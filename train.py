import sys
sys.path.insert(0,'caffe_ext/python');
import utils as ut;
from multiprocessing import Process, Queue, Pool
import matplotlib.pyplot as plt
import numpy as np
import os, subprocess, time, math, random, shutil, signal, glob, re
from caffe import layers as L, params as P, to_proto, NetSpec, get_solver, Net
from caffe.proto import caffe_pb2
import caffe
import deepcontext_config

# Basic Configuration
patch_sz=(96,96) # size of sampled patches
batch_sz=512     # max patches in a single batch
gap = 48         # gap between patches
noise = 7        # jitter by at most this many pixels

def netset(n, nm, l):
  setattr(n, nm, l);
  return getattr(n,nm);

def conv_relu(n, bottom, name, ks, nout, stride=1, pad=0, group=1, 
              batchnorm=False, weight_filler=dict(type='xavier')):
    conv = netset(n, 'conv'+name, L.Convolution(bottom, kernel_size=ks, stride=stride, 
                         num_output=nout, pad=pad, group=group, 
                         weight_filler=weight_filler))
    convbatch=conv;
    if batchnorm:
      batchnorm = netset(n, 'bn'+name, L.BatchNorm(conv, in_place=True, 
                           param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}]));
      convbatch = batchnorm
    # Note that we don't have a scale/shift afterward, which is different from
    # the original Batch Normalization layer.  Using a scale/shift layer lets
    # the network completely silence the activations in a given layer, which
    # is exactly the behavior that we need to prevent early on.
    relu=netset(n, 'relu'+name, L.ReLU(convbatch, in_place=True))
    return conv, relu 

def fc_relu(n, bottom, name, nout, batchnorm=False):
    fc = netset(n, 'fc'+name, L.InnerProduct(bottom, num_output=nout, 
                        weight_filler = dict(type='xavier')))
    fcbatch=fc;
    if batchnorm:
      batchnorm = netset(n, 'bn'+name, L.BatchNorm(fc, in_place=True,
                           param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}]));
      fcbatch = batchnorm
    relu = netset(n, 'relu'+name, L.ReLU(fcbatch, in_place=True));
    return fc, relu

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet_stack(data, n, use_bn=True):
    conv_relu(n, data, '1', 11, 96, stride=4, pad=5)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    conv_relu(n, n.norm1, '2', 5, 256, pad=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    conv_relu(n, n.norm2, '3', 3, 384, pad=1, batchnorm=use_bn)
    conv_relu(n, n.relu3, '4', 3, 384, pad=1, batchnorm=use_bn)
    conv_relu(n, n.relu4, '5', 3, 256, pad=1, batchnorm=use_bn)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    fc_relu(n, n.pool5, '6', 4096, batchnorm=use_bn)

def gen_net(batch_size=512, use_bn=True):
    n=NetSpec();
    n.data = L.DummyData(shape={"dim":[batch_size,3,96,96]})
    n.select1 = L.DummyData(shape={"dim":[2]})
    n.select2 = L.DummyData(shape={"dim":[2]})
    n.label = L.DummyData(shape={"dim":[2]})
    caffenet_stack(n.data, n, use_bn)
    n.first = L.BatchReindex(n.relu6, n.select1)
    n.second = L.BatchReindex(n.relu6, n.select2)
    n.fc6_concat=L.Concat(n.first, n.second);

    fc_relu(n, n.fc6_concat, '7', 4096, batchnorm=use_bn);
    fc_relu(n, n.relu7, '8', 4096);
    n.fc9 = L.InnerProduct(n.relu8, num_output=8,
                            weight_filler=dict(type='xavier'));
    n.loss = L.SoftmaxWithLoss(n.fc9, n.label, loss_param=dict(normalization=P.Loss.NONE));

    prot=n.to_proto()
    prot.debug_info=True
    return prot;

# image preprocessing.  Note that the input image will modified.
def prep_image(im):

  # for some patches, randomly downsample to as little as 100 total pixels
  if(random.random() < .33):
    origsz=im.shape
    randpix=int(math.sqrt(random.random() * (95 * 95 - 10 * 10) + 10 * 10))
    im=caffe.io.resize(im.astype(np.uint8), (randpix, randpix))
    im=(caffe.io.resize(im, (origsz[0], origsz[1])) * 255).astype(np.float32)

  # randomly drop all but one color channel
  chantokeep=random.randint(0, 2);
  mean=[123, 117, 104]
  for i in range(0, 3):
    if i==chantokeep:
      im[:,:,i]-=np.mean(im[:,:,i])
    else:
      im[:,:,i]=np.random.uniform(0, 1, (im.shape[0], im.shape[1])) - .5

  # Normalize the mean and variance so that gradients are a less useful cue;
  # then scale by 50 so that the variance is roughly the same as the usual
  # AlexNet inputs.
  im=im / np.sqrt(np.mean(np.square(im))) * 50
  return im.transpose(2, 0, 1)

# Sample a patch.  This function defines a grid over the image of patches
# of size patch_sz with a gap of gap between the patches.  The upper-left
# corner of the grid starts at (gridstartx, gridstarty).  We then sample
# the patch at location (x, y) on this grid, jitter by up to noisehalf in
# every direction.  im_shape is the dimensions of the image; an error
# is thrown if (x, y) refers to a patch outside the image frame. 
#
# Returns the coordinates of the sampled patch's upper left corner.
def sample_patch(x, y, gridstartx, gridstarty, patch_sz, gap, noisehalf, 
                 im_shape):
  xpix = gridstartx + x * (patch_sz[1] + gap) \
         + random.randint(-noisehalf, noisehalf)
  xpix2 = min(max(xpix, 0), im_shape[1] - patch_sz[1])
  ypix = gridstarty + y * (patch_sz[0] + gap) \
         + random.randint(-noisehalf, noisehalf)
  ypix2 = min(max(ypix, 0), im_shape[0] - patch_sz[0])
  assert abs(xpix - xpix2) < gap
  assert abs(ypix - ypix2) < gap
  return (xpix2 , ypix2)

# A background thread which loads images, extracts pairs of patches, arranges
# them into batches.
#
# args:
#   dataq: A queue object where the batches of data will be sent.  Batches
#   consist of a tuple of: 
#     (1) a 4-d array of of N patches, or a filename of
#     a file containing the data,
#     (2) a list of pairs of patches to be used as training examples, kept in
#     an array of shape N-by-2,
#     (3) an array of labels for each pair, which are in the range 0-7.
#
#   batch_sz: max number of patches go in each batch
#   imgs: a list of images (see load_imageset)
#   tmpdir: if not None, data batches will be saved here and the resulting
#     filename will be sent through the queue rather than the data. 
#   seed: a seed for the random number generator, required so that different
#     processes produce images in a different order:
#   tid: thread id for use in filenames.
#   patch_sz: size of sampled patches
def imgloader(dataq, batch_sz, imgs, tmpdir, seed, tid, patch_sz): 
  qctr = 0
  curidx = 0
  # order for going through the images
  np.random.seed(seed)
  imgsord=np.random.permutation(len(imgs['filename']))
  # sample this many grids per image 
  num_grids = 4
  gridctr = 0
  # storage for the sampled batch
  perm = []
  label = []
  pats = []
  # index within the current batch
  j = 0;
  # keep returning batches forever.  Each iteration of this loop
  # samples one grid of patches from an image.
  while True:
    # if we've already sampled num_grids in this image, we sample a new one.
    if(gridctr==0):
      while True:
        try:
          im=ut.get_resized_image(imgsord[curidx % len(imgs['filename'])],
                             imgs,
                             {"gri_targpixels":random.randint(150000,450000)})
        except:
          curidx = (curidx + 1) % (len(imgsord))
          print("broken image id " + str(curidx))
          continue;
        curidx = (curidx + 1) % (len(imgsord));
        if(im.shape[0] > patch_sz[0] * 2 + gap + noise and 
           im.shape[1] > patch_sz[1] * 2 + gap + noise):
          break
      gridctr = num_grids;
    # compute where the grid starts, and then comptue its size.
    gridstartx = random.randint(0, patch_sz[1] + gap - 1)
    gridstarty = random.randint(0, patch_sz[0] + gap - 1)
    gridszx = int((im.shape[1] + gap-gridstartx) / (patch_sz[1] + gap))
    gridszy = int((im.shape[0] + gap-gridstarty) / (patch_sz[0] + gap))
    # Whenever we sample and store a patch, we'll put its index in this
    # variable so it's easy to pair it up later.
    grid=np.zeros((gridszy, gridszx), int)

    # if we can't fit the current grid into the batch without going over
    # batch_sz, put the batch in the queue and reset.
    if(gridszx * gridszy + j >= batch_sz):
      pats=map(prep_image, pats)
      data=np.array(pats)
      qctr+=1
      perm=(np.array(perm))
      label=(np.array(label))
      if tmpdir is None:
        dataq.put((np.ascontiguousarray(data), perm, label), timeout=600)
      else:
        fnm=tmpdir + str(tid) + '_' + str(qctr) + '.npy'
        np.save(fnm, data)
        dataq.put((fnm, perm, label), timeout=600)
      perm=[]
      label=[]
      pats=[]
      j=0

    gridctr-=1;
    # for each location in the grid, sample a patch, search up and to the
    # left for patches that can be paired with it, and add them to the batch.
    for y in range(0,gridszy):
      for x in range(0,gridszx):
        (xpix, ypix)=sample_patch(x, y, gridstartx, gridstarty, patch_sz, 
                                  gap, noise, im.shape)
        pats.append(np.copy(
            im[ypix:ypix + patch_sz[0], xpix:xpix + patch_sz[1], :]*255));
        grid[y, x] = j;
        for pair in [(-1,-1), (0,-1), (1,-1), (-1,0)]:
          gridposx = pair[0] + x;
          gridposy = pair[1] + y;
          if(gridposx < 0 or gridposy < 0 or gridposx >= gridszx):
            continue;
          perm.append(np.array([j, grid[gridposy, gridposx]]));
          label.append(pos2lbl(pair));
          perm.append(np.array([grid[gridposy, gridposx],j]))
          label.append(pos2lbl((-pair[0],-pair[1])));
        j+=1;

# convert an (x, y) offset into a single number to use as a label. Labels are:
# 1 2 3
# 4   5
# 6 7 8
def pos2lbl(pos):
  (posx, posy)=pos;
  if(posy==-1):
    lbl = posx + 1;
  elif(posy == 0):
    lbl = (posx + 7) / 2
  else:
    assert(posy == 1);
    lbl = posx + 6;
  return lbl;

# will set these later, need to make it global for signal handler
if 'exp_name' not in locals():
  exp_name='';
def signal_handler(signal, frame):
    print("PYCAFFE IS NOT GUARANTEED TO RETURN CONTROL TO PYTHON WHEN " +
        "INTERRUPTED. That means I can't necessarily clean up temporary files " +
        "and spawned processes. " +
        "You were lucky this time. Run deepcontext_quit() to quit. Next time touch " + 
        exp_name + '_pause to pause safely and ' + exp_name + '_quit to quit.')

# return a dict containing two fields 'dir' (a string) and 'filename' (a list 
# of strings) such that
# scipy.misc.imread(imgs['dir'] + imgs['filename'][idx]) will return
# an image.  Make sure the order of imgs['filename'] is deterministic, since
# the code uses the index in this list as an ID for each image.
def load_imageset():
  datadir=deepcontext_config.imagenet_dir;
  imgs={};
  imgs['dir']=datadir+'train/';
  names=[];
  with open(datadir + 'train.txt', 'rb') as f:
    for line in f:
      row=line.split();
      names.append(row[0]);
  imgs['filename']=names;
  return imgs;

# The main code body.  
try:
  if 'solver' not in locals():
    exp_name=ut.mfilename();
    # all generated files will be here.
    outdir = deepcontext_config.out_dir + '/' + exp_name + '_out/';
    if deepcontext_config.tmp_dir:
      tmpdir = deepcontext_config.tmp_dir + '/' + exp_name + '_out/';
    else:
      tmpdir = None
    if not os.path.exists(outdir):
      os.mkdir(outdir);
    else:
      try:
        input=raw_input;
      except:
        pass;
      print('=======================================================================');
      print('Found old data. Load most recent snapshot and append to log file (y/N)?');
      inp=input('======================================================================');
      if not 'y' == inp.lower():
        raise RuntimeError("User stopped execution");
        
    if not os.path.exists(tmpdir):
      os.makedirs(tmpdir);
    # by default, we append to the logfile if it's already there.
    #if os.path.exists(outdir + "out.log"):
    #  os.remove(outdir + "out.log")

  # Magic commands to redirect standard output and standard
  # error to a log file for easy plotting of the loss function.  Note that
  # running these commands will screw up your terminal; hence why the whole
  # code is wrapped in a try/finally statement that puts things back the way
  # they were.
  prevOutFd = os.dup(sys.stdout.fileno())
  prevErrFd = os.dup(sys.stderr.fileno()) 
  tee = subprocess.Popen(["tee", "-a", outdir + "out.log"], stdin=subprocess.PIPE)
  os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
  os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

  # if the solver hasn't been set up yet, do so now.  Otherwise assume that 
  # we're continuing.
  if 'solver' not in locals():
    if os.path.isfile(exp_name + '_pause'):
      os.remove(exp_name + '_pause')
    if os.path.isfile(exp_name + '_quit'):
      os.remove(exp_name + '_quit')

    with open(outdir+'network.prototxt','w') as f:
      f.write(str(gen_net()));
    with open(outdir+'network_no_bn.prototxt','w') as f:
      f.write(str(gen_net(use_bn=False)));

    ut.mkorender('solver_mko.prototxt', outdir + 'solver.prototxt', 
                 base_lr=1e-5, outdir=outdir, weight_decay=0)

    print('setting gpu')
    caffe.set_mode_gpu();
    print('constructing solver')
    solver = caffe.get_solver(outdir + 'solver.prototxt');

    # Find earlier solvers and restore them
    fils=glob.glob(outdir + 'model_iter_*.solverstate');
    if(len(fils)>0):
      idxs=[];
      for f in fils:
        idxs.append(int(re.findall('\d+',os.path.basename(f))[0]));
      idx=np.argmax(np.array(idxs));
      solver.restore(outdir + os.path.basename(fils[idx]));

    # we occasionally read out the parameters in this list and save the norm
    # of the update out to disk, so we can make sure they're updating at
    # the right speed.
    track=[];
    for bl in solver.net.params:
      if not 'bn' in bl:
        track.append(bl);
    nrm=dict();
    intval={};
    trackold={};
    for tracknm in track:
      intval[tracknm]=[];
      nrm[tracknm]=[];

    curstep = 0;

    # load the images
    imgs=load_imageset();

    # start the data prefetching threads.
    dataq=[]
    procs=[]
    i=0
    for i in range(0,3):
      dataq.append(Queue(5))
      procs.append(Process(target=imgloader, 
                       args=(dataq[-1], batch_sz, imgs, tmpdir,
                             (hash(outdir)+i) % 1000000, #random seed
                             i, patch_sz)))
      procs[-1].start()
    def deepcontext_quit():
      for proc in procs:
        proc.terminate()
      time.sleep(2)
      shutil.rmtree(tmpdir)
      os.kill(os.getpid(), signal.SIGKILL)
    signal.signal(signal.SIGINT, signal_handler)
  
  # The main loop over batches.
  while True:
    start=time.time();
    (datafnm, perm, label)=dataq[curstep % len(dataq)].get(timeout=600)
    print("queue size: " + str(dataq[curstep % len(dataq)].qsize()))

    if(tmpdir is None):
      d=datafnm
    else:
      d=np.load(datafnm,mmap_mode='r')
      os.remove(datafnm)

    # input the patch data
    solver.net.blobs['data'].reshape(*d.shape)
    solver.net.blobs['data'].data[:]=d[:]

    # input the patch pairings
    solver.net.blobs['select1'].reshape(*(perm.shape[0],))
    solver.net.blobs['select1'].data[:]=perm[:,0]
    solver.net.blobs['select2'].reshape(*(perm.shape[0],))
    solver.net.blobs['select2'].data[:]=perm[:,1]

    # input the labels
    solver.net.blobs['label'].reshape(*label.shape)
    solver.net.blobs['label'].data[:]=label

    print("data input time: " + str(time.time()-start));

    # take a step
    solver.step(1)
    print("norm_loss: " + str(solver.net.blobs['loss'].data /
          (label.shape[0])));
    print("solver step time: " + str(time.time() - start));
    dobreak=False;
    broken=[];
    
    msg = (' Please examine the situation and re-execute ' + exp_name + 
           '.py to continue.')
    if curstep % 100 == 0:
      start = time.time()
      print("getting param statistics...")
      for tracknm in track:
        try:
          intval[tracknm].append(np.sqrt(np.sum(np.square(
              solver.net.params[tracknm][0].data - trackold[tracknm]))));
          if (intval[tracknm][-1] > 10 * intval[tracknm][-2] 
              and curstep > 100) \
              or np.isnan(intval[tracknm][-1]):
            print(tracknm + " changed a suspiciously large amount." + msg)
            dobreak = True; 
            broken.append(tracknm);
        except:
          print("init " + tracknm + " statistics")
        trackold[tracknm]=np.copy(solver.net.params[tracknm][0].data);
        nrmval=np.sqrt(np.sum(solver.net.params[tracknm][0].data * 
                       solver.net.params[tracknm][0].data))
        nrm[tracknm].append(nrmval);
      np.save(outdir + 'intval',intval);
      np.save(outdir + 'nrm',nrm);
      print("param statistics time: " + str(time.time()-start));

      val = np.sum(solver.net.params["fc8"][0].data);
      if np.isnan(val) or val > 1e10:
        print("fc8 activations look broken to me." + msg)
        dobreak = True;
      val2 = np.max(np.abs(solver.net.blobs["pool1"].diff));
      if np.isnan(val2) or val2 > 1e8:
        print("pool1 diffs look broken to me." + msg)
        dobreak = True;

    curstep += 1;
    if dobreak:
      break;
    if os.path.isfile(exp_name + '_pause'):
      break;
    if os.path.isfile(exp_name + '_quit'):
      # Need to kill the subprocesses and delete the temporary files.
      deepcontext_quit();
except KeyboardInterrupt:
  if 'procs' in locals():
    handler(None,None)
  raise;
finally:
  if 'prevOutFd' in locals():
    os.dup2(prevOutFd, sys.stdout.fileno())
    os.close(prevOutFd)
    os.dup2(prevErrFd, sys.stderr.fileno())
    os.close(prevErrFd)
