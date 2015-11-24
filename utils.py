import inspect
import re
from mako.template import Template
from mako import exceptions
from mako.lookup import TemplateLookup
import os
from scipy import misc
import numpy as np
import caffe

# the filename for the root file that's being run.
def mfilename():
  stack = inspect.stack();
  filepath = (stack[1][1]);
  dotidx=[m.start() for m in re.finditer('\.', filepath)]
  filenm = filepath[0:dotidx[len(dotidx)-1]];
  dotidx=[m.start() for m in re.finditer('\/', filepath)]
  try:
    # this command fails in python <3
    filenm = filenm[dotidx[len(dotidx)-1]+1:];
  except:
    pass;
  return filenm;

def mkorender(tplfile,outfile,*args,**kwargs):
  mylookup = TemplateLookup(directories=[os.getcwd()])
  tpl=Template(filename=tplfile,lookup=mylookup);
  mkorendertpl(tpl,outfile,*args,**kwargs);

def mkorendertpl(tpl,outfile,*args,**kwargs):
  with open(outfile,"w") as f:
    try:
      f.write(tpl.render(*args,**kwargs));
    except:
      print((exceptions.text_error_template().render()))
      raise;

def mkorenderstr(strtpl,outfile,*args,**kwargs):
  mylookup = TemplateLookup(directories=[os.getcwd()])
  tpl=Template(strtpl,lookup=mylookup);
  mkorendertpl(tpl,outfile,*args,**kwargs);

def get_image(idx,dataset):
  im=misc.imread(dataset['dir'] + dataset['filename'][idx]);
  if (len(im.shape)==2):
    im=np.concatenate((im[:,:,None],im[:,:,None],im[:,:,None]),axis=2);
  return im;

def get_resized_image(idx,dataset,conf={}):
  targpixels=None;
  if 'targpixels' in dataset:
    targpixels=dataset['targpixels']
  if 'gri_targpixels' in conf:
    targpixels=conf['gri_targpixels'];
  maxdim=conf.get('gri_maxdim',None);
  im=get_image(idx,dataset);
  try:
    if(len(im.shape)==2):
      print("found grayscale image");
      im=np.array([im,im,im]).transpose(1,2,0);
    elif(im.shape[2]==4):
      print("found 4-channel png with channel 4 min "+str(np.min(im[:,:,3])));
      im=im[:,:,0:3];
  except:
    print("image id: " + str(idx));
    raise
    
  im=im.astype(np.float32)/255;
  if targpixels is not None:
    npixels = float(im.shape[0]*im.shape[1])
    # TODO: this has issues with the image boundary, as it samples zeros 
    # outside the image bounds.
    im=caffe.io.resize_image(im,(int(im.shape[0]*np.sqrt(targpixels/npixels)),int(im.shape[1]*np.sqrt(targpixels/npixels))))
  elif maxdim is not None:
    immax=max(im.shape[0:2]);
    ratio=float(maxdim)/float(immax);
    im=caffe.io.resize_image(im,(int(im.shape[0]*ratio),int(im.shape[1]*ratio)))
  return im;


