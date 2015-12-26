#!/usr/bin/python2

import os,sys
sys.path.insert(0,'caffe_ext/python');
import caffe
import numpy as np
import utils as ut
import csv,math
import subprocess, sys
import string
import copy
import argparse
from google.protobuf import text_format

parser = argparse.ArgumentParser(description='convert a network using ' +
   'batch normalization into an equivalent network that does not.  Assumes that ' +
   'parameter layer names have the form \'conv*\' or \'fc*\' and are directly ' +
   'followed by their respective batch norm layers.  These BatchNorm layers must ' +
   'have names which are identical to the names of the layers that they modify, ' +
   'except that \'conv\' or \'fc\' is replaced by \'bn\'.  E.g. conv3 is directly ' +
   'followed by \'bn3\'.  Does not copy fc7, fc8, or fc9.')
parser.add_argument('indefinition', type=str,
                   help='input network definition (prototxt)')
parser.add_argument('inmodel', type=str,
                   help='input network parameters (caffemodel)')
parser.add_argument('outdefinition', type=str,
                   help='output network definition (prototxt)')
parser.add_argument('outmodel', type=str,
                   help='output network parameters (caffemodel; will be overwritten)')

args = parser.parse_args()

net2= caffe.Net(args.indefinition,args.inmodel,caffe.TRAIN);
net=caffe.Net(args.outdefinition,caffe.TRAIN);
tocopy=net.params;

netdef = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(args.indefinition).read(), netdef)

for prm in tocopy:
  if('fc7' in prm or 'fc8' in prm or 'fc9' in prm):
    print 'skipping ' + prm;
    continue;
  if(prm.startswith('fc')):
    bnprm='bn'+prm[2:];
  elif(prm.startswith('conv')):
    bnprm='bn'+prm[4:];
  else:
    print 'warning: ' + prm + ' has parameters but I can\'t infer its layer type.'
    continue;
  if bnprm not in net2.params:
    print bnprm + ' not found, just copying ' + prm;
    for i in range(0,len(net2.params[prm])):
      net.params[prm][i].data[...]=np.copy(net2.params[prm][i].data[...]);
    continue;
  if net2.params[prm][0].data.shape != net.params[prm][0].data.shape:
    print 'warning: ' + prm + ' has parameters but they are different sizes in the different protos.  skipping.'
    continue;
  print 'removing batchnorm from ' + prm;

  for i in range(0,len(net2.params[prm])):


    prmval=np.copy(net2.params[prm][i].data).reshape(net.params[prm][i].data.shape);
    meanval=np.copy(net2.params[bnprm][0].data);
    stdval=np.copy(net2.params[bnprm][1].data);

    meanval/=net2.params[bnprm][2].data[...].reshape(-1);
    stdval/=net2.params[bnprm][2].data[...].reshape(-1);
    eps=None;
    for j in range(0,len(netdef.layer)):
      if str(netdef.layer[j].name) == bnprm:
        eps=netdef.layer[j].batch_norm_param.eps;
    if eps is None:
      raise ValueError("Unable to get epsilon for layer " + nbprm);
    stdval+=eps;
    #stdval-=np.square(meanval)+1e-5; # for the old bn layer
    stdval=np.sqrt(stdval);
    if(i==1):
      prmval/=stdval[:];
      prmval-=meanval[:]/stdval[:];
    else:
      if prm.startswith('fc'):
        prmval/=stdval.reshape((-1,1));
      else:
        prmval/=stdval.reshape((-1,1,1,1));
    net.params[prm][i].data[:]=prmval
net.save(args.outmodel);
