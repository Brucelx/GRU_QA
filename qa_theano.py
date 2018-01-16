#-*-coding:utf-8-*-
#!/usr/bin/env python
import os, sys, timeit, random, operator

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import memory_profiler
import pickle

reload(sys)
sys.setdefaultencoding( "utf-8" )


