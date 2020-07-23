# scitific cal
import numpy as np
from scipy.spatial.distance import pdist, squareform
import copy
import math
# plotting
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
# system
from time import time
import sys
import os
import gc
import subprocess # Call the command line
from subprocess import call
import pdb
# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau

## Import local modules (Prof.JX-W's python code)
RWOF_dir = os.path.expanduser("/home/luning/Documents/utility/pythonLib")
RWOF_dir_1 = os.path.expanduser("/home/luning/Documents/utility/pythonLib/python_openFoam")
sys.path.append(RWOF_dir)
sys.path.append(RWOF_dir_1)

# import the modules you need
#import foamFileOperation as foamOp
class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Net, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('hidden', torch.nn.Linear(n_feature, n_hidden))
        self.features.add_module('active1', Swish())
        self.features.add_module('hidden2', torch.nn.Linear(n_hidden, n_hidden))
        self.features.add_module('active2', Swish())
        self.features.add_module('hidden3', torch.nn.Linear(n_hidden, n_hidden))
        self.features.add_module('active3', Swish())
        #self.features.add_module('hidden3', torch.nn.Linear(n_hidden, n_hidden))
        #self.features.add_module('active3', Swish())
        self.features.add_module('predict', torch.nn.Linear(n_hidden, 3))
        
    def forward(self, x):
        return self.features(x)
    
    def reset_parameters(self, verbose=False):
        #TODO: where did you define module?
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
        if 'reset_parameters' in dir(module):
            if callable(module.reset_parameters):


                module.reset_parameters()
            if verbose:
                print("Reset parameters in {}".format(module))

