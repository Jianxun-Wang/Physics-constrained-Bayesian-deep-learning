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
#from time import time
import time
import sys
import os
import gc
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
# local import
import FCN
from BayesNN import BayesNN
from helpers import log_sum_exp, parameters_to_vector, vector_to_parameters, _check_param_device
from SVGD import SVGD
from args import args, device
from cases import cases

# Specific hyperparameter for SVGD
n_samples = args.n_samples
noise = args.noise

# deterministic NN structure
nFeature = 2
nNeuron = 20
denseFCNN = FCN.Net(nFeature, nNeuron)
#print(denseFCNN)

# Bayesian NN
bayes_nn = BayesNN(denseFCNN, n_samples=n_samples, noise=noise).to(device)
# specifying training case
train_case  = cases(bayes_nn, 'stenosis_hard')
#time.sleep(10)
train_loader, train_size = train_case.dataloader()
ntrain = train_size

# Initialize SVGD
svgd = SVGD(bayes_nn,train_loader)
print('Start training.........................................................')
tic = time.time()
epochs = args.epochs
data_likelihod = []
eq_likelihood = []
rec_lamda = []
rec_beta_eq = []
rec_log_beta = []
rec_log_alpha = []
LOSS,LOSSB,LOSS1,LOSS2,LOSS3,LOSSD = [], [], [], [], [], []

for epoch in range(epochs):
    data_likelihod, eq_likelihood, rec_lamda, rec_beta_eq, rec_log_beta,rec_log_alpha, LOSS,LOSSB,LOSS1,LOSS2,LOSS3,LOSSD = svgd.train(epoch, data_likelihod, eq_likelihood, rec_lamda, rec_beta_eq, rec_log_beta, rec_log_alpha, LOSS,LOSSB,LOSS1,LOSS2,LOSS3,LOSSD)
training_time = time.time() - tic
print('finished in ',training_time)
torch.save(bayes_nn.state_dict(),"test1e-2_1p.pt")
#np.savetxt('datalikelihood.csv',data_likelihod)
#np.savetxt('eqlikelihood.csv',eq_likelihood)
np.savetxt('Loss.csv',LOSS)
np.savetxt('LOSSB.csv',LOSSB)
np.savetxt('LOSS1.csv',LOSS1)
np.savetxt('LOSS2.csv',LOSS2)
np.savetxt('LOSS3.csv',LOSS3)
np.savetxt('LOSSD.csv',LOSSD)
np.savetxt('log_beta.csv',rec_log_beta)
np.savetxt('beta_eq.csv',rec_beta_eq)

# plot
train_case.plot()

