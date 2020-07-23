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
import pdb
import subprocess # Call the command line
from subprocess import call
# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.ticker as ticker
# local import
from FCN import Net
from args import args, device


class BayesNN(nn.Module):
    """Define Bayesian netowrk

    """
    def __init__(self, model, n_samples=2, noise=1e-6):
        super(BayesNN, self).__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model {} is not a Module subclass".format(
                torch.typename(model)))
        self.n_samples = n_samples # number of particles (# of perturbed NN)

        # w_i ~ StudentT(w_i | mu=0, lambda=shape/rate, nu=2*shape)
        # for efficiency, represent StudentT params using Gamma params
        # Nick shape  = 0.5, rate = 10
        self.w_prior_shape = 1.
        self.w_prior_rate = 0.05 

        # noise variance 1e-6: beta ~ Gamma(beta | shape, rate)
        ## Nick shape = 10 rate = 10*C0*args*.t**k0 = 10*0.2*0.005^3 = 2.5e-7
        self.beta_prior_shape = (2.)
        self.beta_prior_rate = noise

        print('noise is',noise)
        ################
        # for the equation loglilihood
        self.var_eq = 1e-4
        self.lamda = np.ones((1,))

        ## for stenosis hard use
        self.mu = args.mu
        self.sigma = args.sigma
        self.scale = args.scale
        self.nu = args.nu
        self.rho = args.rho
        self.rInlet = args.rInlet
        self.xStart = args.xStart
        self.xEnd = args.xEnd
        self.dP = args.dP
        self.L = args.L
        ##
        ################
        # replicate `n_samples` instances with the same network as `model`
        instances = []
        for i in range(n_samples):
            new_instance = copy.deepcopy(model)
            #new_instance = Net(1, 20)
            # initialize each model instance with their defualt initialization
            # instead of the prior
            #new_instance.reset_parameters()
            def init_normal(m):
                if type(m) == nn.Linear:
                    nn.init.kaiming_normal_(m.weight)
            new_instance.apply(init_normal)
            print('Reset parameters in model instance {}'.format(i))
            instances.append(new_instance)
            #t.sleep(100)
        
        self.nnets = nn.ModuleList(instances)
        #del instances # delete instances

        # log precision (Gamma) of Gaussian noise
        ## change to a constant log_beta
        
        log_beta = Gamma(self.beta_prior_shape, 
                         self.beta_prior_rate).sample((self.n_samples,)).log().to(device)
        ##
        ## beta equals to 10^3, 1/beta = 1e-3
        #log_beta = (1/noise*torch.ones(self.n_samples,)).log()
        print('log_beta is',log_beta)
        for i in range(n_samples):
            self.nnets[i].log_beta = Parameter(log_beta[i])

            print('log_beta grad is',self.nnets[i].log_beta.requires_grad)

            self.nnets[i].beta_eq = 1/self.var_eq

        print('Total number of parameters: {}'.format(self._num_parameters()))

    
    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name)
            count += param.numel()
        return count

    def __getitem__(self, idx):
        return self.nnets[idx]

    @property
    
    def log_beta(self):
        return torch.tensor([self.nnets[i].log_beta.item() 
            for i in range(self.n_samples)], device=device)
    
    def forward(self, inputs):

        output = []
        for i in range(self.n_samples):
            output.append(self.nnets[i].forward(inputs))
        output = torch.stack(output)

        return output
    def _log_joint(self, index, output, target, outputb, outputin, targetin, outputout, targetout, ntrain):
        """Log joint probability or unnormalized posterior for single model
        instance. Ignoring constant terms for efficiency.
        Can be implemented in batch computation, but memory is the bottleneck.
        Thus here we trade computation for memory, e.g. using for loop.
        Args:
            index (int): model index, 0, 1, ..., `n_samples`
            output (Tensor): y_pred
            target (Tensor): y
            ntrain (int): total number of training data, mini-batch is used to
                evaluate the log joint prob
        Returns:
            Log joint probability (zero-dim tensor)
        """
        # Normal(target | output, 1 / beta * I)
        #print('output.size = ',output.size(0))
        n_train = output.size(0)


        targetb = torch.zeros_like(outputb)
        ### u loglikelihood
        ## sparse domain data
        # u
        log_likelihood = ntrain / output.size(0) * (
                            - 0.5 * self.nnets[index].log_beta.exp()
                            * ((target[:,0:2] - output[:,0:2]).pow(2).sum()
                            + 0.5 * target[:,0:2].numel() * self.nnets[index].log_beta))
        # v
        #log_likelihood = ntrain / output.size(0) * (
                            #- 0.5 * self.nnets[index].log_beta.exp()
                            #* ((target[:,1] - output[:,1]).pow(2).sum()
                            #+ 0.5 * target[:,1].numel() * self.nnets[index].log_beta))
        ## boundary data
        # u 
        log_likelihood += ntrain / outputb.size(0) * (
                            - 0.5 * self.nnets[index].log_beta.exp()
                            * ((targetb[:,0:2] - outputb[:,0:2]).pow(2).sum()
                            + 0.5 * targetb[:,0:2].numel() * self.nnets[index].log_beta))

        # inlet data
        # u
        log_likelihood += ntrain / outputin.size(0) * (
                            - 0.5 * self.nnets[index].log_beta.exp()
                            * ((targetin[:,0:2] - outputin[:,0:2]).pow(2).sum()
                            + 0.5 * targetin[:,0:2].numel() * self.nnets[index].log_beta))

        ## outlet data
        # u
        log_likelihood += ntrain/outputout.size(0) * (
                            - 0.5 * self.nnets[index].log_beta.exp()
                            * ((targetout[:,0:2] - outputout[:,0:2]).pow(2).sum()
                            + 0.5 * targetout[:,0:2].numel() * self.nnets[index].log_beta))

        # log prob of prior of weights, i.e. log prob of studentT
        log_prob_prior_w = torch.tensor(0.).to(device)
        for param in self.nnets[index].features.parameters():
            log_prob_prior_w +=                     torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
        log_prob_prior_w *= -(self.w_prior_shape + 0.5)
        # log prob of prior of log noise-precision (NOT noise precision)
        # noise prior (beta_shape - 1)*log_beta
        log_prob_prior_log_beta = ((self.beta_prior_shape-1) * self.nnets[index].log_beta-  self.beta_prior_rate*self.nnets[index].log_beta.exp() )


        


        return log_likelihood + log_prob_prior_w + log_prob_prior_log_beta
        
    def _log_likeEq(self,index,u,u_x,u_t,u_xx,ntrain):
        #u = output[:,0]
        u = u.view(len(u),-1)
        '''
        x = inputs[:,0]
        t = inputs[:,1]
        u = output[:,0]
        x,t,u = x.view(len(x),-1),t.view(len(t),-1),u.view(len(u),-1)
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        '''
        mu = 0
        res = u_t + u*u_x
        #print('res is',res)
        #print('equation output.size is',output.size(0))
        '''
        log_likelihood = ntrain / output.size(0) * (
                            - 0.5 * self.nnets[index].beta_eq#* (self.nnets[index].log_beta.exp()/ self.nnets[index].log_beta.exp())
                            * (res - 0).pow(2).sum()
                            )
        '''
        loss_f = nn.MSELoss()
        loss = loss_f(res,torch.zeros_like(res))
        #print('log_likelihood is',log_likelihood)
        return loss
        #return log_likelihood


    def _mse(self, index, output, target, ntrain):

        loss_f = nn.MSELoss()
        loss = loss_f(output,target)
        #print('log_likelihood is',log_likelihood)

        return loss
    def criterion(self,index,x,y,ntrain):
        
        x = torch.FloatTensor(x).to(device)
        y = torch.FloatTensor(y).to(device)
        x.requires_grad = True
        y.requires_grad = True


        
        
        net_in = torch.cat((x,y),1)
        
        output = self.nnets[index].forward(net_in)
        u = output[:,0]
        v = output[:,1]
        P = output[:,2]
        u = u.view(len(u),-1)
        v = v.view(len(v),-1)
        P = P.view(len(P),-1)


        # axisymetric
        #R = self.scale * 1/np.sqrt(2*np.pi*self.sigma**2)*torch.exp(-(x-self.mu)**2/(2*self.sigma**2))
        #h = self.rInlet - R

        #u_hard = u*(h**2 - y**2)
        #v_hard = (h**2 -y**2)*v
        u_hard = u
        v_hard = v
        P_hard = (self.xStart-x)*0 + self.dP*(self.xEnd-x)/self.L + 0*y + (self.xStart - x)*(self.xEnd - x)*P
        #P_hard = (-4*x**2+3*x+1)*dP +(xStart - x)*(xEnd - x)*P



        u_x = torch.autograd.grad(u_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_y = torch.autograd.grad(u_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        P_x = torch.autograd.grad(P_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        #P_xx = torch.autograd.grad(P_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        #print('type of nu is',nu.shape)
        loss_1 = (u_hard*u_x+v_hard*u_y-self.nu*(u_xx+u_yy)+1/self.rho*P_x)

        v_x = torch.autograd.grad(v_hard,x,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        
        v_y = torch.autograd.grad(v_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        
        v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        P_y = torch.autograd.grad(P_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        #P_yy = torch.autograd.grad(P_y,y,grad_outputs=torch.ones_like(x),create_graph = True,allow_unused = True)[0]


        loss_2 = (u_hard*v_x+v_hard*v_y - self.nu*(v_xx+v_yy)+1/self.rho*P_y)
        #Main_deriv = torch.cat((u_x,u_xx,u_y,u_yy,P_x,v_x,v_xx,v_y,v_yy,P_y),1)
        loss_3 = (u_x + v_y)
        #loss_3 = u_x**2 + 2*u_y*v_x + v_y**2+1/rho*(P_xx + P_yy)
        #loss_3 = loss_3*100




        # MSE LOSS
        #loss_f = nn.MSELoss()
        #loss_f = nn.L1loss()
        
        #ntrain = 50
        logloss1 = ntrain / output.size(0) * (
                            - 0.5 * self.nnets[index].beta_eq#* (self.nnets[index].log_beta.exp()/ self.nnets[index].log_beta.exp())
                            * (loss_1 - torch.zeros_like(loss_1)).pow(2).sum()
                            )

        logloss2 = ntrain / output.size(0) * (
                            - 0.5 * self.nnets[index].beta_eq#* (self.nnets[index].log_beta.exp()/ self.nnets[index].log_beta.exp())
                            * (loss_2 - torch.zeros_like(loss_2)).pow(2).sum()
                            )
        logloss3 =ntrain / output.size(0) * (
                            - 0.5 * self.nnets[index].beta_eq#* (self.nnets[index].log_beta.exp()/ self.nnets[index].log_beta.exp())
                            * (loss_3 - torch.zeros_like(loss_3)).pow(2).sum()
                            )
        
        '''
        logloss1 = -self.nnets[index].beta_eq* (loss_1 - torch.zeros_like(loss_1)).pow(2).mean()
        logloss2 = -self.nnets[index].beta_eq* (loss_2 - torch.zeros_like(loss_2)).pow(2).mean()
        logloss3 = -self.nnets[index].beta_eq* (loss_3 - torch.zeros_like(loss_3)).pow(2).mean()
        '''
        #loss = loss_f(loss_1,torch.zeros_like(loss_1))+ loss_f(loss_2,torch.zeros_like(loss_2))+loss_f(loss_3,torch.zeros_like(loss_3))
        #return loss
        return (logloss1 + logloss2 + logloss3),loss_1,loss_2,loss_3


    ## implement hard constraint
    def hard_constraint(self, inputs, output):
        x= inputs[:,0]
        y = inputs[:,1]

        u = output[:,0]
        v = output[:,1]
        P = output[:,2]
        x = x.view(len(x),-1)
        y = y.view(len(y),-1)
        u = u.view(len(u),-1)
        v = v.view(len(v),-1)
        P = P.view(len(P),-1)


        # axisymetric
        #R = self.scale * 1/np.sqrt(2*np.pi*self.sigma**2)*torch.exp(-(x-self.mu)**2/(2*self.sigma**2))
        #h = self.rInlet - R

        #u_hard = u*(h**2 - y**2)
        #v_hard = (h**2 -y**2)*v
        u_hard = u
        v_hard = v
        P_hard = (self.xStart-x)*0 + self.dP*(self.xEnd-x)/self.L + 0*y + (self.xStart - x)*(self.xEnd - x)*P

        output_const = torch.cat((u_hard,v_hard,P_hard),1)

        return output_const

    #def predict(self, x_test):
    ### modified for burgers,
    def predict(self, inputs):
        """
        Predictive mean and variance at x_test. (only average over w and beta)
        Args:
            x_test (Tensor): [N, *], test input
        """
        # S x N x oC x oH x oW
        y = self.forward(inputs)
        print('shape of output is',y.shape)
        '''
        x = inputs[:,0]
        y = inputs[:,1]
        x = x.view(len(x),-1)
        y = y.view(len(y),-1)
        u = y[:,0]
        v = y[:,1]
        P = y[:,2]

        R = torch.FloatTensor(yUp)


        u_hard = u*(R**2 - y**2)
        v_hard = (R**2 -y**2)*v
        P_hard = (self.xStart-x)*0 + self.dP*(self.xEnd-xt)/L + 0*y + (self.xStart - x)*(self.xEnd - x)*P

        u_hard = u.view(len(u_hard),-1)
        v_hard = v.view(len(v_hard),-1)
        P_hard = P.view(len(P_hard),-1)
        '''

        #y_pred_mean = y.mean(0)
        # compute predictive variance per pixel
        # N x oC x oH x oW
        #EyyT = (y ** 2).mean(0)
        #EyEyT = y_pred_mean ** 2
        #beta_inv = (- self.log_beta).exp()
        #y_pred_var = beta_inv.mean() + EyyT - EyEyT

        return y

    '''
    def predict(self, x_test):
        """
        Predictive mean and variance at x_test. (only average over w and beta)
        Args:
            x_test (Tensor): [N, *], test input
        """
        # S x N x oC x oH x oW
        y = self.forward(x_test)
        y_pred_mean = y.mean(0)
        # compute predictive variance per pixel
        # N x oC x oH x oW
        EyyT = (y ** 2).mean(0)
        EyEyT = y_pred_mean ** 2
        beta_inv = (- self.log_beta).exp()
        y_pred_var = beta_inv.mean() + EyyT - EyEyT

        return y_pred_mean, y_pred_var


    def propagate(self, mc_loader):
        """
        Mean and Variance statistics of predictive output distribution
        averaging over the input distribution, i.e. uncertainty propagation.

        First compute the conditional predictive mean and var given realizations
        of uncertain surrogate; then compute the statistics of those conditional
        statistics.

        Args:
            mc_loader (torch.utils.data.DataLoader): dataloader for the Monte 
                Carlo data (10,000 is used in this work)

            S: num of samples
            M: num of data
            D: output dimensions
        """
        # First compute conditional statistics
        # S x N x oC x oH x oW
        # self.cpu()
        # x_test = x_test.cpu()
        # print('here')

        # S x oC x oH x oW
        output_size = mc_loader.dataset[0][1].size()
        cond_Ey = torch.zeros(self.n_samples, *output_size, device=device)
        cond_Eyy = torch.zeros_like(cond_Ey)

        for _, (x_mc, _) in enumerate(mc_loader):
            x_mc = x_mc.to(device)
            # S x B x oC x oH x oW            
            y = self.forward(x_mc)
            cond_Ey += y.mean(1)
            cond_Eyy += y.pow(2).mean(1)
        cond_Ey /= len(mc_loader)
        cond_Eyy /= len(mc_loader)
        beta_inv = (- self.log_beta).exp()
        print('Noise variances: {}'.format(beta_inv))
        
        y_cond_pred_var = cond_Eyy - cond_Ey ** 2 \
                     + beta_inv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # compute statistics of conditional statistics
        return cond_Ey.mean(0), cond_Ey.var(0), \
               y_cond_pred_var.mean(0), y_cond_pred_var.var(0)
    '''
