import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
import copy
import math
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.ticker as ticker
from time import time
import time as t
import sys
import os
import gc

## Import local modules (Prof.JX-W's python code)
RWOF_dir = os.path.expanduser("/home/luning/Documents/utility/pythonLib")
RWOF_dir_1 = os.path.expanduser("/home/luning/Documents/utility/pythonLib/python_openFoam")
sys.path.append(RWOF_dir)
sys.path.append(RWOF_dir_1)
import subprocess # Call the command line
from subprocess import call
# import the modules you need
#import foamFileOperation as foamOp
from FCN import Net
from BayesNN import BayesNN
from helpers import log_sum_exp, parameters_to_vector, vector_to_parameters, _check_param_device
from args import args, device

n_samples = args.n_samples
lr = args.lr
lr_noise = args.lr_noise
ntrain = args.ntrain
class SVGD(object):
    """
    Args:
        model (nn.Module): The model to be instantiated `n_samples` times
        data_loader (utils.data.DataLoader): For training and testing
        n_samples (int): Number of samples for uncertain parameters
    """

    def __init__(self, bayes_nn,train_loader):
        """
        For-loop implementation of SVGD.
        Args:
            bayes_nn (nn.Module): Bayesian NN
            train_loader (utils.data.DataLoader): Training data loader
            logger (dict)
        """
        self.bayes_nn = bayes_nn
        self.train_loader = train_loader
        self.n_samples = n_samples
        self.optimizers, self.schedulers = self._optimizers_schedulers(
                                            lr, lr_noise)
    def _squared_dist(self, X):
        """Computes squared distance between each row of `X`, ||X_i - X_j||^2
        Args:
            X (Tensor): (S, P) where S is number of samples, P is the dim of 
                one sample
        Returns:
            (Tensor) (S, S)
        """
        XXT = torch.mm(X, X.t())
        XTX = XXT.diag()
        return -2.0 * XXT + XTX + XTX.unsqueeze(1)
    


    def _Kxx_dxKxx(self, X):
        
        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in eqn (8)
        of reference SVGD paper.
        Args:
            X (Tensor): (S, P), design matrix of samples, where S is num of
                samples, P is the dim of each sample which stacks all params
                into a (1, P) row. Thus P could be 1 millions.
        """
        squared_dist = self._squared_dist(X)
        l_square = 0.5 * squared_dist.median() / math.log(self.n_samples)
        Kxx = torch.exp(-0.5 / l_square * squared_dist)
        # matrix form for the second term of optimal functional gradient
        # in eqn (8) of SVGD paper
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square

        return Kxx, dxKxx


        '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    ''' 
    def svgd_kernel(self, h = -1):
        sq_dist = pdist(self.theta)
        pairwise_dists = squareform(sq_dist)**2
        print('shape of theta is',self.theta.shape)
        print('shape of sq_dist is',sq_dist.shape)
        print('shape of pairwise_dists is',pairwise_dists.shape)
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel
        
        Kxy = np.exp( -pairwise_dists / h**2 / 2)
        print('Kxy.shape is',Kxy.shape)
        time.sleep(1)
        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        print('dxkxy.shape is',dxkxy.shape)
        time.sleep(1)
        return (Kxy, dxkxy)
    



    def _optimizers_schedulers(self, lr, lr_noise):
        """Initialize Adam optimizers and schedulers (ReduceLROnPlateau)
        Args:
            lr (float): learning rate for NN parameters `w`
            lr_noise (float): learning rate for noise precision `log_beta`
        """
        optimizers = []
        schedulers = []
        for i in range(self.n_samples):
            parameters = [{'params': [self.bayes_nn[i].log_beta],'lr':lr_noise},
                    {'params': self.bayes_nn[i].features.parameters()}]
            #parameters = [{'params': self.bayes_nn[i].features.parameters()}]

            optimizer_i = torch.optim.Adam(parameters, lr=lr)
            optimizers.append(optimizer_i)
            schedulers.append(ReduceLROnPlateau(optimizer_i, 
                    mode='min', factor=0.99, patience=100000, verbose=True))
        return optimizers, schedulers
    
    def train(self, epoch,data_likelihod, eq_likelihood, rec_lamda, rec_beta_eq, rec_log_beta, rec_log_alpha,LOSS,LOSSB,LOSS1,LOSS2,LOSS3,LOSSD):
        
        self.bayes_nn.train()
        mse = 0.       
        # add noise after sampling
        Data_sparse = np.load('xyuvp_sparse_separate_3sec.npz')
        sp_xin, sp_yin, sp_uin, sp_xout, sp_yout, sp_uout, sp_xdom,sp_ydom,sp_udom,sp_vdom,sp_Pdom,sp_xb,sp_yb,sp_ub,sp_vb,sp_Pb = self._dataset(Data_sparse)

        Data_b = np.load('boundary_data.npz')
        x_left, x_right, x_up, x_down = Data_b['x_l'], Data_b['x_r'], Data_b['x_u'], Data_b['x_d']
        y_left, y_right, y_up, y_down = Data_b['y_l'], Data_b['y_r'], Data_b['y_u'], Data_b['y_d']
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x_l_sample, x_r_sample, x_u_sample, x_d_sample, y_l_sample, y_r_sample, y_u_sample, y_d_sample = self._bound_sample(x_left, x_right, x_up, x_down, y_left, y_right, y_up, y_down)
            # paste boundary
            xb, yb = self._paste_b(x_l_sample, x_r_sample, x_u_sample, x_d_sample, y_l_sample,y_r_sample,y_u_sample,y_d_sample)
            xb.requires_grad = True
            yb.requires_grad = True
            net_inb = torch.cat((xb,yb),1)
            ##add noise
            noise_lv = 0.05
            sp_udom, sp_vdom, sp_Pdom = self._addnoise(noise_lv,sp_udom,sp_vdom,sp_Pdom)
            ## paste sp data
            sp_x, sp_y, sp_u, sp_v, sp_P = self._paste_d(sp_xdom, sp_xb, sp_ydom, sp_yb, sp_udom, sp_ub, sp_vdom, sp_vb, sp_Pdom, sp_Pb)
            sp_P = torch.zeros_like(sp_x)
            ### modified for burgers
            sp_x.reuqires_grad = True
            sp_y.requires_grad = True
            sp_inputs = torch.cat((sp_x,sp_y),1)
            sp_target = torch.cat((sp_u,sp_v,sp_P),1)
            ## paste inlet
            x_in, y_in, u_in = self._paste_in(sp_xin, sp_yin, sp_uin)
            x_in.requires_grad = True
            y_in.requires_grad = True
            inlet_input = torch.cat((x_in,y_in),1)
            inlet_target = torch.cat((u_in,torch.zeros_like(u_in),torch.zeros_like(u_in)),1)
            ## paste outlet
            x_out, y_out, u_out = self._paste_in(sp_xout, sp_yout, sp_uout)
            x_out.requires_grad = True
            y_out.requires_grad = True
            outlet_input = torch.cat((x_out, y_out), 1)
            outlet_target = torch.cat((u_out, torch.zeros_like(u_out), torch.zeros_like(u_out)), 1)
            self.bayes_nn.zero_grad()
            output = torch.zeros_like(x).to(device)
            sp_output = torch.zeros_like(sp_target).to(device)
            outputb = torch.zeros([net_inb.shape[0],3]).to(device)
            outputin = torch.zeros_like(inlet_target)
            outputout = torch.zeros_like(outlet_target)
            # all gradients of log joint probability: (S, P)
            grad_log_joint = []
            # all model parameters (particles): (S, P)
            theta = []
            # store the joint probabilities
            log_joint = 0.
            loss_1, loss_2, loss_3,loss_d, loss_b = 0, 0, 0, 0, 0
            for i in range(self.n_samples):
                #####################
                ###modified for sparse data stenosis
                ## forward for training data
                sp_output_i = self.bayes_nn[i].forward(sp_inputs)
                outputb_i = self.bayes_nn[i].forward(net_inb)
                output_in_i = self.bayes_nn[i].forward(inlet_input)
                output_out_i = self.bayes_nn[i].forward(outlet_input)
                # why a detach here
                sp_output += sp_output_i.detach()
                outputb += outputb_i.detach()
                outputin += output_in_i.detach()
                outputout += output_out_i.detach()
                ## loss for unlabelled points
                log_eq_i,loss_1i,loss_2i,loss_3i =  self.bayes_nn.criterion(i,x,y,ntrain)
                ## loss for labelled points
                log_joint_i = self.bayes_nn._log_joint(i, sp_output_i, sp_target, outputb_i,output_in_i,inlet_target, output_out_i, outlet_target, ntrain)
                log_joint_i_1 = log_joint_i
                ### for monity purpose
                loss_1, loss_2, loss_3, loss_d, loss_b = self._monitor(loss_1, loss_2, loss_3, loss_d, loss_b, loss_1i, loss_2i, loss_3i, sp_output_i, sp_target, outputb_i)
                ###
                log_joint_i += log_eq_i
                log_joint_i.backward()
                if (i==0) :
                    rec_log_beta.append(self.bayes_nn[i].log_beta.item())
                    rec_beta_eq.append(self.bayes_nn[i].beta_eq)
                #####
                # backward frees memory for computation graph
                # computation below does not build computation graph
                # extract parameters and their gradients out from models
                vec_param, vec_grad_log_joint = parameters_to_vector(
                    self.bayes_nn[i].parameters(), both=True)
                grad_log_joint.append(vec_grad_log_joint.unsqueeze(0))
                theta.append(vec_param.unsqueeze(0))
            # calculating the kernel matrix and its gradients
            theta = torch.cat(theta)
            Kxx, dxKxx = self._Kxx_dxKxx(theta)
            grad_log_joint = torch.cat(grad_log_joint)
            grad_logp = torch.mm(Kxx, grad_log_joint)
            # negate grads here!!!
            grad_theta = - (grad_logp + dxKxx) / self.n_samples
            ## switch back to 1 particle
            #grad_theta = grad_log_joint
            # explicitly deleting variables does not release memory :(
       
            # update param gradients
            for i in range(self.n_samples):
                vector_to_parameters(grad_theta[i,:],
                    self.bayes_nn[i].parameters(), grad=True)
                
                self.optimizers[i].step()
            # WEAK: no loss function to suggest when to stop or
            # approximation performance
            #mse = F.mse_loss(output / self.n_samples, target).item()

            loss_1/= self.n_samples
            loss_2/=self.n_samples
            loss_3/=self.n_samples
            loss_d/=self.n_samples
            loss_b/=self.n_samples
            #print('len(self.train_loader)',len(self.train_loader))
            if batch_idx % 20 ==0:
                loss = (loss_1+loss_2+loss_3)/3
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAvgLoss: {:.10f}\tAvgLossB: {:.10f}\tAvgLossD: {:.10f}'.format(
                    epoch, batch_idx * len(x), len(self.train_loader)*len(x),
                    100. * batch_idx / len(self.train_loader), loss.item(),loss_b.item(),loss_d.item()))
                LOSS.append(loss)
                LOSSB.append(loss_b)
                LOSS1.append(loss_1)
                LOSS2.append(loss_2)
                LOSS3.append(loss_3)
                LOSSD.append(loss_d)

        if epoch%100==0:
            self._savept(epoch, rec_log_beta, rec_beta_eq, rec_log_alpha)
        return data_likelihod, eq_likelihood, rec_lamda, rec_beta_eq, rec_log_beta, rec_log_alpha, LOSS,LOSSB,LOSS1,LOSS2,LOSS3,LOSSD
        #np.savetxt('.csv',rec_lamda)
        #for i in range(self.n_samples):
            #self.schedulers[i].step(rmse_train)
    def _plot(self,x,y,u):
        plt.figure()
        plt.scatter(x,y,c = u)
        plt.colorbar()
        plt.show()
    def _dataset(self,Data):
        sp_xin = Data['xinlet']
        sp_yin = Data['yinlet']
        sp_uin = Data['uinlet']
        sp_xout = Data['xoutlet']
        sp_yout = Data['youtlet']
        sp_uout = Data['uoutlet']
        sp_xdom = Data['xdom']
        #print('x_size is',sparse_x.shape)
        sp_ydom = Data['ydom']
        sp_udom = Data['udom']
        sp_vdom = Data['vdom']
        sp_Pdom = Data['pdom']
        sp_xb = Data['xb']
        sp_yb = Data['yb']
        sp_ub = Data['ub']
        sp_vb = Data['vb']
        sp_Pb = Data['pb']
        return sp_xin, sp_yin, sp_uin, sp_xout, sp_yout, sp_uout, sp_xdom, sp_ydom, sp_udom, sp_vdom, sp_Pdom, sp_xb, sp_yb, sp_ub, sp_vb, sp_Pb
    def _bound_sample(self, x_left, x_right, x_up, x_down, y_left, y_right, y_up, y_down, ratio = 4, device = device):
        perm_x_left = np.random.randint(len(x_left), size=args.bound_sample)     
        perm_y_left = perm_x_left

        x_l_sample =torch.Tensor(x_left[perm_x_left]).to(device)
        y_l_sample = torch.Tensor(y_left[perm_y_left]).to(device)
        # right boudanry sample
        perm_x_right = np.random.randint(len(x_right), size=args.bound_sample)
        perm_y_right = perm_x_right

        x_r_sample = torch.Tensor(x_right[perm_x_right]).to(device)
        y_r_sample = torch.Tensor(y_right[perm_y_right]).to(device)
        
        # up boundary sample
        perm_x_up = np.random.randint(len(x_up), size=ratio*args.bound_sample)
        perm_y_up = perm_x_up
        x_u_sample = torch.Tensor(x_up[perm_x_up]).to(device)
        y_u_sample = torch.Tensor(y_up[perm_y_up]).to(device)


        # low boundary sample
        perm_x_down = np.random.randint(len(x_down), size=ratio*args.bound_sample)
        perm_y_down = perm_x_down
        x_d_sample = torch.Tensor(x_down[perm_x_down]).to(device)
        y_d_sample = torch.Tensor(y_down[perm_y_down]).to(device)
        return x_l_sample, x_r_sample, x_u_sample, x_d_sample, y_l_sample, y_r_sample, y_u_sample, y_d_sample
    def _addnoise(self,noise_lv, sp_udom, sp_vdom, sp_Pdom):
        for i in range(0,len(sp_udom)):
            u_error = np.random.normal(0, noise_lv*np.abs(sp_udom[i]), 1)
            
            v_error = np.random.normal(0, noise_lv*np.abs(sp_vdom[i]), 1)
            p_error = np.random.normal(0, noise_lv*np.abs(sp_Pdom[i]), 1)
            sp_udom[i] += u_error
            sp_vdom[i] += v_error
            sp_Pdom[i] += p_error

        return sp_udom, sp_vdom, sp_Pdom
    def _paste_b(self,x_l_sample, x_r_sample, x_u_sample, x_d_sample, y_l_sample,y_r_sample,y_u_sample,y_d_sample, device = device):
        xb =torch.cat((x_l_sample,x_r_sample,x_u_sample,x_d_sample),0).to(device)
        yb = torch.cat((y_l_sample,y_r_sample,y_u_sample,y_d_sample),0).to(device)
        xb = xb.view(len(xb),-1)
        yb = yb.view(len(yb),-1)
        return xb, yb
    def _paste_d(self, sp_xdom, sp_xb, sp_ydom, sp_yb, sp_udom, sp_ub, sp_vdom, sp_vb, sp_Pdom, sp_Pb, device = device):
        ##
        sp_x = np.concatenate((sp_xdom,sp_xb),0)
        sp_y = np.concatenate((sp_ydom,sp_yb),0)
        sp_u = np.concatenate((sp_udom,sp_ub),0)
        sp_v = np.concatenate((sp_vdom,sp_vb),0)
        sp_P = np.concatenate((sp_Pdom,sp_Pb),0)
        sp_x, sp_y, sp_u, sp_v, sp_P = sp_x[...,None], sp_y[...,None], sp_u[...,None], sp_v[...,None], sp_P[...,None]
        sp_data = np.concatenate((sp_x,sp_y,sp_u,sp_v,sp_P),1)
        
        
        ##
        # for sparase stenosis
        sp_x, sp_y, sp_u, sp_v, sp_P = torch.Tensor(sp_data[:,0]).to(device), torch.Tensor(sp_data[:,1]).to(device), torch.Tensor(sp_data[:,2]).to(device), torch.Tensor(sp_data[:,3]).to(device), torch.Tensor(sp_data[:,4]).to(device)
        sp_x, sp_y, sp_u, sp_v, sp_P = sp_x.view(len(sp_x), -1), sp_y.view(len(sp_y), -1), sp_u.view(len(sp_u), -1), sp_v.view(len(sp_v), -1), sp_P.view(len(sp_P),-1)
        return sp_x, sp_y, sp_u, sp_v, sp_P
    def _paste_in(self,x_in, y_in, u_in,device = device):
        x_in = torch.Tensor(x_in).to(device)
        y_in = torch.Tensor(y_in).to(device)
        u_in = torch.Tensor(u_in).to(device)
        x_in, y_in, u_in = x_in.view(len(x_in),-1), y_in.view(len(y_in),-1), u_in.view(len(u_in),-1)
        return x_in, y_in, u_in

    def _monitor(self, loss_1, loss_2, loss_3, loss_d, loss_b, loss_1i, loss_2i, loss_3i, sp_output_i, sp_target, outputb_i):
        loss_f = nn.MSELoss()
        loss_1 += loss_f(loss_1i,torch.zeros_like(loss_1i))
        loss_2 += loss_f(loss_2i,torch.zeros_like(loss_2i))
        loss_3 += loss_f(loss_3i,torch.zeros_like(loss_3i))
        loss_d += loss_f(sp_output_i, sp_target)
        loss_b += loss_f(outputb_i, torch.zeros_like(outputb_i))

        return loss_1, loss_2, loss_3, loss_d, loss_b
    def _savept(self, epoch, rec_log_beta, rec_beta_eq, rec_log_alpha):
        torch.save(self.bayes_nn.state_dict(),"test"+str(epoch)+".pt")

        np.savetxt('log_beta.csv',rec_log_beta)
        np.savetxt('beta_eq.csv',rec_beta_eq)
        np.savetxt('log_alpha.csv',rec_log_alpha)

