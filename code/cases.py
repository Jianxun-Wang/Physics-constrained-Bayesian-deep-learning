import pandas as pd
import numpy as np
import matplotlib.pylab as plt
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
import sys
import os
import gc
import time
from FCN import Net
from BayesNN import BayesNN
from helpers import log_sum_exp, parameters_to_vector, vector_to_parameters, _check_param_device
from SVGD import SVGD
from args import args,device
import random
random.seed(20)
class cases:
	def __init__(self,bayes_nn,name):
		self.name = name
		self.epochs = args.epochs
		self.bayes_nn = bayes_nn
		if self.name == 'regression':
			self.noise_data =1
	
	def dataloader(self):
		if self.name == 'regression':
			train_size = args.batch_size
			

			X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
			y = self._f(X, sigma=self.noise_data)
			y_true = self._f(X, sigma=0.0)
			plt.scatter(X, y, marker='+', label='Training data')
			plt.plot(X, y_true, label='Truth')
			plt.title('Noisy training data and ground truth')
			plt.legend();
			plt.show()

			X_train, Y_train = torch.Tensor(X), torch.Tensor(y)                                     
			X_test, Y_test = torch.Tensor(X), torch.Tensor(y)

			data = torch.utils.data.TensorDataset(X_train, Y_train)

			train_loader = torch.utils.data.DataLoader(data, batch_size=train_size, shuffle=True)
			return train_loader,train_size

		elif self.name == 'stenosis_hard':
			train_size = args.batch_size
			N_y = 30
			L = 1
			xStart = 0
			xEnd = xStart+L
			rInlet = 0.05

			nPt = 100
			unique_x = np.linspace(xStart, xEnd, nPt)
			sigma = 0.1
			scale = 0.005
			mu = 0.5*(xEnd-xStart)
			x_2d = np.tile(unique_x,N_y)
			x = x_2d
			x = np.reshape(x,(len(x),1))

			Data = np.load('xyuvp_uinlet.npz')
			x = Data['x']
			y = Data['y']
			u = Data['u']
			v = Data['v']
			P = Data['p']
			x = x[...,None]
			y = y[...,None]
			u = u[...,None]
			v = v[...,None]
			P = P[...,None]
			#print('x.shape is',x.shape)
			R = scale * 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
			nu = 1e-3
			yUp = rInlet - R
			yDown = -rInlet + R
			plt.scatter(x, yUp)
			plt.scatter(x, yDown)
			plt.scatter(x,y)
			plt.axis('equal')
			plt.show()
			############################

			np.savez('stenosis_hard_coord',x = x,y = y,yUp = yUp,u = u, v = v, P = P)
			################
			data = torch.utils.data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y))
			
			train_loader = torch.utils.data.DataLoader(data, batch_size=train_size, shuffle=True)
			print('len(data is)', len(data))
			print('len(dataloader is)', len(train_loader))
			return train_loader,train_size
		else:
			raise Exception("error,no such model") 
	def plot(self):
		if self.name == 'regression':
			"""Evaluate model during training. 
			Print predictions including 4 rows:
				1. target
				2. predictive mean
				3. error of the above two
				4. two sigma of predictive variance
			Args:
				test_fixed (Tensor): (2, N, *), `test_fixed[0]` is the fixed test input, 
					`test_fixed[1]` is the corresponding target
			"""
			self.bayes_nn.load_state_dict(torch.load("test.pt",map_location = 'cpu'))
			self.bayes_nn.eval()
			
			x = np.linspace(-0.5, 0.5, 100).reshape(-1, 1)
			y = self._f(x, sigma=self.noise_data)
			ytrue = self._f(x, sigma=0.0)
			xt,yt = torch.Tensor(x), torch.Tensor(y)
			xt, yt = xt.to(device), yt.to(device)
			with torch.no_grad():
				y_pred_mean, y_pred_var = self.bayes_nn.predict(xt)

			pred = y_pred_mean.cpu().detach().numpy().ravel()
			var = y_pred_var.cpu().detach().numpy().ravel()
			plt.figure()
			plt.plot(x.ravel() ,pred,label = 'Prediction')
			plt.scatter(x,y,label  ='Data')
			plt.plot(x, ytrue, label='Truth')
			plt.fill_between(x.ravel(),pred+var,pred - var,alpha = 0.5,label = 'Uncertainty')
			plt.legend()
			plt.show()


		elif self.name =='stenosis_hard':
			#self.bayes_nn.load_state_dict(torch.load("test1e-2_1p.pt",map_location = 'cpu'))
			self.bayes_nn.load_state_dict(torch.load("test1500.pt",map_location = 'cpu'))

			self.bayes_nn.eval()
			Data = np.load('stenosis_hard_coord.npz')
			x = Data['x']
			y = Data['y']
			u = Data['u']
			v = Data['v']
			P = Data['P']
			u_CFD = u
			v_CFD = v
			P_CFD = P
			print('u_CFD is', u_CFD)
			print('v_CFD is', v_CFD)
			yUp = Data['yUp']
			xt,yt = torch.Tensor(x), torch.Tensor(y)
			Rt= torch.Tensor(yUp).to(device)
			print('Rt.requires_grad is',Rt.requires_grad)
			xt,yt = xt.view(len(xt),-1), yt.view(len(yt),-1)
			xt.requires_grad = True
			yt.requires_grad = True
			xt, yt = xt.to(device), yt.to(device)
			inputs = torch.cat((xt,yt),1)
			#with torch.no_grad():
			print('inputs is',inputs)
			y_pred_mean = self.bayes_nn.forward(inputs)
			#pred = y_pred_mean.cpu().detach().numpy()
			pred = y_pred_mean

			for i in range (0,pred.shape[0]):
				# hard constraint u
				#pred[i,:,0] *= (Rt[:,0]**2 - yt[:,0]**2)
				# hard constraint v
				#pred[i,:,1] *= (Rt[:,0]**2 -yt[:,0]**2)
				# hard constraint P
				pred[i,:,2] = (args.xStart-xt[:,0])*0 + args.dP*(args.xEnd-xt[:,0])/args.L + 0*yt[:,0] + (args.xStart - xt[:,0])*(args.xEnd - xt[:,0])*pred[i,:,2] 
			print('pred.shape is',pred.shape)
			mean = pred.mean(0)
			EyyT = (pred ** 2).mean(0)
			EyEyT = mean ** 2
			beta_inv = (- self.bayes_nn.log_beta).exp()
			print('beta_inv.mean',beta_inv.mean())
			var = beta_inv.mean() + EyyT - EyEyT

			#var = (pred.std(0))**2
			print('mean.shape',mean.shape)
			print('var.shape',var.shape)
			u_hard = mean[:,0]
			v_hard = mean[:,1]
			P_hard = mean[:,2]
			u_hard = u_hard.view(len(u_hard),-1)
			v_hard = v_hard.view(len(v_hard),-1)
			P_hard = P_hard.view(len(P_hard),-1)
			u_hard = u_hard.cpu().detach().numpy()
			v_hard = v_hard.cpu().detach().numpy()
			P_hard = P_hard.cpu().detach().numpy()
			var_u = var[:,0]
			var_v = var[:,1]
			var_P = var[:,2]
			var_u = var_u.view(len(var_u),-1)
			var_v = var_v.view(len(var_v),-1)
			var_P = var_P.view(len(var_P),-1)
			var_u = var_u.cpu().detach().numpy()
			var_v = var_v.cpu().detach().numpy()
			var_P = var_P.cpu().detach().numpy()
	

			#plot_x = 0.4
			#plot_y = 0.045
			plot_x= 0.4*np.max(x)
			plot_y = 0.95*np.max(y)
			fontsize = 18
			#axis_limit = [-0.5, 0.5, -0.5, 0.2]
			noise_lv = 0.05
			print('shape of u is',u.shape)
			print('shape of v is',v.shape)
			print('shape of P is',P.shape)
			u_noiseCFD = np.zeros_like(u)
			v_noiseCFD = np.zeros_like(v)
			P_noiseCFD = np.zeros_like(P)
			for i in range(0,len(u)):
				u_error = np.random.normal(0, noise_lv*np.abs(u[i]), 1)
				#print('std is',noise_lv*np.abs(sparse_udom[i]))
				#print('np.random.normal(0, noise_lv*np.abs(sparse_udom[i]), 1)',np.random.normal(0, noise_lv*np.abs(sparse_udom[i]), 1))
				v_error = np.random.normal(0, noise_lv*np.abs(v[i]), 1)
				p_error = np.random.normal(0, noise_lv*np.abs(P[i]), 1)
				u_noiseCFD[i] = u[i] + u_error
				v_noiseCFD[i] = v[i] + v_error
				P_noiseCFD[i] = P[i] + p_error
			

			Data_sparse = np.load('xyuvp_sparse_separate_3sec.npz')
			sparse_x = Data_sparse['xdom']
			print('x_size is',sparse_x.shape)
			sparse_y = Data_sparse['ydom']
			sparse_u = Data_sparse['udom']
			sparse_v = Data_sparse['vdom']
			xinlet  = Data_sparse['xinlet']
			yinlet = Data_sparse['yinlet']
			uinlet = Data_sparse['uinlet']
			xoutlet = Data_sparse['xoutlet']
			youtlet = Data_sparse['youtlet']
			uoutlet = Data_sparse['uoutlet']
			xb = Data_sparse['xb']
			yb = Data_sparse['yb']
			ub = Data_sparse['ub']
			xb_full = Data_sparse['xb_full']
			yb_full = Data_sparse['yb_full']


			xtrain = np.concatenate((xinlet,xoutlet,sparse_x),0)
			ytrain = np.concatenate((yinlet, youtlet, sparse_y),0)

			##
			loss_f = nn.MSELoss()
			print('u_hard is', u_hard)
			print('u_CFD is', u_CFD)

			# accruacy of u
			error_u = loss_f(torch.Tensor(u_hard),torch.Tensor(u_CFD)).item()
			# accuracy of v
			error_v = loss_f(torch.Tensor(v_hard),torch.Tensor(v_CFD)).item()
			# accuracy of P
			error_P = loss_f(torch.Tensor(P_hard),torch.Tensor(P_CFD)).item()
			
			## relative norm
			ut = torch.Tensor(u_CFD)
			vt = torch.Tensor(v_CFD)
			pt = torch.Tensor(P_CFD)

			u_CFDnorm = loss_f(ut ,torch.zeros_like(ut)).item()
			v_CFDnorm = loss_f(vt, torch.zeros_like(vt)).item()
			P_CFDnorm = loss_f(pt, torch.zeros_like(pt)).item()

			print('u_CFDnorm is', np.sqrt(u_CFDnorm))
			print('v_CFDnorm is', np.sqrt(v_CFDnorm))
			print('P_CFDnorm is', np.sqrt(P_CFDnorm))
			
			np.savetxt('u_CFDnorm.csv', np.array([np.sqrt(u_CFDnorm)]))
			np.savetxt('v_CFDnorm.csv', np.array([np.sqrt(v_CFDnorm)]))
			np.savetxt('P_CFDnorm.csv', np.array([np.sqrt(P_CFDnorm)]))
			

			relative_error_u = np.sqrt(error_u/u_CFDnorm)
			relative_error_v = np.sqrt(error_v/v_CFDnorm)
			relative_error_P = np.sqrt(error_P/P_CFDnorm)

			print('relative norm |u - u_CFD|/|u_CFD|', relative_error_u)
			print('relative norm |v - v_CFD|/|v_CFD|', relative_error_v)
			print('relative norm |P - P_CFD|/|P_CFD|', relative_error_P)

			np.savetxt('Relative_error_u.csv',np.array([relative_error_u]))
			np.savetxt('Relative_error_v.csv',np.array([relative_error_v]))
			np.savetxt('Relative_error_P.csv',np.array([relative_error_P]))

			###
			
			## Std u mean
			uq_u_mean = np.sqrt(var_u).mean()
			## Std v mean
			uq_v_mean = np.sqrt(var_v).mean()
			## Std P mean
			uq_P_mean = np.sqrt(var_P).mean()

			## Std u max
			uq_u_max = np.sqrt(var_u).max()
			## Std v max
			uq_v_max = np.sqrt(var_v).max()
			## Std P max
			uq_P_max = np.sqrt(var_P).max()
			#
			#print('uq_u.shape is', uq_u.shape)
			np.savetxt('error_u.csv',np.array([error_u]))
			np.savetxt('error_v.csv',np.array([error_v]))
			np.savetxt('error_P.csv',np.array([error_P]))

			np.savetxt('uq_umean.csv',np.array([uq_u_mean]))
			np.savetxt('uq_vmean.csv',np.array([uq_v_mean]))
			np.savetxt('uq_Pmean.csv',np.array([uq_P_mean]))

			np.savetxt('uq_umax.csv',np.array([uq_u_max]))
			np.savetxt('uq_vmax.csv',np.array([uq_v_max]))
			np.savetxt('uq_Pmax.csv',np.array([uq_P_max]))

			print('test loss u is',error_u)
			print('test loss v is',error_v)
			print('test loss P is',error_P)

			print('mean uq u is',uq_u_mean)
			print('mean uq v is',uq_v_mean)
			print('mean uq P is',uq_P_mean)

			print('max uq u is',uq_u_max)
			print('max uq v is',uq_v_max)
			print('max uq P is',uq_P_max)

			plt.figure()
			plt.subplot(2,1,1)
			#plt.scatter(x, y, c= np.sqrt(var_u)/u_hard, label = 'u_hard_var')
			plt.scatter(x, y, c= np.sqrt(var_u), label = 'u_hard_std', cmap = 'coolwarm')
			plt.text(plot_x, plot_y, r'u Std', {'color': 'b', 'fontsize': fontsize})
			#plt.axis('equal')
			plt.colorbar()
			plt.savefig('softuNN_std_noise15.png',bbox_inches = 'tight')

			plt.figure()
			plt.subplot(2,1,1)
			plt.scatter(x, y, c= u_hard, label = 'uhard', cmap = 'coolwarm', vmin = min(u_CFD), vmax = max(u_CFD))
			plt.text(plot_x, plot_y, r'u Mean', {'color': 'b', 'fontsize': fontsize})
			plt.colorbar()
			#plt.axis('equal')
			plt.savefig('softuNN_mean_noise15.png',bbox_inches = 'tight')

			plt.figure()
			plt.subplot(2,1,1)
			plt.scatter(x, y, c= u_noiseCFD, label = 'u CFD', cmap = 'coolwarm', vmin = min(u_CFD), vmax = max(u_CFD))
			plt.colorbar()
			plt.scatter(xtrain, ytrain, marker = 'x', c = 'black')
			plt.text(plot_x, plot_y, r'u CFD', {'color': 'b', 'fontsize': fontsize})
			#plt.scatter(x, y, c= np.sqrt(var_v), label = 'v_hard_std')
			#plt.scatter(x,y, c = np.sqrt(var_v)/v_hard, label = 'v_hard_std')
			#plt.axis('equal')
			
			plt.savefig('u_CFD_noise15.png',bbox_inches = 'tight')


			plt.figure()
			plt.subplot(2,1,1)
			#plt.scatter(x, y, c= np.sqrt(var_u)/u_hard, label = 'u_hard_var')
			plt.scatter(x, y, c= np.sqrt(var_v), label = 'u_hard_std',vmin = 0.001, cmap = 'coolwarm')
			plt.text(plot_x, plot_y, r'v Std', {'color': 'b', 'fontsize': fontsize})
			plt.colorbar()
			#plt.savefig('u_hard_var.png',bbox_inches = 'tight')
			#plt.figure()
			plt.subplot(2,1,2)
			plt.scatter(x, y, c= v_hard, label = 'uhard', cmap = 'coolwarm')
			plt.text(plot_x, plot_y, r'v Mean', {'color': 'b', 'fontsize': fontsize})
			#plt.scatter(x, y, c= np.sqrt(var_v), label = 'v_hard_std')
			#plt.scatter(x,y, c = np.sqrt(var_v)/v_hard, label = 'v_hard_std')
			plt.colorbar()
			plt.savefig('v_hard_var.png',bbox_inches = 'tight')
			plt.figure()
			plt.scatter(x,y, c = P_hard, label = 'P_hard_std', cmap = 'coolwarm')
			plt.colorbar()
			plt.savefig('P_hard_var.png',bbox_inches = 'tight')
			#plt.scatter(x,y,label  ='Data')
			plt.show()

			print('mean of stdvar_u',np.mean(np.sqrt(var_u)))
			print('mean of std var_v',np.mean(np.sqrt(var_v)))

		else:
			raise Exception("error,no such model") 


	def _f(self,x, sigma):
		epsilon = np.random.randn(*x.shape) * sigma
		return  10*np.sin(2 * np.pi * (x)) + epsilon



