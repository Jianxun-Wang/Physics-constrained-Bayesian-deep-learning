import argparse
import torch
import json
import random
from pprint import pprint
#from utils.misc import mkdirs
import time

# always uses cuda if avaliable

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='FCN with SVGD')       

        # model
        self.add_argument('-ns', '--n-samples', type=int, default=5, help='(5-30) number of model instances in SVGD')        

        # data
        self.add_argument('--ntrain', type=int, default=4500, help="number of training data")

        
        # training
        self.add_argument('--epochs', type=int, default=2000, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=1e-3, help='learnign rate')
        self.add_argument('--lr-noise', type=float, default=1e-5, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=50, help='batch size for training')
        #self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
        self.add_argument('--noise', type=float, default=1e-6, help='noise in prior')
        self.add_argument('--bound_sample', type = int, default = 10, help= 'number of boundary sample' )
        #####
        # for stenosis_hard use only
        self.add_argument('--mu', type=float, default=0.5, help='narrowest/widest part of the stenosis/anuerysm')
        self.add_argument('--sigma', type = float, default = 0.1, help = 'degree of stenosis/aneurysm, remain constant')
        self.add_argument('--scale', type = float, default = 0.005, help = 'control the degree of stenosis/aneurysm, is tunable')
        self.add_argument('--nu', type = float, default = 1e-3, help = 'viscosity')
        self.add_argument('--rho', type = float, default = 1, help = 'fluid density')
        self.add_argument('--rInlet', type = float, default = 0.05, help = 'inlet width')
        self.add_argument('--xStart', type = float, default = 0, help = 'start of x')
        self.add_argument('--xEnd', type = float, default = 1, help = 'end of x')
        self.add_argument('--L', type = float, default = 1, help = 'length of the L')
        self.add_argument('--dP', type = float, default = 0.1, help = 'prssure drop')

        

    def parse(self):
        args = self.parse_args()
        

        # seed
        #if args.seed is None:
            #args.seed = random.randint(1, 10000)
        #print("Random Seed: ", args.seed)
        #random.seed(args.seed)
        #torch.manual_seed(args.seed)

        print('Arguments:')
        pprint(vars(args))
        

        return args

# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
