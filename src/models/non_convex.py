import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange


class TwoLayerReLU(nn.Module):
    
    def __init__(self, dXin, dW1, dYpred=1):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(TwoLayerReLU, self).__init__()
        self.device = torch.device(device) # use gpu if availble
        self.net = nn.Sequential(
            nn.Linear(dXin, dW1, bias=False),
            nn.ReLU(),
            nn.Linear(dW1, dYpred, bias=False)
        ).to(self.device)
        
        self.net.apply(self.init_weights)
        return

    def init_weights(self, l):
        if isinstance(l, nn.Linear):
            nn.init.xavier_uniform_(l.weight)
            # l.bias.data.fill_(0.01)
        return

    def forward(self, x):
        x = self.net(x)
        return x

    def loss_with_reg(self, pred, y, beta=1e-2):
      '''
      Loss Function from Equation 2 in https://arxiv.org/pdf/2002.10553.pdf
      Code sourced from https://github.com/pilancilab/convex_nn/blob/e401184311dafbfa5ef9196941d3ddf003823fa4/convexnn_pytorch_stepsize_fig.py#L293
      '''
      loss = 0.5 * torch.norm(pred - y)**2
      
      ## l2 norm on first layer weights, l1 squared norm on second layer
      for layer, p in enumerate(self.net.parameters()):
          if layer == 0:
              loss += beta/2 * torch.norm(p)**2
          else:
              loss += beta/2 * sum([torch.norm(p[:, j], 1)**2 for j in range(p.shape[1])])
      
      return loss

    def train(self, train_data, test_data, n_epochs, lr, beta):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainloss=[] 
        testloss=[]


        lossfn=self.loss_with_reg
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)


        progress_bar=trange(n_epochs)
        for e in progress_bar:
            trainloss_epoch=0
            testloss_epoch=0
            for i, train_point in enumerate(train_data):
                # Forward propagation of training data.
                features = train_point[:, :-1].float().to(self.device)
                label = train_point[:, [-1]].float().to(self.device)
                pred = self.net(features)
                loss = lossfn(pred, label, beta) # REGULARIZATION NEEDS TO BE ADDED, see eqn (2) in http://proceedings.mlr.press/v119/pilanci20a/pilanci20a.pdf
                trainloss_epoch+=loss.cpu().detach().item()
            
                # Back propagation w/ SGD to update network weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Perform network loss analysis on testing data. (don't update)
            predictions = np.zeros((len(test_data)))
            for j, test_point in enumerate(test_data):
                features = test_point[:, :-1].float().to(self.device)
                label = test_point[:, [-1]].float().to(self.device)
                pred = self.net(features)
                predictions[j] = pred
                testloss_epoch += lossfn(pred, label, beta).cpu().detach().item()


            trainloss.append(trainloss_epoch / len(train_data))
            testloss.append(testloss_epoch / len(test_data))
            progress_bar.set_description(f'{trainloss[-1]:.4f}')

        return (trainloss, testloss, predictions)
