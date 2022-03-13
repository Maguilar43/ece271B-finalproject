#!/usr/bin/env python3

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange


class ConvexStockNet(nn.Module):

    def __init__(self, n_features, n_regions, n_outputs=1):
        '''
        Args:
            n_features (int): The number of features per data point.
            n_regions  (int): The number of data masks D_i to be created (see eq. 8).
            n_outputs  (int): The number of outputs from the network.

            NOTE: The number of hidden units in the resulting ReLU network
                  is influenced by n_regions.
                    n_units <= n_regions
                  depending on the sparisty of the convex problem solution.
        '''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(ConvexStockNet, self).__init__()

        # Network Information
        self.device = torch.device(device)
        self.n_features = n_features
        self.n_regions = n_regions
        self.n_outputs = n_outputs

        # Convex Problem Variables (Don't move parameters to device, move model to device)
        self.U = nn.Parameter(data=torch.zeros(self.n_features, self.n_regions), requires_grad=True)
        self.V = nn.Parameter(data=torch.zeros(self.n_features, self.n_regions), requires_grad=True)

        # Create random sample of sign patterns (keep consistent for all batches)
        self.S = torch.randn(self.n_features, self.n_regions).to(device)

        return

    def forward(self, x):

        # worreid the translation doesn't work
        # x = x @ self.W1
        # x = F.relu(x)
        # x = x @ self.W2

        Dmat = torch.sign(F.relu(x @ self.S))
        Xu, Xv = torch.matmul(x, self.U), torch.matmul(x, self.V)
        DXu, DXv = torch.mul(Dmat, Xu), torch.mul(Dmat, Xv)
        x = DXu.sum(axis=1) - DXv.sum(axis=1)

        return x

    def train(self, train_set, test_set, n_epochs, lr=1e-20, lamb=1e-2, rho=1e-2):
        '''
        Args:
            train_set (DataLoader): The dataset to be used for training.
            test_set  (DataLoader): The dataset to be used for validation.
            n_epochs         (int): The number iterations through each batch.
            lr             (float): Learning rate of SGD optimizer.
            lamb           (float): Lagrange multiplier for l2l1-norm term. 1e-2
            rho            (float): Lagrange multiplier for constraints.
        '''

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        n_examples = train_set.batch_size

        residual_loss = torch.nn.MSELoss(reduction='sum')
        # residual_loss = torch.nn.MSELoss(reduction='sum')
        test_loss_fn=nn.MSELoss()
        
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        train_loss = []

        train_loss_per_epoch=[] # added this to just look at visulation
        test_loss_per_epoch=[]   # non convex test
        non_conv_train_loss_per_epoch=[] # non conv  
        non_conv_train_loss_per_epoch2=[] #slop  

        test_residual_loss=[]

        progress_bar = trange(n_epochs)
        for t in progress_bar:
            epoch_train_loss=0 # added this for visualization temporarily
            non_conv_epoch_train_loss=0
            for i, batch in enumerate(train_set):
                batch = batch.to(self.device).float()
                data, labels = batch[:, :-1], batch[:, -1]

                Dmat = torch.sign(F.relu(data @ self.S))

                Xu = torch.matmul(data, self.U)
                Xv = torch.matmul(data, self.V)
                DXu = torch.mul(Dmat, Xu)
                DXv = torch.mul(Dmat, Xv)

                # Term 1: Residual L2 Error
                residual_error = residual_loss(DXu.sum(axis=1) - DXv.sum(axis=1), labels)
                non_conv_epoch_train_loss+=residual_error.cpu().detach().item()

                # Term 2: Mixed-Norm Error
                norm_U = torch.linalg.norm(self.U, axis=0).sum()
                norm_V = torch.linalg.norm(self.V, axis=0).sum()

                # Term 3: Constraints
                # NOTE: Pilanci's code has a max() in there, not sure why.
                # NOTE: Also, I can use sum here rather than inner product assuming
                #       that all the rho's are the same.
                constraint = torch.sum(F.relu(-2*DXu + Xu)) + torch.sum(F.relu(-2*DXv + Xv))

                # Total Loss
                total_loss = residual_error + lamb * (norm_U + norm_V) + rho * constraint
                t_loss = total_loss.cpu().detach().item()
                epoch_train_loss+=t_loss; # added this for visualization temporarily
                train_loss.append(t_loss)
                progress_bar.set_description(f'{t_loss:.4f}')

                # Back propagation w/ SGD to update network weights.
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            train_loss_per_epoch.append(epoch_train_loss/ len(train_set)) # just changed this for visualization temporarily
            non_conv_train_loss_per_epoch.append(non_conv_epoch_train_loss/ len(train_set))

            self.W2 = torch.linalg.norm(self.U, axis=0)
            self.W1 = self.U / self.W2

            non_conv_epoch_train_loss2=0
            for i, batch in enumerate(train_set):
                batch = batch.to(self.device).float()
                data, labels = batch[:, :-1], batch[:, -1]
                preds=self.forward(data)
                non_conv_epoch_train_loss2+=test_loss_fn(preds, labels).cpu().detach().item()

            non_conv_train_loss_per_epoch2.append(non_conv_epoch_train_loss2/ len(train_set))

            epoch_test_loss=0
            res_test_loss=0
            for i, batch in enumerate(test_set):
                batch = batch.to(self.device).float()
                data, labels = batch[:, :-1], batch[:, -1]
                preds=self.forward(data)
                epoch_test_loss+=test_loss_fn(preds, labels).cpu().detach().item()

                Dmat = torch.sign(F.relu(data @ self.S))
                Xu = torch.matmul(data, self.U)
                Xv = torch.matmul(data, self.V)
                DXu = torch.mul(Dmat, Xu)
                DXv = torch.mul(Dmat, Xv)

                # Term 1: Residual L2 Error
                residual_error = residual_loss(DXu.sum(axis=1) - DXv.sum(axis=1), labels)
                res_test_loss+=residual_error.cpu().detach().item()

            test_residual_loss.append(res_test_loss/len(test_set))

            test_loss_per_epoch.append(epoch_test_loss/ len(test_set))

        # return np.asarray(train_loss)

        self.W2 = torch.linalg.norm(self.U, axis=0)
        self.W1 = self.U / self.W2

        return (np.asarray(train_loss), np.asarray(train_loss_per_epoch),np.asarray(non_conv_train_loss_per_epoch),np.asarray(test_loss_per_epoch),np.asarray(non_conv_train_loss_per_epoch2), np.asarray(test_residual_loss),self.U)
