#!/usr/bin/env python3

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ConvexStockNet(nn.Module):

    def __init__(self, n_features, n_regions, n_outputs=1):

        super(ConvexStockNet, self).__init__()

        self.params = {
            'n_features': n_features,
            'n_regions': n_regions,
            'n_outputs': n_outputs
        }

        self.U = nn.Parameter(data=torch.zeros(n_features, n_regions).normal_(std=1/n_regions), requires_grad=True)
        self.V = nn.Parameter(data=torch.zeros(n_features, n_regions).normal_(std=1/n_regions), requires_grad=True)
        self.S = torch.randn(n_features, n_regions)

        return

    def forward(self, X):
        Dmat = torch.sign(F.relu(X @ self.S))
        Xu_v = X @ (self.U - self.V)
        DXu_v = Dmat * Xu_v
        Y_hat = DXu_v.sum(dim=1, keepdim=False)
        return Y_hat

    def train(self, train_set, val_set, n_epochs, lr=1e-20, beta=1e-2, rho=1e-2):

        l2_loss = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        n_batch = len(train_set)
        relu = nn.ReLU()

        train_loss = np.zeros([n_epochs])
        val_loss = np.zeros([n_epochs])

        progress_bar = tqdm(range(n_epochs))
        for t in progress_bar:
            for i, batch in enumerate(train_set):

                # Process data
                batch = batch.to(self.U.device).float()
                data, labels = batch[:, :-1], batch[:, -1]

                # Compute terms of convex problem.
                '''
                Loss Function from Equation 8 in https://arxiv.org/pdf/2002.10553.pdf
                '''
                self.S = self.S.to(self.U.device)
                Dmat = torch.sign(F.relu(data @ self.S))
                Xu = data @ self.U
                Xv = data @ self.V
                DXu = (Dmat * Xu).sum(axis=1)
                DXv = (Dmat * Xv).sum(axis=1)
                l2_norm_error = 0.5 * l2_loss(DXu - DXv, labels)
                mixed_norm_error = torch.linalg.norm(self.U, dim=0).sum() + torch.linalg.norm(self.V, dim=0).sum()
                constraint1 = -2 * DXu + Xu.sum(axis=1)
                constraint2 = -2 * DXu + Xu.sum(axis=1)
                constraint = torch.sum(constraint1) + torch.sum(constraint2)

                # Objective
                obj = l2_norm_error + beta * mixed_norm_error + rho * constraint

                # Use SGD to optimize
                optimizer.zero_grad()
                obj.backward()
                optimizer.step()
                                
                train_loss[t] += obj.cpu().detach().item() / len(train_set)

            progress_bar.set_description(f'Obj Loss = {train_loss[t]:.3f}')

            # Test
            predictions = []
            for j, batch in enumerate(val_set):
                batch = batch.to(self.U.device).float()
                data, labels = batch[:, :-1], batch[:, -1]
                Dmat = torch.sign(F.relu(data @ self.S))
                Xu = data @ self.U
                Xv = data @ self.V
                DXu = (Dmat * Xu).sum(axis=1)
                DXv = (Dmat * Xv).sum(axis=1)
                l2_norm_error = l2_loss(DXu - DXv, labels)
                mixed_norm_error = torch.norm(self.U, dim=0).sum() + torch.norm(self.V, dim=0).sum()
                constraint1 = -2 * DXu + Xu.sum(axis=1)
                constraint2 = -2 * DXu + Xu.sum(axis=1)
                constraint = torch.sum(constraint1) + torch.sum(constraint2)

                obj = l2_norm_error + beta * mixed_norm_error + rho * constraint
                val_loss[t] += obj.cpu().detach().item() / len(val_set)

                # DX(u - v) can serve as our prediction in the convex network.
                predictions.append((DXu - DXv).cpu().detach().numpy())

            predictions = np.array(predictions)

        return (train_loss, val_loss, predictions)
