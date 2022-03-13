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

        self.U = nn.Parameter(data=torch.zeros(n_features, n_regions), requires_grad=True)
        self.V = nn.Parameter(data=torch.zeros(n_features, n_regions), requires_grad=True)
        self.S = nn.Parameter(data=torch.randn(n_features, n_regions), requires_grad=False)

        return

    def forward(self, x):
        self.W2 = torch.linalg.norm(self.U, axis=0)
        self.W1 = self.U / self.W2
        x = F.relu(x @ self.W1) @ self.W2
        return x

    def train(self, train_set, val_set, n_epochs, lr=1e-20, lamb=1e-2, rho=1e-2):

        l2_loss = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        n_batch = len(train_set)
        relu = nn.ReLU()

        train_loss = np.zeros([n_epochs, n_batch])
        val_loss = np.zeros([n_epochs, n_batch])
        progress_bar = tqdm(range(n_epochs))
        for t in progress_bar:
            for i, batch in enumerate(train_set):

                # Process data
                batch = batch.to(self.U.device).float()
                data, labels = batch[:, :-1], batch[:, -1]

                # Compute terms of convex problem.
                Dmat = torch.sign(F.relu(data @ self.S))
                Xu = data @ self.U
                Xv = data @ self.V
                DXu = Dmat * Xu
                DXv = Dmat * Xv
                l2_norm_error = l2_loss(DXu.sum(axis=1) - DXv.sum(axis=1), labels)
                mixed_norm_error = torch.linalg.norm(self.U, ord=2, dim=1).sum() + torch.linalg.norm(self.V, ord=2, dim=1).sum()
                constraint1 = F.relu(-2 * DXu + Xu)
                constraint2 = F.relu(-2 * DXu + Xu)
                constraint = torch.sum(constraint1) + torch.sum(constraint2)

                # Objective
                obj = l2_norm_error + lamb * mixed_norm_error + rho * constraint
                train_loss[t, i] = obj.cpu().detach().item()

                # Use SGD to optimize
                optimizer.zero_grad()
                obj.backward()
                optimizer.step()

                # Test
                for j, batch in enumerate(val_set):
                    batch = batch.to(self.U.device).float()
                    data, labels = batch[:, :-1], batch[:, -1]
                    Dmat = torch.sign(F.relu(data @ self.S))
                    Xu = data @ self.U
                    Xv = data @ self.V
                    DXu = Dmat * Xu
                    DXv = Dmat * Xv
                    l2_norm_error = l2_loss(DXu.sum(axis=1) - DXv.sum(axis=1), labels)
                    mixed_norm_error = torch.norm(self.U, dim=1).sum() + torch.norm(self.V, dim=1).sum()
                    constraint1 = F.relu(-2 * DXu + Xu)
                    constraint2 = F.relu(-2 * DXu + Xu)
                    constraint = torch.sum(constraint1) + torch.sum(constraint2)
                    obj = l2_norm_error + lamb * mixed_norm_error + rho * constraint
                    val_loss[t, i] += obj.cpu().detach().item()

                progress_bar.set_description(f'Loss = {train_loss[t, i]:.3f}')

        return train_loss, val_loss
