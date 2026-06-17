# ### This notebook presents the architectures of the two recommendation systems tested within this framework
# # 1. Imports

import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
from pathlib import Path
import pickle
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import optuna
import logging

# # 2. MLP recommender Architecture

class MLP(nn.Module):
    def __init__(self, hidden_size, **kw):
        super(MLP, self).__init__()
        user_size = kw['num_items']
        item_size = kw['num_items']
        self.device = kw['device']
        self.users_fc = nn.Linear(user_size, hidden_size, bias = True).to(self.device)
        self.items_fc = nn.Linear(item_size, hidden_size, bias = True).to(self.device)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_tensor, item_tensor):
        user_vec = self.users_fc(user_tensor.to(self.device))
        item_vec = self.items_fc(item_tensor.to(self.device))
        output = torch.matmul(user_vec, item_vec.T).to(self.device)
        return self.sigmoid(output).to(self.device)

# # 3. VAE recommender Architecture

class VAE(nn.Module):
    def __init__(self, model_conf, **kw):
        super(VAE, self).__init__()
        self.device = kw['device'] 
        num_items = kw['num_items'] 
        self.num_items = num_items
        self.enc_dims = [self.num_items] + model_conf['enc_dims']
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]
        self.dropout = model_conf['dropout']
        self.softmax = nn.Softmax(dim=1)
        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.anneal_cap = model_conf['anneal_cap']

        self.eps = 1e-6
        self.anneal = 0.
        self.update_count = 0
        
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.ReLU())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.ReLU())
                
        self.to(self.device)

    def forward(self, rating_matrix):
        """
        Forward pass
        :param rating_matrix: rating matrix
        """
        # encoder
        if len(rating_matrix.shape) == 1:
            rating_matrix = torch.unsqueeze(rating_matrix, 0)
        h = F.dropout(F.normalize(rating_matrix, dim=-1), p=self.dropout, training=self.training)
        for layer in self.encoder:
            h = layer(h)

        # sample
        mu_q = h[:, :self.enc_dims[-1]]
        logvar_q = h[:, self.enc_dims[-1]:]  # log sigmod^2  batch x 200
        std_q = torch.exp(0.5 * logvar_q)  # sigmod batch x 200
        
        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q

        output = sampled_z
        for layer in self.decoder:
            output = layer(output)

        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return output, kl_loss
        else:
            return self.softmax(output)   
        
    def train_one_epoch(self, dataset, optimizer, batch_size, alpha=0.5):
        """
        Train model for one epoch
        :param dataset: given data
        :param optimizer: choice of optimizer
        :param batch_size: batch size
        :return: model loss
        """
        self.train()

        train_matrix = dataset

        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / batch_size))
        perm = np.random.permutation(num_training)

        loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx]).to(self.device)

            if self.total_anneal_steps > 0:
                self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                self.anneal = self.anneal_cap

            pred_matrix, kl_loss = self.forward(batch_matrix)

            # cross_entropy
            total_ce = -(F.log_softmax(pred_matrix, 1) * batch_matrix)
            ce_hist = total_ce[:,:self.num_items].sum(1).mean()
            ce_demo = total_ce[:,self.num_items:].sum(1).mean()
            ce_loss = ce_hist+alpha*ce_demo

            batch_loss = ce_loss + kl_loss * self.anneal

            batch_loss.backward()
            optimizer.step()

            self.update_count += 1

            loss += batch_loss
            if b % 200 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

    def predict(self, eval_users, test_batch_size):
        """
        Predict the model on test set
        :param eval_users: evaluation (test) user
        :param eval_pos: position of the evaluated (test) item
        :param test_batch_size: batch size for test set
        :return: predictions
        """
        with torch.no_grad():
            input_matrix = torch.Tensor(eval_users).to(self.device)
            preds = np.zeros_like(input_matrix.cpu())

            num_data = input_matrix.shape[0]
            num_batches = int(np.ceil(num_data / test_batch_size))
            perm = list(range(num_data))
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
                    
                test_batch_matrix = input_matrix[batch_idx]
                batch_pred_matrix = self.forward(test_batch_matrix)
                batch_pred_matrix.masked_fill(test_batch_matrix.bool(), float('-inf'))
                preds[batch_idx] = batch_pred_matrix.detach().cpu().numpy()
        return preds

# # 4. LGCN

import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
from pathlib import Path
import pickle
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import optuna
import logging
import scipy.sparse as sp

class LightGCN(nn.Module):
    def __init__(self, hidden_size, **kw):
        super(LightGCN, self).__init__()
        self.num_users = kw['num_users']
        self.num_items = kw['num_items']
        self.device = kw['device']
        self.embedding_dim = hidden_size
        self.n_layers = kw.get('n_layers', 3)
        self.alpha = kw.get('alpha', 0.1)  # Degree of propagation
        
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim).to(self.device)
        self.sigmoid = nn.Sigmoid()
        
        self.graph = self.build_graph(kw['user_item_matrix'])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def build_graph(self, user_item_matrix):
        user_item_matrix = sp.coo_matrix(user_item_matrix)
        rows, cols = user_item_matrix.row, user_item_matrix.col
        data = user_item_matrix.data
        
        adj = sp.coo_matrix((data, (rows, cols)), shape=(self.num_users + self.num_items, self.num_users + self.num_items))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        row_sum = np.array(adj.sum(1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv).tocoo()
        
        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse.FloatTensor(indices, values, shape).to(self.device)
    
    def forward(self, user_tensor, item_tensor):
        all_embeddings = self.get_embeddings()
        
        user_embeddings = all_embeddings[:self.num_users]
        item_embeddings = all_embeddings[self.num_users:]
        
        user_vec = user_embeddings[user_tensor.long()]
        item_vec = item_embeddings[item_tensor.long()]
        
        output = torch.matmul(user_vec, item_vec.mT).to(self.device)
        return self.sigmoid(output)
    
    def get_embeddings(self):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        
        embeddings_list = [all_embeddings]
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.graph, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        lightgcn_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_embeddings = torch.mean(lightgcn_embeddings, dim=1)
        
        return lightgcn_embeddings
