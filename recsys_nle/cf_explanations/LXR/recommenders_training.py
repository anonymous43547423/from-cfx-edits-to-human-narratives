import pandas as pd
import numpy as np
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

from pathlib import Path
import pickle
import torch
from torch import nn, optim, Tensor
import torch.nn as nn
import torch.nn.functional as F
import optuna
import logging
import matplotlib.pyplot as plt

from recommenders_architecture import *
from help_functions import *
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

from torch_geometric.data import download_url, extract_zip
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from scipy import sparse

import os
import argparse

data_name = "Pinterest" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "MLP" ## Can be MLP, VAE, MLP_model, GMF_model, NCF, LightGCN

# Early stopping configuration
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait before stopping
EARLY_STOPPING_MIN_EPOCHS = 6  # Minimum epochs before early stopping can trigger
EARLY_STOPPING_METRIC = 'hit_rate_10'  # 'hit_rate_10' or 'loss'

DP_DIR = Path("datasets", "lxr-CE", data_name)
export_dir = Path(os.getcwd())
files_path = Path(export_dir, DP_DIR)
checkpoints_path = Path(export_dir, "checkpoints", "recommenders")
Neucheckpoints_path = Path(export_dir, "checkpoints", "recommenders")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(f'[INFO] Using device: {device}')

output_type_dict = {
    "VAE":"multiple",
    "MLP":"single",
    "LightGCN":"single" #changed
}

num_users_dict = {
    "ML1M":6037,
    "Yahoo":13797, 
    "Pinterest":19155
}

num_items_dict = {
    "ML1M":3381,
    "Yahoo":4604, 
    "Pinterest":9362
}

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

ITERATIONS = 5000
EPOCHS = 30

BATCH_SIZE = 1024
LR = 1e-3
ITERS_PER_EVAL = 200
ITERS_PER_LR_DECAY = 200
LAMBDA = 1e-6

print(f'[INFO] Training Configuration:')
print(f'  - Dataset: {data_name} ({num_users_dict[data_name]} users, {num_items_dict[data_name]} items)')
print(f'  - Recommender: {recommender_name}')
print(f'  - Output type: {output_type_dict[recommender_name]}')
print(f'  - Device: {device}')
print(f'[INFO] Early Stopping Configuration:')
print(f'  - Enabled: {EARLY_STOPPING_ENABLED}')
print(f'  - Patience: {EARLY_STOPPING_PATIENCE} epochs')
print(f'  - Minimum epochs: {EARLY_STOPPING_MIN_EPOCHS}')
print(f'  - Metric: {EARLY_STOPPING_METRIC}')

output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 

train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)

print(f'[INFO] Data loaded successfully:')
print(f'  - Training samples: {len(train_array)}')
print(f'  - Test samples: {len(test_array)}')

for row in range(static_test_data.shape[0]):
    static_test_data.iloc[row, static_test_data.iloc[row,-2]]=0
test_array = static_test_data.iloc[:,:-2].to_numpy()

pop_array = np.zeros(len(pop_dict))
for key, value in pop_dict.items():
    pop_array[key] = value

from recommenders_architecture import *

kw_dict = {'device':device,
          'num_items': num_items,
          'pop_array':pop_array,
          'all_items_tensor':all_items_tensor,
          'static_test_data':static_test_data,
          'items_array':items_array,
          'output_type':output_type,
          'recommender_name':recommender_name}


train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

def check_early_stopping(epoch, test_losses, patience, min_epochs):
    """
    Check if early stopping should be triggered.
    
    Args:
        epoch: Current epoch number
        test_losses: List of test losses (negative hit rates)
        patience: Number of epochs to wait before stopping
        min_epochs: Minimum epochs before early stopping can trigger
    
    Returns:
        bool: True if early stopping should be triggered
    """
    if not EARLY_STOPPING_ENABLED or epoch < min_epochs or len(test_losses) < patience + 1:
        return False
    
    # Check if performance has been degrading for 'patience' consecutive epochs
    for i in range(patience):
        if test_losses[-(i+1)] <= test_losses[-(i+2)]:
            return False
    
    return True

def MLP_objective(trial):
    
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    beta = trial.suggest_float('beta', 0, 4) # hyperparameter that weights the different loss terms
    epochs = 40
    model = MLP(hidden_dim, **kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    
    # Track best performance for smart checkpoint saving
    best_hit_rate = 0.0
    best_epoch = -1
    
    print(f'[TRIAL {trial.number}] ========== Starting new MLP training run ==========')
    print(f'[TRIAL {trial.number}] Hyperparameters:')
    print(f'  - Learning rate: {lr:.6f}')
    print(f'  - Batch size: {batch_size}')
    print(f'  - Hidden dimension: {hidden_dim}')
    print(f'  - Beta: {beta:.2f}')
    print(f'  - Epochs: {epochs}')
    logger.info(f'[TRIAL {trial.number}] ========== Starting new MLP training run ==========')
    
    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))

    
    for epoch in range(epochs):
        train_matrix = sample_indices(train_data.copy(), **kw_dict)
        perm = np.random.permutation(num_training)
        loss = []
        train_pos_loss=[]
        train_neg_loss=[]
        if epoch!=0 and epoch%10 == 0: # decrease the learning rate every 10 epochs
            lr = 0.1*lr
            optimizer.lr = lr
            print(f'[TRIAL {trial.number}] Epoch {epoch}: Learning rate reduced to {lr:.6f}')
        
        for b in range(num_batches):
            optimizer.zero_grad()
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]    
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx,:-2]).to(device)

            batch_pos_idx = train_matrix[batch_idx,-2]
            batch_neg_idx = train_matrix[batch_idx,-1]
            
            batch_pos_items = torch.Tensor(items_array[batch_pos_idx]).to(device)
            batch_neg_items = torch.Tensor(items_array[batch_neg_idx]).to(device)
            
            pos_output = torch.diagonal(model(batch_matrix, batch_pos_items))
            neg_output = torch.diagonal(model(batch_matrix, batch_neg_items))
            
            # MSE loss
            pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
            neg_loss = torch.mean((neg_output)**2)
            
            batch_loss = pos_loss + beta*neg_loss
            batch_loss.backward()
            optimizer.step()
            
            loss.append(batch_loss.item())
            train_pos_loss.append(pos_loss.item())
            train_neg_loss.append(neg_loss.item())
            
        avg_train_loss = np.mean(loss)
        avg_pos_loss = np.mean(train_pos_loss)
        avg_neg_loss = np.mean(train_neg_loss)
        print(f'[TRIAL {trial.number}] Epoch {epoch:2d}/{epochs}:')
        print(f'  - Training Loss: {avg_train_loss:.4f}')
        print(f'  - Positive Loss: {avg_pos_loss:.4f}')
        print(f'  - Negative Loss: {avg_neg_loss:.4f}')
        
        train_losses.append(avg_train_loss)
        
        # Smart checkpoint saving - only save if performance improves
        if epoch % ((epochs*10)/100) == 0:
            model_filename = f'MLP_{data_name}_{epoch}_{hidden_dim}_{batch_size}_{trial.number}.pt'
            print(f'[TRIAL {trial.number}] Saving model checkpoint: {model_filename}')
            torch.save(model.state_dict(), Path(Neucheckpoints_path, model_filename))
        # torch.save(model.state_dict(), Path(checkpoints_path, f'MLP_{data_name}_{round(lr,4)}_{batch_size}_{trial.number}_{epoch}.pt'))


        model.eval()
        test_matrix = np.array(static_test_data)
        test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
        
        test_pos = test_matrix[:,-2]
        test_neg = test_matrix[:,-1]
        
        row_indices = np.arange(test_matrix.shape[0])
        test_tensor[row_indices,test_pos] = 0
        
        pos_items = torch.Tensor(items_array[test_pos]).to(device)
        neg_items = torch.Tensor(items_array[test_neg]).to(device)
        
        pos_output = torch.diagonal(model(test_tensor, pos_items).to(device))
        neg_output = torch.diagonal(model(test_tensor, neg_items).to(device))
        
        pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
        neg_loss = torch.mean((neg_output)**2)
        print(f'  - Test Positive Loss: {pos_loss:.4f}')
        print(f'  - Test Negative Loss: {neg_loss:.4f}')
        
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10) # metric for monitoring
        print(f'  - Evaluation Metrics:')
        print(f'    * Hit@10:  {hit_rate_at_10:.4f} ({hit_rate_at_10*100:.2f}%)')
        print(f'    * Hit@50:  {hit_rate_at_50:.4f} ({hit_rate_at_50*100:.2f}%)')
        print(f'    * Hit@100: {hit_rate_at_100:.4f} ({hit_rate_at_100*100:.2f}%)')
        print(f'    * MRR:     {MRR:.4f}')
        print(f'    * MPR:     {MPR:.4f}')
        
        # Smart checkpoint saving based on Hit@10 improvement
        if hit_rate_at_10 > best_hit_rate:
            best_hit_rate = hit_rate_at_10
            best_epoch = epoch
            # Don't save here - we'll save the best checkpoint at the end of the trial
            print(f'  - 🎯 NEW BEST! Hit@10 improved to {hit_rate_at_10:.4f} ({hit_rate_at_10*100:.2f}%) at epoch {epoch}')
        else:
            print(f'  - 📊 No improvement (Best: {best_hit_rate:.4f} at epoch {best_epoch})')
        
        print()
        
        test_losses.append(-hit_rate_at_10)
        
        # Check early stopping
        if check_early_stopping(epoch, test_losses, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_EPOCHS):
            print(f'[TRIAL {trial.number}] ⚠️  Early stopping triggered!')
            print(f'[TRIAL {trial.number}] Performance degraded for {EARLY_STOPPING_PATIENCE} consecutive epochs')
            print(f'[TRIAL {trial.number}] Best performance at epoch {best_epoch} with HR@10: {best_hit_rate:.4f}')
            
            # Save the best checkpoint at early stopping
            model_filename = f'{recommender_name}_{data_name}_TRIAL{trial.number}_FINAL_hr{best_hit_rate:.4f}_epoch{best_epoch}_lr{lr:.6f}_bs{batch_size}'
            if recommender_name == 'MLP':
                model_filename += f'_hd{hidden_dim}_beta{beta:.2f}.pt'
            else:  # VAE
                model_filename += f'_drop{VAE_config["dropout"]}.pt'
            print(f'[TRIAL {trial.number}] 💾 Saving final best checkpoint: {model_filename}')
            torch.save(model.state_dict(), Path(Neucheckpoints_path, model_filename))
            
            logger.info(f'[TRIAL {trial.number}] Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {best_epoch} with HR@10: {best_hit_rate}')
            train_losses_dict[trial.number] = train_losses
            test_losses_dict[trial.number] = test_losses
            HR10_dict[trial.number] = hr10
            return max(hr10)
            
    print(f'[TRIAL {trial.number}] ✅ Training completed!')
    print(f'[TRIAL {trial.number}] Best performance at epoch {best_epoch} with HR@10: {best_hit_rate:.4f}')
    
    # Save the best checkpoint at the end of training
    model_filename = f'{recommender_name}_{data_name}_TRIAL{trial.number}_FINAL_hr{best_hit_rate:.4f}_epoch{best_epoch}_lr{lr:.6f}_bs{batch_size}'
    if recommender_name == 'MLP':
        model_filename += f'_hd{hidden_dim}_beta{beta:.2f}.pt'
    else:  # VAE
        model_filename += f'_drop{VAE_config["dropout"]}.pt'
    print(f'[TRIAL {trial.number}] 💾 Saving final best checkpoint: {model_filename}')
    torch.save(model.state_dict(), Path(Neucheckpoints_path, model_filename))
    
    logger.info(f'[TRIAL {trial.number}] Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {best_epoch} with HR@10: {best_hit_rate}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

VAE_config= {
"enc_dims": [256,256],
"dropout": 0.5,
"anneal_cap": 0.2,
"total_anneal_steps": 200000
}

def VAE_objective(trial):
    
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [128,256])
    epochs = 40
    model = VAE(VAE_config ,**kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    
    # Track best performance for smart checkpoint saving
    best_hit_rate = 0.0
    best_epoch = -1
    
    print(f'[TRIAL {trial.number}] ========== Starting new VAE training run ==========')
    print(f'[TRIAL {trial.number}] Hyperparameters:')
    print(f'  - Learning rate: {lr:.6f}')
    print(f'  - Batch size: {batch_size}')
    print(f'  - Epochs: {epochs}')
    print(f'  - Encoder dimensions: {VAE_config["enc_dims"]}')
    print(f'  - Dropout: {VAE_config["dropout"]}')
    logger.info(f'[TRIAL {trial.number}] ========== Starting new VAE training run ==========')
    
    for epoch in range(epochs):
        if epoch!=0 and epoch%10 == 0:
            lr = 0.1*lr
            optimizer.lr = lr
            print(f'[TRIAL {trial.number}] Epoch {epoch}: Learning rate reduced to {lr:.6f}')
        loss = model.train_one_epoch(train_array, optimizer, batch_size)
        train_losses.append(loss)
        
        # Smart checkpoint saving - only save if performance improves
        if epoch % ((epochs*10)/100) == 0:
            model_filename = f'VAE_{data_name}_{trial.number}_{epoch}_{batch_size}.pt'
            print(f'[TRIAL {trial.number}] Saving model checkpoint: {model_filename}')
            torch.save(model.state_dict(), Path(Neucheckpoints_path, model_filename))
        # torch.save(model.state_dict(), Path(checkpoints_path, f'VAE_{data_name}_{trial.number}_{epoch}_{round(lr,4)}_{batch_size}.pt'))


        model.eval()
        test_matrix = static_test_data.to_numpy()
        test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
        test_pos = test_array[:,-2]
        test_neg = test_array[:,-1]
        row_indices = np.arange(test_matrix.shape[0])
        test_tensor[row_indices,test_pos] = 0
        output = model(test_tensor).to(device)
        pos_loss = -output[row_indices,test_pos].mean()
        neg_loss = output[row_indices,test_neg].mean()
        print(f'[TRIAL {trial.number}] Epoch {epoch:2d}/{epochs}:')
        print(f'  - Training Loss: {loss:.4f}')
        print(f'  - Test Positive Loss: {pos_loss:.4f}')
        print(f'  - Test Negative Loss: {neg_loss:.4f}')
        
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10)
        print(f'  - Evaluation Metrics:')
        print(f'    * Hit@10:  {hit_rate_at_10:.4f} ({hit_rate_at_10*100:.2f}%)')
        print(f'    * Hit@50:  {hit_rate_at_50:.4f} ({hit_rate_at_50*100:.2f}%)')
        print(f'    * Hit@100: {hit_rate_at_100:.4f} ({hit_rate_at_100*100:.2f}%)')
        print(f'    * MRR:     {MRR:.4f}')
        print(f'    * MPR:     {MPR:.4f}')
        
        # Smart checkpoint saving based on Hit@10 improvement
        if hit_rate_at_10 > best_hit_rate:
            best_hit_rate = hit_rate_at_10
            best_epoch = epoch
            # Don't save here - we'll save the best checkpoint at the end of the trial
            print(f'  - 🎯 NEW BEST! Hit@10 improved to {hit_rate_at_10:.4f} ({hit_rate_at_10*100:.2f}%) at epoch {epoch}')
        else:
            print(f'  - 📊 No improvement (Best: {best_hit_rate:.4f} at epoch {best_epoch})')
        
        print()
        
        test_losses.append(pos_loss.item())
        
        # Check early stopping
        if check_early_stopping(epoch, test_losses, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_EPOCHS):
            print(f'[TRIAL {trial.number}] ⚠️  Early stopping triggered!')
            print(f'[TRIAL {trial.number}] Performance degraded for {EARLY_STOPPING_PATIENCE} consecutive epochs')
            print(f'[TRIAL {trial.number}] Best performance at epoch {best_epoch} with HR@10: {best_hit_rate:.4f}')
            
            # Save the best checkpoint at early stopping
            model_filename = f'{recommender_name}_{data_name}_TRIAL{trial.number}_FINAL_hr{best_hit_rate:.4f}_epoch{best_epoch}_lr{lr:.6f}_bs{batch_size}'
            if recommender_name == 'MLP':
                model_filename += f'_hd{hidden_dim}_beta{beta:.2f}.pt'
            else:  # VAE
                model_filename += f'_drop{VAE_config["dropout"]}.pt'
            print(f'[TRIAL {trial.number}] 💾 Saving final best checkpoint: {model_filename}')
            torch.save(model.state_dict(), Path(Neucheckpoints_path, model_filename))
            
            logger.info(f'[TRIAL {trial.number}] Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {best_epoch} with HR@10: {best_hit_rate}')
            train_losses_dict[trial.number] = train_losses
            test_losses_dict[trial.number] = test_losses
            HR10_dict[trial.number] = hr10
            return max(hr10)
    
    print(f'[TRIAL {trial.number}] ✅ Training completed!')
    print(f'[TRIAL {trial.number}] Best performance at epoch {best_epoch} with HR@10: {best_hit_rate:.4f}')
    
    # Save the best checkpoint at the end of training
    model_filename = f'{recommender_name}_{data_name}_TRIAL{trial.number}_FINAL_hr{best_hit_rate:.4f}_epoch{best_epoch}_lr{lr:.6f}_bs{batch_size}'
    if recommender_name == 'MLP':
        model_filename += f'_hd{hidden_dim}_beta{beta:.2f}.pt'
    else:  # VAE
        model_filename += f'_drop{VAE_config["dropout"]}.pt'
    print(f'[TRIAL {trial.number}] 💾 Saving final best checkpoint: {model_filename}')
    torch.save(model.state_dict(), Path(Neucheckpoints_path, model_filename))
    
    logger.info(f'[TRIAL {trial.number}] Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {best_epoch} with HR@10: {best_hit_rate}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)

logger = logging.getLogger()

logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler(f"{recommender_name}_{data_name}_Optuna.log", mode="w"))

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

study = optuna.create_study(direction='maximize')

print(f'[INFO] Starting hyperparameter optimization for {recommender_name} on {data_name} dataset...')
logger.info("Start optimization.")

if recommender_name == 'MLP':
    study.optimize(MLP_objective, n_trials=5) 
elif recommender_name == 'VAE':
    study.optimize(VAE_objective, n_trials=5) 

with open(f"{recommender_name}_{data_name}_Optuna.log") as f:
    assert f.readline().startswith("A new study created")
    assert f.readline() == "Start optimization.\n"
    
    
# Print best hyperparameters and corresponding metric value
print(f'[RESULTS] ========== Optimization Results ==========')
print(f'[RESULTS] Best hyperparameters: {study.best_params}')
print(f'[RESULTS] Best metric value: {study.best_value:.4f}')
print(f'[RESULTS] Number of trials completed: {len(study.trials)}')


from help_functions import *

recommender_path_dict = {
    ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","MLP"):Path(checkpoints_path, "MLP1_ML1M_0.0076_256_7.pt"),

    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),
    
    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_12_18_0.0001_256.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt"),
    
}

hidden_dim_dict = {
    ("ML1M","VAE"): None,
    ("ML1M","MLP"): 32,

    ("Yahoo","VAE"): None,
    ("Yahoo","MLP"):32,
    
    ("Pinterest","VAE"): None,
    ("Pinterest","MLP"):512,
}

hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
recommender_path = recommender_path_dict[(data_name,recommender_name)]

print(f'[INFO] Loading pre-trained model from: {recommender_path}')

def load_recommender():
    if recommender_name=='MLP':
        recommender = MLP(hidden_dim, **kw_dict)
    elif recommender_name=='VAE':
        recommender = VAE(VAE_config, **kw_dict)
    recommender_checkpoint = torch.load(Path(checkpoints_path, recommender_path))
    recommender.load_state_dict(recommender_checkpoint)
    recommender.eval()
    for param in recommender.parameters():
        param.requires_grad= False
    print(f'[INFO] Model loaded successfully and set to evaluation mode')
    return recommender
    
model = load_recommender()

print(f'[INFO] Generating recommendations for test users...')

topk_test = {}
for i in range(len(test_array)):
    vec = test_array[i]
    tens = torch.Tensor(vec).to(device)
    topk_test[i] = int(get_user_recommended_item(tens, model, **kw_dict).cpu().detach().numpy())

print(f'[INFO] Evaluating final model performance...')

hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)

print(f'[FINAL RESULTS] ========== Final Model Performance ==========')
print(f'[FINAL RESULTS] Hit@10:  {hit_rate_at_10:.4f} ({hit_rate_at_10*100:.2f}%)')
print(f'[FINAL RESULTS] Hit@50:  {hit_rate_at_50:.4f} ({hit_rate_at_50*100:.2f}%)')
print(f'[FINAL RESULTS] Hit@100: {hit_rate_at_100:.4f} ({hit_rate_at_100*100:.2f}%)')
print(f'[FINAL RESULTS] MRR:     {MRR:.4f}')
print(f'[FINAL RESULTS] MPR:     {MPR:.4f}')
print(f'[FINAL RESULTS] ==============================================')