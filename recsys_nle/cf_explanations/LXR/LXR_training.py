# # Imports

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
import copy
import optuna
import logging
import matplotlib.pyplot as plt
import random
import ipynb
import wandb
import importlib


#Begin add -----------------------------------------------------------------------------------------
import os,argparse
parser = argparse.ArgumentParser(description="List files in a directory that start with a given keyword.")
# Add the arguments
parser.add_argument('--directory', type=str, default="VAE_ML1M_0.0007_128_10.pt", nargs='?')
parser.add_argument('--model', type=str, default="VAE", nargs='?')
parser.add_argument('--lambda_pos', type=float, default=35, nargs='?')
parser.add_argument('--lambda_neg', type=float, default=7, nargs='?')
parser.add_argument('--alpha', type=int, default=1, nargs='?')
parser.add_argument('--learning_rate', type=float, default=0.004, nargs='?')
parser.add_argument('--trial', type=int, default=0 , nargs='?')
parser.add_argument('--whereSaved', type=str, default='', nargs='?')
# Parse the arguments
args = parser.parse_args()
recommender_name = args.model
# End add -----------------------------------------------------------------------------------------


data_name = "ML1M" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "VAE" ## Can be MLP, VAE
print(f'------ Runnig {recommender_name} on {data_name} -----------')
DP_DIR = Path("datasets", data_name)
export_dir = Path(os.getcwd())
files_path = Path(export_dir, DP_DIR)
checkpoints_path = Path(export_dir, "checkpoints", "recommenders", recommender_name)
Neucheckpoints_path = Path(export_dir, "checkpoints", "recommenders", recommender_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_type_dict = {
    "VAE":"multiple",
    "MLP":"single"
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



recommender_path_dict = {
    ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","MLP"):Path(checkpoints_path, "MLP1_ML1M_0.0076_256_7.pt"),
    
    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),

    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_0.0002_32_12.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt")
    
}

hidden_dim_dict = {
    ("ML1M","VAE"): None,
    ("ML1M","MLP"): 32,

    ("Yahoo","VAE"): None,
    ("Yahoo","MLP"):32,
    
    ("Pinterest","VAE"): None,
    ("Pinterest","MLP"):512

}

output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 

hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
recommender_path = recommender_path_dict[(data_name,recommender_name)]

# ## Data imports and preprocessing

train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
train_data['user_id'] = train_data.index
test_data['user_id'] = test_data.index
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)

pop_array = np.zeros(len(pop_dict))
for key, value in pop_dict.items():
    pop_array[key] = value

kw_dict = {'device':device,
          'num_items': num_items,
          'pop_array':pop_array,
          'all_items_tensor':all_items_tensor,
          'static_test_data':static_test_data,
          'items_array':items_array,
          'output_type':output_type,
          'recommender_name':recommender_name}

# # Recommenders Architecture

from recsys_nle.cf_explanations.LXR import recommenders_architecture as _recommenders_architecture
importlib.reload(_recommenders_architecture)
from recsys_nle.cf_explanations.LXR.recommenders_architecture import *

VAE_config= {
"enc_dims": [512,128],
"dropout": 0.5,
"anneal_cap": 0.2,
"total_anneal_steps": 200000
}

def load_recommender():
    if recommender_name=='MLP':
        recommender = MLP(hidden_dim, **kw_dict)
    elif recommender_name=='VAE':
        recommender = VAE(VAE_config, **kw_dict)

    recommender_checkpoint = torch.load(Path(checkpoints_path, recommender_path), map_location=torch.device('cpu'))
    recommender.load_state_dict(recommender_checkpoint)
    recommender.eval()
    for param in recommender.parameters():
        param.requires_grad= False
    return recommender
    
recommender = load_recommender()

# # Help functions

from recsys_nle.cf_explanations.LXR import help_functions as _help_functions
importlib.reload(_help_functions)
from recsys_nle.cf_explanations.LXR.help_functions import *

# ## Load / create top recommended items dict

create_dicts = True
if create_dicts:
    top1_train = {}
    top1_test = {}
    for i in range(train_array.shape[0]):
        user_index = train_array[i][-1]
        user_tensor = torch.Tensor(train_array[i][:-1]).to(device)
        top1_train[user_index] = int(get_user_recommended_item(user_tensor, recommender, **kw_dict))
    for i in range(test_array.shape[0]):
        user_index = test_array[i][-1]
        user_tensor = torch.Tensor(test_array[i][:-1]).to(device)
        top1_test[user_index] = int(get_user_recommended_item(user_tensor, recommender, **kw_dict))
        
    with open(Path(files_path,f'top1_train_{data_name}_{recommender_name}.pkl'), 'wb') as f:
        pickle.dump(top1_train, f)
    with open(Path(files_path,f'top1_test_{data_name}_{recommender_name}.pkl'), 'wb') as f:
        pickle.dump(top1_test, f)
else:
    with open(Path(files_path,f'top1_train_{data_name}_{recommender_name}.pkl'), 'rb') as f:
        top1_train = pickle.load(f)
    with open(Path(files_path,f'top1_test_{data_name}_{recommender_name}.pkl'), 'rb') as f:
        top1_test = pickle.load(f)

# # LXR Architecture

class Explainer(nn.Module):
    def __init__(self, user_size, item_size, hidden_size):
        super(Explainer, self).__init__()
        
        self.users_fc = nn.Linear(in_features = user_size, out_features=hidden_size).to(device)
        self.items_fc = nn.Linear(in_features = item_size, out_features=hidden_size).to(device)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features = hidden_size*2, out_features=hidden_size).to(device),
            nn.Tanh(),
            nn.Linear(in_features = hidden_size, out_features=user_size).to(device),
            nn.Sigmoid()
        ).to(device)
        
        
    def forward(self, user_tensor, item_tensor):
        user_output = self.users_fc(user_tensor.float())
        item_output = self.items_fc(item_tensor.float())
        combined_output = torch.cat((user_output, item_output), dim=-1)
        expl_scores = self.bottleneck(combined_output).to(device)
        return expl_scores

# # Train functions

class LXR_loss(nn.Module):
    def __init__(self, lambda_pos, lambda_neg, alpha):
        super(LXR_loss, self).__init__()
        
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.alpha = alpha
        
        
    def forward(self, user_tensors, items_tensors, items_ids, pos_masks):
        neg_masks = torch.sub(torch.ones_like(pos_masks), pos_masks)
        x_masked_pos = user_tensors * pos_masks
        x_masked_neg = user_tensors * neg_masks
        if output_type=='single':
            x_masked_res_pos = torch.diag(recommender_run(x_masked_pos, recommender, items_tensors, item_id=items_ids, wanted_output = 'single', **kw_dict))
            x_masked_res_neg = torch.diag(recommender_run(x_masked_neg, recommender, items_tensors, item_id=items_ids, wanted_output = 'single', **kw_dict))
        else:
            x_masked_res_pos_before = recommender_run(x_masked_pos, recommender, items_tensors, item_id=items_ids, wanted_output = 'vector', **kw_dict)
            x_masked_res_neg_before = recommender_run(x_masked_neg, recommender, items_tensors, item_id=items_ids, wanted_output = 'vector', **kw_dict)
            rows=torch.arange(len(items_ids))
            x_masked_res_pos = x_masked_res_pos_before[rows, items_ids] 
            x_masked_res_neg = x_masked_res_neg_before[rows, items_ids] 
            
            
        pos_loss = -torch.mean(torch.log(x_masked_res_pos))
        neg_loss = torch.mean(torch.log(x_masked_res_neg))
        l1 = x_masked_pos[user_tensors>0].mean()
        combined_loss = self.lambda_pos*pos_loss + self.lambda_neg*neg_loss + self.alpha*l1
        
        return combined_loss, pos_loss, neg_loss, l1

#LXR based similarity
def find_LXR_mask(user_tensor, item_id, item_tensor, explainer):
    expl_scores = explainer(user_tensor, item_tensor)
    x_masked = user_tensor*expl_scores
    item_sim_dict = {i: x_masked[i].item() for i in range(len(x_masked))}    

    return item_sim_dict

# evaluate LXR pos@20 and neg@20 scores on test set
def calculate_pos_neg_k(user_tensor, item_id, items_tensor, num_of_bins, explainer, k=20):
    
    POS_masked = user_tensor
    NEG_masked = user_tensor
    user_hist_size = int(torch.sum(user_tensor))

    bins = [0] + [len(x) for x in np.array_split(np.arange(user_hist_size), num_of_bins, axis=0)]

    POS_at_20 = [0] * (num_of_bins+1)
    NEG_at_20 = [0] * (num_of_bins+1)
    total_items = 0
    
    #returns original tensor
    sim_items = find_LXR_mask(user_tensor, item_id, items_tensor, explainer)
    POS_sim_items=list(sorted(sim_items.items(), key=lambda item: item[1],reverse=True))[0:user_hist_size]
    NEG_sim_items  = list(sorted(dict(POS_sim_items).items(), key=lambda item: item[1],reverse=False))
    
    for i in range(len(bins)):
        total_items += bins[i]
        
        POS_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        for j in POS_sim_items[:total_items]:
            POS_masked[j[0]] = 1
        POS_masked = user_tensor - POS_masked # remove the masked items from the user history 
        
        NEG_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        for j in NEG_sim_items[:total_items]:
            NEG_masked[j[0]] = 1
        NEG_masked = user_tensor - NEG_masked # remove the masked items from the user history 
        
        POS_index = get_index_in_the_list(POS_masked, user_tensor, item_id, recommender, **kw_dict)+1
        NEG_index = get_index_in_the_list(NEG_masked, user_tensor, item_id, recommender, **kw_dict)+1        
        
        POS_at_20[i] = 1 if POS_index <= 20 else 0
        NEG_at_20[i] = 1 if NEG_index <=20 else 0

    res = [np.array(POS_at_20), np.array(NEG_at_20)]
    return res

# # LXR training
# ### Utilizing Optuna for hyperparameter optimization and WandB (Weights and Biases) for experiment tracking and logging.

torch.manual_seed(42)
np.random.seed(42)

num_of_rand_users = 200 # number of users for evaluations
random_rows = np.random.choice(test_array.shape[0], num_of_rand_users, replace=False)
random_sampled_array = test_array[random_rows]

def lxr_training(trial):
    
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01)
    alpha = trial.suggest_categorical('alpha', [1]) # set alpha to be 1, change other hyperparameters
    lambda_neg = trial.suggest_float('lambda_neg', 0,50)
    lambda_pos = trial.suggest_float('lambda_pos', 0,50)
    if bool(args.trial):
        print('Optuna trial')
        learning_rate = trial.suggest_float('learning_rate', 0.003, 0.006)
        alpha = trial.suggest_categorical('alpha', [1]) # set alpha to be 1, change other hyperparameters
        lambda_neg = trial.suggest_float('lambda_neg', 7,10)
        lambda_pos = trial.suggest_float('lambda_pos', 32,39)
    else:
        print('Manual trial')
        learning_rate = args.learning_rate
        alpha = args.alpha
        lambda_neg = args.lambda_neg
        lambda_pos = args.lambda_pos
    # learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01)
    # alpha = trial.suggest_categorical('alpha', [1]) # set alpha to be 1, change other hyperparameters
    # lambda_neg = trial.suggest_float('lambda_neg', 0,50)
    # lambda_pos = trial.suggest_float('lambda_pos', 0,50)
    batch_size = trial.suggest_categorical('batch_size', [32,64,128,256])
    explainer_hidden_size = trial.suggest_categorical('explainer_hidden_size', [32,64,128])
    epochs = 10

    print(f'model is {args.directory}')


    # batch_size = trial.suggest_categorical('batch_size', [128])
    # explainer_hidden_size = trial.suggest_categorical('explainer_hidden_size', [128])

    # batch_size = trial.suggest_categorical('batch_size', [32,64,128,256])
    # explainer_hidden_size = trial.suggest_categorical('explainer_hidden_size', [32,64,128])
    # epochs = 40 
    epochs = 25
    
    wandb.init(
        project=f"{data_name}_{recommender_name}_LXR_training",
        name=f"trial_{trial.number}",
        config={
        'learning_rate' : learning_rate,
        'alpha' : alpha,
        'lambda_neg' : lambda_neg,
        'lambda_pos' : lambda_pos,
        'batch_size' : batch_size,
        'explainer_hidden_size' : explainer_hidden_size,
        'architecture' : 'LXR_combined',
        'activation_function' : 'Tanh',
        'loss_type' : 'logloss',
        'optimize_for' : 'pos_at_20',
        'epochs':epochs
        })

    loader = torch.utils.data.DataLoader(train_array, batch_size=batch_size, shuffle=True)
    num_batches = int(np.ceil(train_array.shape[0] / batch_size))


    num_of_bins = 10
    run_pos_at_20 = []
    run_neg_at_20 = []
    metric_for_monitoring = []
    train_losses = []

    recommender.eval()
    # explainer = Explainer(num_features, num_items, explainer_hidden_size).to(device) #change
    explainer = Explainer(train_data.shape[1]-1, num_items, explainer_hidden_size).to(device)
    optimizer_comb = torch.optim.Adam(explainer.parameters(), learning_rate)
    loss_func = LXR_loss(lambda_pos, lambda_neg, alpha)

    print('======================== new run ========================')
    print(f'Start with lambda_pos = {lambda_pos}, lambda_neg = {lambda_neg}, alpha_parameter = {alpha}')    
    print(f'Learning rate = {learning_rate}, batch size = {batch_size}, explainer hidden size = {explainer_hidden_size}')
    print('=========================================================')

    for epoch in range(epochs):
        if epoch%15 == 0 and epoch>0: # decrease learning rate every 15 epochs
            learning_rate*= 0.1
            optimizer_comb.lr = learning_rate

        POS_at_20_lxr = np.zeros(num_of_bins+1)
        NEG_at_20_lxr = np.zeros(num_of_bins+1)
        train_loss = 0
        total_pos_loss=0
        total_neg_loss=0
        total_l1_loss=0

        explainer.train()
        for batch_index, samples in enumerate(loader):
            # prepare data for explainer:
            user_tensors = torch.Tensor(samples[:,:-1]).to(device)
            user_ids = samples[:,-1]
            top1_item = np.array([top1_train[int(x)] for x in user_ids])
            items_vectors = items_array[top1_item]
            items_tensors = torch.Tensor(items_vectors).to(device)
            n = user_tensors.shape[0]

            # zero grad:
            optimizer_comb.zero_grad()
            # forward:
            expl_scores = explainer(user_tensors, items_tensors)

            # caclulate loss
            comb_loss, pos_loss, neg_loss, l1 = loss_func(user_tensors, items_tensors, top1_item, expl_scores)
            train_loss += comb_loss*n
            total_pos_loss += pos_loss*n
            total_neg_loss += neg_loss*n
            total_l1_loss += l1*n

            # back propagation
            comb_loss.backward()
            optimizer_comb.step()

        train_metrics = {"train/train_loss": train_loss,
                         "train/pos_loss": total_pos_loss,
                         "train/neg_loss": total_neg_loss,
                         "train/l1_loss": total_l1_loss,
                         "train/epoch": epoch}

        torch.save(explainer.state_dict(), Path(checkpoints_path, f'LXR_{data_name}_{recommender_name}_{trial.number}_{epoch}_{explainer_hidden_size}_{lambda_pos}_{lambda_neg}.pt'))

        #Monitoring on POS metric after each epoch
        explainer.eval()
        for j in range(random_sampled_array.shape[0]):
            user_id = random_sampled_array[j][-1]
            user_tensor = torch.Tensor(random_sampled_array[j][:-1]).to(device)
            top1_item_test = top1_test[user_id]
            item_vector = items_array[top1_item_test]
            items_tensor = torch.Tensor(item_vector).to(device)

            res = calculate_pos_neg_k(user_tensor, top1_item_test, items_tensor, num_of_bins, explainer, k=20)
            POS_at_20_lxr += res[0]
            NEG_at_20_lxr += res[1]

        last_pos_at_20 = np.mean(POS_at_20_lxr)/random_sampled_array.shape[0]
        last_neg_at_20 = np.mean(NEG_at_20_lxr)/random_sampled_array.shape[0]
        run_pos_at_20.append(last_pos_at_20)
        run_neg_at_20.append(last_neg_at_20)
        metric_for_monitoring.append(last_pos_at_20.item())

        val_metrics = {
            "val/pos_at_20": last_pos_at_20,
            "val/neg_at_20": last_neg_at_20
        }
        
        wandb.log({**train_metrics, **val_metrics})
        print(f'Finished epoch {epoch} with run_pos_at_20 {last_pos_at_20} and run_neg_at_20 {last_neg_at_20}')
        print(f'Train loss = {train_loss}')

        torch.save(explainer.state_dict(), Path(Neucheckpoints_path, f'{args.whereSaved}/LXR_{data_name}_{recommender_name}_{trial.number}_{epoch}_{explainer_hidden_size}_{lambda_pos}_{lambda_neg}.pt'))
        print(f'LXR_{data_name}_{recommender_name}_{trial.number}_{epoch}_{explainer_hidden_size}_{lambda_pos}_{lambda_neg}.pt')
        if epoch>=5: # early stop conditions - if both pos@20 and neg@20 are getting worse in the past 4 epochs
            if run_pos_at_20[-2]<run_pos_at_20[-1] and run_pos_at_20[-3]<run_pos_at_20[-2] and run_pos_at_20[-4]<run_pos_at_20[-3]:
                if run_neg_at_20[-2]>run_neg_at_20[-1] and run_neg_at_20[-3]>run_neg_at_20[-2] and run_neg_at_20[-4]>run_neg_at_20[-3]:
                    print(f'Early stop at trial with lambda_pos = {lambda_pos}, lambda_neg = {lambda_neg}, alpha_parameter = {alpha}. Best results at epoch {np.argmin(run_pos_at_20)} with value {np.min(run_pos_at_20)}')
                    return np.min(metric_for_monitoring) # return the best pos@20 value in this trial

    print(f'Stop at trial with lambda_pos = {lambda_pos}, lambda_neg = {lambda_neg}, alpha_parameter = {alpha}. Best results at epoch {np.argmin(run_pos_at_20)} with value {np.min(run_pos_at_20)}')    
    return np.min(metric_for_monitoring) # return the best pos@20 value in this trial

# ### Save logs in text file, optimize using Optuna

# Testing Wandb

# import wandb
# import random

# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.02,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
#     "epochs": 10,
#     }
# )

# # simulate training
# epochs = 10
# offset = random.random() / 5
# for epoch in range(2, epochs):
#     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
#     loss = 2 ** -epoch + random.random() / epoch + offset

#     # log metrics to wandb
#     wandb.log({"acc": acc, "loss": loss})

# # [optional] finish the wandb run, necessary in notebooks
# wandb.finish()

# logger = logging.getLogger()

# logger.setLevel(logging.INFO)  # Setup the root logger.
# logger.addHandler(logging.FileHandler(f"{data_name}_{recommender_name}_explainer_training.log", mode="w"))

# optuna.logging.enable_propagation()  # Propagate logs to the root logger.
# optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

# study = optuna.create_study(direction='minimize')

# logger.info("Start optimization.")
# study.optimize(lxr_training, n_trials=5)

# with open(f"{data_name}_{recommender_name}_explainer_training.log") as f:
#     assert f.readline().startswith("A new study created")
#     assert f.readline() == "Start optimization.\n"
    
    
# # Print best hyperparameters and corresponding metric value
# print("Best hyperparameters: {}".format(study.best_params))
# print("Best metric value: {}".format(study.best_value))

# wandb.finish()
