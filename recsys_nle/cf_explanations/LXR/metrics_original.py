# ### This notebook produces the metrics for a specific recommendation system and dataset.
# # Imports
# 

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
import torch.nn.functional as F
import optuna
import logging
import matplotlib.pyplot as plt
import ipynb
import importlib
import sys

import os,argparse
parser = argparse.ArgumentParser(description="List files in a directory that start with a given keyword.")

parser.add_argument('--directoryRec', type=str, default="VAE_ML1M_0.0007_128_10.pt", nargs='?')
parser.add_argument('--directoryLXR', type=str, default="LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426LXRMAIN.pt", nargs='?')
parser.add_argument('--SizeLXR', type=int, default=128, nargs='?')
parser.add_argument('--evaluation_method', type=str, default='binned',
                   choices=['binned', 'onebyone', 'onebyoneuntil20p'],
                   help='Evaluation method: binned (10%% increments), onebyone (item-by-item removal), or onebyoneuntil20p (item-by-item removal up to 20%% of user history) (default: binned)')

# Parse the arguments
args = parser.parse_args()

data_name = "ML1M" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "VAE" ### Can be MLP, VAE, LightGCN
DP_DIR = Path("datasets", data_name)
export_dir = Path(os.getcwd())
files_path = Path(export_dir, DP_DIR)
checkpoints_path = Path(export_dir, "checkpoints", "recommenders","VAE")
Neucheckpoints_path = Path(export_dir, "checkpoints", "recommenders", "VAE")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

output_type_dict = {
    "VAE":"multiple",
    "MLP":"single",
    "LightGCN":"single"
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
    # ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","VAE"): Path(Neucheckpoints_path, args.directoryRec),
    ("ML1M","MLP"):Path(checkpoints_path, "MLP_ML1M_2_512.pt"),
    ("ML1M", "LightGCN"): Path(checkpoints_path, "LightGCN_ML1M_iter10k.pt"),
    
    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "5MLP_Yahoo_5_256.pt"),
    
    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_0_11_128.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt")
}

hidden_dim_dict = {
    ("ML1M","VAE"): [512,128], #change from [512,256]
    ("ML1M","MLP"): 512, #change
    ("ML1M", "LightGCN"): 64,
    
    ("Yahoo","VAE"): [512,128],
    ("Yahoo","MLP"):256,

    ("Pinterest","VAE"): [256,256],  #change
    ("Pinterest","MLP"):512,

}

LXR_checkpoint_dict = {
    # ("ML1M","VAE"): ('LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426.pt',128),
    ("ML1M","VAE"): (args.directoryLXR, args.SizeLXR),
    ("ML1M","MLP"): ('sel_2_LXR_ML1M_MLP_0_3_32_46.54931757000438_19.14191145091549.pt',32),
    ("ML1M", "LightGCN"): ('LXR_ML1M_LightGCN_0_1_64_6.820005312417243_26.942911135968924.pt',64),

    ("Yahoo","VAE"): ('LXR_Yahoo_VAE_neg-1.5pos_combined_19_26_128_18.958765029913238_4.92235962483309.pt',128),
    ("Yahoo","MLP"):('sel_5YahooLXR_Yahoo_MLP_0_39_32_11.107168366441384_7.177773458235631.pt',32),

    ("Pinterest","VAE"): ('LXR_Pinterest_VAE_comb_4_27_32_6.3443735346179855_1.472868807603448.pt',32),
    ("Pinterest","MLP"):('LXR_Pinterest_MLP_0_5_16_10.059416809308486_0.705778173474644.pt',16),
}

output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 
hidden_dim = hidden_dim_dict[(data_name,recommender_name)]

recommender_path = recommender_path_dict[(data_name,recommender_name)]
lxr_path = LXR_checkpoint_dict[(data_name,recommender_name)][0]
lxr_dim = LXR_checkpoint_dict[(data_name,recommender_name)][1]

# ## Data and baselines imports

train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)

test_array = static_test_data.iloc[:,:-2].to_numpy()
create_dictionaries = False

with open(Path(files_path, f'jaccard_based_sim_{data_name}.pkl'), 'rb') as f:
    jaccard_dict = pickle.load(f) 

with open(Path(files_path, f'cosine_based_sim_{data_name}.pkl'), 'rb') as f:
    cosine_dict = pickle.load(f) 

with open(Path(files_path, f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f) 

with open(Path(files_path, f'tf_idf_dict_{data_name}.pkl'), 'rb') as f:
    tf_idf_dict = pickle.load(f) 

with open(Path(files_path, f'item_to_cluster_{recommender_name}_{data_name}.pkl'), 'rb') as f:
    item_to_cluster = pickle.load(f) 

with open(Path(files_path, f'shap_values_{recommender_name}_{data_name}.pkl'), 'rb') as f:
    shap_values= pickle.load(f) 

for i in range(num_items):
    for j in range(i, num_items):
        jaccard_dict[(j,i)]= jaccard_dict[(i,j)]
        cosine_dict[(j,i)]= cosine_dict[(i,j)]

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

from ipynb.fs.defs.recommenders_architecture import *
importlib.reload(ipynb.fs.defs.recommenders_architecture)
from ipynb.fs.defs.recommenders_architecture import *

VAE_config= {
"enc_dims": hidden_dim,
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

# # LXR definition and loading

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

def load_explainer():
    explainer = Explainer(num_items, num_items, lxr_dim)
    lxr_checkpoint = torch.load(Path(Neucheckpoints_path, lxr_path), map_location=torch.device('cpu'))
    explainer.load_state_dict(lxr_checkpoint)
    explainer.eval()
    for param in explainer.parameters():
        param.requires_grad= False
    return explainer

    
explainer = load_explainer()

# # Help functions

from ipynb.fs.defs.help_functions import *
importlib.reload(ipynb.fs.defs.help_functions)
from ipynb.fs.defs.help_functions import *

# # Baselines functions
# ### Every function produces explanations for a designated baseline, resulting in a dictionary that maps items from the user's history to their explanation scores based on that baseline.

sys.path.append('../baselines') 
from ipynb.fs.defs.lime import *
from ipynb.fs.defs.lime import *
importlib.reload(ipynb.fs.defs.lime)
from ipynb.fs.defs.lime import *

lime = LimeBase(distance_to_proximity)

#User based similarities using Jaccard
def find_jaccard_mask(x, item_id, user_based_Jaccard_sim):
    user_hist = x # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    item_jaccard_dict = {}
    for i,j in enumerate(user_hist>0):
        if j:
            if (i,item_id) in user_based_Jaccard_sim:
                item_jaccard_dict[i]=user_based_Jaccard_sim[(i,item_id)] # add Jaccard similarity between items
            else:
                item_jaccard_dict[i] = 0            

    return item_jaccard_dict

#Cosine based similarities between users and items
def find_cosine_mask(x, item_id, item_cosine):
    user_hist = x # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    item_cosine_dict = {}
    for i,j in enumerate(user_hist>0):
        if j:
            if (i,item_id) in item_cosine:
                item_cosine_dict[i]=item_cosine[(i,item_id)]
            else:
                item_cosine_dict[i]=0

    return item_cosine_dict

def find_lime_mask(x, item_id, min_pert, max_pert, num_of_perturbations, kernel_func, feature_selection, recommender, num_samples=10, method = 'POS', **kw_dict):
    user_hist = x # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    lime.kernel_fn = kernel_func
    neighborhood_data, neighborhood_labels, distances, item_id = get_lime_args(user_hist, item_id, recommender, all_items_tensor, min_pert = min_pert, max_pert = max_pert, num_of_perturbations = num_of_perturbations, seed = item_id, **kw_dict)
    if method=='POS':
        most_pop_items  = lime.explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, item_id, num_samples, feature_selection, pos_neg='POS')
    if method=='NEG':
        most_pop_items  = lime.explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, item_id, num_samples, feature_selection ,pos_neg='NEG')
        
    return most_pop_items 

def find_fia_mask(user_tensor, item_tensor, item_id, recommender):
    y_pred = recommender_run(user_tensor, recommender, item_tensor, item_id, **kw_dict).to(device)
    items_fia = {}
    user_hist = user_tensor.cpu().detach().numpy().astype(int)
    
    for i in range(num_items):
        if(user_hist[i] == 1):
            user_hist[i] = 0
            user_tensor = torch.FloatTensor(user_hist).to(device)
            y_pred_without_item = recommender_run(user_tensor, recommender, item_tensor, item_id, 'single', **kw_dict).to(device)
            infl_score = y_pred - y_pred_without_item
            items_fia[i] = infl_score
            user_hist[i] = 1

    return items_fia

def find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model, top_k):
    """
    Generate counterfactual explanation using ACCENT method.
    
    ACCENT analyzes differential influence of historical items across top-k recommendations.
    """
    items_accent = defaultdict(float)
    factor = top_k - 1

    # Get top-k recommended items
    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender_model, **kw_dict).keys())
    
    if top_k == 1:
        top_k_indices = [sorted_indices[0]]
    else:
        top_k_indices = sorted_indices[:top_k]
   
    for iteration, item_k_id in enumerate(top_k_indices):
        # Create item tensor for this top-k recommended item
        item_vector = items_array[item_k_id]
        item_tensor_k = torch.FloatTensor(item_vector).to(device)
              
        # Compute influence of historical items on this recommendation
        # Use original user_tensor (unchanged history)
        fia_dict = find_fia_mask(user_tensor, item_tensor_k, item_k_id, recommender_model)
         
        # Aggregate influences with weighted difference
        if not iteration:
            for key in fia_dict.keys():
                items_accent[key] = fia_dict[key] * factor
        else:
            for key in fia_dict.keys():
                items_accent[key] -= fia_dict[key]
       
    # Negate all scores
    for key in items_accent.keys():
        items_accent[key] *= -1    

    return items_accent


def find_accent_mask_refactored(user_tensor, user_id, item_tensor, item_id, recommender_model, top_k):
    """
    Generate counterfactual explanation using ACCENT method (corrected implementation).
    
    This implements ACCENT's core idea: compute gap influences between the top-1 
    recommendation and other top-k candidates. For each interaction z in user history:
        gap_influence(z) = sum over candidates i: [I(z, rec) - I(z, i)]
    
    where I(z, y) is the influence of interaction z on prediction y (computed via FIA).
    """
    items_accent = defaultdict(float)

    # Get top-k recommended items
    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender_model, **kw_dict).keys())
    
    if top_k == 1:
        # With k=1, there's no gap to compute
        top_k_indices = [sorted_indices[0]]
        item_vector = items_array[top_k_indices[0]]
        item_tensor_single = torch.FloatTensor(item_vector).to(device)
        return find_fia_mask(user_tensor, item_tensor_single, top_k_indices[0], recommender_model)
    
    top_k_indices = sorted_indices[:top_k]
    
    # rec is the top-1 recommendation we want to explain
    rec = top_k_indices[0]
    
    # Compute influence of each interaction on rec (once)
    rec_item_vector = items_array[rec]
    rec_item_tensor = torch.FloatTensor(rec_item_vector).to(device)
    fia_rec = find_fia_mask(user_tensor, rec_item_tensor, rec, recommender_model)
    
    # For each candidate in top-k (excluding rec), compute gap influence
    for cand in top_k_indices[1:]:
        # Compute influence of each interaction on this candidate
        cand_item_vector = items_array[cand]
        cand_item_tensor = torch.FloatTensor(cand_item_vector).to(device)
        fia_cand = find_fia_mask(user_tensor, cand_item_tensor, cand, recommender_model)
        
        # Accumulate gap influence: I(z, rec) - I(z, cand)
        for z in fia_rec.keys():
            gap_influence = fia_rec[z] - fia_cand.get(z, 0.0)
            items_accent[z] += gap_influence
    
    return items_accent


def find_shapley_mask(user_tensor, user_id, model, shap_values, item_to_cluster):
       
    item_shap = {}
    shapley_values = shap_values[shap_values[:, 0].astype(int) == user_id][:,1:]
    user_vector = user_tensor.cpu().detach().numpy().astype(int)
       
    for i in np.where(user_vector.astype(int) == 1)[0]:
        items_cluster = item_to_cluster[i]
        item_shap[i] = shapley_values.T[int(items_cluster)][0]
 
    return item_shap 

def find_lxr_mask(x, item_tensor):
    
    user_hist = x
    expl_scores = explainer(user_hist, item_tensor)
    x_masked = user_hist*expl_scores
    item_sim_dict = {}
    for i,j in enumerate(x_masked>0):
        if j:
            item_sim_dict[i]=x_masked[i] 
        
    return item_sim_dict

# # Evaluation help functions

def single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender_model, user_id = None, mask_type = None):
    '''
    This function invokes various explanation functions
    and returns a dictionary of explanations, sorted by their scores.
    '''
    user_hist_size = np.sum(user_vector)

    if mask_type == 'lime':
        POS_sim_items = find_lime_mask(user_vector, item_id, 50, 100, 150, distance_to_proximity,'highest_weights', recommender_model, num_samples=user_hist_size, **kw_dict)
        NEG_sim_items = find_lime_mask(user_vector, item_id, 50, 100, 150, distance_to_proximity,'highest_weights', recommender_model, num_samples=user_hist_size, method = 'NEG', **kw_dict)
    else:
        if mask_type == 'jaccard':
            sim_items = find_jaccard_mask(user_tensor, item_id, jaccard_dict)
        elif mask_type == 'cosine':
            sim_items = find_cosine_mask(user_tensor, item_id, cosine_dict)
        elif mask_type == 'shap':
            sim_items = find_shapley_mask(user_tensor, user_id, recommender_model, shap_values, item_to_cluster)
        elif mask_type == 'accent':
            sim_items = find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model, 5)
        elif mask_type == 'lxr':
            sim_items = find_lxr_mask(user_tensor, item_tensor)
        
        POS_sim_items  = list(sorted(sim_items.items(), key=lambda item: item[1],reverse=True))[0:user_hist_size]
        
    return POS_sim_items

def single_user_metrics(user_vector, user_tensor, item_id, item_tensor, num_of_bins, recommender_model, expl_dict, **kw_dict):
    
    '''
    This function takes the explanation dictionary as input. 
    It iteratively removes the top 10% of items
    that provide the most explanation from the user's history
    and calculates the metric values for the resulting counterfactual user vector.
    Ultimately, it returns all the metrics calculated at each step of masking.
    '''
    
    POS_masked = user_tensor
    NEG_masked = user_tensor
    POS_masked[item_id]=0
    NEG_masked[item_id]=0
    user_hist_size = np.sum(user_vector)
    
    
    bins=[0]+[len(x) for x in np.array_split(np.arange(user_hist_size), num_of_bins, axis=0)]
    
    POS_at_1 = [0]*(len(bins))
    POS_at_5 = [0]*(len(bins))
    POS_at_3 = [0]*(len(bins))
    POS_at_10=[0]*(len(bins))
    POS_at_20=[0]*(len(bins))
    POS_at_50=[0]*(len(bins))
    POS_at_100=[0]*(len(bins))
    
    NEG_at_1 = [0]*(len(bins))
    NEG_at_5 = [0]*(len(bins))
    NEG_at_3 = [0]*(len(bins))
    NEG_at_10 = [0]*(len(bins))
    NEG_at_20 = [0]*(len(bins))
    NEG_at_50 = [0]*(len(bins))
    NEG_at_100 = [0]*(len(bins))
    
    DEL = [0]*(len(bins))
    INS = [0]*(len(bins))
    
    NDCG = [0]*(len(bins))

    
    POS_sim_items = expl_dict
    NEG_sim_items  = list(sorted(dict(POS_sim_items).items(), key=lambda item: item[1],reverse=False))
    
    total_items=0
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
        
        POS_ranked_list = get_top_k(POS_masked, user_tensor, recommender_model, **kw_dict)
        
        if item_id in list(POS_ranked_list.keys()):
            POS_index = list(POS_ranked_list.keys()).index(item_id)+1
        else:
            POS_index = num_items
        NEG_index = get_index_in_the_list(NEG_masked, user_tensor, item_id, recommender_model, **kw_dict)+1

        # for pos:
        POS_at_1[i] = 1 if POS_index <=1 else 0
        POS_at_3[i] = 1 if POS_index <=3 else 0
        POS_at_5[i] = 1 if POS_index <=5 else 0
        POS_at_10[i] = 1 if POS_index <=10 else 0
        POS_at_20[i] = 1 if POS_index <=20 else 0
        POS_at_50[i] = 1 if POS_index <=50 else 0
        POS_at_100[i] = 1 if POS_index <=100 else 0

        # for neg:
        NEG_at_1[i] = 1 if NEG_index <=1 else 0
        NEG_at_3[i] = 1 if NEG_index <=3 else 0
        NEG_at_5[i] = 1 if NEG_index <=5 else 0
        NEG_at_10[i] = 1 if NEG_index <=10 else 0
        NEG_at_20[i] = 1 if NEG_index <=20 else 0
        NEG_at_50[i] = 1 if NEG_index <=50 else 0
        NEG_at_100[i] = 1 if NEG_index <=100 else 0

        # for del:
        DEL[i] = float(recommender_run(POS_masked, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())

        # for ins:
        INS[i] = float(recommender_run(user_tensor-POS_masked, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())

        #for NDCG:
        NDCG[i]= get_ndcg(list(POS_ranked_list.keys()),item_id, **kw_dict)
        
    res = [DEL, INS, NDCG, POS_at_1, POS_at_3, POS_at_5, POS_at_10, POS_at_20, POS_at_50, POS_at_100,  NEG_at_1, NEG_at_3, NEG_at_5, NEG_at_10, NEG_at_20, NEG_at_50, NEG_at_100]
    for i in range(len(res)):
        res[i] = np.array(res[i])
        
    return res

create_dictionaries = False # if it is the first time generating the explanations

if create_dictionaries:
    import time
    recommender.eval()
    # Evaluate the model on the test set
    
    jaccard_expl_dict = {}
    cosine_expl_dict = {}
    lime_expl_dict = {}
    accent_expl_dict = {}
    shap_expl_dict = {}
    lxr_expl_dict = {}    


    with torch.no_grad():
        for i in range(test_array.shape[0]):
            if i%500 == 0:
                print(i)
            start_time = time.time()
            user_vector = test_array[i]
            user_tensor = torch.FloatTensor(user_vector).to(device)
            user_id = int(test_data.index[i])

            item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).detach().cpu().numpy())
            item_vector =  items_array[item_id]
            item_tensor = torch.FloatTensor(item_vector).to(device)

            user_vector[item_id] = 0
            user_tensor[item_id] = 0

            # jaccard_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'jaccard')
            # cosine_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'cosine')
            # lime_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'lime')
            # accent_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'accent')
            # shap_expl_dict[user_id] = single_user_expl(user_vector, user_tensor,item_id, item_tensor, num_items, recommender, mask_type= 'shap',user_id = user_id)
            lxr_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'lxr')
            prev_time = time.time()
            #print("User {}, total time: {:.2f}".format(i,prev_time - start_time))


        # with open(Path(files_path,f'{recommender_name}_jaccard_expl_dict.pkl'), 'wb') as handle:
        #     pickle.dump(jaccard_expl_dict, handle)

        # with open(Path(files_path,f'{recommender_name}_cosine_expl_dict.pkl'), 'wb') as handle:
        #     pickle.dump(cosine_expl_dict, handle)

        # with open(Path(files_path,f'{recommender_name}_lime_expl_dict.pkl'), 'wb') as handle:
        #     pickle.dump(lime_expl_dict, handle)

        # with open(Path(files_path,f'{recommender_name}_accent_expl_dict.pkl'), 'wb') as handle:
        #     pickle.dump(accent_expl_dict, handle) 
            
        # with open(Path(files_path,f'{recommender_name}_shap_expl_dict.pkl'), 'wb') as handle:
        #     pickle.dump(shap_expl_dict, handle)

        with open(Path(files_path,f'{recommender_name}_lxr_expl_dict.pkl'), 'wb') as handle:
            pickle.dump(lxr_expl_dict, handle)

def eval_one_expl_type(expl_name):
    '''
    This function aggregates explanations for all test users
    and computes the average metric values across the entire test set.
    '''
    
    print(f' ============ Start explaining {data_name} {recommender_name} by {expl_name} ============')
    with open(Path(files_path,f'{recommender_name}_{expl_name}_expl_dict.pkl'), 'rb') as handle:
        expl_dict = pickle.load(handle)
    recommender.eval()
    # Evaluate the model on the test set
    num_of_bins = 11


    users_DEL = np.zeros(num_of_bins)
    users_INS = np.zeros(num_of_bins)
    NDCG = np.zeros(num_of_bins)
    POS_at_1 = np.zeros(num_of_bins)
    POS_at_3 = np.zeros(num_of_bins)
    POS_at_5 = np.zeros(num_of_bins)
    POS_at_10 = np.zeros(num_of_bins)
    POS_at_20 = np.zeros(num_of_bins)
    POS_at_50 = np.zeros(num_of_bins)
    POS_at_100 = np.zeros(num_of_bins)
    NEG_at_1 = np.zeros(num_of_bins)
    NEG_at_3 = np.zeros(num_of_bins)
    NEG_at_5 = np.zeros(num_of_bins)
    NEG_at_10 = np.zeros(num_of_bins)
    NEG_at_20 = np.zeros(num_of_bins)
    NEG_at_50 = np.zeros(num_of_bins)
    NEG_at_100 = np.zeros(num_of_bins)

    num_of_bins=10


    with torch.no_grad():
        for i in range(test_array.shape[0]):
            start_time = time.time()
            user_vector = test_array[i]
            user_tensor = torch.FloatTensor(user_vector).to(device)
            user_id = int(test_data.index[i])

            item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).detach().cpu().numpy())
            item_vector =  items_array[item_id]
            item_tensor = torch.FloatTensor(item_vector).to(device)

            user_vector[item_id] = 0
            user_tensor[item_id] = 0

            user_expl = expl_dict[user_id]

            res = single_user_metrics(user_vector, user_tensor, item_id, item_tensor, num_of_bins, recommender, user_expl, **kw_dict)
            users_DEL += res[0]
            users_INS += res[1]
            NDCG += res[2]
            POS_at_1 += res[3]
            POS_at_3 += res[4]
            POS_at_5 += res[5]
            POS_at_10 += res[6]
            POS_at_20 += res[7]
            POS_at_50 += res[8]
            POS_at_100 += res[9]
            NEG_at_1 += res[10]
            NEG_at_3 += res[11]
            NEG_at_5 += res[12]
            NEG_at_10 += res[13]
            NEG_at_20 += res[14]
            NEG_at_50 += res[15]
            NEG_at_100 += res[16]

            if i%500 == 0:
                prev_time = time.time()
                print("User {}, total time: {:.2f}".format(i,prev_time - start_time))

    a = i+1

    print(f'users_DEL_{expl_name}: ', np.mean(users_DEL)/a)
    print(f'users_INS_{expl_name}: ', np.mean(users_INS)/a)
    print(f'NDCG_{expl_name}: ', np.mean(NDCG)/a)
    print(f'POS_at_1_{expl_name}: ', np.mean(POS_at_1)/a)
    print(f'POS_at_3_{expl_name}: ', np.mean(POS_at_3)/a)
    print(f'POS_at_5_{expl_name}: ', np.mean(POS_at_5)/a)
    print(f'POS_at_10_{expl_name}: ', np.mean(POS_at_10)/a)
    print(f'POS_at_20_{expl_name}: ', np.mean(POS_at_20)/a)
    print(f'POS_at_50_{expl_name}: ', np.mean(POS_at_50)/a)
    print(f'POS_at_100_{expl_name}: ', np.mean(POS_at_100)/a)
    print(f'NEG_at_1_{expl_name}: ', np.mean(NEG_at_1)/a)
    print(f'NEG_at_3_{expl_name}: ', np.mean(NEG_at_3)/a)
    print(f'NEG_at_5_{expl_name}: ', np.mean(NEG_at_5)/a)
    print(f'NEG_at_10_{expl_name}: ', np.mean(NEG_at_10)/a)
    print(f'NEG_at_20_{expl_name}: ', np.mean(NEG_at_20)/a)
    print(f'NEG_at_50_{expl_name}: ', np.mean(NEG_at_50)/a)
    print(f'NEG_at_100_{expl_name}: ', np.mean(NEG_at_100)/a)

    
    print(np.mean(users_DEL)/a , np.mean(users_INS)/a, np.mean(NDCG)/a , np.mean(POS_at_1)/a , np.mean(POS_at_3)/a, np.mean(NEG_at_1)/a, np.mean(NEG_at_3)/a, np.mean(POS_at_5)/a , np.mean(NEG_at_5)/a, np.mean(POS_at_10)/a , np.mean(NEG_at_10)/a , np.mean(POS_at_20)/a , np.mean(NEG_at_20)/a, np.mean(POS_at_50)/a , np.mean(NEG_at_50)/a, np.mean(POS_at_100)/a , np.mean(NEG_at_100)/a)

# expl_names_list = ['lxr' ,'lime','jaccard', 'cosine','shap','accent']
expl_names_list = ['lxr'] # specify the names of the baselines for which you wish to calculate the metrics values.

for expl_name in expl_names_list:
    eval_one_expl_type(expl_name)