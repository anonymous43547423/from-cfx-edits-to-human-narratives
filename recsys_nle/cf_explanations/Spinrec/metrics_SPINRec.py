
# Imports and initial settings

import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import datetime
import gc
np.random.seed(42)

export_dir = Path(os.getcwd())
checkpoints_path = Path(export_dir, "checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_type_dict = {
    "VAE":"multiple",
    "MLP":"single",
    "NCF": "single"}

num_users_dict = {
    "ML1M":6037,
    "Yahoo":13797, 
    "Pinterest":19155}

num_items_dict = {
    "ML1M":3381,
    "Yahoo":4604, 
    "Pinterest":9362}


recommender_path_dict = {
    ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","MLP"):Path(checkpoints_path, "MLP1_ML1M_0.0076_256_7.pt"),
    ("ML1M","NCF"):Path(checkpoints_path, "NCF_ML1M_5e-05_64_16.pt"),
    
    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),
    ("Yahoo","NCF"):Path(checkpoints_path, "NCF_Yahoo_0.001_64_21_0.pt"),
    
    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_12_18_0.0001_256.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt"),
    ("Pinterest","NCF"):Path(checkpoints_path, "NCF2_Pinterest_9e-05_32_9_10.pt"),}


hidden_dim_dict = {
    ("ML1M","VAE"): None,
    ("ML1M","MLP"): 32,
    ("ML1M","NCF"): 8,

    ("Yahoo","VAE"): None,
    ("Yahoo","MLP"):32,
    ("Yahoo","NCF"):8,
    
    ("Pinterest","VAE"): None,
    ("Pinterest","MLP"):512,
    ("Pinterest","NCF"): 64,
}

# Important to edit:

data_names = ["ML1M"]
#data_names = ["ML1M", "Yahoo", "Pinterest"]

recommender_names = ["NCF"]
# recommender_names = ["MLP", "VAE", "NCF"]

expl_names_list = ['PI']

# Parameters

num_steps = 10
num_of_random_users = 10
list_of_nums = [1,2,4,5,8,10] #,15,20,25,30,35,40,45,50]
num_lists_needed = len(list_of_nums)
method = 'sample_random_user'

new_file_name = f"NEW_NAME_OF_YOUR_CHOICE"
new_file_name

# Import Functions form other notebooks

from recommenders_architecture import *
from help_functions import *
from SPINRec_functions import *


# Evaluation help functions

def single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender_model, all_items_tensor, user_id = None, mask_type = None):
    user_hist_size = np.sum(user_vector)
    
    if mask_type == 'PI':
        sim_items = find_ip_mask(model=recommender_model, user_tensor=user_tensor, item_id=item_id, all_items_tensor=all_items_tensor, num_steps=num_steps, method=method, device=device, recommender_name=recommender_name, train_array=train_array, pop_array=pop_array)   
    else:
        print("Wrong notebook!!")
        
    POS_sim_items  = list(sorted(sim_items.items(), key=lambda item: item[1],reverse=True))[0:user_hist_size]
    return POS_sim_items

# START HERE

for data_name in data_names:
    
    DP_DIR = Path("processed_data", data_name)
    files_path = Path(export_dir, DP_DIR)

    num_users = num_users_dict[data_name] 
    num_items = num_items_dict[data_name] 
    num_features = num_items_dict[data_name]
    
    with open(Path(files_path, f'pop_dict_{data_name}.pkl'), 'rb') as f:
        pop_dict = pickle.load(f) 
    pop_array = np.zeros(len(pop_dict))
    for key, value in pop_dict.items():
        pop_array[key] = value

    # Data 
    train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
    test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
    
    train_array = train_data.to_numpy()
    test_array = test_data.to_numpy()
    items_array = np.eye(num_items)
    all_items_tensor = torch.Tensor(items_array).to(device)

    
    for recommender_name in recommender_names:
        output_type = output_type_dict[recommender_name]
        hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
        
        recommender_path = recommender_path_dict[(data_name,recommender_name)]

        kw_dict = {'device':device,
                  'num_items': num_items,
                  'demographic':False,
                  'num_features':num_features,
                  'pop_array':pop_array,
                  'all_items_tensor':all_items_tensor,
                  'items_array':items_array,
                  'output_type':output_type,
                  'recommender_name':recommender_name}


        recommender = load_recommender(data_name, hidden_dim, checkpoints_path, recommender_path, **kw_dict)

        file_mode = 'a' if os.path.exists(new_file_name) else 'w'
        with open(new_file_name, file_mode) as file:
            file.write(f' ============ This stats are for {data_name} dataset ============\n')
            file.write(f' ============ & for the recommender {recommender_name} ============\n')
            
        file_mode = 'a'
        
        for expl_name in expl_names_list:
            if expl_name == "PI":
                with open(new_file_name, file_mode) as file:
                    file.write(f' ============ Start explaining by {expl_name} ============\n')
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f' ============ Start time {now} ============\n')

                ip_path = Path(files_path, "PI", recommender_name, method)

                recommender.eval()
                users_DEL = {}
                users_INS = {}
                NDCG = {}
                POS_at_5 = {}
                POS_at_10 = {}
                POS_at_20 = {}
                
                all_dicts = [users_DEL, users_INS, NDCG, POS_at_5, POS_at_10, POS_at_20]

                # Append the required number of empty lists to each list in all_lists
                for each_dict in all_dicts:
                    for n in range(num_lists_needed):
                        each_dict[f"num_of_users_{list_of_nums[n]}"] = []

                num_of_bins=10

                for i in range(test_array.shape[0]):
                    if i % 400 == 0:
                        with open(new_file_name, file_mode) as file:
                            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            file.write(f' ============ User number {i} has started at {now} \n')
                    user_vector = test_array[i]
                    user_tensor = torch.FloatTensor(user_vector).to(device)
                    user_id = int(test_data.index[i])
                    item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).detach().cpu().numpy())
                    item_vector =  items_array[item_id]
                    item_tensor = torch.FloatTensor(item_vector).to(device)
                    user_vector[item_id] = 0
                    user_tensor[item_id] = 0

                    user_DEL = []
                    user_INS = []
                    user_NDCG = []
                    user_POS_at_5 = []
                    user_POS_at_10 = []
                    user_POS_at_20 = []

                    for j in range(1, num_of_random_users + 1):
                        user_expl_dict = {}
                        user_expl = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, all_items_tensor=kw_dict['all_items_tensor'], mask_type= 'PI')
                        user_expl_dict[user_id] = user_expl
                        num = str(j)

                        new_dict_path = Path(ip_path, num, f'PI_expl_dict_user_{i}.pkl')
                        new_dict_path.parent.mkdir(parents=True, exist_ok=True) 

                        with open(new_dict_path, 'wb') as handle:
                            pickle.dump(user_expl_dict, handle)

                        with torch.no_grad():
                            res = single_user_metrics(user_vector, user_tensor, item_id, item_tensor, num_of_bins, recommender, user_expl, **kw_dict)
                            
                            user_DEL.append(np.mean(res[0]))
                            user_INS.append(np.mean(res[1]))
                            user_NDCG.append(np.mean(res[2]))
                            user_POS_at_5.append(np.mean(res[3]))
                            user_POS_at_10.append(np.mean(res[4]))
                            user_POS_at_20.append(np.mean(res[5]))

                            if j in list_of_nums:
                                key_name = f"num_of_users_{j}"

                                users_DEL[key_name].append(min(user_DEL))
                                users_INS[key_name].append(max(user_INS))
                                NDCG[key_name].append(min(user_NDCG))
                                POS_at_5[key_name].append(min(user_POS_at_5))
                                POS_at_10[key_name].append(min(user_POS_at_10))
                                POS_at_20[key_name].append(min(user_POS_at_20))


        with open(new_file_name, file_mode) as file:                        
            final = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f' \n')
            file.write(f' ============ All users are done at {final} ============\n')

            results_path = Path(files_path, "PI", recommender_name, method, "results")      
            results_path.mkdir(parents=True, exist_ok=True)   

            ip_final_statistics = {}
            for k in range(num_lists_needed):
                spot = f"num_of_users_{list_of_nums[k]}"
                ip_final_statistics[f"{list_of_nums[k]}_random"] = [np.mean(users_DEL[spot]), np.mean(users_INS[spot]), np.mean(reciprocal[spot]), np.mean(NDCG[spot]), np.mean(POS_at_1[spot]), np.mean(NEG_at_1[spot]), np.mean(rank_at_1[spot]), np.mean(POS_at_5[spot]), np.mean(NEG_at_5[spot]), np.mean(rank_at_5[spot]), np.mean(POS_at_10[spot]), np.mean(NEG_at_10[spot]), np.mean(rank_at_10[spot]), np.mean(POS_at_20[spot]), np.mean(NEG_at_20[spot]), np.mean(rank_at_20[spot]), np.mean(POS_at_50[spot]), np.mean(NEG_at_50[spot]), np.mean(rank_at_50[spot]), np.mean(POS_at_100[spot]), np.mean(NEG_at_100[spot]), np.mean(rank_at_100[spot])]
                file.write(f' ============ Results for {list_of_nums[k]} random users: ============ \n')
                file.write(f'{ip_final_statistics[f"{list_of_nums[k]}_random"]} \n')

            with open(Path(results_path,f'PI_{method}_final_statistics.pkl'), 'wb') as handle:
                pickle.dump(ip_final_statistics, handle)
                
            file.write(f"All Saved \n")

print("Done")

