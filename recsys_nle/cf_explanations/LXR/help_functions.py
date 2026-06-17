# ### This notebook includes the framework's functions that are being used in all notebooks.
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
import torch.nn.functional as F
import copy

# # Help Functions

# a function that samples different train data variation for a diverse training
def sample_indices(data, **kw):
    num_items = kw['num_items']
    pop_array = kw['pop_array']
    
    matrix = np.array(data)[:,:num_items] # keep only items columns, remove demographic features columns
    zero_indices = []
    one_indices = []

    for row in matrix:
        zero_idx = np.where(row == 0)[0]
        one_idx = np.where(row == 1)[0]
        probs = pop_array[zero_idx]
        probs = probs/ np.sum(probs)

        sampled_zero = np.random.choice(zero_idx, p = probs) # sample negative interactions according to items popularity 
        zero_indices.append(sampled_zero)

        sampled_one = np.random.choice(one_idx) # sample positive interactions from user's history
        data.iloc[row, sampled_one] = 0
        one_indices.append(sampled_one)

    data['pos'] = one_indices
    data['neg'] = zero_indices
    return np.array(data)

# a function that returns a specific item's rank in user's recommendations list
def get_index_in_the_list(user_tensor, original_user_tensor, item_id, recommender, **kw):
    top_k_list = list(get_top_k(user_tensor, original_user_tensor, recommender, **kw).keys())
    return top_k_list.index(item_id)

# returns a dictionary of items and recommendations scores for a specific user
def get_top_k(user_tensor, original_user_tensor, model, **kw):
    all_items_tensor = kw['all_items_tensor']
    num_items = kw['num_items']
    
    item_prob_dict = {}
    output_model = [float(i) for i in recommender_run(user_tensor, model, all_items_tensor, None, 'vector', **kw).cpu().detach().numpy()]
    original_user_vector = np.array(original_user_tensor.cpu())[:num_items]
    catalog = np.ones_like(original_user_vector)- original_user_vector
    output = catalog*output_model
    for i in range(len(output)):
        if catalog[i] > 0:
            item_prob_dict[i]=output[i]
    sorted_items_by_prob  = sorted(item_prob_dict.items(), key=lambda item: item[1],reverse=True)
    return dict(sorted_items_by_prob)

# a function that wraps the different recommenders types 
# returns user's scores with respect to a certain item or for all items 
def recommender_run(user_tensor, recommender, item_tensor = None, item_id= None, wanted_output = 'single', **kw):
    output_type=kw['output_type']
    if output_type == 'single':
        if wanted_output == 'single':
            return recommender(user_tensor, item_tensor)
        else:
            return recommender(user_tensor, item_tensor).squeeze()
    else:
        if wanted_output == 'single':
            return recommender(user_tensor).squeeze()[item_id]
        else:
            return recommender(user_tensor).squeeze()

# evaluate recommenders on test set and return HR@10, HR@50, HR@100, MRR and MPR
def recommender_evaluations(recommender, **kw):
    static_test_data = kw['static_test_data'].copy()
    device = kw['device']
    items_array = kw['items_array']
    num_items = kw['num_items']

    counter_10 = 0
    counter_50 = 0
    counter_100 = 0
    RR = 0
    PR = 0
    temp_test_array = np.array(static_test_data)
    n = temp_test_array.shape[0]
    for i in range(n):
        item_id = temp_test_array[i][-2]
        item_tensor = items_array[item_id]
        user_tensor = torch.Tensor(temp_test_array[i][:-2]).to(device)
        user_tensor[item_id]=0
        index = get_index_in_the_list(user_tensor, user_tensor, item_id, recommender, **kw) +1 
        if index <= 10:
            counter_10 +=1 
        if index <= 50:
            counter_50 +=1 
        if index <= 100:
            counter_100 +=1             
        RR += np.reciprocal(index)
        PR += index/num_items
        
    return counter_10/n, counter_50/n, counter_100/n,  RR/n, PR*100/n

# get user's top recommended item
def get_user_recommended_item(user_tensor, recommender, user_id= None, **kw):
    all_items_tensor = kw['all_items_tensor']
    num_items = kw['num_items']
    user_res = recommender_run(user_tensor, recommender, all_items_tensor, None, 'vector', **kw)[:num_items]
    user_tensor = user_tensor[:num_items]
    user_catalog = torch.ones_like(user_tensor)-user_tensor
    user_recommenations = torch.mul(user_res, user_catalog)
    return(torch.argmax(user_recommenations))

# calculate the ndcg score of the restored recommendations list after perturbating the user's data.
def get_ndcg(ranked_list, target_item, **kw):
    device = kw['device']
    if target_item not in ranked_list:
        return 0.0

    target_idx = torch.tensor(ranked_list.index(target_item), device=device)
    dcg = torch.reciprocal(torch.log2(target_idx + 2))

    return dcg.item()