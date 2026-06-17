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
import ipynb
import importlib
import random


# Create the interpolation steps between the baseline and the target
def get_interpolated_values(baseline, target, num_steps):
    """this function returns a tensor of all the vecrots interpolation steps"""
    baseline = baseline.cpu()
    target = target.cpu()

    delta = target - baseline

    # Make steps between 0 and 1 
    scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis]
        
    shape = (num_steps + 1,) + delta.shape
    deltas = scales * np.broadcast_to(delta.detach().cpu().numpy(), shape)
    interpolated_activations = baseline + deltas

    return interpolated_activations  #.to(device).clone().detach().requires_grad_(True)



# 2
# Gradient calculations with respect to the recommended item

def get_grads_wrt_to_user_tensor(model, user_tensor, all_items_tensor, item_id, recommender_name, device):
    model.eval()
    model.zero_grad()

    for param in model.parameters():
        param.requires_grad = True
        
    # Different implementation of the recommenders 
    if recommender_name == "VAE":
        preds = model(user_tensor)[0]
    else: # MLP or NCF
        preds = model(user_tensor, all_items_tensor)
    
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[item_id] = 1

    score = torch.sum(one_hot * preds)
    score.backward()
    
    with torch.no_grad():
        vector_grad = user_tensor.grad.detach()
    user_tensor.requires_grad = False
    return vector_grad




# 3
# Sampling methods
# randomly sample a user vector from the training set and using it as the baseline

def create_baseline_random_user(train_array, device):
    i = np.random.randint(0, train_array.shape[0]) # Randomly select index
    base = train_array[i]  #This is the baseline!
    base_tensor = torch.Tensor(base).to(device)
    
    return base_tensor

# Sampling method - sample uniformly a number between zero to one for each item in the vector. Use this sampeled vector as the baseline
def create_baseline_random_items(user_tensor):
    baseline_tensor = torch.zeros_like(user_tensor)

    for i in range(user_tensor.numel()):  # numel() gives the total number of elements in the tensor
        random_sample = np.random.rand()
        baseline_tensor[i] = random_sample
        
    return baseline_tensor

# Sampling method - sample a number and decide 0 or 1 according to the popularity of the item
def create_baseline_by_pop(user_tensor, pop_array):
    baseline_tensor = torch.zeros_like(user_tensor)

    for i in range(user_tensor.numel()):  # numel() gives the total number of elements in the tensor
        random_sample = np.random.rand()
        baseline_tensor[i] = 1 if random_sample < pop_array[i] else 0
        
    return baseline_tensor

# 4
# Find the explanation map
def find_ip_mask(model, user_tensor, item_id, all_items_tensor, num_steps, method, device, recommender_name, train_array, pop_array):
    if method == "base":
        baseline = torch.zeros_like(user_tensor)
    elif method == "sample_random_user":  
        baseline = create_baseline_random_user(train_array, device)
    elif method == "sample_random_items":
        baseline = create_baseline_random_items(user_tensor)
    elif method == "sample_items_by_pop": 
        baseline = create_baseline_by_pop(user_tensor, pop_array)
    
    
    interpolations = get_interpolated_values(baseline, user_tensor, num_steps)
    
    gradients = []
    count = -1 
    for i in interpolations: 
        count += 1
        if count != 0:
            i = i.to(device).detach()
            i.requires_grad = True
            grad_tensor = get_grads_wrt_to_user_tensor(model=model, user_tensor=i, all_items_tensor=all_items_tensor, item_id=item_id, recommender_name=recommender_name, device=device)
            gradients.append(grad_tensor)

    stacked_gradients = torch.stack(gradients, dim=0)
    ip_explanation = torch.mean(stacked_gradients, dim=0)
   
    x_masked = user_tensor*ip_explanation 
    
    item_sim_dict = {}
    for i,j in enumerate(x_masked):
        if j:
            item_sim_dict[i]=x_masked[i] 
        
    return item_sim_dict

