"""Sample code to generate counterfactual explanations using multiple methods."""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict

# Import architectures
from recommenders_architecture import VAE
from LXR_training import Explainer
from lime import LimeBase, distance_to_proximity, get_lime_args
from help_functions import recommender_run, get_top_k

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_name = "ML1M"
num_items = 3381
num_users = 6037

# Paths
base_path = Path(__file__).parent.parent.parent.parent
data_path = base_path / "datasets" / data_name
checkpoint_path = base_path / "checkpoints" / "recommenders" / "VAE"

# Load data
train_data = pd.read_csv(data_path / f"train_data_{data_name}.csv", index_col=0)
test_data = pd.read_csv(data_path / f"test_data_{data_name}.csv", index_col=0)
static_test_data = pd.read_csv(data_path / f"static_test_data_{data_name}.csv", index_col=0)
test_data['user_id'] = test_data.index

# Load all required dictionaries for different explanation methods
with open(data_path / f"pop_dict_{data_name}.pkl", "rb") as f:
    pop_dict = pickle.load(f)

with open(data_path / f"jaccard_based_sim_{data_name}.pkl", "rb") as f:
    jaccard_dict = pickle.load(f)

with open(data_path / f"cosine_based_sim_{data_name}.pkl", "rb") as f:
    cosine_dict = pickle.load(f)

with open(data_path / f"item_to_cluster_VAE_{data_name}.pkl", "rb") as f:
    item_to_cluster = pickle.load(f)

with open(data_path / f"shap_values_VAE_{data_name}.pkl", "rb") as f:
    shap_values = pickle.load(f)

# Complete symmetric dictionaries
for i in range(num_items):
    for j in range(i, num_items):
        jaccard_dict[(j, i)] = jaccard_dict[(i, j)]
        cosine_dict[(j, i)] = cosine_dict[(i, j)]

# Prepare arrays
train_array = train_data.to_numpy()
test_array = static_test_data.iloc[:, :-2].to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)
pop_array = np.array([pop_dict.get(i, 0) for i in range(len(pop_dict))])

# Load recommender
VAE_config = {"enc_dims": [512, 128], "dropout": 0.5, "anneal_cap": 0.2, "total_anneal_steps": 200000}
kw_dict = {
    "device": device,
    "num_items": num_items,
    "pop_array": pop_array,
    "all_items_tensor": all_items_tensor,
    "static_test_data": static_test_data,
    "items_array": items_array,
    "output_type": "multiple",
    "recommender_name": "VAE"
}
recommender = VAE(VAE_config, **kw_dict)
recommender.load_state_dict(torch.load(checkpoint_path / "VAE_ML1M_0.0007_128_10.pt", map_location=device))
recommender.eval()
for param in recommender.parameters():
    param.requires_grad = False

# Load explainer
explainer = Explainer(num_items, num_items, 128)
explainer.load_state_dict(torch.load(checkpoint_path / "LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426LXRMAIN.pt", map_location=device))
explainer.eval()
for param in explainer.parameters():
    param.requires_grad = False

# Initialize LIME
lime = LimeBase(distance_to_proximity)


# Baseline explanation functions
def find_jaccard_mask(user_vector, item_id):
    """Generate counterfactual explanation using Jaccard similarity."""
    user_hist = user_vector.copy()
    user_hist[item_id] = 0
    item_jaccard_dict = {}
    for i, j in enumerate(user_hist > 0):
        if j:
            if (i, item_id) in jaccard_dict:
                item_jaccard_dict[i] = jaccard_dict[(i, item_id)]
            else:
                item_jaccard_dict[i] = 0
    return item_jaccard_dict


def find_cosine_mask(user_vector, item_id):
    """Generate counterfactual explanation using Cosine similarity."""
    user_hist = user_vector.copy()
    user_hist[item_id] = 0
    item_cosine_dict = {}
    for i, j in enumerate(user_hist > 0):
        if j:
            if (i, item_id) in cosine_dict:
                item_cosine_dict[i] = cosine_dict[(i, item_id)]
            else:
                item_cosine_dict[i] = 0
    return item_cosine_dict


def find_lime_mask(user_vector, item_id, recommender):
    """Generate counterfactual explanation using LIME."""
    user_hist = user_vector.copy()
    user_hist[item_id] = 0
    user_hist_size = int(np.sum(user_hist))
    
    user_tensor = torch.FloatTensor(user_hist).to(device)
    item_tensor = torch.FloatTensor(items_array[item_id]).to(device)
    
    neighborhood_data, neighborhood_labels, distances, item_id_out = get_lime_args(
        user_hist, item_id, recommender, all_items_tensor,
        min_pert=50, max_pert=100, num_of_perturbations=150,
        seed=item_id, **kw_dict
    )
    
    most_pop_items = lime.explain_instance_with_data(
        neighborhood_data, neighborhood_labels, distances,
        item_id_out, user_hist_size, 'highest_weights', pos_neg='POS'
    )
    
    return dict(most_pop_items)


def find_fia_mask(user_tensor, item_tensor, item_id, recommender):
    """Generate counterfactual explanation using Feature Influence Analysis."""
    y_pred = recommender_run(user_tensor, recommender, item_tensor, item_id, **kw_dict).to(device)
    items_fia = {}
    user_hist = user_tensor.cpu().detach().numpy().astype(int)
    
    for i in range(num_items):
        if user_hist[i] == 1:
            user_hist[i] = 0
            user_tensor_temp = torch.FloatTensor(user_hist).to(device)
            y_pred_without_item = recommender_run(
                user_tensor_temp, recommender, item_tensor, item_id, 'single', **kw_dict
            ).to(device)
            infl_score = y_pred - y_pred_without_item
            items_fia[i] = float(infl_score.cpu().detach().numpy())
            user_hist[i] = 1
    
    return items_fia


def find_accent_mask(user_tensor, item_id, recommender, top_k=5):
    """Generate counterfactual explanation using Accent."""
    items_accent = defaultdict(float)
    factor = top_k - 1
    user_accent_hist = user_tensor.cpu().detach().numpy().astype(int).copy()
    
    # Get top-k items
    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender, **kw_dict).keys())
    
    if top_k == 1:
        top_k_indices = [sorted_indices[0]]
    else:
        top_k_indices = sorted_indices[:top_k]
    
    for iteration, item_k_id in enumerate(top_k_indices):
        # Set top-k items to 0 in the user's history
        user_accent_hist[item_k_id] = 0
        user_tensor_temp = torch.FloatTensor(user_accent_hist).to(device)
        
        item_vector = items_array[item_k_id]
        item_tensor_temp = torch.FloatTensor(item_vector).to(device)
        
        # Check influence of the items in the history on this specific item in top-k
        fia_dict = find_fia_mask(user_tensor_temp, item_tensor_temp, item_k_id, recommender)
        
        # Sum up all differences between influence on top1 and other top-k values
        if not iteration:
            for key in fia_dict.keys():
                items_accent[key] = fia_dict[key] * factor
        else:
            for key in fia_dict.keys():
                items_accent[key] -= fia_dict[key]
    
    for key in items_accent.keys():
        items_accent[key] *= -1
    
    return dict(items_accent)


def find_shapley_mask(user_tensor, user_id):
    """Generate counterfactual explanation using SHAP values."""
    item_shap = {}
    shapley_values = shap_values[shap_values[:, 0].astype(int) == user_id][:, 1:]
    user_vector = user_tensor.cpu().detach().numpy().astype(int)
    
    for i in np.where(user_vector.astype(int) == 1)[0]:
        items_cluster = item_to_cluster[i]
        item_shap[i] = shapley_values.T[int(items_cluster)][0]
    
    return item_shap


def find_lxr_mask(user_tensor, item_tensor):
    """Generate counterfactual explanation using LXR."""
    expl_scores = explainer(user_tensor, item_tensor)
    x_masked = user_tensor * expl_scores
    item_sim_dict = {}
    for i, j in enumerate(x_masked > 0):
        if j:
            item_sim_dict[i] = float(x_masked[i].cpu().detach().numpy())
    
    return item_sim_dict


# Generate counterfactual explanations for test instances
def get_counterfactual_explanation(user_tensor, user_vector, user_id, item_id, explainer, recommender, method='lxr'):
    """Generate counterfactual explanation for a user-item pair using the specified method."""
    item_tensor = torch.Tensor(items_array[item_id]).to(device)
    
    if method == 'lxr':
        sim_items = find_lxr_mask(user_tensor, item_tensor)
    elif method == 'jaccard':
        sim_items = find_jaccard_mask(user_vector, item_id)
    elif method == 'cosine':
        sim_items = find_cosine_mask(user_vector, item_id)
    elif method == 'lime':
        sim_items = find_lime_mask(user_vector, item_id, recommender)
    elif method == 'accent':
        sim_items = find_accent_mask(user_tensor, item_id, recommender)
    elif method == 'shap':
        sim_items = find_shapley_mask(user_tensor, user_id)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Sort by explanation score
    sorted_items = sorted(sim_items.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_items


# Define which methods to compare
methods = ['lxr', 'jaccard', 'cosine', 'lime', 'accent', 'shap']

# Loop through test instances
print(f"Processing {len(test_array)} test instances...")
num_samples = min(5, len(test_array))  # Show first 5 examples

for i in range(num_samples):
    user_vector = test_array[i][:num_items]
    user_tensor = torch.Tensor(user_vector).to(device)
    user_id = int(test_data.index[i])
    
    # Get recommended item
    user_res = recommender(user_tensor)[:num_items]
    user_catalog = torch.ones_like(user_tensor) - user_tensor
    user_recommendations = torch.mul(user_res.squeeze(), user_catalog)
    recommended_item = int(torch.argmax(user_recommendations))
    
    print(f"\n{'='*80}")
    print(f"User {user_id}: Recommended item {recommended_item}")
    print(f"{'='*80}")
    
    # Get counterfactual explanations using all methods
    for method in methods:
        try:
            cf_explanation = get_counterfactual_explanation(
                user_tensor, user_vector, user_id, recommended_item,
                explainer, recommender, method=method
            )
            print(f"\n{method.upper():10} - Top 5: {cf_explanation[:5]}")
        except Exception as e:
            print(f"\n{method.upper():10} - Error: {e}")

print("\n" + "="*80)
print("Done!")
print("="*80)


# Sample output:
# ------ Runnig VAE on ML1M -----------
# Processing 1208 test instances...

# User 4829: Recommended item 996
# Top 5 counterfactual items: [(236, 0.9364509582519531), (719, 0.9303868412971497), (2379, 0.898897647857666), (2959, 0.6616955995559692), (1807, 0.3589291572570801)]

# User 4830: Recommended item 559
# Top 5 counterfactual items: [(1142, 0.8641871809959412), (269, 0.8050122261047363), (1013, 0.6266318559646606), (2339, 0.4472600817680359), (994, 0.20843404531478882)]

# User 4831: Recommended item 269
# Top 5 counterfactual items: [(547, 0.9902001023292542), (2262, 0.9068177938461304), (1013, 0.9056581854820251), (48, 0.888690173625946), (559, 0.8068826198577881)]

# User 4832: Recommended item 954
# Top 5 counterfactual items: [(1076, 0.9999886751174927), (1023, 0.9993877410888672), (652, 0.9981630444526672), (3091, 0.9965981841087341), (959, 0.995537281036377)]

# User 4833: Recommended item 773
# Top 5 counterfactual items: [(931, 0.9999998807907104), (1025, 0.9999998807907104), (1028, 0.9999998807907104), (1594, 0.9999998807907104), (2734, 0.9999998807907104)]

# User 4834: Recommended item 1950
# Top 5 counterfactual items: [(2339, 1.0), (2515, 1.0), (3168, 1.0), (2346, 0.9999997615814209), (2424, 0.9999996423721313)]

# User 4835: Recommended item 3029
# Top 5 counterfactual items: [(2987, 1.0), (3086, 0.9999998807907104), (3081, 0.9999996423721313), (3201, 0.9999991655349731), (1716, 0.9999990463256836)]

# User 4836: Recommended item 1075
# Top 5 counterfactual items: [(705, 0.9999998807907104), (924, 0.9999998807907104), (1172, 0.9999997615814209), (770, 0.9999996423721313), (223, 0.9999994039535522)]

# User 4837: Recommended item 1063
# Top 5 counterfactual items: [(1023, 0.990998387336731), (2429, 0.9840937256813049), (37, 0.9739718437194824), (2008, 0.9569032192230225), (160, 0.9413785338401794)]

# User 4838: Recommended item 489
# Top 5 counterfactual items: [(994, 0.9957962036132812), (841, 0.9946692585945129), (719, 0.7835817933082581), (1676, 0.7748198509216309), (1048, 0.5706575512886047)]
