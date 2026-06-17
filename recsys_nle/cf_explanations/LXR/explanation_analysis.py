"""Sample code to generate counterfactual explanations using multiple methods."""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict






# Import architectures
try:
    from recommenders_architecture import VAE
    from LXR_training import Explainer
    from lime import LimeBase, distance_to_proximity, get_lime_args
    from help_functions import recommender_run, get_top_k

    # Import SPINRec functions
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "Spinrec"))
    from SPINRec_functions import find_ip_mask

    # Import utility functions
    from utility import get_movie_name, analyze_missing_movies
except ImportError:
    # When imported as a package, use absolute imports.
    # Note: Explainer is not imported here because LXR_training has argparse at module level.
    # Functions that need Explainer should only be run when executing directly.
    from recsys_nle.cf_explanations.LXR.recommenders_architecture import VAE
    from recsys_nle.cf_explanations.LXR.lime import LimeBase, distance_to_proximity, get_lime_args
    from recsys_nle.cf_explanations.LXR.help_functions import recommender_run, get_top_k
    from recsys_nle.cf_explanations.LXR.utility import get_movie_name, analyze_missing_movies

    import sys
    sys.path.append(str(Path(__file__).parent.parent / "Spinrec"))
    from recsys_nle.cf_explanations.Spinrec.SPINRec_functions import find_ip_mask


def setup_paths(data_name):
    """Set up all necessary paths for data and checkpoints."""
    base_path = Path(__file__).parent.parent.parent.parent
    data_path = base_path / "datasets" / data_name
    checkpoint_path = base_path / "checkpoints" / "recommenders" / "VAE"
    return base_path, data_path, checkpoint_path


def load_data_and_movies(data_path, data_name, num_items):
    """Load training data, test data, popularity dictionary, and movie information."""
    # Load data
    train_data = pd.read_csv(data_path / f"train_data_{data_name}.csv", index_col=0)
    test_data = pd.read_csv(data_path / f"test_data_{data_name}.csv", index_col=0)
    static_test_data = pd.read_csv(data_path / f"static_test_data_{data_name}.csv", index_col=0)
    test_data['user_id'] = test_data.index
    
    with open(data_path / f"pop_dict_{data_name}.pkl", "rb") as f:
        pop_dict = pickle.load(f)
    
    # Load additional dictionaries for baseline methods
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
    
    # Load movie details from .dat file
    movies_df = pd.read_csv(
        data_path / "datafiles" / "movies.dat",
        sep='::',
        engine='python',
        encoding='latin-1',
        names=['MovieID', 'MovieName', 'Genre']
    )
    movies_dict = dict(zip(movies_df['MovieID'] - 1, movies_df['MovieName']))  # Adjust for 0-indexing
    
    # Optional: Analyze movie data for diagnostics
    # analyze_missing_movies(movies_df, movies_dict, num_items)
    
    return train_data, test_data, static_test_data, pop_dict, movies_df, movies_dict, jaccard_dict, cosine_dict, item_to_cluster, shap_values


def prepare_arrays(train_data, test_data, static_test_data, pop_dict, num_items, device):
    """Prepare numpy arrays and tensors for the models."""
    train_array = train_data.to_numpy()
    test_array = static_test_data.iloc[:, :-2].to_numpy()
    items_array = np.eye(num_items)
    all_items_tensor = torch.Tensor(items_array).to(device)
    pop_array = np.array([pop_dict.get(i, 0) for i in range(len(pop_dict))])
    
    return train_array, test_array, items_array, all_items_tensor, pop_array


def load_models(checkpoint_path, num_items, pop_array, all_items_tensor, items_array, static_test_data, device):
    """Load and initialize the recommender and explainer models."""
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
    explainer.load_state_dict(
        torch.load(
            checkpoint_path / "LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426LXRMAIN.pt",
            map_location=device
        )
    )
    explainer.eval()
    for param in explainer.parameters():
        param.requires_grad = False
    
    # Initialize LIME
    lime = LimeBase(distance_to_proximity)
    
    return recommender, explainer, lime, kw_dict


# Baseline explanation functions
def find_jaccard_mask(user_vector, item_id, jaccard_dict):
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


def find_cosine_mask(user_vector, item_id, cosine_dict):
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


def find_lime_mask(user_vector, item_id, recommender, lime, all_items_tensor, kw_dict):
    """Generate counterfactual explanation using LIME."""
    user_hist = user_vector.copy()
    user_hist[item_id] = 0
    user_hist_size = int(np.sum(user_hist))
    
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


def find_fia_mask(user_tensor, item_tensor, item_id, recommender, num_items, kw_dict, device):
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


def find_accent_mask(user_tensor, item_id, recommender, num_items, items_array, kw_dict, device, top_k=5):
    """
    Generate counterfactual explanation using ACCENT method.
    
    ACCENT analyzes how historical items differentially influence top-k recommendations.
    For each top-k recommended item, it computes which historical items influence it,
    then aggregates with weighted differences to highlight the most important items.
    """
    items_accent = defaultdict(float)
    factor = top_k - 1
    
    # Get top-k recommended items (not in user's history)
    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender, **kw_dict).keys())
    
    if top_k == 1:
        top_k_indices = [sorted_indices[0]]
    else:
        top_k_indices = sorted_indices[:top_k]
    
    for iteration, item_k_id in enumerate(top_k_indices):
        # Create item tensor for this top-k recommended item
        item_vector = items_array[item_k_id]
        item_tensor_temp = torch.FloatTensor(item_vector).to(device)
        
        # Compute influence of historical items on this recommendation
        # Use original user_tensor (unchanged history)
        fia_dict = find_fia_mask(user_tensor, item_tensor_temp, item_k_id, recommender, num_items, kw_dict, device)
        
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
    
    return dict(items_accent)


def find_accent_mask_refactored(user_tensor, item_id, recommender, num_items, items_array, kw_dict, device, top_k=5):
    """
    Generate counterfactual explanation using ACCENT method (corrected implementation).
    
    This implements ACCENT's core idea: compute gap influences between the top-1 
    recommendation and other top-k candidates. For each interaction z in user history:
        gap_influence(z) = sum over candidates i: [I(z, rec) - I(z, i)]
    
    where I(z, y) is the influence of interaction z on prediction y (computed via FIA).
    
    This produces an aggregate "importance mask" that explains why rec is ranked 
    higher than other top-k items. Interactions with high positive scores are 
    those that most strongly push rec ahead of alternatives.
    
    Args:
        user_tensor: Original user interaction history (unchanged throughout)
        item_id: Target item to explain (unused in current implementation)
        recommender: Recommender model
        num_items: Total number of items in catalog
        items_array: Identity matrix for items (num_items x num_items)
        kw_dict: Additional keyword arguments for recommender
        device: torch device
        top_k: Number of top recommendations to consider
        
    Returns:
        Dictionary mapping interaction indices to gap influence scores
    """
    items_accent = defaultdict(float)
    
    # Get top-k recommended items (not in user's history)
    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender, **kw_dict).keys())
    
    if top_k == 1:
        # With k=1, there's no gap to compute
        top_k_indices = [sorted_indices[0]]
        # Return empty dict or influence on single item
        item_vector = items_array[top_k_indices[0]]
        item_tensor = torch.FloatTensor(item_vector).to(device)
        return dict(find_fia_mask(user_tensor, item_tensor, top_k_indices[0], recommender, num_items, kw_dict, device))
    
    top_k_indices = sorted_indices[:top_k]
    
    # rec is the top-1 recommendation we want to explain
    rec = top_k_indices[0]
    
    # Compute influence of each interaction on rec (once)
    rec_item_vector = items_array[rec]
    rec_item_tensor = torch.FloatTensor(rec_item_vector).to(device)
    fia_rec = find_fia_mask(user_tensor, rec_item_tensor, rec, recommender, num_items, kw_dict, device)
    
    # For each candidate in top-k (excluding rec), compute gap influence
    for cand in top_k_indices[1:]:
        # Compute influence of each interaction on this candidate
        cand_item_vector = items_array[cand]
        cand_item_tensor = torch.FloatTensor(cand_item_vector).to(device)
        fia_cand = find_fia_mask(user_tensor, cand_item_tensor, cand, recommender, num_items, kw_dict, device)
        
        # Accumulate gap influence: I(z, rec) - I(z, cand)
        # Positive values mean interaction z pushes rec higher than cand
        for z in fia_rec.keys():
            gap_influence = fia_rec[z] - fia_cand.get(z, 0.0)
            items_accent[z] += gap_influence
    
    return dict(items_accent)


def find_shapley_mask(user_tensor, user_id, shap_values, item_to_cluster):
    """Generate counterfactual explanation using SHAP values."""
    item_shap = {}
    shapley_values = shap_values[shap_values[:, 0].astype(int) == user_id][:, 1:]
    user_vector = user_tensor.cpu().detach().numpy().astype(int)
    
    for i in np.where(user_vector.astype(int) == 1)[0]:
        items_cluster = item_to_cluster[i]
        item_shap[i] = shapley_values.T[int(items_cluster)][0]
    
    return item_shap


def find_lxr_mask(user_tensor, item_tensor, explainer):
    """Generate counterfactual explanation using LXR."""
    expl_scores = explainer(user_tensor, item_tensor)
    x_masked = user_tensor * expl_scores
    item_sim_dict = {}
    for i, j in enumerate(x_masked > 0):
        if j:
            item_sim_dict[i] = float(x_masked[i].cpu().detach().numpy())
    
    return item_sim_dict


def find_spinrec_mask(user_tensor, item_id, recommender, all_items_tensor, device, train_array, pop_array, num_steps=50, method='base'):
    """Generate counterfactual explanation using SPINRec (Integrated Gradients)."""
    item_sim_dict = find_ip_mask(
        model=recommender,
        user_tensor=user_tensor,
        item_id=item_id,
        all_items_tensor=all_items_tensor,
        num_steps=num_steps,
        method=method,
        device=device,
        recommender_name='VAE',
        train_array=train_array,
        pop_array=pop_array
    )
    # Convert tensor values to float
    return {k: float(v.cpu().detach().numpy()) for k, v in item_sim_dict.items()}


def get_counterfactual_explanation(
    user_tensor, user_vector, user_id, item_id, explainer, recommender, items_array, device,
    method='lxr', jaccard_dict=None, cosine_dict=None, shap_values=None, item_to_cluster=None,
    lime=None, all_items_tensor=None, kw_dict=None, num_items=None, train_array=None, pop_array=None
):
    """Generate counterfactual explanation for a user-item pair using the specified method."""
    item_tensor = torch.Tensor(items_array[item_id]).to(device)
    
    if method == 'lxr':
        sim_items = find_lxr_mask(user_tensor, item_tensor, explainer)
    elif method == 'jaccard':
        sim_items = find_jaccard_mask(user_vector, item_id, jaccard_dict)
    elif method == 'cosine':
        sim_items = find_cosine_mask(user_vector, item_id, cosine_dict)
    elif method == 'lime':
        sim_items = find_lime_mask(user_vector, item_id, recommender, lime, all_items_tensor, kw_dict)
    elif method == 'accent':
        sim_items = find_accent_mask(user_tensor, item_id, recommender, num_items, items_array, kw_dict, device)
        # Negate accent scores so that high = supports the recommendation,
        # consistent with all other methods.  find_accent_mask returns scores
        # where high = supports alternatives (due to its final negation step).
        sim_items = {k: -v for k, v in sim_items.items()}
    elif method == 'shap':
        sim_items = find_shapley_mask(user_tensor, user_id, shap_values, item_to_cluster)
    elif method == 'spinrec':
        sim_items = find_spinrec_mask(user_tensor, item_id, recommender, all_items_tensor, device, train_array, pop_array, method='sample_items_by_pop')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Sort by explanation score
    sorted_items = sorted(sim_items.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_items


def process_test_instances(
    test_array, test_data, num_items, recommender, explainer, items_array, movies_dict, device, output_file,
    method='lxr', jaccard_dict=None, cosine_dict=None, shap_values=None, item_to_cluster=None,
    lime=None, all_items_tensor=None, kw_dict=None, max_instances=300, train_array=None, pop_array=None
):
    """Loop through test instances and generate counterfactual explanations."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Processing {len(test_array)} test instances using {method.upper()} method...\n")
        f.write("=" * 80 + "\n\n")
        
        for i in range(min(max_instances, len(test_array))):
            user_vector = test_array[i][:num_items]
            user_tensor = torch.Tensor(user_vector).to(device)
            user_id = int(test_data.index[i])
            
            # Get recommended item
            user_res = recommender(user_tensor)[:num_items]
            user_catalog = torch.ones_like(user_tensor) - user_tensor
            user_recommendations = torch.mul(user_res.squeeze(), user_catalog)
            recommended_item = int(torch.argmax(user_recommendations))
            
            # Get counterfactual explanation
            cf_explanation = get_counterfactual_explanation(
                user_tensor, user_vector, user_id, recommended_item, explainer, recommender, items_array, device,
                method=method, jaccard_dict=jaccard_dict, cosine_dict=cosine_dict,
                shap_values=shap_values, item_to_cluster=item_to_cluster,
                lime=lime, all_items_tensor=all_items_tensor, kw_dict=kw_dict, num_items=num_items,
                train_array=train_array, pop_array=pop_array
            )
            
            # Get movie names
            recommended_movie = get_movie_name(recommended_item, movies_dict)
            cf_movies = [(get_movie_name(item_id, movies_dict), score) for item_id, score in cf_explanation[:5]]
            
            # Write results to file
            f.write(f"User {user_id}:\n")
            f.write(f"  Recommended movie: {recommended_movie}\n")
            f.write(f"  Top 5 counterfactual items:\n")
            for idx, (movie_name, score) in enumerate(cf_movies, 1):
                f.write(f"    {idx}. {movie_name} (score: {score:.4f})\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Done!\n")
    
    print(f"Results written to: {output_file}")


def validate_cf_explanations(
    test_array, test_data, num_items, recommender, explainer, items_array, movies_dict, device, output_file,
    method='lxr', jaccard_dict=None, cosine_dict=None, shap_values=None, item_to_cluster=None,
    lime=None, all_items_tensor=None, kw_dict=None, max_instances=300, max_removals=5,
    train_array=None, pop_array=None
):
    """
    Validate counterfactual explanations by iteratively removing items from user history.
    
    For each test instance:
    1. Get the original recommendation
    2. Get counterfactual explanation (ranked items to remove)
    3. Remove items one at a time in order of CF ranking
    4. Check if recommendation changes after each removal
    5. If recommendation changes within max_removals, record as successful
    """
    successful_cases = []
    
    print(f"Validating CF explanations using {method.upper()} for {min(max_instances, len(test_array))} test instances...")
    
    for i in range(min(max_instances, len(test_array))):
        user_vector = test_array[i][:num_items]
        user_tensor = torch.Tensor(user_vector).to(device)
        user_id = int(test_data.index[i])
        
        # Get original recommendation
        user_res = recommender(user_tensor)[:num_items]
        user_catalog = torch.ones_like(user_tensor) - user_tensor
        user_recommendations = torch.mul(user_res.squeeze(), user_catalog)
        original_recommendation = int(torch.argmax(user_recommendations))
        
        # Get counterfactual explanation
        cf_explanation = get_counterfactual_explanation(
            user_tensor, user_vector, user_id, original_recommendation, explainer, recommender, items_array, device,
            method=method, jaccard_dict=jaccard_dict, cosine_dict=cosine_dict,
            shap_values=shap_values, item_to_cluster=item_to_cluster,
            lime=lime, all_items_tensor=all_items_tensor, kw_dict=kw_dict, num_items=num_items,
            train_array=train_array, pop_array=pop_array
        )
        
        # Try removing items one at a time
        modified_tensor = user_tensor.clone()
        removed_items = []
        recommendation_changed = False
        changed_at_step = None
        new_recommendation = None
        
        for step in range(min(max_removals, len(cf_explanation))):
            # Get the item to remove (highest CF score)
            item_to_remove, cf_score = cf_explanation[step]
            
            # Remove the item from user history
            modified_tensor[item_to_remove] = 0
            removed_items.append((item_to_remove, cf_score))
            
            # Get new recommendation
            user_res_modified = recommender(modified_tensor)[:num_items]
            user_catalog_modified = torch.ones_like(modified_tensor) - modified_tensor
            user_recommendations_modified = torch.mul(user_res_modified.squeeze(), user_catalog_modified)
            new_recommendation = int(torch.argmax(user_recommendations_modified))
            
            # Check if recommendation changed
            if new_recommendation != original_recommendation:
                recommendation_changed = True
                changed_at_step = step + 1  # 1-indexed for readability
                break
        
        # If recommendation changed within max_removals, record it
        if recommendation_changed:
            successful_cases.append({
                'user_id': user_id,
                'original_recommendation': original_recommendation,
                'new_recommendation': new_recommendation,
                'removed_items': removed_items,
                'changed_at_step': changed_at_step
            })
    
    # Write successful cases to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"SUCCESSFUL COUNTERFACTUAL EXPLANATIONS ({method.upper()})\n")
        f.write(f"Found {len(successful_cases)} cases where recommendation changed within {max_removals} removals\n")
        f.write("=" * 80 + "\n\n")
        
        for case in successful_cases:
            user_id = case['user_id']
            original_rec = case['original_recommendation']
            new_rec = case['new_recommendation']
            removed_items = case['removed_items']
            changed_at_step = case['changed_at_step']
            
            f.write(f"User {user_id}:\n")
            f.write(f"  Original recommendation: {get_movie_name(original_rec, movies_dict)}\n")
            f.write(f"  New recommendation: {get_movie_name(new_rec, movies_dict)}\n")
            f.write(f"  Recommendation changed after removing {changed_at_step} item(s)\n")
            f.write(f"  Items removed (in order):\n")
            for idx, (item_id, score) in enumerate(removed_items, 1):
                f.write(f"    {idx}. {get_movie_name(item_id, movies_dict)} (CF score: {score:.4f})\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Total successful cases: {len(successful_cases)} out of {min(max_instances, len(test_array))}\n")
        f.write(f"Success rate: {len(successful_cases) / min(max_instances, len(test_array)) * 100:.2f}%\n")
    
    print(f"Validation results written to: {output_file}")
    print(f"Success rate: {len(successful_cases)}/{min(max_instances, len(test_array))} ({len(successful_cases) / min(max_instances, len(test_array)) * 100:.2f}%)")


def validate_cf_explanations_with_ranking(
    test_array, test_data, num_items, recommender, explainer, items_array, movies_dict, device, output_file,
    method='lxr', jaccard_dict=None, cosine_dict=None, shap_values=None, item_to_cluster=None,
    lime=None, all_items_tensor=None, kw_dict=None, max_instances=300, max_removals=5, target_rank=3,
    train_array=None, pop_array=None
):
    """
    Validate counterfactual explanations by checking if original top-1 recommendation drops out of top-k.
    
    For each test instance:
    1. Get the original top-1 recommendation
    2. Get counterfactual explanation (ranked items to remove)
    3. Remove items one at a time in order of CF ranking
    4. Check if original top-1 is now outside top-k after each removal
    5. If original top-1 drops out of top-k within max_removals, record as successful
    
    Args:
        target_rank: The ranking threshold (e.g., 3 means original top-1 should drop out of top-3)
    """
    successful_cases = []
    
    print(f"Validating CF explanations using {method.upper()} with ranking metric (target_rank={target_rank}) for {min(max_instances, len(test_array))} test instances...")
    
    for i in range(min(max_instances, len(test_array))):
        user_vector = test_array[i][:num_items]
        user_tensor = torch.Tensor(user_vector).to(device)
        user_id = int(test_data.index[i])
        
        # Get original recommendation and rankings
        user_res = recommender(user_tensor)[:num_items]
        user_catalog = torch.ones_like(user_tensor) - user_tensor
        user_recommendations = torch.mul(user_res.squeeze(), user_catalog)
        original_top1 = int(torch.argmax(user_recommendations))
        
        # Get counterfactual explanation
        cf_explanation = get_counterfactual_explanation(
            user_tensor, user_vector, user_id, original_top1, explainer, recommender, items_array, device,
            method=method, jaccard_dict=jaccard_dict, cosine_dict=cosine_dict,
            shap_values=shap_values, item_to_cluster=item_to_cluster,
            lime=lime, all_items_tensor=all_items_tensor, kw_dict=kw_dict, num_items=num_items,
            train_array=train_array, pop_array=pop_array
        )
        
        # Try removing items one at a time
        modified_tensor = user_tensor.clone()
        removed_items = []
        ranking_dropped = False
        changed_at_step = None
        new_rank_of_original = None
        top_k_after_removal = None
        
        for step in range(min(max_removals, len(cf_explanation))):
            # Get the item to remove (highest CF score)
            item_to_remove, cf_score = cf_explanation[step]
            
            # Remove the item from user history
            modified_tensor[item_to_remove] = 0
            removed_items.append((item_to_remove, cf_score))
            
            # Get new recommendations ranking
            user_res_modified = recommender(modified_tensor)[:num_items]
            user_catalog_modified = torch.ones_like(modified_tensor) - modified_tensor
            user_recommendations_modified = torch.mul(user_res_modified.squeeze(), user_catalog_modified)
            
            # Get top-k recommendations
            top_k_values, top_k_indices = torch.topk(user_recommendations_modified, target_rank)
            top_k_items = top_k_indices.cpu().numpy().tolist()
            
            # Check if original top-1 is now outside top-k
            if original_top1 not in top_k_items:
                ranking_dropped = True
                changed_at_step = step + 1  # 1-indexed for readability
                
                # Find the actual rank of the original top-1
                all_sorted_indices = torch.argsort(user_recommendations_modified, descending=True)
                new_rank_of_original = (all_sorted_indices == original_top1).nonzero(as_tuple=True)[0].item() + 1
                top_k_after_removal = top_k_items
                break
        
        # If ranking dropped within max_removals, record it
        if ranking_dropped:
            successful_cases.append({
                'user_id': user_id,
                'original_top1': original_top1,
                'removed_items': removed_items,
                'changed_at_step': changed_at_step,
                'new_rank': new_rank_of_original,
                'top_k_after_removal': top_k_after_removal
            })
    
    # Write successful cases to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"SUCCESSFUL COUNTERFACTUAL EXPLANATIONS ({method.upper()} - RANKING METRIC)\n")
        f.write(f"Target: Push original top-1 recommendation out of top-{target_rank}\n")
        f.write(f"Found {len(successful_cases)} cases where ranking dropped within {max_removals} removals\n")
        f.write("=" * 80 + "\n\n")
        
        for case in successful_cases:
            user_id = case['user_id']
            original_top1 = case['original_top1']
            removed_items = case['removed_items']
            changed_at_step = case['changed_at_step']
            new_rank = case['new_rank']
            top_k_after_removal = case['top_k_after_removal']
            
            f.write(f"User {user_id}:\n")
            f.write(f"  Original top-1 recommendation: {get_movie_name(original_top1, movies_dict)}\n")
            f.write(f"  New rank after removals: {new_rank}\n")
            f.write(f"  Dropped out of top-{target_rank} after removing {changed_at_step} item(s)\n")
            f.write(f"  New top-{target_rank} recommendations:\n")
            for idx, item_id in enumerate(top_k_after_removal, 1):
                f.write(f"    {idx}. {get_movie_name(item_id, movies_dict)}\n")
            f.write(f"  Items removed (in order):\n")
            for idx, (item_id, score) in enumerate(removed_items, 1):
                f.write(f"    {idx}. {get_movie_name(item_id, movies_dict)} (CF score: {score:.4f})\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Total successful cases: {len(successful_cases)} out of {min(max_instances, len(test_array))}\n")
        f.write(f"Success rate: {len(successful_cases) / min(max_instances, len(test_array)) * 100:.2f}%\n")
    
    print(f"Ranking validation results written to: {output_file}")
    print(f"Success rate: {len(successful_cases)}/{min(max_instances, len(test_array))} ({len(successful_cases) / min(max_instances, len(test_array)) * 100:.2f}%)")


def validate_cf_explanations_with_full_topk_change(
    test_array, test_data, num_items, recommender, explainer, items_array, movies_dict, device, output_file,
    method='lxr', jaccard_dict=None, cosine_dict=None, shap_values=None, item_to_cluster=None,
    lime=None, all_items_tensor=None, kw_dict=None, max_instances=300, max_removals=5, target_rank=3,
    train_array=None, pop_array=None
):
    """
    Validate counterfactual explanations by checking if ALL original top-k items drop out of top-k.
    
    For each test instance:
    1. Get the original top-k recommendations
    2. Get counterfactual explanation for the top-1 (ranked items to remove)
    3. Remove items one at a time in order of CF ranking
    4. Check if ALL original top-k items are now outside the new top-k after each removal
    5. If all original top-k drop out within max_removals, record as successful
    
    Args:
        target_rank: The k value for top-k (e.g., 3 means checking top-3)
    """
    successful_cases = []
    
    print(f"Validating CF explanations using {method.upper()} with full top-{target_rank} change for {min(max_instances, len(test_array))} test instances...")
    
    for i in range(min(max_instances, len(test_array))):
        user_vector = test_array[i][:num_items]
        user_tensor = torch.Tensor(user_vector).to(device)
        user_id = int(test_data.index[i])
        
        # Get original top-k recommendations
        user_res = recommender(user_tensor)[:num_items]
        user_catalog = torch.ones_like(user_tensor) - user_tensor
        user_recommendations = torch.mul(user_res.squeeze(), user_catalog)
        
        # Get original top-k items
        original_top_k_values, original_top_k_indices = torch.topk(user_recommendations, target_rank)
        original_top_k_items = set(original_top_k_indices.cpu().numpy().tolist())
        original_top1 = int(original_top_k_indices[0])
        
        # Get counterfactual explanation for top-1
        cf_explanation = get_counterfactual_explanation(
            user_tensor, user_vector, user_id, original_top1, explainer, recommender, items_array, device,
            method=method, jaccard_dict=jaccard_dict, cosine_dict=cosine_dict,
            shap_values=shap_values, item_to_cluster=item_to_cluster,
            lime=lime, all_items_tensor=all_items_tensor, kw_dict=kw_dict, num_items=num_items,
            train_array=train_array, pop_array=pop_array
        )
        
        # Try removing items one at a time
        modified_tensor = user_tensor.clone()
        removed_items = []
        all_dropped = False
        changed_at_step = None
        new_top_k_after_removal = None
        remaining_original_items = None
        
        for step in range(min(max_removals, len(cf_explanation))):
            # Get the item to remove (highest CF score)
            item_to_remove, cf_score = cf_explanation[step]
            
            # Remove the item from user history
            modified_tensor[item_to_remove] = 0
            removed_items.append((item_to_remove, cf_score))
            
            # Get new recommendations ranking
            user_res_modified = recommender(modified_tensor)[:num_items]
            user_catalog_modified = torch.ones_like(modified_tensor) - modified_tensor
            user_recommendations_modified = torch.mul(user_res_modified.squeeze(), user_catalog_modified)
            
            # Get new top-k recommendations
            new_top_k_values, new_top_k_indices = torch.topk(user_recommendations_modified, target_rank)
            new_top_k_items = set(new_top_k_indices.cpu().numpy().tolist())
            
            # Check if ALL original top-k items are now outside new top-k
            remaining_in_topk = original_top_k_items.intersection(new_top_k_items)
            
            if len(remaining_in_topk) == 0:
                # All original top-k items have dropped out
                all_dropped = True
                changed_at_step = step + 1  # 1-indexed for readability
                new_top_k_after_removal = list(new_top_k_items)
                remaining_original_items = remaining_in_topk
                break
        
        # If all original top-k dropped within max_removals, record it
        if all_dropped:
            successful_cases.append({
                'user_id': user_id,
                'original_top_k': list(original_top_k_items),
                'new_top_k': new_top_k_after_removal,
                'removed_items': removed_items,
                'changed_at_step': changed_at_step
            })
    
    # Write successful cases to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"SUCCESSFUL COUNTERFACTUAL EXPLANATIONS ({method.upper()} - FULL TOP-{target_rank} CHANGE)\n")
        f.write(f"Target: ALL original top-{target_rank} items must drop out of new top-{target_rank}\n")
        f.write(f"Found {len(successful_cases)} cases where all top-{target_rank} changed within {max_removals} removals\n")
        f.write("=" * 80 + "\n\n")
        
        for case in successful_cases:
            user_id = case['user_id']
            original_top_k = case['original_top_k']
            new_top_k = case['new_top_k']
            removed_items = case['removed_items']
            changed_at_step = case['changed_at_step']
            
            f.write(f"User {user_id}:\n")
            f.write(f"  Original top-{target_rank} recommendations:\n")
            for idx, item_id in enumerate(original_top_k, 1):
                f.write(f"    {idx}. {get_movie_name(item_id, movies_dict)}\n")
            f.write(f"  New top-{target_rank} recommendations (ALL different):\n")
            for idx, item_id in enumerate(new_top_k, 1):
                f.write(f"    {idx}. {get_movie_name(item_id, movies_dict)}\n")
            f.write(f"  Complete change achieved after removing {changed_at_step} item(s)\n")
            f.write(f"  Items removed (in order):\n")
            for idx, (item_id, score) in enumerate(removed_items, 1):
                f.write(f"    {idx}. {get_movie_name(item_id, movies_dict)} (CF score: {score:.4f})\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Total successful cases: {len(successful_cases)} out of {min(max_instances, len(test_array))}\n")
        f.write(f"Success rate: {len(successful_cases) / min(max_instances, len(test_array)) * 100:.2f}%\n")
    
    print(f"Full top-{target_rank} change validation results written to: {output_file}")
    print(f"Success rate: {len(successful_cases)}/{min(max_instances, len(test_array))} ({len(successful_cases) / min(max_instances, len(test_array)) * 100:.2f}%)")


def limit_cf_item_interactions(
    cf_items: list[tuple[int, float]],
    user_tensor: torch.Tensor,
    target_movie_id: int,
    recommender: object,
    num_items: int,
    device: torch.device,
    max_removals: int,
    target_rank: int,
) -> tuple[list[tuple[int, float]], bool]:
    """Limit CF items to the minimal set that changes the recommendation ranking."""
    # Try removing items one at a time
    modified_tensor = user_tensor.clone()
    removed_items = []
    ranking_dropped = False

    for step in range(min(max_removals, len(cf_items))):
        # Get the item to remove (highest CF score)
        item_to_remove, cf_score = cf_items[step]

        # Remove the item from user history
        modified_tensor[item_to_remove] = 0
        removed_items.append((item_to_remove, cf_score))

        # Get new recommendations ranking
        user_res_modified = recommender(modified_tensor)[:num_items]
        user_catalog_modified = torch.ones_like(modified_tensor) - modified_tensor
        user_recommendations_modified = torch.mul(user_res_modified.squeeze(), user_catalog_modified)

        # Get top-k recommendations
        top_k_values, top_k_indices = torch.topk(user_recommendations_modified, target_rank)
        top_k_items = top_k_indices.cpu().numpy().tolist()

        # Check if original top-1 is now outside top-k
        if target_movie_id not in top_k_items:
            ranking_dropped = True
            break

    return removed_items, ranking_dropped


def main():
    """Main function to run the counterfactual explanation analysis."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_name = "ML1M"
    num_items = 3381
    num_users = 6037
    
    # Setup paths
    base_path, data_path, checkpoint_path = setup_paths(data_name)
    
    # Load data and movies
    train_data, test_data, static_test_data, pop_dict, movies_df, movies_dict, jaccard_dict, cosine_dict, item_to_cluster, shap_values = load_data_and_movies(
        data_path, data_name, num_items
    )
    
    # Prepare arrays
    train_array, test_array, items_array, all_items_tensor, pop_array = prepare_arrays(
        train_data, test_data, static_test_data, pop_dict, num_items, device
    )
    
    # Load models
    recommender, explainer, lime, kw_dict = load_models(
        checkpoint_path, num_items, pop_array, all_items_tensor, items_array, static_test_data, device
    )
    
    # Define which methods to compare
    methods = ['lxr', 'jaccard', 'cosine', 'lime', 'accent', 'shap', 'spinrec']
    
    # Run analysis for each method
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Running analysis for method: {method.upper()}")
        print(f"{'='*80}\n")
        
        # Output file paths for this method
        cf_results_dir = base_path / "CF Results"
        cf_results_dir.mkdir(exist_ok=True)
        
        output_file = cf_results_dir / f"counterfactual_explanations_{method}.txt"
        validation_output_file = cf_results_dir / f"good_cf_explanations_{method}.txt"
        ranking_validation_output_file = cf_results_dir / f"good_cf_explanations_ranking_{method}.txt"
        full_topk_change_output_file = cf_results_dir / f"good_cf_explanations_full_topk_change_{method}.txt"
        
        # Process test instances and generate explanations
        process_test_instances(
            test_array, test_data, num_items, recommender, explainer, items_array, movies_dict, device, output_file,
            method=method, jaccard_dict=jaccard_dict, cosine_dict=cosine_dict,
            shap_values=shap_values, item_to_cluster=item_to_cluster,
            lime=lime, all_items_tensor=all_items_tensor, kw_dict=kw_dict, max_instances=300,
            train_array=train_array, pop_array=pop_array
        )
        
        # Validate counterfactual explanations (changes top-1)
        validate_cf_explanations(
            test_array, test_data, num_items, recommender, explainer, items_array, movies_dict, device, validation_output_file,
            method=method, jaccard_dict=jaccard_dict, cosine_dict=cosine_dict,
            shap_values=shap_values, item_to_cluster=item_to_cluster,
            lime=lime, all_items_tensor=all_items_tensor, kw_dict=kw_dict, max_instances=300, max_removals=5,
            train_array=train_array, pop_array=pop_array
        )
        
        # Validate counterfactual explanations with ranking metric (push out of top-k)
        validate_cf_explanations_with_ranking(
            test_array, test_data, num_items, recommender, explainer, items_array, movies_dict, device, ranking_validation_output_file,
            method=method, jaccard_dict=jaccard_dict, cosine_dict=cosine_dict,
            shap_values=shap_values, item_to_cluster=item_to_cluster,
            lime=lime, all_items_tensor=all_items_tensor, kw_dict=kw_dict, max_instances=300, max_removals=5, target_rank=3,
            train_array=train_array, pop_array=pop_array
        )
        
        # Validate counterfactual explanations with full top-k change (all original top-k items drop out)
        validate_cf_explanations_with_full_topk_change(
            test_array, test_data, num_items, recommender, explainer, items_array, movies_dict, device, full_topk_change_output_file,
            method=method, jaccard_dict=jaccard_dict, cosine_dict=cosine_dict,
            shap_values=shap_values, item_to_cluster=item_to_cluster,
            lime=lime, all_items_tensor=all_items_tensor, kw_dict=kw_dict, max_instances=300, max_removals=3, target_rank=3,
            train_array=train_array, pop_array=pop_array
        )
    
    print(f"\n{'='*80}")
    print("Analysis complete for all methods!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
