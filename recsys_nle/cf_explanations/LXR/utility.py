"""Utility functions for counterfactual explanation analysis."""

import pandas as pd


def get_movie_name(item_id, movies_dict):
    """Get movie name from item ID."""
    return movies_dict.get(item_id, f"Unknown (ID: {item_id})")


def analyze_missing_movies(movies_df, movies_dict, num_items):
    """Analyze which movie indices are missing and why."""
    print("\n" + "=" * 80)
    print("MOVIE DATA ANALYSIS")
    print("=" * 80)
    
    # Get all expected item IDs (0-indexed)
    expected_ids = set(range(num_items))
    
    # Get actual IDs in movies_dict
    actual_ids = set(movies_dict.keys())
    
    # Find missing IDs
    missing_ids = expected_ids - actual_ids
    
    # Statistics
    print(f"\nTotal items in model: {num_items}")
    print(f"Movies loaded from CSV: {len(movies_df)}")
    print(f"Movies in dictionary: {len(actual_ids)}")
    print(f"Missing movie entries: {len(missing_ids)}")
    
    if missing_ids:
        print(f"\nMissing item IDs (showing first 20): {sorted(missing_ids)[:20]}")
        if len(missing_ids) > 20:
            print(f"... and {len(missing_ids) - 20} more")
    
    # Check for gaps in MovieID sequence in CSV
    movie_ids_in_csv = sorted(movies_df['MovieID'].values)
    print(f"\nMovieID range in CSV: {movie_ids_in_csv[0]} to {movie_ids_in_csv[-1]}")
    print(f"Expected range (0-indexed): 0 to {num_items - 1}")
    print(f"Expected range (1-indexed): 1 to {num_items}")
    
    # Check for duplicate MovieIDs
    duplicates = movies_df[movies_df.duplicated(subset=['MovieID'], keep=False)]
    if not duplicates.empty:
        print(f"\nWarning: Found {len(duplicates)} duplicate MovieIDs")
        print(duplicates[['MovieID', 'MovieName']].head(10))
    
    # Check maximum MovieID
    max_movie_id = movies_df['MovieID'].max()
    print(f"\nMaximum MovieID in CSV: {max_movie_id}")
    print(f"After 0-indexing: {max_movie_id - 1}")
    
    # Show sample of loaded movies
    print("\nSample of loaded movies (first 10):")
    for item_id in range(min(10, num_items)):
        if item_id in movies_dict:
            print(f"  ID {item_id}: {movies_dict[item_id]}")
        else:
            print(f"  ID {item_id}: MISSING")
    
    # Check if any CSV lines were skipped due to parsing errors
    print("\n" + "=" * 80)
    print()


def get_counterfactual_explanation(user_tensor, item_id, explainer, recommender, items_array, device):
    """Generate counterfactual explanation for a user-item pair."""
    import torch
    
    item_tensor = torch.Tensor(items_array[item_id]).to(device)
    expl_scores = explainer(user_tensor, item_tensor)
    x_masked = user_tensor * expl_scores
    
    item_sim_dict = {i: x_masked[i].item() for i in range(len(x_masked)) if user_tensor[i] > 0}
    sorted_items = sorted(item_sim_dict.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_items
