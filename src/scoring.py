import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# Import the logger from our existing module
from .data_loader import log_message


"""Helper and Utility Methods"""
def _scale_features(features_df: pd.DataFrame) -> np.ndarray:
    """Scales the feature data to have zero mean and unit variance."""
    log_message("Scaling features for clustering model...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    return scaled_features


def _get_wallet_clusters(scaled_features: np.ndarray, features_df: pd.DataFrame) -> pd.Series:
    """Applies the HDBSCAN algorithm to group wallets into behavioral clusters."""
    log_message("Running HDBSCAN to identify wallet clusters...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        gen_min_span_tree=True,
        metric='euclidean'
    )
    clusterer.fit(scaled_features)
    log_message(f"Identified {clusterer.labels_.max() + 1} distinct clusters and {np.sum(clusterer.labels_ == -1)} outliers.")
    return pd.Series(clusterer.labels_, index=features_df.index, name="cluster_label")


def _calculate_heuristic_score(features_df: pd.DataFrame) -> pd.Series:
    """
    Calculates a raw credit score based on a re-balanced heuristic model.
    """
    log_message("Calculating scores using re-balanced heuristic model...")
    base_score = 500
    scores = pd.Series(base_score, index=features_df.index)

    # --- PENALTIES FOR RISKY BEHAVIOR ---
    # Increased penalty for liquidations. This should be the biggest single factor.
    scores -= features_df['liquidation_count'] * 400

    # It targets wallets whose minimum health factor dropped below a risky threshold (e.g., 1.5)
    risky_hf_wallets = features_df['min_health_factor_proxy'] < 1.5
    scores[risky_hf_wallets] -= (1.5 - features_df[risky_hf_wallets]['min_health_factor_proxy']) * 100

    # Add a penalty for high leverage (borrowing a lot relative to collateral)
    scores -= (features_df['borrow_to_deposit_ratio'] * 50).clip(upper=150)
    
    # Small penalty for very high frequency (potential bot behavior)
    scores -= (features_df['transaction_frequency'].clip(lower=10) - 10) * 5

    # --- REWARDS FOR RESPONSIBLE BEHAVIOR ---
    scores += (features_df['wallet_age_days'] * 0.3).clip(upper=75)
    
    # Reward for a high average health factor, but only for those who manage debt.
    # We ignore the artificially high HF of non-borrowers.
    borrowers_mask = features_df['total_borrow_usd'] > 0
    scores[borrowers_mask] += (features_df[borrowers_mask]['mean_health_factor_proxy'].clip(upper=10) * 10)

    # Reward for repaying a high fraction of borrowed amounts
    scores += (features_df['repay_to_borrow_ratio'] * 50).clip(upper=50)
    
    # Reward for being a net depositor (providing liquidity)
    scores += (np.log1p(features_df['net_deposit_usd'].clip(lower=0)) * 5).clip(upper=75)
    
    scores += (features_df['unique_active_days'] * 1.0).clip(upper=50)
    
    return scores


def _normalize_scores(scores: pd.Series) -> pd.Series:
    """
    Normalizes scores to a 0-1000 scale using quantile-based ranking
    with an intra-bin adjustment for tie-breaking and granularity.

    :param scores: The Series of raw, calculated scores.
    :return: A pandas Series of final scores, scaled from 0 to 1000.
    """
    log_message("Normalizing scores with intra-bin ranking for max granularity...")

    # Step 1: Assign each wallet to a percentile bin (0-99)
    try:
        # We aim for 100 bins, each representing 10 points on the final scale
        quantiles = pd.qcut(scores, 100, labels=False, duplicates='drop')
    except ValueError:
        # Fallback for data that can't be split into 100 bins
        quantiles = pd.qcut(scores.rank(method='first'), 100, labels=False, duplicates='drop')

    # Step 2: Calculate the base score from the bin (e.g., bin 83 -> 830)
    base_score = quantiles * 10
    
    # Step 3: Calculate a fine-grained "bonus" score within each bin
    # We group the raw scores by their quantile and rank them from 0 to 1 within that group
    grouped = scores.groupby(quantiles)
    # rank(pct=True) gives a rank from 0.0 to 1.0 within each group
    intra_bin_rank = grouped.rank(pct=True)
    
    # The bonus will be from 0 to 9, adding granularity inside the 10-point bin
    bonus_score = (intra_bin_rank * 9).fillna(0)
    
    # Step 4: Combine base and bonus for the final score
    final_score = base_score + bonus_score
    
    return final_score.astype(int)


"""Main Method"""
def generate_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the entire scoring pipeline.
    """
    log_message("Starting scoring pipeline...")
    
    scaled_features = _scale_features(features_df)
    cluster_labels = _get_wallet_clusters(scaled_features, features_df)
    raw_scores = _calculate_heuristic_score(features_df)
    
    final_scores = _normalize_scores(raw_scores)
    
    log_message("Assembling final results...")
    results_df = pd.DataFrame({
        'credit_score': final_scores,
        'cluster_label': cluster_labels
    }, index=features_df.index)

    log_message("Scoring pipeline complete.", level="SUCCESS")
    
    return results_df.sort_values(by='credit_score', ascending=False)