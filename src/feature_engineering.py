import pandas as pd
import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm

from .data_loader import log_message


"""Helper and Utility Methods"""
def _calculate_history_features(df: pd.DataFrame, grouped_data: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Calculates wallet history and stability features.

    :param df: The raw transaction DataFrame.
    :param grouped_data: The DataFrame grouped by userWallet.
    :return: A DataFrame with history features per wallet.
    """
    log_message("Calculating: Wallet History & Stability Features...")
    
    # Time between first and last transaction
    min_ts = grouped_data['timestamp'].min()
    max_ts = grouped_data['timestamp'].max()
    # Add 1 second to avoid zero-day ages for wallets with one transaction
    wallet_age_seconds = (max_ts - min_ts).dt.total_seconds() + 1
    
    features = pd.DataFrame({
        'wallet_age_days': wallet_age_seconds / (60 * 60 * 24),
        'total_transactions': grouped_data.size(),
        'unique_active_days': grouped_data['timestamp'].apply(lambda x: x.dt.date.nunique())
    })
    
    features['transaction_frequency'] = features['total_transactions'] / features['unique_active_days']
    return features


def _calculate_financial_features(grouped_data: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Calculates financial and scale-related features.

    :param grouped_data: The DataFrame grouped by userWallet.
    :return: A DataFrame with financial features per wallet.
    """
    log_message("Calculating: Financial Health & Scale Features...")

    # Calculate total USD value for each key action
    financials = {
        action: grouped_data.apply(lambda x: x[x['action'] == action]['amountUSD'].sum())
        for action in ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']
    }
    
    features = pd.DataFrame({
        'total_deposit_usd': financials['deposit'],
        'total_borrow_usd': financials['borrow'],
        'total_repay_usd': financials['repay'],
        'total_redeem_usd': financials['redeemunderlying'],
        'total_liquidation_usd': financials['liquidationcall'],
        'average_transaction_value_usd': grouped_data['amountUSD'].mean()
    })

    # Net liquidity provided to the protocol
    features['net_deposit_usd'] = features['total_deposit_usd'] - features['total_redeem_usd']
    return features


def _calculate_risk_features(df: pd.DataFrame, grouped_data: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Calculates risk, responsibility, and liquidation-related features.

    :param df: The raw transaction DataFrame.
    :param grouped_data: The DataFrame grouped by userWallet.
    :return: A DataFrame with risk features per wallet.
    """
    log_message("Calculating: Risk & Responsibility Features...")

    # Count of liquidations is a critical risk indicator
    liquidation_count = grouped_data.apply(lambda x: (x['action'] == 'liquidationcall').sum())
    
    # --- Health Factor Proxy Calculation ---
    # We define collateral changes and debt changes.
    collateral_change = df.apply(lambda row: row['amountUSD'] if row['action'] in ['deposit', 'repay'] else -row['amountUSD'] if row['action'] in ['redeemunderlying', 'borrow'] else 0, axis=1)
    debt_change = df.apply(lambda row: row['amountUSD'] if row['action'] == 'borrow' else -row['amountUSD'] if row['action'] == 'repay' else 0, axis=1)

    temp_df = df[['userWallet', 'timestamp']].copy()
    temp_df['collateral_change'] = collateral_change
    temp_df['debt_change'] = debt_change
    
    # Sort by time to calculate cumulative state correctly
    temp_df = temp_df.sort_values(by=['userWallet', 'timestamp'])
    
    # Calculate cumulative collateral and debt for each wallet over time
    temp_df['collateral_balance'] = temp_df.groupby('userWallet')['collateral_change'].cumsum()
    temp_df['debt_balance'] = temp_df.groupby('userWallet')['debt_change'].cumsum()
    
    # Calculate the health factor proxy at each point in time. Avoid division by zero.
    # Add a small epsilon to the denominator.
    temp_df['health_factor_proxy'] = temp_df['collateral_balance'] / (temp_df['debt_balance'] + 1e-6)
    
    # We are interested in the *minimum* health factor a user ever had, and their average.
    # Replace infinite values (from division by zero) with a large number, as it implies no debt.
    temp_df.replace([np.inf, -np.inf], 1000, inplace=True) 
    
    risk_proxies = temp_df.groupby('userWallet')['health_factor_proxy'].agg(['min', 'mean'])
    risk_proxies.rename(columns={'min': 'min_health_factor_proxy', 'mean': 'mean_health_factor_proxy'}, inplace=True)

    features = pd.DataFrame({
        'liquidation_count': liquidation_count
    })
    
    features = features.join(risk_proxies)
    return features


def _calculate_behavioral_features(financial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates behavioral pattern features from financial totals.

    :param financial_df: DataFrame containing the calculated financial totals.
    :return: A DataFrame with behavioral ratio features.
    """
    log_message("Calculating: Behavioral Pattern Features...")
    
    features = pd.DataFrame(index=financial_df.index)
    
    # Ratio of how much a user repays vs. borrows. Add epsilon for stability.
    features['repay_to_borrow_ratio'] = financial_df['total_repay_usd'] / (financial_df['total_borrow_usd'] + 1e-6)
    
    # Ratio of how much a user borrows against their deposits.
    features['borrow_to_deposit_ratio'] = financial_df['total_borrow_usd'] / (financial_df['total_deposit_usd'] + 1e-6)

    return features


"""Main Method""" 
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the entire feature engineering pipeline.

    Takes the raw transaction data and produces a feature matrix where
    each row represents a unique user wallet and each column is a
    feature describing their behavior.

    :param df: The cleaned DataFrame from the data_loader.
    :return: A feature matrix DataFrame ready for model training.
    """
    log_message("Starting feature engineering pipeline...")
    
    # Ensure data is sorted by time for cumulative calculations
    df = df.sort_values(by='timestamp')

    # Create the grouped object for efficient calculations
    grouped_by_wallet = df.groupby('userWallet')

    # Run each feature calculation module
    tqdm.pandas(desc="Engineering Features")
    
    history_features = _calculate_history_features(df, grouped_by_wallet)
    financial_features = _calculate_financial_features(grouped_by_wallet)
    risk_features = _calculate_risk_features(df, grouped_by_wallet)
    behavioral_features = _calculate_behavioral_features(financial_features)
    
    # Combine all features into a single DataFrame
    log_message("Combining all feature sets...")
    features_df = history_features
    features_df = features_df.join(financial_features)
    features_df = features_df.join(risk_features)
    features_df = features_df.join(behavioral_features)
    
    # Post-processing and Cleanup
    log_message("Performing final cleanup...")
    
    # Fill NaN values. Wallets that never performed an action (e.g., borrow) will have NaNs.
    # Filling with 0 is a reasonable default for counts and totals.
    features_df.fillna(0, inplace=True)
    
    # Clean up any infinite values that may have slipped through.
    features_df.replace([np.inf, -np.inf], 0, inplace=True)

    log_message(f"Feature engineering complete. Created {features_df.shape[1]} features for {features_df.shape[0]} wallets.", level="SUCCESS")
    
    return features_df