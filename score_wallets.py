import pandas as pd
from typing import NoReturn
import os

# Import our custom modules
from src import data_loader
from src import feature_engineering
from src import scoring

# Ignore all warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")


# Define constants for file paths and settings
DATA_FILE_PATH = "data/raw/user-wallet-transactions.json"
OUTPUT_DIR = "output"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "wallet_scores.csv")
ENABLE_LOGGING = True

def main() -> NoReturn:
    """
    Main function to run the end-to-end credit scoring pipeline.
    
    Pipeline Steps:
    1. Loads transaction data from the JSON file.
    2. Engineers a feature matrix for each wallet.
    3. Generates a credit score based on features and clustering.
    4. Saves the final scores to a CSV file.
    """
    # --- Step 0: Configure Environment ---
    data_loader.set_logging(ENABLE_LOGGING)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # --- Step 1: Load Transaction Data ---
    raw_df = data_loader.load_transactions(DATA_FILE_PATH)

    # --- Step 2: Engineer Features ---
    features_df = feature_engineering.engineer_features(raw_df)
    features_df.to_csv("data/processed/feature_matrix.csv", index=False)

    # --- Step 3: Generate Scores ---
    final_scores_df = scoring.generate_scores(features_df)
    
    # --- Step 4: Save and Display Results ---
    data_loader.log_message(f"Saving final scores to {OUTPUT_FILE_PATH}", level="SUCCESS")
    final_scores_df.to_csv(OUTPUT_FILE_PATH)

    print("\n\n" + "="*60)
    print("                Credit Scoring Pipeline Complete")
    print("="*60 + "\n")

    print("--- Top 10 Wallets by Credit Score ---\n")
    print(final_scores_df.head(10))

    print("\n\n--- Bottom 10 Wallets by Credit Score ---\n")
    print(final_scores_df.tail(10))

    print("\n\n--- Score Distribution Summary ---\n")
    print(final_scores_df['credit_score'].describe())


if __name__ == "__main__":
    main()