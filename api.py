import pandas as pd
import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Dict

# Ignore All Warnings 
import warnings 
warnings.filterwarnings("ignore")

# Import our custom modules from the src package
from src import data_loader, feature_engineering, scoring

"""API Configuration and Setup"""
# Define constants for file paths
DATA_FILE_PATH = "data/raw/user-wallet-transactions.json"
OUTPUT_DIR = "output"
SCORES_FILE_PATH = os.path.join(OUTPUT_DIR, "wallet_scores.csv")

# Create the FastAPI app instance
app = FastAPI(
    title="Aave Wallet Credit Scoring API",
    description="An API to generate and retrieve credit scores for Aave V2 wallets based on historical transaction data.",
    version="1.0.0",
)


"""Pydantic Models""" 
class ScoreResponse(BaseModel):
    """Defines the structure for a single score response."""
    userWallet: str = Field(..., example="0x003a85c562730b196f7cba202a2515f2ff855736")
    credit_score: int = Field(..., example=839)
    cluster_label: int = Field(..., example=-1)
    
class TriggerResponse(BaseModel):
    """Defines the structure for the scoring trigger response."""
    message: str
    output_file: str


"""Helpers"""
def run_scoring_pipeline():
    """
    A wrapper function that runs the entire pipeline from start to finish.
    This is what the background task will execute.
    """
    # Disable verbose logging for API runs to keep the console clean
    data_loader.set_logging(False)
    
    print("--- Background Task: Started Scoring Pipeline ---")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Transaction Data
    raw_df = data_loader.load_transactions(DATA_FILE_PATH)

    # Step 2: Engineer Features
    features_df = feature_engineering.engineer_features(raw_df)

    # Step 3: Generate Scores
    final_scores_df = scoring.generate_scores(features_df)
    
    # Step 4: Save Output
    final_scores_df.to_csv(SCORES_FILE_PATH)
    
    print(f"--- Background Task: Pipeline Complete. Scores saved to {SCORES_FILE_PATH} ---")



"""API Endpoints"""
@app.get("/")
def read_root() -> Dict[str, str]:
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Aave Credit Scoring API. Visit /docs for details."}


@app.post("/score/generate", response_model=TriggerResponse, status_code=202)
def generate_all_scores(background_tasks: BackgroundTasks) -> TriggerResponse:
    """
    Triggers the full credit scoring pipeline as a background task.
    
    This will process the `user_transactions.json` file and generate the
    `wallet_scores.csv` output file without blocking the API.
    """
    background_tasks.add_task(run_scoring_pipeline)
    return {
        "message": "Credit scoring pipeline has been triggered in the background.",
        "output_file": SCORES_FILE_PATH
    }


@app.get("/score/{wallet_address}", response_model=ScoreResponse)
def get_wallet_score(wallet_address: str) -> ScoreResponse:
    """
    Retrieves the pre-computed credit score for a specific wallet address.

    Note: The scoring pipeline must be run at least once via the
    `/score/generate` endpoint before using this endpoint.
    """
    # Normalize the input address to lowercase to ensure match
    wallet_address = wallet_address.lower()

    # Check if the scores file exists
    if not os.path.exists(SCORES_FILE_PATH):
        raise HTTPException(
            status_code=404, 
            detail="Scores file not found. Please trigger the scoring pipeline first via a POST to /score/generate."
        )
        
    # Read the CSV and look for the wallet
    scores_df = pd.read_csv(SCORES_FILE_PATH, index_col='userWallet')
    
    # Normalize the index to lowercase
    scores_df.index = scores_df.index.str.lower()
    
    if wallet_address not in scores_df.index:
        raise HTTPException(
            status_code=404, 
            detail=f"Wallet address '{wallet_address}' not found in the scoring results."
        )
    
    # Retrieve the score data for the wallet
    wallet_data = scores_df.loc[wallet_address]
    
    return ScoreResponse(
        userWallet=wallet_address,
        credit_score=int(wallet_data['credit_score']),
        cluster_label=int(wallet_data['cluster_label'])
    )