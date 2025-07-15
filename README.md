# Aave Wallet Credit Scoring Model

### Project Goal
This project delivers a robust machine learning system to address the challenge of assigning a credit score between **0 and 1000** to Aave V2 wallets. Based on a sample of 100,000 raw transactions, the model analyzes historical on-chain behavior to identify reliable and responsible usage (higher scores) versus risky, bot-like, or previously liquidated behavior (lower scores).

### Key Features
*   **Robust Scoring Logic:** Implements a hybrid model using heuristic rules and unsupervised clustering (HDBSCAN) to ensure scores are both explainable and data-driven.
*   **Effective Distribution:** Utilizes quantile normalization to guarantee scores are meaningfully distributed across the entire 0-1000 range.
*   **One-Step Script:** Includes a primary executable, `score_wallets.py`, to generate all wallet scores from the raw JSON file with a single command.
*   **FastAPI Integration:** Exposes the model's functionality through a simple, fast, and well-documented API.
*   **In-depth Analysis:** Provides a detailed `analysis.md` file with score distributions and behavioral analysis of high and low-scoring wallets.

---

## Methodology & Architecture

### Methodology
The core of this project is a **hybrid unsupervised learning approach**, chosen because no pre-labeled data for "good" or "bad" wallets exists.

1.  **Feature Engineering:** Raw transaction logs are aggregated for each unique wallet to create a rich feature set describing its stability, financial health, risk profile, and on-chain behavior. Key features include `liquidation_count`, `min_health_factor_proxy`, `wallet_age_days`, and `repay_to_borrow_ratio`.
2.  **Heuristic Scoring:** A weighted model applies penalties for risky actions (e.g., liquidations, high leverage) and rewards for responsible actions (e.g., long-term liquidity provision, high health factor).
3.  **Clustering & Normalization:** Wallets are clustered using **HDBSCAN** to identify natural behavioral groups. The raw heuristic scores are then normalized using **quantile ranking** to ensure a granular and well-distributed final score from 0-1000.

### Project Architecture
The codebase is organized into modular, single-responsibility components for maintainability and clarity.

```
aave-credit-scoring/
│
├── README.md              # This file: Project overview and instructions
├── analysis.md            # In-depth analysis of scoring results
├── requirements.txt       # All Python dependencies
│
├── score_wallets.py       # Use Case 1: Main executable script for batch scoring
├── api.py                 # Use Case 2: FastAPI server to serve scores
│
├── data/
│   └── raw/user_transactions.json
│
├── output/
│   └── wallet_scores.csv  # Final generated scores
│
└── src/
    ├── data_loader.py         # Handles data loading and validation.
    ├── feature_engineering.py # Creates the feature matrix from raw data.
    └── scoring.py             # Applies the scoring model and logic.
```

---

## How to Use This Project

### 1. Initial Setup
First, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/ananya868/aave-credit-scoring.git
cd aave-credit-scoring

# Install all required packages
pip install -r requirements.txt
```
*Note: Ensure the `user-wallet-transactions.json` file is placed in the `data/raw/` directory.*

### 2. Use Case 1: Generate Scores via Command-Line Script
This is the primary deliverable for generating the `wallet_scores.csv` file.

1.  **Run the script:**
    ```bash
    python score_wallets.py
    ```
   
2.  **Find the output:** The script will process all transactions and save the final results in `output/wallet_scores.csv`.

### 3. Use Case 2: Interact with the Scoring API
The FastAPI server allows you to trigger the scoring pipeline and retrieve individual scores on the fly.

1.  **Start the API server:**
    ```bash
    uvicorn api:app --reload
    ```
    The API will be live at `http://127.0.0.1:8000`.

2.  **Interact with the API:** Open a new terminal to use `curl` or navigate your browser to **`http://127.0.0.1:8000/docs`** for a user-friendly, interactive interface.

    *   **A) Trigger the scoring pipeline (run this first):**
        This will process the data in the background and create the `wallet_scores.csv` file.
        ```bash
        curl -X POST http://127.0.0.1:8000/score/generate
        ```
        *(Wait a minute for the background task to complete).*

    *   **B) Retrieve the score for a specific wallet:**
        Once the pipeline has run, you can query for any wallet address.
        ```bash
        curl http://127.0.0.1:8000/score/0x003a85c562730b196f7cba202a2515f2ff855736
        ```
