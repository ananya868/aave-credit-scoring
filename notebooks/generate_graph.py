import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define file paths
OUTPUT_DIR = "output"
SCORES_FILE = os.path.join(OUTPUT_DIR, "wallet_scores.csv")
IMAGE_FILE = os.path.join(OUTPUT_DIR, "score_distribution.png")

# Load the data
try:
    df = pd.read_csv(SCORES_FILE)
except FileNotFoundError:
    print(f"Error: The file {SCORES_FILE} was not found.")
    print("Please run 'python score_wallets.py' first to generate the scores.")
    exit()

# Create the plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

sns.histplot(df['credit_score'], bins=50, kde=True, ax=ax, color='#2c7bb6')

# Set titles and labels
ax.set_title('Distribution of Wallet Credit Scores', fontsize=18, pad=20)
ax.set_xlabel('Credit Score (0-1000)', fontsize=12)
ax.set_ylabel('Number of Wallets', fontsize=12)
ax.set_xlim(0, 1000)

# Save the figure
plt.savefig(IMAGE_FILE, dpi=300, bbox_inches='tight')

print(f"Graph successfully saved to {IMAGE_FILE}")