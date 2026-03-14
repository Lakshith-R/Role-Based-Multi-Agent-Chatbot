import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# For pretty display in notebooks (optional)
from tabulate import tabulate

# Set plotting style
sns.set(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

# Load CSV exported from Google Forms
df = pd.read_csv("evaluation_responses.csv")

# Rename columns for readability if needed
df.columns = [col.strip() for col in df.columns]

# Drop timestamp if exists
if "Timestamp" in df.columns:
    df.drop("Timestamp", axis=1, inplace=True)

# Map Likert scale to numerical values
likert_mapping = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5
}

df_numeric = df.replace(likert_mapping)

# Compute summary stats
summary_stats = df_numeric.describe().transpose()
summary_stats = summary_stats[["mean", "std", "min", "max"]].round(2)

# Save summary as table
summary_stats.to_csv("feedback_summary_stats.csv")

# Display in terminal
print("\n📊 Likert Scale Evaluation Summary:\n")
print(tabulate(summary_stats, headers="keys", tablefmt="github"))

# Create output folder for plots
os.makedirs("plots", exist_ok=True)

# Plot bar chart for each question
for col in df.columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, palette="viridis", order=likert_mapping.keys())
    plt.title(f"Responses for: {col}", fontsize=12)
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"plots/{col[:40].replace(' ', '_')}.png")
    plt.close()

print("\n✅ Bar plots saved in 'plots/' folder.")
