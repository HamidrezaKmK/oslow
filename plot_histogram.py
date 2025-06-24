import os
import wandb
import matplotlib.pyplot as plt
from wandb.apis.public import Run
import json


# Configuration
RUNS_DIR = "/h/liaidan/oslow/runs"  # Typically ~/wandb or ./wandb
RUN_PREFIX = "sigmoid"  # e.g., "nonparametric_laplace"
METRIC_NAME = "evaluation/best_backward_penalty"
SAVE_PATH = f"/h/liaidan/oslow/{RUN_PREFIX}_backward_penalty_histogram.png"  # Path to save the plot

def get_final_metric(run_dir):
    """Extract metric from latest-run subdirectory"""
    # Navigate to latest-run/files
    latest_run_dir = os.path.join(run_dir, "wandb/latest-run")
    if not os.path.exists(latest_run_dir):
        print(f"latest-run not found in {run_dir}")
        return None
        
    summary_file = os.path.join(latest_run_dir, "files", "wandb-summary.json")
    
    if not os.path.exists(summary_file):
        print(f"Summary file not found in {latest_run_dir}")
        return None
    
    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)
        return summary.get(METRIC_NAME)
    except Exception as e:
        print(f"Error loading {summary_file}: {str(e)}")
        return None

# Get all run directories with the prefix
all_runs = [os.path.join(RUNS_DIR, d) 
            for d in os.listdir(RUNS_DIR) 
            if d.startswith(RUN_PREFIX)]

# Collect metrics
penalties = []
for run_path in all_runs:
    metric_value = get_final_metric(run_path)
    if metric_value is not None:
        penalties.append(metric_value)


# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(penalties, bins=15, edgecolor='black', alpha=0.7)
plt.xlabel("Best Backward Penalty")
plt.ylabel("Frequency")
plt.title(f"Distribution of {METRIC_NAME} ({len(penalties)} runs)")
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"Plot saved to {SAVE_PATH}")

# Show the plot (optional)
plt.show()