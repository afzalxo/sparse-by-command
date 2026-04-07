import pandas as pd
import numpy as np

CSV_PATH = "carla_ue4_dataset_base/data.csv"

df = pd.read_csv(CSV_PATH)

# 1. Identify groups
is_turn = df['instruction'].str.contains('turn|change')
is_stop = df['instruction'].str.contains('stop|brake')
is_straight = df['instruction'].str.contains('follow|go straight')

# 2. Define target counts based on the rarest important class (turns)
# We want to keep 100% of turns, and downsample the rest to match.
turn_count = len(df[is_turn])
print(f"Total turns available: {turn_count}")

new_candidates = np.zeros(len(df), dtype=int)

for idx, row in df.iterrows():
    cmd = row['instruction']
    
    if "change" in cmd:
        new_candidates[idx] = 1
    elif "turn right" in cmd:
        if np.random.rand() < 0.7187:
            new_candidates[idx] = 1
    elif "turn left" in cmd:
        new_candidates[idx] = 1
    elif "straight" in cmd:
        if np.random.rand() < 0.6062:
            new_candidates[idx] = 1
    elif "follow" in cmd:
        if np.random.rand() < 0.1642:
            new_candidates[idx] = 1
    elif "stop" in cmd or "brake" in cmd:
        if np.random.rand() < 0.2766:
            new_candidates[idx] = 1

df['is_train_candidate'] = new_candidates
print(f"New candidate count: {df['is_train_candidate'].sum()}")
df.to_csv(CSV_PATH, index=False)
