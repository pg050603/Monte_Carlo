import pandas as pd
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')

avg = 1
std_dev = .1
num_reps = 500
num_simulations = 1000

pct_to_target = np.random.normal(avg, std_dev, num_reps).round(2)

print(pct_to_target[:10])
