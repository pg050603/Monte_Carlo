import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

avg = 1
std_dev = .1
num_reps = 500

num_simulations = 1000

def sales_setup(ave, std_dev, num_reps):
    pct_to_target = np.random.normal(avg, std_dev, num_reps).round(2)
    target_values = [75_000, 100_000, 200_000, 300_000, 400_000, 500_000]
    likelihood = [.3, .3, .2, .1, .05, .05]
    sales_target = np.random.choice(target_values, num_reps, p=likelihood)
    df = pd.DataFrame(index=range(num_reps), data={'Pct_To_Target': pct_to_target,
                                                'Sales_Target': sales_target})
    df['Sales'] = df['Pct_To_Target'] * df['Sales_Target']
    return df

def calc_commission_rate(x):
    """ Return the commission rate based on the table:
    0-90% = 2%
    91-99% = 3%
    >= 100 = 4%
    """
    if x <= .90:
        return .02
    if x <= .99:
        return .03
    else:
        return .04

def apply_commission(df):
    df['Commission_Rate'] = df['Pct_To_Target'].apply(calc_commission_rate)
    df['Commission_Amount'] = df['Commission_Rate'] * df['Sales']
    return df

def MC_simulator(num_reps):
    # Define a dataframe to store the results from each simulation that we want to analyze
    all_stats = pd.DataFrame(columns=['Sales', 'Commission_Amount', 'Sales_Target'])

    # Loop through many simulations
    for i in range(num_simulations):
        df = sales_setup(avg, std_dev, num_reps)
        df = apply_commission(df)

        # We want to track sales, commission amounts, and sales targets over all the simulations
        all_stats.loc[i] = [df['Sales'].sum().round(0),
                            df['Commission_Amount'].sum().round(0),
                            df['Sales_Target'].sum().round(0)]

    return all_stats

a = MC_simulator(num_reps)

a['Away_From_Target'] = a['Sales_Target'] - a['Sales']

# Create a histogram of the away from target values
plt.figure(figsize=(10, 6))
sns.histplot(data=a, x='Away_From_Target', bins=30)
plt.title('Histogram of Away From Target')
plt.xlabel('Away From Target')
plt.ylabel('Frequency')
plt.show()





