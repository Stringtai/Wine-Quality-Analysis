import numpy as np
import pandas as pd
from time import time
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import visuals as vs  # Custom visualization module (ensure it's available)

# Load the dataset
data = pd.read_csv("winequality-red.csv", sep=";")
n_wines = data.shape[0]  # Total number of wines

# Filter wines based on quality ratings (Fixing incorrect conditions)
quality_above_6 = data.loc[data['quality'] >= 7]
n_above_6 = quality_above_6.shape[0]

quality_below_5 = data.loc[data['quality'] < 5]
n_below_5 = quality_below_5.shape[0]

quality_between_5 = data.loc[(data['quality'] >= 5) & (data['quality'] <= 6)]
n_between_5 = quality_between_5.shape[0]

# Percentage of wines with quality rating 7 and above
greater_percent = (n_above_6 * 100) / n_wines

# Print statistics
print("Total number of wine data: {}".format(n_wines))
print("Wines with rating 7 and above: {}".format(n_above_6))
print("Wines with rating less than 5: {}".format(n_below_5))
print("Wines with rating 5 and 6: {}".format(n_between_5))
print("Percentage of wines with quality 7 and above: {:.2f}%".format(greater_percent))

# Display summary statistics of the dataset
display(np.round(data.describe()))

# Scatter matrix to visualize relationships between features
pd.plotting.scatter_matrix(data, alpha=0.3, figsize=(40, 40), diagonal='kde')

# Compute and visualize correlation matrix
correlations = data.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(correlations, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.title("Feature Correlation Heatmap")
plt.show()

# Joint plot for pH vs Fixed Acidity
fixedAcidity_pH = data[['pH', 'fixed acidity']]
sns.jointplot(x='fixed acidity', y='pH', data=fixedAcidity_pH, kind='scatter', height=6)
plt.show()

# Joint plot for Citric Acid vs Fixed Acidity
g = sns.jointplot(x='fixed acidity', y='citric acid', data=data, height=6)
g.plot_joint(sns.regplot, scatter_kws={'s': 10})
g.plot_marginals(sns.histplot)
plt.show()

# Bar plot for Quality vs Volatile Acidity (Ensure 'volatileAcidity_quality' is defined)
fig, axs = plt.subplots(ncols=1, figsize=(10, 6))
sns.barplot(x='quality', y='volatile acidity', data=data, ax=axs)
plt.title('Quality vs Volatile Acidity')
plt.tight_layout()
plt.show()

# Bar plot for Quality vs Alcohol (Ensure 'quality_alcohol' is defined)
fig, axs = plt.subplots(ncols=1, figsize=(10, 6))
sns.barplot(x='quality', y='alcohol', data=data, ax=axs)
plt.title('Quality vs Alcohol')
plt.tight_layout()
plt.show()

# Detect and display outliers for each feature
for feature in data.keys():
    Q1 = np.percentile(data[feature], q=25)
    Q3 = np.percentile(data[feature], q=75)
    interquartile_range = Q3 - Q1
    step = 1.5 * interquartile_range
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])
