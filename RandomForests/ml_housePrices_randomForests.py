# Loading stuff
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

sns.set()
pd.set_option('max_columns', 1000)
warnings.filterwarnings('ignore')

# %matplotlib inline
# load data
df_train = pd.read_csv('/Users/josephlefournour/Projects/VSCode/MachineLearningPython/RandomForests/train.csv')

#explore data columns
print(df_train.columns.values)
print('No. variables:', len(df_train.columns.values))

num_missing = df_train.isnull().sum()
percent = num_missing / df_train.isnull().count()

df_missing = pd.concat([num_missing, percent], axis=1, keys=['MissingValues', 'Fraction'])
df_missing = df_missing.sort_values('Fraction', ascending=False)
df_missing[df_missing['MissingValues'] > 0]

variables_to_keep = df_missing[df_missing['MissingValues'] == 0].index
df_train = df_train[variables_to_keep]

# Build the correlation matrix
matrix = df_train.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(matrix, vmax=0.7, square=True)
plt.show()


interesting_variables = matrix['SalePrice'].sort_values(ascending=False)

# Filter out the target variables (SalePrice) and variables with a low correlation score (v such that -0.6 <= v <= 0.6)
interesting_variables = interesting_variables[abs(interesting_variables) >= 0.6]
interesting_variables = interesting_variables[interesting_variables.index != 'SalePrice']
interesting_variables

values = np.sort(df_train['OverallQual'].unique())
print('Unique values of "OverallQual":', values)

data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
data.plot.scatter(x='OverallQual', y='SalePrice')
plt.show()

cols = interesting_variables.index.values.tolist() + ['SalePrice']
sns.pairplot(df_train[cols], size=2.5)
plt.show()

# Build the correlation matrix
matrix = df_train[cols].corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, vmax=1.0, square=True)
