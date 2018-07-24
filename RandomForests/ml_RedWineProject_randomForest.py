# import numpy good support for efficient numerical computation
import numpy as np 

# this supoorts data frames
import pandas as pd 

# import functions for machine learning. 
# The first one will be the train_test_split() function from the model_selection module. 
# This module contains many utilities that will help us choose between models.
# sampling helper
from sklearn.model_selection import train_test_split

# Next, we'll import the entire preprocessing module. 
# This contains utilities for scaling, transforming, and wrangling data
# import preprocessing modules
from sklearn import preprocessing

#import random forest family models
from sklearn.ensemble import RandomForestRegressor

#import cross-validation pipeline, this is a tool that helps
# with data cross - validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Next, let's import some metrics we can use to evaluate our model performance later.
from sklearn.metrics import mean_squared_error, r2_score

# And finally, we'll import a way to persist our model for future use.
# Import module for saving scikit-learn models
from sklearn.externals import joblib

#load red wine data - we can use CSV_read function that works even using a URL
# import wine data from remote URLPython
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url , sep=";")

#Now let's take a look at the first 5 rows of data:
#print (data.head())

# lets look more into the data
# shape
#print(data.shape)

#summary stats
#print(data.describe())

# split the data into training and testing sets
# Splitting the data into training and test sets at the beginning of your modeling workflow 
# is crucial for getting a realistic estimate of your model's performance. 

# First, let's separate our target (y) features from our input (X) features:
# Separate target from training featuresPython
y = data.quality
X = data.drop('quality', axis=1)

# This allows us to take advantage of Scikit-Learn's useful train_test_split function:
# As you can see, we'll set aside 20% of the data as a test set for evaluating our model. 
# We also set an arbitrary "random state" (a.k.a. seed) so that we can reproduce our results. 
# Finally, it's good practice to stratify your sample by the target variable. 
# This will ensure your training set looks similar to your test set, 
# making your evaluation metrics more reliable
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)


# Lazy scaling
#X_train_scaled = preprocessing.scale(X_train)
#print(X_train_scaled)
#print(X_train_scaled.mean(axis=0))
#print (X_train_scaled.std(axis=0))


# So instead of directly invoking the scale function, we'll be using a feature in Scikit-Learn 
# called the Transformer API. The Transformer API allows you to "fit" a preprocessing step 
# using the training data the same way you'd fit a model... 
# ...and then use the same transformation on future data sets! 
# Here's what that process looks like: 
# Fit the transformer on the training set (saving the means and standard deviations) 
# Apply the transformer to the training set (scaling the training data) 
# Apply the transformer to the test set (using the same means and standard deviations) 
# This makes your final estimate of model performance more realistic, 
# and it allows to insert your preprocessing steps into a cross-validation pipeline

# Fitting the Transformer APIPython
scaler = preprocessing.StandardScaler().fit(X_train)

# Now, the scaler object has the saved means and standard deviations for each feature in the training set

# Applying transformer to training data
X_train_scaled = scaler.transform(X_train)
#print(X_train_scaled.mean(axis=0))
#print (X_train_scaled.std(axis=0))

# Applying transformer to test data
X_test_scaled = scaler.transform(X_test)
print (X_test_scaled.mean(axis=0))
print (X_test_scaled.std(axis=0))

# Notice how the scaled features in the test set are not perfectly centered at zero with unit variance! 
# This is exactly what we'd expect, as we're transforming the test set using the means from the training set, not from the test set itself.

# In practice, when we set up the cross-validation pipeline, 
# we won't even need to manually fit the Transformer API. 
# Instead, we'll simply declare the class object, like so:

# Pipeline with preprocessing and modelPython

pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))

#This is exactly what it looks like: a modeling pipeline that first transforms the data using StandardScaler() 
# and then fits a model using a random forest regressor.

# Declare Hyper parameters

# There are two types of parameters we need to worry about: model parameters and hyperparameters. 
# Models parameters can be learned directly from the data (i.e. regression coefficients), 
# while hyperparameters cannot.

# Hyperparameters express "higher-level" structural information about the model, 
# and they are typically set before training the model

# As an example, let's take our random forest for regression: 
# Within each decision tree, the computer can empirically decide where to create branches based on 
# either mean-squared-error (MSE) or mean-absolute-error (MAE). 
# Therefore, the actual branch locations are MODEL parameters.
# However, the algorithm does not know which of the two criteria, MSE or MAE, that it should use. 
# The algorithm also cannot decide how many trees to include in the forest. 
# These are examples of HYPERparameters that the user must set

# We can list the tunable hyperparameters like so:
# print(pipeline.get_params())

# Now, let's declare the hyperparameters we want to tune through cross-validation.
# Declare hyperparameters to tunePython
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

                  