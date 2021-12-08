#!/usr/bin/env python
# coding: utf-8

# <div class='bar_title'></div>
# 
# *Practical Data Science*
# 
# # Machine Learning Introduction
# 
# Nikolai Stein<br>
# Chair of Information Systems and Business Analytics
# 
# Winter Semester 21/22

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction-and-Data-Set" data-toc-modified-id="Introduction-and-Data-Set-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction and Data Set</a></span></li><li><span><a href="#Loading-the-Data" data-toc-modified-id="Loading-the-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Loading the Data</a></span></li><li><span><a href="#Select-data-for-modeling" data-toc-modified-id="Select-data-for-modeling-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Select data for modeling</a></span><ul class="toc-item"><li><span><a href="#Selecting-the-prediction-target" data-toc-modified-id="Selecting-the-prediction-target-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Selecting the prediction target</a></span></li><li><span><a href="#Choosing-&quot;Features&quot;" data-toc-modified-id="Choosing-&quot;Features&quot;-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Choosing "Features"</a></span></li></ul></li><li><span><a href="#Building-Models-in-Scikit-learn" data-toc-modified-id="Building-Models-in-Scikit-learn-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Building Models in Scikit-learn</a></span><ul class="toc-item"><li><span><a href="#Our-first-Decision-Tree" data-toc-modified-id="Our-first-Decision-Tree-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Our first Decision Tree</a></span></li></ul></li><li><span><a href="#Model-Validation" data-toc-modified-id="Model-Validation-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Model Validation</a></span><ul class="toc-item"><li><span><a href="#Metrics" data-toc-modified-id="Metrics-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Metrics</a></span></li></ul></li><li><span><a href="#In-Sample-vs.-Out-of-Sample-Scores" data-toc-modified-id="In-Sample-vs.-Out-of-Sample-Scores-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>In-Sample vs. Out-of-Sample Scores</a></span></li><li><span><a href="#Model-Tuning" data-toc-modified-id="Model-Tuning-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Model Tuning</a></span></li><li><span><a href="#Training-a-Random-Forest" data-toc-modified-id="Training-a-Random-Forest-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Training a Random Forest</a></span></li><li><span><a href="#Missing-Value-Imputation" data-toc-modified-id="Missing-Value-Imputation-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Missing Value Imputation</a></span><ul class="toc-item"><li><span><a href="#Simple-Imputation" data-toc-modified-id="Simple-Imputation-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Simple Imputation</a></span></li><li><span><a href="#Advanced-Imputation" data-toc-modified-id="Advanced-Imputation-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Advanced Imputation</a></span></li></ul></li><li><span><a href="#Encoding-Categorical-Variables" data-toc-modified-id="Encoding-Categorical-Variables-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Encoding Categorical Variables</a></span><ul class="toc-item"><li><span><a href="#Label-Encoding" data-toc-modified-id="Label-Encoding-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Label Encoding</a></span></li><li><span><a href="#One-hot-Encoding" data-toc-modified-id="One-hot-Encoding-10.2"><span class="toc-item-num">10.2&nbsp;&nbsp;</span>One-hot Encoding</a></span></li></ul></li><li><span><a href="#Creating-Model-Pipelines" data-toc-modified-id="Creating-Model-Pipelines-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Creating Model Pipelines</a></span><ul class="toc-item"><li><span><a href="#Define-Preprocessing-steps" data-toc-modified-id="Define-Preprocessing-steps-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>Define Preprocessing steps</a></span></li><li><span><a href="#Define-the-Model" data-toc-modified-id="Define-the-Model-11.2"><span class="toc-item-num">11.2&nbsp;&nbsp;</span>Define the Model</a></span></li><li><span><a href="#Create-and-Evaluate-the-Pipeline" data-toc-modified-id="Create-and-Evaluate-the-Pipeline-11.3"><span class="toc-item-num">11.3&nbsp;&nbsp;</span>Create and Evaluate the Pipeline</a></span></li></ul></li><li><span><a href="#Wrapping-up" data-toc-modified-id="Wrapping-up-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Wrapping up</a></span></li></ul></div>

# ## Introduction and Data Set
# Credits: Most of the material of this lecture is adopted from www.kaggle.com
# 
# This lecture provides an overview of how machine learning models can be used for real problems. We will build models as well as a machine learning pipeline based on the following scenario:

# Your cousin has made millions of dollars speculating on real estate. He's offered to become business partners with you because of your interest in data science. He'll supply the money, and you'll supply models that predict how much various houses are worth.
# 
# You ask your cousin how he's predicted real estate values in the past and he says it is just intuition. But more questioning reveals that he's identified price patterns from houses he has seen in the past, and he uses those patterns to make predictions for new houses he is considering.
# 
# Instead of using intuition to make good decision, we want to train a machine learning model to predict the value of new houses.

# ## Loading the Data
# The first step in any machine learning project is to load and familiarize yourself with the data. To this end, we can use the pandas library from last week and load the dataset with the following commands:

# In[2]:


import pandas as pd


# In[3]:


melbourne_file_path = "https://github.com/NikoStein/pds_data/raw/main/data/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data.head()


# For simplicity we remove rows with missing values for this example. Note that a missing value can sometimes be a valuable information. 

# In[4]:


print(melbourne_data.shape)
melbourne_data.dropna(axis=0, inplace=True)
print(melbourne_data.shape)


# ## Select data for modeling
# On a first glimpse, we see that our dataset has too many variables to wrap our heads around. How can we pare down this overwhelming amount of data to something we can understand?
# 
# We'll start by picking a few variables using our intuition. To choose variables, we'll need to see a list of all columns in the dataset. That is done with the columns property of the DataFrame:

# In[51]:


melbourne_data.columns


# ### Selecting the prediction target
# 
# To train a predictive model using supervised machine learning techniques, we have to identify the target variable. In the problem at hand, we want to predict the house prices. This information is encoded in the column Price.
# 
# By convention, the target variable is called **y**. 

# In[6]:


y = melbourne_data['Price']


# ### Choosing "Features"
# The columns that serve as input for our model (and are later used to make predictions) are called "features." Sometimes, you will use all columns except the target as features. Other times you'll be better off with fewer features.
# 
# For now, we'll build a model with only a few features. 
# 
# We select multiple features by providing a list of column names inside brackets. Each item in that list should be a string (with quotes).
# 

# In[7]:


melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']


# By convention, this data is called **X**.

# In[8]:


X = melbourne_data[melbourne_features]


# Let's quickly review the data we'll be using to predict house prices using the describe method and the head method, which shows the top few rows. Visually checking your data with these commands is an important part of a data scientist's job. You'll frequently find surprises in the dataset that deserve further inspection.

# In[9]:


X.head()


# In[10]:


X.describe()


# ## Building Models in Scikit-learn
# For now, we will use the scikit-learn library to create our models. As you will see in the upcoming section, this library is written as sklearn in the code. Scikit-learn offers a lot of powerful features and is easily the most popular library for modeling tabular data.

# The steps to building and using a model in Scikit-learn are:
# * Define:  What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# * Fit: Capture patterns from our input data.
# * Predict: Make predictions using input variables and the trained model.
# * Evaluate: Determine how accurate the model's predictions are.

# ### Our first Decision Tree
# Here is a simple example of defining a decision tree model with scikit-learn and fitting it with the features and target variable selected above:

# In[11]:


from sklearn.tree import DecisionTreeRegressor

# Define
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit
melbourne_model.fit(X, y)


# Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run. This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose.

# We now have a fitted model that we can use to make predictions.
# 
# In practice, you'll want to make predictions for new houses coming on the market rather than the houses we already have prices for. But we'll make predictions for the first few rows of the training data to see how the predict function works.

# In[12]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


# ## Model Validation
# 
# We have successfully trained our very first model. However, we have no clue how good our model is. Yet, measuring model quality is the key to iteratively improving our models.
# 
# In most (though not all) applications, the relevant measure of model quality is predictive accuracy. In other words, will the model's predictions be close to what actually happens.

# ### Metrics
# 
# To evaluate the performance of our model we need to find a way to summarize the model quality in an understandable way. If we compare predicted and actual home values in our example dataset for 10,000 houses, we will find a mix of good and bad predictions. However, looking through a list of 10,000 predicted and actual values would be pointless. We need to summarize this into a single metric.

# There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error (**MAE**). Let's break down this metric starting with the last word, error.
# 
# The prediction error for each house is: 
# 
# ``error=actual−predicted``

# So, if a house cost \$150,000 and you predicted it would cost \$100,000 the error is \$50,000.
# 
# With the MAE metric, we take the absolute value of each error. This converts each error to a positive number. We then take the average of those absolute errors. This is our measure of model quality.

# We could implement a function to calculate this metric (or any other metric) on our dataframe. However, Scikit-learn provides implementations of the most common metrics that can be easily imported.

# In[13]:


from sklearn.metrics import mean_absolute_error


# So lets use our decision tree model to make predictions for all observations in our dataset and calculate the MAE:

# In[14]:


predicted_home_prices = melbourne_model.predict(X)
mae = mean_absolute_error(y, predicted_home_prices)

print("The MAE of our model is: {}".format(mae))


# ## In-Sample vs. Out-of-Sample Scores

# The MAE of our model looks very promising. However, we used a single "sample" of houses for both building the model and evaluating it. Hence, the measure we just computed can be called an "in-sample" score.
# 
# Trusting the in-sample score to evaluate a model is very dangerous. Imagine that there is a variable in the dataset that is unrelated to the home price (e.g., the name of the current owner). However, in the sample of data we used to build the model, all names are unique and hence, all house prices in the sample can be explained by this feature. Our model will see this pattern and it will try to apply it to new datasets. 

# Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model. The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. This data is called validation data.

# The scikit-learn library has a function train_test_split to break up the data into two pieces. We'll use some of that data as training data to fit the model, and we'll use the other data as validation data to calculate the MAE.

# In[16]:


from sklearn.model_selection import train_test_split


# In[19]:


# Split the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Build the model
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(train_X, train_y)

# Evaluate the performance
val_predictions = melbourne_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)

print("The MAE of our model is: {}".format(val_mae))


# The MAE for the in-sample data was about \$500. Out-of-sample it is more than \$250,000.
# 
# This is the difference between a model that is almost exactly right, and one that is unusable for most practical purposes. As a point of reference, the average home value in the validation data is about \$1.1 million. So the error in new data is about a quarter of the average home value.
# 
# There are many ways to improve this model, such as experimenting to find better features or different model types.

# ## Model Tuning
# 
# Now that we have a reliable way to measure the model performance, we can experiment with different parameters of the decision tree (or entirely different models) and see which combination gives us the best predictions. You can find the available models and parameters in the Scikit-learn [documentation](https://scikit-learn.org/stable/modules/classes.html).
# 
# In this simple example we will stick with our decision tree model and only vary one of the parameters.
# 
# You can see in the decision tree [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) that the model has many parameters (more than you'll want or need for a long time). The most important parameters determine the tree's depth. 

# In practice, it's not uncommon for a tree to have 10 splits between the top level (all houses) and a leaf. As the tree gets deeper, the dataset gets sliced up into leaves with fewer houses. If a tree only had 1 split, it divides the data into 2 groups. If each group is split again, we would get 4 groups of houses. Splitting each of those again would create 8 groups. If we keep doubling the number of groups by adding more splits at each level, we'll have $2^{10}$  groups of houses by the time we get to the 10th level. That's 1024 leaves.
# 
# When we divide the houses amongst many leaves, we also have fewer houses in each leaf. Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses). This is a phenomenon called **overfitting**, where a model matches the training data almost perfectly, but does poorly in validation and other new data. 

# On the flip side, if we make our tree very shallow, it doesn't divide up the houses into very distinct groups. At an extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses. Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason). When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called **underfitting**.
# 
# Since we care about accuracy on new data, which we estimate from our validation data, we want to find the sweet spot between underfitting and overfitting.

# There are a few alternatives for controlling the tree depth, and many allow for some routes through the tree to have greater depth than other routes. But the max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting. The more leaves we allow the model to make, the more we move from the underfitting area in the above graph to the overfitting area.
# 
# We write a short utility function to help compare MAE scores from different values for max_leaf_nodes:

# In[20]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# Next, we loop over different values for the parameter to compare the in-sample and the out-of-sample performance of our model:

# In[21]:


for max_leaf_nodes in [2, 5, 50, 500, 5000, 10000]:
    is_mae = get_mae(max_leaf_nodes, X, X, y, y)
    oos_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t In-sample:  %d \t Out-of-sample:  %d" %(max_leaf_nodes, is_mae, oos_mae))


# Here's the takeaway: Models can suffer from either:
# 
# - Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
# - Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.
# 
# We use validation data, which isn't used in model training, to measure a candidate model's accuracy. This lets us try many candidate models and keep the best one.

# ## Training a Random Forest
# 
# Decision trees leave us with a difficult trade-off. A deep tree with lots of leaves will overfit because each prediction is coming from historical data from only the few houses at its leaf. But a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions in the raw data.
# 
# Even today's most sophisticated modeling techniques face this tension between underfitting and overfitting. But, many models have clever ideas that can lead to better performance. We'll look at the random forest as an example.

# The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters.

# Thanks to our Scikit-learn modeling pipeline we can reuse most of our code to train a random forest model with 100 trees.

# In[22]:


from sklearn.ensemble import RandomForestRegressor


# In[24]:


# Define
forest_model = RandomForestRegressor(random_state=1, n_estimators=100)

# Fit
forest_model.fit(train_X, train_y)

# Evaluate
melb_preds = forest_model.predict(val_X) 
print("The MAE of our model is: {}".format(mean_absolute_error(val_y, melb_preds)))


# There is likely room for further improvement, but this is a big improvement over the best decision tree error of 243,000. There are parameters which allow you to change the performance of the Random Forest much as we changed the maximum depth of the single decision tree. But one of the best features of Random Forest models is that they generally work reasonably even without this tuning.

# ## Missing Value Imputation
# 
# We just finished training our first machine learning models. To further improve the predictive power of the models we will have to work on our dataset.
# 
# We will start with handling missing values in the data. Most machine learning libraries (including scikit-learn) give an error if we try to build a model using data with missing values. So we'll need to choose a strategy to handle missing values.
# 
# We have already used a very simple strategy and dropped all rows containing missing values in the first example. To evaluate different approaches we will first load the full dataset and create a train-test split. (Note: As we cannot apply all imputation functions (e.g., mean) to categorical data we will only use numerical predictions in this simple example

# In[25]:


# Load dataset
data = pd.read_csv(melbourne_file_path)

# Target variable
y = data['Price']

# Drop non-numeric variables
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)


# ### Simple Imputation
# 
# One popular way to handle missing values is called imputation. Here, we fill in the missing values with some number. For instance, we can fill in the mean value along each column. The imputed value won't be exactly right in most cases, but it usually leads to more accurate models than you would get from dropping the column entirely.

# In[27]:


from sklearn.impute import SimpleImputer


# In[28]:


# Imputation
simple_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(simple_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(simple_imputer.transform(X_valid))

# "Repair" column names
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns


# To evaluate the performance of the approach, we modify our helper function (get_mae) to train and evaluate our model on different datasets:

# In[29]:


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# In[30]:


mae_imputation = score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid)
print("MAE using Imputation: {}".format(mae_imputation))


# ### Advanced Imputation
# 
# We see that the imputation approach performs much better compared to the simple solution dropping all rows with NA values.
# 
# Imputation is the standard approach, and it usually works well. However, imputed values may be systematically above or below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique in some other way. In that case, your model would make better predictions by considering which values were originally missing.
# 
# In the advanced imputation approach, we impute the missing values, as before. And, additionally, for each column with missing entries in the original dataset, we add a new column that shows the location of the imputed entries.

# In[33]:


# Make a copy of the original datasets to avoid chaning the original data frame
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Find all columns with missing values:
cols_with_missing = X_train.columns.values[X_train.isna().sum() > 0]

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
    
# Imputation
simple_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(simple_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(simple_imputer.transform(X_valid_plus))

# "Repair" column names
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns


# In[35]:


mae_imputation_advanced = score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)
print("MAE using Imputation: {}".format(mae_imputation_advanced))


# As we see, advanced imputation does not improve the performance of our model in the problem at hand. In general, advanced imputation will meaningfully improve results in some cases. In other cases, it doesn't help at all.

# ## Encoding Categorical Variables
# 
# Until now we only used numerical features for our models. However, valuable information is often encoded in categorical variables (e.g., gender, city, job). 
# 
# If we simply plug these variables into machine learning models we will get an error. Hence, we need to find an  appropriate preprocessing to capture the information hidden in categorical variables. 
# 
# The easiest approach to deal with categorical variables is to drop them from the dataset (that is what we have done before). However, this approach will only produce satisfying results if the dropped columns did not contain useful information.

# ### Label Encoding
# 
# One common approach to handle categorical variables is called label encoding. Here, we assign each unique value to a different integer (e.g., bad = 0, neutral = 1, good = 2). This assumption makes sense in this example, because there is an indisputable ranking to the categories. Not all categorical variables have a clear ordering in the values, but we refer to those that do as ordinal variables. For tree-based models (like decision trees and random forests), you can expect label encoding to work well with ordinal variables.
# 
# For simplicity, we will drop columns with missing values for the following evaluation.

# In[36]:


# Load dataset
data = pd.read_csv(melbourne_file_path)

# Drop NA
data.dropna(axis=0, inplace=True)

# Separate target from predictors
y = data['Price']
X = data.drop(['Price'], axis=1)

# Train-test split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)


# As we do not want to use all categorical variables we focus on those with a limited number of categories:

# In[37]:


low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

print(low_cardinality_cols)


# ...and combine them with the numerical variables:

# In[38]:


# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep only selected columns
cols_to_keep = low_cardinality_cols + numerical_cols
X_train = X_train_full[cols_to_keep].copy()
X_valid = X_valid_full[cols_to_keep].copy()

X_train.head()


# We can now perform label encoding on our new dataset using the functions provided by Scikit-learn. Subsequently, we can evaluate our approach by using our score_dataset utility function.

# In[39]:


from sklearn.preprocessing import LabelEncoder


# In[40]:


# Make a copy to protect original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply label encoder
label_encoder = LabelEncoder()
for col in low_cardinality_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
    

# Evaluate performance
mae_label_encoding = score_dataset(label_X_train, label_X_valid, y_train, y_valid)
print("MAE using Label Encoding: {}".format(mae_label_encoding))


# ### One-hot Encoding
# 
# One-hot encoding creates new binary columns indicating the presence (or absence) of each possible value in the original data. 
# 
# In contrast to label encoding, one-hot encoding does not assume an ordering of the categories. Thus, you can expect this approach to work particularly well if there is no clear ordering in the categorical data. We refer to categorical variables without an intrinsic ranking as nominal variables.
# 
# One-hot encoding generally does not perform well if the categorical variable takes on a large number of values (i.e., you generally won't use it for variables taking more than 15 different values).

# Again, we can use Scikit-learn functions to implement one-hot encodings:

# In[41]:


from sklearn.preprocessing import OneHotEncoder


# In[42]:


# Appley one-hot encoder to each column with categorical data
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
one_hot_cols_train = pd.DataFrame(one_hot_encoder.fit_transform(X_train[low_cardinality_cols]))
one_hot_cols_valid = pd.DataFrame(one_hot_encoder.transform(X_valid[low_cardinality_cols]))

# Repair index 
one_hot_cols_train.index = X_train.index
one_hot_cols_valid.index = X_valid.index

# Remove categorical columns and replace with one-hot encoding
num_X_train = X_train.drop(low_cardinality_cols, axis=1)
num_X_valid = X_valid.drop(low_cardinality_cols, axis=1)
one_hot_X_train = pd.concat([num_X_train, one_hot_cols_train], axis=1)
one_hot_X_valid = pd.concat([num_X_valid, one_hot_cols_valid], axis=1)

# Evaluate performance
one_hot_encoding = score_dataset(one_hot_X_train, one_hot_X_valid, y_train, y_valid)
print("MAE using One-hot Encoding: {}".format(one_hot_encoding))


# ## Creating Model Pipelines
# 
# Up to now, we learned how to prepare our datasets, train, tune, and evaluate powerful models. However, we wrote lots of code and functions to perform all the required tasks. Scikit-learn pipelines are a simple way to keep our data preprocessing and modeling code organized. Specifically, a pipeline bundles preprocessing and modeling steps so we can use the whole bundle as if it were a single step.
# 
# Using pipelines provides multiple benefits:
# * Cleaner Code
# * Fewer Bugs
# * Easier to Productionize
# * More Options for Model Validation

# We will build a pipeline using all numerical variables as well as the low cardinatlity categorical variables

# In[45]:


# Load dataset
data = pd.read_csv(melbourne_file_path)

# Separate target from predictors
y = data['Price']
X = data[cols_to_keep]

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
X_train.head()


# Writing a pipeline in Scikit-learn can be broken down into 3 steps:
# 1. Define preprocessing steps
# 2. Define the model
# 3. Create and evaluate the pipeline

# ### Define Preprocessing steps
# 
# We use the ``ColumnTransformer`` class to bundle together different preprocessing steps. To this end, we will impute missing values in the numerical columns and impute missing values and use one-hot encoding in the categorical columns.

# In[46]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[48]:


# Preprocessing numerical columns
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

# Bundle both preprocessors
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, low_cardinality_cols)
])


# ### Define the Model
# Next, we define a random forest model.

# In[49]:


model = RandomForestRegressor(n_estimators=100, random_state=1)


# ### Create and Evaluate the Pipeline
# 
# Finally, we use the ``Pipeline`` class to define a pipeline that bundles the preprocessing and modeling steps. There are a few important things to notice:
# 
# * With the pipeline, we preprocess the training data and fit the model in a single line of code. (In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps. This becomes especially messy if we have to deal with both numerical and categorical variables!)
# * With the pipeline, we supply the unprocessed features in X_valid to the predict() command, and the pipeline automatically preprocesses the features before generating predictions. (However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)

# In[50]:


# Bundle preprocessing and modeling code in a pipeline
complete_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Preprocess the raw training data and fit the model
complete_pipeline.fit(X_train, y_train)

# Preprocess the raw validation data and make predictions
preds = complete_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print("MAE using the complete pipeline: {}".format(score))


# ## Wrapping up
# 
# In this lecture we learned how to build powerful machine learning models leveraging numerical as well as categorical variables. Additionally, we learned about model pipelines which are helpful for creating reproducible and understandable code. 
# 
# To keep improving, view the [scikit-learn documentation](https://scikit-learn.org/stable/) and keep working on your own projects!
