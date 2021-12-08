#!/usr/bin/env python
# coding: utf-8

# # <div class='bar_title'></div>
# 
# *Practical Data Science*
# 
# # Descriptive Analytics with Pandas
# 
# Nikolai Stein<br>
# Chair of Information Systems and Business Analytics
# 
# Winter Semester 21/22

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Motivation" data-toc-modified-id="Motivation-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Motivation</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Two-ways-of-exploring-data" data-toc-modified-id="Two-ways-of-exploring-data-1.0.1"><span class="toc-item-num">1.0.1&nbsp;&nbsp;</span>Two ways of exploring data</a></span></li></ul></li><li><span><a href="#Exploratory-data-analysis-(EDA)" data-toc-modified-id="Exploratory-data-analysis-(EDA)-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Exploratory data analysis (EDA)</a></span><ul class="toc-item"><li><span><a href="#Building-intuition-about-the-data" data-toc-modified-id="Building-intuition-about-the-data-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Building intuition about the data</a></span></li></ul></li></ul></li><li><span><a href="#Introducing-Pandas" data-toc-modified-id="Introducing-Pandas-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Introducing Pandas</a></span><ul class="toc-item"><li><span><a href="#Typical-Use-Cases" data-toc-modified-id="Typical-Use-Cases-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Typical Use Cases</a></span></li><li><span><a href="#Pandas-within-the-data-science-toolkit" data-toc-modified-id="Pandas-within-the-data-science-toolkit-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Pandas within the data science toolkit</a></span></li><li><span><a href="#Installation-and-Import" data-toc-modified-id="Installation-and-Import-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Installation and Import</a></span></li></ul></li><li><span><a href="#Series-and-DataFrames" data-toc-modified-id="Series-and-DataFrames-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Series and DataFrames</a></span><ul class="toc-item"><li><span><a href="#Creating-DataFrames-from-scratch" data-toc-modified-id="Creating-DataFrames-from-scratch-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Creating DataFrames from scratch</a></span></li><li><span><a href="#Indexing-DataFrames" data-toc-modified-id="Indexing-DataFrames-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Indexing DataFrames</a></span></li><li><span><a href="#Reading-and-writing-CSVs" data-toc-modified-id="Reading-and-writing-CSVs-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Reading and writing CSVs</a></span></li></ul></li><li><span><a href="#Basic-DataFrame-operations:-Viewing" data-toc-modified-id="Basic-DataFrame-operations:-Viewing-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Basic DataFrame operations: Viewing</a></span><ul class="toc-item"><li><span><a href="#Viewing-your-data" data-toc-modified-id="Viewing-your-data-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Viewing your data</a></span></li><li><span><a href="#Getting-info-about-your-data-.info()" data-toc-modified-id="Getting-info-about-your-data-.info()-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Getting info about your data <code>.info()</code></a></span></li><li><span><a href="#Getting-info-about-your-data:-.shape" data-toc-modified-id="Getting-info-about-your-data:-.shape-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Getting info about your data: <code>.shape</code></a></span></li><li><span><a href="#Describing-your-variables" data-toc-modified-id="Describing-your-variables-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Describing your variables</a></span></li><li><span><a href="#Describing-Categorical-Variables" data-toc-modified-id="Describing-Categorical-Variables-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Describing Categorical Variables</a></span></li><li><span><a href="#Relationships-between-continuous-variables" data-toc-modified-id="Relationships-between-continuous-variables-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Relationships between continuous variables</a></span></li></ul></li><li><span><a href="#Basic-DataFrame-operations:-Data-Cleaning" data-toc-modified-id="Basic-DataFrame-operations:-Data-Cleaning-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Basic DataFrame operations: Data Cleaning</a></span><ul class="toc-item"><li><span><a href="#Handling-duplicates-(1)" data-toc-modified-id="Handling-duplicates-(1)-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Handling duplicates (1)</a></span></li><li><span><a href="#Handling-duplicates-(2)" data-toc-modified-id="Handling-duplicates-(2)-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Handling duplicates (2)</a></span></li><li><span><a href="#Inplace-Operations" data-toc-modified-id="Inplace-Operations-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Inplace Operations</a></span></li><li><span><a href="#Column-cleanup" data-toc-modified-id="Column-cleanup-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Column cleanup</a></span></li><li><span><a href="#Renaming-Columns" data-toc-modified-id="Renaming-Columns-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Renaming Columns</a></span></li><li><span><a href="#Operating-on-many-columns" data-toc-modified-id="Operating-on-many-columns-5.6"><span class="toc-item-num">5.6&nbsp;&nbsp;</span>Operating on many columns</a></span></li><li><span><a href="#Working-with-missing-values" data-toc-modified-id="Working-with-missing-values-5.7"><span class="toc-item-num">5.7&nbsp;&nbsp;</span>Working with missing values</a></span></li><li><span><a href="#Identify-missing-values" data-toc-modified-id="Identify-missing-values-5.8"><span class="toc-item-num">5.8&nbsp;&nbsp;</span>Identify missing values</a></span></li><li><span><a href="#Removing-null-values" data-toc-modified-id="Removing-null-values-5.9"><span class="toc-item-num">5.9&nbsp;&nbsp;</span>Removing null values</a></span></li><li><span><a href="#Imputation" data-toc-modified-id="Imputation-5.10"><span class="toc-item-num">5.10&nbsp;&nbsp;</span>Imputation</a></span></li></ul></li><li><span><a href="#Basic-DataFrame-operations:-Slicing,-selecting,-extracting" data-toc-modified-id="Basic-DataFrame-operations:-Slicing,-selecting,-extracting-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Basic DataFrame operations: Slicing, selecting, extracting</a></span><ul class="toc-item"><li><span><a href="#By-column" data-toc-modified-id="By-column-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>By column</a></span></li><li><span><a href="#By-row" data-toc-modified-id="By-row-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>By row</a></span></li><li><span><a href="#Conditional-selections" data-toc-modified-id="Conditional-selections-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Conditional selections</a></span></li><li><span><a href="#Combining-conditions" data-toc-modified-id="Combining-conditions-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Combining conditions</a></span></li><li><span><a href="#Combining-conditions-(2)" data-toc-modified-id="Combining-conditions-(2)-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>Combining conditions (2)</a></span></li><li><span><a href="#Combining-conditions-(3)" data-toc-modified-id="Combining-conditions-(3)-6.6"><span class="toc-item-num">6.6&nbsp;&nbsp;</span>Combining conditions (3)</a></span></li></ul></li><li><span><a href="#Aggregation-and-Grouping" data-toc-modified-id="Aggregation-and-Grouping-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Aggregation and Grouping</a></span><ul class="toc-item"><li><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Column-indexing" data-toc-modified-id="Column-indexing-7.0.0.1"><span class="toc-item-num">7.0.0.1&nbsp;&nbsp;</span>Column indexing</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#Applying-functions" data-toc-modified-id="Applying-functions-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Applying functions</a></span></li><li><span><a href="#Plotting" data-toc-modified-id="Plotting-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Plotting</a></span><ul class="toc-item"><li><span><a href="#Plotting-from-an-IPython-notebook" data-toc-modified-id="Plotting-from-an-IPython-notebook-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Plotting from an IPython notebook</a></span><ul class="toc-item"><li><span><a href="#Matplotlib-options" data-toc-modified-id="Matplotlib-options-9.1.1"><span class="toc-item-num">9.1.1&nbsp;&nbsp;</span>Matplotlib options</a></span></li></ul></li><li><span><a href="#Scatterplot" data-toc-modified-id="Scatterplot-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Scatterplot</a></span></li><li><span><a href="#Histogram" data-toc-modified-id="Histogram-9.3"><span class="toc-item-num">9.3&nbsp;&nbsp;</span>Histogram</a></span></li><li><span><a href="#Boxplot" data-toc-modified-id="Boxplot-9.4"><span class="toc-item-num">9.4&nbsp;&nbsp;</span>Boxplot</a></span></li></ul></li><li><span><a href="#Wrapping-up" data-toc-modified-id="Wrapping-up-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Wrapping up</a></span></li></ul></div>

# ## Motivation

# Most of the material of this lecture is adopted from 

# - https://github.com/LearnDataSci/article-resources 
# - [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) - Essential Tools for Working with Data" By Jake VanderPlas,  O'Reilly Media

# <img src="https://jakevdp.github.io/PythonDataScienceHandbook/figures/PDSH-cover.png" style="width:50%" />

# #### Two ways of exploring data
# 
# <img src="https://github.com/NikoStein/pds_data/raw/main/images/02/explore_data.png" style="width:60%" />

# ### Exploratory data analysis (EDA)

# EDA allows to
# * better understand the data
# * build an intuition about the data
# * generate hypotheses
# * assess assumptions
# * find insights

# With EDA we can
# * get comfortable with the data
# * find magic features
# * find mistakes or odd values

# #### Building intuition about the data

# <img src="https://github.com/NikoStein/pds_data/raw/main/images/02/intuitive.png" style="width:80%" />

# + Is *336* a Typo?
# + Do we misinterpret the feature (*age in years*)?

# ## Introducing Pandas
# 
# The *pandas* package is the most important tool at the disposal of Data Scientists and Analysts working in Python today. The powerful machine learning and glamorous visualization tools may get all the attention, but pandas is the backbone of most data projects. 
# 
# >\[*pandas*\] is derived from the term "**pan**el **da**ta", an econometrics term for data sets that include observations over multiple time periods for the same individuals. — [Wikipedia](https://en.wikipedia.org/wiki/Pandas_%28software%29)
# 
# We will cover the essential bits of information about pandas, including how to install it, its uses, and how it works with other common Python data analysis packages such as **matplotlib**.

# ### Typical Use Cases
# 
# Pandas has so many uses that it might make sense to list the things it can't do instead of what it can do. 
# 
# For example, say you want to explore a dataset stored in a CSV on your computer. Pandas will extract the data from that CSV into a DataFrame — a table, basically — then let you do things like:
# 
# - Calculate statistics and answer questions about the data, like
#     - What's the average, median, max, or min of each column? 
#     - Does column A correlate with column B?
#     - What does the distribution of data in column C look like?
# - Clean the data by doing things like removing missing values and filtering rows or columns by some criteria
# - Visualize the data with help from Matplotlib. Plot bars, lines, histograms, bubbles, and more. 
# - Store the cleaned, transformed data back into a CSV, other file or database
# 
# Before you jump into predictive or prescriptive modeling you need to have a good understanding of the nature of your dataset and pandas is a great avenue through which to do that.
# 

# ### Pandas within the data science toolkit
# 
# Not only is the pandas library a central component of the data science toolkit but it is used in conjunction with other libraries in that collection. 
# 
# Pandas is built on top of the **NumPy** package, meaning a lot of the structure of NumPy is used or replicated in Pandas. Data in pandas is often used to feed statistical analysis in **SciPy**, plotting functions from **Matplotlib**, and machine learning algorithms in **Scikit-learn**.
# 
# Jupyter Notebooks offer a good environment for using pandas to do data exploration and modeling, but pandas can also be used in text editors just as easily.
# 
# Jupyter Notebooks give us the ability to execute code in a particular cell as opposed to running the entire file. This saves a lot of time when working with large datasets and complex transformations. Notebooks also provide an easy way to visualize pandas’ DataFrames and plots.
# 
# 

# ### Installation and Import
# Pandas is an easy package to install. Open up your terminal program (for Mac users) or command line (for PC users) and install it using either of the following commands:
# 
# `conda install pandas`
# 
# OR 
# 
# `pip install pandas`
# 
# Google Colab has pandas pre-installed.
# 
# To import pandas we usually import it with a shorter name since it's used so much:

# In[55]:


import pandas as pd


# ## Series and DataFrames
# 
# The primary two components of pandas are the `Series` and `DataFrame`. 
# 
# A `Series` is essentially a column vector, and a `DataFrame` is a multi-dimensional table made up of a collection of Series. 
# 
# <img src="https://github.com/NikoStein/pds_data/raw/main/images/02/series-and-dataframe.png" style="width:60%" />
# 
# DataFrames and Series are quite similar in that many operations that you can do with one you can do with the other, such as filling in null values and calculating the mean.

# ### Creating DataFrames from scratch
# 
# Creating DataFrames right in Python is good to know and quite useful when testing new methods and functions you find in the pandas docs. There are *many* ways to create a DataFrame from scratch, but a great option is to just use a simple `dict`. 
# 
# Let's say we are working on a fulfillment project. We want to have a row for each customer and one colum for order and delivery amount each. To organize this as a dictionary for pandas we could do something like:

# In[56]:


data = {
    'ordered': [3, 2, 7, 3], 
    'delivered': [0, 2, 4, 2],
}
purchases = pd.DataFrame(data)
purchases


# ### Indexing DataFrames
# 
# Each *(key, value)* item in `data` corresponds to a *column* in the resulting DataFrame. The **Index** of this DataFrame was given to us on creation as the numbers 0-3, but we could also create our own when we initialize the DataFrame. 
# 
# Let's have customer names as our index: 

# In[5]:


purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])
purchases


# So now we could **loc**ate a customer's order status by using their name:

# In[6]:


purchases.loc['June']


# ### Reading and writing CSVs
# It’s quite simple to load data from various file formats into a DataFrame. With CSV files all you need is a single line to load in the data.

# In[57]:


df = pd.read_csv('https://raw.githubusercontent.com/NikoStein/pds_data/main/data/purchases.csv')
df


# After extensive data preparation you likely want to save it as a file of your choice. Similar to the ways we read in data, pandas provides intuitive commands to save it:

# In[7]:


df.to_csv('new_purchases.csv')


# ## Basic DataFrame operations: Viewing
# 
# DataFrames possess hundreds of methods and other operations that are crucial to any analysis. As a beginner, you should know the operations that perform simple transformations of your data and those that provide fundamental statistical analysis.
# 
# Let's load in the IMDB movies dataset to begin. We're loading this dataset from a CSV and designating the movie titles to be our index.

# In[58]:


movies_df = pd.read_csv("https://raw.githubusercontent.com/NikoStein/pds_data/main/data/IMDB-Movie-Data.csv", index_col="Title")


# ### Viewing your data
# 
# The first thing to do when opening a new dataset is print out a few rows to keep as a visual reference.
# 
# Typically when we load in a dataset, we like to view the first five or so rows to see what's under the hood. Here we can see the names of each column, the index, and examples of values in each row.
# 
# `.head()` outputs the **first** five rows of your DataFrame by default, but we could also pass a number as well: `movies_df.head(10)` would output the top ten rows, for example. To see the **last** five rows use `.tail()`.

# In[10]:


movies_df.head(3)


# ### Getting info about your data `.info()`
# 
# `.info()` should be one of the very first commands you run after loading your data. `.info()` provides the essential details about your dataset, such as the number of rows and columns, the number of non-null values, what type of data is in each column, and how much memory your DataFrame is using. 
# 
# Notice in our movies dataset we have some obvious missing values in the `Revenue` and `Metascore` columns. We'll look at how to handle those in a bit.

# In[11]:


movies_df.info()


# ### Getting info about your data: `.shape`
# 
# Another fast and useful attribute is `.shape`, which outputs just a tuple of (rows, columns).

# In[12]:


movies_df.shape


# Note that `.shape` has no parentheses and is a simple tuple of format (rows, columns). So we have **1000 rows** and **11 columns** in our movies DataFrame.
# 
# The `.shape` command is used a lot when cleaning and transforming data. For example, you might filter some rows based on some criteria and then want to know quickly how many rows were removed.

# ### Describing your variables
# Using `describe()` on an entire DataFrame we can get a summary of the distribution of continuous variables. Understanding which numbers are continuous also comes in handy when thinking about the type of plot to use to represent your data visually. 

# In[13]:


movies_df.describe()


# ### Describing Categorical Variables
# `.describe()` can also be used on a categorical variable to get the count of rows, unique count of categories, top category, and freq of top category:

# In[14]:


movies_df['Genre'].describe()


# `.value_counts()` can tell us the frequency of all values in a column:

# In[15]:


movies_df['Genre'].value_counts().head(10)


# ### Relationships between continuous variables
# Examining bivariate relationships comes in handy when you have an outcome or dependent variable in mind and would like to see the features most correlated to the increase or decrease of the outcome. You can visually represent bivariate relationships with scatterplots (seen below in the plotting section). 
# 
# By using the correlation method `.corr()` we can generate the relationship between each continuous variable:

# In[16]:


movies_df.corr()


# ## Basic DataFrame operations: Data Cleaning
# 
# ### Handling duplicates (1)
# Duplicate management is key in most settings involving real data. It is a central data intgegration challenge and we want to be able to perform some basic activities using pandas. This dataset does not have duplicate rows, but it is always important to verify you aren't aggregating duplicate rows. 
# 
# To demonstrate, let's simply just double up our movies DataFrame by appending it to itself. Using `append()` will return a copy without affecting the original DataFrame. We are capturing this copy in `temp` so we aren't working with the real data:

# In[17]:


temp_df = movies_df.append(movies_df)
temp_df.shape


# Now we can try dropping duplicates. Just like `append()`, the `drop_duplicates()` method will also return a copy of your DataFrame, but this time with duplicates removed. Calling `.shape` confirms we're back to the 1000 rows of our original dataset.

# In[18]:


temp_df = temp_df.drop_duplicates()
temp_df.shape


# ### Handling duplicates (2)
# 
# Another important argument for `drop_duplicates()` is `keep`, which has three possible options:
# * `first`: (default) Drop duplicates except for the first occurrence.
# * `last`: Drop duplicates except for the last occurrence.
# * `False`: Drop all duplicates.
# 
# Since we didn't define the `keep` arugment in the previous example it was defaulted to `first`. This means that if two rows are the same pandas will drop the second row and keep the first row. Watch what happens to `temp_df`:

# In[19]:


temp_df = movies_df.append(movies_df)  # make a new copy
temp_df = temp_df.drop_duplicates(keep=False)
temp_df.shape


# Since all rows were duplicates, `keep=False` dropped them all resulting in zero rows being left over. If you're wondering why you would want to do this, one reason is that it allows you to locate all duplicates in your dataset. When conditional selections are shown below you'll see how to do that.

# ### Inplace Operations
# 
# It's a little verbose to keep assigning DataFrames to the same variable like in this example. For this reason, pandas has the `inplace` keyword argument on many of its methods. Using `inplace=True` will modify the DataFrame object in place:

# In[20]:


temp_df = movies_df.append(movies_df)  # make a new copy

temp_df.drop_duplicates(inplace=True, keep=False)

temp_df.shape


# ### Column cleanup
# 
# Many times datasets will have verbose column names with symbols, upper and lowercase words, spaces, and typos. To make selecting data by column name easier we can spend a little time cleaning up their names. Here's how to print the column names of our dataset:

# In[21]:


movies_df.columns


# Not only does `.columns` come in handy if you want to rename columns by allowing for simple copy and paste, it's also useful if you need to understand why you are receiving a `Key Error` when selecting data by column.

# ### Renaming Columns
# 
# We can use the `.rename()` method to rename certain or all columns via a `dict`. We don't want parentheses, so let's rename those:

# In[22]:


movies_df.rename(columns={
        'Runtime (Minutes)': 'Runtime', 
        'Revenue (Millions)': 'Revenue_millions'
    }, inplace=True)
movies_df.columns


# Excellent. But what if we want to lowercase all names? Instead of using `.rename()` we could also set a list of names to the columns like so:

# In[23]:


movies_df.columns = ['rank', 'genre', 'description', 'director', 'actors', 'year', 'runtime', 
                     'rating', 'votes', 'revenue_millions', 'metascore']
movies_df.columns


# ### Operating on many columns
# With thousands of columns the previous approach is too much work. Instead of just renaming each column manually we can do a list comprehension:

# In[24]:


movies_df.columns = [col.lower() for col in movies_df]

movies_df.columns


# `list` (and `dict`) comprehensions come in handy a lot when working with pandas and data in general.
# 
# It's a good idea to lowercase, remove special characters, and replace spaces with underscores if you'll be working with a dataset for some time.

# ### Working with missing values
# 
# When exploring data, you’ll most likely encounter missing or null values, which are essentially placeholders for non-existent values. Most commonly you'll see Python's `None` or NumPy's `np.nan`, each of which are handled differently in some situations.
# 
# There are two options in dealing with nulls: 
# 
# 1. Get rid of rows or columns with nulls
# 2. Replace nulls with non-null values, a technique known as **imputation**
# 
# Let's calculate to total number of nulls in each column of our dataset. The first step is to check which cells in our DataFrame are null:

# ### Identify missing values
# `isnull()` returns a DataFrame where each cell is either True or False depending on that cell's null status.
# 

# In[25]:


movies_df.isnull().head(5)


# `.isnull()` just by iteself isn't very useful, and is usually used in conjunction with other methods, like `sum()`. To count the number of nulls in each column we use an aggregate function for summing: 

# In[26]:


movies_df.isnull().sum()


# ### Removing null values
# 
# Data Scientists and Analysts regularly face the dilemma of dropping or imputing null values, and is a decision that requires intimate knowledge of your data and its context. Overall, removing null data is only suggested if you have a small amount of missing data.
# 
# Remove nulls is pretty simple:

# In[27]:


df_clean = movies_df.dropna()
df_clean.shape


# This operation deletes any **row** with at least a single null value. In our case, it removes 128 rows where `revenue_millions` is null and 64 rows where `metascore` is null. This obviously seems like a waste since there's perfectly good data in the other columns of those dropped rows. That's why we'll look at imputation next.

# ### Imputation
# 
# Imputation is a conventional feature engineering technique used to keep valuable data that have null values. There may be instances where dropping every row with a null value removes too big a chunk from your dataset, so instead we can impute that null with another value, usually the **mean** or the **median** of that column. 
# 
# Let's look at imputing the missing values in the `revenue_millions` column:

# In[28]:


revenue = movies_df['revenue_millions']


# Using square brackets is the general way we select columns in a DataFrame. `revenue` is a Series and we can calculate its mean and fill the nulls using `fillna()`:

# In[29]:


revenue_mean = revenue.mean()
print(revenue_mean)
revenue.fillna(revenue_mean, inplace=True)


# We have now replaced all nulls in `revenue` with the mean of the column:

# In[30]:


movies_df.isnull().sum()


# Imputing an entire column with the same value like this is a basic example. It would be a better idea to try a more granular imputation by Genre or Director. For example, you would find the mean of the revenue generated in each genre individually and impute the nulls in each genre with that genre's mean.
# 

# ## Basic DataFrame operations: Slicing, selecting, extracting
# 
# Up until now we've focused on some basic summaries of our data. We've learned about simple column extraction using single brackets, and we imputed null values in a column using `fillna()`. Below are the other methods of slicing, selecting, and extracting you'll need to use constantly.
# 
# It's important to note that, although many methods are the same, DataFrames and Series have different attributes, so you'll need be sure to know which type you are working with or else you will receive attribute errors. 
# 
# Let's look at working with columns first.

# ### By column
# 
# You already saw how to extract a column using square brackets like this:

# In[31]:


genre_col = movies_df['genre']
type(genre_col)


# This will return a *Series*. To extract a column as a *DataFrame*, you need to pass a list of column names:

# In[32]:


genre_col = movies_df[['genre']]
type(genre_col)


# Since it's just a list, adding another column name is easy:

# In[33]:


subset = movies_df[['genre', 'rating']]
subset.head()


# ### By row
# For rows, we have two options: 
# - `.loc` - **loc**ates by name
# - `.iloc`- **loc**ates by numerical **i**ndex

# In[34]:


prom = movies_df.loc["Prometheus"]
prom[{"description","director"}]


# With `iloc` we give it the numerical index of Prometheus:

# In[35]:


prom = movies_df.iloc[1]
prom[{"description","director"}]


# ### Conditional selections
# We’ve gone over how to select columns and rows, but what if we want to make a conditional selection? 
# 
# For example, what if we want to filter our movies DataFrame to show only films directed by Ridley Scott or films with a rating greater than or equal to 8.0?
# 
# To do that, we take a column from the DataFrame and apply a Boolean condition to it. Here's an example of a Boolean condition:

# In[36]:


movies_df[movies_df['director'] == "Ridley Scott"].head()


# ### Combining conditions
# We can make some richer conditionals by using logical operators `|` for "or" and `&` for "and".
# 
# Let's filter the the DataFrame to show only movies by Christopher Nolan OR Ridley Scott:

# In[37]:


movies_df[(movies_df['director'] == 'Christopher Nolan') | (movies_df['director'] == 'Ridley Scott')].head()


# ### Combining conditions (2)
# We need to make sure to group evaluations with parentheses so Python knows how to evaluate the conditional.
# 
# `(movies_df['director'] == 'Christopher Nolan') | (movies_df['director'] == 'Ridley Scott')`
# 
# Using the `isin()` method we could make this more concise:

# In[38]:


selection = ['Christopher Nolan', 'Ridley Scott']
movies_df[movies_df['director'].isin(selection)].head()


# ### Combining conditions (3)
# Let's say we want all movies that were released between 2005 and 2010, have a rating above 8.0, but made below the 25th percentile in revenue.

# In[39]:


movies_df[
    ((movies_df['year'] >= 2005) & (movies_df['year'] <= 2010))
    & (movies_df['rating'] > 8.0)
    & (movies_df['revenue_millions'] < movies_df['revenue_millions'].quantile(0.25))
]


# ## Aggregation and Grouping
# An essential piece of analysis of large data is efficient summarization: computing aggregations like ``sum()``, ``mean()``, ``median()``, ``min()``, and ``max()``, in which a single number gives insight into the nature of a potentially large dataset.
# In this section, we'll explore aggregations in Pandas, from simple operations akin to what we've seen on NumPy arrays, to more sophisticated operations based on the concept of a ``groupby``.

# The following table summarizes some other built-in Pandas aggregations:
# 
# | Aggregation              | Description                     |
# |--------------------------|---------------------------------|
# | ``count()``              | Total number of items           |
# | ``first()``, ``last()``  | First and last item             |
# | ``mean()``, ``median()`` | Mean and median                 |
# | ``min()``, ``max()``     | Minimum and maximum             |
# | ``std()``, ``var()``     | Standard deviation and variance |
# | ``mad()``                | Mean absolute deviation         |
# | ``prod()``               | Product of all items            |
# | ``sum()``                | Sum of all items                |
# 
# These are all methods of ``DataFrame`` and ``Series`` objects.

# Let's calculate some statistics 

# In[40]:


movies_df.std()


# In[41]:


movies_df['revenue_millions'].median()


# ###### GroupBy: Split, Apply, Combine
# Simple aggregations can give you a flavor of your dataset, but often we would prefer to aggregate conditionally on some label or index: this is implemented in the so-called groupby operation. The name "group by" comes from a command in the SQL database language, but it is perhaps more illuminative to think of it in the terms first coined by Hadley Wickham of Rstats fame: split, apply, combine.
# 
# <img src="https://github.com/jakevdp/PythonDataScienceHandbook/raw/master/notebooks/figures/03.08-split-apply-combine.png" style="width:60%" />

# ##### Column indexing
# 
# The ``GroupBy`` object supports column indexing in the same way as the ``DataFrame``, and returns a modified ``GroupBy`` object.
# For example:

# In[42]:


movies_df.groupby('genre')


# In[43]:


movies_df.groupby('genre')['revenue_millions']


# In[44]:


movies_df.groupby('genre')['revenue_millions'].median().sort_values(ascending=False)


# ## Applying functions
# 
# It is possible to iterate over a DataFrame or Series as you would with a list, but doing so — especially on large datasets — is very slow.
# 
# An efficient alternative is to `apply()` a function to the dataset. For example, we could use a function to convert movies with an 8.0 or greater to a string value of "good" and the rest to "bad" and use this transformed values to create a new column.
# 
# First we would create a function that, when given a rating, determines if it's good or bad:

# In[45]:


def rating_function(x):
    if x >= 8.0:
        return "good"
    else:
        return "bad"


# Now we want to send the entire rating column through this function, which is what `apply()` does:

# In[46]:


movies_df["rating_category"] = movies_df["rating"].apply(rating_function)
movies_df.head(2)


# Besides being much more concise than a loop structure, using `apply()` will also be much faster than iterating manually over rows because pandas is utilizing vectorization.

# ## Plotting
# 
# Another great thing about pandas is that it integrates with Matplotlib, so you get the ability to plot directly off DataFrames and Series. To get started we need to import Matplotlib:

# In[47]:


import matplotlib.pyplot as plt


# ### Plotting from an IPython notebook
# 
# Plotting interactively within an IPython notebook can be done with the ``%matplotlib`` command, and works in a similar way to the IPython shell.
# In the IPython notebook, you also have the option of embedding graphics directly in the notebook, with two possible options:
# 
# - ``%matplotlib notebook`` will lead to *interactive* plots embedded within the notebook
# - ``%matplotlib inline`` will lead to *static* images of your plot embedded in the notebook
# 
# We will generally opt for ``%matplotlib inline``

# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')


# A simple plot

# In[49]:


import numpy as np
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');


# #### Matplotlib options

# If you are using Matplotlib from within a script, the function ``plt.show()`` is your friend.
# ``plt.show()`` starts an event loop, looks for all currently active figure objects, and opens one or more interactive windows that display your figure or figures.

# In[50]:


plt.rcParams.update({'font.size': 20, 'figure.figsize': (10, 8)}) # set font and plot size to be larger


# Now we can begin. There won't be a lot of coverage on plotting, but it should be enough to explore you're data easily.
# 
# **Side note:**
# For categorical variables utilize Bar Charts* and Boxplots.  For continuous variables utilize Histograms, Scatterplots, Line graphs, and Boxplots.

# ### Scatterplot
# 
# Let's plot the relationship between ratings and revenue. All we need to do is call `.plot()` on `movies_df` with some info about how to construct the plot:

# In[51]:


movies_df.plot(kind='scatter', x='rating', y='revenue_millions', title='Revenue (millions) vs Rating');


# The semicolon is not a syntax error, just a way to hide extra output in Jupyter notebooks.

# ### Histogram
# If we want to plot a simple Histogram based on a single column, we can call plot on a column:

# In[52]:


movies_df['rating'].plot(kind='hist', title='Rating');


# ### Boxplot
# Using a Boxplot we can visualize the rating quartiles of the n directors with the most movies:

# In[53]:


n=10
directors = movies_df['director'].value_counts()[:n].index.tolist()
chart = movies_df[movies_df['director'].isin(directors)].boxplot(column=['rating'], by="director")
chart.set_xticklabels(chart.get_xticklabels(), rotation=90);


# ## Wrapping up
# 
# Exploring, cleaning, transforming, and visualization data with pandas in Python is an essential skill in data science. Just cleaning wrangling data is 80% of your job as a Data Scientist. After a few projects and some practice, you should be very comfortable with most of the basics.
# 
# To keep improving, view the [extensive tutorials](https://pandas.pydata.org/pandas-docs/stable/tutorials.html) offered by the official pandas docs, follow along with a few [Kaggle kernels](https://www.kaggle.com/kernels), and keep working on your own projects!
