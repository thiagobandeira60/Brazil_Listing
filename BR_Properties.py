# %% markdown
# # Brazil Properties Cleaning Project
# %% markdown
# # Inital Exploratory Analysis
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
br1 = pd.read_csv('brazil_properties_rent.csv')
# %%
br2 = pd.read_csv('brazil_properties_sell.csv')
# %%
br1.head(3)
# %%
br2.head(3)
# %%
br1.columns
# %%
br2.columns
# %%
br1.shape
# %%
br2.shape
# %%
br1.dtypes
# %%
br2.dtypes
# %%
br1.isnull().sum()
# %%
br2.isnull().sum()
# %%
51323 * 100 / 97353
# %%
# The 2 datasets ar similar, except for the location column in the rent data. When we merge the 2 datasets the location column,
# that already has over 52% of missing values in the smaller dataset, will have over 90% of missing data.
# Therefore, we are dropping that column and concatenating both datasets.
# %%
br1 = br1.drop(['location'], axis=1)
# %%
br1.shape
# %%
# Noticed that the 'lat_lon' column in the br2 dataset is named as 'lat-lon'
br2.rename(columns = {'lat-lon': 'lat_lon'}, inplace=True)
# %%
br_full = pd.concat([br1, br2], sort=False)
# %%
br_full.head(3)
# %%
br_full.shape
# %%
br_full.columns
# %%
br_full.info()
# %%
# The 2 datasets were successfully concatenated.
# %%
br_full.dtypes
# %%
br_full.isnull().sum()
# %%
# Let's analyse some columns individually to see if they should be dropped or not.
# %%
# The column created on needs to be changed to date type.
# %%
br_full.geonames_id.describe()
# %%
# The column geonames_id contains, basically, only missing values. Therefore, it will be dropped.
# %%
br_full.lat_lon.describe()
# %%
br_full['lat_lon'].sample(10)
# %%
br_full.lat.loc[816556]
# %%
br_full['lat_lon'].nunique()
# %%
br_full['lat'].nunique()
# %%
br_full['lon'].nunique()
# %%
# Since we have 2 distinct columns for latitude and longitude, and don't need the lat_log column that only gives us a string.
# The lat_long column will be dropped.
# %%
br_full.currency.sample(10)
# %%
br_full.currency.unique()
# %%
br_full.currency.nunique()
# %%
# Currency needs to be standardized. We should probably create Currancy in R$ and in USD dollars.
# %%
br_full.operation.unique()
# %%
br_full.property_type.unique()
# %%
br_full.place_with_parent_names.sample(10)
# %%
br_full.place_with_parent_names.describe()
# %%
# The column place_name contains the city names for each observation. The column place_with_parent_names contains more information
# about the location. It would be interesting to have the states as well. We will extract the state information and put it
# in a new column.
# %%
br_full.price.describe()
# %%
# The column price seems to have some weird values. We have properties costing over 100,000,000.00 and others costing 0.
# We need to investigate the outliers and eliminate those that don't make sense.
# %%
br_full.price_aprox_local_currency.sample(10)
# %%
br_full.price.loc[[356239, 644653, 150757, 192553, 44493, 849261, 6270, 40369, 173015, 78318]]
# %%
br_full.loc[44493]
# %%
# There are some observations that have the same id but they are different. It's due to the concatenation at the beggining.
# In this case, we need to reindex the dataset.
# %%
br_full = br_full.reset_index(drop=True)
# %%
br_full.shape
# %%
br_full.loc[44493]
# %%
# Now it looks like every entry has its own index number.
# %%
# Not sure about what the price_aprox_local_currency column means, bit it looks like it's an aproximation to the actual price in
# the local currency. The price_aprox_usd is probably the same price, but in dollars.
# The dataset doesn't contains a code-book.
# %%
br_full.surface_total_in_m2.describe()
# %%
br_full.surface_covered_in_m2.describe()
# %%
# As the price column, the surface_total_in_m2 and the surface_covered_in_m2 columns need to be investigated for outliers.
# All the price columns contain a significant number of outliers, but they also contain crucial information for analysis.
# Therefore, they shouldn't be dropped.
# %%
br_full.price_per_m2.describe()
# %%
br_full.floor.unique()
# %%
# The variable floor has over 90% of missing data. Based on that, the floor column will be dropped.
# %%
br_full.rooms.sample(10)
# %%
br_full.rooms.describe()
# %%
# The variable rooms needs to be investigated for outliers. That column contains a lot of missing values, but the information
# is relevant and will be used for analysis. Therefore, the column won't be dropped, but the missing values will be imputed.
# %%
br_full.expenses.sample(10)
# %%
br_full.expenses.describe()
# %%
br_full.shape
# %%
704934 * 100 / 970353
# %%
# The column expense has over 72% of missing data. It will be dropped since it also doesn't seems to contain relevant information.
# %%
br_full.description.sample(5)
# %%
# The description column needs to be further investigated.
# %%
br_full.columns
# %%
# The title and image_thumnail columns will contribuit too little for the analysis, thus they will be dropped.
# %%
# Summary of what needs to be done in order to clean the data:

# 1. The first thing is to drop the columns that will not help the analysis since they either contain too many missing values
# or irrelevant information. The columns are: geonames_id, lat_lon, floor, expenses, properati_url, title, and image_thumbnail.

# 2. The column created_on needs to be changed to date type. It will be good to also extract the year, month, and name of the
# month and create 3 more columns with that information for future analysis.

# 3. Currency needs to be standardized. We should probably create Currancy in R$ and in USD dollars.

# 4. The column place_name contains the city names for each observation. The column place_with_parent_names contains more information
# about the location. It would be interesting to have the states as well. We will extract the state information and put it
# in a new column.

# 5. Not sure about what the price_aprox_local_currency column means, but it looks like it's an aproximation to the actual price in
# the local currency. The price_aprox_usd is probably the same price, but in dollars.
# The dataset doesn't contains a code-book.

# 6. Some columns need to be further investigated for outliers: price, price_aprox_local_currency, price_aprox_usd,
# surface_total_in_m2, surface_covered_in_m2, price_usd_per_m2, price_per_m2, rooms.

# 7. The description column needs to be further investigated.

# 8. We need to deal with missing data and imputation.
# %%
br_full.to_csv('br_full.csv')
# %% markdown
# # Brazil Properties Cleaning Practice
# %% markdown
# # Actual Cleaning
# %%
# Summary of what needs to be done in order to clean the data:

# 1. The first thing is to drop the columns that will not help the analysis since they either contain too many missing values
# or irrelevant information. The columns are: geonames_id, lat_lon, floor, expenses, properati_url, title, and image_thumbnail.

# 2. The column created_on needs to be changed to date type. It will be good to also extract the year, month, and name of the
# month and create 3 more columns with that information for future analysis.

# 3. Currency needs to be standardized. We should probably create Currancy in R$ and in USD dollars.

# 4. The column place_name contains the city names for each observation. The column place_with_parent_names contains more information
# about the location. It would be interesting to have the states as well. We will extract the state information and put it
# in a new column.

# 5. Not sure about what the price_aprox_local_currency column means, but it looks like it's an aproximation to the actual price in
# the local currency. The price_aprox_usd is probably the same price, but in dollars.
# The dataset doesn't contains a code-book.

# 6. Some columns need to be further investigated for outliers: price, price_aprox_local_currency, price_aprox_usd,
# surface_total_in_m2, surface_covered_in_m2, price_usd_per_m2, price_per_m2, rooms.

# 7. The description column needs to be further investigated.

# 8. We need to deal with missing data and imputation.
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
# %%
br_full = pd.read_csv('br_full.csv')
# %%
br_full.head(3)
# %%
br_full.columns
# %%
# When I saved the file, in the other ipython notebook, I created, aacidentally, and new index column.
# Now I'm removing that column.
# %%
br_full = br_full.drop(columns=['Unnamed: 0'], axis=1)
# %%
br_full.columns
# %%
br_full.dtypes
# %%
br_full.isnull().sum()
# %%
br_full.shape
# %% markdown
# ## 1. Dropping the columns
# %%
br_full.info()
# %%
br_messy = br_full.drop(columns = ['geonames_id', 'lat_lon', 'floor', 'expenses', 'properati_url',
                                   'title', 'image_thumbnail'], axis=1)
# %%
br_messy.info()
# %% markdown
# ## 2. Changing the column 'created_on' to the datetime format and creating the additional columns
# %%
br_messy['created_on'] = pd.to_datetime(br_messy['created_on'])
# %%
br_messy.info()
# %%
br_messy.head(1)
# %%
br_messy['year'] = pd.DatetimeIndex(br_messy['created_on']).year
# %%
br_messy.head(2)
# %%
br_messy['month'] = pd.DatetimeIndex(br_messy['created_on']).month
# %%
br_messy.head(2)
# %%
br_messy['month_name'] = br_messy['month'].apply(lambda x: calendar.month_abbr[x])
# %%
br_messy.head(2)
# %% markdown
# ## 3. Dealing with Currency column
# %%
br_messy[['operation', 'currency', 'price', 'price_aprox_local_currency', 'price_aprox_usd']].sample(15)
# %%
br_messy['currency'].unique()
# %%
br_messy.groupby('currency').count()
# %%
242 * 100 / 970025
# %%
type(br_messy['currency'])
# %%
# Invetigating a little further, we found out that the observations containing MXN and USD currencies are only about
# 0.02% of the data, so intead of standardizing it, we are just dropping those rows and reindexing.
# %%
br_messy.drop(br_messy[br_messy.currency == 'MXN'].index, inplace=True)
# %%
br_messy.drop(br_messy[br_messy.currency == 'USD'].index, inplace=True)
# %%
br_messy.groupby('currency').count()
# %%
br_messy.currency.isnull().sum()
# %% markdown
# ## 4. Getting the State column
# %%
br_messy.shape
# %%
br_messy.place_with_parent_names.sample(10)
# %%
br_messy.place_with_parent_names.tail(10)
# %%
br_messy.place_with_parent_names.head(10)
# %%
# As we can see, the second argument is the state. That's the argument we need.
# %%
type(br_messy.place_with_parent_names)
# %%
br_messy['PWPN_string'] = br_messy.place_with_parent_names.str.split('|')
# %%
br_messy.PWPN_string.head()
# %%
br_messy['State'] = br_messy.PWPN_string.str.get(2)
# %%
br_messy.State.head()
# %%
br_messy.State.sample(60)
# %%
br_messy.State.unique()
# %%
# Now we are all good, except that we have a value(s) that doesn't make sense: Miami.
# Therefore, we should investigate it further.
# %%
br_messy.loc[br_messy['State'] == 'Miami', :]
# %%
# We found 143 observations that don't make sense, since that describe houses or apartments in the US and not in Brazil.
# Since we are talking about only 143 observations, we can just drop them.
# %%
br_messy.drop(br_messy[br_messy.State == 'Miami'].index, inplace=True)
# %%
br_messy.State.unique()
# %%
# Now we don't have Miami anymore.
# %%
# To finish, we'll keep the place_with_parent_names column (since it may be useful in the future), but we'll drop the
# column PWPN_string we created.
# %%
br_messy.drop(columns = 'PWPN_string', axis=1, inplace=True)
# %%
br_messy.columns
# %%
br_messy = br_messy.reset_index(drop=True)
# %%
br_messy.head()
# %% markdown
# ## 5. Investigating the 'aprox' columns
# %%
subset = br_messy.loc[:, ['price', 'price_aprox_local_currency', 'price_aprox_usd']]
# %%
subset.sample(15)
# %%
dollar_price = br_messy['price_aprox_local_currency'] / br_messy['price_aprox_usd']
# %%
dollar_price.sample(10)
# %%
# The dollar price is around 3.20 during that period.
# It seems our suspects were almost right. The column price_aprox_local_currency is the actual price in local currency, the column
# price_aprox_usd is the same value in dollars, and the column price is an aproximation.
# %%
# Since that's the case, we can leave the columns there. They can be used later for analysis.
# %% markdown
# ## 6. Investigating for Outliers
# %%
# Some columns need to be further investigated for outliers: price, price_aprox_local_currency, price_aprox_usd,
# surface_total_in_m2, surface_covered_in_m2, price_usd_per_m2, price_per_m2, rooms.
# %%
# Let's first create some boxplots to have a general idea of the outliers.
# %%
sns.boxplot(br_messy['price'], orient = 'v')
plt.gcf().set_size_inches(15, 8)
# %%
sns.scatterplot(y = br_messy['price'], x = br_messy['State'])
plt.gcf().set_size_inches(15, 8)
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
plt.show()
# %%
# The columns price, price_aprox_local_currency, and price_aprox_usd basically show the same amounts, so the changes
# we'll perform in one of the will serve to the others as well.
# %%
# Let's see the observations where the price is 60 million or more make sense.
# %%
br_messy.loc[(br_messy['price'] >= 60000000), :]
# %%
test = br_messy.loc[(br_messy['price'] >= 60000000), :]
# %%
test.describe()
# %%
br_messy.loc[(br_messy['price'] >= 60000000), :].count()
# %%
test[['price', 'description']].sample(15)
# %%
# There are just about 57 observations where the house's prices are more than 60 million. By looking at the description
# of a random sample of them, it's easy to notice that most of the observations don't make sense.
# Therefore, let's drop those rows and take care of those 3 columns.
# %%
br_messy.drop(br_messy[br_messy.price >= 60000000].index, inplace=True)
# %%
sns.scatterplot(y = br_messy['price'], x = br_messy['State'])
plt.gcf().set_size_inches(15, 8)
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
plt.show()
# %%
# Now let's analyse the surface_total_in_m2 and surface_covered_in_m2 columns.
# Let's try to find some erroneous values by looking if we have observations where the covered area is greater than the
# total area, which would be impossible.
# %%
br_messy[['surface_total_in_m2', 'surface_covered_in_m2']].sample(10)
# %%
br_messy.loc[br_messy['surface_total_in_m2'] < br_messy['surface_covered_in_m2']]
# %%
# We have 23,203 rows where the covered area is greater than the total area, which doen't make any sense. Just by
# investigating some of them, we can see that there are some erroneous values, such as row 968,861 where the covered area is 160 m2
# and the total area is 0.
# We will, then, drop those rows.
# %%
test2 = br_messy.loc[br_messy['surface_total_in_m2'] < br_messy['surface_covered_in_m2']]
# %%
test2.head()
# %%
test2.shape
# %%
br_almost = br_messy.drop(br_messy[br_messy['surface_total_in_m2'] < br_messy['surface_covered_in_m2']].index)
# %%
br_almost.shape
# %%
br_almost.loc[br_almost['surface_total_in_m2'] < br_almost['surface_covered_in_m2']]
# %%
# As we can see, we got rid of those erroneous observations.
# %%
br_almost.info()
# %%
# Let's see if the price_per_m2 column makes sense by multiplying it with the surface covered and comparing the result with
# the price column.
# %%
br_almost['price_calc'] = br_almost['surface_covered_in_m2'] * br_almost['price_per_m2']
# %%
br_almost[['price', 'price_calc', 'surface_total_in_m2', 'surface_covered_in_m2', 'price_per_m2']].sample(10)
# %%
sns.set()
sns.scatterplot(x = 'price', y = 'price_calc', markers='o', data=br_almost)
plt.xlabel('price')
plt.ylabel('calculated price')
plt.gcf().set_size_inches(15, 8)
plt.show()
# %%
# Since the relationship is linear, we can assume that the columns surface_covered_in_m2, price_per_m2, and price_usd_per_m2
# do not contain mistakes.
# Now we can drop the price_calc column.
# %%
br_almost = br_almost.drop(columns = ['price_calc'], axis=1)
# %%
br_almost.shape
# %%
# The last thing is to analyze the rooms column.
# %%
br_almost.rooms.describe()
# %%
sns.scatterplot(y = br_almost['rooms'], x = br_messy['State'])
plt.gcf().set_size_inches(15, 8)
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
plt.show()
# %%
# Let's analyze the observations that contain over 10 rooms and then over 20 rooms.
# %%
br_almost.loc[br_almost['rooms'] >= 10]
# %%
br_almost.loc[br_almost['rooms'] >= 20]
# %%
# There are just a few thousand rows that contain more than 10 rooms, and just a few hundred containing more than
# 20 rooms. However, by inspecting some of the observations, it's possible to see that these are outliers but make sense.
# Therefore, they won't be dropped.
# %% markdown
# ## 7. Description Column
# %%
br_almost.description.describe()
# %%
br_almost.description.sample(10)
# %%
# Although I don't think the description column will be useful for analysis, we can leave it for now.
# The description column will not be dropped right now.
# %%
# Now, we have only to deal with the missing data.
# %%
br_almost.to_csv('br_almost.csv', index=False)
# %% markdown
# # Brazil Properties Cleaning Practice
# %% markdown
# # Actual Cleaning - Missing Data
# %%
# Summary of what needs to be done in order to clean the data:

# 1. The first thing is to drop the columns that will not help the analysis since they either contain too many missing values
# or irrelevant information. The columns are: geonames_id, lat_lon, floor, expenses, properati_url, title, and image_thumbnail.

# 2. The column created_on needs to be changed to date type. It will be good to also extract the year, month, and name of the
# month and create 3 more columns with that information for future analysis.

# 3. Currency needs to be standardized. We should probably create Currancy in R$ and in USD dollars.

# 4. The column place_name contains the city names for each observation. The column place_with_parent_names contains more information
# about the location. It would be interesting to have the states as well. We will extract the state information and put it
# in a new column.

# 5. Not sure about what the price_aprox_local_currency column means, but it looks like it's an aproximation to the actual price in
# the local currency. The price_aprox_usd is probably the same price, but in dollars.
# The dataset doesn't contains a code-book.

# 6. Some columns need to be further investigated for outliers: price, price_aprox_local_currency, price_aprox_usd,
# surface_total_in_m2, surface_covered_in_m2, price_usd_per_m2, price_per_m2, rooms.

# 7. The description column needs to be further investigated.

# 8. We need to deal with missing data and imputation.
# %% markdown
# ## 8. Dealing with Missing Data
# %%
# This is the last step of the cleaning process where we'll deal with missing data.
# There are many approaches regarding missing data, and we will investigate further to see which ones suit our dataset and
# columns better.
# %%
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer
import fancyimpute
from fancyimpute import KNN
from fancyimpute import IterativeImputer
# %%
br_almost = pd.read_csv('br_almost.csv')
# %%
br_almost.head()
# %%
br_almost.info()
# %%
# Before start working with missing data, let's first reorder our columns.We don't need the place_with_parent_names anymore.
# Since we standardized the currency, we won't need that column anymore.
# %%
br_almost = br_almost[['created_on', 'year', 'month', 'month_name', 'place_name', 'State', 'price', 'price_aprox_local_currency',
                       'price_aprox_usd', 'surface_total_in_m2', 'surface_covered_in_m2', 'price_usd_per_m2',
                       'price_per_m2', 'rooms', 'operation', 'property_type', 'lat', 'lon', 'description']]
# %%
br_almost.head()
# %%
br_almost.info()
# %%
# Now we can start dealing with the missing data.
# %%
# We need to have a general idea of the missing values before actually cleaning them.
# %%
msno.matrix(br_almost)
# %%
# Our approach here will be to see the the data distribution and to find the imputation technique that is
# closest to that distribuition. We will also compare R-squared and coefficients.
# %%
# We'll start by making some basic imputations:
# 1. mean imputation
# 2. median imputation
# 3. mode imputation
# 4. constant imputation (0)
# %% markdown
# ### 1. The mean imputation
# %%
# Before starting, we will separate the dataset in 3 different datasets: 1 containing non-numeric data, 1 containg the
# columns from price to rooms, and another one containing lat and lon columns.
# The reason is that we are gonna try a different method of imputation for the coordenates and we don't want non-numeric data
# on the imputation tests, so it doesn't give us an error message.
# After the best method is decided and the imputations are done, we will then concatenate the datasets.
# %%
df_1 = br_almost[['created_on', 'year', 'month', 'month_name', 'place_name', 'State', 'operation', 'property_type',
                  'description']]

df_2 = br_almost[['price', 'price_aprox_local_currency','price_aprox_usd', 'surface_total_in_m2',
                  'surface_covered_in_m2', 'price_usd_per_m2', 'price_per_m2', 'rooms']]

df_3 = br_almost[['lat', 'lon']]
# %%
df_2_mean = df_2.copy(deep=True)
# %%
mean_imputer = SimpleImputer(strategy = 'mean')
# %%
df_2_mean.iloc[:, :] = mean_imputer.fit_transform(df_2_mean)
# %%
df_2_mean.head()
# %% markdown
# ### 2. The median imputation
# %%
df_2_median = df_2.copy(deep=True)
median_imputer = SimpleImputer(strategy = 'median')
df_2_median.iloc[:, :] = median_imputer.fit_transform(df_2_median)
df_2_median.head()
# %% markdown
# ### 3. The mode imputation
# %%
df_2_mode = df_2.copy(deep=True)
mode_imputer = SimpleImputer(strategy = 'most_frequent')
df_2_mode.iloc[:, :] = mode_imputer.fit_transform(df_2_mode)
df_2_mode.head()
# %% markdown
# ### 4. The constant imputation
# %%
df_2_constant = df_2.copy(deep=True)
constant_imputer = SimpleImputer(strategy = 'constant', fill_value = 0)
df_2_constant.iloc[:, :] = constant_imputer.fit_transform(df_2_constant)
df_2_constant.head()
# %% markdown
# ### Now let's visualize result distribution from those imputation methods
# %%
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
nullity = df_2['price'].isnull() + df_2['rooms'].isnull()
imputations = {'Mean Imputation': df_2_mean, 'Median Imputation': df_2_median,
               'Most Frequent Imputation': df_2_mode, 'Constant Imputation': df_2_constant}

for ax, df_key in zip(axes.flatten(), imputations):
    imputations[df_key].plot(x = 'price', y = 'rooms', kind = 'scatter',
                             alpha = 0.5, c = nullity, cmap = 'rainbow', ax = ax,
                             colorbar = False, title = df_key)
# %%
# As we can see from the charts above, any of the imputation methods we tried will work well.
# The read line depicts the imputed values, and they don't follow the pattern of the data.
# We'll need to try some more advanced techniques such as:

# KNN imputation
# However, since the dataset is very large, KNN approach is not possible.

# Mice imputation
# %% markdown
# ### MICE Imputation
# %%
df_2_mice = df_2.copy(deep=True)
mice_imputer = IterativeImputer()
df_2_mice.iloc[:, :] = mice_imputer.fit_transform(df_2_mice)
# %%
df_2_mice.head()
# %%
# Now we need to focus on the latitude and longitude imputations.
# %%
# Even by splitting the data in 3 distinct datasets, it's not possible to use KNN imputation for latitude and longitude
# duo to the dataset size and hardware limitations.
# Furthermore, the columns have over 50% of missing data, so the best approach in this case is to drop them.
# %%
# We will evaluate which model performed best on df_2 imputations, select it, and concatenate the datasets df_1 and df_2.
# %%
# Plot graphs of imputed DataFrames and the complete case
df_2['price'].plot(kind='kde', c='red', linewidth=3)
df_2_mean['price'].plot(kind='kde')
df_2_median['price'].plot(kind='kde')
df_2_mode['price'].plot(kind='kde')
df_2_constant['price'].plot(kind='kde')
df_2_mice['price'].plot(kind='kde')

# Create labels for the six DataFrames
labels = ['Baseline (Complete Case)', 'Mean Imputation', 'Median Imputation', 'Mode Imputation',
          'Constant Imputation', 'MICE Imputation']
plt.legend(labels)

# Setting the x-label as 'price'
plt.xlabel('Price')

plt.gcf().set_size_inches(20, 15)
plt.show()
# %%
# In the above graph we can easily see that the mode imputation is the less efficient choice, since its distribution distantiates
# from the base case distribution.
# However, it's still hard to compare the other distributions with the base case.
# Let's plot the base case with each distribution separately, so we can have a better look at the graphs.
# %%
df_2['price'].plot(kind='kde', c='red', linewidth=3)
df_2_mean['price'].plot(kind='kde')

labels = ['Baseline (Complete Case)', 'Mean Imputation']
plt.legend(labels)

plt.xlabel('Price')

plt.gcf().set_size_inches(20, 15)
plt.show()
# %%
df_2['price'].plot(kind='kde', c='red', linewidth=3)
df_2_median['price'].plot(kind='kde')

labels = ['Baseline (Complete Case)', 'Median Imputation']
plt.legend(labels)

plt.xlabel('Price')

plt.gcf().set_size_inches(20, 15)
plt.show()
# %%
df_2['price'].plot(kind='kde', c='red', linewidth=3)
df_2_constant['price'].plot(kind='kde')

labels = ['Baseline (Complete Case)', 'Constant Imputation']
plt.legend(labels)

plt.xlabel('Price')

plt.gcf().set_size_inches(20, 15)
plt.show()
# %%
df_2['price'].plot(kind='kde', c='red', linewidth=3)
df_2_mice['price'].plot(kind='kde')

labels = ['Baseline (Complete Case)', 'Mice Imputation']
plt.legend(labels)

plt.xlabel('Price')

plt.gcf().set_size_inches(20, 15)
plt.show()
# %%
# The distributions the are closest to the base case are the Mean abd the Mice distributions.
# However, the Mice distribution would give us negative values for the surface areas (total and covered), which would likely
# change the distribution as we get rid of them.
# Therefore, the mean distribution will be the chosen one.
# %%
df_2_mean.shape
# %%
df_1.shape
# %%
br_clean = pd.concat([df_1, df_2_mean], axis=1)
# %%
br_clean.head()
# %%
br_clean.shape
# %%
# Now let's make sure the surface_covered is not greater than the surface_total
# %%
br_clean = br_clean.drop(br_clean[br_clean['surface_total_in_m2'] < br_clean['surface_covered_in_m2']].index)
# %%
br_clean['rooms'] = br_clean['rooms'].round(0)
# %%
br_clean['rooms'] = br_clean['rooms'].astype(int)
# %%
br_clean.head()
# %%
br_clean = br_clean[['created_on', 'year', 'month', 'month_name', 'place_name', 'State', 'price', 'price_aprox_local_currency',
                       'price_aprox_usd', 'surface_total_in_m2', 'surface_covered_in_m2', 'price_usd_per_m2',
                       'price_per_m2', 'rooms', 'operation', 'property_type']]
# %%
br_clean.head()
# %%
br_clean.info()
# %%
