# Empirical Analysis of Forward Returns
## Steve Hall
### Dec. 2019
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression

from utils import pct_returns
from utils import absolute_diff

# MANUAL INPUTS
output_dir = 'C:\devl\LeadingIndicators\output'
response_var = 'USGG10YR' # select variable to forecast
scaler = StandardScaler()
pca = PCA()

# Load and clean data
raw_px = pd.read_excel('C:\devl\LeadingIndicators\Data_tests_v3.xlsx', 
                         sheet_name='Clean', index_col='Dates')
desc = raw_px.describe().transpose()
min_rows = int(desc['count'].min())
clean_px = raw_px.iloc[-min_rows:]

variables_diff = ['FFFives', 'TwosTens', 'FivesThirties'] # zList of vars for 1st diff
levels_pct_chgs = clean_px[clean_px.columns.difference(variables_diff)]
levels_abs_diff = clean_px[variables_diff]

# SELECT RESPONSE VARIABLES
response = clean_px[response_var]
weeks = 8 # forecast period
forecast_pd = weeks*5  # Number of days in forecast period.  
# Shift back for fwd returns.
forecast_rets = pct_returns(response, -forecast_pd, 0) 
Y = forecast_rets.to_frame().dropna()

# PREDICTOR VARIABLES
# One-month reversals.
x_pct = pct_returns(levels_pct_chgs, 1, 21).dropna()
x_diff = absolute_diff(levels_abs_diff, 1, 21).dropna() # 21 days in a month.
x = x_pct.merge(x_diff, how='left', left_index=True, right_index=True)
x_columns = x.columns
# TODO: create day forward chaining logic
split = len(x) - 500 # timeseries split
x_train = x.iloc[:split]
scaled_features = scaler.fit_transform(x_train)
pca_features = pca.fit_transform(scaled_features)
x_1MR = pd.DataFrame(pca_features, index=x_train.index)
x_1MR_train = x_1MR.add_prefix('1MR_')

# Generate test data
x_test = x.iloc[split+5:] # leave a gap due to overlap
scaled_features_test = scaler.transform(x_test)
pca_features_test  = pca.transform(scaled_features_test)
x_1MR = pd.DataFrame(pca_features_test, index=x_test.index)
x_1MR_test = x_1MR.add_prefix('1MR_')

# Repeat for momentum factor (11 month return, 1 month ago)
# One-month pct return.
x_pct = pct_returns(levels_pct_chgs, 21, 253).dropna()
x_diff = absolute_diff(levels_abs_diff, 21, 253).dropna() # 21 days in a month.
x = x_pct.merge(x_diff, how='left', left_index=True, right_index=True)
x_columns = x.columns
# TODO: create day forward chaining logic
split = len(x) - 750 # timeseries split
x_train = x.iloc[:split]
scaled_features = scaler.fit_transform(x_train)
pca_features = pca.fit_transform(scaled_features)
x_MOMO = pd.DataFrame(pca_features, index=x_train.index)
x_MOMO_train = x_MOMO.add_prefix('MOMO_')
# Repeat for test data
x_test = x.iloc[split+5:]
scaled_features_test = scaler.transform(x_test)
pca_features_test  = pca.transform(scaled_features_test)
x_MOMO = pd.DataFrame(pca_features_test, index=x_test.index)
x_MOMO_test = x_MOMO.add_prefix('MOMO_')

X_train = x_MOMO_train.merge(x_1MR_train, how='left', left_index=True, right_index=True)
X_train = X_train.merge(Y, how='left', left_index=True, right_index=True).dropna()
Y_train = X_train.pop(response_var)

X_test = x_MOMO_test.merge(x_1MR_test, how='left', left_index=True, right_index=True)
X_test = X_test.merge(Y, how='left', left_index=True, right_index=True).dropna()
Y_test = X_test.pop(response_var)

# VARIABLE SELECTION
selector = SelectKBest(f_regression, k=20)
selector.fit(X_train, Y_train)
# Get columns to keep
cols = selector.get_support()
# Create new dataframe with only desired columns, or overwrite existing
select_X_train = X_train.loc[:,cols]
select_X_test = X_test.loc[:,cols]
   
# Describe predictor set
predictor_stats = select_X_train.describe()

# Create dictionary of estimators
ESTIMATORS = {
    "Tree": DecisionTreeRegressor(max_depth=10),
    "KNN": KNeighborsRegressor(),
    "LinearRegression": LinearRegression(),
    "Ridge": RidgeCV(),
    "NeuralNet": MLPRegressor(hidden_layer_sizes=(100,100),
                                 learning_rate_init=0.001,
                                 max_iter=400,
                                 random_state=0, #keep random number the same
                                 early_stopping=True)}

Y_test_predict = dict() # dictionary of OOS test predictions

# Hyperparameters
num_neighbors = [50, 75, 100, 125, 150]

for name, estimator in ESTIMATORS.items():
    if name == 'KNN':
        for i, wgts in enumerate(['uniform', 'distance']):
            for j, k in enumerate(num_neighbors):
                model = KNeighborsRegressor(k, weights=wgts)
                model.fit(select_X_train, Y_train)
                prediction = model.predict(select_X_test)
                Y_test_predict[name + '_' + wgts + '_' + str(k)] = prediction
    else:
        model = estimator.fit(select_X_train, Y_train)
        prediction = model.predict(select_X_test)
        Y_test_predict[name] = prediction

# Add straight line to forecast
ave_Y_train = Y_train.mean()
Y_test_predict['Naive'] = np.full((len(select_X_test),),ave_Y_train) 

# Iterate through keys in dictionary to get test residuals and RMSE
residuals = dict()
results = dict()
for key, item in Y_test_predict.items(): # item is Y prediction
    residuals[key] = Y_test - item
    results[key] = np.sqrt(mean_squared_error(Y_test,item))

# Create scatter plots of prediction v. actuals
for key, item in Y_test_predict.items():
    plt.scatter(item, Y_test)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.axis('tight')
    plt.title(key + '_' + 'OOS test')
    plt.show()

# Create line plots of prediction
for key, item in Y_test_predict.items():
    plt.plot(Y_test.index, Y_test, label='Actuals')
    plt.plot(Y_test.index, item, label='Predicted')
    plt.ylabel('8-week % Change in Response')
    plt.xlabel('Dates')
    plt.axis('tight')
    plt.legend()
    plt.title(key + '_' + 'OOS test')
    plt.show()
    
# Create line plots of residuals
for key, item in residuals.items():
    plt.plot(item.index, item, label='Resids')
    plt.ylabel('OOS Residuals')
    plt.xlabel('Dates')
    plt.axis('tight')
    plt.legend()
    plt.title(key + '_' + 'OOS test')
    plt.show()

# Nearest Neighbors Analysis
neigh = neighbors.NearestNeighbors(150)
neigh.fit(scaler.fit_transform(X))
last_obs = final_X.iloc[-1]
scaled_last_obs = scaler.transform([last_obs])
dist, indices = neigh.kneighbors(scaled_last_obs)
row_idx = indices.flatten()
nearest_neighbors = X.iloc[row_idx]

# Append last observation to nearest neighbors for output
output_df = nearest_neighbors.append(last_obs) # appends to last row