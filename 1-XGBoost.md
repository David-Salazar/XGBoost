# XGBoost

## Fit/Predict

You can use the scikit-learn `.fit() / .predict()` paradigm that you are already familiar to build your XGBoost models, as the xgboost library has a scikit-learn compatible API!

```PYTHOn
# Import xgboost
import xgboost as xgb

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))
```

## Cross Validation

Using the `xgboost` API, we get one function cross-validation:

```python
# Create the DMatrix: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params= params, nfold=3, num_boost_round=5, metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])
```

## Loss functions

Loss function names in xgboost:

- reg:linear - use for regression problems
- reg:logistic - use for classification problems when you want just decision, not probability
- binary:logistic - use when you want probability rather than just decision

## Feature Importance

```python
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(X, y)

# Create the parameter dictionary: params
params = {"objective": "reg:linear","max_depth":4}

# Train the model: xg_reg
xg_reg = xgb.train(dtrain = housing_dmatrix, params = params)

# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()
```

# Fine Tuning

## Early stopping rounds

```python
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain = housing_dmatrix, params = params,
                    nfold = 3, metrics = "rmse", as_pandas = True,
                    seed = 123, early_stopping_rounds = 10,
                    num_boost_round = 50)

# Print cv_results
print(cv_results)
```

## Grid Search

```python
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator = gbm,
                        param_grid = gbm_param_grid,
                        scoring = "neg_mean_squared_error",
                        cv = 4,
                        verbose = 1)


# Fit grid_mse to the data
grid_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
```

## Random Search

```python 
# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators=10)

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(estimator = gbm,
                 param_distributions = gbm_param_grid,
                 scoring = "neg_mean_squared_error",
                 n_iter = 5,
                 cv = 4,
                 verbose = 1)


# Fit randomized_mse to the data
randomized_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
```

