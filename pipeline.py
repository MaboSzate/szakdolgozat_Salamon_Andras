from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb


def original_pipeline(X_train, y_train, split, method, importance="gain"):
    scaler = StandardScaler()  # sztenderdizáció
    # paraméterrácsok a különböző módszerekhez
    if method == "SVR":
        reg = SVR(kernel="rbf")
        param_grid = {'reg__C': [1, 2, 4, 8, 16, 32, 64, 100],
                      'reg__gamma': [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1],
                      'reg__epsilon': [0.001, 0.01, 0.1, 1]}
    if method == "RF":
        reg = RandomForestRegressor(random_state=42, n_estimators=500)
        param_grid = {'reg__max_depth': [2, 3, 5, 8],
                      'reg__min_samples_leaf': [0.1, 0.15, 0.2, 0.25],
                      'reg__max_features': [3, 5, 8, 10]}
    if method == "LGB":
        reg = lgb.LGBMRegressor(verbosity=-1, data_sample_strategy='goss', boosting_type='dart', random_state=42,
                                importance_type=importance)
        param_grid = {'reg__num_leaves': [5, 15, 31],
                      'reg__n_estimators': [50, 100, 200, 400],
                      'reg__learning_rate': [0.1, 0.01, 0.005, 0.001]}
    if method == "XGB":
        reg = xgb.XGBRegressor(random_state=42, tree_method='hist')
        param_grid = {'reg__n_estimators': [100, 250, 400],
                      'reg__max_depth': [3, 5, 8],
                      'reg__learning_rate': [0.01, 0.005, 0.001]}
    pipe = Pipeline(steps=[("scaler", scaler), ("reg", reg)])  # itt a reg a becslés, a módszertől függ, hogy milyen
    search = GridSearchCV(pipe, param_grid, cv=split, scoring='neg_mean_squared_error')
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.9f):" % search.best_score_)
    print(search.best_params_)
    return search.best_estimator_, -search.best_score_


def final_pipeline(X_train, y_train, split, importance="gain"):
    scaler = StandardScaler()
    reg = lgb.LGBMRegressor(verbosity=-1, data_sample_strategy='goss', boosting_type='dart', random_state=42,
                            importance_type=importance)
    param_grid = {'reg__num_leaves': [5, 10, 15, 31],  # itt már a 10 is szerepel
                  'reg__n_estimators': [50, 100, 200, 400],
                  'reg__learning_rate': [0.1, 0.01, 0.005, 0.001],
                  'reg__lambda_l2': [0, 1, 10]}  # regularizációs paraméter
    pipe = Pipeline(steps=[("scaler", scaler), ("reg", reg)])
    search = GridSearchCV(pipe, param_grid, cv=split, scoring='neg_mean_squared_error')
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.9f):" % search.best_score_)
    print(search.best_params_)
    return search.best_estimator_, -search.best_score_
