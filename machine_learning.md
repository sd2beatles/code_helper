## 1. XGboost and Light GBM
key parameters

1.n_estimators: the number of trees involved in the boosting(트리의 갯수)
              : the higher number of tress leads to the higher accuracy but leads to dissapointing performances due to overfitting
              
2.learning_rate: sacale parameter. The lower learing rate could lead to the higher accuracy at the same time delaying the processing time
                 and higher chance of over-fitting problems
                 
3.max_depth: the depth of tress

4.min_child_samples: the minum number of records in the leaf. This is one of the mechanisms to avoid overfitting problems

5.min_child_weight: 
![image](https://user-images.githubusercontent.com/53164959/81254970-70663880-9067-11ea-810c-2468bec13480.png)

5.boosting: specifying the type of algorith gbdf: the general gradient boosting , rf:random forest

6.subsample: the size of sample to construct the tree. Let's say subsample=0.5, the model randomly samples a half of training data prior to constructing the tree Also, this helps to avoid the overfitting. 

7.colsample_bytree:the ratio of random selection of input features

8.num_leaves:the maxium number of leaves for each tree


![image](https://user-images.githubusercontent.com/53164959/81252886-19119980-9062-11ea-8e7e-6e0adfa416f7.png)

```python
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

fit_params={'early_stopping_rounds':30,
            'eval_metrics':'logloss', #Default: ‘l2’ for LGBMRegressor, ‘logloss’ for LGBMClassifier, ‘ndcg’ for LGBMRanker.
            'eval_set':[(X_test,to_categorical(y_test))],
            'eval_names':['valid'],
            'verbose':100,
            'categorical_feature':'auto' # Feature names. If ‘auto’ and data is pandas DataFrame, data columns names are used.
           }

#This parameter defines the number of HP points to be tested
n_HP_points_to_test = 100

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=n_HP_points_to_test,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)

```
## 2. GridSearch

```python
def auc_scores(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    sm=SMOTE(random_state=0)
    X_over,y_over=sm.fit_sample(X,y)
    params={
             'rf__n_estimators':[5,10],
             'rf__max_depth':[2,4,6],
             'rf__min_samples_split':[4,8,12]
            }
    pipeline=Pipeline([('sc',sc),
                       ('rf',rf)])
    grid_search=GridSearchCV(pipeline,param_grid=params,scoring='roc_auc',cv=4)
    grid_search.fit(X_over,y_over)
    y_pred=grid_search.predict(X_test)
    results=roc_auc_score(y_test,y_pred)
    return results
```

## 3. RandomizedSearch
```python
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'rf__n_estimators': n_estimators,
               'rf__max_features': max_features,
               'rf__max_depth': max_depth,
               'rf__min_samples_split': min_samples_split,
               'rf__min_samples_leaf': min_samples_leaf,
               'rf__bootstrap': bootstrap}

pipeline=Pipeline([('sc',sc),('rf',rf)])
rf_grid_search=RandomizedSearchCV(pipeline,param_distributions=random_grid,n_iter=100,cv=5,n_jobs=-1,verbose=2)
rf_grid_search.fit(X_re,y_re)




```

