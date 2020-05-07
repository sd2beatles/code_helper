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
