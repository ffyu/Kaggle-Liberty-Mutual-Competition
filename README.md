## My solution for Kaggle's Liberty Mutual Competition:

https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction

### Final private leader board ranking: 17/2236 (top 1%)

The script **"submission.py"** produces the prediction based on an ensemble of several XGBoost models with different specifications.

The dependencies include:
 - [numpy](http://docs.scipy.org/doc/numpy/user/install.html)
 - [pandas](http://pandas.pydata.org/pandas-docs/stable/)
 - [scikit-learn](http://scikit-learn.org/stable/)
 - [xgboost](https://github.com/dmlc/xgboost)

To run the script, you need to download two raw files **"train.csv"** and **"test.csv"**, and two processed files **"train_numeric_features.csv"** and **"test_numeric_features.csv"** to the same folder. 

The numeric features are out-of-fold mean, median, and std of the categorical variables

Set the global variable **"FOLDER"** in **"submission.py"** to the same folder that just downloaded all datat files.

You should be good to go. The running time is around 5 hours in a i5-8G laptop.

The final ensemble includes 10 different xgboost models with varying model parameters, features and target variable transformations. 

The choice of those 10 models comes from a candidate of 185 models with broader model specifications. The incorporation and weights of each model is determined based on a forward stepwise searching algorithm with replacement. The code for training those 185 models and finding the best combinations are too lengthy to show here.

Feifei Yu

FeifeiYu1204@gmail.com
