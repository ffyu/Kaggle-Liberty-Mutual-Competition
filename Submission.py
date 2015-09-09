import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


FOLDER = './data/'
FILE1 = 'train.csv'
FILE2 = 'test.csv'
FILE3 = 'train_numeric_features.csv'
FILE4 = 'test_numeric_features.csv'


class BuildXgb():

    def __init__(self, model_idx, y_transform, feature_transform, xgb_params, folder=FOLDER):

        self.model_idx = model_idx
        self.y_transform = y_transform
        self.feature_transform = feature_transform
        self.xgb_params = xgb_params
        self.folder = folder

        self._train = None
        self._test = None
        self._rgr = None

    def _feature_select(self, df_train, df_test):

	# Drop four features
        return df_train.drop(['T1_V10', 'T1_V13', 'T2_V7', 'T2_V10'], axis=1), \
               df_test.drop(['T1_V10', 'T1_V13', 'T2_V7', 'T2_V10'], axis=1)

    def _data_clean_factor(self, df_train, df_test):

	# Convert non-numeric features to factors
        for col_name in df_train.columns.tolist():
            if df_train[col_name].dtype == 'object':
                le = LabelEncoder()
                le.fit(list(df_train[col_name]) + list(df_test[col_name]))
                df_train.loc[:, col_name] = le.transform(df_train[col_name])
                df_test.loc[:, col_name] = le.transform(df_test[col_name])
        return df_train.astype(float), df_test.astype(float)

    def _data_clean_ohe(self, df_train, df_test):

	# Convert non-numeric features using one-hot-encoding
        df_train['Train_Set'] = 1
        df_test['Train_Set'] = 0
        df = pd.concat([df_train, df_test])
        df.reset_index(inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        for col_name in df.columns.tolist():
            if df_train[col_name].dtype == 'object':
                df = pd.concat([df, pd.get_dummies(df[col_name], prefix=col_name)], axis=1)
                df.drop([col_name], axis=1, inplace=True)
        # Split the train and test again
        df_train_cleaned = pd.DataFrame(df[df['Train_Set'] == 1])
        df_test_cleaned = pd.DataFrame(df[df['Train_Set'] == 0])
        df_train_cleaned.drop(['Train_Set'], axis=1, inplace=True)
        df_test_cleaned.drop(['Hazard', 'Train_Set'], axis=1, inplace=True)
        return df_train_cleaned.astype(float), df_test_cleaned.astype(float)

    def _data_clean_all_ohe(self, df_train, df_test):

	# Convert all features using one-hot-encoding
        df_train['Train_Set'] = 1
        df_test['Train_Set'] = 0
        df = pd.concat([df_train, df_test])
        df.reset_index(inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        for col_name in df.columns.tolist():
            if col_name != 'Train_Set' and col_name != 'Id' and col_name != 'Hazard' and col_name != 'T2_V1' and col_name != 'T2_V2':
                df = pd.concat([df, pd.get_dummies(df[col_name], prefix=col_name)], axis=1)
                df.drop([col_name], axis=1, inplace=True)
        # Split the train and test again
        df_train_cleaned = pd.DataFrame(df[df['Train_Set'] == 1])
        df_test_cleaned = pd.DataFrame(df[df['Train_Set'] == 0])
        df_train_cleaned.drop(['Train_Set'], axis=1, inplace=True)
        df_test_cleaned.drop(['Hazard', 'Train_Set'], axis=1, inplace=True)
        return df_train_cleaned.astype(float), df_test_cleaned.astype(float)

    def _data_clean_all_factor(self, df_train, df_test):

	# Convert all features to factors
        for col_name in df_train.columns.tolist():
            if col_name != 'Id' and col_name != 'Hazard':
                le = LabelEncoder()
                le.fit(list(df_train[col_name]) + list(df_test[col_name]))
                df_train.loc[:, col_name] = le.transform(df_train[col_name])
                df_test.loc[:, col_name] = le.transform(df_test[col_name])
        return df_train.astype(float), df_test.astype(float)

    def _data_clean_numeric(self, df_train, df_test):

	# Helper function
        return df_train, df_test

    def _pre_processing(self, df_train, df_test):

	# Get rid of un-useful features
        if self.feature_transform != 'numeric':
            df_train, df_test = self._feature_select(df_train, df_test)

        # Apply feature transform strategies
        options = {'factor': self._data_clean_factor,
                   'ohe': self._data_clean_ohe,
                   'all_ohe': self._data_clean_all_ohe,
                   'all_factor': self._data_clean_all_factor,
                   'numeric': self._data_clean_numeric}

        self._train, self._test = options[self.feature_transform](df_train, df_test)

    def _fit(self):

        # model training
        if self.y_transform == 'log':
            y = np.log(self._train['Hazard'].values)
        elif self.y_transform == 'sqrt':
            y = np.sqrt(self._train['Hazard'].values)
        else:
            y = self._train['Hazard'].values

        X = self._train.drop(['Id', 'Hazard'], axis=1).values

        plst = self.xgb_params[0]
        num_rounds = self.xgb_params[1]

        train_xgb = xgb.DMatrix(X, label=y)
        self._rgr = xgb.train(plst, train_xgb, num_rounds)

    def _predict(self):

        # predict for test set
        X_test = self._test.drop(['Id'], axis=1).values
        test_xgb = xgb.DMatrix(X_test)

        if self.y_transform == 'log':
            y_predict = np.exp(self._rgr.predict(test_xgb))
        elif self.y_transform == 'sqrt':
            y_predict = np.power(self._rgr.predict(test_xgb), 2)
        else:
            y_predict = self._rgr.predict(test_xgb)

        # write to file
        test_id = self._test['Id'].astype(int).values
        df_result = pd.DataFrame({'Id': test_id, 'Hazard': y_predict})
        df_result.set_index('Id', inplace=True)
        df_result.to_csv(self.folder+'Model{}.csv'.format(self.model_idx))

    def build(self, df_train, df_test):
		
        # Build the model
        self._pre_processing(df_train, df_test)
        self._fit()
        self._predict()


def ensemble_results(y_list, w):
	
    # Calculate ensemble results given predictions and weights
    combined = np.empty([y_list[0].shape[0], len(y_list)])
    for col, y in enumerate(y_list):
        combined[:, col] = y

    return np.average(combined, axis=1, weights=w)


def main():
	
    # Read in the raw and processed data
    df_train = pd.read_csv(FOLDER+FILE1)
    df_test = pd.read_csv(FOLDER+FILE2)
    df_train_numeric = pd.read_csv(FOLDER+FILE3)
    df_test_numeric = pd.read_csv(FOLDER+FILE4)

    # Build Model 1
    params = {'objective': 'reg:linear',
              'eta': 0.005,
              'subsample': 0.7,
              'max_depth': 9,
              'min_child_weight': 6,
              'colsample_bytree': 0.7,
              'silent': 1
              }
    num_rounds = 1000
    bx = BuildXgb(model_idx=1, y_transform=None, feature_transform='factor', xgb_params=[list(params.items()), num_rounds])
    bx.build(df_train, df_test)
    print 'Model 1 has been built!'

    # Build Model 2
    params = {'objective': 'count:poisson',
              'eta': 0.005,
              'subsample': 0.9,
              'max_depth': 6,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 5000
    bx = BuildXgb(model_idx=2, y_transform=None, feature_transform='factor', xgb_params=[list(params.items()), num_rounds])
    bx.build(df_train, df_test)
    print 'Model 2 has been built!'

    # Build Model 3
    params = {'objective': 'count:poisson',
              'eta': 0.005,
              'subsample': 0.7,
              'max_depth': 6,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 5000
    bx = BuildXgb(model_idx=3, y_transform=None, feature_transform='factor', xgb_params=[list(params.items()), num_rounds])
    bx.build(df_train, df_test)
    print 'Model 3 has been built!'

    # Build Model 4
    params = {'objective': 'reg:linear',
              'eta': 0.005,
              'subsample': 0.9,
              'max_depth': 6,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 5000
    bx = BuildXgb(model_idx=4, y_transform='log', feature_transform='factor', xgb_params=[list(params.items()), num_rounds])
    bx.build(df_train, df_test)
    print 'Model 4 has been built!'

    # Build Model 5
    params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.7,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 5000
    bx = BuildXgb(model_idx=5, y_transform='log', feature_transform='factor', xgb_params=[list(params.items()), num_rounds])
    bx.build(df_train, df_test)
    print 'Model 5 has been built!'

    # Build Model 6
    params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 5000
    bx = BuildXgb(model_idx=6, y_transform='sqrt', feature_transform='factor', xgb_params=[list(params.items()), num_rounds])
    bx.build(df_train, df_test)
    print 'Model 6 has been built!'

    # Build Model 7
    params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 5000
    bx = BuildXgb(model_idx=7, y_transform='sqrt', feature_transform='ohe', xgb_params=[list(params.items()), num_rounds])
    bx.build(df_train, df_test)
    print 'Model 7 has been built!'

    # Build Model 8
    params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 5000
    bx = BuildXgb(model_idx=8, y_transform='sqrt', feature_transform='all_ohe', xgb_params=[list(params.items()), num_rounds])
    bx.build(df_train, df_test)
    print 'Model 8 has been built!'

    # Build Model 9
    params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 5000
    bx = BuildXgb(model_idx=9, y_transform='sqrt', feature_transform='all_factor', xgb_params=[list(params.items()), num_rounds])
    bx.build(df_train, df_test)
    print 'Model 9 has been built!'

    # Build Model 10
    params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 5000
    bx = BuildXgb(model_idx=10, y_transform='sqrt', feature_transform='numeric', xgb_params=[list(params.items()), num_rounds])
    bx.build(df_train_numeric, df_test_numeric)
    print 'Model 10 has been built!'

    # Define Weights
    w = [1, 1, 1, 1, 1, 3, 3, 3, 4, 2]

    # Ensemble results
    y_list = []
    for model_idx in range(1, 11):
        df = pd.read_csv(FOLDER+'Model{}.csv'.format(model_idx))
        y_list.append(df['Hazard'].values)
    y_predict = ensemble_results(y_list, w)
    print "\nEnsemble Completed!\n"

    # Prepare submission file
    test_id = df_test['Id'].astype(int).values
    df_result = pd.DataFrame({'Id': test_id, 'Hazard': y_predict})
    df_result.set_index('Id', inplace=True)
    df_result.to_csv(FOLDER+'submission.csv')

if __name__ == '__main__':
    main()
