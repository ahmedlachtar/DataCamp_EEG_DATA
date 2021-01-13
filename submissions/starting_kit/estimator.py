import numpy as np
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from numpy.fft import fft, fftfreq


class DemocracyEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.categorical_transformer = OneHotEncoder()
        self.categorical_feature = ['device']
        self.signals_feature = ['signals']
        self.estimators = {
            "EP": GradientBoostingClassifier(),
            "IN": GradientBoostingClassifier(),
            "MU": GradientBoostingClassifier(),
            "MW": GradientBoostingClassifier(),
        }
        self.transformers = {
            "EP": ColumnTransformer(
                transformers=[
                    ('num', FunctionTransformer(
                        lambda X: feature_extractor(X)), self.signals_feature),
                ]),
            "IN": ColumnTransformer(
              transformers=[
                   ('num', FunctionTransformer(
                       lambda X: feature_extractor(X)), self.signals_feature),
                ]),
            "MU": ColumnTransformer(
                transformers=[
                    ('num', FunctionTransformer(
                        lambda X: feature_extractor(X)), self.signals_feature),
                ]),
            "MW": ColumnTransformer(
                transformers=[
                    ('num', FunctionTransformer(
                        lambda X: feature_extractor(X)), self.signals_feature),
                ]),
        }

    def fit(self, X, y):
        X_dict = {}
        y_dict = {}
        for key in self.estimators.keys():
            X_dict[key] = X.loc[X['device'] == key]
            y_dict[key] = y[X['device'] == key]

        for key in self.estimators.keys():
            self.transformers[key].fit(X_dict[key])
            transformed_samples = self.transformers[key].transform(X_dict[key])
            self.estimators[key].fit(transformed_samples, y_dict[key])

        return self

    def predict(self, X):
        y_pred = np.empty((X.shape[0]))
        for k in range(X.shape[0]):
            row = X.loc[[k], :]
            device = row.loc[k, "device"]
            row = self.transformers[device].transform(row)
            y_pred[k] = self.estimators[device].predict(row)
        return y_pred

    def predict_proba(self, X):

        prob_pred = np.empty((X.shape[0], 11))
        for k in range(X.shape[0]):
            row = X.loc[[k], :]
            device = row.loc[k, "device"]
            row = self.transformers[device].transform(row)
            # [0,4:])))
            row_prediction = self.estimators[device].predict_proba(row)
            if device == "IN":
                row_prediction = np.append(0, row_prediction[0])
            prob_pred[k, :] = np.array(row_prediction)
        return np.array(prob_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == np.array(y))


def feature_extractor(X_df):

    X_df = X_df['signals'].reset_index(drop=True)
    n_channels = len(X_df[0])
    n_freq = 3
    n_features = 2 + 2*n_freq
    feature_array = np.zeros((len(X_df), n_channels*n_features))
    for k, (_, x) in enumerate(X_df.iteritems()):
        len_list = len(x[0])
        # x is a multi-channel signals
        for i in range(n_channels):
            ft = fft(x[i])
            freqs_ft = fftfreq(len_list)
            ft, freqs_ft = ft[freqs_ft > 0], freqs_ft[freqs_ft > 0]
            magnitude_spectrum = np.abs(ft)
            indices = (-magnitude_spectrum).argsort()[:n_freq]
            freqs = freqs_ft[indices]
            terms = magnitude_spectrum[indices]

            feature_array[k, n_features*i:(n_features*(i+1))] = np.concatenate(
                (
                    # wavelet_features,
                    freqs, terms, np.mean(x[i]).reshape(-1), np.std(x[i]).reshape(-1)))
    return feature_array


transformer_m = FunctionTransformer(
    lambda X: feature_extractor(X))


categorical_features = ['device']
categorical_transformer = OneHotEncoder()


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', "passthrough", ['signals'])
    ])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.

# clf = Pipeline(steps=[('preprocessor', preprocessor),
#                      ('classifier', DemocracyEstimator())])

clf = Pipeline(steps=[('classifier', DemocracyEstimator())])


def get_estimator():
    return clf
