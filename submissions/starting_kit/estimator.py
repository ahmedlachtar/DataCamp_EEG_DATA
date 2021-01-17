import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from numpy.fft import fft, fftfreq
import pywt

class DemocracyEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.categorical_transformer = OneHotEncoder()
        self.categorical_feature = ['device']
        self.signals_feature = ['signals']
        self.devices = ["MW", "EP", "IN", "MU"]
        self.estimators = {}
        self.transformers = {}
        self.imputers = {}
        self.pipelines = {}
        
        for device in self.devices:
            self.estimators[device] = GradientBoostingClassifier()
            self.transformers[device] = ColumnTransformer(
                transformers=[
                    ('num', FunctionTransformer(
                        lambda X: feature_extractor(X)), self.signals_feature),
                ])
            self.imputers[device] = SimpleImputer(
                missing_values=np.nan, strategy='mean')
            self.pipelines[device] = Pipeline([
                ('transformer', self.transformers[device]),
                ('imputer', self.imputers[device]),
                ('estimator', self.estimators[device]),
                ])

    def fit(self, X, y):
        X_dict = {}
        y_dict = {}
        
        for key in self.estimators.keys():
            X_dict[key] = X.loc[X['device'] == key]
            y_dict[key] = y[X['device'] == key]

        
        for key in self.estimators.keys():
            print("Fitting estimator of device: ",key, "with df_train.shape=",X_dict[key].shape)
            """ self.transformers[key].fit(X_dict[key])
            transformed_samples = self.transformers[key].transform(X_dict[key])
            try:
                self.estimators[key].fit(transformed_samples, y_dict[key])
            except:
                print(transformed_samples) """

            self.pipelines[key].fit(X_dict[key], y_dict[key])

        print("Fitting phase finished...")    
        return self

    def predict(self, X):
        y_pred = np.empty((X.shape[0]))
        
        for k in range(X.shape[0]):
            row = X.loc[[k], :]
            device = row.loc[k, "device"]

            y_pred[k] = self.pipelines[device].predict(row)
        
        return y_pred

    def predict_proba(self, X):
        prob_pred = np.empty((X.shape[0], 11))
        prob_pred[:] = np.nan
        n = X.shape[0]
        
        if n == 7144 :
            step = "Computing train score:"
        
        else : 
            step = "Computing cv score:"
        
        for k in range(n):
            row = X.loc[[k], :]
            device = row.loc[k, "device"]

            aux = self.pipelines[device].predict_proba(row)[0]
            
            if aux.size == 11:
                prob_pred[k, :] = aux
            
            else:
                prob_pred[k, :] = np.concatenate(([0], aux))
                
            if k%500 == 0:
                print(step,k,"/",n)
                
        return np.array(prob_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == np.array(y))


def feature_extractor(X):
    X = X.reset_index(drop=True)
    n_channels = len(X.loc[0,"signals"])
    X_df = X['signals']
    n_freq = 2
    n_features = 10
    feature_array = np.zeros((len(X_df), n_channels*n_features))
    
    for k, (_, x) in enumerate(X_df.iteritems()):
        len_list = len(x[0])
        for i in range(n_channels):
            # Wavelet
            (ca, cd) = pywt.dwt(x[i],'haar')
            cat = pywt.threshold(ca, np.std(ca)/2)
            cdt = pywt.threshold(cd, np.std(cd)/2)
            wv_feat =list()
            wv_feat.append(np.mean(cat))
            wv_feat.append(np.max(cat))
            wv_feat.append(np.min(cat))


            wv_feat.append(np.mean(cdt))
            wv_feat.append(np.max(cdt))
            wv_feat.append(np.min(cdt))


            wv_feat = np.array(wv_feat)
            
            # FFT
            ft = fft(x[i])
            freqs_ft = fftfreq(len_list)
            ft, freqs_ft = ft[freqs_ft > 0], freqs_ft[freqs_ft > 0]
            magnitude_spectrum = np.abs(ft)
            indices = (-magnitude_spectrum).argsort()[:n_freq]
            freqs = freqs_ft[indices]
            fft_feat = np.concatenate(
                (freqs, np.mean(x[i]).reshape(-1), np.std(x[i]).reshape(-1)))
            
            # Concat
            feature_array[k, (n_features*i):((n_features*(i+1)))] = np.concatenate((fft_feat,wv_feat))
            
    return feature_array


clf = Pipeline(steps=[('classifier', DemocracyEstimator())])


def get_estimator():
    return clf