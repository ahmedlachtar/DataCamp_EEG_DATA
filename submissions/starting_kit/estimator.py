import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


def get_mean(X):
    index = X.columns.get_loc("data")

    mean_array = np.array(X)
    n = mean_array.shape[0]

    L = mean_array[:, index]
    for i in range(n):
        mean_array[i, index] = round(np.mean(L[i]), 4)
    return mean_array


transformer_m = FunctionTransformer(
    lambda X: get_mean(X))

passthrough_cols = [
    'id',
    'event',
    'size'
]

transformer = make_column_transformer(
            # (OrdinalEncoder(), ['device']),
            # (OneHotEncoder(), ['channel']),
            (transformer_m, ['data']),
            ('passthrough', passthrough_cols)
)

pipe = make_pipeline(transformer,
                     RandomForestClassifier(n_estimators=200))


def get_estimator():
    return pipe
