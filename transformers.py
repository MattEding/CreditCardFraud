import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin


class TimeToHour(TransformerMixin):
    def fit(self, X, y=None):
        bins = len(np.histogram_bin_edges(X['Time'], bins='auto'))
        cuts = pd.cut(X['Time'], bins)
        counts = cuts.value_counts()
        midnight = int(counts.index[-1].mid)
        self.midnight = midnight
        return self

    def transform(self, X):
        day_seconds = 24 * 60 * 60
        hour_seconds = 60 * 60
        X['Hour'] = ((X['Time'] - self.midnight) % day_seconds) / hour_seconds
        return X


class AmountCentsOnly(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Cents'] = (X['Amount'] % 1).round(2)
        return X


class Log1pAmount(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['Log1pAmount'] = np.log1p(X['Amount'])
        return X
