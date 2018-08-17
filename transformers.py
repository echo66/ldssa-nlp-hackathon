from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state


class AutomatedTruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.9, random_state=None, incr=+5):
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.incr = incr
        
    def fit(self, X, y=None):
        max_components = X.shape[1]
        rs = check_random_state(self.random_state)
        self.max_components = max_components
        for n_components in range(1, max_components, self.incr):
            self.svd = TruncatedSVD(n_components=n_components, random_state=rs)
            self.svd.fit(X)
            total_variance = self.svd.explained_variance_ratio_.sum()
            print(n_components, total_variance)
            if self.variance_threshold <= total_variance:
                break
                
    def transform(self, X):
        return self.svd.transform(X)