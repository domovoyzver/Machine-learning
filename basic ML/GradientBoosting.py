import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomRegressor:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []

    def fit(self, x, y):
        self._estimators = []
        y_pred = np.full_like(y, np.mean(y), dtype=np.float64)
        self._init_prediction = np.mean(y)

        for i in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=None if self.random_state is None else self.random_state
            )
            tree.fit(x, residuals)
            update = tree.predict(x)
            y_pred += self.learning_rate * update
            self._estimators.append(tree)

    def predict(self, x):
        pred = np.full(x.shape[0], self._init_prediction, dtype=np.float64)
        for tree in self._estimators:
            pred += self.learning_rate * tree.predict(x)
        return pred

    @property
    def estimators_(self):
        return self._estimators


class GBCustomClassifier:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        self._estimators = []
        y = y.astype(np.float64)
        raw_pred = np.full_like(y, np.log((y.mean() + 1e-10) / (1 - y.mean() + 1e-10)), dtype=np.float64)
        self._init_prediction = raw_pred.copy()

        for i in range(self.n_estimators):
            prob = self._sigmoid(raw_pred)
            grad = prob - y
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=None if self.random_state is None else self.random_state
            )
            tree.fit(x, -grad)
            update = tree.predict(x)
            raw_pred += self.learning_rate * update
            self._estimators.append(tree)

    def predict_proba(self, x):
        raw_pred = np.full(x.shape[0], self._init_prediction[0], dtype=np.float64)
        for tree in self._estimators:
            raw_pred += self.learning_rate * tree.predict(x)
        prob = self._sigmoid(raw_pred)
        return np.vstack([1 - prob, prob]).T

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    @property
    def estimators_(self):
        return self._estimators
