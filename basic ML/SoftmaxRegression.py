import numpy as np


class SoftmaxRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=100,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.coef_ = None
        self.intercept_ = None

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def get_penalty_grad(self):
        if self.penalty == "l2":
            return 2 * self.alpha * self.coef_
        elif self.penalty == "l1":
            return self.alpha * np.sign(self.coef_)
        return 0

    def fit(self, x, y):
        n_samples, n_features = x.shape
        n_classes = np.max(y) + 1

        self.coef_ = np.zeros((n_features, n_classes))
        self.intercept_ = np.zeros(n_classes)

        best_loss = np.inf
        no_improve = 0

        for i in range(self.max_iter):
            if self.shuffle:
                index = np.arange(n_samples)
                np.random.shuffle(index)
                x, y = x[index], y[index]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                x_batch, y_batch = x[start:end], y[start:end]

                y_one_hot = np.eye(n_classes)[y_batch]

                logits = x_batch @ self.coef_ + self.intercept_
                probs = self.softmax(logits)
                error = (probs - y_one_hot) / len(y_batch)

                grad_w = x_batch.T @ error + self.get_penalty_grad()
                grad_w0 = np.sum(error, axis=0)

                self.coef_ -= self.eta0 * grad_w
                self.intercept_ -= self.eta0 * grad_w0

            val_loss = np.mean(-np.sum(y_one_hot * np.log(probs + 1e-15), axis=1))
            if self.early_stopping:
                if val_loss + self.tol < best_loss:
                    best_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.n_iter_no_change:
                        break

    def predict_proba(self, x):
        logits = x @ self.coef_ + self.intercept_
        return self.softmax(logits)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
