import numpy as np


class LinearRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
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

        self._coef = None
        self._intercept = None

        self.rng = np.random.RandomState(random_state) if random_state is not None else np.random

    def get_penalty_grad(self):
        if self.penalty == "l2":
            return 2 * self.alpha * self._coef
        elif self.penalty == "l1":
            return self.alpha * np.sign(self._coef)
        else:
            return 0

    def fit(self, x, y):
        n_strings, n_features = x.shape
        self._coef = np.zeros(n_features)
        self._intercept = 0.0

        if self.early_stopping:
            val_size = int(n_strings * self.validation_fraction)
            x_train, y_train = x[:-val_size], y[:-val_size]
            x_val, y_val = x[-val_size:], y[-val_size:]
        else:
            x_train, y_train = x, y

        no_improve = 0
        best_loss = float("inf")

        for _ in range(self.max_iter):
            index = np.arange(len(x_train))
            if self.shuffle:
                self.rng.shuffle(index)
            x_train, y_train = x_train[index], y_train[index]

            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                y_pred = x_batch @ self._coef + self._intercept
                error = y_pred - y_batch

                grad_w = 2 * (x_batch.T @ error) / len(y_batch)
                grad_w0 = 2 * np.mean(error)

                self._coef -= self.eta0 * (grad_w + self.get_penalty_grad())
                self._intercept -= self.eta0 * grad_w0

            if self.early_stopping:
                val_loss = np.mean((x_val @ self._coef + self._intercept - y_val) ** 2)
                if val_loss + self.tol < best_loss:
                    best_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.n_iter_no_change:
                        break

    def predict(self, x):
        return x @ self._coef + self._intercept

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = np.array(value)

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
