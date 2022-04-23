from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.mu_ = []
        for c in self.classes_:
            self.mu_.append(X[y == c].mean(axis=0))

        self.mu_ = np.array(self.mu_)
        self.pi_ = np.array([[c / y.shape[0]] for c in counts])

        self.cov_ = np.zeros(shape=(X.shape[1], X.shape[1]))
        for c in self.classes_:
            Xg = X[y == c, :]
            self.cov_ += self.pi_[c] * np.cov(Xg.T, bias=True)

        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        x_rows = X.shape[0]
        pred_arr = list()

        for i in range(x_rows):
            arg_arr = list()
            for k in range(len(self.classes_)):
                a_k = self._cov_inv @ self.mu_[k]
                b_k = np.log(self.pi_[k]) - 0.5 * self.mu_[k].T @ self._cov_inv @ self.mu_[k]
                arg_arr.append(a_k.T @ X[i] + b_k)

            pred = self.classes_[np.argmax(arg_arr)]
            pred_arr.append(pred)

        return np.array(pred_arr)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        def prob(x, c):
            return \
                np.log(self.pi_[c]) \
                + x.T @ self._cov_inv @ self.mu_[c] \
                - 0.5 * self.mu_[c].T @ self._cov_inv @ self.mu_[c]

        x_rows = X.shape[0]
        mat = np.ndarray((x_rows, len(self.classes_)))

        for i in range(x_rows):
            for k in range(len(self.classes_)):
                mat[i, k] = prob(X[i], k)

        return mat

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        pred = self.predict(X).flatten()
        return misclassification_error(y, pred)
