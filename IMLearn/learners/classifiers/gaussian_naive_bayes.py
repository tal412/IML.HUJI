from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_, counts = np.unique(y, return_counts=True)
        self.mu_ = []
        self.vars_ = []
        for c in self.classes_:
            self.mu_.append(X[y == c].mean(axis=0))
            self.vars_.append(X[y == c].var(axis=0, ddof=1))

        self.vars_ = np.array(self.vars_)
        self.mu_ = np.array(self.mu_)
        self.pi_ = np.array([[c / y.shape[0]] for c in counts])

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
        return np.take(self.classes_, np.argmax(self.likelihood(X), axis=1))

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

        x_rows, x_cols = X.shape
        mat = np.ndarray((x_rows, len(self.classes_)))

        cov_array = []
        for k in range(len(self.classes_)):
            cov_k = np.zeros(shape=(x_cols, x_cols))
            np.fill_diagonal(cov_k, self.vars_[k])
            cov_array.append(cov_k)

        for i, x in enumerate(X):
            for k in range(len(self.classes_)):
                # Calculating gaussian log-likelihood, same as Ex 1
                cov_inv = np.linalg.inv(cov_array[k])

                mat[i, k] = \
                    np.log(self.pi_[k]) \
                    - 0.5 * np.log(2 * np.pi * np.linalg.det(cov_array[k])) + \
                    - 0.5 * (x - self.mu_[k]).T @ cov_inv @ (x - self.mu_[k])

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
