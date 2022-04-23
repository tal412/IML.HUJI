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
            self.vars_.append(X[y == c].var(axis=0))

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
        x_cols = X.shape[1]
        cov_array = []

        for k in range(len(self.classes_)):
            cov_k = np.zeros(shape=(x_cols, x_cols))
            np.fill_diagonal(cov_k, self.vars_[k])
            cov_array.append(cov_k)

        preds_list = list()
        for x in X:
            args_list = list()
            for c in self.classes_:
                inv_cov = np.linalg.inv(cov_array[c])
                inv_cov_det = np.linalg.det(inv_cov)
                likelihood = 0.5 * np.log(inv_cov_det) - 0.5 * (x - self.mu_[c]).T @ inv_cov @ (x - self.mu_[c])
                post = np.log(self.pi_[c]) + likelihood
                args_list.append(post)

            preds_list.append(self.classes_[np.argmax(args_list)])

        return np.array(preds_list)

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
            cov_inv = self._get_inv_cov(X)
            return \
                np.log(self.pi_[c]) \
                + x.T @ cov_inv @ self.mu_[c] \
                - 0.5 * self.mu_[c].T @ cov_inv @ self.mu_[c]

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

    def _get_inv_cov(self, X):
        x_cols = X.shape[1]
        cov_mat = np.zeros(shape=(x_cols, x_cols))

        for i in range(x_cols):
            for k in range(len(self.classes_)):
                cov_mat[i, i] += self.vars_[k, i] * self.pi_[k]

        return np.linalg.inv(cov_mat)

