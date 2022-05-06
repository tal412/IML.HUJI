from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        loss_min = 0

        for sign, j in product([-1, 1], range(X.shape[1])):
            loss, threshold = self._find_threshold(X[j], y, sign)
            if loss < loss_min:
                self.sign_ = sign
                self.threshold_ = threshold
                self.j_ = j
                loss_min = loss

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        best_feature_col = X[:self.j_]
        return np.array([self.sign_ if value >= self.threshold_ else -self.sign_ for value in best_feature_col])

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sort_idx = np.argsort(values)
        sorted_vals = values[sort_idx]
        sorted_labels = labels[sort_idx]

        loss_arr = []
        for threshold in sorted_vals:
            signs_arr = np.zeros(len(sorted_vals))
            np.where(sorted_vals >= threshold, signs_arr, -1)

            plus_sign_indexes = [i for i, v in enumerate(signs_arr) if v == sign]
            minus_sign_indexes = [i for i, v in enumerate(signs_arr) if v == -sign]

            plus_sign_loss =\
                np.sum(np.take(signs_arr, plus_sign_indexes) != np.take(sorted_labels, plus_sign_indexes))

            minus_sign_loss =\
                np.sum(np.take(signs_arr, minus_sign_indexes) != np.take(sorted_labels, minus_sign_indexes))

            total_loss = plus_sign_loss + minus_sign_loss
            loss_arr.append(total_loss)

        min_loss_id = np.argmin(loss_arr)
        return sorted_vals[min_loss_id], loss_arr[min_loss_id]


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
