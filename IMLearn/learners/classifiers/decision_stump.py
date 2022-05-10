from __future__ import annotations

import sys
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
        loss_min = np.inf
        self.fitted_ = True

        for sign, j in product([-1, 1], range(X.shape[1])):
            threshold, loss = self._find_threshold(X[:, j], y, sign)
            if loss < loss_min:
                loss_min = loss
                self.sign_ = sign
                self.threshold_ = threshold
                self.j_ = j

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
        chosen_feature = X[:, self.j_]
        pred = self.sign_ * ((chosen_feature >= self.threshold_) * 2 - 1)
        return pred

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
        D = abs(labels)
        labels = np.sign(labels)
        sorted_vals, sorted_labels, D = values[sort_idx], labels[sort_idx], D[sort_idx]
        thresholds = np.concatenate(
            [
                [-np.inf],
                [(sorted_vals[i]+sorted_vals[i+1])/2 for i in range(len(sorted_vals)-1)],
                [np.inf]]
        )
        min_threshold_loss = np.sum(D[sorted_labels == sign])
        losses = np.append(min_threshold_loss, min_threshold_loss - np.cumsum(D * (sorted_labels * sign)))
        min_loss_idx = np.argmin(losses)

        if min_loss_idx == 0:
            final_threshold = -np.inf

        elif min_loss_idx == values.shape[0]:
            final_threshold = np.inf
        else:
            final_threshold = thresholds[min_loss_idx]

        return final_threshold, losses[min_loss_idx]

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
