from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """



    train_score = 0
    validation_score = 0

    folders = np.remainder(np.arange(X.shape[0]), cv)

    for i in range(cv):
        train_X = X[folders != i]
        train_y = y[folders != i]

        estimator.fit(train_X, train_y)
        train_pred = estimator.predict(train_X)
        train_score += scoring(train_y, train_pred)

        validate_X = X[folders == i]
        validate_y = y[folders == i]

        validate_pred = estimator.predict(validate_X)
        validation_score += scoring(validate_y, validate_pred)

    return train_score/cv, validation_score/cv


