from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = 'browser'


def polynomial_value(x):
    return (x + 3)*(x + 2)*(x + 1)*(x - 1)*(x - 2)


def generate_dataset_from_polynomial(n_samples, noise):
    X = np.linspace(-1.2, 2, n_samples)

    y = np.zeros(shape=(n_samples, 1))
    noiseless_y = np.zeros(shape=(n_samples, 1))

    for i in range(n_samples):
        value = polynomial_value(X[i])
        noiseless_y[i] = value
        y[i] = value

    noise = np.array(np.random.normal(0, noise, n_samples)).reshape(n_samples, 1)
    y += noise

    return X, y, noiseless_y


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X, y, noiseless_y = generate_dataset_from_polynomial(n_samples, noise)
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X), pd.Series(y.flatten()), 2/3)

    x_values_1 = [train_x[0], test_x[0], X]
    y_values_1 = [train_y, test_y, pd.Series(noiseless_y.flatten())]
    names_1 = ["Train set", "Test set", "Noiseless"]
    colors = ["red", "blue", "black"]

    qs1_fig = go.Figure()
    qs1_fig.update_layout(
        title=f"{n_samples} samples between -1.2 and 2 and their predicted values with {noise} noise",
        xaxis_title="Samples in the range [-1.2,2]",
        yaxis_title="Predicted values [colored] and true value [black]"
    )
    for i in range(len(x_values_1)):
        opacity = 1
        if i == 2:
            opacity = 0.5

        qs1_fig.add_trace(
            go.Scatter(
                x=x_values_1[i],
                y=y_values_1[i],
                mode='markers',
                marker=go.scatter.Marker(color=colors[i],opacity=opacity),
                name=names_1[i],
                showlegend=True
            )
        )

    qs1_fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    qs2_fig = go.Figure()
    qs2_fig.update_layout(
        title=f"Train error and validation error as a function of polynomial degree with {n_samples} samples and {noise} noise",
        xaxis_title="Degree",
        yaxis_title="Error"
    )

    pol_degree = []
    avg_train_arr = []
    avg_validation_arr = []

    y_values_2 = [avg_train_arr, avg_validation_arr]
    names_2 = ["Average train error", "Average validation error"]

    for k in range(11):
        pol_degree.append(k)
        pol_model = PolynomialFitting(k)
        avg_train, avg_validation = cross_validate(pol_model, train_x[0], train_y.to_numpy(), mean_square_error, 5)
        avg_train_arr.append(avg_train)
        avg_validation_arr.append(avg_validation)

    for i in range(2):
        qs2_fig.add_trace(
            go.Scatter(
                x=pol_degree,
                y=y_values_2[i],
                mode='markers+lines',
                marker=go.scatter.Marker(color=colors[i]),
                name=names_2[i],
                showlegend=True
            )
        )

    qs2_fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_deg = np.argmin(avg_validation_arr)
    best_estimator = PolynomialFitting(min_deg)
    best_estimator.fit(train_x[0], train_y.to_numpy())
    best_validation_err = mean_square_error(test_y.to_numpy(), best_estimator.predict(test_x[0]))

    print(f"n_samples: {n_samples}, noise: {noise}")
    print(f"K*: {min_deg}")
    print(f"Validation error: {best_validation_err}")
    print("-------------------------------------------------------")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x = X[0: n_samples, :]
    train_y = np.array(y[0:n_samples]).reshape(-1, 1)
    train_y = train_y.reshape(-1, 1)

    test_x = X[n_samples+1: X.shape[0], :]
    test_y = np.array(y[n_samples+1:]).reshape(-1, 1)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_range = np.linspace(0.01, 3, n_evaluations)

    avg_train_err_ridge = []
    avg_validation_ridge = []

    avg_train_err_lasso = []
    avg_validation_lasso = []

    y_values_7 = [avg_train_err_ridge, avg_validation_ridge, avg_train_err_lasso, avg_validation_lasso]
    names_7 = [
        "Train - Ridge",
        "Validation - Ridge",
        "Train - Lasso",
        "Validation - Lasso"
    ]
    colors = ["red", "blue", "black", "green"]

    for lam in lambda_range:
        ridge_model = RidgeRegression(lam)
        ridge_train, ridge_validation = cross_validate(ridge_model, train_x, train_y, mean_square_error, 5)
        avg_train_err_ridge.append(ridge_train[0])
        avg_validation_ridge.append(ridge_validation[0])

        lasso_model = Lasso(lam)
        lasso_train, lasso_validation = cross_validate(lasso_model, train_x, train_y, mean_square_error, 5)
        avg_train_err_lasso.append(lasso_train[0])
        avg_validation_lasso.append(lasso_validation[0])

    qs7_fig = go.Figure()
    qs7_fig.update_layout(
        title=f"Error rates for different regularization parameter with"
              f" {n_samples} samples and {n_evaluations} evaluations",
        xaxis_title="Lambda",
        yaxis_title="Average err ( Train \\ Test )"
    )

    for i in range(4):
        qs7_fig.add_trace(
            go.Scatter(
                x=lambda_range,
                y=y_values_7[i],
                mode='markers+lines',
                marker=go.scatter.Marker(color=colors[i]),
                name=names_7[i],
                showlegend=True
            )
        )

    qs7_fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_lam_index_ridge = np.argmin(avg_validation_ridge)
    min_lam_ridge = lambda_range[min_lam_index_ridge]
    print(f"Best ridge lambda: {min_lam_ridge})")
    best_ridge = RidgeRegression(min_lam_ridge)
    best_ridge.fit(train_x, train_y)
    best_ridge_validation_err = mean_square_error(test_y, best_ridge.predict(test_x))
    print(f"Best ridge validation err: {best_ridge_validation_err})")
    print()

    min_lam_index_lasso = np.argmin(avg_validation_lasso)
    min_lam_lasso = lambda_range[min_lam_index_lasso]
    print(f"Best lasso lambda: {min_lam_lasso})")
    best_lasso = Lasso(min_lam_lasso)
    best_lasso.fit(train_x, train_y)
    best_lasso_validation_err = mean_square_error(test_y, best_lasso.predict(test_x))
    print(f"Best lasso validation err: {best_lasso_validation_err})")
    print()

    linear_model = LinearRegression()
    linear_model.fit(train_x, train_y)
    print(f"Linear validation err: {linear_model.loss(test_x, test_y)})")
    print("-------------------------------------------------------")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter(n_samples=50)



