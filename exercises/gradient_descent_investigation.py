import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'
pio.templates.default = "simple_white"

def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(model, **kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])
        return

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                               etas: Tuple[float] = (1, .1, .01, .001)):

    functions = {
        "L1": L1,
        "L2": L2
    }

    for f_name, f in functions.items():
        for eta in etas:
            init_weights = np.copy(init)
            current_norm = f(init_weights)
            callback, values, weights = get_gd_state_recorder_callback()

            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback, out_type="best")
            best = gd.fit(current_norm, None, None)

            qs1_fig = plot_descent_path(module=f, descent_path=np.array(weights), title=f"for eta {eta} and norm {f_name}")
            qs1_fig.show()

            qs3_fig = go.Figure()
            qs3_fig.update_layout(
                title=f"Convergence rate for eta {eta} and norm {f_name}",
                xaxis_title="Iteration number",
                yaxis_title="Norm value",
                title_x=0.5
            )

            iterations = [i for i in range(len(values))]
            qs3_fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=values,
                    mode="markers",
                    marker=dict(color="blue"),
                    showlegend=False
                )
            )

            qs3_fig.show()

            print(f"Best loss of {f_name} for eta {eta} is {np.abs(0-f(best).compute_output())}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    qs5_fig = go.Figure()
    qs5_fig.update_layout(
        title=f"Convergence rate for L1 using exponential rate with different gamma values",
        xaxis_title="Iteration number",
        yaxis_title="Norm value",
        title_x=0.5
    )

    for i, gamma in enumerate(gammas):
        init_weights_L1 = np.copy(init)
        current_norm_L1 = L1(init_weights_L1)
        callback_L1, values, weights_L1 = get_gd_state_recorder_callback()
        gd_1 = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback_L1, out_type="best")
        best = gd_1.fit(current_norm_L1, None, None)

        if gamma == 0.95:
            print(f"Best loss of L1 for eta {eta} and gamma 0.95 is {np.abs(0 - L1(best).compute_output())}")
            init_weights_L2 = np.copy(init)
            current_norm_L2 = L2(init_weights_L2)
            callback_L2, values, weights_L2 = get_gd_state_recorder_callback()
            gd_2 = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback_L2, out_type="best")
            gd_2.fit(current_norm_L2, None, None)

            qs7_fig_1 = plot_descent_path(module=L1, descent_path=np.array(weights_L1), title=f"for L1 and gamma 0.95")
            qs7_fig_1.show()

            qs7_fig_2 = plot_descent_path(module=L2, descent_path=np.array(weights_L2), title=f"for L2 and gamma 0.95")
            qs7_fig_2.show()

        iterations = [i for i in range(len(values))]
        qs5_fig.add_trace(
            go.Scatter(
                x=iterations,
                y=values,
                mode="markers",
                name=f"Gamma: {gamma}",
                showlegend=True,
            )
        )

    qs5_fig.show()

    qs6_fig = go.Figure()
    qs6_fig.update_layout(
        title=f"Convergence rate for L1 using exponential rate with different gamma values",
        xaxis_title="Iteration number",
        yaxis_title="Norm value",
        title_x=0.5
    )


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Q8 ----------------------------------------------------

    log_model = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000))
    log_model.fit(X_train.to_numpy(), y_train.to_numpy())
    prob_vec = log_model.predict_proba(X_train.to_numpy())

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_train, prob_vec)

    qs8_fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color="red",)],

        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    qs8_fig.show()

    # Q9 ----------------------------------------------------

    qs9_best_alpha = thresholds[np.argmax(tpr-fpr)]
    print(f"Best alpha is {qs9_best_alpha}")

    best_alpha_pred = prob_vec >= qs9_best_alpha
    test_err = misclassification_error(y_train.to_numpy(), best_alpha_pred)
    print(f"Test error using best alpha is: {test_err}")

    # Q10 and Q11 ----------------------------------------------------

    train_errs_l1 = []
    val_errs_l1 = []
    train_errs_l2 = []
    val_errs_l2 = []

    possible_lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for lam in possible_lambdas:
        l1_log_model = LogisticRegression(
            solver=
            GradientDescent(
                learning_rate=FixedLR(1e-4),
                max_iter=20000,
            ),
            penalty="l1",
            alpha=0.5,
            lam=lam
        )

        l2_log_model = LogisticRegression(
            solver=
            GradientDescent(
                learning_rate=FixedLR(1e-4),
                max_iter=20000,
            ),
            penalty="l2",
            alpha=0.5,
            lam=lam
        )

        train_err_l1, val_err_l1 = \
            cross_validate(
                l1_log_model,
                X_train.to_numpy(),
                y_train.to_numpy(),
                misclassification_error
            )
        train_errs_l1.append(train_err_l1)
        val_errs_l1.append(val_err_l1)

        train_err_l2, val_err_l2 = \
            cross_validate(
                l2_log_model,
                X_train.to_numpy(),
                y_train.to_numpy(),
                misclassification_error
            )

        train_errs_l2.append(train_err_l2)
        val_errs_l2.append(val_err_l2)

    min_err_l1 = np.min(val_errs_l1)
    min_err_l2 = np.min(val_errs_l2)

    best_lam_index_l1 = val_errs_l1.index(min_err_l1)
    best_lam_l1 = possible_lambdas[best_lam_index_l1]

    best_lam_index_l2 = val_errs_l2.index(min_err_l2)
    best_lam_l2 = possible_lambdas[best_lam_index_l2]

    best_l1_log_model = LogisticRegression(
        solver=
        GradientDescent(
            learning_rate=FixedLR(1e-4),
            max_iter=20000,
        ),
        penalty="l1",
        alpha=0.5,
        lam=best_lam_l1
    )
    best_l1_log_model.fit(X_train.to_numpy(), y_train.to_numpy())

    best_l2_log_model = LogisticRegression(
        solver=
        GradientDescent(
            learning_rate=FixedLR(1e-4),
            max_iter=20000,
        ),
        penalty="l2",
        alpha=0.5,
        lam=best_lam_l2
    )
    best_l2_log_model.fit(X_train.to_numpy(), y_train.to_numpy())

    test_err_l1 = misclassification_error(
        y_test.to_numpy(),
        best_l1_log_model.predict(X_test.to_numpy())
    )

    test_err_l2 = misclassification_error(
        y_test.to_numpy(),
        best_l2_log_model.predict(X_test.to_numpy())
    )

    print(f"Best lambda for l1 model is {best_lam_l1} and best test error is {np.round(test_err_l1,2)}")
    print(f"Best lambda for l2 model is {best_lam_l2} and best test error is {np.round(test_err_l2,2)}")


if __name__ == '__main__':

    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
