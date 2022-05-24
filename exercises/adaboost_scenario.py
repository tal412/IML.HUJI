import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = 'browser'
pio.templates.default = "simple_white"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost = AdaBoost(DecisionStump, n_learners)
    adaBoost.fit(train_X, train_y)

    training_loss = []
    test_loss = []

    classifiers_indexes = np.arange(1, n_learners)

    for i in range(1, n_learners + 1):
        print(i)
        training_loss.append(adaBoost.partial_loss(train_X, train_y, i))
        test_loss.append(adaBoost.partial_loss(test_X, test_y, i))

    qs1_fig = go.Figure()

    qs1_fig.update_layout(
        title=f"Train and test errors of AdaBoost with noise = {noise}",
        xaxis_title="Learners count",
        yaxis_title="Error"
    )

    qs1_fig.add_scatter(
        x=classifiers_indexes,
        y=training_loss,
    )

    qs1_fig.add_scatter(
        x=classifiers_indexes,
        y=test_loss,
    )

    qs1_fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    qs2_fig = make_subplots(2, 2, subplot_titles=[f"AdaBoost ensemble trained up to iteration number {t}" for t in T])

    for index, num_of_learners in enumerate(T):
        def predict(g):
            return adaBoost.partial_predict(g, num_of_learners)

        currTrace = [
            decision_surface(predict, lims[0], lims[1], showscale=False),
            go.Scatter(
                x=test_X[:, 0],
                y=test_X[:, 1],
                mode="markers",
                marker=dict(color=test_y),
                showlegend=False)]

        qs2_fig.add_traces(currTrace, rows=int(index / 2) + 1, cols=(index % 2) + 1)

    qs2_fig.update_layout(title="AdaBoost ensemble trained for different numbers of iterations")
    qs2_fig.show()

    # Question 3: Decision surface of best performing ensemble
    min_loss_iter = np.argmin(test_loss).astype(int)
    acc = 1 - test_loss[min_loss_iter]

    def qs3_predict(g):
        return adaBoost.partial_predict(g, min_loss_iter)

    qs3_fig = go.Figure()
    qs3_fig.update_layout(
        title=f"Decision Boundary of AdaBoost with Ensemble size of {min_loss_iter} learners and {acc} accuracy",
    )

    qs3_trace = [
        decision_surface(qs3_predict, lims[0], lims[1], showscale=False),
        go.Scatter(
            x=test_X[:, 0],
            y=test_X[:, 1],
            mode="markers",
            marker=dict(color=test_y),
            showlegend=False)]

    qs3_fig.add_traces(qs3_trace)
    qs3_fig.show()

    # Question 4: Decision surface with weighted samples
    def qs4_predict(g):
        return adaBoost.partial_predict(g, min_loss_iter)

    D = 5 * (adaBoost.D_ / np.max(adaBoost.D_))
    qs4_fig = go.Figure()

    qs4_fig.add_traces(
        [decision_surface(qs4_predict, lims[0], lims[1], showscale=False),
            go.Scatter(
                x=train_X[:, 0],
                y=train_X[:, 1],
                mode="markers",
                showlegend=False,
                marker=dict(color=train_y.astype(int),
                            colorscale=[custom[0], custom[-1]],
                            line=dict(color="black", width=1),
                            size=D))
        ]
    )

    qs4_fig.update_layout(title=f"Decision surface with weighted samples")
    qs4_fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
