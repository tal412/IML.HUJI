import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data_frame = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    data_frame = data_frame[data_frame["Temp"] > -70]
    data_frame["DayOfYear"] = data_frame["Date"].dt.day_of_year

    return data_frame


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data_frame = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data_frame = data_frame[data_frame["Country"] == "Israel"]
    qs_2_fig = px.scatter(israel_data_frame,
                          title="Temperature as a function of day of year in Israel",
                          x="DayOfYear",
                          y="Temp",
                          color=israel_data_frame.Year.astype(str))
    qs_2_fig.show()

    # Question 3 - Exploring differences between countries
    # raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()
