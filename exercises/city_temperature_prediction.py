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
    qs_2_fig_A = px.scatter(israel_data_frame,
                            title="Temperature as a function of day of year in Israel",
                            x="DayOfYear",
                            y="Temp",
                            color=israel_data_frame.Year.astype(str))
    qs_2_fig_A.show()

    temp_group_israel = israel_data_frame.groupby(['Month']).Temp.agg(['std'])
    qs_2_fig_B = px.bar(temp_group_israel)
    qs_2_fig_B.show()

    # Question 3 - Exploring differences between countries
    temp_group_country =\
        data_frame.groupby(['Country', 'Month'],
                           as_index=False
                           ).agg(dict(Temp=['mean', 'std']))
    qs_3_fig = px.line(
        x=temp_group_country['Month'],
        y=temp_group_country[('Temp', 'mean')],
        color=temp_group_country['Country'],
        labels=dict(x='Month', y='Mean Temp', color='Country'),
        error_y=temp_group_country[('Temp', 'std')],
        title='Temp\'s mean by month with std err'
    )

    qs_3_fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y =\
        split_train_test(israel_data_frame, israel_data_frame[['Temp']])

    k_vals = []
    loss_vals_q_4 = []
    for i in range(1, 11):
        k_vals.append(i)
        pf = PolynomialFitting(k=i)
        pf.fit(np.asarray(train_x['DayOfYear']), np.asarray(train_y['Temp']))
        loss =\
            float("{:.2f}".format(
                pf.loss(np.asarray(test_x['DayOfYear']), np.asarray(test_y['Temp']))
            ))
        loss_vals_q_4.append(loss)
        print(loss)

    qs_4_fig = px.bar(
        x=k_vals,
        y=loss_vals_q_4,
        labels=dict(
            x='Value of k',
            y='Loss of our fitted polynomial'
        )
    )
    qs_4_fig.show()

    # Question 5 - Evaluating fitted model on different countries
    loss_vals_q_5 = []
    pf = PolynomialFitting(k=5)
    pf.fit(
        np.asarray(israel_data_frame['DayOfYear']),
        np.asarray(israel_data_frame['Temp'])
    )

    countries = ['Israel', 'Jordan', 'The Netherlands', 'South Africa']
    for country in countries:

        data = data_frame[data_frame['Country'] == country]
        loss = pf.loss(
            np.asarray(data['DayOfYear']),
            np.asarray(data['Temp'])
        )

        loss_vals_q_5.append(loss)

    qs_5_fig = px.bar(
        x=countries,
        y=loss_vals_q_5,
        title="Israel err over other countries",
        labels=dict(
            x='Country',
            y='Loss val of a model trained with Israel\'s data'
        ))
    qs_5_fig.show()



