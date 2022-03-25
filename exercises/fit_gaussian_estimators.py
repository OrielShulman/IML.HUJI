from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from typing import Tuple

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    mean = 10
    var = 1

    # Question 1 - Draw samples and print fitted model
    samples, univariate_g = _Q1(mean=mean, var=var)

    # Question 2 - Empirically showing sample mean is consistent
    _Q2(mean=mean, var=var, samples=samples)

    # Question 3 - Plotting Empirical PDF of fitted model
    _Q3(samples=samples, univariate=univariate_g)


def _Q1(mean: float, var: float) -> Tuple[np.ndarray, UnivariateGaussian]:
    """
    Using numpy.random.normal draw 1000 samples x1,..., x1000 ~ N (10,1) and fit a
    univariate Gaussian. Print the estimated expectation and variance.

    Output format should be (expectation, variance).
    """
    samples = np.random.normal(mean, var, size=1000)
    fit = UnivariateGaussian()
    fit.fit(samples)
    print((fit.mu_, fit.var_))
    return samples, fit


def _Q2(mean: float, var: float, samples: np.ndarray) -> None:
    """
    Over previously drawn samples, fit a series of models of increasing samples size: 10, 20,...,100,
    110,...1000. Plot the absolute distance between the estimated- and true value of the expectation,
    as a function of the sample size. Provide meaningful axis names and title.
    """
    fit_2 = UnivariateGaussian()
    size_range = range(10, 1001, 10)
    samples_2 = [samples[:i] for i in range(10, samples.size + 1, 10)]
    estimated_exp = np.array([np.abs(fit_2.fit(sample).mu_ - mean) for sample in samples_2])

    f_title = r"$\text{Absolute distance between the estimated and true value of the expectation - 10}$"
    x_label = r"$\text{ - number of samples}$"
    y_label = r"$|\mu - 10|$"
    f = go.Figure(
        [go.Scatter(x=[i for i in size_range], y=estimated_exp, mode='lines', name="Absolute value"),
         go.Scatter(x=[i for i in size_range], y=np.zeros(estimated_exp.shape), mode='lines', name='0')],
        layout=go.Layout(title=f_title, xaxis_title=x_label, yaxis_title=y_label, height=300))
    f.show()


def _Q3(samples: np.ndarray, univariate: UnivariateGaussian) -> None:
    """
    Compute the PDF of the previously drawn samples using the model fitted in question 1.
    Plot the empirical PDF function under the fitted model. That is, create a scatter plot with the
    ordered sample values along the x-axis and their PDFs (using the UnivariateGaussian.pdf
    function) along the y-axis. Provide meaningful axis names and title. What are you expecting
    to see in the plot?

    """
    samples_PDF = univariate.pdf(samples)

    f_title = f"Empirical PDF function under the fitted model for mu = {format(univariate.mu_, '.3f')}, " \
              f"var = {format(univariate.var_, '.3f')}"
    # .format(mu=format(univariate.mu_, '.3f'), var=format(univariate.var_, '.3f'))
    x_title = "ordered sample values"
    y_title = "PDF of sample"
    f = go.Figure(
        [go.Scatter(x=samples, y=samples_PDF, mode='markers', name="sample PDF")],
        layout=go.Layout(title=f_title, xaxis_title=x_title, yaxis_title=y_title, height=300))
    f.show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()

