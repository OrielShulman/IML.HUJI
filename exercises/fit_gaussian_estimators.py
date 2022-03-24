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
    samples_1, univariate_1 = _Q1(mean=mean, var=var)

    # Question 2 - Empirically showing sample mean is consistent
    _Q2(mean=mean, var=var, samples=samples_1)

    # Question 3 - Plotting Empirical PDF of fitted model
    # _Q3(samples=samples_1, univariate=univariate_1)


def _Q1(mean: float, var: float) -> Tuple[np.ndarray, UnivariateGaussian]:
    samples = np.random.normal(mean, var, size=1000)
    fit = UnivariateGaussian()
    fit.fit(samples)
    print((fit.mu_, fit.var_))
    return samples, fit


def _Q2(mean: float, var: float, samples: np.ndarray) -> None:
    fit_2 = UnivariateGaussian()
    size_range = range(10, 1001, 10)
    samples_2 = [samples[:i] for i in range(10, samples.size + 1, 10)]

    # samples_2 = np.array([np.random.normal(mean, var, size=i) for i in size_range], dtype=np.ndarray)
    estimated_exp = np.array([np.abs(fit_2.fit(sample).mu_ - mean) for sample in samples_2])

    f = go.Figure(
        [go.Scatter(x=[i for i in size_range], y=estimated_exp, mode='lines', name="Absolute value"),
         go.Scatter(x=[i for i in size_range], y=np.zeros(estimated_exp.shape), mode='lines', name='0')],
        layout=go.Layout(
            title=r"$\text{Absolute distance between the estimated and true value of the expectation - 10}$",
            xaxis_title=r"$\text{ - number of samples}$", yaxis_title=r"$|\mu - 10|$", height=300))
    f.show()


def _Q3(samples: np.ndarray, univariate: UnivariateGaussian) -> None:
    samples_PDF = univariate.pdf(samples)

    title = "Empirical PDF function under the fitted model for mu = {mu}, var = {var}"\
        .format(mu=format(univariate.mu_, '.3f'), var=format(univariate.var_, '.3f'))

    x_title = "ample value"
    y_title = "PDF of sample"
    f = go.Figure(
        [go.Scatter(x=samples, y=samples_PDF, mode='markers', name="sample PDF")],
        layout=go.Layout(title=title, xaxis_title=x_title, yaxis_title=y_title, height=300))
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

    # samples = np.arange(0, 120)
    # print(f"samples of size {samples.size}:\n {samples}\n")
    #
    # print(f"samples[:3]:\n{samples[:3]}\n")
    #
    # print(f"slices:\n{[i for i in range(10, samples.size + 1, 10)]}\n")
    #
    # samples_2 = [samples[:i] for i in range(10, samples.size + 1, 10)]
    #
    # # print(f"sliced samples:\n{samples_2}\n")
    # for i, slice in enumerate(samples_2):
    #     print(f"arr {i+1}:\n{slice}\n")
