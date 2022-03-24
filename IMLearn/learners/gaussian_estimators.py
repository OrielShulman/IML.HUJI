from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet

ERR_NOT_FITTED = "Estimator must first be fitted before calling `pdf` function"


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> None:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # self.mu_ = np.divide(np.sum(X), len(X))

        self.mu_ = X.mean()

        if self.biased_:
            self.var_ = np.var(X)
        else:
            self.var_ = np.var(X, ddof=1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(ERR_NOT_FITTED)
        # TODO: test
        numerator = np.exp(-(np.divide(np.square(X - self.mu_), (2 * self.var_))))
        denominator = np.sqrt(2 * np.pi * self.var_)
        return np.divide(numerator, denominator).reshape(X.shape)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """

        # num_after_log = -0.5 * np.sum(np.divide(np.square(X.T - mu), 2 * sigma)) * X.size
        # dom_after_log = -0.5 * np.log(2 * np.pi * sigma) * X.size
        # return  num_after_log + dom_after_log

        # TODO: from 9. and google log likelihood

        numerator = np.prod(np.exp(-np.divide(np.square(X.T - mu), 2 * sigma)))

        denominator = np.power(2 * np.pi * sigma, 0.5 * X.size)

        return np.log(np.divide(numerator, denominator))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self) -> None:
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # TODO: test

        self.mu_ = X.mean(axis=1)

        self.cov_ = np.var(X)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(ERR_NOT_FITTED)

        # TODO: not correct - might be the log likelihood
        d = 5  # TODO: ?

        numerator = np.prod(np.exp(-0.5 * (X - self.mu_).T @ inv(self.cov_) @ (X - self.mu_)))
        denominator = np.power(np.power(2 * np.pi, d) * det(self.cov_), 0.5 * X.size)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        raise NotImplementedError()


if __name__ == '__main__':
    # tests for UnivariateGaussian:

    sample_test_0 = np.random.randint(1, 15, size=20)
    print(f"{'-' * 30}\nsample 0:\n{sample_test_0}\nshape:{sample_test_0.shape}\n")

    numer = np.exp(-(np.divide(np.square(sample_test_0 - 1), (2 * 2))))
    deno = np.sqrt(2 * np.pi * 2)
    sample_test_1 = np.divide(numer, deno)
    print(f"{'-' * 30}\nsample 1:\n{sample_test_1}\nshape:\n{sample_test_1.shape}\n")

    sample_test_2 = sample_test_1.reshape(sample_test_0.shape)
    print(f"{'-' * 30}\nsample 2:\n{sample_test_2}\nshape:\n{sample_test_2.shape}\n")

    sample_test_3 = np.random.randint(1, 15, size=20)
    print(f"{'-' * 30}\nsample 3:\n{sample_test_3}\nshape:\n{sample_test_3.shape}\nsize:\n{sample_test_3.size}\n")

    sample_test_4 = np.random.randint(1, 15, (4, 5))
    print(f"{'-' * 30}\nsample 4:\n{sample_test_4}\nshape:\n{sample_test_4.shape}\nsize:\n{sample_test_4.size}\n")

    sample_test_5 = np.array([1, 3, 0])
    prod = np.prod(sample_test_5)
    print(f"{'-' * 30}\nsample 5:\n{sample_test_5}\nnp.prod:\n{prod}\n")

    sample_test_6 = np.random.normal(10, 1, size=10)
    print(f"{'-' * 30}\nsample 4:\n{sample_test_6}\nshape:\n{sample_test_6.shape}\nsize:\n{sample_test_6.size}\n")

    # tests for MultivariateGaussian:
