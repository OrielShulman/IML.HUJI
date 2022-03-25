from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet

ERR_NOT_FITTED = "Estimator must first be fitted before calling `pdf` function"

# MG:
# fit:
# - axis=0 | axis=1 (cols)
# - rowvar=False | rowvar=True (for rows variables)
# pdf:
# - float return value?
# - used X.size instead of len(self.mu_)
# l_l:
# - used X.shape[0] instead of len(mu)
# - used mu.size instead of len(X)


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
        # equations 1.2 & 1.3 in course material
        # self.mu_ = np.divide(np.sum(X), X.size)
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

        #  1.1.2 in course materials
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
        # equation 1.6 from course material

        # numerator = np.prod(np.exp(-np.divide(np.square(X.T - mu), 2 * sigma)))
        # denominator = np.power(2 * np.pi * sigma, 0.5 * X.size)
        # return np.log(np.divide(numerator, denominator))

        # applying natural log on the LL:
        return -0.5 * (X.size * np.log(2 * np.pi * sigma) + (np.sum(np.square(X - mu)) / sigma))


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

        self.mu_ = X.mean(axis=0)

        #  1.2.4 in course materials
        self.cov_ = np.cov(rowvar=False)
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

        # TODO: delete later
        print(f"pdf:\nlen(self.mu_) -->  X.size\nX.size = {X.size}, len(self.mu_) = {len(self.mu_)}\n")
        assert X.size == len(self.mu_), "X.size == self.mu_ (Delete this later)"

        #  1.2.5 in course materials
        d = X.size  # number of random variables
        numerator = np.exp(-0.5 * (X - self.mu_).T @ inv(self.cov_) @ (X - self.mu_))
        denominator = np.sqrt(np.power(2 * np.pi, d) * det(self.cov_))

        return np.divide(numerator, denominator)

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
        # TODO: delete later
        print(f"log_likelihood:\n"
              f"len(mu) --> X.shape[0]\n"
              f"len(X) --> mu.size\n"
              f"len(mu) = {len(mu)}, X.shape[0] = {X.shape[0]} | len(X) = {len(X)}, mu.size = {mu.size}")

        print(f"|||||| {X.shape[0]} <-> {mu.size} | m <-> p ||||||")

        assert X.shape[0] == mu.size, "X.size == self.mu_ (Delete this later)"

        #  derived from 1.2.5 in course materials
        m = X.shape[0]  # number of random variables (n_samples)
        p = mu.size  # (n_features)

        e1 = -0.5 * m * p * np.log(2 * np.pi)

        e2 = -0.5 * m * np.log(det(cov))

        e3 = -0.5 * np.sum((X - mu).T @ inv(cov) @ (X - mu))  # np.sum([(x-mu).T @ inv(cov) @ (x-mu) for x in X])

        return e1 + e2 + e3


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
    sample_test_7 = np.arange(34)
    print(f"{'-' * 30}\nshape:\n{sample_test_7.shape}\nsize:\n{sample_test_7.size}\nlen:\n{len(sample_test_7)}\n")

    x = np.ones((3, 4))
    print(x)
    print(len(x))
    print(x.size)
