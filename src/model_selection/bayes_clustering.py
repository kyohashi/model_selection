# -*- coding:utf-8 -*-

from .utils.check_datashape import check_array
from abc import ABCMeta, abstractmethod
import pymc3 as pm
import numpy as np


class _BaseBC(metaclass=ABCMeta):
    """ Abtract base class for Bayes Clustering model selector"""

    @abstractmethod
    def _check_X(self, X):
        """ check if input X is appropriate """
        pass

    @abstractmethod
    def fit(self, X):
        """ fit the model to data """
        pass


class GMM_BC(_BaseBC):
    """
    Bayes Clustering with Gaussian Mixture Model (GMM).

    The hyperparameter K standing for the number of clusters is estimated by WAIC.
    For details on WAIC, see the paper:
        "A Widely Applicable Bayesian Information Criterion
         JOURNAL OF MACHINE LEARNING RESEARCH, Vol.14, page. 867-897, 2013"

    Attributes
    ----------
    K_cand: the candidates of K to be calculated IC
    X : 2D array (n_samples, n_features)

    Model settings and Examples
    ----------------------------
    See https://github.com/kyohashi/model_selection/blob/master/gmm_usecase.ipynb
    """

    def __init__(self, K_cand=np.arange(2, 5+1), n_samples=2000):
        self.K_cand = K_cand
        self.n_samples = n_samples

        # To be stored set [K, MCMC samples] over K_cand
        self.result = {}

    def _check_X(self, X):
        return check_array(X)

    def fit(self, X):
        return self._fit(X)

    def _fit(self, X):
        # check if input X is 2D array
        X = self._check_X(X)

        # model evaluation by WAIC over K_cand
        for K in self.K_cand:
            print('MCMC sampling: K={}'.format(K))
            self._model_eval(X, K, self.n_samples)

    def _multivariate_normal_dist(self, init_mu, suffix=""):
        if not isinstance(suffix, str):
            suffix = str(suffix)
        data_dim = len(init_mu)

        # prior of covariance
        sd_dist = pm.HalfCauchy.dist(beta=2.5)
        packed_chol = pm.LKJCholeskyCov(
            'cov'+suffix, eta=2, n=data_dim, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(
            data_dim, packed_chol, lower=True)
        # prior of mean
        mu = pm.MvNormal('mu'+suffix, mu=0,
                         cov=np.eye(data_dim), shape=data_dim)
        return pm.MvNormal.dist(mu, chol=chol)

    def _model_eval(self, X, K, n_samples):
        # setup model
        with pm.Model() as model:
            data_dim = X.shape[1]
            # prior of mixture ratio
            w = pm.Dirichlet('w', a=np.ones(K))
            # setup the likelihood
            init_mu = np.zeros(data_dim)
            components = [self._multivariate_normal_dist(
                init_mu, suffix=k) for k in range(K)]
            like = pm.Mixture(
                'like', w=w, comp_dists=components, observed=X)

        # fit model
        with model:
            trace = pm.sample(2000, step=pm.NUTS(),
                              start=pm.find_MAP(), tune=1000)

        # store the result
        self.result['K='+str(K)] = trace

    def compare(self, trace, ic='waic', scale='deviance'):
        return pm.compare(trace, ic=ic, scale=scale)
