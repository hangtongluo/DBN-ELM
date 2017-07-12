# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:25:20 2017

@author: Administrator
"""

#==============================================================================
# from abc import ABCMeta, abstractmethod
# import numpy as np
# from scipy.linalg import pinv2
# from sklearn.utils import as_float_array, check_random_state  #, atleast2d_or_csr
# from sklearn.utils.extmath import safe_sparse_dot
# from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
# from sklearn.preprocessing import LabelBinarizer
#==============================================================================

from abc import ABCMeta, abstractmethod
from math import sqrt
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state  #, atleast2d_or_csr
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from scipy.linalg import pinv2
from sklearn.utils import as_float_array
from sklearn.preprocessing import LabelBinarizer
           
__all__ = ["SimpleRandomHiddenLayer",
           "ELMRegressor",
           "ELMClassifier",
           "SimpleELMRegressor",
           "SimpleELMClassifier"]

def atleast2d_or_csr(X):
    return X

# Abstract Base Class for random hidden layers
class BaseRandomHiddenLayer(BaseEstimator, TransformerMixin):
    __metaclass__ = ABCMeta

    _internal_activation_funcs = dict()

    # take n_hidden and random_state, init components_ and
    # input_activations_
    def __init__(self, n_hidden=20, random_state=0, activation_func=None,
                 activation_args=None):

        self.n_hidden = n_hidden
        self.random_state = random_state
        self.activation_func = activation_func
        self.activation_args = activation_args

        self.components_ = dict()
        self.input_activations_ = None

        # keyword args for internally defined funcs
        self._extra_args = dict()

    @abstractmethod
    def _generate_components(self, X):
        """Generate components of hidden layer given X"""

    @abstractmethod
    def _compute_input_activations(self, X):
        """Compute input activations given X"""

    # compute input activations and pass them
    # through the hidden layer transfer functions
    # to compute the transform
    def _compute_hidden_activations(self, X):
        """Compute hidden activations given X"""

        self._compute_input_activations(X)

        acts = self.input_activations_

        if (callable(self.activation_func)):
            args_dict = self.activation_args if (self.activation_args) else {}
            X_new = self.activation_func(acts, **args_dict)
        else:
            func_name = self.activation_func
            func = self._internal_activation_funcs[func_name]

            X_new = func(acts, **self._extra_args)

        return X_new

    # perform fit by generating random components based
    # on the input array
    def fit(self, X, y=None):
        """Generate a random hidden layer.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training set: only the shape is used to generate random component
            values for hidden units

        y : is not used: placeholder to allow for usage in a Pipeline.

        Returns
        -------
        self
        """
        X = atleast2d_or_csr(X)

        self._generate_components(X)

        return self

    # perform transformation by calling compute_hidden_activations
    # (which will normally call compute_input_activations first)
    def transform(self, X, y=None):
        """Generate the random hidden layer's activations given X as input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            Data to transform

        y : is not used: placeholder to allow for usage in a Pipeline.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_components]
        """
        X = atleast2d_or_csr(X)

        if (self.components_ is None):
            raise ValueError('No components initialized')

        return self._compute_hidden_activations(X)


class SimpleRandomHiddenLayer(BaseRandomHiddenLayer):
    """Simple Random Hidden Layer transformer

    Creates a layer of units as a specified functions of an activation
    value determined by the dot product of the input and a random vector
    plus a random bias term:

     f(a), s.t. a = dot(x, hidden_weights) + bias

    and transfer function f() which defaults to numpy.tanh if not supplied
    but can be any callable that returns an array of the same shape as
    its argument (input activation array, shape [n_samples, n_hidden])

    Parameters
    ----------
    `n_hidden` : int, optional (default=20)
        Number of units to generate

    `activation_func` : {callable, string} optional (default='tanh')
        Function used to transform input activation
        It must be one of 'tanh', 'sine', 'tribas', 'sigmoid', 'hardlim' or
        a callable.  If none is given, 'tanh' will be used. If a callable
        is given, it will be used to compute the hidden unit activations.

    `activation_args` : dictionary, optional (default=None)
        Supplies keyword arguments for a callable activation_func

    `random_state`  : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        hidden unit weights at fit time.

    Attributes
    ----------
    `input_activations_` : numpy array of shape [n_samples, n_hidden]
        Array containing dot(x, hidden_weights) + bias for all samples

    `components_` : dictionary containing two keys:
        `bias_weights_`   : numpy array of shape [n_hidden]
        `hidden_weights_` : numpy array of shape [n_features, n_hidden]

    See Also
    --------
    ELMRegressor, ELMClassifier, SimpleELMRegressor, SimpleELMClassifier,
    RBFRandomHiddenLayer
    """

    #
    # internal transfer function (RBF) definitions
    #

    # triangular transfer function
    _tribas = (lambda x: np.clip(1.0 - np.fabs(x), 0.0, 1.0))

    # sigmoid transfer function
    _sigmoid = (lambda x: 1.0/(1.0 + np.exp(-x)))

    # hard limit transfer function
    _hardlim = (lambda x: np.array(x > 0.0, dtype=float))

    # internal transfer function table
    _internal_activation_funcs = {'sine': np.sin,
                                  'tanh': np.tanh,
                                  'tribas': _tribas,
                                  'sigmoid': _sigmoid,
                                  'hardlim': _hardlim
                                  }

    # default setup, plus initialization of activation_func
    def __init__(self, n_hidden=20, random_state=None,
                 activation_func='tanh', activation_args=None):

        super(SimpleRandomHiddenLayer, self).__init__(n_hidden,
                                                      random_state,
                                                      activation_func,
                                                      activation_args)

        if (isinstance(self.activation_func, str)):
            func_names = self._internal_activation_funcs.keys()
            if (self.activation_func not in func_names):
                msg = "unknown transfer function '%s'" % self.activation_func
                raise ValueError(msg)

    def _generate_components(self, X):
        """Generate components of hidden layer given X"""

        rand_state = check_random_state(self.random_state)
        n_features = X.shape[1]

        b_size = self.n_hidden
        hw_size = (n_features, self.n_hidden)

        self.components_['biases'] = rand_state.normal(size=b_size)
        self.components_['weights'] = rand_state.normal(size=hw_size)

    def _compute_input_activations(self, X):
        """Compute input activations given X"""

        b = self.components_['biases']
        w = self.components_['weights']

        self.input_activations_ = safe_sparse_dot(X, w)
        self.input_activations_ += b


# BaseELM class, regressor and hidden_layer attributes
# and provides defaults for docstrings
class BaseELM(BaseEstimator):
    """
    Base class for ELMs.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    __metaclass__ = ABCMeta

    def __init__(self, hidden_layer, regressor):
        self.regressor = regressor
        self.hidden_layer = hidden_layer

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """

    @abstractmethod
    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """

class ELMRegressor(BaseELM, RegressorMixin):
    """
    ELMRegressor is a regressor based on the Extreme Learning Machine.

    An Extreme Learning Machine (ELM) is a single layer feedforward
    network with a random hidden layer components and ordinary linear
    least squares fitting of the hidden->output weights by default.
    [1][2]

    Parameters
    ----------
    `hidden_layer` : random_hidden_layer instance, optional
        (default=SimpleRandomHiddenLayer(random_state=0))

    `regressor`    : regressor instance, optional (default=None)
        If provided, this object is used to perform the regression from hidden
        unit activations to the outputs and subsequent predictions.  If not
        present, an ordinary linear least squares fit is performed

    Attributes
    ----------
    `coefs_` : numpy array
        Fitted regression coefficients if no regressor supplied.

    `fitted_` : bool
        Flag set when fit has been called already.

    `hidden_activations_` : numpy array of shape [n_samples, n_hidden]
        Hidden layer activations for last input.

    See Also
    --------
    RBFRandomHiddenLayer, SimpleRandomHiddenLayer, ELMRegressor, ELMClassifier
    SimpleELMRegressor, SimpleELMClassifier

    References
    ----------
    .. [1] http://www.extreme-learning-machines.org
    .. [2] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
          Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
              2006.
    """

    def __init__(self,
                 hidden_layer=SimpleRandomHiddenLayer(random_state=0),
                 regressor=None):

        super(ELMRegressor, self).__init__(hidden_layer, regressor)

        self.coefs_ = None
        self.fitted_ = False
        self.hidden_activations_ = None

    def _fit_regression(self, y):
        """
        fit regression using internal linear regression
        or supplied regressor
        """
        if (self.regressor is None):
            self.coefs_ = safe_sparse_dot(pinv2(self.hidden_activations_), y)
        else:
            self.regressor.fit(self.hidden_activations_, y)

        self.fitted_ = True

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """
        # fit random hidden layer and compute the hidden layer activations
        self.hidden_activations_ = self.hidden_layer.fit_transform(X)

        # solve the regression from hidden activations to outputs
        self._fit_regression(as_float_array(y, copy=True))

        return self

    def _get_predictions(self, X):
        """get predictions using internal least squares/supplied regressor"""
        if (self.regressor is None):
            preds = safe_sparse_dot(self.hidden_activations_, self.coefs_)
        else:
            preds = self.regressor.predict(self.hidden_activations_)

        return preds

    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        if (not self.fitted_):
            raise ValueError("ELMRegressor not fitted")

        # compute hidden layer activations
        self.hidden_activations_ = self.hidden_layer.transform(X)

        # compute output predictions for new hidden activations
        predictions = self._get_predictions(X)

        return predictions        
        
class ELMClassifier(BaseELM, ClassifierMixin):
    """
    ELMClassifier is a classifier based on the Extreme Learning Machine.

    An Extreme Learning Machine (ELM) is a single layer feedforward
    network with a random hidden layer components and ordinary linear
    least squares fitting of the hidden->output weights by default.
    [1][2]

    Parameters
    ----------
    `hidden_layer` : random_hidden_layer instance, optional
        (default=SimpleRandomHiddenLayer(random_state=0))

    `regressor`    : regressor instance, optional (default=None)
        If provided, this object is used to perform the regression from hidden
        unit activations to the outputs and subsequent predictions.  If not
        present, an ordinary linear least squares fit is performed

    Attributes
    ----------
    `classes_` : numpy array of shape [n_classes]
        Array of class labels

    `binarizer_` : LabelBinarizer instance
        Used to transform class labels

    `elm_regressor_` : ELMRegressor instance
        Performs actual fit of binarized values

    See Also
    --------
    RBFRandomHiddenLayer, SimpleRandomHiddenLayer, ELMRegressor, ELMClassifier
    SimpleELMRegressor, SimpleELMClassifier

    References
    ----------
    .. [1] http://www.extreme-learning-machines.org
    .. [2] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
              Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
              2006.
    """
    def __init__(self,
                 hidden_layer=SimpleRandomHiddenLayer(random_state=0),
                 regressor=None):

        super(ELMClassifier, self).__init__(hidden_layer, regressor)

        self.classes_ = None
        self.binarizer_ = LabelBinarizer(-1, 1)
        self.elm_regressor_ = ELMRegressor(hidden_layer, regressor)

    def decision_function(self, X):
        """
        This function return the decision function values related to each
        class on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]

        Returns
        -------
        C : array of shape [n_samples, n_classes] or [n_samples,]
            Decision function values related to each class, per sample.
            In the two-class case, the shape is [n_samples,]
        """
        return self.elm_regressor_.predict(X)

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """
        self.classes_ = np.unique(y)

        y_bin = self.binarizer_.fit_transform(y)

        self.elm_regressor_.fit(X, y_bin)
        return self

    def predict(self, X):
        """Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        raw_predictions = self.decision_function(X)
        class_predictions = self.binarizer_.inverse_transform(raw_predictions)

        return class_predictions
		

		
#	 def predict_proba(self, X):
#        """Probability estimates.
#        Parameters
#       ----------
#        X : {array-like, sparse matrix}, shape (n_samples, n_features)
#           The input data.
#        Returns
#        -------
#        y_prob : array-like, shape (n_samples, n_classes)
#            The predicted probability of the sample for each class in the
#            model, where classes are ordered as they are in `self.classes_`.
#        """
#        y_pred = self.decision_function(X)
#        y_pred = y_pred.ravel()
#
#        return np.vstack([1 - y_pred, y_pred]).T


# ELMRegressor with default SimpleRandomHiddenLayer
class SimpleELMRegressor(BaseEstimator, RegressorMixin):
    """
    SimpleELMRegressor is a regressor based on the Extreme Learning Machine.

    An Extreme Learning Machine (ELM) is a single layer feedforward
    network with a random hidden layer components and ordinary linear
    least squares fitting of the hidden->output weights by default.
    [1][2]

    SimpleELMRegressor is a wrapper for an ELMRegressor that uses a
    SimpleRandomHiddenLayer and passes the __init__ parameters through
    to the hidden layer generated by the fit() method.

    Parameters
    ----------
    `n_hidden` : int, optional (default=20)
        Number of units to generate in the SimpleRandomHiddenLayer

    `activation_func` : {callable, string} optional (default='tanh')
        Function used to transform input activation
        It must be one of 'tanh', 'sine', 'tribas', 'sigmoid', 'hardlim' or
        a callable.  If none is given, 'tanh' will be used. If a callable
        is given, it will be used to compute the hidden unit activations.

    `activation_args` : dictionary, optional (default=None)
        Supplies keyword arguments for a callable activation_func

    `random_state`  : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        hidden unit weights at fit time.

    Attributes
    ----------
    `elm_regressor_` : ELMRegressor object
        Wrapped object that actually performs the fit.

    See Also
    --------
    RBFRandomHiddenLayer, SimpleRandomHiddenLayer, ELMRegressor, ELMClassifier
    SimpleELMRegressor, SimpleELMClassifier

    References
    ----------
    .. [1] http://www.extreme-learning-machines.org
    .. [2] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
          Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
              2006.
    """

    def __init__(self, n_hidden=20,
                 activation_func='tanh', activation_args=None,
                 random_state=None):

        self.n_hidden = n_hidden
        self.activation_func = activation_func
        self.activation_args = activation_args
        self.random_state = random_state

        self.elm_regressor_ = None

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """
        rhl = SimpleRandomHiddenLayer(n_hidden=self.n_hidden,
                                      activation_func=self.activation_func,
                                      activation_args=self.activation_args,
                                      random_state=self.random_state)

        self.elm_regressor_ = ELMRegressor(hidden_layer=rhl)
        self.elm_regressor_.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        if (self.elm_regressor_ is None):
            raise ValueError("SimpleELMRegressor not fitted")

        return self.elm_regressor_.predict(X)        
        
# ELMClassifier with default SimpleRandomHiddenLayer
class SimpleELMClassifier(BaseEstimator, ClassifierMixin):
    """
    SimpleELMClassifier is a classifier based on the Extreme Learning Machine.

    An Extreme Learning Machine (ELM) is a single layer feedforward
    network with a random hidden layer components and ordinary linear
    least squares fitting of the hidden->output weights by default.
    [1][2]

    SimpleELMClassifier is a wrapper for an ELMClassifier that uses a
    SimpleRandomHiddenLayer and passes the __init__ parameters through
    to the hidden layer generated by the fit() method.

    Parameters
    ----------
    `n_hidden` : int, optional (default=20)
        Number of units to generate in the SimpleRandomHiddenLayer

    `activation_func` : {callable, string} optional (default='tanh')
        Function used to transform input activation
        It must be one of 'tanh', 'sine', 'tribas', 'sigmoid', 'hardlim' or
        a callable.  If none is given, 'tanh' will be used. If a callable
        is given, it will be used to compute the hidden unit activations.

    `activation_args` : dictionary, optional (default=None)
        Supplies keyword arguments for a callable activation_func

    `random_state`  : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        hidden unit weights at fit time.

    Attributes
    ----------
    `classes_` : numpy array of shape [n_classes]
        Array of class labels

    `elm_classifier_` : ELMClassifier object
        Wrapped object that actually performs the fit

    See Also
    --------
    RBFRandomHiddenLayer, SimpleRandomHiddenLayer, ELMRegressor, ELMClassifier
    SimpleELMRegressor, SimpleELMClassifier

    References
    ----------
    .. [1] http://www.extreme-learning-machines.org
    .. [2] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
          Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
              2006.
    """

    def __init__(self, n_hidden=20,
                 activation_func='tanh', activation_args=None,
                 random_state=None):

        self.n_hidden = n_hidden
        self.activation_func = activation_func
        self.activation_args = activation_args
        self.random_state = random_state

        self.elm_classifier_ = None

    @property
    def classes_(self):
        return self.elm_classifier_.classes_

    def decision_function(self, X):
        """
        This function return the decision function values related to each
        class on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]

        Returns
        -------
        C : array of shape [n_samples, n_classes] or [n_samples,]
            Decision function values related to each class, per sample.
            In the two-class case, the shape is [n_samples,]
        """
        return self.elm_classifier_.decision_function(X)

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """
        rhl = SimpleRandomHiddenLayer(n_hidden=self.n_hidden,
                                      activation_func=self.activation_func,
                                      activation_args=self.activation_args,
                                      random_state=self.random_state)

        self.elm_classifier_ = ELMClassifier(hidden_layer=rhl)
        self.elm_classifier_.fit(X, y)

        return self

    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        if (self.elm_classifier_ is None):
            raise ValueError("SimpleELMClassifier not fitted")

        return self.elm_classifier_.predict(X)


