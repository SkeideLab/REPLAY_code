# -*- coding: utf-8 -*-
"""
A class for general Cross-Decoding Analysis.

@author: Christopher Postzich
@github: mcpost
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_array
from tqdm import trange


class CrossDecoding_MEEG(BaseEstimator, ClassifierMixin):
    """
    Cross-temporal decoding for M/EEG data with scikit-learn compatibility. 
    
    This class trains classifiers on neural data at specified time points and 
    applies them to decode activity across all time points in test data.
    Supports single time point training, baseline augmentation, and various
    classifier output modes.
    
    Parameters:
    -----------
    base_estimator : sklearn estimator
        Base classifier to use for decoding. Must have fit/predict methods.
    training_time : float or None, default=None
        Specific time point (in seconds) for training. If None, trains on all timepoints.
    baseline_time : float or None, default=None
        Time point (in seconds) for baseline period. If provided, adds baseline
        trials with label 0 to training data.
    times : array-like
        Time points corresponding to the last dimension of input data.
    include_zero : bool, default=True
        Whether to include zero labels in output classes.
    n_jobs : int, default=1
        Number of parallel jobs (not implemented yet).
    random_state : int or None, default=None
        Random state for reproducible baseline trial selection.
    verbose : bool, default=True
        Whether to show progress bars during fitting/prediction.
        
    Attributes:
    -----------
    classes_ : array
        Unique class labels.
    decoders_ : dict
        Dictionary of trained classifiers for each class/timepoint.
    training_indices_ : array
        Time indices used for training.
    is_fitted_ : bool
        Whether the estimator has been fitted.
    """
    
    def __init__(self, base_estimator, train_times=None, test_times=None,
                 training_timepoint=None, baseline_timepoint=None, 
                 include_zero=True, n_jobs=1, random_state=None, 
                 verbose=True):
        self.base_estimator = base_estimator
        self.train_times = train_times
        self.training_timepoint = training_timepoint
        self.baseline_timepoint = baseline_timepoint
        self.test_times = test_times
        self.include_zero = include_zero
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
    def _validate_input(self, X, y=None):
        """Validate input arrays and extract time information."""
        X = check_array(X, allow_nd=True)
        
        if X.ndim != 3:
            raise ValueError(f"X must be 3D array (n_samples, n_features, n_times), got {X.ndim}D")
            
        if y is not None:
            y = check_array(y, ensure_2d=False)
            if len(y) != X.shape[0]:
                raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} vs {len(y)}")
                
        return X, y
    
    def _get_time_indices(self, n_times):
        """Get time indices for training based on training_time parameter."""
        if self.train_times is None:
            if self.training_timepoint is not None:
                raise ValueError("times must be provided when training_time is specified")
            return np.arange(n_times)
        
        times_array = np.array(self.train_times)
        if len(times_array) != n_times:
            raise ValueError(f"Length of times ({len(times_array)}) must match last dimension of X ({n_times})")
            
        if self.training_timepoint is not None:
            # Find closest time point
            time_idx = np.argmin(np.abs(times_array - self.training_timepoint))
            return np.array([time_idx])
        else:
            return np.arange(n_times)
    
    def _prepare_training_data(self, X, y):
        """Prepare training data with optional baseline augmentation."""
        n_samples, n_features, n_times = X.shape
        
        # Get training time indices
        training_indices = self._get_time_indices(n_times)
        
        # Add baseline data if specified
        baseline_data = None
        if self.baseline_timepoint is not None:
            if self.train_times is None:
                raise ValueError("times must be provided when baseline_time is specified")
                
            baseline_idx = np.argmin(np.abs(np.array(self.train_times) - self.baseline_timepoint))
            
            # Select baseline trials (same number as original trials)
            rng = np.random.RandomState(self.random_state)
            baseline_trial_indices = rng.choice(n_samples, n_samples, replace=False)
            
            baseline_X = X[baseline_trial_indices, :, baseline_idx]
            baseline_y = np.zeros(len(baseline_trial_indices), dtype=y.dtype)
            baseline_data = (baseline_X, baseline_y)
        
        return training_indices, baseline_data
    
    def fit(self, X, y):
        """
        Fit the cross-temporal decoder.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features, n_times)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_input(X, y)
        n_samples, n_features, n_times = X.shape
        
        # Prepare training data
        training_indices, baseline_data = self._prepare_training_data(X, y)
        
        # Determine unique classes
        train_labels = y.copy()
        if baseline_data is not None:
            train_labels = np.concatenate([train_labels, baseline_data[1]])
        
        if self.include_zero:
            self.classes_ = np.unique(train_labels)
            _,self.classes_indices_ = np.unique(self.classes_, return_index=True)
        else:
            self.classes_ = np.unique(train_labels[train_labels > 0])
            _,self.classes_indices_ = np.unique(self.classes_, return_index=True)
        
        # Store training indices
        self.training_indices_ = training_indices
        self.n_training_times_ = len(training_indices)
        
        # Train classifiers
        self.decoders_ = {}
        
        # Progress bar setup
        desc = "Training classifiers"
        disable_tqdm = not self.verbose
        
        # Train one classifier per training timepoint
        for i, train_time_idx in enumerate(trange(len(training_indices), desc=desc, disable=disable_tqdm)):
            actual_time_idx = training_indices[train_time_idx]
            
            # Prepare training data for this timepoint
            train_X = X[:, :, actual_time_idx]
            train_y = y.copy()
            
            # Add baseline data if specified
            if baseline_data is not None:
                train_X = np.vstack([train_X, baseline_data[0]])
                train_y = np.concatenate([train_y, baseline_data[1]])
            
            # Train classifiers for this timepoint
            timepoint_decoders = {}
            
            # Train classifier
            timepoint_decoders['model'] = clone(self.base_estimator)
            timepoint_decoders['model'].fit(train_X, train_y)
            
            self.decoders_[actual_time_idx] = timepoint_decoders
        
        self.is_fitted_ = True
        return self
    
    def _get_prediction_method(self, method_name, train_time_idx):
        """Get the appropriate prediction method from trained classifiers for a specific training time."""
        methods = {}
        timepoint_decoders = self.decoders_[train_time_idx]
        
        decoder = timepoint_decoders['model']  
        if hasattr(decoder, method_name):
            methods['model'] = getattr(decoder, method_name)
        else:
            raise AttributeError(f"Base estimator does not have {method_name} method")
                
        return methods
    
    def _apply_prediction_method(self, X, method_name):
        """Apply prediction method across all timepoints."""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("This CrossDecoding_MEEG instance is not fitted yet.")
        
        X, _ = self._validate_input(X)
        n_samples, n_features, n_times = X.shape
        
        # Determine output shape based on training configuration
        if len(self.training_indices_) == 1:
            # training timepoint: output shape (n_samples, n_classes, n_times)
            if method_name == 'predict_proba':
                output_shape = (n_samples, len(self.classes_), n_times)
            elif method_name == 'decision_function':
                if len(self.classes_) == 2:
                    output_shape = (n_samples, n_times)
                else:
                    output_shape = (n_samples, len(self.classes_), n_times)
            else:  # predict
                output_shape = (n_samples, n_times)
        else:
            # Multiple training timepoints: cross-temporal matrix
            # Output shape: (n_samples, n_classes/1, n_training_times, n_test_times)
            if method_name == 'predict_proba':
                output_shape = (n_samples, len(self.classes_), len(self.training_indices_), n_times)
            elif method_name == 'decision_function':
                if len(self.classes_) == 2:
                    output_shape = (n_samples, len(self.training_indices_), n_times)
                else:
                    output_shape = (n_samples, len(self.classes_), len(self.training_indices_), n_times)
            else:  # predict
                output_shape = (n_samples, len(self.training_indices_), n_times)
        
        if method_name == 'predict':
            predictions = np.zeros(output_shape, dtype=int)
        else:
            predictions = np.zeros(output_shape)
        
        # Progress bar setup
        desc = f"Applying {method_name}"
        disable_tqdm = not self.verbose
        
        # Apply prediction for each training timepoint and test timepoint
        for train_idx, train_time_idx in enumerate(trange(len(self.training_indices_), 
                                                          desc=f"{desc} (training times)", 
                                                          disable=disable_tqdm)):
            actual_train_time_idx = self.training_indices_[train_time_idx]
            
            # Get prediction methods for this training timepoint
            prediction_methods = self._get_prediction_method(method_name, actual_train_time_idx)
            
            # Apply to each test timepoint
            for test_t in range(n_times):
                X_t = X[:, :, test_t]  # Shape: (n_samples, n_features)
                
                # Apply multi-class classifier in single model
                method = prediction_methods['model']
                pred_t = method(X_t)
                
                if len(self.training_indices_) == 1:
                    # Single training timepoint
                    if method_name == 'predict':
                        predictions[:, test_t] = pred_t[self.classes_indices_]
                    elif method_name == 'decision_function' and len(self.classes_) == 2:
                        predictions[:, test_t] = pred_t[self.classes_indices_]
                    else:
                        predictions[:, :, test_t] = pred_t[:,self.classes_indices_]
                else:
                    # Multiple training timepoints
                    if method_name == 'predict':
                        predictions[:, train_idx, test_t] = pred_t[self.classes_indices_]
                    elif method_name == 'decision_function' and len(self.classes_) == 2:
                        predictions[:, train_idx, test_t] = pred_t[self.classes_indices_]
                    else:
                        predictions[:, :, train_idx, test_t] = pred_t[:,self.classes_indices_]
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities across all timepoints.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features, n_times)
            Data to predict.
            
        Returns
        -------
        probabilities : array
            Predicted class probabilities for each sample and timepoint.
            Shape depends on training configuration:
            - Single training time: (n_samples, n_classes, n_times)
            - Multiple training times: (n_samples, n_classes, n_training_times, n_times)
        """
        return self._apply_prediction_method(X, 'predict_proba')
    
    def predict(self, X):
        """
        Predict classes across all timepoints.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features, n_times)
            Data to predict.
            
        Returns
        -------
        predictions : array
            Predicted classes for each sample and timepoint.
            Shape depends on training configuration:
            - Single training time: (n_samples, n_times) or (n_samples, n_classes, n_times)
            - Multiple training times: (n_samples, n_training_times, n_times) or 
                                     (n_samples, n_classes, n_training_times, n_times)
        """
        return self._apply_prediction_method(X, 'predict')
    
    def decision_function(self, X):
        """
        Compute decision function across all timepoints.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features, n_times)
            Data to predict.
            
        Returns
        -------
        decisions : array
            Decision function values for each sample and timepoint.
            Shape depends on training configuration:
            - Single training time: (n_samples, n_times) or (n_samples, n_classes, n_times)
            - Multiple training times: (n_samples, n_training_times, n_times) or
                                     (n_samples, n_classes, n_training_times, n_times)
        """
        return self._apply_prediction_method(X, 'decision_function')
    
    def transform(self, X):
        """
        Transform data using predict_proba (alias for compatibility).
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features, n_times)
            Data to transform.
            
        Returns
        -------
        transformed : array, shape (n_samples, n_classes, n_times)
            Transformed data (predicted probabilities).
        """
        return self.predict_proba(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'base_estimator': self.base_estimator,
            'train_times': self.train_times,
            'training_timepoint': self.training_timepoint,
            'baseline_timepoint': self.baseline_timepoint,
            'test_times': self.test_times,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
        
        if deep and hasattr(self.base_estimator, 'get_params'):
            base_params = self.base_estimator.get_params(deep=True)
            for key, value in base_params.items():
                params[f'base_estimator__{key}'] = value
                
        return params
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        base_params = {}
        estimator_params = {}
        
        for key, value in params.items():
            if key.startswith('base_estimator__'):
                base_params[key[16:]] = value  # Remove 'base_estimator__' prefix
            else:
                estimator_params[key] = value
        
        # Set base estimator parameters
        if base_params and hasattr(self.base_estimator, 'set_params'):
            self.base_estimator.set_params(**base_params)
        
        # Set this estimator's parameters
        for key, value in estimator_params.items():
            setattr(self, key, value)
            
        return self