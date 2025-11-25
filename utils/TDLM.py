# -*- coding: utf-8 -*-
"""
Temporally Delayed Linear Model (TDLM)

This module implements the TDLM method for estimating sequentiality in 
probability time series data. The method uses lagged regression to detect
sequential reactivation patterns in neural data.

Created on Mon Mar 17 15:53:51 2025
@author: Christopher Postzich
"""

from itertools import permutations, combinations
import math
from typing import Optional, Tuple, List, Literal

import numpy as np
from scipy.linalg import toeplitz
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def create_transition_matrix(
    label: np.ndarray, 
    transitions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create forward and backward transition matrices from label and transition data.
    
    Parameters
    ----------
    label : np.ndarray
        Array of state labels
    transitions : np.ndarray
        Array of transition indices
        
    Returns
    -------
    tuple of np.ndarray
        Forward transition matrix and its transpose (backward matrix)
    """
    idx_corr = np.min(label)
    tm_forward = np.zeros((len(label), len(label)), dtype=int)
    
    for i in range(len(transitions)):
        if transitions[i]:
            tm_forward[i, transitions[i] - idx_corr] += 1
            
    return tm_forward, tm_forward.T


class TDLM:
    """
    Temporally Delayed Linear Model for sequenceness analysis.
    
    This class implements methods to detect sequential replay patterns in 
    multivariate time series data by comparing empirical lagged correlations
    against model transition structures.
    
    Parameters
    ----------
    max_lag : int, default=60
        Maximum time lag (in time points) to consider for temporal delays
    bin_lag : int, default=10
        Bin size for grouping lags in regression. If None or 0, uses max_lag
    uperm_method : {'uperms', 'full'}, default='uperm'
        Method for generating permutation matrices:
        - 'uperms': Unique permutations using random sampling (for large matrices)
        - 'full': All possible row and column permutations
    k : int, default=30
        Number of unique permutations to generate when using 'uperms' method
    model_tm : np.ndarray, optional
        Model transition matrix representing the hypothesized sequence structure
        
    Attributes
    ----------
    empirical_tm : np.ndarray or None
        Empirical transition matrix computed from data via lagged regression
    model_tm_perms : list of np.ndarray or None
        List of permuted transition matrices for null hypothesis testing
    """
    
    def __init__(
        self, 
        max_lag: int = 60, 
        bin_lag: int = 10, 
        uperm_method: Literal['uperms', 'full'] = 'uperms', 
        k: int = 30,
        model_tm: Optional[np.ndarray] = None
    ):
        self.max_lag = max_lag
        self.bin_lag = bin_lag
        self.uperm_method = uperm_method
        self.k = k
        
        self.model_tm = model_tm
        self.empirical_tm: Optional[np.ndarray] = None
        self.model_tm_perms: Optional[List[np.ndarray]] = None
        
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        **kwargs
    ) -> np.ndarray:
        """
        Fit the TDLM to data and compute sequenceness measures.
        
        This method performs lagged regression on the input data to compute an
        empirical transition matrix, then decomposes it into forward, backward,
        symmetric, and offset components relative to the model transition matrix.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_states, n_timepoints) containing
            probability or activation values for each state over time
        y : np.ndarray
            Label array indicating the number of states
        **kwargs
            model_tm : np.ndarray, optional
                Model transition matrix (overrides instance attribute)
            
        Returns
        -------
        np.ndarray
            Sequenceness measures of shape (4, max_lag) containing:
            - Row 0: Forward sequenceness
            - Row 1: Backward sequenceness  
            - Row 2: Symmetric component
            - Row 3: Offset/baseline (intercept)
        """
        model_tm = kwargs.get('model_tm', self.model_tm)
        
        if model_tm is not None:
            self.model_tm = model_tm
            
        # Compute empirical transition matrix via lagged regression
        self.empirical_tm = self._lagged_regression(X, y.size)
        
        # Create regression design matrix with forward, backward, identity, and constant
        regr_model = np.vstack([
            model_tm.flatten('F'),  # Forward transitions
            model_tm.T.flatten('F'),  # Backward transitions
            np.eye(y.size).flatten('F'),  # Identity (symmetric)
            np.ones(y.size**2)  # Offset/baseline intercept
        ])
        
        # Decompose empirical matrix into components via least squares
        sequenceness = np.linalg.pinv(regr_model).T @ self.empirical_tm.T
        
        return sequenceness

    def permutations(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        **kwargs
    ) -> np.ndarray:
        """
        Compute sequenceness for permuted transition matrices (null distribution).
        
        This generates a null distribution for statistical testing by computing
        sequenceness measures for randomly permuted versions of the model
        transition matrix.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix (same as in fit method)
        y : np.ndarray
            Label array indicating the number of states
        **kwargs
            model_tm_perms : list of np.ndarray, optional
                Pre-computed permutation matrices
            
        Returns
        -------
        np.ndarray
            Sequenceness for permutations, shape (n_perms, 4, max_lag)
        """
        model_tm_perms = kwargs.get('model_tm_perms', self.model_tm_perms)

        if np.any(model_tm_perms):
            self.model_tm_perms = model_tm_perms
        else:
            self.model_tm_perms = self._get_tm_permutations()
        
        if self.empirical_tm is not None: 
            sequenceness_perms = np.zeros((len(self.model_tm_perms), 4, self.max_lag))
            
            for itmp, tmp in enumerate(self.model_tm_perms):
                # Build regression model for this permutation
                regr_model = np.vstack([
                    tmp.flatten('F'), 
                    tmp.T.flatten('F'), 
                    np.eye(y.size).flatten('F'), 
                    np.ones(y.size**2)
                ])
                sequenceness_perms[itmp, :, :] = np.linalg.pinv(regr_model).T @ self.empirical_tm.T   
        
        return sequenceness_perms
    
    
    def fit_chunk(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        chunk_size: int, 
        overlap: int, 
        **kwargs
    ) -> np.ndarray:
        """
        Fit TDLM to data in overlapping chunks for time-resolved analysis.
        
        This method enables tracking of how sequenceness evolves over time by
        analyzing the data in sliding windows.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_states, n_timepoints)
        y : np.ndarray
            Label array indicating the number of states
        chunk_size : int
            Size of each time window in timepoints
        overlap : int
            Number of overlapping timepoints between consecutive windows
        **kwargs
            model_tm : np.ndarray, optional
                Model transition matrix
            
        Returns
        -------
        np.ndarray
            Time-resolved sequenceness of shape (n_chunks, 4, max_lag)
        """
        model_tm = kwargs.get('model_tm', self.model_tm)
        
        if model_tm is not None:
            self.model_tm = model_tm
        
        # Build regression model once (same for all chunks)
        regr_model = np.vstack([
            model_tm.flatten('F'), 
            model_tm.T.flatten('F'), 
            np.eye(y.size).flatten('F'), 
            np.ones(y.size**2)
        ])
        
        # Calculate step size and number of chunks
        step = chunk_size - overlap
        n_chunks = max(1, (X.shape[1] - overlap) // step)
        
        # Initialize output arrays
        self.empirical_tm = np.zeros((n_chunks, self.max_lag, y.size**2))
        sequenceness = np.zeros((n_chunks, 4, self.max_lag))
        
        # Process each chunk
        for chs in range(n_chunks):
            start_idx = chs * step
            end_idx = start_idx + chunk_size
            
            # Compute empirical transition matrix for this chunk
            self.empirical_tm[chs, :, :] = self._lagged_regression(
                X[:, start_idx:end_idx], 
                y.size
            )
            
            # Decompose into sequenceness components
            sequenceness[chs, :, :] = np.linalg.pinv(regr_model).T @ self.empirical_tm[chs, :, :].T
        
        return sequenceness

    def permutations_chunk(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        **kwargs
    ) -> np.ndarray:
        """
        Compute permutation null distribution for chunked analysis.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        y : np.ndarray
            Label array indicating the number of states
        **kwargs
            model_tm_perms : list of np.ndarray, optional
                Pre-computed permutation matrices
            
        Returns
        -------
        np.ndarray
            Permuted sequenceness, shape (n_chunks, n_perms, 4, max_lag)
        """
        model_tm_perms = kwargs.get('model_tm_perms', self.model_tm_perms)

        if np.any(model_tm_perms):
            self.model_tm_perms = model_tm_perms
        else:
            self.model_tm_perms = self._get_tm_permutations()
        
        if self.empirical_tm is not None: 
            sequenceness_perms = np.zeros((
                self.empirical_tm.shape[0], 
                len(self.model_tm_perms), 
                4, 
                self.max_lag
            ))
            
            # Iterate over chunks and permutations
            for iemptm, emptm in enumerate(self.empirical_tm):
                for itmp, tmp in enumerate(self.model_tm_perms):
                    regr_model = np.vstack([
                        tmp.flatten('F'), 
                        tmp.T.flatten('F'), 
                        np.eye(y.size).flatten('F'), 
                        np.ones(y.size**2)
                    ])
                    sequenceness_perms[iemptm, itmp, :, :] = np.linalg.pinv(regr_model).T @ emptm.T   
            
        return sequenceness_perms
    
    
    def replay_onsets(
        self, 
        X: np.ndarray, 
        lag: int, 
        **kwargs
    ) -> np.ndarray:
        """
        Detect onset times of replay events in the data.
        
        This method identifies timepoints where sequential reactivation patterns
        begin by finding peaks in the correlation between the data and its
        lagged, model-projected version.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_states, n_timepoints)
        lag : int
            Time lag to use for detecting replays
        **kwargs
            thresh_percentile : float, default=95
                Percentile threshold for detecting significant replay events
            minimum_offset : int, default=10
                Minimum time separation between detected replay onsets
            model_tm : np.ndarray, optional
                Model transition matrix
            
        Returns
        -------
        np.ndarray
            Array of timepoint indices where replay events begin
        """
        # Extract parameters with defaults
        thresh_percentile = kwargs.get('thresh_percentile', 95)
        minimum_offset = kwargs.get('minimum_offset', 10)
        model_tm = kwargs.get('model_tm', self.model_tm)
        
        if model_tm is not None:
            self.model_tm = model_tm
        
        P = model_tm
        
        # Create time-lagged version of data (shifted forward)
        Proj = np.zeros(X.shape)
        for tlu in range(X.shape[0]):
            Proj[tlu, :X.shape[-1] - tlu * lag] = X[tlu, tlu * lag:]
        
        # Project original data through model transition matrix
        Orig = np.matmul(P, X)
        
        # Compute element-wise correlation between original and projected
        R = np.sum(Orig * Proj, 0)
        
        # Threshold to find significant events
        R[R <= np.percentile(R, thresh_percentile)] = 0
        
        # Label connected components (continuous replay events)
        labeled_array, num_features = label(R)
        
        # Find peak within each connected component
        temp = []
        for f in range(1, num_features + 1):
            component_mask = labeled_array == f
            peak_idx = np.where(component_mask)[0][np.argmax(R[component_mask])]
            temp.append(peak_idx)
        temp = np.atleast_1d(temp)
        
        # Filter to ensure minimum separation between onsets
        replay_onsets = temp[np.hstack([False, np.diff(temp) > minimum_offset])]
        
        return replay_onsets
    
    def _lagged_regression(
        self, 
        X: np.ndarray, 
        label_size: int
    ) -> np.ndarray:
        """
        Perform lagged regression to compute empirical transition matrix.
        
        This internal method creates a design matrix with time-lagged predictors
        for each state and performs regression to estimate temporal dependencies.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_states, n_timepoints)
        label_size : int
            Number of distinct states
            
        Returns
        -------
        np.ndarray
            Empirical transition matrix of shape (max_lag, label_size^2)
        """
        # Determine effective bin lag
        if (self.bin_lag is None) or (self.bin_lag == 0):
            eff_bin_lag = self.max_lag
        else:
            eff_bin_lag = self.bin_lag
        
        # Create Toeplitz matrices for each state (lagged predictors)
        lagged_pred = np.zeros((X.shape[-1], self.max_lag * label_size))
        for s in range(label_size):
            lagged_pred[:, self.max_lag * s:self.max_lag * (s + 1)] = toeplitz(
                X[s, :].ravel(order='F'), 
                np.zeros((1, self.max_lag + 1)).ravel(order='F')
            )[:, 1:]
        
        # Perform regression for each lag bin
        betas = np.zeros((label_size * self.max_lag, label_size))
        intercept = np.ones((lagged_pred.shape[0], 1))
        
        for ilag in range(eff_bin_lag):
            # Select predictors at this lag across all states
            pred_idx = np.arange(
                ilag,
                label_size * self.max_lag,
                self.bin_lag
            )
            
            # Combine lagged predictors with intercept
            P = np.concatenate([lagged_pred[:, pred_idx], intercept], 1)
            
            # Solve least squares regression
            b = np.linalg.pinv(P) @ X.T
            
            # Store regression coefficients (exclude intercept)
            betas[pred_idx, :] = b[0:-1, :]
        
        # Reshape to (max_lag, label_size^2) format
        return np.reshape(betas, (self.max_lag, label_size**2), order='F')
        
    def _get_tm_permutations(self, **kwargs) -> List[np.ndarray]:
        """
        Generate permuted versions of the model transition matrix.
        
        Creates a set of transition matrices with permuted state orders,
        excluding the original forward matrix, backward matrix, and identity.
        Used for null hypothesis testing.
        
        Parameters
        ----------
        **kwargs
            model_tm : np.ndarray, optional
                Model transition matrix to permute
            uperm_method : str, optional
                Permutation method ('uperms' or 'full')
            k : int, optional
                Number of permutations for 'uperms' method
            
        Returns
        -------
        list of np.ndarray
            List of permuted transition matrices
        """
        model_tm = kwargs.get('model_tm', self.model_tm)
        uperm_method = kwargs.get('uperm_method', self.uperm_method)
        k = kwargs.get('k', self.k)
        
        if uperm_method == 'full':
            
            # Generate all possible matrices
            n_mat = model_tm.shape[0]
            n_ones = np.sum(model_tm.flatten())
            all_perms = []
            
            # Total positions in 4x4 matrix
            total_positions = n_mat**2
            
            # Choose 3 positions out of 16 for the ones
            for positions in combinations(range(total_positions), n_ones):
                # Create a zero matrix
                matrix = np.zeros((n_mat, n_mat), dtype=int)
                
                # Place ones at selected positions
                for pos in positions:
                    row = pos // n_mat
                    col = pos % n_mat
                    matrix[row, col] = 1
                
                all_perms.append(matrix)
            
        elif uperm_method == 'uperms': 
            # Use unique permutations with random sampling (efficient for large n)
            (nPerms, pInds, Perms) = self._uperms(range(model_tm.shape[0]), k=k+2)
            all_perms = [model_tm[i, :][:, i] for i in pInds]
            
        else:
            # Standard permutations (permute rows only)
            n = model_tm.shape[0]
            all_perms = []
            
            for row_perm in permutations(range(n)):
                for col_perm in permutations(range(n)):
                    # Apply permutation to both dimensions
                    permuted = model_tm[np.ix_(row_perm, col_perm)]
                    all_perms.append(permuted)
        
        
        # Filter out excluded matrices
        filtered_perms = []
        for matrix in all_perms:
            # Skip if any row has more than one 1
            if np.any(np.sum(matrix, axis=1) > 1):
                continue
            # Skip if it's the forward matrix (original)
            if np.array_equal(matrix, model_tm):
                continue
            # Skip if it's the backward matrix (transpose)
            if np.array_equal(matrix, model_tm.T):
                continue
            # Skip if all ones are on diagonal (identity-like)
            if np.trace(matrix) == matrix.sum():
                continue
            
            # Check for duplicates (including transposes)
            is_duplicate = False
            for existing in filtered_perms:
                if np.array_equal(matrix, existing): #or np.array_equal(matrix.T, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_perms.append(matrix)
        
        return filtered_perms
        
    
    def _uperms(self, X, k=None):
        """
        Created on Thu Mar 10 16:05:07 2022

        1:1 copied implementation from the MATLAB reference implementation found at
        https://github.com/YunzheLiu/TDLM

        results of both functions are bit wise equivalent
        @author: Simon Kern
        
        #uperms: unique permutations of an input vector or rows of an input matrix
        # Usage:  nPerms              = uperms(X)
        #        [nPerms pInds]       = uperms(X, k)
        #        [nPerms pInds Perms] = uperms(X, k)
        #
        # Determines number of unique permutations (nPerms) for vector or matrix X.
        # Optionally, all permutations' indices (pInds) are returned. If requested,
        # permutations of the original input (Perms) are also returned.
        #
        # If k < nPerms, a random (but still unique) subset of k of permutations is
        # returned. The original/identity permutation will be the first of these.
        #
        # Row or column vector X results in Perms being a [k length(X)] array,
        # consistent with MATLAB's built-in perms. pInds is also [k length(X)].
        #
        # Matrix X results in output Perms being a [size(X, 1) size(X, 2) k]
        # three-dimensional array (this is inconsistent with the vector case above,
        # but is more helpful if one wants to easily access the permuted matrices).
        # pInds is a [k size(X, 1)] array for matrix X.
        #
        # Note that permutations are not guaranteed in any particular order, even
        # if all nPerms of them are requested, though the identity is always first.
        #
        # Other functions can be much faster in the special cases where they apply,
        # as shown in the second block of examples below, which uses perms_m.
        #
        # Examples:
        #  uperms(1:7),       factorial(7)        # verify counts in simple cases,
        #  uperms('aaabbbb'), nchoosek(7, 3)      # or equivalently nchoosek(7, 4).
        #  [n pInds Perms] = uperms('aaabbbb', 5) # 5 of the 35 unique permutations
        #  [n pInds Perms] = uperms(eye(3))       # all 6 3x3 permutation matrices
        #
        #  # A comparison of timings in a special case (i.e. all elements unique)
        #  tic; [nPerms P1] = uperms(1:20, 5000); T1 = toc
        #  tic; N = factorial(20); S = sample_no_repl(N, 5000);
        #  P2 = zeros(5000, 20);
        #  for n = 1:5000, P2(n, :) = perms_m(20, S(n)); end
        #  T2 = toc # quicker (note P1 and P2 are not the same random subsets!)
        #  # For me, on one run, T1 was 7.8 seconds, T2 was 1.3 seconds.
        #
        #  # A more complicated example, related to statistical permutation testing
        #  X = kron(eye(3), ones(4, 1));  # or similar statistical design matrix
        #  [nPerms pInds Xs] = uperms(X, 5000); # unique random permutations of X
        #  # Verify correctness (in this case)
        #  G = nan(12,5000); for n = 1:5000; G(:, n) = Xs(:,:,n)*(1:3)'; end
        #  size(unique(G', 'rows'), 1)    # 5000 as requested.
        #
        # See also: randperm, perms, perms_m, signs_m, nchoosek_m, sample_no_repl
        # and http://www.fmrib.ox.ac.uk/fsl/randomise/index.html#theory

        # Copyright 2010 Ged Ridgway
        # http://www.mathworks.com/matlabcentral/fileexchange/authors/27434
        """
        # Count number of repetitions of each unique row, and get representative x
        X = np.array(X).squeeze()
        assert len(X) > 1

        if X.ndim == 1:
            uniques, uind, c = np.unique(X, return_index=True, return_counts=True)
        else:
            # [u uind x] = unique(X, 'rows'); % x codes unique rows with integers
            uniques, uind, c = np.unique(X, axis=0, return_index=True, return_counts=True)

        uniques = uniques.tolist()
        x = np.array([uniques.index(i) for i in X.tolist()])

        c = sorted(c)
        nPerms = np.prod(np.arange(c[-1] + 1, np.sum(c) + 1)) / np.prod(
            [math.factorial(x) for x in c[:-1]]
        )
        nPerms = int(nPerms)
        #% computation of permutation
        # Basics
        n = len(X)
        if k is None or k > nPerms:
            k = nPerms
            # default to computing all unique permutations

        #% Identity permutation always included first:
        pInds = np.zeros([int(k), n]).astype(np.uint32)
        Perms = pInds.copy()
        pInds[0, :] = np.arange(0, n)
        Perms[0, :] = x

        # Add permutations that are unique
        u = 0
        # to start with
        while u < k - 1:
            pInd = np.random.permutation(int(n))
            pInd = np.array(pInd).astype(
                int
            )  # just in case MATLAB permutation was monkey patched
            if x[pInd].tolist() not in Perms.tolist():
                u += 1
                pInds[u, :] = pInd
                Perms[u, :] = x[pInd]
        #%
        # Construct permutations of input
        if X.ndim == 1:
            Perms = np.repeat(np.atleast_2d(X), k, 0)
            for n in np.arange(1, k):
                Perms[n, :] = X[pInds[n, :]]
        else:
            Perms = np.repeat(np.atleast_3d(X), k, axis=2)
            for n in np.arange(1, k):
                Perms[:, :, n] = X[pInds[n, :], :]
        return (nPerms, pInds, Perms)
    
    
    def plot_empirical_matrices(
        self, 
        **kwargs
    ) -> Optional[Axes]:
        """
        Visualize the empirical transition matrices.
        
        Creates a grid of subplots showing all empirical matrices with their
        values displayed in each cell.
        
        Parameters
        ----------
        **kwargs
            time_array : np.ndarray of lag times, default=None
                Array that links the last dimension to the time domain 
                (e.g. [10,20,30, ..., max_lag]). Needs to be of size max_lag.
            lags : np.ndarray of lags, default=None
                Allows for a subset of lags to display (e.g. [20, 50, 100, ...]). 
                If none (default), all lags are plotted. If used, time_array 
                needs to exist. (e.g. [20, 50, 100, ...])
            axes : np.ndarray of Axes, optional
                Pre-existing axes to plot on
            return_handles : bool, default=False
                If True, return figure and axes handles
            
        Returns
        -------
        Axes or tuple of (Figure, Axes), optional
            Axes object if axes were provided and return_handles is False,
            or (fig, axes) tuple if return_handles is True
        """
        time_array = kwargs.get('time_array', None)
        lags = kwargs.get('lags', None)
        axes = kwargs.get('axes', None)
        return_handles = kwargs.get('return_handles', False)
        
        emp_tm = np.reshape(
            self.empirical_tm, 
            (self.max_lag, self.model_tm.shape[0], self.model_tm.shape[0]), 
            order='F'
        )
        
        if time_array is None:
            time_array = np.arange(1, emp_tm.shape[0]+1)

        if lags is None:
            lags = time_array
            
        # Create figure if axes not provided
        if axes is None:
            fig, axes = plt.subplots(
                nrows=int(np.sqrt(lags.shape[0])), 
                ncols=int(np.round(np.sqrt(lags.shape[0]) + 1)), 
                figsize=(9, 8)
            )

        for i,ax in enumerate(axes.flat):
            
            if emp_tm.shape[0] > i:
                idx = (time_array == lags[i])
                ax.imshow(
                    emp_tm[idx,:,:].squeeze(),
                    vmin=np.percentile(emp_tm.flatten(), 2.5),
                    vmax=np.percentile(emp_tm.flatten(), 97.5),
                    cmap='Blues', 
                    aspect='equal'
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Lag: {lags[i]}', {'fontsize': 10})
            else:
                # Hide unused subplots
                ax.set_axis_off()
        
        
        if return_handles:
            return fig, axes
        if axes is not None:
            return ax
    
    
    def plot_permutation_matrices(
        self, 
        **kwargs
    ) -> Optional[Axes]:
        """
        Visualize the permuted transition matrices.
        
        Creates a grid of subplots showing all permutation matrices with their
        values displayed in each cell.
        
        Parameters
        ----------
        **kwargs
            axes : np.ndarray of Axes, optional
                Pre-existing axes to plot on
            return_handles : bool, default=False
                If True, return figure and axes handles
            
        Returns
        -------
        Axes or tuple of (Figure, Axes), optional
            Axes object if axes were provided and return_handles is False,
            or (fig, axes) tuple if return_handles is True
        """
        axes = kwargs.get('axes', None)
        return_handles = kwargs.get('return_handles', False)
        
        # Create figure if axes not provided
        if axes is None:
            fig, axes = plt.subplots(
                nrows=int(np.sqrt(len(self.model_tm_perms))), 
                ncols=int(np.round(np.sqrt(len(self.model_tm_perms)) + 1)), 
                figsize=(7, 6)
            )
        
        # Plot each permutation matrix
        for i, ax in enumerate(axes.flat):
            if len(self.model_tm_perms) > i:
                mat = self.model_tm_perms[i]
                
                # Display matrix as image
                ax.imshow(mat, cmap='gray_r', vmin=0, vmax=1, aspect='equal')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add cell values as text
                for row in range(mat.shape[0]):
                    for col in range(mat.shape[1]):
                        val = mat[row, col]
                        ax.text(
                            col, row, f'{val:.0f}', 
                            ha='center', va='center', fontsize=14,
                            color='white' if abs(val) > 0.5 else 'black'
                        )
            else:
                # Hide unused subplots
                ax.set_axis_off()
        
        if return_handles:
            return fig, axes
        if axes is not None:
            return ax