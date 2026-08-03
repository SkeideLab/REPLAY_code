# -*- coding: utf-8 -*-
"""
Cross-Correlation based Sequenceness Analysis

Estimates replay sequenceness by computing pairwise cross-correlations between
stimulus probability time series at varying lags, then averaging over
forward and backward transition pairs defined by a transition matrix.

The forward/backward pair definitions follow the same transition-matrix
convention used in TDLM.py, making the two methods directly comparable.

Created on 2026-05-11
@author: Christopher Postzich
"""

from itertools import permutations as _permutations
from typing import List, Optional, Tuple

import numpy as np


def create_transition_matrix(
    label: np.ndarray, transitions: np.ndarray
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


class CrossCorrelationSequenceness:
    """
    Cross-correlation based sequenceness analysis.

    For each lag τ, computes the Pearson cross-correlation matrix
    C[i, j, τ] = corr(X[i, t], X[j, t+τ]) and then averages over
    the transition pairs defined by a forward transition matrix.

    Parameters
    ----------
    max_lag : int, default=50
        Maximum lag (in time points) to compute correlations for.

    Attributes
    ----------
    xcorr_ : np.ndarray or None
        Cached cross-correlation tensor (n_states, n_states, max_lag)
        from the most recent call to fit(). Reused by permutations().
    model_tm : np.ndarray or None
        Forward transition matrix currently in use.
    model_tm_perms : list of np.ndarray or None
        Permuted transition matrices for null-hypothesis testing.
    """

    def __init__(self, max_lag: int = 50):
        self.max_lag = max_lag
        self.model_tm: Optional[np.ndarray] = None
        self.xcorr_: Optional[np.ndarray] = None
        self.model_tm_perms: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Public API (mirrors TDLM interface)
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        classes: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute cross-correlation sequenceness for one participant.

        Parameters
        ----------
        X : np.ndarray, shape (n_states, n_timepoints)
            Stimulus probability time series, one row per state.
        classes : np.ndarray
            Array of state labels (used only to determine n_states,
            consistent with the TDLM interface).
        **kwargs
            model_tm : np.ndarray, optional
                Forward transition matrix. Overrides the instance attribute.

        Returns
        -------
        np.ndarray, shape (3, max_lag)
            Row 0: forward sequenceness (mean corr over forward pairs)
            Row 1: backward sequenceness (mean corr over backward pairs)
            Row 2: net sequenceness (forward − backward)
        """
        model_tm = kwargs.get("model_tm", self.model_tm)
        if model_tm is not None:
            self.model_tm = model_tm

        self.xcorr_ = self._compute_xcorr(X)
        return self._sequenceness_from_xcorr(self.xcorr_, self.model_tm)

    def permutations(
        self,
        X: np.ndarray,
        classes: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute sequenceness for permuted transition matrices (null distribution).

        If fit() was called previously on the same X, the cached cross-
        correlation tensor is reused so the correlations are only computed once.

        Parameters
        ----------
        X : np.ndarray, shape (n_states, n_timepoints)
        classes : np.ndarray
        **kwargs
            model_tm_perms : list of np.ndarray, optional
                Pre-computed permutation matrices. If not supplied and not
                cached, they are generated automatically.

        Returns
        -------
        np.ndarray, shape (n_perms, 3, max_lag)
        """
        model_tm_perms = kwargs.get("model_tm_perms", self.model_tm_perms)
        if model_tm_perms is not None:
            self.model_tm_perms = model_tm_perms
        else:
            self.model_tm_perms = self._get_tm_permutations(self.model_tm)

        if self.xcorr_ is None:
            self.xcorr_ = self._compute_xcorr(X)

        n_perms = len(self.model_tm_perms)
        sequenceness_perms = np.zeros((n_perms, 3, self.max_lag))
        for ip, tm_perm in enumerate(self.model_tm_perms):
            sequenceness_perms[ip] = self._sequenceness_from_xcorr(self.xcorr_, tm_perm)

        return sequenceness_perms

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_xcorr(self, X: np.ndarray) -> np.ndarray:
        """
        Build the full cross-correlation tensor at all lags.

        Parameters
        ----------
        X : np.ndarray, shape (n_states, n_timepoints)

        Returns
        -------
        np.ndarray, shape (n_states, n_states, max_lag)
            xcorr[i, j, τ] = Pearson r between X[i, t] and X[j, t+τ+1].
        """
        n_states, T = X.shape
        xcorr = np.zeros((n_states, n_states, self.max_lag))

        for lag in range(1, self.max_lag + 1):
            Xn = X[:, : T - lag]  # present:  (n_states, T-lag)
            Xl = X[:, lag:]  # future:   (n_states, T-lag)
            n_t = T - lag

            std_n = Xn.std(1, keepdims=True)
            std_l = Xl.std(1, keepdims=True)

            # Safe normalisation — flat signals give zero cross-correlation
            Zn = np.where(std_n > 0, (Xn - Xn.mean(1, keepdims=True)) / std_n, 0.0)
            Zl = np.where(std_l > 0, (Xl - Xl.mean(1, keepdims=True)) / std_l, 0.0)

            xcorr[:, :, lag - 1] = (Zn @ Zl.T) / n_t

        return xcorr

    def _sequenceness_from_xcorr(
        self,
        xcorr: np.ndarray,
        tm: np.ndarray,
    ) -> np.ndarray:
        """
        Project a cross-correlation tensor onto a transition matrix.

        Parameters
        ----------
        xcorr : np.ndarray, shape (n_states, n_states, max_lag)
        tm : np.ndarray, shape (n_states, n_states)
            Forward transition matrix (non-zero entries define forward pairs).

        Returns
        -------
        np.ndarray, shape (3, max_lag)
            [forward, backward, net]
        """
        fwd_mask = tm > 0  # (n_states, n_states)
        bwd_mask = tm.T > 0

        # Average correlation over all forward / backward state pairs
        forward = xcorr[fwd_mask].mean(0)  # (max_lag,)
        backward = xcorr[bwd_mask].mean(0)  # (max_lag,)

        return np.stack([forward, backward, forward - backward], axis=0)

    def _get_tm_permutations(
        self,
        model_tm: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Generate permuted transition matrices for null-hypothesis testing.

        Applies all row-and-column permutations of the labels and filters out:
        the original forward matrix, its transpose (backward), identity-like
        matrices, and duplicates.

        Parameters
        ----------
        model_tm : np.ndarray, shape (n_states, n_states)

        Returns
        -------
        list of np.ndarray
        """
        n = model_tm.shape[0]
        filtered: List[np.ndarray] = []

        for perm in _permutations(range(n)):
            perm = list(perm)
            mat = model_tm[np.ix_(perm, perm)]

            if np.any(np.sum(mat, axis=1) > 1):
                continue
            if np.array_equal(mat, model_tm):
                continue
            if np.array_equal(mat, model_tm.T):
                continue
            if mat.sum() > 0 and np.trace(mat) == mat.sum():
                continue
            if any(np.array_equal(mat, e) for e in filtered):
                continue

            filtered.append(mat)

        return filtered
