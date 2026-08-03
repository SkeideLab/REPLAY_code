# -*- coding: utf-8 -*-
"""
Plotting Functions

Created on Wed Jul 24 20:00:35 2024
@author: Christopher Postzich
@github: mcpost
"""

import mne
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from utils.utils import std_error, chan_grid as _CHAN_GRID

### Helper Functions


def clear_axis(ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels("")
    ax.set_yticklabels("")


### Preprocessing Plot Functions


def plot_data_quant(trialnum_loc, rest_dur, labels, ytick_names, **kwargs):

    # Extracting optional parameters with defaults
    width = kwargs.get("width", 0.2)
    colors = kwargs.get("colors", ("#1f77b4ff", "#ff7f0eff", "#2ca02cff"))
    hspace = kwargs.get("hspace", 0.05)
    wspace = kwargs.get("wspace", 0.15)
    save = kwargs.get("save", None)

    y = np.arange(trialnum_loc.shape[0])

    fig, ax = plt.subplots(
        ncols=2,
        nrows=1,
        sharey=True,
        gridspec_kw={
            "height_ratios": [1],
            "width_ratios": [1, 1],
            "hspace": hspace,
            "wspace": wspace,
        },
        figsize=(11, 5),
    )
    for i, (tn, c) in enumerate(zip(trialnum_loc.T, colors)):
        y_cur = y + width * i - width * (trialnum_loc.shape[1] // 2)
        ax[0].barh(y_cur, tn, width, color=c)
    ax[0].set_yticks(y, ytick_names)
    ax[0].set_xlabel("Trial Number")
    ax[0].legend(labels)
    ax[1].barh(y, rest_dur.squeeze(), width + 0.4, color="blue")
    ax[1].set_xlabel("Resting Signal (s)")
    if save:
        plt.savefig(save, bbox_inches="tight")


def plot_cohens_kappa(
    kappa_values,
    scale="landis",
    ax=None,
    title=None,
    custom_scale=None,
    show_boxplot=True,
    show_points=True,
    jitter=0.05,
    boxplot_kwargs=None,
    scatter_kwargs=None,
    text_kwargs=None,
):
    """
    Plot Cohen's kappa values horizontally with optional boxplot and individual points.
    Background is colored according to interpretation scales for Cohen's kappa.

    Parameters:
    -----------
    kappa_values : array-like
        Array of Cohen's kappa values to plot
    scale : str, optional
        Which interpretation scale to use: 'cicchetti', 'fleiss', 'landis', 'regier'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.
    title : str, optional
        Plot title
    custom_scale : dict, optional
        Custom scale dictionary with required keys 'borders' and 'labels' and optional key 'colors'
        Example: {'borders': [0.2, 0.4, 0.6, 1.0],
                 'labels': ['Poor', 'Fair', 'Good', 'Excellent'],
                 'colors': ['#FF9999', '#FFCC99', '#CCFF99', '#99FF99']}
    show_boxplot : bool, optional
        Whether to show boxplot
    show_points : bool, optional
        Whether to show individual points
    jitter : float, optional
        Amount of vertical jitter for points
    boxplot_kwargs : dict, optional
        Additional keyword arguments to pass to boxplot (defaults to upper half placement)
    scatter_kwargs : dict, optional
        Additional keyword arguments to pass to scatter plot (defaults to lower half placement)
    text_kwargs : dict, optional
        Additional keyword arguments to pass to text labels

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    # Define the scales
    scales = {
        "cicchetti": {
            "borders": [0.4, 0.6, 0.75, 1.0],
            "labels": ["Poor", "Fair", "Good", "Excellent"],
            "colors": ["#FF9999", "#FFCC99", "#CCFF99", "#99FF99"],
        },
        "fleiss": {
            "borders": [0.4, 0.75, 1.0],
            "labels": ["Poor", "Fair to Good", "Excellent"],
            "colors": ["#FF9999", "#FFCC99", "#99FF99"],
        },
        "landis": {
            "borders": [0.2, 0.4, 0.6, 0.8, 1.0],
            "labels": ["Slight", "Fair", "Moderate", "Substantial", "Almost Perfect"],
            "colors": ["#FF9999", "#FFCC99", "#FFFFCC", "#CCFF99", "#99FF99"],
        },
        "regier": {
            "borders": [0.2, 0.4, 0.6, 0.8, 1.0],
            "labels": [
                "Unacceptable",
                "Questionable",
                "Good",
                "Very Good",
                "Excellent",
            ],
            "colors": ["#FF9999", "#FFCC99", "#FFFFCC", "#CCFF99", "#99FF99"],
        },
    }

    # Use custom scale if provided
    if custom_scale is not None:
        if "borders" not in custom_scale or "labels" not in custom_scale:
            raise ValueError("Custom scale must contain 'borders' and 'labels' keys")

        # Create a copy to avoid modifying the original
        selected_scale = custom_scale.copy()

        # Generate colors if not provided
        if "colors" not in selected_scale:
            n_colors = len(selected_scale["borders"])
            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", ["#FF9999", "#99FF99"], N=n_colors
            )
            selected_scale["colors"] = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    else:
        if scale not in scales:
            raise ValueError(f"Scale must be one of {list(scales.keys())} or a custom dict")
        selected_scale = scales[scale]

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Add background patches
    prev_border = 0
    for i, border in enumerate(selected_scale["borders"]):
        rect = Rectangle(
            (prev_border, 0),
            border - prev_border,
            1,
            linewidth=1,
            edgecolor="gray",
            facecolor=selected_scale["colors"][i],
            alpha=0.3,
        )
        ax.add_patch(rect)

        # Add text label in the middle of each patch
        midpoint = (prev_border + border) / 2

        # Default text parameters
        default_text_kwargs = {
            "horizontalalignment": "center",
            "verticalalignment": "bottom",
            "fontsize": 10,
            "alpha": 0.7,
            "y": 0.05,  # Position text at the bottom
        }

        # Update with user provided text kwargs if any
        if text_kwargs is not None:
            default_text_kwargs.update(text_kwargs)

        ax.text(
            midpoint,
            default_text_kwargs.pop("y"),
            selected_scale["labels"][i],
            **default_text_kwargs,
        )

        prev_border = border

    # Plot boxplot if requested
    if show_boxplot:
        # Default boxplot parameters (upper half)
        default_boxplot_kwargs = {
            "vert": False,
            "patch_artist": True,
            "widths": 0.3,
            "positions": [0.75],  # Position in the upper half
            "boxprops": {"facecolor": "white", "alpha": 0.7},
            "medianprops": {"color": "black"},
            "whiskerprops": {"color": "black"},
            "capprops": {"color": "black"},
            "flierprops": {"markeredgecolor": "gray"},
        }

        # Update with user provided boxplot kwargs if any
        if boxplot_kwargs is not None:
            default_boxplot_kwargs.update(boxplot_kwargs)

        # Extract and remove box style properties to handle separately
        box_props = default_boxplot_kwargs.pop("boxprops", {"facecolor": "white", "alpha": 0.7})
        median_props = default_boxplot_kwargs.pop("medianprops", {"color": "black"})
        whisker_props = default_boxplot_kwargs.pop("whiskerprops", {"color": "black"})
        cap_props = default_boxplot_kwargs.pop("capprops", {"color": "black"})
        flier_props = default_boxplot_kwargs.pop("flierprops", {"markeredgecolor": "gray"})

        bp = ax.boxplot(kappa_values, **default_boxplot_kwargs)

        # Apply style properties
        plt.setp(bp["boxes"], **box_props)
        plt.setp(bp["medians"], **median_props)
        plt.setp(bp["whiskers"], **whisker_props)
        plt.setp(bp["caps"], **cap_props)
        plt.setp(bp["fliers"], **flier_props)

    # Plot individual points if requested
    if show_points:
        # Default scatter parameters (lower half)
        default_scatter_kwargs = {"color": "black", "s": 8, "alpha": 0.7, "zorder": 3}

        # Update with user provided scatter kwargs if any
        if scatter_kwargs is not None:
            default_scatter_kwargs.update(scatter_kwargs)

        # Add jitter to y position (default to lower half)
        y_pos = np.random.normal(0.25, jitter, size=len(kappa_values))
        ax.scatter(kappa_values, y_pos, **default_scatter_kwargs)

    # Set axis limits and remove y-ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    # Add a title if provided
    if title:
        ax.set_title(title)
    else:
        scale_name = scale.capitalize() if scale in scales else "Custom"
        ax.set_title(f"Cohen's Kappa Values ({scale_name} Scale)")

    # Add x-axis label
    ax.set_xlabel("Cohen's Kappa")

    # Add gridlines
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Tight layout
    plt.tight_layout()

    return fig, ax


### Analysis Plot Functions


def plot_correlation(x, y, **kwargs):
    """
    Plot a nice correlation plot between x and y.

    Parameters:
    -----------
    x : float or array-like, shape (n, )
        X-axis values.
    y : float or array-like, shape (n, )
        Y-axis values.

    Keyword Arguments:
    -----------------
    scatter_kwargs : dict, optional
        Additional keyword arguments for method plt.scatter()
    polyfit : bool, default=False
        Whether to plot a polynomial plot over the scatter plot data
    polyfit_kwargs : dict, optional
        Additional keyword arguments for method np.polyfit()
    line_kwargs : dict, optional
        Additional keyword arguments for method plt.plot() for the trendline
    save : str, default=None
        File path to save the figure.
    return_handles : bool, default=False
        If True, returns figure and axis handles instead of showing the plot.
    ax : axes object, default=None
        A matplotlib axes object to plot.

    Plotting Customization:
    ----------------------
    line_kwargs : dict, optional
        Additional kwargs for plt.plot() (e.g., linestyle, marker, color)

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects if return_handles=True
        Otherwise displays the plot
    """
    # Extracting optional parameters with defaults
    scatter_kwargs = kwargs.get("scatter_kwargs", dict())
    polyfit = kwargs.get("polyfit", False)
    polyfit_kwargs = kwargs.get("polyfit_kwargs", dict(deg=1))
    line_kwargs = kwargs.get("line_kwargs", dict())
    save = kwargs.get("save", None)
    return_handles = kwargs.get("return_handles", False)
    ax = kwargs.get("ax", None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(x, y, **scatter_kwargs)

    if polyfit:
        b, a = np.polyfit(x, y, **polyfit_kwargs)
        # Create x sequence
        xseq = np.linspace(np.min(x), np.max(x), num=100)
        # Plot regression line
        ax.plot(xseq, a + b * xseq, **line_kwargs)

    # Save or return
    if save:
        plt.savefig(save, bbox_inches="tight")
    if return_handles:
        return fig, ax
    if ax is not None:
        return ax


def plot_cond(data, x, labels, **kwargs):
    """
    Plot condition data with optional customization.

    Parameters
    ----------
    data : list of numpy.ndarray
        Input data. Each array should have shape (trials, time/freq points).
    x : array-like
        X-axis values (time or frequency points).
    labels : list of str
        Labels for each condition/dataset.

    Keyword Arguments
    -----------------
    data_format : {'erp', 'freq'}, optional
        Type of data for appropriate axis labeling.
    se_alpha : float, default=0.2
        Transparency of standard error fill.
    save : str, optional
        File path to save the figure.
    return_handles : bool, default=False
        If True, returns figure and axis handles instead of showing the plot.
    axes : matplotlib axes, default=None
        Axes object to pass as input.

    Plotting Customization
    ----------------------
    line_kwargs : dict, optional
        Additional kwargs for plt.plot() (e.g., linestyle, marker, color)
    fill_kwargs : dict, optional
        Additional kwargs for plt.fill_between()
    axhline_kwargs : dict, optional
        Additional kwargs for horizontal zero line
    axvline_kwargs : dict, optional
        Additional kwargs for vertical zero line
    title : str, optional
        Title for the plot
    xlabel : str, optional
        Custom x-axis label (overrides data_format)
    ylabel : str, optional
        Custom y-axis label (overrides data_format)

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects if return_handles=True
        Otherwise displays the plot or returns ax if ax was provided
    """
    # Extracting optional parameters with defaults
    data_format = kwargs.get("data_format", None)
    se_alpha = kwargs.get("se_alpha", 0.2)
    save = kwargs.get("save", None)
    return_handles = kwargs.get("return_handles", False)
    ax = kwargs.get("axes", None)

    # Plotting kwargs
    line_kwargs = kwargs.get("line_kwargs", {})
    fill_kwargs = kwargs.get("fill_kwargs", {})
    axhline_kwargs = kwargs.get("axhline_kwargs", {"y": 0.0, "color": "k", "linestyle": "-"})
    axvline_kwargs = kwargs.get("axvline_kwargs", {"x": 0.0, "color": "k", "linestyle": "-"})

    # Create or use existing axes
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        created_fig = False

    # Plot zero lines
    ax.axhline(**axhline_kwargs)
    ax.axvline(**axvline_kwargs)

    # Plot data
    for dat, lab in zip(data, labels):
        # Plot mean with optional customization
        ax.plot(x, dat.mean(0), label=lab, **line_kwargs)

        # Fill between with standard error
        ax.fill_between(
            x,
            np.mean(dat, 0) - std_error(dat, 0),
            np.mean(dat, 0) + std_error(dat, 0),
            alpha=se_alpha,
            **fill_kwargs,
        )

    # Add legend
    ax.legend()

    # Set labels based on data format or custom inputs
    if data_format == "erp":
        xlabel = kwargs.get("xlabel", "times")
        ylabel = kwargs.get("ylabel", "$\mu$V")
    elif data_format == "freq":
        xlabel = kwargs.get("xlabel", "frequencies")
        ylabel = kwargs.get("ylabel", "power")
    else:
        xlabel = kwargs.get("xlabel", "x")
        ylabel = kwargs.get("ylabel", "y")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set title if provided
    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    # Save if requested
    if save and created_fig:
        plt.savefig(save, bbox_inches="tight")

    # Return based on context
    if return_handles:
        return fig, ax
    elif ax is not None:
        return ax
    else:
        plt.show()


def plot_chan_cond(data, x, channels, labels, **kwargs):
    """
    Plot channel condition data with optional customization. Calls plot_cond internally after averaging over selected channels.

    Parameters
    ----------
    data : list of numpy.ndarray
        Input data. Each array should have shape (trials, channels, time/freq points).
    x : array-like
        X-axis values (time or frequency points).
    channels : int or array-like of int
        Channel indices to plot. Can be a single integer or list/array of integers.
    labels : list of str
        Labels for each condition/dataset.

    Keyword Arguments
    -----------------
    All kwargs from plot_cond are supported, plus:

    save : str, optional
        File path to save the figure.
    return_handles : bool, default=False
        If True, returns figure and axis handles instead of showing the plot.
    ax : matplotlib axes, default=None
        Axes object to pass to plot_cond.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects if return_handles=True
        Otherwise displays the plot or returns ax if ax was provided
    """
    # Ensure channels is a numpy array of integers
    if isinstance(channels, (int, np.integer)):
        channels = [channels]
    channels = np.atleast_1d(channels)

    # Extract parameters that are specific to this level
    save = kwargs.get("save", None)
    return_handles = kwargs.get("return_handles", False)
    ax_input = kwargs.get("ax", None)

    # Process data: select channels and average over them
    processed_data = []
    for dat in data:
        # Select specified channels, compute mean across those channels
        channel_data = dat[:, channels, :].mean(1)
        processed_data.append(channel_data)

    # Call plot_cond with processed data and all kwargs
    result = plot_cond(processed_data, x, labels, **kwargs)

    # Handle saving and returning
    if save and ax_input is None and not return_handles:
        plt.savefig(save, bbox_inches="tight")

    return result


def plot_topo_cond(data, x, channels, labels, **kwargs):
    """
    Create a topographic plot of channel conditions with interactive features. Calls plot_chan_cond internally for individual channel plots.

    Parameters
    ----------
    data : list of numpy.ndarray
        Input data. Each array should have shape (trials, channels, time/freq points).
    x : array-like
        X-axis values (time or frequency points).
    channels : list of str
        List of channel names corresponding to the data.
    labels : list of str
        Labels for each condition/dataset.

    Keyword Arguments
    -----------------
    data_format : {'erp', 'freq'}, optional
        Type of data for appropriate axis labeling.
    figsize : tuple, optional
        Figure size (default: (15, 10))
    se_alpha : float, default=0.1
        Transparency of standard error fill.
    line_kwargs : dict, optional
        Additional kwargs for plt.plot()
    fill_kwargs : dict, optional
        Additional kwargs for plt.fill_between()
    axhline_kwargs : dict, optional
        Additional kwargs for horizontal zero line
    axvline_kwargs : dict, optional
        Additional kwargs for vertical zero line
    title : str, optional
        Overall figure title
    save : str, optional
        File path to save the figure.
    return_handles : bool, default=False
        If True, returns figure and axis handles instead of showing the plot.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes array if return_handles=True
        Otherwise displays the plot
    """
    # Predefined channel grid
    chan_grid = {
        "F9": 0,
        "F7": 1,
        "F3": 2,
        "Fz": 3,
        "F4": 4,
        "F8": 5,
        "F10": 6,
        "FC5": 8,
        "FC3": 9,
        "FCz": 10,
        "FC4": 11,
        "FC6": 12,
        "T7": 14,
        "C3": 16,
        "C4": 18,
        "T8": 20,
        "CP5": 22,
        "CP3": 23,
        "CP4": 25,
        "CP6": 26,
        "TP9": 28,
        "P7": 29,
        "P3": 30,
        "Pz": 31,
        "P4": 32,
        "P8": 33,
        "TP10": 34,
        "POz": 38,
        "O1": 44,
        "Oz": 45,
        "O2": 46,
    }
    rev_chan_grid = {chan_grid[k]: k for k in chan_grid.keys()}

    # Extract topo-specific parameters
    figsize = kwargs.get("figsize", (15, 10))
    save = kwargs.get("save", None)
    return_handles = kwargs.get("return_handles", False)
    se_alpha = kwargs.get("se_alpha", 0.1)  # Default for topo is lower

    # Prepare kwargs to pass down to plot_cond (remove topo-specific ones)
    plot_kwargs = kwargs.copy()
    plot_kwargs.pop("figsize", None)
    plot_kwargs.pop("save", None)
    plot_kwargs.pop("return_handles", None)
    plot_kwargs["se_alpha"] = se_alpha  # Use topo default

    # Create figure
    fig, axes = plt.subplots(
        7,
        7,
        subplot_kw={},
        gridspec_kw={
            "height_ratios": [1] * 7,
            "width_ratios": [1] * 7,
            "hspace": 0.18,
            "wspace": 0.1,
        },
        figsize=figsize,
    )

    # Set overall title if provided
    if "title" in kwargs:
        fig.suptitle(kwargs["title"])
        # Remove title from kwargs passed to subplots
        plot_kwargs_subplot = plot_kwargs.copy()
        plot_kwargs_subplot.pop("title", None)
    else:
        plot_kwargs_subplot = plot_kwargs

    # Plot data for each channel
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i in rev_chan_grid.keys():
            # Get channel index
            chan_idx = channels.index(rev_chan_grid[i])

            # Use plot_cond for this channel by passing the axis
            plot_kwargs_subplot["axes"] = ax
            plot_kwargs_subplot["title"] = rev_chan_grid[i]

            # Extract data for this single channel and call plot_cond
            channel_data = [d[:, chan_idx, :] for d in data]
            plot_cond(channel_data, x, labels, **plot_kwargs_subplot)

            # Update title font size
            ax.set_title(rev_chan_grid[i], fontsize=10)
            ax.set_xlim(x[0], x[-1])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels("")
            ax.set_yticklabels("")
            ax.legend().remove()
            ax.axis("on")

        # Add legend to a specific subplot
        if i == 42:
            ax.axis("on")
            ax.plot(np.array([0]), np.array([[0]] * len(labels)).T, label=labels)
            ax.legend()
            ax.set_xlim(x[0], x[-1])

    # Interactive click event to plot individual channel
    def on_click(event):
        if event.inaxes is not None:
            channel_index = [i for i, ax in enumerate(axes.flat) if ax == event.inaxes][0]
            if channel_index in rev_chan_grid.keys():
                # Plot individual channel when clicked using plot_chan_cond
                chan_idx = channels.index(rev_chan_grid[channel_index])
                click_kwargs = plot_kwargs.copy()
                click_kwargs["title"] = f"{rev_chan_grid[channel_index]}"
                plot_chan_cond(data, x, chan_idx, labels, **click_kwargs)

    # Connect click event to the main figure
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Save or return
    if save:
        plt.savefig(save, bbox_inches="tight")

    if return_handles:
        return fig, ax
    else:
        plt.show()


# Plot Classifier


def plot_sliding_classifier(data, time, **kwargs):
    """
    Plot time-resolved classifier performance with optional significance markers.

    Creates a publication-ready plot of sliding window classifier results,
    showing mean performance over time with standard error shading and
    markers for statistically significant time clusters.

    Parameters
    ----------
    data : numpy.ndarray
        Classification performance data with shape:
        - 1D: (n_timepoints,) for single subject/pre-averaged data
        - 2D: (n_subjects/folds, n_timepoints) for group-level analysis
    time : numpy.ndarray
        Time vector in seconds corresponding to each timepoint.

    Keyword Arguments
    -----------------
    average : bool, default=True
        If True, compute and display mean +/- standard error across first axis.
        If False, plot all traces without averaging.
    chance : float, default=0.5
        Chance level for classifier (displayed as horizontal dashed line).
    se_alpha : float, default=0.2
        Alpha transparency for standard error shading.
    perm_info : dict, optional
        Permutation test results for significance marking. Expected keys:
        - 'cluster_dict': list of cluster dictionaries from permutation test
        - 'statistic': str, name of statistic ('cluster_area' or 'cluster_tsum')
        - 'cluster_max_pval': float, threshold for significance
    line_kwargs : dict, default=dict(linestyle='-', color='tab:blue')
        Keyword arguments passed to ax.plot() for the main line.
    fill_kwargs : dict, default=dict(color='tab:blue')
        Keyword arguments passed to ax.fill_between() for error shading.
    sct_kwargs : dict, default=dict(s=15, marker='o', color='tab:blue')
        Keyword arguments passed to ax.scatter() for significance markers.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
    save : str, optional
        File path to save the figure. If None, figure is not saved.
    return_handles : bool, default=False
        If True, returns (fig, ax) tuple.

    Returns
    -------
    matplotlib.axes.Axes or tuple
        Returns the axes object, or (fig, ax) tuple if return_handles=True.

    Examples
    --------
    >>> # Basic plot with significance markers
    >>> plot_sliding_classifier(
    ...     classifier_scores,  # shape (20, 100)
    ...     times,              # shape (100,)
    ...     perm_info=perm_results,
    ...     chance=1/3
    ... )
    >>>
    >>> # Custom styling on existing axes
    >>> fig, ax = plt.subplots()
    >>> plot_sliding_classifier(
    ...     scores, times, ax=ax,
    ...     line_kwargs=dict(color='red', linewidth=2),
    ...     fill_kwargs=dict(color='red'),
    ...     se_alpha=0.3
    ... )
    """
    # Extract optional parameters with defaults
    average = kwargs.get("average", True)
    save = kwargs.get("save", None)
    chance = kwargs.get("chance", 0.5)
    se_alpha = kwargs.get("se_alpha", 0.2)
    perm_info = kwargs.get("perm_info", None)
    line_kwargs = kwargs.get("line_kwargs", dict(linestyle="-", color="tab:blue"))
    fill_kwargs = kwargs.get("fill_kwargs", dict(color="tab:blue"))
    sct_kwargs = kwargs.get("sct_kwargs", dict(s=15, marker="o", color="tab:blue"))
    ax = kwargs.get("ax", None)
    return_handles = kwargs.get("return_handles", False)

    # Set alpha for fill
    fill_kwargs["alpha"] = se_alpha

    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()

    # Configure x-axis ticks (every 100ms, labeled every 200ms)
    xtickmarks = np.arange(np.min(time), np.max(time) + 0.1, 0.1)
    xticklabels = ["" if int(t * 10) % 2 else f"{1000 * t:4.0f}" for t in xtickmarks]

    # Compute mean and standard error if averaging
    if average:
        if data.ndim == 2:
            mean_data = np.mean(data, axis=0)
            stderr_data = std_error(data, 0)
        else:
            mean_data = data
            stderr_data = np.zeros(data.shape)
    else:
        mean_data = data.T

    # Plot reference lines
    ax.axhline(chance, color="k", linestyle="--")
    ax.axvline(0.0, color="k", linestyle="-")

    # Plot data
    ax.plot(time, mean_data, **line_kwargs)
    if average:
        ax.fill_between(time, mean_data - stderr_data, mean_data + stderr_data, **fill_kwargs)

    # Configure axes
    ax.set_xticks(xtickmarks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Classification Performance")

    # Add significance markers from permutation test
    if average and perm_info:
        if "cluster_dict" in perm_info:
            # Legacy wrapper format: {'cluster_dict': ..., 'cluster_max_pval': ...}
            stat_info_use = perm_info["cluster_dict"]
            _alpha = perm_info.get("cluster_max_pval", 0.05)
        else:
            # Direct stat_info from sign_flip_permtest / label_shuffle_cv_permtest
            stat_info_use = perm_info
            _alpha = 0.05
        add_significance(
            ax,
            stat_info_use,
            plot_type="linepoints",
            times=time,
            alpha_thresh=_alpha,
            y_pos=chance - 0.02,
            scatter_kwargs=sct_kwargs,
        )

    # Save figure if path provided
    if save:
        plt.savefig(save, bbox_inches="tight")

    # Return handles
    if return_handles:
        return fig, ax
    return ax


def plot_topo_class_pattern(data, time, epoch, **kwargs):
    """
    Create topographic plots for multiple classes at specified time point(s).

    Parameters
    ----------
    data : list of numpy.ndarray
        Input data arrays. Each array should be 3D with dimensions
        (channels x classes x time).
    time : int, list, or numpy.ndarray
        Time point(s) to plot.
        - If int: Single time point
        - If list/array: Multiple time points (plotted as columns)
    epoch : mne.Epoch
        MNE Epoch object containing channel and time information.

    Keyword Arguments
    ----------------
    label : list of str
        Labels for each class/row.
    row_break : integer

    vlim_prctile : tuple, optional
        Percentile limits for color scaling. Default is (5, 95).
    resolution : int, optional
        Resolution of the topomap. Default is 256.
    save : str, optional
        File path to save the figure. If None, figure is not saved.
    return_handles : bool, optional
        If True, returns figure and axis handles. Default is True.
    figsize : tuple, optional
        Figure size in inches. Default is (3.5, 8).
    title : str, optional
        Overall figure title.
    cmap : str, optional
        Colormap to use for topoplots. Default is matplotlib's default.
    colorbar_label : str, optional
        Label for the colorbar. Default is '$\mu$V'.
    dpi : int, optional
        Dots per inch for the figure. Default is matplotlib's default.

    Returns
    -------
    tuple or None
        If return_handles is True, returns (figure, axis_list).
        Otherwise, returns None.
    """
    # Extracting optional parameters with defaults
    label = kwargs.get("label", None)
    row_break = kwargs.get("row_break", 1)
    vlim = kwargs.get("vlim", None)
    vlim_prctile = kwargs.get("vlim_prctile", (5, 95))
    resolution = kwargs.get("resolution", 256)
    save = kwargs.get("save", None)
    return_handles = kwargs.get("return_handles", False)
    times_label = kwargs.get("times_label", [f"{1000 * t:6.0f} ms" for t in time])
    data_scaler = kwargs.get("data_scaler", 1)

    # Figure customization parameters
    figsize = kwargs.get("figsize", (3.5, 8))
    title = kwargs.get("title", None)
    cmap = kwargs.get("cmap", None)
    colorbar_label = kwargs.get("colorbar_label", "$\mu$V")
    ylabel_fontsize = kwargs.get("ylabel_fontsize", 11)
    ylabel_rotation = kwargs.get("ylabel_rotation", 90)
    dpi = kwargs.get("dpi", None)
    title_kwargs = kwargs.get("title_kwargs", None)

    # Normalize time input to numpy array
    if isinstance(time, int) or isinstance(time, float):
        time = [time]
    time = np.atleast_1d(time)

    # Transform timing values to index values
    time_idx = [int(np.argmin(np.abs(epoch.times - t))) for t in time]

    #
    row_mulitplier = 1
    if isinstance(label, str):
        nrows = 1
        ncols = len(time_idx)
    elif isinstance(label, list):
        nrows = len(label)
        ncols = len(time_idx)
        time_idx *= nrows
    else:
        nrows = row_break
        ncols = len(time_idx) // (row_break)
        row_mulitplier = 0

    # Compute global color limits based on all specified time points
    if vlim:
        vlims = np.atleast_1d(vlim)
    else:
        vlims = np.percentile(
            np.array([data_scaler * p[:, :, time_idx].flatten() for p in data]).flatten(),
            vlim_prctile,
        )

    # Create figure with custom gridspec
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # If a title is provided, add it
    if title:
        fig.suptitle(title)

    # Create gridspec with minimal spacing
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,  # +1 for colorbar
        width_ratios=[1] * ncols + [0.05],
        height_ratios=[1] * nrows,
        hspace=0.02,
        wspace=0.1,
    )

    # Adjust the overall gridspec position
    gs.update(left=0.02, right=0.93)

    # Create subplots and plot topographies
    ax = []
    topomap_objects = []

    for row in range(nrows):
        row_axes = []
        row_topos = []

        for col in range(ncols):
            # Create subplot
            current_ax = fig.add_subplot(gs[row, col])
            row_axes.append(current_ax)

            # Plot topomap
            plot_kwargs = {
                "res": resolution,
                "vlim": vlims,
                "axes": current_ax,
                "show": False,
            }
            if cmap:
                plot_kwargs["cmap"] = cmap

            top = mne.viz.plot_topomap(
                np.array(
                    [
                        data_scaler * p[:, row_mulitplier * row, time_idx[row * (ncols) + col]]
                        for p in data
                    ]
                ).mean(0),
                epoch.info,
                **plot_kwargs,
            )
            row_topos.append(top)

            # Add time label to first plot in row
            if label:
                if col == 0:
                    current_ax.set_ylabel(
                        label[row], fontsize=ylabel_fontsize, rotation=ylabel_rotation
                    )

                # Add time point label to first plot in column
                if row == 0:
                    current_ax.set_title(times_label[row * (ncols) + col], **title_kwargs)
            else:
                current_ax.set_title(times_label[row * (ncols) + col], **title_kwargs)

        ax.append(row_axes)
        topomap_objects.append(row_topos)

    # Add colorbar to last column
    middle_row = nrows // 2
    cbar_ax = fig.add_subplot(gs[middle_row, -1])
    cbar = fig.colorbar(topomap_objects[middle_row][0][0], cax=cbar_ax)
    cbar.set_label(colorbar_label, labelpad=1)

    # Save figure if path is provided
    if save:
        plt.savefig(save, bbox_inches="tight")

    # Return figure handles if requested
    if return_handles:
        return fig, ax, cbar


def plot_generalizing_classifier(data, timex, timey, **kwargs):
    """
    Plot temporal generalization matrix for classifier cross-decoding results.

    Creates a 2D heatmap or contour plot showing classifier performance across
    training and testing time combinations, with optional significance masking
    from cluster-based permutation tests.

    Parameters
    ----------
    data : numpy.ndarray
        2D array of classification performance with shape (n_train_times, n_test_times).
        Values typically represent accuracy, AUC, or probability.
    timex : numpy.ndarray
        Time vector for x-axis (test times) in seconds.
    timey : numpy.ndarray
        Time vector for y-axis (train times) in seconds.

    Keyword Arguments
    -----------------
    perm_info : dict, optional
        Permutation test results for significance masking. Expected keys:
        - 'cluster_dict': dict with 'pos' and 'neg' cluster lists
        - 'statistic': str, name of statistic ('cluster_area' or 'cluster_tsum')
        - 'cluster_max_pval': float, threshold for significance
    display : str, default='imshow'
        Display mode: 'imshow' for heatmap, 'contourf' for filled contours.
    n_levels : int, default=20
        Number of contour levels for 'contourf' display mode.
    imshow_kwargs : dict, optional
        Keyword arguments passed to ax.imshow(). Defaults include extent and origin.
    contourf_kwargs : dict, optional
        Keyword arguments passed to ax.contourf() for filled contours.
    line_kwargs : dict, default=dict(color='k', linestyle='--', alpha=0.5)
        Keyword arguments for reference lines (zero lines and diagonal).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
    cmap : str, default='RdBu_r'
        Colormap for the plot.
    vmin, vmax : float, optional
        Color limits. If not provided, uses 2.5 and 97.5 percentiles.
    colorbar_label : str, default='Classification Performance'
        Label for the colorbar.
    save : str, optional
        File path to save the figure. If None, figure is not saved.
    return_handles : bool, default=False
        If True, returns (fig, ax, cax) tuple.

    Returns
    -------
    matplotlib.axes.Axes or tuple
        Returns the axes object, or (fig, ax, cax) tuple if return_handles=True.

    Notes
    -----
    For the 'imshow' display mode, non-significant regions are dimmed using
    alpha transparency based on the significance mask.

    For the 'contourf' display mode, significant cluster boundaries are outlined
    with black contour lines.

    Examples
    --------
    >>> # Basic temporal generalization plot
    >>> plot_generalizing_classifier(
    ...     cross_temp_matrix,  # shape (80, 120)
    ...     test_times,         # shape (120,)
    ...     train_times,        # shape (80,)
    ...     perm_info=perm_results
    ... )
    >>>
    >>> # Contour plot with custom colormap
    >>> fig, ax = plt.subplots()
    >>> plot_generalizing_classifier(
    ...     data, test_times, train_times,
    ...     ax=ax, display='contourf', cmap='viridis'
    ... )
    """
    # Extract optional parameters with defaults
    perm_info = kwargs.get("perm_info", None)
    display = kwargs.get("display", "imshow")
    n_levels = kwargs.get("n_levels", 20)
    cmap = kwargs.get("cmap", "RdBu_r")
    colorbar_label = kwargs.get("colorbar_label", "Classification Performance")
    ax = kwargs.get("ax", None)
    save = kwargs.get("save", None)
    return_handles = kwargs.get("return_handles", False)
    line_kwargs = kwargs.get("line_kwargs", dict(color="k", linestyle="--", alpha=0.5))

    # Determine color limits
    vmin = kwargs.get("vmin", np.percentile(data.flatten(), 2.5))
    vmax = kwargs.get("vmax", np.percentile(data.flatten(), 97.5))

    # Default kwargs for imshow
    imshow_defaults = {
        "extent": [timex[0], timex[-1], timey[0], timey[-1]],
        "origin": "lower",
        "aspect": "auto",
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
    }
    imshow_kwargs = kwargs.get("imshow_kwargs", {})
    imshow_defaults.update(imshow_kwargs)
    imshow_kwargs = imshow_defaults

    # Create figure if no axes provided
    if ax is None:
        fig, (ax, cax) = plt.subplots(
            nrows=1,
            ncols=2,
            gridspec_kw={
                "height_ratios": [1],
                "width_ratios": [1, 0.06],
                "hspace": 0,
                "wspace": 0.01,
            },
            figsize=(10, 8),
        )
    else:
        fig = ax.get_figure()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

    # Plot based on display mode
    if display == "imshow":
        img = ax.imshow(data, **imshow_kwargs)

    elif display == "contourf":
        X, Y = np.meshgrid(timex, timey)
        levels = np.linspace(vmin, vmax, n_levels)
        img = ax.contourf(X, Y, data, levels=levels, cmap=cmap, extend="both")

    else:
        raise ValueError(f"Unknown display mode: {display}. Use 'imshow' or 'contourf'.")

    # Overlay significance using add_significance
    if perm_info:
        if "cluster_dict" in perm_info:
            # Legacy wrapper format: {'cluster_dict': ..., 'cluster_max_pval': ...}
            stat_info_use = perm_info["cluster_dict"]
            _alpha = perm_info.get("cluster_max_pval", 0.05)
        else:
            # Direct stat_info from sign_flip_permtest / label_shuffle_cv_permtest
            stat_info_use = perm_info
            _alpha = 0.05
        sig_plot_type = "contour" if display == "contourf" else "alpha"
        add_significance(
            ax,
            stat_info_use,
            plot_type=sig_plot_type,
            timex=timex,
            timey=timey,
            alpha_thresh=_alpha,
        )

    # Add reference lines
    ax.axhline(0, **line_kwargs)
    ax.axvline(0, **line_kwargs)

    # Add diagonal line (identity line where train time = test time)
    diag_min = max(timey[0], timex[0])
    diag_max = min(timey[-1], timex[-1])
    ax.plot([diag_min, diag_max], [diag_min, diag_max], **line_kwargs)

    # Add colorbar
    fig.colorbar(img, cax=cax, label=colorbar_label)

    # Save figure if path provided
    if save:
        plt.savefig(save, bbox_inches="tight")

    # Return handles
    if return_handles:
        return fig, ax, cax
    return ax


# Plot Sequenceness Results


def plot_sequenceness(data, lags, perms, **kwargs):

    # Extracting optional parameters with defaults
    save = kwargs.get("save", None)
    se_alpha = kwargs.get("se_alpha", 0.2)
    plot_subs = kwargs.get("plot_subs", False)
    pool_permutations = kwargs.get("pool_permutations", False)
    plot_dist_perm = kwargs.get("plot_dist_perm", None)
    plot_max_perm = kwargs.get("plot_max_perm", None)
    return_handles = kwargs.get("return_handles", False)
    return_cbar_handle = kwargs.get("return_cbar_handle", False)
    ax = kwargs.get("ax", None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if pool_permutations:
        perm_dist = np.reshape(perms, (perms.shape[0] * perms.shape[1], perms.shape[-1]))
    else:
        perm_dist = np.mean(perms, 0)

    if isinstance(plot_dist_perm, str):
        if plot_dist_perm == "lines":
            ax.plot(lags, perm_dist.T, color="k", alpha=0.1)
        elif plot_dist_perm == "area":
            ax.fill_between(
                lags,
                np.min(perm_dist, 0),
                np.max(perm_dist, 0),
                color="k",
                alpha=0.1,
                label="Permutation Distribution",
            )

    if plot_max_perm:
        perm_dist_abs = np.max(np.abs(perm_dist), 1).flatten()
        if isinstance(plot_max_perm, str):
            ax.axhline(
                np.max(perm_dist_abs),
                color="k",
                linestyle="--",
                label="Max Permutation",
            )
            ax.axhline(-np.max(perm_dist_abs), color="k", linestyle="--")
        if isinstance(plot_dist_perm, float):
            ax.axhline(
                np.percentile(perm_dist_abs, plot_dist_perm),
                color="k",
                linestyle="--",
                label=f"{plot_dist_perm}% Permutation",
            )
            ax.axhline(-np.percentile(perm_dist_abs, plot_dist_perm), color="k", linestyle="--")

    ax.axhline(0.0, color="k", linestyle="-")
    if plot_subs:
        norm = Normalize(
            vmin=np.min(plot_subs["color_values"]),
            vmax=np.max(plot_subs["color_values"]),
        )
        sm = plt.cm.ScalarMappable(cmap=plot_subs["cmap"], norm=norm)
        sm.set_array([])  # This is necessary for the colorbar to work
        for p in range(data.shape[0]):
            color = plt.get_cmap(plot_subs["cmap"])(norm(plot_subs["color_values"][p]))
            ax.plot(lags, data[p, :], color=color)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(plot_subs["cbar_title"], fontsize=12)
    else:
        ax.plot(lags, np.mean(data, 0), label="Evidence for Replay")
        if data.shape[0] > 1:
            ax.fill_between(
                lags,
                np.mean(data, 0) - std_error(data, 0),
                np.mean(data, 0) + std_error(data, 0),
                alpha=se_alpha,
            )
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim((0, np.max(lags)))
    ax.set_ylabel("sequenceness")
    ax.set_xlabel("time lag (ms)")
    ax.legend()
    if save:
        plt.savefig(save, bbox_inches="tight")
    if return_handles:
        if return_cbar_handle:
            return fig, ax, cbar
        else:
            return fig, ax
    if ax is not None:
        if return_cbar_handle:
            return ax, cbar
        else:
            return ax


def plot_sequenceness_subjects(data, lags, perms, **kwargs):
    """
    Plot individual subject sequenceness traces with the group mean overlaid.

    Parameters
    ----------
    data : np.ndarray, shape (n_subs, n_lags)
        Per-subject sequenceness (e.g. forward − backward).
    lags : np.ndarray, shape (n_lags,)
        Lag values in ms.
    perms : np.ndarray, shape (n_subs, n_perms, n_lags)
        Per-subject permutation null distributions.

    Keyword Arguments
    -----------------
    ax : matplotlib.axes.Axes, optional
    sub_color : color-spec, default='C0'
        Line colour for individual subject traces.
    sub_alpha : float, default=0.3
        Opacity of individual subject traces.
    mean_color : color-spec, default='k'
        Colour of the group mean line.
    mean_lw : float, default=2.0
        Line-width of the group mean.
    se_alpha : float, default=0.2
        Opacity of the ±SEM shading around the mean.
    plot_max_perm : bool, default=True
        If True, draw dashed horizontal lines at ±max(permutation distribution).
    return_handles : bool, default=False
    save : str or None, default=None

    Returns
    -------
    ax or (fig, ax)
    """
    ax = kwargs.get("ax", None)
    sub_color = kwargs.get("sub_color", "C0")
    sub_alpha = kwargs.get("sub_alpha", 0.3)
    mean_color = kwargs.get("mean_color", "k")
    mean_lw = kwargs.get("mean_lw", 2.0)
    se_alpha = kwargs.get("se_alpha", 0.2)
    plot_max_perm = kwargs.get("plot_max_perm", True)
    return_handles = kwargs.get("return_handles", False)
    save = kwargs.get("save", None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if plot_max_perm:
        perm_dist = np.mean(perms, 0)  # (n_perms, n_lags)
        thresh = np.max(np.abs(perm_dist).max(1))
        ax.axhline(thresh, color="k", linestyle="--", linewidth=0.8, label="Max perm.")
        ax.axhline(-thresh, color="k", linestyle="--", linewidth=0.8)

    ax.axhline(0.0, color="k", linestyle="-", linewidth=0.5)

    for s in range(data.shape[0]):
        ax.plot(lags, data[s], color=sub_color, alpha=sub_alpha, linewidth=0.8)

    mn = data.mean(0)
    se = std_error(data, 0)
    ax.plot(lags, mn, color=mean_color, linewidth=mean_lw, label="Mean")
    ax.fill_between(lags, mn - se, mn + se, color=mean_color, alpha=se_alpha)

    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim((0, np.max(lags)))
    ax.set_ylabel("sequenceness")
    ax.set_xlabel("time lag (ms)")
    ax.legend()

    if save:
        plt.savefig(save, bbox_inches="tight")

    if return_handles:
        return fig, ax
    return ax


def add_significance(
    ax,
    stat_info,
    plot_type="area",
    times=None,
    timex=None,
    timey=None,
    alpha_thresh=0.05,
    **kwargs,
):
    """
    Add significance indicators to an existing 1D or 2D plot.

    Accepts stat_info directly from sign_flip_permtest or
    label_shuffle_cv_permtest, or the legacy pre-processed dict used by
    add_signif_timepts (keys: times, mask, corrected_pvals, permtest_pval).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    stat_info : dict
        Cluster format (both permtest functions):
            {'pos': [cluster_dicts], 'neg': [cluster_dicts]}
            Each cluster dict must have 'cluster_stat_pval' (or fallback
            'cluster_tsum_pval' / 'cluster_area_pval') and 'cluster_index'.
        Max format (label_shuffle_cv_permtest test='max'):
            {'sig_times': array, 'pval': float}
        Legacy format (add_signif_timepts input):
            {'times': array, 'mask': array, 'corrected_pvals': array,
             'permtest_pval': float}
    plot_type : str, default='area'
        1D modes: 'area', 'lines', 'linepoints', 'points', 'bar'
        2D modes: 'alpha' (dim non-significant), 'contour' (outline
                  cluster boundaries), 'hatch' (hatch significant areas).
        'area' is mapped to 'alpha' automatically in 2D mode.
    times : array-like, optional
        Full (uncropped) time axis for 1D mode. Required for 1D mode unless
        using legacy format (which includes 'times' in stat_info).
    timex : array-like, optional
        Full test-time axis for 2D mode (x / column dimension).
    timey : array-like, optional
        Full train-time axis for 2D mode (y / row dimension).
    alpha_thresh : float, default=0.05
        Significance threshold. Standard convention: p < alpha_thresh.

    Keyword Arguments
    -----------------
    label : str
        Match color to an existing line (1D 'points' / 'area' mode).
    y_pos : float
        Y-position for 'lines', 'linepoints', 'bar' (1D).
    alpha : float, default=0.3
        Transparency for patches and fills (1D).
    height : float
        Patch height for 'lines' and 'bar' (1D).
    use_pval_alpha : bool, default=False
        Modulate alpha by p-value in 'linepoints' (1D).
    use_pval_colormap : bool, default=False
        Color-code bars by p-value in 'bar' (1D).
    colormap : str, default='viridis_r'
        Colormap for p-value coding (1D 'bar').
    color : str or tuple
        Override color (1D).
    patch_kwargs : dict
        Extra kwargs for Rectangle patches (1D 'area', 'bar').
    line_kwargs : dict
        Extra kwargs for line plotting (1D 'lines').
    scatter_kwargs : dict
        Extra kwargs for scatter (1D 'linepoints', 'points').
    dim_alpha : float, default=0.7
        Opacity of non-significant overlay (2D 'alpha' mode).
    contour_kwargs : dict
        Extra kwargs for contour/contourf (2D 'contour', 'hatch').

    Returns
    -------
    artists : list
        Matplotlib artists added to the axes.
    """
    is_2d = (timex is not None) and (timey is not None)
    is_legacy = ("mask" in stat_info) and ("permtest_pval" in stat_info)

    # ------------------------------------------------------------------ #
    # Build sig_mask and corrected_pvals from stat_info                   #
    # ------------------------------------------------------------------ #
    if is_legacy:
        times = np.asarray(stat_info["times"])
        mask_arr = np.asarray(stat_info["mask"])
        corrected_pvals = np.asarray(stat_info["corrected_pvals"], dtype=float)
        sig_mask = (mask_arr > 0) & (corrected_pvals < stat_info["permtest_pval"])

    elif "sig_times" in stat_info:
        # Max-statistic format
        sig_coords = np.asarray(stat_info["sig_times"])
        global_pval = float(stat_info.get("pval", 1.0))
        if is_2d:
            sig_mask = np.zeros((len(timey), len(timex)), dtype=bool)
            corrected_pvals = np.full(sig_mask.shape, np.nan)
            if global_pval < alpha_thresh and len(sig_coords) > 0:
                sig_mask[sig_coords[:, 0], sig_coords[:, 1]] = True
                corrected_pvals[sig_coords[:, 0], sig_coords[:, 1]] = global_pval
        else:
            times = np.asarray(times)
            sig_mask = np.zeros(len(times), dtype=bool)
            corrected_pvals = np.full(len(times), np.nan)
            if global_pval < alpha_thresh and len(sig_coords) > 0:
                sig_mask[sig_coords[:, -1]] = True
                corrected_pvals[sig_coords[:, -1]] = global_pval

    elif "pos" in stat_info or "neg" in stat_info:
        # Cluster format (sign_flip_permtest or label_shuffle_cv_permtest)
        if is_2d:
            sig_mask = np.zeros((len(timey), len(timex)), dtype=bool)
            corrected_pvals = np.full(sig_mask.shape, np.nan)
        else:
            times = np.asarray(times)
            sig_mask = np.zeros(len(times), dtype=bool)
            corrected_pvals = np.full(len(times), np.nan)

        for direction in ("pos", "neg"):
            for cluster in stat_info.get(direction, []):
                pval = cluster.get(
                    "cluster_stat_pval",
                    cluster.get(
                        "cluster_tsum_pval",
                        cluster.get("cluster_area_pval", 1.0),
                    ),
                )
                if pval < alpha_thresh:
                    idx = np.asarray(cluster["cluster_index"])
                    if is_2d:
                        # Validate bounds
                        if (idx[:, 0] >= len(timey)).any() or (idx[:, 1] >= len(timex)).any():
                            raise ValueError(
                                f"Cluster index out of bounds for "
                                f"timey ({len(timey)}) x timex ({len(timex)}) grid."
                            )
                        sig_mask[idx[:, 0], idx[:, 1]] = True
                        corrected_pvals[idx[:, 0], idx[:, 1]] = pval
                    else:
                        time_idx = idx[:, -1]
                        sig_mask[time_idx] = True
                        corrected_pvals[time_idx] = pval
    else:
        raise ValueError(
            "stat_info format not recognized. Expected keys: 'pos'/'neg' "
            "(cluster format), 'sig_times' (max format), or 'mask'/'permtest_pval' "
            "(legacy format)."
        )

    if not np.any(sig_mask):
        return []

    artists = []

    # ------------------------------------------------------------------ #
    # 2D rendering                                                         #
    # ------------------------------------------------------------------ #
    if is_2d:
        timex = np.asarray(timex)
        timey = np.asarray(timey)
        dim_alpha = kwargs.get("dim_alpha", 0.7)
        contour_kwargs = kwargs.get("contour_kwargs", {})
        extent = [timex[0], timex[-1], timey[0], timey[-1]]
        X, Y = np.meshgrid(timex, timey)

        # map 'area' → 'alpha' for 2D
        _plot_type_2d = "alpha" if plot_type == "area" else plot_type

        if _plot_type_2d == "alpha":
            overlay = np.ones((*sig_mask.shape, 4))
            overlay[sig_mask, 3] = 0.0
            overlay[~sig_mask, 3] = dim_alpha
            img = ax.imshow(
                overlay,
                extent=extent,
                origin="lower",
                aspect="auto",
                zorder=3,
                interpolation="nearest",
            )
            artists.append(img)

        elif _plot_type_2d == "contour":
            ckw = {"colors": "black", "linewidths": 1.5, **contour_kwargs}
            cs = ax.contour(X, Y, sig_mask.astype(float), levels=[0.5], **ckw)
            artists.append(cs)

        elif _plot_type_2d == "hatch":
            hkw = {"hatches": ["//"], "alpha": 0, **contour_kwargs}
            cf = ax.contourf(X, Y, sig_mask.astype(float), levels=[0.5, 1.5], **hkw)
            artists.append(cf)

        else:
            raise ValueError(
                f"Unknown plot_type for 2D mode: {plot_type!r}. Use 'alpha', 'contour', or 'hatch'."
            )
        return artists

    # ------------------------------------------------------------------ #
    # 1D rendering (logic from add_signif_timepts)                        #
    # ------------------------------------------------------------------ #
    label = kwargs.get("label", None)
    y_pos = kwargs.get("y_pos", None)
    alpha = kwargs.get("alpha", 0.3 if plot_type != "linepoints" else 1.0)
    height = kwargs.get("height", None)
    use_pval_alpha = kwargs.get("use_pval_alpha", False)
    use_pval_colormap = kwargs.get("use_pval_colormap", False)
    colormap = kwargs.get("colormap", "viridis_r")
    custom_color = kwargs.get("color", None)

    patch_kwargs = kwargs.get("patch_kwargs", {})
    line_kwargs = kwargs.get("line_kwargs", {})
    scatter_kwargs = kwargs.get("scatter_kwargs", {})

    if custom_color is not None:
        color = custom_color
    elif label is not None:
        color = None
        for line in ax.get_lines():
            if line.get_label() == label:
                color = line.get_color()
                break
        if color is None:
            color = "red"
    else:
        color = "red"

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    if height is None:
        height = y_range * 0.05

    if y_pos is None:
        if plot_type in ["lines", "linepoints", "bar"]:
            y_pos = y_min - y_range * 0.1

    if len(times) > 1:
        dt = times[1] - times[0]
    else:
        dt = 1.0

    if plot_type == "area":
        diff_mask = np.diff(np.concatenate(([False], sig_mask, [False])).astype(int))
        starts = np.where(diff_mask == 1)[0]
        ends = np.where(diff_mask == -1)[0]

        patch_kwargs_merged = {"alpha": alpha, "color": color, **patch_kwargs}

        for start, end in zip(starts, ends):
            x_start = times[start]
            width = times[end - 1] - times[start] + dt
            rect = Rectangle((x_start, y_min), width, y_range, **patch_kwargs_merged)
            ax.add_patch(rect)
            artists.append(rect)

    elif plot_type == "lines":
        sig_times = times[sig_mask]
        if len(sig_times) > 0:
            diff_times = np.diff(sig_times)
            median_dt = np.median(np.diff(times))
            breaks = np.where(diff_times > 2 * median_dt)[0] + 1

            segments = []
            start_idx = 0
            for break_idx in breaks:
                segments.append([sig_times[start_idx], sig_times[break_idx - 1]])
                start_idx = break_idx
            segments.append([sig_times[start_idx], sig_times[-1]])

            line_kwargs_merged = {
                "color": color,
                "alpha": alpha,
                "linewidth": height * 20,
                **line_kwargs,
            }

            for seg_start, seg_end in segments:
                line = ax.plot([seg_start, seg_end], [y_pos, y_pos], **line_kwargs_merged)[0]
                artists.append(line)

    elif plot_type == "linepoints":
        sig_times = times[sig_mask]
        sig_pvals = corrected_pvals[sig_mask]

        if len(sig_times) > 0:
            scatter_kwargs_merged = {"color": color, "s": 20, **scatter_kwargs}

            if use_pval_alpha and not np.all(np.isnan(sig_pvals)):
                valid_pval_mask = ~np.isnan(sig_pvals)
                if np.any(valid_pval_mask):
                    norm_pvals = 1.0 - (
                        sig_pvals[valid_pval_mask] / np.max(sig_pvals[valid_pval_mask])
                    )
                    alphas = 0.3 + 0.7 * norm_pvals

                    for i, (t, a) in enumerate(zip(sig_times[valid_pval_mask], alphas)):
                        scatter_kwargs_point = {**scatter_kwargs_merged, "alpha": a}
                        point = ax.scatter(t, y_pos, **scatter_kwargs_point)
                        artists.append(point)

                    nan_mask = np.isnan(sig_pvals)
                    if np.any(nan_mask):
                        scatter_kwargs_merged["alpha"] = alpha
                        points = ax.scatter(
                            sig_times[nan_mask],
                            [y_pos] * np.sum(nan_mask),
                            **scatter_kwargs_merged,
                        )
                        artists.append(points)
            else:
                scatter_kwargs_merged["alpha"] = alpha
                points = ax.scatter(sig_times, [y_pos] * len(sig_times), **scatter_kwargs_merged)
                artists.append(points)

    elif plot_type == "points":
        sig_times = times[sig_mask]

        if len(sig_times) > 0 and label is not None:
            target_line = None
            for line in ax.get_lines():
                if line.get_label() == label:
                    target_line = line
                    break

            if target_line is not None:
                line_x = target_line.get_xdata()
                line_y = target_line.get_ydata()

                sig_y = np.interp(sig_times, line_x, line_y)

                scatter_kwargs_merged = {
                    "color": color,
                    "s": 30,
                    "alpha": alpha,
                    "zorder": 10,
                    **scatter_kwargs,
                }
                points = ax.scatter(sig_times, sig_y, **scatter_kwargs_merged)
                artists.append(points)

    elif plot_type == "bar":
        sig_times = times[sig_mask]
        sig_pvals = corrected_pvals[sig_mask]

        if len(sig_times) > 0:
            if use_pval_colormap and not np.all(np.isnan(sig_pvals)):
                cmap = cm.get_cmap(colormap)
                valid_pval_mask = ~np.isnan(sig_pvals)

                if np.any(valid_pval_mask):
                    valid_pvals = sig_pvals[valid_pval_mask]
                    norm_pvals = (valid_pvals - np.min(valid_pvals)) / (
                        np.max(valid_pvals) - np.min(valid_pvals)
                    )
                    colors = cmap(norm_pvals)

                    patch_kwargs_merged = {"alpha": alpha, **patch_kwargs}

                    for i, (t, c) in enumerate(zip(sig_times[valid_pval_mask], colors)):
                        rect = Rectangle(
                            (t - dt / 2, y_pos),
                            dt,
                            height,
                            color=c,
                            **patch_kwargs_merged,
                        )
                        ax.add_patch(rect)
                        artists.append(rect)

                    nan_mask = np.isnan(sig_pvals)
                    if np.any(nan_mask):
                        patch_kwargs_merged["color"] = color
                        for t in sig_times[nan_mask]:
                            rect = Rectangle((t - dt / 2, y_pos), dt, height, **patch_kwargs_merged)
                            ax.add_patch(rect)
                            artists.append(rect)
            else:
                patch_kwargs_merged = {"alpha": alpha, "color": color, **patch_kwargs}
                for t in sig_times:
                    rect = Rectangle((t - dt / 2, y_pos), dt, height, **patch_kwargs_merged)
                    ax.add_patch(rect)
                    artists.append(rect)

    else:
        raise ValueError(
            f"Unknown plot_type for 1D mode: {plot_type!r}. "
            "Use 'area', 'lines', 'linepoints', 'points', or 'bar'."
        )

    return artists
