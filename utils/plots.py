# -*- coding: utf-8 -*-
"""
Plotting Functions

@author: Christopher Postzich
@github: mcpost
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm
from utils.utils import std_error


### Preprocessing Plot Functions


def plot_data_quant(trialnum_loc, rest_dur, labels, ytick_names, **kwargs):
    
    # Extracting optional parameters with defaults
    width = kwargs.get('width', 0.2)
    colors = kwargs.get('colors', ('#1f77b4ff','#ff7f0eff','#2ca02cff'))
    hspace = kwargs.get('hspace', 0.05)
    wspace = kwargs.get('wspace', 0.15)
    save = kwargs.get('save', None)
    
    y = np.arange(trialnum_loc.shape[0])
    
    fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, 
                           gridspec_kw={'height_ratios': [1], 'width_ratios': [1,1],
                                        'hspace': hspace, 'wspace': wspace}, 
                           figsize=(11,5))
    for i,(tn,c) in enumerate(zip(trialnum_loc.T, colors)):
        y_cur = y + width*i - width*(trialnum_loc.shape[1]//2)
        ax[0].barh(y_cur, tn, width, color=c) 
    ax[0].set_yticks(y, ytick_names) 
    ax[0].set_xlabel("Trial Number")  
    ax[0].legend(labels) 
    ax[1].barh(y, rest_dur.squeeze(), width+0.4, color='blue')
    ax[1].set_xlabel("Resting Signal (s)")
    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_detrending(epochs, data_detr, trend1, trend30, weight30, chan_list, trl_list, **kwargs):
    
    # Extracting optional parameters with defaults
    xlim = kwargs.get('xlim', (-4.0, 8.0))
    ylim = kwargs.get('ylim', (-0.00015, 0.00015))
    hspace = kwargs.get('hspace', 0.01)
    wspace = kwargs.get('wspace', 0.02)
    save = kwargs.get('save', None)
    
    fig, axes = plt.subplots(
        nrows=len(chan_list)+1, ncols=len(trl_list), sharex='col', sharey='row',
        gridspec_kw={'height_ratios': [1]*(len(chan_list)+1), 'width_ratios': [1]*len(trl_list),
                     'hspace': hspace, 'wspace': wspace},
        figsize=(9.97, 8.03)
    )
    
    for i, ch in enumerate(chan_list):
        for j, trl in enumerate(trl_list):
            axes[i, j].plot(epochs.times, epochs._data[trl, i, :], color='C0')
            axes[i, j].plot(epochs.times, data_detr[trl, :, i], color='C1')
            axes[i, j].plot(epochs.times, trend1[trl, :, i], color='red')
            axes[i, j].plot(epochs.times, trend30[trl, :, i], color='green')
            axes[i, j].set_xlim(xlim)
            axes[i, j].set_ylim(ylim)
            if i == len(chan_list) - 1:
                axes[len(chan_list), j].pcolormesh(
                    epochs.times, np.arange(weight30.shape[-1]), weight30[trl, :, :].T, cmap='Greys'
                )
                axes[len(chan_list), j].set_xlim(xlim)
                axes[len(chan_list), j].set_ylabel('ch. weights')
                axes[len(chan_list), j].set_xlabel('samples')
    if save:
        plt.savefig(save, bbox_inches='tight')
        plt.close()


def plot_autoreject_results(ar, epochs, reject_log, **kwargs):
    
    # Extracting optional parameters with defaults
    bl_wind = kwargs.get('bl_wind', (-0.2, 0.0))
    save = kwargs.get('save', None)
    aspect = kwargs.get('aspect', 100000)
    
    fig, axes = plt.subplots(nrows=2, ncols=2,
                             gridspec_kw={'height_ratios': [1,1], 'width_ratios': [1,1],
                                          'hspace': 0.05, 'wspace': 0.15},
                             figsize=(11,7))
    loss = ar.loss_['eeg'].mean(axis=-1)
    im1 = axes[0,0].matshow(loss.T * 1e6, cmap=plt.get_cmap('viridis'))
    axes[0,0].set_xticks(range(len(ar.consensus)), ['%.1f' % c for c in ar.consensus])
    axes[0,0].set_yticks(range(len(ar.n_interpolate)), ar.n_interpolate)
    # Draw rectangle at location of best parameters
    idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
    rect = Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor='r', facecolor='none')
    axes[0,0].add_patch(rect)
    axes[0,0].xaxis.tick_bottom()
    axes[0,0].set_xlabel(r'Consensus percentage $\kappa$')
    axes[0,0].set_ylabel(r'Max sensors interpolated $\rho$')
    axes[0,0].set_title('EEG:\nMean cross validation error')
    plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    # Draw rectangle at location of best parameters
    idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
    rect = Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor='r', facecolor='none')
    axes[0,1].axhline(0.0, color="k", linestyle="--")
    axes[0,1].axvline(0.0, color="k", linestyle="--")
    axes[0,1].plot(epochs.times, epochs.apply_baseline(bl_wind)[~reject_log.bad_epochs].average().data.mean(0), 'b', label='Drop Trials')
    axes[0,1].plot(epochs.times, epochs.apply_baseline(bl_wind).average().data.mean(0), 'r', label='All Trials')
    axes[0,1].set_xlabel(r'Time (s)')
    axes[0,1].set_ylabel(r'$\mu$V')
    axes[0,1].set_title('EEG: Signal with and without correction')
    axes[0,1].legend()
    axes[0,1].set_aspect(aspect)
    gs = axes[1, 1].get_gridspec()
    axes[1,0].remove()
    axes[1,1].remove()
    axbig = fig.add_subplot(gs[1,0:2])
    ar.get_reject_log(epochs).plot(orientation='horizontal',show_names=1, aspect=5, ax=axbig)
    if save:
        plt.savefig(save, bbox_inches='tight')
        plt.close()



def plot_topo_trials(epochs):
    # Get the layout from the montage
    pos = mne.find_layout(epochs.info).pos

    # Create a figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Create subplots at channel positions
    for idx, ch_name in enumerate(epochs.info['ch_names']):
        # Calculate position for the subplot
        x, y = pos[idx, 0], pos[idx, 1]
        width = height = 0.1  # Adjust the width and height for each subplot as needed

        sub_ax = fig.add_axes([x - width / 2, y - height / 2, width, height])
        
        # Plot trials for each channel
        for epoch in range(len(epochs)):
            sub_ax.plot(epochs.times, epochs.get_data()[epoch, idx, :], alpha=0.5)
        
        sub_ax.set_title(ch_name, fontsize=10)
        sub_ax.set_xlim(epochs.times[0], epochs.times[-1])
        sub_ax.axis('off')  # Turn off axis

    # Hide the main axis
    ax.axis('off')
    plt.show()

    
def plot_topo_trials_new(epochs):
    chan_grid = {
        'F9':    0, 'F7':    1, 'F3':    2, 'Fz':    3, 'F4':    4, 'F8':    5, 'F10':  6, 
                    'FC5':   8, 'FC3':   9, 'FCz':  10, 'FC4':  11, 'FC6':  12,
        'T7':   14,             'C3':   16,             'C4':   18,             'T8':   20, 
                    'CP5':  22, 'CP3':  23,             'CP4':  25, 'CP6':  26, 
        'TP9':  28, 'P7':   29, 'P3':   30, 'Pz':   31, 'P4':   32, 'P8':   33, 'TP10': 34, 
                                            'POz':  38, 
                                'O1':   44, 'Oz':   45, 'O2':   46
        }

    rev_chan_grid = {chan_grid[k]: k for k in chan_grid.keys()}


    # Create a figure
    fig_trial, ax = plt.subplots(7,7, subplot_kw={},
                           gridspec_kw={'height_ratios': [1]*7, 'width_ratios': [1]*7,
                                        'hspace': 0.05, 'wspace': 0.1},
                           figsize=(15, 10))
    for i,a in enumerate(ax.flat):
        a.axis('off')
        if i in rev_chan_grid.keys():
            a.plot(epochs.times, epochs._data[:, epochs.ch_names.index(rev_chan_grid[i]), :].T, alpha=0.5)
            a.set_title(rev_chan_grid[i], fontsize=10)
            a.set_xlim(epochs.times[0], epochs.times[-1])

    # Function to handle click events on subplots
    def on_click(event):
        if event.inaxes is not None:
            channel_index = [i for i,a in enumerate(ax.flat) if a == event.inaxes][0]
            if channel_index in rev_chan_grid.keys():
                plot_trial_data(epochs.ch_names.index(rev_chan_grid[channel_index]))

    # Function to plot trial data for the selected channel
    def plot_trial_data(channel_index):
        fig_chan, ax_trial = plt.subplots(figsize=(10, 6))
        lines = []
        for trial in range(epochs._data.shape[0]):
            line, = ax_trial.plot(epochs.times, epochs._data[trial, channel_index, :].T, label=f'Trial {trial+1}')
            lines.append(line)
        ax_trial.set_xlabel('Timepoints')
        ax_trial.set_ylabel('Value')
        ax_trial.set_title(f'Trial data for Channel {channel_index + 1}')

        # Function to handle hover events
        def on_hover(event):
            for line in lines:
                if line.contains(event)[0]:
                    line.set_linewidth(2)  # Highlight the line
                else:
                    line.set_linewidth(1)  # Reset other lines
            fig_chan.canvas.draw()

        # Function to handle click events
        def onclick(event):
            for trl,line in enumerate(lines):
                if line.contains(event)[0]:
                    ax_trial.annotate(f"Trial {epochs.selection[trl]}", (event.xdata, event.ydata), xytext=(10, 10),
                                      textcoords='offset points', arrowprops=dict(arrowstyle="->"),
                                      color=line.get_color(), bbox=dict(boxstyle="round,pad=0.3", 
                                                                        edgecolor='black', facecolor='white'))
                    fig_chan.canvas.draw()
                    break

        # Connect hover and click events to the trial data plot
        fig_chan.canvas.mpl_connect('motion_notify_event', on_hover)
        fig_chan.canvas.mpl_connect('button_press_event', onclick)

        plt.show()

    # Connect click event to the main figure
    fig_trial.canvas.mpl_connect('button_press_event', on_click)

    plt.show()





def plot_cohens_kappa(kappa_values, scale='landis', ax=None, title=None, custom_scale=None, 
                      show_boxplot=True, show_points=True, jitter=0.05, 
                      boxplot_kwargs=None, scatter_kwargs=None, text_kwargs=None):
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
        'cicchetti': {
            'borders': [0.4, 0.6, 0.75, 1.0],
            'labels': ['Poor', 'Fair', 'Good', 'Excellent'],
            'colors': ['#FF9999', '#FFCC99', '#CCFF99', '#99FF99']
        },
        'fleiss': {
            'borders': [0.4, 0.75, 1.0],
            'labels': ['Poor', 'Fair to Good', 'Excellent'],
            'colors': ['#FF9999', '#FFCC99', '#99FF99']
        },
        'landis': {
            'borders': [0.2, 0.4, 0.6, 0.8, 1.0],
            'labels': ['Slight', 'Fair', 'Moderate', 'Substantial', 'Almost Perfect'],
            'colors': ['#FF9999', '#FFCC99', '#FFFFCC', '#CCFF99', '#99FF99']
        },
        'regier': {
            'borders': [0.2, 0.4, 0.6, 0.8, 1.0],
            'labels': ['Unacceptable', 'Questionable', 'Good', 'Very Good', 'Excellent'],
            'colors': ['#FF9999', '#FFCC99', '#FFFFCC', '#CCFF99', '#99FF99']
        }
    }
    
    # Use custom scale if provided
    if custom_scale is not None:
        if 'borders' not in custom_scale or 'labels' not in custom_scale:
            raise ValueError("Custom scale must contain 'borders' and 'labels' keys")
        
        # Create a copy to avoid modifying the original
        selected_scale = custom_scale.copy()
        
        # Generate colors if not provided
        if 'colors' not in selected_scale:
            n_colors = len(selected_scale['borders'])
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FF9999', '#99FF99'], N=n_colors)
            selected_scale['colors'] = [cmap(i/(n_colors-1)) for i in range(n_colors)]
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
    for i, border in enumerate(selected_scale['borders']):
        rect = Rectangle((prev_border, 0), border - prev_border, 1, 
                                 linewidth=1, 
                                 edgecolor='gray', 
                                 facecolor=selected_scale['colors'][i], 
                                 alpha=0.3)
        ax.add_patch(rect)
        
        # Add text label in the middle of each patch
        midpoint = (prev_border + border) / 2
        
        # Default text parameters
        default_text_kwargs = {
            'horizontalalignment': 'center',
            'verticalalignment': 'bottom',
            'fontsize': 10,
            'alpha': 0.7,
            'y': 0.05  # Position text at the bottom
        }
        
        # Update with user provided text kwargs if any
        if text_kwargs is not None:
            default_text_kwargs.update(text_kwargs)
            
        ax.text(midpoint, default_text_kwargs.pop('y'), selected_scale['labels'][i], **default_text_kwargs)
        
        prev_border = border
    
    # Plot boxplot if requested
    if show_boxplot:
        # Default boxplot parameters (upper half)
        default_boxplot_kwargs = {
            'vert': False,
            'patch_artist': True,
            'widths': 0.3,
            'positions': [0.75],  # Position in the upper half
            'boxprops': {'facecolor': 'white', 'alpha': 0.7},
            'medianprops': {'color': 'black'},
            'whiskerprops': {'color': 'black'},
            'capprops': {'color': 'black'},
            'flierprops': {'markeredgecolor': 'gray'}
        }
        
        # Update with user provided boxplot kwargs if any
        if boxplot_kwargs is not None:
            default_boxplot_kwargs.update(boxplot_kwargs)
        
        # Extract and remove box style properties to handle separately
        box_props = default_boxplot_kwargs.pop('boxprops', {'facecolor': 'white', 'alpha': 0.7})
        median_props = default_boxplot_kwargs.pop('medianprops', {'color': 'black'})
        whisker_props = default_boxplot_kwargs.pop('whiskerprops', {'color': 'black'})
        cap_props = default_boxplot_kwargs.pop('capprops', {'color': 'black'})
        flier_props = default_boxplot_kwargs.pop('flierprops', {'markeredgecolor': 'gray'})
        
        bp = ax.boxplot(kappa_values, **default_boxplot_kwargs)
        
        # Apply style properties
        plt.setp(bp['boxes'], **box_props)
        plt.setp(bp['medians'], **median_props)
        plt.setp(bp['whiskers'], **whisker_props)
        plt.setp(bp['caps'], **cap_props)
        plt.setp(bp['fliers'], **flier_props)
    
    # Plot individual points if requested
    if show_points:
        # Default scatter parameters (lower half)
        default_scatter_kwargs = {
            'color': 'black',
            's': 8,
            'alpha': 0.7,
            'zorder': 3
        }
        
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
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
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
    scatter_kwargs = kwargs.get('scatter_kwargs', dict())
    polyfit = kwargs.get('polyfit', False)
    polyfit_kwargs = kwargs.get('polyfit_kwargs', dict(deg=1))
    line_kwargs = kwargs.get('line_kwargs', dict())
    save = kwargs.get('save', None)
    return_handles = kwargs.get('return_handles', False)
    ax = kwargs.get('ax', None)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    
    ax.scatter(x, y, **scatter_kwargs)
    
    if polyfit:
        b, a = np.polyfit(x, y, **polyfit_kwargs)
        # Create x sequence
        xseq = np.linspace(np.min(x), np.max(x), num=100)
        # Plot regression line
        ax.plot(xseq, a + b * xseq, **line_kwargs)
    
    
    # Save or return
    if save:
        plt.savefig(save, bbox_inches='tight')
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
    data_format = kwargs.get('data_format', None)
    se_alpha = kwargs.get('se_alpha', 0.2)
    save = kwargs.get('save', None)
    return_handles = kwargs.get('return_handles', False)
    ax = kwargs.get('axes', None)
    
    # Plotting kwargs
    line_kwargs = kwargs.get('line_kwargs', {})
    fill_kwargs = kwargs.get('fill_kwargs', {})
    axhline_kwargs = kwargs.get('axhline_kwargs', {'y': 0.0, 'color': 'k', 'linestyle': '-'})
    axvline_kwargs = kwargs.get('axvline_kwargs', {'x': 0.0, 'color': 'k', 'linestyle': '-'})
    
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
        ax.fill_between(x, 
                        np.mean(dat, 0) - std_error(dat, 0), 
                        np.mean(dat, 0) + std_error(dat, 0), 
                        alpha=se_alpha, **fill_kwargs)

    # Add legend
    ax.legend()
    
    # Set labels based on data format or custom inputs
    if data_format == 'erp':
        xlabel = kwargs.get('xlabel', 'times')
        ylabel = kwargs.get('ylabel', '$\mu$V')
    elif data_format == 'freq':
        xlabel = kwargs.get('xlabel', 'frequencies')
        ylabel = kwargs.get('ylabel', 'power')
    else:
        xlabel = kwargs.get('xlabel', 'x')
        ylabel = kwargs.get('ylabel', 'y')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set title if provided
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    
    # Save if requested
    if save and created_fig:
        plt.savefig(save, bbox_inches='tight')
    
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
    save = kwargs.get('save', None)
    return_handles = kwargs.get('return_handles', False)
    ax_input = kwargs.get('ax', None)
    
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
        plt.savefig(save, bbox_inches='tight')
    
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
        'F9':    0, 'F7':    1, 'F3':    2, 'Fz':    3, 'F4':    4, 'F8':    5, 'F10':  6, 
                    'FC5':   8, 'FC3':   9, 'FCz':  10, 'FC4':  11, 'FC6':  12,
        'T7':   14,             'C3':   16,             'C4':   18,             'T8':   20, 
                    'CP5':  22, 'CP3':  23,             'CP4':  25, 'CP6':  26, 
        'TP9':  28, 'P7':   29, 'P3':   30, 'Pz':   31, 'P4':   32, 'P8':   33, 'TP10': 34, 
                                            'POz':  38, 
                                'O1':   44, 'Oz':   45, 'O2':   46
    }
    rev_chan_grid = {chan_grid[k]: k for k in chan_grid.keys()}
    
    # Extract topo-specific parameters
    figsize = kwargs.get('figsize', (15, 10))
    save = kwargs.get('save', None)
    return_handles = kwargs.get('return_handles', False)
    se_alpha = kwargs.get('se_alpha', 0.1)  # Default for topo is lower
    
    # Prepare kwargs to pass down to plot_cond (remove topo-specific ones)
    plot_kwargs = kwargs.copy()
    plot_kwargs.pop('figsize', None)
    plot_kwargs.pop('save', None)
    plot_kwargs.pop('return_handles', None)
    plot_kwargs['se_alpha'] = se_alpha  # Use topo default
    
    # Create figure
    fig, axes = plt.subplots(7, 7, 
                          subplot_kw={},
                          gridspec_kw={
                              'height_ratios': [1]*7, 
                              'width_ratios': [1]*7,
                              'hspace': 0.18, 
                              'wspace': 0.1
                          },
                          figsize=figsize)
    
    # Set overall title if provided
    if 'title' in kwargs:
        fig.suptitle(kwargs['title'])
        # Remove title from kwargs passed to subplots
        plot_kwargs_subplot = plot_kwargs.copy()
        plot_kwargs_subplot.pop('title', None)
    else:
        plot_kwargs_subplot = plot_kwargs
    
    # Plot data for each channel
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i in rev_chan_grid.keys():
            # Get channel index
            chan_idx = channels.index(rev_chan_grid[i])
            
            # Use plot_cond for this channel by passing the axis
            plot_kwargs_subplot['axes'] = ax
            plot_kwargs_subplot['title'] = rev_chan_grid[i]
            
            # Extract data for this single channel and call plot_cond
            channel_data = [d[:, chan_idx, :] for d in data]
            plot_cond(channel_data, x, labels, **plot_kwargs_subplot)
            
            # Update title font size
            ax.set_title(rev_chan_grid[i], fontsize=10)
            ax.set_xlim(x[0], x[-1])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels('')
            ax.set_yticklabels('')
            ax.legend().remove()
            ax.axis('on')
        
        # Add legend to a specific subplot
        if i == 42:
            ax.axis('on')
            ax.plot(np.array([0]), np.array([[0]]*len(labels)).T, label=labels)
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
                click_kwargs['title'] = f'{rev_chan_grid[channel_index]}'
                plot_chan_cond(data, x, chan_idx, labels, **click_kwargs)
    
    # Connect click event to the main figure
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Save or return
    if save:
        plt.savefig(save, bbox_inches='tight')
    
    if return_handles:
        return fig, ax
    else:
        plt.show()



# Plot Classifier

def plot_sliding_classifier(data, time, **kwargs):
    
    # Extracting optional parameters with defaults
    average = kwargs.get('average', True)
    save = kwargs.get('save', None)
    chance = kwargs.get('chance', 0.5)
    se_alpha = kwargs.get('se_alpha', 0.2)
    perm_info = kwargs.get('perm_info', None)
    sct_kwargs = kwargs.get('sct_kwargs', dict(s=15, marker='o', color='b'))
    ax = kwargs.get('ax', None)
    return_handles = kwargs.get('return_handles', False)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    
    xtickmarks = np.arange(np.min(time), np.max(time)+0.1, 0.1)
    xticklabels = ['' if int(t*10) % 2 else f'{1000*t:4.0f}' for t in xtickmarks]                    
    
    if average:
        if data.ndim == 2:
            mean_data = np.mean(data, axis=0)
            stderr_data = std_error(data, 0)
        else:
            mean_data = data
            stderr_data = np.zeros(data.shape)
    else:
        mean_data = data.T
        
    
    ax.axhline(chance, color="k", linestyle="--")
    ax.axvline(0.0, color="k", linestyle="-")
    ax.plot(time, mean_data)
    if average:
        ax.fill_between(time, mean_data - stderr_data, mean_data + stderr_data, 
                        alpha=se_alpha)
    ax.set_xticks(xtickmarks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("times (ms)")
    ax.set_ylabel("classification accuracy (%)")  # Area Under the Curve
    if average and perm_info:
        for pi in perm_info['cluster_dict']:
            if pi[perm_info['statistic']+'_pval'] > (1-perm_info['cluster_max_pval']):
                ax.scatter(time[pi['cluster_index'][:,-1]], (chance-0.02)*np.ones(pi['cluster_index'].shape[0]), **sct_kwargs)
                #ax.axvspan(time[pi['cluster_index'][0]], time[pi['cluster_index'][1]], color='red', alpha=0.3, edgecolor=None)
    if save:
        plt.savefig(save, bbox_inches='tight')
    if return_handles:
        return fig, ax
    if ax is not None:
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
    label = kwargs.get('label', None)
    row_break = kwargs.get('row_break', 1)
    vlim = kwargs.get('vlim', None)
    vlim_prctile = kwargs.get('vlim_prctile', (5, 95))
    resolution = kwargs.get('resolution', 256)
    save = kwargs.get('save', None)
    return_handles = kwargs.get('return_handles', False)
    times_label = kwargs.get('times_label', [f'{1000*t:6.0f} ms' for t in time])
    data_scaler = kwargs.get('data_scaler', 1)
    
    # Figure customization parameters
    figsize = kwargs.get('figsize', (3.5, 8))
    title = kwargs.get('title', None)
    cmap = kwargs.get('cmap', None)
    colorbar_label = kwargs.get('colorbar_label', '$\mu$V')
    ylabel_fontsize = kwargs.get('ylabel_fontsize', 11)
    ylabel_rotation = kwargs.get('ylabel_rotation', 90)
    dpi = kwargs.get('dpi', None)
    title_kwargs = kwargs.get('title_kwargs', None)
    
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
        ncols = (len(time_idx) // (row_break))
        row_mulitplier = 0
    
    # Compute global color limits based on all specified time points
    if vlim: 
        vlims = np.atleast_1d(vlim)
    else:
        vlims = np.percentile(
            np.array([data_scaler*p[:,:,time_idx].flatten() for p in data]).flatten(), 
            vlim_prctile
        )
    
    # Create figure with custom gridspec
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # If a title is provided, add it
    if title:
        fig.suptitle(title)
    
    # Create gridspec with minimal spacing
    gs = fig.add_gridspec(
        nrows=nrows, 
        ncols=ncols+1,  # +1 for colorbar
        width_ratios=[1]*ncols + [0.05],
        height_ratios=[1]*nrows,
        hspace=0.02,
        wspace=0.1
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
                'res': resolution, 
                'vlim': vlims, 
                'axes': current_ax, 
                'show': False
            }
            if cmap:
                plot_kwargs['cmap'] = cmap
            
            top = mne.viz.plot_topomap(
                np.array([data_scaler*p[:,row_mulitplier*row,time_idx[row*(ncols)+col]] for p in data]).mean(0),
                epoch.info, **plot_kwargs
            )
            row_topos.append(top)
            
            # Add time label to first plot in row
            if label:
                if col == 0:
                    current_ax.set_ylabel(label[row], 
                                          fontsize=ylabel_fontsize, 
                                          rotation=ylabel_rotation)
                
                # Add time point label to first plot in column
                if row == 0:
                    current_ax.set_title(times_label[row*(ncols)+col], **title_kwargs)
            else:
                current_ax.set_title(times_label[row*(ncols)+col], **title_kwargs)
        
        ax.append(row_axes)
        topomap_objects.append(row_topos)
    
    # Add colorbar to last column
    middle_row = nrows // 2
    cbar_ax = fig.add_subplot(gs[middle_row, -1])
    cbar = fig.colorbar(topomap_objects[middle_row][0][0], cax=cbar_ax)
    cbar.set_label(colorbar_label, labelpad=1)
    
    # Save figure if path is provided
    if save:
        plt.savefig(save, bbox_inches='tight')
    
    # Return figure handles if requested
    if return_handles:
        return fig, ax, cbar



# Plot Data Decoding 

def plot_data_decoding(data, srate, segment, **kwargs):
    
    # Extracting optional parameters with defaults
    plot_max_val = kwargs.get('plot_max_val', True)
    save = kwargs.get('save', None)
    return_handles = kwargs.get('return_handles', True)
    
    time_vec = np.arange(0, data.shape[1]/srate, 1/srate)
    time_seg = np.arange(segment[0], segment[1], 1/srate)
    max_values = np.argmax(data[:,int(segment[0]*srate):int(segment[1]*srate)],0)
    fig, ax = plt.subplots(nrows = 2, ncols = 1)
    ax[0].plot(time_vec, data[:,:].T)#, label=labels)
    ax[0].axvspan(time_seg[0], time_seg[-1], 0.01, 0.98, fill=True, facecolor = 'b',edgecolor=None, alpha=0.3)
    ax[0].axvspan(time_seg[0], time_seg[-1], 0.01, 0.98, fill=False, edgecolor='k', linewidth=1.6, linestyle='--', zorder=10)
    ax[1].plot(time_seg, data[:,int(segment[0]*srate):int(segment[1]*srate)].T)
    if plot_max_val:
        for i in range(data.shape[0]):
            ax[1].scatter(time_seg[max_values==i], max_values[max_values==i]/30+1.1, s=2, marker='.')
    ax[0].legend()
    if save:
        plt.savefig(save, bbox_inches='tight')
    if return_handles:
        return fig, ax


# Plot Sequenceness Results

def plot_sequenceness(data, lags, perms, **kwargs):
    
    # Extracting optional parameters with defaults
    save = kwargs.get('save', None)
    se_alpha = kwargs.get('se_alpha', 0.2)
    plot_subs = kwargs.get('plot_subs', False)
    pool_permutations = kwargs.get('pool_permutations', False)
    plot_dist_perm = kwargs.get('plot_dist_perm', None)
    plot_max_perm = kwargs.get('plot_max_perm', None)
    return_handles = kwargs.get('return_handles', False)
    return_cbar_handle = kwargs.get('return_cbar_handle', False)
    ax = kwargs.get('ax', None)                
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4))
    
    if pool_permutations:
        perm_dist = np.reshape(perms, (perms.shape[0]*perms.shape[1], perms.shape[-1]))
    else:
        perm_dist = np.mean(perms, 0)
    
    if isinstance(plot_dist_perm, str):
        if plot_dist_perm == 'lines':
            ax.plot(lags, perm_dist.T, color='k', alpha=0.1)
        elif plot_dist_perm == 'area':
            ax.fill_between(lags, np.min(perm_dist, 0), np.max(perm_dist, 0), color='k', alpha=0.1, label = 'Permutation Distribution')
    
    if plot_max_perm:
        perm_dist_abs = np.max(np.abs(perm_dist),1).flatten()
        if isinstance(plot_max_perm, str):
            ax.axhline(np.max(perm_dist_abs), color="k", linestyle="--", label = 'Max Permutation')
            ax.axhline(-np.max(perm_dist_abs), color="k", linestyle="--")
        if isinstance(plot_dist_perm, float):
            ax.axhline(np.percentile(perm_dist_abs, plot_dist_perm), color="k", linestyle="--", label = f'{plot_dist_perm}% Permutation')
            ax.axhline(-np.percentile(perm_dist_abs, plot_dist_perm), color="k", linestyle="--")
    
    ax.axhline(0.0, color="k", linestyle="-")
    if plot_subs:
        norm = Normalize(vmin=np.min(plot_subs['color_values']), 
                         vmax=np.max(plot_subs['color_values']))
        sm = plt.cm.ScalarMappable(cmap=plot_subs['cmap'], norm=norm)
        sm.set_array([])  # This is necessary for the colorbar to work
        for p in range(data.shape[0]):
            color = plt.get_cmap(plot_subs['cmap'])(norm(plot_subs['color_values'][p]))
            ax.plot(lags, data[p,:], color=color)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(plot_subs['cbar_title'], fontsize=12)
    else:
        ax.plot(lags, data.mean(0), label='Evidence for Replay')
        if data.shape[0] > 1:
            ax.fill_between(lags, np.mean(data,0) - std_error(data, 0), np.mean(data,0) + std_error(data, 0), alpha=se_alpha)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim((0,np.max(lags)))
    ax.set_ylabel("sequenceness")
    ax.set_xlabel("time lag (ms)")
    ax.legend()
    if save:
        plt.savefig(save, bbox_inches='tight')
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




def add_signif_timepts(ax, perm_info, plot_type='area', **kwargs):
    """
    Add significance indicators to an existing plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to add significance indicators to.
    perm_info : dict
        Dictionary containing significance information with keys:
        - 'stat_data': array of statistic values (t, t^2, F values, etc.)
        - 'dims': string indicating data dimensions ('times', 'freq_times', 'chan_times', 'x_y_z')
        - 'times': array of time points
        - 'threshold_pval': threshold for creating clusters/significant time points
        - 'method': statistic method (e.g., "tmax", "F", "cluster_area", "cluster_tsum", "cluster_depth")
        - 'permtest_pval': threshold for significance against surrogate distribution
        - 'corrected_pvals': array same size as stat_data with corrected p-values (can contain NaN)
        - 'mask': array same size as stat_data, 0 where no significance, >0 integers for clusters
    plot_type : str, default='area'
        Type of significance visualization:
        - 'area': Background patches with alpha transparency
        - 'lines': Horizontal lines at specified position
        - 'linepoints': Points in horizontal line at specified position
        - 'points': Points along the data lines
        - 'bar': Vertical bars/patches at specified position
        
    Keyword Arguments:
    -----------------
    label : str, optional
        Label to match with existing plot lines for color consistency
    y_pos : float, optional
        Y-position for 'lines', 'linepoints', and 'bar' plot types
    alpha : float, default=0.3
        Transparency level for patches and fills
    height : float, optional
        Height for 'lines' and 'bar' plot types (default: 5% of y-axis range)
    use_pval_alpha : bool, default=False
        Whether to use p-values to modulate alpha in 'linepoints'
    use_pval_colormap : bool, default=False
        Whether to use p-values to color-code 'bar' plots
    colormap : str, default='viridis_r'
        Colormap for p-value coding (reversed so lower p-vals are brighter)
    color : str or tuple, optional
        Custom color (overrides label-based color matching)
    
    Plot-specific kwargs:
    --------------------
    patch_kwargs : dict, optional
        Additional kwargs for Rectangle patches (area, bar plot types)
    line_kwargs : dict, optional
        Additional kwargs for line plotting (lines plot type)
    scatter_kwargs : dict, optional
        Additional kwargs for scatter plots (linepoints, points plot types)
    
    Returns:
    --------
    artists : list
        List of matplotlib artists added to the plot
    """
    
    # Extract parameters
    label = kwargs.get('label', None)
    y_pos = kwargs.get('y_pos', None)
    alpha = kwargs.get('alpha', 0.3 if plot_type != 'linepoints' else 1.0)
    height = kwargs.get('height', None)
    use_pval_alpha = kwargs.get('use_pval_alpha', False)
    use_pval_colormap = kwargs.get('use_pval_colormap', False)
    colormap = kwargs.get('colormap', 'viridis_r')
    custom_color = kwargs.get('color', None)
    
    # Extract plot-specific kwargs
    patch_kwargs = kwargs.get('patch_kwargs', {})
    line_kwargs = kwargs.get('line_kwargs', {})
    scatter_kwargs = kwargs.get('scatter_kwargs', {})
    
    # Get times and significance info
    times = perm_info['times']
    #stat_data = perm_info['stat_data']
    #dims = perm_info['dims']
    #threshold_pval = perm_info['threshold_pval']
    #method = perm_info['method']
    permtest_pval = perm_info['permtest_pval']
    corrected_pvals = perm_info['corrected_pvals']
    mask = perm_info['mask']
    
    
    # Determine color based on label matching or custom color
    if custom_color is not None:
        color = custom_color
    elif label is not None:
        # Try to match color with existing line
        color = None
        for line in ax.get_lines():
            if line.get_label() == label:
                color = line.get_color()
                break
        if color is None:
            color = 'red'  # fallback color
    else:
        color = 'red'  # default color
    
    # Get axis limits for positioning
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Set default height if not provided
    if height is None:
        height = y_range * 0.05
    
    # Set default y_pos if not provided
    if y_pos is None:
        if plot_type in ['lines', 'linepoints', 'bar']:
            y_pos = y_min - y_range * 0.1
    
    artists = []
    
    # Create boolean mask for significant time points
    sig_mask = (mask > 0) & (corrected_pvals < permtest_pval)
    
    if not np.any(sig_mask):
        return artists  # No significant time points
    
    # Get time step for patch width calculation
    if len(times) > 1:
        dt = times[1] - times[0]
    else:
        dt = 1.0
    
    # Plot based on selected type
    if plot_type == 'area':
        # Create background patches for significant regions
        # Find continuous regions of significance
        diff_mask = np.diff(np.concatenate(([False], sig_mask, [False])).astype(int))
        starts = np.where(diff_mask == 1)[0]
        ends = np.where(diff_mask == -1)[0]
        
        patch_kwargs_merged = {'alpha': alpha, 'color': color, **patch_kwargs}
        
        for start, end in zip(starts, ends):
            x_start = times[start]
            width = times[end-1] - times[start] + dt
            rect = Rectangle((x_start, y_min), width, y_range, **patch_kwargs_merged)
            ax.add_patch(rect)
            artists.append(rect)
    
    elif plot_type == 'lines':
        # Create horizontal lines for significant regions
        sig_times = times[sig_mask]
        if len(sig_times) > 0:
            # Group consecutive time points into line segments
            diff_times = np.diff(sig_times)
            median_dt = np.median(np.diff(times))
            breaks = np.where(diff_times > 2 * median_dt)[0] + 1
            
            segments = []
            start_idx = 0
            for break_idx in breaks:
                segments.append([sig_times[start_idx], sig_times[break_idx-1]])
                start_idx = break_idx
            segments.append([sig_times[start_idx], sig_times[-1]])
            
            line_kwargs_merged = {'color': color, 'alpha': alpha, 'linewidth': height*20, **line_kwargs}
            
            for seg_start, seg_end in segments:
                line = ax.plot([seg_start, seg_end], [y_pos, y_pos], **line_kwargs_merged)[0]
                artists.append(line)
    
    elif plot_type == 'linepoints':
        # Plot points in horizontal line
        sig_times = times[sig_mask]
        sig_pvals = corrected_pvals[sig_mask]
        
        if len(sig_times) > 0:
            scatter_kwargs_merged = {'color': color, 's': 20, **scatter_kwargs}
            
            if use_pval_alpha and not np.all(np.isnan(sig_pvals)):
                # Use p-values to modulate alpha
                valid_pval_mask = ~np.isnan(sig_pvals)
                if np.any(valid_pval_mask):
                    # Normalize p-values to alpha range [0.3, 1.0]
                    norm_pvals = 1.0 - (sig_pvals[valid_pval_mask] / np.max(sig_pvals[valid_pval_mask]))
                    alphas = 0.3 + 0.7 * norm_pvals
                    
                    # Plot each point with individual alpha
                    for i, (t, a) in enumerate(zip(sig_times[valid_pval_mask], alphas)):
                        scatter_kwargs_point = {**scatter_kwargs_merged, 'alpha': a}
                        point = ax.scatter(t, y_pos, **scatter_kwargs_point)
                        artists.append(point)
                        
                    # Plot NaN p-value points with default alpha
                    nan_mask = np.isnan(sig_pvals)
                    if np.any(nan_mask):
                        scatter_kwargs_merged['alpha'] = alpha
                        points = ax.scatter(sig_times[nan_mask], [y_pos]*np.sum(nan_mask), **scatter_kwargs_merged)
                        artists.append(points)
            else:
                # Uniform alpha
                scatter_kwargs_merged['alpha'] = alpha
                points = ax.scatter(sig_times, [y_pos]*len(sig_times), **scatter_kwargs_merged)
                artists.append(points)
    
    elif plot_type == 'points':
        # Plot points along the data lines
        sig_times = times[sig_mask]
        
        if len(sig_times) > 0 and label is not None:
            # Find the matching line to get y-values
            target_line = None
            for line in ax.get_lines():
                if line.get_label() == label:
                    target_line = line
                    break
            
            if target_line is not None:
                line_x = target_line.get_xdata()
                line_y = target_line.get_ydata()
                
                # Interpolate y-values at significant time points
                sig_y = np.interp(sig_times, line_x, line_y)
                
                scatter_kwargs_merged = {'color': color, 's': 30, 'alpha': alpha, 'zorder': 10, **scatter_kwargs}
                points = ax.scatter(sig_times, sig_y, **scatter_kwargs_merged)
                artists.append(points)
    
    elif plot_type == 'bar':
        # Create vertical bars/patches
        sig_times = times[sig_mask]
        sig_pvals = corrected_pvals[sig_mask]
        
        if len(sig_times) > 0:
            if use_pval_colormap and not np.all(np.isnan(sig_pvals)):
                # Use colormap based on p-values
                cmap = cm.get_cmap(colormap)
                valid_pval_mask = ~np.isnan(sig_pvals)
                
                if np.any(valid_pval_mask):
                    # Normalize p-values for colormap
                    valid_pvals = sig_pvals[valid_pval_mask]
                    norm_pvals = (valid_pvals - np.min(valid_pvals)) / (np.max(valid_pvals) - np.min(valid_pvals))
                    colors = cmap(norm_pvals)
                    
                    patch_kwargs_merged = {'alpha': alpha, **patch_kwargs}
                    
                    # Plot bars with individual colors
                    for i, (t, c) in enumerate(zip(sig_times[valid_pval_mask], colors)):
                        rect = Rectangle((t - dt/2, y_pos), dt, height, 
                                       color=c, **patch_kwargs_merged)
                        ax.add_patch(rect)
                        artists.append(rect)
                    
                    # Plot NaN p-value bars with default color
                    nan_mask = np.isnan(sig_pvals)
                    if np.any(nan_mask):
                        patch_kwargs_merged['color'] = color
                        for t in sig_times[nan_mask]:
                            rect = Rectangle((t - dt/2, y_pos), dt, height, **patch_kwargs_merged)
                            ax.add_patch(rect)
                            artists.append(rect)
            else:
                # Uniform color bars
                patch_kwargs_merged = {'alpha': alpha, 'color': color, **patch_kwargs}
                for t in sig_times:
                    rect = Rectangle((t - dt/2, y_pos), dt, height, **patch_kwargs_merged)
                    ax.add_patch(rect)
                    artists.append(rect)
    
    return artists