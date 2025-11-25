# -*- coding: utf-8 -*-
"""
Utility functions.

@author: Christopher Postzich
@github: mcpost
"""

import numpy as np
from tqdm import trange
import scipy.stats as scistats
from skfda.inference.hotelling import hotelling_t2
from skimage import measure


# Channel Grid
chan_grid = {
    'F9':    0, 'F7':    1, 'F3':    2, 'Fz':    3, 'F4':    4, 'F8':    5, 'F10':  6, 
                'FC5':   8, 'FC3':   9, 'FCz':  10, 'FC4':  11, 'FC6':  12,
    'T7':   14,             'C3':   16,             'C4':   18,             'T8':   20, 
                'CP5':  22, 'CP3':  23,             'CP4':  25, 'CP6':  26, 
    'TP9':  28, 'P7':   29, 'P3':   30, 'Pz':   31, 'P4':   32, 'P8':   33, 'TP10': 34, 
                                        'POz':  38, 
                            'O1':   44, 'Oz':   45, 'O2':   46
    }




# Calculate Standard Error
def std_error(data, axis=0, ddof=1):
    """Helper Function to compute Standard Error of the Mean."""
    return np.std(data, axis=axis, ddof=ddof)/np.sqrt(data.shape[axis])


def sign_flip_permtest(data, n_permutations=1000, clust_thresh_pval=0.05, **kwargs):
    """Sign-flip Permutation Test."""
    # Extracting optional parameters with defaults
    chance_lev = kwargs.get('chance_lev', 0.0)
    add_info = kwargs.get('add_info', False)
    
    df = data.shape[0] - 1
    
    if data.ndim == 3:
        dims = [1,1]
        surrogate_tmap = np.zeros((n_permutations, data.shape[-2], data.shape[-1]))
        real_tmap = scistats.ttest_1samp(data - chance_lev, 0)[0]
    else:
        dims = [1]
        surrogate_tmap = np.zeros((n_permutations, data.shape[-1]))
        real_tmap = scistats.ttest_1samp(data - chance_lev, 0)[0][np.newaxis,:]
    
    for perm in trange(n_permutations, desc = 'Sign Flip Permutations'):
        sign_flip = np.sign(2*np.random.rand(data.shape[0],*dims)-1)
        surrogate_tmap[perm,:],_ = scistats.ttest_1samp(sign_flip*(data - chance_lev), 0)


    # Initialize data arrays
    surrogate_clust_info = np.zeros((n_permutations,2))
    for perm in trange(n_permutations, desc = 'Find Surrogage Clusters: '):
        if data.ndim == 3:
            temp = surrogate_tmap[perm]
        else:
            temp = surrogate_tmap[perm][np.newaxis,:]
        # Find positive surrogate clusters
        clust_map_pos = temp >= scistats.t.ppf(1-clust_thresh_pval, df)
        labels = measure.label(clust_map_pos)
        props = measure.regionprops(labels)
        
        if labels.sum() > 0:
            surrogate_clust_info[perm,0] = np.argmax([cl.area for cl in props])
            surrogate_clust_info[perm,1] = np.max([temp[cl.coords[:,0], cl.coords[:,1]].sum() for cl in props])
            

    clust_map_pos = real_tmap >= scistats.t.ppf(1-clust_thresh_pval, df)
    labels = measure.label(clust_map_pos)
    props = measure.regionprops(labels)
    real_clust_info = [[]]*len(props)
    for icl,cl in enumerate(props):
        real_clust_info[icl] = dict(cluster_area = int(cl.area),
                                    cluster_area_pval = np.mean(cl.area > surrogate_clust_info[:,0]),
                                    cluster_tsum = np.sum(real_tmap[cl.coords[:,0], cl.coords[:,1]].flatten()),
                                    cluster_tsum_pval = np.mean(np.sum(real_tmap[cl.coords[:,0], cl.coords[:,1]].flatten()) > surrogate_clust_info[:,1]),
                                    cluster_index = cl.coords
                                    )    
    
    if add_info:
        return real_clust_info, surrogate_clust_info, real_tmap, surrogate_tmap
    else:
        return real_clust_info



def hotelling_t2_test(fd1, fd2):
    """Compute Hotelling's T² statistic and p-value for two functional data samples."""
    stat = hotelling_t2(fd1, fd2)
    
    n1 = fd1.n_samples
    n2 = fd2.n_samples
    p = fd1.grid_points[0].shape[0]  # number of time points
    
    # Convert T² to F-statistic
    # F = [(n1 + n2 - p - 1) / (p * (n1 + n2 - 2))] * T²
    f_stat = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * stat
    
    # Degrees of freedom
    df = (p, n1 + n2 - p - 1)
    
    # Compute p-value
    p_value = 1 - scistats.f.cdf(f_stat, df[0], df[1])
    
    return stat, f_stat, p_value, df