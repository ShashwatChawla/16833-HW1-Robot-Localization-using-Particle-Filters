'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        # New Array of resampled wts
        X_bar_resampled =  np.zeros_like(X_bar)
        
        # Varibles for sampler
        num_samples = X_bar.shape[0]
        step        = 1/num_samples
        u_rand      = np.random.uniform(0, step)
        
        if np.sum(X_bar[:, 3]) == 0:
            X_bar_resampled = X_bar
            print(f"[WARN]Returning X_bar as it is")
            return X_bar_resampled
        # Normalized Wt. arr
        normalized_wts = X_bar[:, 3]/np.sum(X_bar[:, 3])
        
        # Cumalitive Wt. arr 
        cumulative_wts = np.cumsum(normalized_wts)
        
        i = 0
        # Sample Particles
        for j in range(0, num_samples):
            resampling_wt = u_rand + j*step
            for c_idx, c_wt in enumerate(cumulative_wts):
                if resampling_wt <= c_wt:
                    X_bar_resampled[i, :3] = X_bar[c_idx, :3]
                    break
                
                elif resampling_wt > c_wt and resampling_wt < cumulative_wts[c_idx+1]:
                    X_bar_resampled[i, :3] = X_bar[c_idx+1, :3]
                    break
            
            # print(f"Particle '{i}' replaced with '{c_idx}'")
            i += 1
        

        # For Testing Purposes
        # X_bar_resampled =  np.zeros_like(X_bar)
        # X_bar_resampled = X_bar 
        
        return X_bar_resampled
