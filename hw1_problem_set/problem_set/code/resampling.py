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
    def __init__(self, num_particles):
        """
        TODO : Initialize resampling process parameters here
        """
        self.num_particles = num_particles
        self.step = 1/self.num_particles

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
        u_rand      = np.random.uniform(0, self.step)
        idx = -1

        if np.sum(X_bar[:, 3]) == 0:
            print(f"[WARN]Returning X_bar | Sum is zero")
            return X_bar
        

        # Normalised X_bar
        X_bar[:, 3] = X_bar[:, 3]/np.sum(X_bar[:, 3])
        
        # # Cumalitive Wt. arr 
        # cumulative_wts = np.cumsum(normalized_wts)
        
        cum_wt = X_bar[0, 3]
        for p_ in range(0, self.num_particles):
            u_wt = u_rand + (p_)*self.step
            while u_wt > cum_wt and idx+1 < self.num_particles: 
                idx += 1
                # Move to them cumalitive wt.
                cum_wt += X_bar[idx, 3]
            X_bar_resampled[p_, :] = X_bar[idx, :]
            # Equal weight to all
            X_bar_resampled[p_, 3] = 1/self.num_particles
        return X_bar_resampled
