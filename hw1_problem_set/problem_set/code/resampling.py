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
    def __init__(self, num_particles, occupancy_map, kidnap_test=False):
        """
        TODO : Initialize resampling process parameters here
        """
        self.num_particles = num_particles
        self.step = 1/self.num_particles
        self.occupancy_map = occupancy_map
        self._robot_kidnap = kidnap_test
        if self._robot_kidnap:
            # Detect kidnap if Mean/Max wt ratio below this threshold 
            self._kidnap_threshold = 0.2    
            
            print("[INFO] Kidnap Detection Enabled")
        else:
            print("[INFO] Kidnap Detection Disabled")
        

    def init_particles_freespace(self, num_particles, occupancy_map):

        # initialize [x, y, theta] positions in world_frame for all particles
        """
        TODO : Add your code here
        This version converges faster than init_particles_random
        """
        X_bar_init = np.zeros((num_particles, 4))

        freespace_rows, freespace_cols = np.where(occupancy_map == 0)

        # Rows are y-axis
        valid_pts = list(zip(freespace_cols, freespace_rows))
        
        if len(valid_pts) >= num_particles:
            rand_pts = np.random.choice(len(valid_pts), num_particles, replace=False)
            selected_pts = np.array([valid_pts[i] for i in rand_pts])
        else:
            print(f"[ERROR] Not enough points to sample from free-space")
            exit()
        
        # Values are in map-frame
        selected_pts *= 10
        # Randomly sample theta-values
        theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))
        # initialize weights for all particles
        w0_vals = np.ones((num_particles, 1), dtype=np.float64)
        w0_vals = w0_vals / num_particles

        X_bar_init = np.hstack((selected_pts, theta0_vals, w0_vals))
        
        return X_bar_init

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
        Algorithm Adapted from:
            Probabilistic Robotic S.Thrun
            Pg:110 | Algorithm Low_Variance Sampler
        """
        X_bar_resampled = np.zeros_like(X_bar)
        num_particles = X_bar.shape[0]  # Number of particles (rows in X)
    
        wt = X_bar[:, -1]

        # If kidnap detection is enabled
        if self._robot_kidnap:
            # Average over all particles:
            max_wt = np.max(wt)
            mean_wt = np.average(wt)
            mean_max_r = mean_wt/max_wt

            if mean_max_r < self._kidnap_threshold:
                print("[WARN] Robot Kidnap Detected !!!!!!!!!!!!!!!!!")
                print("[INF0] Resampling Particles in Free-Space")
                # Initialise Particles in free-space
                self.init_particles_freespace(self.num_particles, self.occupancy_map)


        wt = wt / np.sum(wt)
        r = np.random.uniform(0, 1/num_particles)
    
        c = wt[0]
        i = 0

        # Loop over all particles
        for m in range(num_particles):
            u = r + (m / num_particles)  # Resampling threshold

            # Move through the particles until we find the one corresponding to u
            while u > c:
                i += 1
                c += wt[i]

            # Add the selected particle (excluding the weight) to the resampled set
            X_bar_resampled[m] = X_bar[i]
        
        # Reset weights
        X_bar_resampled[:, -1] = X_bar[:, -1]

        return X_bar_resampled

