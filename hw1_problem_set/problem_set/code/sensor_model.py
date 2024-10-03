'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
import os

from map_reader import MapReader
# from numba import njit, prange
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map, seed):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        # For Reproducability
        np.random.seed(seed)


        self._z_hit = 11
        self._z_short = 0
        self._z_max = 1
        self._z_rand =800
        self._sigma_hit = 3
        self._lambda_short = 0.01
        self._max_range = 800
        
    
        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 1
        
        # Map meta-data
        self.occupancy_map  = occupancy_map
        self.map_width, self.map_height = occupancy_map.shape
        # Used in p_max and p_rand, optionally in ray casting
        self.map_resolution = 10
    
        # Laser offset(as per ReadME)
        self._laser_offset = 25

        
        self._path_sensor_map = 'particle_rays.npy' 
        # Load sensor-map
        if os.path.exists(self._path_sensor_map): 
            self.sensor_map = np.load(self._path_sensor_map)
        else:
            print("[WARN] Couldn't find sensor map. Trying to generate")
            self.create_sensor_map()
            self.sensor_map = np.load(self._path_sensor_map)
        print("[INFO] Sensor map loaded successfully")

# Function to trace rays and create hash-map
    def create_sensor_map(self):
        # For now collecting just free space
        freespace_rows, freespace_cols = np.where(self.occupancy_map == 0.0)

        num_pts_ = freespace_cols.shape[0]

        # Store angle arrs
        angles_  = np.linspace(0, 360, 360, endpoint=False)        
        cos_angles_ = np.cos(np.deg2rad(angles_)).reshape(-1,1)
        sin_angles_ = np.sin(np.deg2rad(angles_)).reshape(-1,1)
        
        # Store ray_len arr
        ray_len_ = np.linspace(0, self._max_range, int(self._max_range + 1), endpoint=True).reshape(-1,1)
        
        # Final particle arr; shape: [x,y,360]
        particle_rays_arr_ = np.full((self.map_width, self.map_height, angles_.shape[0]), 0)
        
        # Occupancy map mask: 0 for free space, 1 for obstacles, -1 for out of bounds
        map_mask_ = (self.occupancy_map >= self._min_probability).astype(int) 
        map_mask_[self.occupancy_map == -1] = 1      

        # Iterate for each particle
        for idx in tqdm(range(0, num_pts_)):
            # x is col. 
            free_x = freespace_cols[idx]
            free_y = freespace_rows[idx]

            # Ray endpoints
            map_x_idxs_ = (free_x +  ray_len_ * cos_angles_.T ).astype(int)  
            map_y_idxs_ = (free_y +  ray_len_ * sin_angles_.T ).astype(int)  

            # Mask for valid indices (within bounds)
            valid_mask = (
                            (map_x_idxs_ >= 0) & (map_x_idxs_ < self.map_width ) &
                            (map_y_idxs_ >= 0) & (map_y_idxs_ < self.map_height)
                         )

             # Clip to prevent out-of-bounds 
            map_x_idxs_clipped = np.clip(map_x_idxs_, 0, self.map_width  - 1)
            map_y_idxs_clipped = np.clip(map_y_idxs_, 0, self.map_height - 1)
            
            # Out-of bound are obstacles 
            occupancy_masked = np.where(valid_mask, map_mask_[map_y_idxs_clipped, map_x_idxs_clipped], 1)
            
            rays_ = np.full(angles_.shape[0], self._max_range, dtype=np.float64)

            for ang_idx in range(angles_.shape[0]):
                # Find the index of the first hit for each angle
                hit_idx = np.argmax(occupancy_masked[:, ang_idx])  # Returns the first occurrence of 1 (obstacle)

                # If there's a hit, assign ray length
                if occupancy_masked[hit_idx, ang_idx]:
                    rays_[ang_idx] = hit_idx
                
            particle_rays_arr_[free_x, free_y, :] = rays_

        # Save the numpy array
        np.save(self._path_sensor_map, particle_rays_arr_)
        print(f"Sensor map saved in file:{self._path_sensor_map}")


# centre2laser transform function  
    def centre2laser_transform_vec(self, x_t):

        # Compute the new x and y positions using vectorized operations
        x_t1_off_x = x_t[:, 0] + self._laser_offset * np.cos(x_t[:, 2])
        x_t1_off_y = x_t[:, 1] + self._laser_offset * np.sin(x_t[:, 2])

        # Combine the new positions with the original orientations (theta)
        x_t1_off = np.column_stack((x_t1_off_x, x_t1_off_y, x_t[:, 2]))
        
        return x_t1_off

    
# Get expected measurements for a particle (all-angles; used for visualization)
    def zt_k_star_particle(self, x_t1):
        # Compute the new x and y positions using vectorized operations
        x_t1_off_x = x_t1[0] + self._laser_offset * np.cos(x_t1[2])
        x_t1_off_y = x_t1[1] + self._laser_offset * np.sin(x_t1[2])

        # Combine the new positions with the original orientations (theta)
        x_t1_off = np.array([x_t1_off_x, x_t1_off_y, x_t1[2]])
        
        # Tranform into map-frame & search the array
        map_x, map_y = np.floor_divide([x_t1_off[0], x_t1_off[1]], self.map_resolution).astype(int)
        
        angles_ = np.rad2deg(x_t1_off[2])
        laser_min, laser_max = angles_ - 90, angles_ + 90
        
        laser_min = np.mod(laser_min + 360, 360).astype(int)
        laser_max = np.mod(laser_max + 360, 360).astype(int)

        z_tk_star = np.zeros((1, 180), dtype=np.float64)

        if laser_min < laser_max:
                z_tk_star = self.sensor_map[map_x, map_y, laser_min:laser_max]

        else:
            # Slice in parts
            z_tk_star = np.concatenate((
                                                self.sensor_map[map_x, map_y, laser_min:],  
                                                self.sensor_map[map_x, map_y, :laser_max]   
                                        ))
        
        return z_tk_star
    



# Get expected measurements for particles
    def zt_k_star_vec(self, x_t1_off):        

        # Convert to map idxs
        map_x, map_y = np.floor_divide([x_t1_off[:, 0], x_t1_off[:, 1]], self.map_resolution).astype(int)

        angles_ = np.rad2deg(x_t1_off[:, 2])
        laser_min, laser_max = angles_ - 90, angles_ + 90
        
        laser_min = np.mod(laser_min + 360, 360).astype(int)
        laser_max = np.mod(laser_max + 360, 360).astype(int)

        z_tk_star = np.zeros((x_t1_off.shape[0], 180), dtype=np.float64)

        for i, (min_angle, max_angle) in enumerate(zip(laser_min, laser_max)):
            if min_angle < max_angle:
                z_tk_star[i, :] = self.sensor_map[map_x[i], map_y[i], min_angle:max_angle]
            else:
                # Slice in parts
                z_tk_star[i, :] = np.concatenate((
                                                    self.sensor_map[map_x[i], map_y[i], min_angle:],  
                                                    self.sensor_map[map_x[i], map_y[i], :max_angle]   
                                                ))
        return z_tk_star


# Beam Range Finder Model
    def beam_range_finder_model_vectorized(self, z_t1_k, x_t1_particles):
        """
        param[in] z_t1_k : laser range readings [array of 180 values] at time t
        param[in] x_t1_particles : array of particle states, shape [num_particles, 3] where each row is [x, y, theta] at time t [world_frame]
        param[out] prob_zt1_arr : array of likelihoods for each particle, shape [num_particles]
        """
        num_particles = x_t1_particles.shape[0]
        
        # Centre to Laser Transform 
        x_t1_off = self.centre2laser_transform_vec(x_t1_particles)
        # Get Expected Laser Measurements
        z_t1_k_star = self.zt_k_star_vec(x_t1_off)
        # z_t1_k_star = np.clip(z_t1_k_star, 0, self._max_range)
        
        # Clip & Tranform in map-frame 
        z_t1_k = np.divide(z_t1_k, self.map_resolution) 
        z_t1_k = np.clip(z_t1_k, 0, self._max_range)

        # Sub-sample
        z_t1_k = z_t1_k[::self._subsampling]
        z_t1_k_star = z_t1_k_star[::self._subsampling] 


        ##### Beam Range Finder Model
        p_hit_   = np.zeros((num_particles, z_t1_k.shape[0]), dtype=np.float64)
        p_short_ = np.zeros((num_particles, z_t1_k.shape[0]), dtype=np.float64)
        p_rand_  = np.zeros((num_particles, z_t1_k.shape[0]), dtype=np.float64)
        p_max_   = np.zeros((num_particles, z_t1_k.shape[0]), dtype=np.float64)

        # p_hit condition
        hit_idxs = (z_t1_k <= self._max_range)
        p_hit_[:, hit_idxs] = np.exp( -0.5 * np.square( (z_t1_k[hit_idxs] - z_t1_k_star[:, hit_idxs]) / self._sigma_hit) )
        p_hit_[:, hit_idxs] /= self._sigma_hit * np.sqrt(2*np.pi)  
        
        # p_rand condition
        p_rand_[:, hit_idxs] = 1 / self._max_range 
        
        # p_max condition
        hit_idxs = (z_t1_k >= self._max_range)
        p_max_[:, hit_idxs] = 1

        # p_short condition
        p_short_ = self._lambda_short * np.exp(-self._lambda_short * z_t1_k)
        p_short_ = np.where(z_t1_k > z_t1_k_star, 0, p_short_)

        # Full-model probability 
        p_hit = self._z_hit * p_hit_
        p_short = self._z_short * p_short_
        p_rand = self._z_rand * p_rand_
        p_max = self._z_max * p_max_
        
        p = p_hit + p_short + p_rand + p_max
        q = np.sum(np.log(p), axis=1)
        # q = np.exp(q)
        
        # plt.figure(2)
        # plt.cla()
        # plt.plot(np.sum(p_hit, axis=1), label="p_hit")
        # plt.plot(np.sum(p_short, axis=1), label="p_short")
        # plt.plot(np.sum(p_rand, axis=1), label="p_rand")
        # plt.plot(np.sum(p_max, axis=1), label="p_max")
        # plt.plot(np.sum(p, axis=1), label="p")
        # plt.legend()
        
        
        return q

