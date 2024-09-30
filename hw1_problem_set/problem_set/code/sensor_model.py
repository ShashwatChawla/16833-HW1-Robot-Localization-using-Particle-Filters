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

from map_reader import MapReader
import pickle
# from numba import njit, prange
# import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        # Original Values
        # self._z_hit = 10
        # self._z_short = 0.1
        # self._z_max = 0.1
        # self._z_rand = 10

        # These are for log
        # self._z_hit = 31      
        # self._z_short = 1.75
        # self._z_max = 1.0
        # self._z_rand = 80

        self._z_hit = 35      
        self._z_short = 1.75
        self._z_max = 1.0
        self._z_rand = 80



        # self._z_hit = 22      
        # self._z_short = 1.75
        # self._z_max = 3.0
        # self._z_rand = 10


        # self._z_hit = 31      
        # self._z_short = 1.75
        # self._z_max = 2.0
        # self._z_rand = 8


        # Keeping it small by assuming its good lidar.
        self._sigma_hit = 1.0   
        self._lambda_short = 0.01

        
    
        # Used in p_max and p_rand, optionally in ray casting
        self.map_resolution = 10
        self._max_range = 1000
        self._max_range /= self.map_resolution

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        # ADDED By ME

        # Map meta-data
        self.occupancy_map  = occupancy_map
        self.map_width, self.map_height = occupancy_map.shape
        

        # Laser offset(as per ReadME)
        self.laser_offset = 25
        
        # with open('particle_rays.pkl', 'rb') as f:
        #     self.particle_rays = pickle.load(f)
        
        self.particle_rays = np.load('particle_rays.npy')

        # with open('particle_rays.npy', 'rb') as f:
        #     self.particle_rays = pickle.load(f)
        

        print("Sensor map loaded successfully.")

    
    def vec_centre2laser_transform(self, x_t1_particles, offset=25):

        # Compute the new x and y positions using vectorized operations
        x_t1_l_x = x_t1_particles[:, 0] + offset * np.cos(x_t1_particles[:, 2])
        x_t1_l_y = x_t1_particles[:, 1] + offset * np.sin(x_t1_particles[:, 2])

        # Combine the new positions with the original orientations (theta)
        x_t1_l_particles = np.column_stack((x_t1_l_x, x_t1_l_y, x_t1_particles[:, 2]))
        
        return x_t1_l_particles

    # Function to Tranform from Centre to Laser offset
    def centre2laser_transform(self, x_t1, offset=None, visualize=True):
        
        if offset is None:
            offset = self.laser_offset
            
        
        x_t1_l = np.array([(x_t1[0] + offset*np.cos(x_t1[2])), 
                           (x_t1[1] + offset*np.sin(x_t1[2])),
                            x_t1[2]], dtype=np.float64)

            
        return x_t1_l 

    # Function to trace rays  
    def trace_rays(self):
        # For now collecting extra pts and not just free space
        freespace_rows, freespace_cols = np.where(self.occupancy_map < 0.35)

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
        for idx in range(0, num_pts_):
            # x is col. 
            free_x = freespace_cols[idx]
            free_y = freespace_rows[idx]

            # Ray endpoints
            map_x_idxs_ = (free_x +  ray_len_ * cos_angles_.T ).astype(int)  
            map_y_idxs_ = (free_y +  ray_len_ * sin_angles_.T ).astype(int)  

            # Mask for valid indices (within bounds)
            valid_mask = (
                            (map_x_idxs_ >= 0) & (map_x_idxs_ < self.map_width) &
                            (map_y_idxs_ >= 0) & (map_y_idxs_ < self.map_height)
                         )

             # Clip to prevent out-of-bounds 
            map_x_idxs_clipped = np.clip(map_x_idxs_, 0, self.map_width - 1)
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
                
            # print(f"Saving for particle num:{idx} | out of:{num_pts_}")
            particle_rays_arr_[free_x, free_y, :] = rays_

        # Save the numpy array
        np.save('particle_rays.npy', particle_rays_arr_)
        print("Particle Rays saved successfully as numpy array.")


    # Function to perform ray-tracing
    def ray_tracing(self, x_t1):
        z_t1_arr_expected = np.zeros((180), dtype=np.float64)
        
        # Trace ray for each angle
        for idx, beam_ang in enumerate(np.arange(-90, 90, 1)):
            
            beam_ang_rad = (beam_ang/180.0)*np.pi
            
            # Trace ray for till max-range
            for ray_len in range(0, self._max_range+ self.map_resolution, self.map_resolution):
                ray_x = x_t1[0] + ray_len*np.cos(x_t1[2] + beam_ang_rad)
                ray_y = x_t1[1] + ray_len*np.sin(x_t1[2] + beam_ang_rad) 

                map_x = int(ray_x/self.map_resolution)
                map_y = int(ray_y/self.map_resolution)
                
                # Give max-ranges for out-of-bound rays
                if map_x < 0 or map_x >= self.map_width or map_y < 0 or map_y >= self.map_height:
                    z_t1_arr_expected[idx] = self._max_range
                    break

                # Probability > Obstacle prob
                elif self.occupancy_map[map_y, map_x] > self._min_probability:
                    z_t1_arr_expected[idx] = ray_len
                    break
                         
                # If near to max-range
                elif np.isclose(ray_len, self._max_range):
                    z_t1_arr_expected[idx] = self._max_range
                    break

        return z_t1_arr_expected
    

    def expected_rays(self, x_t1_transformed):
        map_x, map_y = np.floor_divide([x_t1_transformed[0], x_t1_transformed[1]], self.map_resolution).astype(int)
        return self.particle_rays[map_x,map_y,:]

    def expected_ray_measurements_vec(self, x_t1_transformed):        

        # Convert to map idxs
        map_x, map_y = np.floor_divide([x_t1_transformed[:, 0], x_t1_transformed[:, 1]], self.map_resolution).astype(int)

        angles_ = np.rad2deg(x_t1_transformed[:, 2])
        laser_min, laser_max = angles_ - 90, angles_ + 90
        
        laser_min = np.mod(laser_min + 360, 360).astype(int)
        laser_max = np.mod(laser_max + 360, 360).astype(int)

        z_t1_arr_expected_particles = np.zeros((x_t1_transformed.shape[0], 180), dtype=np.float64)


        for i, (min_angle, max_angle) in enumerate(zip(laser_min, laser_max)):
            if min_angle < max_angle:
                z_t1_arr_expected_particles[i, :] = self.particle_rays[map_x[i], map_y[i], min_angle:max_angle]
            else:
                # Slice in two parts
                z_t1_arr_expected_particles[i, :] = np.concatenate((
                                                    self.particle_rays[map_x[i], map_y[i], min_angle:],  
                                                    self.particle_rays[map_x[i], map_y[i], :max_angle]   
                                                                   ))

        return z_t1_arr_expected_particles




    def expected_ray_measurements(self, x_t1_transformed, ang_arr=np.arange(-np.pi, np.pi, np.pi/180)):
        
        z_t1_arr_expected=np.zeros((180), dtype=np.float64)

        
        # Convert to Occupany-map
        map_x, map_y = np.floor_divide([x_t1_transformed[0], x_t1_transformed[1]], self.map_resolution)

        # Search Ray-Tracing Hash-map 
        if (map_x, map_y) in self.particle_rays:
            expected_ray_vals = self.particle_rays[(map_x, map_y)]
        else:
            # Probability zero since the particle is out of free-space(probably lol)
            # print(f"[ERROR] Expected Measurements unavailable for :{x_t1_transformed}")
            return z_t1_arr_expected    
        
        theta_ = x_t1_transformed[2] 

        # Consider just till two decimals
        laser_min = round(theta_ - np.pi/2, 2)
        laser_max = round(theta_ + np.pi/2, 2)
        
        # Wrap laser_min and laser_max if out-of bounds
        if laser_min < -np.pi:
            laser_min += 2 * np.pi  
        if laser_max > np.pi:
            laser_max -= 2 * np.pi  

        # Basic Condition 
        if laser_min <= laser_max:
            closest_min_idx = np.argmin(np.abs(ang_arr - laser_min))
            closest_max_idx = closest_min_idx + 180
            z_t1_arr_expected = np.array(expected_ray_vals[closest_min_idx:closest_max_idx])
        else:
            closest_min_idx = np.argmin(np.abs(ang_arr - laser_min))
            closest_max_idx = np.argmin(np.abs(ang_arr - laser_max))

            # Combine values from two segments
            first_segment = expected_ray_vals[closest_min_idx:]
            second_segment = expected_ray_vals[:closest_max_idx]

            # This should not be required. Debug
            diff = 180 - len(first_segment) + len(second_segment) 
            if diff > 0:
                second_segment = expected_ray_vals[:closest_max_idx + diff]
            # Adjust to ensure the total length is exactly 180 readings
            z_t1_arr_expected = np.concatenate((first_segment, second_segment))[:180]
            
        return z_t1_arr_expected

  

    def beam_range_finder_model_vectorized(self, z_t1_arr, x_t1_particles):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1_particles : array of particle states, shape [num_particles, 3] where each row is [x, y, theta] at time t [world_frame]
        param[out] prob_zt1_arr : array of likelihoods for each particle, shape [num_particles]
        """

        
        num_particles = x_t1_particles.shape[0]
        
        prob_zt_particles = np.ones((num_particles), dtype=np.float64)
        
        # Centre to Laser Transform 
        x_t1_transformed_particles = self.vec_centre2laser_transform(x_t1_particles)

        z_t1_arr_expected_particles = self.expected_ray_measurements_vec(x_t1_transformed_particles)
        

        # TODO: Rerun Map-tracing with this
        # Go in map-frame: Increases speed(e^- vals will descreased while calculating probabilites)
        # IMP: Max range has also been changed
        # z_t1_arr_expected_particles = np.divide(z_t1_arr_expected_particles, self.map_resolution)
        z_t1_arr = np.divide(z_t1_arr, self.map_resolution) 
        
        
        ##### Vectorized Beam Model
        p_hit_beams   = np.zeros((num_particles, z_t1_arr.shape[0]), dtype=np.float64)
        p_short_beams = np.zeros((num_particles, z_t1_arr.shape[0]), dtype=np.float64)
        p_rand_beams  = np.zeros((num_particles, z_t1_arr.shape[0]), dtype=np.float64)
        p_max_beams   = np.zeros((num_particles, z_t1_arr.shape[0]), dtype=np.float64)

        # p_hit condition
        valid_hit_idxs = (z_t1_arr >= 0) & (z_t1_arr <= self._max_range)
        p_hit_beams[:, valid_hit_idxs] = np.exp(-np.square(z_t1_arr[valid_hit_idxs] - z_t1_arr_expected_particles[:, valid_hit_idxs]) / (2 * self._sigma_hit**2))
        # TODO: Required or not?
        p_hit_beams[:, valid_hit_idxs] /= self._sigma_hit * np.sqrt(2*np.pi)
        
        # p_rand condition
        p_rand_beams[:, (z_t1_arr >= 0) & (z_t1_arr < self._max_range)] = 1 / self._max_range * self.map_resolution
        

        # p_max condition (TODO: Ask if vals outside max range be considered here?)
        p_max_beams[:, (z_t1_arr >= self._max_range)] = 1

        
        # p_short condition (TODO: Vectorize somehow)
        for i in range(num_particles):
            valid_short_idxs = (z_t1_arr >= 0) & (z_t1_arr <= z_t1_arr_expected_particles[i])
            p_short_beams[i, valid_short_idxs] = self._lambda_short * np.exp(-self._lambda_short * z_t1_arr[valid_short_idxs])
            
        # Full-model probability 
        p_beams = self._z_hit * p_hit_beams + self._z_short * p_short_beams + self._z_rand * p_rand_beams + self._z_max * p_max_beams
        
        # For Numerical Stability
        p_beams[p_beams < 1e-6] = 1e-6


        prob_zt_particles = np.sum(np.log(p_beams), axis=1)
        # prob_zt_particles = np.exp(prob_zt_particles)
        
        # prob_zt_particles = np.prod(p_beams, axis=1)
        # print(f"prob_z_t :{np.max(prob_zt_particles)}")
# 
        # print(f"z_hit  :  {np.sum(self._z_hit*p_hit_beams, axis=1)}")
        # print(f"z_short:  {np.sum(self._z_short*p_short_beams, axis=1)}")
        # print(f"z_rand :  {np.sum(self._z_rand*p_rand_beams, axis=1)}")
        # print(f"z_max  :  {np.sum(self._z_max*p_max_beams, axis=1)}")
        # print("############################")
        return prob_zt_particles

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """

        # return z_t1_arr
        
        # Centre to Laser Transform
        x_t1_transformed = self.centre2laser_transform(x_t1, visualize=False)

        # Expected Ray Measurments
        z_t1_arr_expected = self.expected_ray_measurements(x_t1_transformed)

        # Particle out-of Bound
        if sum(z_t1_arr_expected) == 0:
            return 0
        
        ##### Vectorized Beam Model
        p_hit_beams   = np.zeros(z_t1_arr.shape, dtype=np.float64)
        p_short_beams = np.zeros(z_t1_arr.shape, dtype=np.float64)
        p_rand_beams  = np.zeros(z_t1_arr.shape, dtype=np.float64)
        p_max_beams   = np.zeros(z_t1_arr.shape, dtype=np.float64)
        prob_zt1 = 1.0
        
        
        # p_hit condition
        valid_hit_idxs = (z_t1_arr >= 0) & (z_t1_arr <= self._max_range) 
        p_hit_beams[valid_hit_idxs] = np.exp(-( np.square(z_t1_arr[valid_hit_idxs] -z_t1_arr_expected[valid_hit_idxs]) / (2*np.square(self._sigma_hit)) ))
        
        # p_short condition
        valid_short_idxs = (z_t1_arr >= 0) & (z_t1_arr <= z_t1_arr_expected)
        p_short_beams[valid_short_idxs] = self._lambda_short*np.exp(-self._lambda_short*z_t1_arr[valid_short_idxs])

        # p_rand condition 
        valid_rand_idxs = (z_t1_arr >= 0) & (z_t1_arr < self._max_range)
        p_rand_beams[valid_rand_idxs] = 1/self._max_range

        # p_max condition (float fcks up w/tout tolerance)
        # valid_max_idxs = np.isclose(z_t1_arr, self._max_range, rtol=0.01)
        valid_max_idxs = (z_t1_arr == self._max_range)
            
        p_max_beams[valid_max_idxs] = 1

        
        # Full-model probability 
        p_beams = self._z_hit*p_hit_beams + self._z_short*p_short_beams + self._z_rand*p_rand_beams + self._z_max*p_max_beams
        
        # Added for Numerical Stability
        p_beams[p_beams < 1e-9] = 1e-9  

        # prob_zt1 = np.sum(p_beams)/p_beams.shape
        prob_zt1 = np.sum(np.log(p_beams))
        
        # print(f"prob max :{}")
        # For testing-purposes
        # prob_zt1 = 1.0

        return prob_zt1
