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
        # self._z_hit = 1
        # self._z_short = 0.1
        # self._z_max = 0.1
        # self._z_rand = 100

        self._z_hit = 15
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 5



        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        # ADDED By ME

        # Map meta-data
        self.occupancy_map  = occupancy_map
        self.map_width, self.map_height = occupancy_map.shape
        self.map_resolution = 10

        # Laser offset(as per ReadME)
        self.laser_offset = 25
        
    
    # Function to Tranform from Centre to Laser offset
    def centre2laser_transform(self, x_t1, offset=None, visualize=True):
        
        if offset is None:
            offset = self.laser_offset
            
        
        x_t1_l = np.array([(x_t1[0] + offset*np.cos(x_t1[2])), 
                           (x_t1[1] + offset*np.sin(x_t1[2])),
                            x_t1[2]], dtype=np.float64)


        if visualize is True:
            x_locs = x_t1[0] / 10.0
            y_locs = x_t1[1] / 10.0
            scat1 = plt.scatter(x_locs, y_locs, c='r', marker='o', s=10)

            # Alternative arrow marker(w/t degree rot)
            # arrow_length = 0.25
            # arrow_width = 0.1
            # dx = arrow_length * np.cos(x_t1[2])  # X-component of arrow direction
            # dy = arrow_length * np.sin(x_t1[2])  # Y-component of arrow direction
            # scat1 = plt.arrow(x_locs, y_locs, dx, dy, head_width=arrow_width, head_length=arrow_width, fc='red', ec='red')
    
            
            x_locs_l = x_t1_l[0] / 10.0
            y_locs_l = x_t1_l[1] / 10.0
            scat2 = plt.scatter(x_locs_l, y_locs_l, c='g', marker='o', s=10)
            
            plt.pause(1)
            scat1.remove()
            scat2.remove()
            
        return x_t1_l 

    # Function to perform ray-tracing
    def ray_tracing(self, x_t1):
        z_t1_arr_expected = np.zeros((180), dtype=np.float64)
        
        # Trace ray for each angle
        for beam_ang in range(1,180+1):
            beam_ang_rad = (beam_ang/180.0)*np.pi
            # Trace ray for till max-range 
            for ray_len in range(0, self._max_range+ self.map_resolution, self.map_resolution):
                ray_x = x_t1[0] + ray_len*np.cos(x_t1[2] + beam_ang_rad)
                ray_y = x_t1[1] + ray_len*np.sin(x_t1[2] + beam_ang_rad) 

                map_x = int(ray_x/self.map_resolution)
                map_y = int(ray_y/self.map_resolution)

                # Give max-ranges for out-of-bound rays
                if map_x < 0 or map_x >= self.map_width or map_y < 0 or map_y >= self.map_height:
                    z_t1_arr_expected[beam_ang-1] = self._max_range
                    break
                
                # Probability > Obstacle prob
                if self.occupancy_map[map_x, map_y] > self._min_probability:
                    z_t1_arr_expected[beam_ang-1] = ray_len
                    break
                
                # If near to max-range
                if np.isclose(ray_len, self._max_range):
                    z_t1_arr_expected[beam_ang-1] = self._max_range
                    break

        return z_t1_arr_expected
                    
                

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """

        # Centre to Laser Offset Transform
        x_t1_offset = self.centre2laser_transform(x_t1, visualize=False)
        
        # Ray-Tracing
        z_t1_arr_expected = self.ray_tracing(x_t1_offset)
        
        # Beam Model
        prob_zt1 = 1.0
        p_hit, p_short, p_max, p_rand = 0.0, 0.0, 0.0, 0.0
        
        scaling_coeff = 10.0
        for idx, z_t in enumerate(z_t1_arr):            
            # Hit Model
            if z_t >= 0 and z_t <= self._max_range:
                p_hit = np.exp(-( np.square(z_t -z_t1_arr_expected[idx]) / (2*np.square(self._z_hit)) ))
            
            
            # Short Model
            if z_t >= 0 and z_t <= self._max_range:
                p_short = self._lambda_short*np.exp(-self._lambda_short*z_t)
            
            
            # Max Range Model
            if z_t == self._max_range:
                p_max = 1
            
            # Random Measurement Model
            if z_t >= 0 and z_t < self._max_range:
                p_rand = 1/self._max_range
            
            
            # Full model probability 
            prob_zt_beam = (self._z_hit*p_hit + self._z_short*p_short + \
                           self._z_max*p_max + self._z_rand*p_rand)*scaling_coeff
            
            
            # Multiply prob for each beam
            prob_zt1 *= (prob_zt_beam)

        # For testing-purposes
        # exit()
        # prob_zt1 = 1.0
        return prob_zt1
