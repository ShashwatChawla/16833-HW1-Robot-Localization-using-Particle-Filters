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
        # self._z_rand = 100

        self._z_hit = 50
        self._z_short = 40
        self._z_max = 10
        self._z_rand = 50
        
        # self._z_hit = 0.05
        # self._z_short = 0.5 
        # self._z_max = 0.05
        # self._z_rand = 0.01

        # Keeping it small by assuming its good lidar.
        self._sigma_hit = 15.0
        # self._sigma_hit = 5.0
        # self._sigma_hit = 50
        self._lambda_short = 0.01

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
        
        with open('particle_rays.pkl', 'rb') as f:
            self.particle_rays = pickle.load(f)
        
        print("Sensor map loaded successfully.")

    
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


    # Function to trace rays for all free locatns in map 
    def trace_map(self):
        freespace_rows, freespace_cols = np.where(self.occupancy_map == 0)
        num_pts = freespace_cols.shape[0]
        ang_arr = np.arange(-3.14, 3.14, np.pi/180)

        # map_rays = np.zeros((num_pts, 2, ang_arr.shape[0]), dtype=np.float64)
        
        particle_rays = {}
        for idx in range(0,num_pts):
            x = freespace_cols[idx]*10
            y = freespace_rows[idx]*10
            
            ang_vals = []
            for theta in ang_arr:
                # print(f"pose is :{x}, {y},{theta}")
                # Trace ray for till max-range
                for ray_len in range(0, self._max_range+ self.map_resolution, self.map_resolution):
                    ray_x = x + ray_len*np.cos(theta)
                    ray_y = y + ray_len*np.sin(theta) 

                    map_x = int(ray_x/self.map_resolution)
                    map_y = int(ray_y/self.map_resolution)
                    
                    # Give max-ranges for out-of-bound rays (this should never happen)
                    if map_x < 0 or map_x >= self.map_width or map_y < 0 or map_y >= self.map_height:
                        ang_vals.append(self._max_range)
                        break

                    # Probability > Obstacle prob
                    elif self.occupancy_map[map_y, map_x] > self._min_probability:
                        ang_vals.append(ray_len)
                        break
                            
                    # If near to max-range
                    elif np.isclose(ray_len, self._max_range):
                        ang_vals.append(self._max_range)
                        break
            # Store for each x & y
            particle_rays[(x, y)] = ang_vals
            print(f"Stored rays for particle pos:{x},{y}")

        # Save the sensor_map to a file using pickle
        import pickle
        with open('sensor_map.pkl', 'wb') as f:
            pickle.dump(particle_rays, f)

        print("Particle Rays Saved successfully.")

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
                

    def ray_trace_measurements(self, x, y):
        angle_values = self.particle_rays[x, y]
        return angle_values 

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """

        z_t1_arr_expected = np.zeros((180), dtype=np.float64)
        
        # Centre to Laser Offset Transform
        x_t1_transformed = self.centre2laser_transform(x_t1, visualize=False)

        # Searhch Hash-map for Ray-Tracing
        map_x, map_y = int(x_t1_transformed[0]/self.map_resolution), int(x_t1_transformed[1]/self.map_resolution)
        if (map_x, map_y) in self.particle_rays:
            exprected_ray_vals = self.particle_rays[(map_x, map_y)]
        else:
            # Probability zero since the particle is out of free-space(probably lol)
            return 0
            
        ang_arr = np.arange(-3.14, 3.14, np.pi/180)

        theta_ = x_t1_transformed[2] 

        # Consider just till two decimals
        laser_min = round(theta_ - 1.57, 2)
        laser_max = round(theta_ + 1.57, 2)
        
        # Wrap laser_min and laser_max if out-of bounds
        if laser_min < -3.14:
            laser_min += 2 * np.pi  
        if laser_max > 3.14:
            laser_max -= 2 * np.pi  

        # Basic Condition 
        if laser_min <= laser_max:
            closest_min_idx = np.argmin(np.abs(ang_arr - laser_min))
            closest_max_idx = closest_min_idx + 180
            z_t1_arr_expected = np.array(exprected_ray_vals[closest_min_idx:closest_max_idx])
        else:
            closest_min_idx = np.argmin(np.abs(ang_arr - laser_min))
            closest_max_idx = np.argmin(np.abs(ang_arr - laser_max))

            # Combine values from two segments
            first_segment = exprected_ray_vals[closest_min_idx:]
            second_segment = exprected_ray_vals[:closest_max_idx]

            # This should not be required. Debug
            diff = 180 - len(first_segment) + len(second_segment) 
            if diff > 0:
                second_segment = exprected_ray_vals[:closest_max_idx + diff]
            # Adjust to ensure the total length is exactly 180 readings
            z_t1_arr_expected = np.concatenate((first_segment, second_segment))[:180]


        scaling_coeff = 1.0
        
        p_hit_beams   = np.zeros(z_t1_arr.shape, dtype=np.float64)
        p_short_beams = np.zeros(z_t1_arr.shape, dtype=np.float64)
        p_rand_beams  = np.zeros(z_t1_arr.shape, dtype=np.float64)
        p_max_beams   = np.zeros(z_t1_arr.shape, dtype=np.float64)
        prob_zt1 = 1.0
        
        ##### Vectorized Beam Model
    
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
        valid_max_idxs = np.isclose(z_t1_arr, z_t1_arr_expected, rtol=0.01)
        p_max_beams[valid_max_idxs] = 1

        
        # Full-model probability 
        p_beams = self._z_hit*p_hit_beams + self._z_short*p_short_beams + self._z_rand*p_rand_beams + self._z_max*p_max_beams
        
        # Added for Numerical Stability
        # p_beams[p_beams < 1e-9] = 1e-9  

        # prob_zt1 = np.sum(p_beams)/p_beams.shape
        prob_zt1 = np.sum(p_beams)
        # prob_zt1 = np.sum(np.log(p_beams))/p_beams.shape
        # prob_zt1 = np.prod(p_beams)
        

        
        # print(f"z_hit :{np.sum(self._z_hit*p_hit_beams)}")
        # print(f"z_short :{np.sum(self._z_short*p_short_beams)}")
        # print(f"z_rand :{np.sum(self._z_rand*p_rand_beams)}")
        # print(f"z_max :{np.sum(self._z_max*p_max_beams)}")
        # if(prob_zt1 > 0.0):
        #     print(f"Laser probs are :{prob_zt1}")
        # print("############################")
        # # exit()
        # For testing-purposes
        # prob_zt1 = 1.0
        return prob_zt1
