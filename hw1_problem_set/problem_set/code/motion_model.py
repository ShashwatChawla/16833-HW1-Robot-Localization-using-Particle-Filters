'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.01
        self._alpha2 = 0.01
        self._alpha3 = 0.01
        self._alpha4 = 0.01

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """ 
        ###################################################
        #  Odom Motion Model-Probalistic Robotics(Ch5)  #  
        ###################################################

        # Delta-change in Odometry 
        del_x, del_y, del_theta = u_t1 - u_t0
        
        del_rot_1 = np.arctan2(del_y, del_x) - u_t0[2]
        del_trans = np.sqrt(np.square(del_x) + np.square(del_y))
        del_rot_2 = del_theta - del_rot_1

        del_rot_1_bar = del_rot_1 - np.random.normal(
            loc=0.0, 
            scale=np.sqrt(
                self._alpha1*np.square(del_rot_1) + 
                self._alpha2*np.square(del_trans) 
                )
            )
        
        del_trans_bar = del_trans - np.random.normal(
            loc=0.0, 
            scale=np.sqrt(
                self._alpha3*np.square(del_trans) + 
                self._alpha4*np.square(del_rot_1) +
                self._alpha4*np.square(del_rot_2) 
            )
        )

        del_rot_2_bar = del_rot_2 - np.random.normal(
            loc=0.0,
            scale=np.sqrt(
                self._alpha1*np.square(del_rot_2) + 
                self._alpha2*np.square(del_trans)
            )
        )        

        x_t1 = np.zeros(3, dtype=np.float64)
        # World-Frame Transformation
        x_t1[0] = x_t0[0] + del_trans_bar*np.cos(x_t0[2] + del_rot_1_bar)
        x_t1[1] = x_t0[1] + del_trans_bar*np.sin(x_t0[2] + del_rot_1_bar)
        x_t1[2] = x_t0[2] + del_rot_1_bar + del_rot_2_bar

        return x_t1 

        return np.random.rand(3)

