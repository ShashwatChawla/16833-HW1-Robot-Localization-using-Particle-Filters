'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel    
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

# Generate a random seed
# seed = np.random.randint(0, 100000)
seed = 78995
# seed = 63630
# Print the seed so you can use it next time
print(f"Random Seed: {seed}")
np.random.seed(seed)


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


# Generate Array frrom pts
def ranges_to_points(ranges):
    pts = []
    for i in range(len(ranges)):
        z = ranges[i]
        theta = np.deg2rad(i - 90)
        
        x = z*np.cos(theta)
        y = z*np.sin(theta)
        pts.append([x, y])
        
    return np.array(pts)


def visualize_timestep(X_bar, tstep, output_path, ranges):
    plt.figure(1 )

    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    # scat = plt.scatter(x_locs, y_locs, c='b', marker='o', s=0.85)
    arrows = plt.quiver(x_locs, y_locs, 25* np.cos(X_bar[:, 2]),25 * np.sin(X_bar[:, 2]), angles='xy', scale_units='xy', scale=0.85, color='b')
    
    # Best Robot position
    x_robot = X_bar[np.argmax(X_bar[:, 3])]
    x_laser = x_robot[0] + 25 * np.cos(x_robot[2])
    y_laser = x_robot[1] + 25 * np.sin(x_robot[2])
    X_laser = np.array([x_laser, y_laser])
    
    laser_pts = ranges_to_points(ranges)
    # transform laser_pts to world frame
    laser_pts_world = np.dot(laser_pts, np.array([[np.cos(- x_robot[2]), -np.sin(- x_robot[2])], [np.sin(- x_robot[2]), np.cos(- x_robot[2])]]))
    laser_pts_world[:, 0] += X_laser[0]
    laser_pts_world[:, 1] += X_laser[1]
    laser_pts_map = laser_pts_world / 10.0  
    
    scat2 = plt.scatter(laser_pts_map[:, 0], laser_pts_map[:, 1], s=0.4, c='r')

    robot_text = f"Robot Position: ({x_robot[0]:.2f}, {x_robot[1]:.2f}, {x_robot[2]:.2f})"
    text = plt.text(5, 770, robot_text, fontsize=7, color='g')
    # robot_kidnap = plt.text(5, 735, "Kidnap Detection Enabled", fontsize=8, color='r')
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)

    # To visualize traced-ray
    # z_t1_k_star = sensor_model.zt_k_star_particle(x_robot) 
    # laser_pts = ranges_to_points(z_t1_k_star)
    # z_star_pts_world = np.dot(laser_pts, np.array([[np.cos(- x_robot[2]), -np.sin(- x_robot[2])], [np.sin(- x_robot[2]), np.cos(- x_robot[2])]]))
    # z_star_pts_world[:, 0] += X_laser[0] / 10
    # z_star_pts_world[:, 1] += X_laser[1] / 10
    # z_star_pts_map = z_star_pts_world   
    # scat3 = plt.scatter(z_star_pts_map[:, 0], z_star_pts_map[:, 1], s=0.4, c='g')

    # scat.remove()
    scat2.remove()  
    text.remove()  
    # scat3.remove()
    arrows.remove()


# Depricated: Visualize expected & rangefinder measurements for a single particle
def visualize_particle_rays(x_t, z_t):    
    x, y = x_t[0]/10, x_t[1]/10
    x_t_ = x_t.reshape(1, 3)
    # print(x_t.shape)
    laser_measurements_ = sensor_model.expected_ray_measurements_vec(x_t_)
    laser_measurements_ = laser_measurements_.reshape(-1)
    angles = np.linspace(x_t[2] - np.pi/2, x_t[2] + np.pi/2, 180)
    
    ray_end_x = x + (laser_measurements_*np.cos(angles))
    ray_end_y = y + (laser_measurements_*np.sin(angles))
    
    lines = []
    for x_, y_ in zip(ray_end_x, ray_end_y):
        line = plt.plot([x, x_], [y, y_], 'y-') 
        lines.extend(line) 
    
    rangefinder_end_x = x + (z_t*np.cos(angles)/10)
    rangefinder_end_y = y + (z_t*np.sin(angles)/10)
    
    rangefinder_lines = []
    for x_, y_ in zip(rangefinder_end_x, rangefinder_end_y):
        line_ = plt.plot([x, x_], [y, y_], 'g-') 
        rangefinder_lines.extend(line_) 
    
    
    
    scat = plt.scatter(x, y, c='r', marker='o')

    plt.pause(0.001)

    for line in lines:
        line.remove()
    for line_ in rangefinder_lines:
        line_.remove()
    scat.remove()

# Function for random particle spread
def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init

# Function for free-space particle spread
def init_particles_freespace(num_particles, occupancy_map):

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

import time
if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=10000, type=int)
    parser.add_argument('--visualize', action='store_true')
    ## Added by Team
    # Argument to Enable robot-kidnapping
    parser.add_argument('--detect_kidnap', action='store_false')
    # Argument to Enable adaptive resampling
    parser.add_argument('--adaptive', action='store_true')
    
    
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')


    num_particles = args.num_particles

    motion_model = MotionModel(seed)
    sensor_model = SensorModel(occupancy_map, seed)
    resampler = Resampling(num_particles, occupancy_map, seed, kidnap_test=args.detect_kidnap)

    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)
    
    
    first_time_idx = True

    # log1- GT:
    # X_bar[:] = np.array([4150, 3900, 3.14, 1])
    # log3- GT:
    # X_bar[:] = np.array([4650, 1630, 0.0, 1])

    tic = time.time()
    for time_idx, line in enumerate(logfile):


        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((X_bar.shape[0], 4), dtype=np.float64)
        u_t1 = odometry_robot


        """
        VECTORIZED MOTION MODEL
        """
        X_bar_new[:, 0:3] = motion_model.update_vectorized(u_t0, u_t1, X_bar[:, 0:3])
        
        
        """
        VECTORIZED SENSOR MODEL
        """
        if (meas_type == "L"):
            z_t = ranges
            X_bar_new[:, 3]  = sensor_model.beam_range_finder_model_vectorized(z_t, X_bar_new)
        else:
            X_bar_new[:, 3] = X_bar[:, 3]
            

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """        
        if (meas_type == "L"):
            if args.adaptive:
                X_bar = resampler.adaptive_low_variance_sampler(X_bar)
            else:
                X_bar = resampler.low_variance_sampler(X_bar)
            
        """
        VISUALIZATION
        """
        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output, ranges)
        

    toc = time.time()
    print("Time taken :", toc-tic)
