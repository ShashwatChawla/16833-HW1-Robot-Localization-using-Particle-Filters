TLDR: The following are notes that were provided on PIAZZA starting in the fall 2021 semester to address common issues people run into when attempting this homework. Its suggested a version of this be provided in future semesters.

You are asked to implement a particle filter for robot localization in this homework. Monty discussed particle filters on 09/09 in class.

We strongly recommend working in groups (of up to 2) for this assignment. This is a challenging assignment, so we recommend starting as soon as possible after Monday's class.

Please use Gradescope to submit your report and solutions.

OH will begin next week, let us know if we can help before then in Piazza, otherwise we will be happy to answer your questions then! 

Here is a list of helpful tips for the homework:

Please consult this before coming to OH since they might help solve some of your questions :)

Sensor Model

You may not need to compute eta in eqn 6.6 of Probabilistic Robotics. Instead, you can simply compute the CDF of your gaussian between 0 and z_max and scale so this is 1.
In general, don't get too caught up on theoretical correctness when computing weights for your distributions. Since you will normalize particle weights in the end, what you need to focus on is the relative weighing between your distributions (p_hit, p_short, p_rand,p_max).
Using a product of probabilities in place of the sum of log probabilities seems to work better for people. You'll have to scale your sensor model (adjusting weight of p_rand) to values close to 1 to avoid numerical issues when using a product.
There is a trade-off between convergence and robustness when tuning the variance and weights of the gaussian you have for phit. Too low variance and too high a weight can cause the filter to converge too soon (to possibly to a wrong location). 

Ray Casting

Using a subset of beams will reduce computation and allow you to debug/tune much faster
Make sure that while doing ray casting you transform the ray origin to the laser sensor coordinates. According to instruct.txt, laser is set 25cm forward (x-axis) from robot's center.
When predicting laser range measurements, you can draw the rays you cast from the particle on the map as a visual debugging tool (don't draw them when you actually run your filter efficiently).

General Parameter Tuning

While there is no ground truth trajectory to refer to, looking at robotmovie.gif can give you a sense of the starting map location of the robot in datalog 1. You can use that information for initializing particles only in a subset of the map. This can speed up your parameter tuning iterations as you now only need a subset of particles for testing. In the final video submission, however, we would still like to see the particle filter converging after having initialized particles throughout free space. 
There isn't much benefit to performing odometry updates at a higher rate than the measurement updates. Skipping odometry-only (measurement type : O) measurements and only propagating particles when a laser measurement (measurement type: L) comes in will help speed up your filter and also reduce the effect of odometry noise.
One way to tune your system could be to first adjust sensor model weights so you achieve consistent convergence to the correct region of the map. Then adjust your motion model so you introduce enough noise to allow your particle to follow a sensible trajectory, without diverging too much. 

Good luck! 

Teaching Staff 
