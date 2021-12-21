# Flex* - From URDF to Flexible Robots Optimal Control
Here we offer an easy to use library that allows you to do optimal control of a given robot, via urdf.
Our tool provide an optimal solution for almost every problem related to a robot that somebody can think of...

<p align="center">
  <img src="https://github.com/marcobiasizzo/flex_video/blob/main/videos/rrbot.gif" width="240" height="240" /> <img src="https://github.com/marcobiasizzo/flex_video/blob/main/videos/jump.gif" width="240" height="240" /> <img src="https://github.com/marcobiasizzo/flex_video/blob/main/videos/panda.gif" width="240" height="240" />
</p>

You can find some automatically generated reports [here](https://htmlpreview.github.io/?https://github.com/marcobiasizzo/flex_videos/blob/main/reports/rrbot_p2p_low_energy_report.html) and [here](https://htmlpreview.github.io/?https://github.com/marcobiasizzo/flex_videos/tree/main/reports/rrbot_p2p_max_speed_report.html).

https://htmlpreview.github.io/?https://github.com/bartaz/impress.js/blob/master/index.html

## Installation Guide
We wrote a simple bash script that set up all the dependencies and needs for you, leaving with a clean installation of the package.

```bash
git clone https://github.com/abcamiletto/urdf2optcontrol.git && cd urdf2optcontrol
./install.sh
```

The needed dependencies are automatically installed with previous command
For a custom installation, just the two following commands are needed:

1. casadi

```bash
pip install casadi
```
    
2. urdf2casadi
    
```bash
git clone https://github.com/abcamiletto/urdf2casadi-light.git
cd urdf2casadi
pip install .
```


To see if it's working, run of the python files in the example folder.


## Example of usage
```python
#!/usr/bin/env python3
from urdf2optcontrol import optimizer
from matplotlib import pyplot as plt 

# URDF options
urdf_path = './urdf/rrbot.urdf'
root = 'link1'
end = 'link3'


# The trajectory in respect we want to minimize the cost function
# If qd isn't given, it will be obtained with differentiation from q
def trajectory_target_(t):
    q = [t] * 2
    return q

# Our cost function
def my_cost_func(q, qd, qdd, ee_pos, u, t):
    return 100*q.T@q + u.T@u/10

# Additional Constraints I may want to set
def my_constraint1(q, q_d, q_dd, ee_pos, u, t):
    return [-30, -30], u, [30, 30]
def my_constraint2(q, q_d, q_dd, ee_pos, u, t):
    return [-15, -15], q_dd, [15, 15]
my_constraints=[my_constraint1, my_constraint2]

# Constraints to be imposed on the final instant of the simulation
# e.g. impose a value of 1 radiant for both the joints
def my_final_constraint1(q, q_d, q_dd, ee_pos, u):
    return [1, 1], q, [1, 1]
my_final_constraints = [my_final_constraint1] 

# Initial Condition in terms of q, qd
in_cond = [0,0] + [0,0]

# Optimization parameters
steps = 40
time_horizon = 1.0    # if not set, it is free (and optimized)

# Load the urdf and calculate the differential equations
optimizer.load_robot(urdf_path, root, end, 
                        get_motor_dynamics=True) # useful only for SEA (default is True)

# Loading the problem conditions
optimizer.load_problem(
    cost_func=my_cost_func,
    control_steps=steps,
    initial_cond=in_cond,
    trajectory_target=trajectory_target_,
    time_horizon=time_horizon,
    constraints=my_constraints, 
    final_constraints=my_final_constraints,
    max_iter=500
    )

# Solving the non linear problem
res = optimizer.solve()

# Print the results!
fig = optimizer.plot_result(show=True)
```


### Things to do:

- [x] Indipendence to number of joints
- [x] Joint limits from URDF
- [x] Reference point different from origin
- [x] Reference Trajectory
- [x] Cost function on final position
- [x] Check performance with SX/MX
- [x] Check sparse Hessian *problem*
- [x] Fix Latex Mathematical Modeling
- [x] Reconstruct the actual optimal input

Second Round

- [x] Easy UI for algebraic constraints
- [x] Auto Parsing of *max_velocity* and *max_effort*
- [x] Friction and Damping modeling 
- [x] URDF parsing to get joint stiffness 
- [x] Control over trajectory derivative
- [x] Installation Guide
- [x] Code Guide 
- [x] Update LaTeX

Third Round

- [x] Test on a multiple arms robot
- [x] Pip package and Auto installation
- [x] SAE modeling

Fourth Round

- [x] SAE with not every joint elastic
- [x] Check damping+friction implementation in urdf2casadi
- [x] Add a parameter to choose whether you want to implement motor dynamics or not
- [x] Fix examples
- [x] Modelling without motor inertias
- [x] Add a method "solve" to optimizer instead of returning on load_problem
- [x] Convert result to numpy array and make sure you can iterate on it
- [x] Return T if we're minimizing it
- [x] Desired trajectory as optional argument
- [x] Insert ee position and q acceleration in cost functions

Fifth Round

- [x] Development of a visualization function for results generating HTML report
- [ ] Implement a path tracking with min time optimization
- [ ] Customization of report layout
- [ ] Link GitHub page for ROS implementation example

To do or not to do?

- [x] Implementation of a minimum time cost function 
- [x] Implementation of fixed final constraints
- [ ] Direct collocation
- [ ] Implement different types of elastic actuators
