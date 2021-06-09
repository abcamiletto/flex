# From URDF to Optimal Control
Here we offer an easy to use library that allows you to do optimal control of a given robot, via urdf.
The optimal control we offer is based on a trajectory target in fixed time.

## Installation Guide
We suppose you have already installed ROS in your system, but it should work anyway.

We wrote a simple bash script that set up all the dependencies and needs for you, leaving with a clean installation of the package.

```bash
git clone https://github.com/abcamiletto/urdf_optcontrol.git && cd urdf_optcontrol
./install.sh
```

The needed dependencies are (automatically installed with previous command):

1. casadi

```bash
pip install casadi
```
    
2. urdf2casadi
    
```bash
git clone https://github.com/mahaarbo/urdf2casadi.git
cd urdf2casadi
pip install .
```


To see if it's working, run the python file in the example folder


## Example of usage
```python
#!/usr/bin/env python3
from urdf_optcontrol import optimizer
from matplotlib import pyplot as plt

# URDF options
urdf_path = '/path/to/urdf/file'
root = 'link1'
end = 'link3'


# The trajectory in respect we want to minimize the cost function
# If qdot isn't given, it will be obtained with differentiation from q
def trajectory_target_(t):
    q = [t] * 2
    qdot = [0] * 2
    return (q, qdot)

# Our cost function
def my_cost_func(q, qd, u, t):
    return 10*q.T@q + u.T@u/10

# Our final term to be added at the end to our cost function
def my_final_term_cost(q_f, qd_f, u_f):
    return 10*q_f.T@q_f

# Additional Constraints I may want to set
def my_constraint1(q, q_dot, u, ee_pos, qq_dot):
    return [-10, -10], u, [10, 10]
def my_constraint2(q, q_dot, u, ee_pos, qq_dot):
    return [-4, -4], q_dot, [4, 4]
def my_constraint3(q, q_dot, u, ee_pos, qq_dot):
    return 0, ee_pos[0]**2 + ee_pos[1]**2 + ee_pos[2]**2, 20
def my_constraint4(q, q_dot, u, ee_pos, qq_dot):
    return [-20, -20], qq_dot, [20, 20]
my_constraints=[my_constraint1, my_constraint2, my_constraint3, my_constraint4]

# Constraints to be imposed on the final instant of the simulation
# e.g. impose a value of 1 radiant for both the joints
def my_final_constraint1(q, q_dot, u, ee_pos, qq_dot):
    return [1, 1], q, [1, 1]
my_final_constraints = [my_final_constraint1]    # if not set, it is free (and optimized)

# Initial Condition in terms of q, qdot
in_cond = [0,0] + [0,0]

# Optimization parameters
steps = 50
time_horizon = 1    # if not set, it is free (and optimized)

# Load the urdf and calculate the differential equations
optimizer.load_robot(urdf_path, root, end)

# Solve an optimal problem with above parameters
# Results will be a dictionary with q, qd, u as keys
res = optimizer.load_problem(
    my_cost_func,
    steps,
    in_cond,
    trajectory_target_,
    time_horizon=time_horizon,
    final_term_cost=my_final_term_cost, 
    my_constraint=my_constraints, 
    my_final_constraint=my_final_constraints,
    max_iter=70
    )

# Print the results!
fig = optimizer.show_result()
plt.show()
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
- [x] Friction and Damping modeling --- *numerical problems*
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
- [ ] ROS utilization guide 
- [x] Add a parameter to choose whether you want to implement motor dynamics or not
- [x] Fix examples
- [x] Modelling without motor inertias
- [ ] Raise error if cost function with theta when in previous case
- [ ] Add a method "solve" to optimizer instead of returning on load_problem
- [ ] Convert result to numpy array and make sure you can iterate on it
- [x] Return T if we're minimizing it
- [ ] Desired trajectory as optional argument
- [ ] Insert ee position and q acceleration in cost functions


To do or not to do?
- [x] Implementation of a minimum time cost function 
- [x] Implementation of fixed final constraints
- [ ] Direct collocation
