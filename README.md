# From URDF to Optimal Control
Here we offer an easy to use library that allows you to do optimal control of a given robot, via urdf.
The optimal control we offer is based on a trajectory target in fixed time.

## Installation Guide
We suppose you have already installed ROS in your system, but it should work anyway.

#### Easy Guide
That's the one-line guide, that will get everything done but needs sudo access. If you wanna play it safe, follow the "hard" one

```bash
git clone https://github.com/abcamiletto/urdf_optcontrol.git
cd urdf_optcontrol
./install.sh
```

To see if it's working, run the python file in the example folder

#### "Hard" Guide
1. Get CasADi: `pip install casadi`
2. Clone our version of urdf2casadi `git clone https://github.com/abcamiletto/urdf2casadi-light.git`
3. Inside it, install the package `pip install .`
4. Clone this repo `git clone https://github.com/abcamiletto/urdf_optcontrol.git`
5. Inside it, install the package `pip install .`

## Example of usage
```python
import urdf_optcontrol as OC
import casadi as cs

# URDF options
urdf_path = 'urdf/rrbot.urdf'
root = 'link1'
end = 'link3'


# The trajectory in respect we want to minimize the cost function
# If qdot isn't given, it will be obtained with differentiation from q
def trajectory_target_(t):
    q = [t] * 2
    qdot = [0] * 2
    return (q, qdot)

# Our cost function
def my_cost_func(q, qd, u):
    return cs.mtimes(q.T, q) + cs.mtimes(u.T, u) / 100

# Our final term to be added at the end to our cost function
def my_final_term_cost(q_f, qd_f, u_f):
    return 10 * cs.mtimes(q_f.T, q_f)

# Additional Constraints i may want to set
def my_constraint1(q, q_dot, u, ee_pos):
    return [-10, -10], u, [10, 10]
def my_constraint2(q, q_dot, u, ee_pos):
    return [-4, -4], q_dot, [4, 4]
def my_constraint3(q, q_dot, u, ee_pos):
    return 0, ee_pos[0]**2 + ee_pos[1]**2 + ee_pos[2]**2, 20
my_constraints=[my_constraint1, my_constraint2, my_constraint3]


# Initial Condition in terms of q, qdot
in_cond = [1] * 2 + [0] * 2

# Optimization parameters
time_horizon = 1
steps = 50

# Load the urdf and calculate the differential equations
urdf_opt = OC.URDFopt(urdf_path, root, end)

# Solve an optimal problem with above parameters
# Results will be a dictionary with q, qd, u as keys
opt = urdf_opt.solve(
    my_cost_func,
    time_horizon,
    steps,
    in_cond,
    trajectory_target_,
    my_final_term_cost,
    my_constraints,
    max_iter=70
    )

# Print the results!
fig = urdf_opt.print_results()
fig.show()
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
- [ ] SAE modeling

