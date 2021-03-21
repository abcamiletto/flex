# URDF2OptimalControl
Here we offer an easy to use library that allows you to do optimal control of a given robot, via urdf.
The optimal control we offer is based on a trajectory target in fixed time.

## Installation Guide
We suppose you have already installed ROS in your system, but it should work anyway.
1. Get CasADi: `pip install casadi`
2. Clone our version of urdf2casadi `git clone https://github.com/abcamiletto/urdf2casadi-light.git`
3. Inside it, install the package `pip install .`
4. Import this package
Since we have yet to make this package "pip installable" you need to import in your python code manually, by having the optimal_control.py in the same folder of your python code or by adding its path to the interpreter.

## Example of usage
```
import optimal_control as OC

# URDF options
urdf_path = "urdf/rrbot.urdf"
root = "link1"
end = "link3"

# The trajectory in respect we want to minimize the cost function
# If qdot isn't given, it will be obtained with differentiation from q
def  trajectory_target(t):
  q = [t]*2
  qdot = [0]*2  
  return q, qdot

# Initial Condition in terms of q, qdot
in_cond = [1]*2 + 0*[2]

# Our cost function 
def  my_cost_func(q, qd, u):
  return cs.mtimes(q.T,q) + cs.mtimes(u.T,u)/100

# Our final term to be added at the end to our cost function
def  my_final_term_cost(q_f, qd_f, u_f):
   return  10*cs.mtimes(q_f.T,q_f) 

# Optimization parameters
time_horizon = 1
steps = 50    

# Load the urdf and calculate the differential equations
urdf_opt = OC.Urdf2Moon(urdf_path, root, end)

# Solve an optimal problem with above parameters
# Results will be a dictionary with q, qd, u as keys
opt = urdf_opt.solve(my_cost_func, time_horizon,
          steps, in_cond, trajectory_target,
          my_final_term_cost, max_iter=70)
          
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

- [ ] Easy UI for algebric constraints
- [x] Auto Parsing of *max_velocity* and *max_effort*
- [x] Friction and Damping modeling --- *numerical problems*
- [x] URDF parsing to get joint stiffness 
- [x] Control over trajectory derivative
- [x] Installation Guide
- [x] Code Guide 
- [x] Update LaTeX

