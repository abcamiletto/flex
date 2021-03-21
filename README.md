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
TODO


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
- [x] Update LaTeX

