import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from .utils import show

class Problem:
    def __init__(self, robot, cost_func, control_steps, initial_cond, trajectory_target=None, time_horizon=None,
                 final_term_cost=None, constraints=None, final_constraints=None,
                 rk_interval=4, max_iter=1000):
        """ Solve method. Takes the problems conditions as input and set up its solution"""
        self.robot = robot

        # Store Robot's useful attributes as Problem Attributes.
        self.__dict__.update(self.robot.__dict__)

        # Store Values
        self.t = cs.SX.sym("t", 1)
        if trajectory_target is None:
            trajectory_target = lambda t: [0]*self.num_joints
        self.traj, self.traj_dot = self.format_trajectory(trajectory_target, self.t)
        self.cost_func = cost_func
        self.final_term_cost = final_term_cost
        self.constraints = constraints
        self.final_constraints = final_constraints
        self.T = cs.MX.sym("T", 1) if time_horizon == None else time_horizon
        self.N = control_steps
        self.rk_intervals = rk_interval
        self.max_iter = max_iter

        # If we get just the position, we set the initial velocity to zero
        if len(initial_cond) == 2 * self.num_joints:
            self.initial_cond = initial_cond
        elif len(initial_cond) == self.num_joints:
            self.initial_cond = initial_cond + [0] * self.num_joints
        else:
            raise ValueError('Initial Conditions should be {} item long: (q, q_dot)'.format(2 * self.num_joints))

        # Calculating the differential equations
        self.f = self._get_diff_eq(self.cost_func)

        # Rewriting the differential equation symbolically with Range-Kutta 4th order
        self.F = self._rk4(self.f, self.T, self.N, self.rk_intervals)

        # Setting up IPOPT non linear problem solver
        self._nlp_solver(self.initial_cond, self.final_term_cost, trajectory_target, self.constraints)

    def _get_diff_eq(self, cost_func):
        '''Function that returns the RhS of the differential equations. See the papers for additional info'''
        
        RHS = []

        # RHS that's in common for all cases
        rhs1 = self.q_dot
        rhs2 = self.M_inv @ (-self.Cq - self.G - self.Fd @ self.q_dot - self.Ff @ cs.sign(self.q_dot))

        # Adjusting RHS for SEA with known inertia. Check the paper for more info.
        if self.sea and self.SEAinertia: 
    
            rhs2 += -self.M_inv @ self.tau_sea
            rhs3 = self.theta_dot
            rhs4 = cs.pinv(self.B) @ (-self.FDsea @ self.q_dot + self.u + self.tau_sea)
            RHS = [rhs1, rhs2, rhs3, rhs4]

            # Adjusting the lenght of the variables
            self.x = cs.vertcat(self.q, self.q_dot, self.theta, self.theta_dot)
            self.num_state_var = self.num_joints * 2
            self.lower_q = self.lower_q * 2
            self.lower_qd = self.lower_qd * 2
            self.upper_q = self.upper_q * 2
            self.upper_qd = self.upper_qd * 2

        # Adjusting RHS for SEA modeling, with motor inertia unknown
        elif self.sea and not self.SEAinertia: 
            rhs2 += -self.M_inv @ self.K @ (self.q - self.u)
            self.upper_u, self.lower_u = self.upper_q, self.lower_q
            RHS = [rhs1, rhs2]

            # State  variable
            self.x = cs.vertcat(self.q, self.q_dot)
            self.num_state_var = self.num_joints

        # Adjusting RHS for classic robot modeling, without any SEA
        else:  
            rhs2 += self.M_inv @ self.u
            RHS = [rhs1, rhs2]

            # State  variable
            self.x = cs.vertcat(self.q, self.q_dot)
            self.num_state_var = self.num_joints

        # Evaluating q_ddot, in order to handle it when given as an input of cost function or constraints
        self.q_ddot_val = self._get_joints_accelerations(rhs2)

        # The differentiation of J will be the cost function given by the user, with our symbolic 
        # variables as inputs
        J_dot = cost_func(  self.q - self.traj,
                            self.q_dot - self.traj_dot,
                            self.q_ddot_val(self.q,self.q_dot, self.u),
                            self.ee_pos(self.q),
                            self.u,
                            self.t
        )

        # Setting the relationship
        self.x_dot = cs.vertcat(*RHS)

        # Defining the differential equation
        f = cs.Function('f',    [self.x, self.u, self.t],  # inputs
                                [self.x_dot, J_dot])  # outputs

        return f

    def _get_joints_accelerations(self, rhs2):
        '''Returns a CASaDi function that maps the q_ddot from other joints info.'''
        q_ddot_val = cs.Function('q_ddot_val', [self.q, self.q_dot, self.u], [rhs2])
        return q_ddot_val

    def _rk4(self, f, T, N, m):
        '''Applying 4th Order Range Kutta for N steps and m Range Kutta intervals'''

        # Defining the time step
        dt = T / N / m

        # Variable definition for RK method
        X0 = cs.MX.sym('X0', self.num_state_var * 2)
        U = cs.MX.sym('U', self.num_joints)
        t = cs.MX.sym('t', 1)

        # Initial value
        X = X0
        Q = 0

        # Integration - 4th order Range Kutta, done m times
        for j in range(m):
            k1, k1_q = f(X, U, t)
            k2, k2_q = f(X + dt / 2 @ k1, U, t)
            k3, k3_q = f(X + dt / 2 @ k2, U, t)
            k4, k4_q = f(X + dt @ k3, U, t)
            # Update the state
            X = X + dt / 6 @ (k1 + 2 * k2 + 2 * k3 + k4)
            # Update the cost function 
            Q = Q + dt / 6 @ (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

        # Getting the differential equation discretized but still symbolical. Paper for more info
        F = cs.Function('F', [X0, U, t], [X, Q], ['x0', 'p', 'time'], ['xf', 'qf'])

        return F

    def _nlp_solver(self, initial_cond, final_term_cost, trajectory_target, constraints):
        '''Setting up the non linear problem settings, manually discretizing each time step'''

        # Start with an empty NLP
        w = []      # free variables vector
        w_g = []    # initial guess
        lbw = []    # lower bounds of inputs
        ubw = []    # upper bounds of inputs 
        J = 0       # initial value of cost func
        g = []      # constraints vector
        lbg = []    # lower bounds of constraints
        ubg = []    # upper bound of constraints

        # Defining a new time variable because we need MX here
        self.t = cs.MX.sym("t", 1)
        self.traj, self.traj_dot = self.format_trajectory(trajectory_target, self.t)

        # With SEA we have double the state variable and double the initial condition. 
        # We assume to start on the equilibrium
        if self.sea and self.SEAinertia:
            initial_cond = initial_cond * 2

        # Finally, we integrate all over the timeframe
        dt = self.T / self.N

        # Defining the variables needed for the integration
        Xk = cs.MX.sym('X0', self.num_state_var * 2)
        # Adding it to the vector that has to be optimized
        w += [Xk]
        # Setting the initial condition
        w_g += initial_cond
        lbw += initial_cond
        ubw += initial_cond

        for k in range(self.N):
            # Variable for the control
            Uk = cs.MX.sym('U_' + str(k), self.num_joints)  # generate the k-th control command, nx1 dimension
            w += [Uk] 
            self.add_input_constraints(w_g, lbw, ubw)

            # Integrate till the end of the interval
            Fk = self.F(x0=Xk, p=Uk, time=dt * k)  # This is the actual integration!
            Xk_next = Fk['xf'] 
            J = J + Fk['qf']

            # Add custom constraints
            if constraints is not None:
                self.add_custom_constraints(g, lbg, ubg, Xk, Uk, constraints)

            # Defining a new state variable
            Xk = cs.MX.sym('X'+str(k+1), self.num_state_var * 2)
            w += [Xk]
            self.add_state_constraints(lbw, ubw, w_g, initial_cond)

            # Imposing equality constraint
            g += [Xk_next - Xk] # Imposing the state variables to work under the differential equation constraints
            lbg += [0] * (2 * self.num_state_var) 
            ubg += [0] * (2 * self.num_state_var)

        # Setting constraints on the last timestep
        if constraints is not None:
            self.add_custom_constraints(g, lbg, ubg, Xk, None, constraints)

        # If we are optimizing for minimum T, we set its boundaries
        if isinstance(self.T, cs.casadi.MX):
            w += [self.T]
            w_g += [1.0]
            lbw += [0.05]
            ubw += [float('inf')]

        # Add a final term cost. If not specified is 0.
        if final_term_cost is not None:
            Q_dd = self.q_ddot_val(Xk[0:self.num_joints], Xk[self.num_joints:2 * self.num_joints], np.array([0]*self.num_joints))
            EE_pos = self.ee_pos(Xk[0:self.num_joints])
            
            J = J + final_term_cost(Xk[0:self.num_joints] - cs.substitute(self.traj, self.t, self.T),
                                    Xk[self.num_joints:2*self.num_joints] - cs.substitute(self.traj_dot, self.t, self.T),
                                    Q_dd,
                                    EE_pos,
                                    Uk)

        # Adding constraint on final timestep
        if self.final_constraints is not None:
            self.add_custom_constraints(g, lbg, ubg, Xk, None, self.final_constraints)

        # Define the problem to be solved
        problem = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
        # NLP solver options
        opts = {}
        opts["ipopt"] = {'max_iter': self.max_iter}
        # Define the solver and add boundary conditions
        solver = cs.nlpsol('solver', 'ipopt', problem, opts)
        self.solver = solver(x0=w_g, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        

    def add_custom_constraints(self, g, lbg, ubg, Xk, Uk, constraints):
        '''Helper function to simplify the settings of the constraints'''
        if Uk is None:
            Uk = np.array([0]*self.num_joints)

        EEk_pos_ = self.ee_pos(Xk[0:self.num_joints])
        Q_ddot_ = self.q_ddot_val(Xk[0:self.num_joints], Xk[self.num_joints:2 * self.num_joints], Uk)

        for constraint in constraints:
            l_bound, f_bound, u_bound = constraint(Xk[0:self.num_joints], Xk[self.num_joints:2 * self.num_joints],
                                                   Q_ddot_, EEk_pos_, Uk)

            if not isinstance(f_bound, list): f_bound = [f_bound]
            if not isinstance(l_bound, list): l_bound = [l_bound]
            if not isinstance(u_bound, list): u_bound = [u_bound]

            g += f_bound
            lbg += l_bound
            ubg += u_bound

    def add_state_constraints(self, lbw, ubw, w_g, initial_cond):
        # Add inequality constraint on state
        w_g += [0] * (2 * self.num_state_var)  # initial guess
        lbw += self.lower_q  # lower bound on q
        lbw += self.lower_qd  # lower bound on q_dot
        ubw += self.upper_q  # upper bound on q
        ubw += self.upper_qd  # upper bound on q_dot

    def add_input_constraints(self, w_g, lbw, ubw):
        w_g += [0] * self.num_joints  # initial guess for each timestep
        # Add inequality constraint on inputs
        lbw += self.lower_u  # lower bound on u
        ubw += self.upper_u  # upper bound on u

    def solve_nlp(self):
        # Solve the NLP
        opt = self.solver['x']

        # If we solved for minimum T, then let's update the results
        if isinstance(self.T, cs.casadi.MX):
            self.T_opt = np.single(opt[-1]).flatten()
            opt = opt[0:-1]
        else:
            self.T_opt = self.T

        # Rewriting the results in a more convenient way for both SEA and non SEA cases
        # For 2 Joints and N timesteps results will be formatted as:
        # [[Q0_0, Q0_1, ... Q0_N][Q1_0, Q1_1, ... Q1_N]] where Qi_j is the value of the ith joint at the jth timestep
        if self.sea and self.SEAinertia:
            self.q_opt = [opt[idx::(3 * (self.num_joints) + 2 * (self.num_joints))] for idx in range(self.num_joints)]
            self.qd_opt = [opt[self.num_joints + idx::(3 * (self.num_joints) + 2 * (self.num_joints))] for idx in
                           range(self.num_joints)]
            self.theta_opt = [opt[self.num_joints * 2 + idx::(3 * (self.num_joints) + 2 * (self.num_joints))] for idx in
                              range(self.num_joints)]
            self.thetad_opt = [
                opt[self.num_joints * 2 + self.num_joints + idx::(3 * (self.num_joints) + 2 * (self.num_joints))] for
                idx in range(self.num_joints)]
            self.u_opt = [
                opt[self.num_joints * 2 + 2 * (self.num_joints) + idx::(3 * (self.num_joints) + 2 * (self.num_joints))]
                for idx in range(self.num_joints)]
        else:
            self.q_opt = [opt[idx::3 * self.num_joints] for idx in range(self.num_joints)]
            self.qd_opt = [opt[self.num_joints + idx::3 * self.num_joints] for idx in range(self.num_joints)]
            self.u_opt = [opt[self.num_joints * 2 + idx::3 * self.num_joints] for idx in range(self.num_joints)]

        # Reconstructing q_ddot
        self.qdd_opt, self.ee_opt = self.evaluate_opt()

        # Formatting the results
        self.result = { 'q': self.casadi2nparray(self.q_opt),
                        'qd': self.casadi2nparray(self.qd_opt),
                        'qdd': self.qdd_opt,
                        'u': self.casadi2nparray(self.u_opt),
                        'T': self.T_opt,
                        'ee_pos': self.ee_opt}
        return self.result

    def casadi2nparray(self, casadi_array):
        '''Convert Casadi MX vectors (optimal solutions) in numpy arrays'''
        list = [np.array(el).flatten() for el in casadi_array]
        return np.array(list)

    def evaluate_opt(self):
        '''Reconstructing all the values of q_ddot reached during the problem'''
        qdd_list = []; ee_list = []
        for idx in range(self.N):  # for every instant
            q = cs.vertcat(*[self.q_opt[idx2][idx] for idx2 in range(self.num_joints)])  # load joints opt values
            qd = cs.vertcat(*[self.qd_opt[idx2][idx] for idx2 in range(self.num_joints)])
            u = cs.vertcat(*[self.u_opt[idx2][idx] for idx2 in range(self.num_joints)])
            qdd = (self.q_ddot_val(q, qd, u)).full().flatten().tolist()  # transform to list
            qdd_list = qdd_list + qdd
            ee = (self.ee_pos(q)).full().flatten().tolist()  # transform to list
            ee_list = ee_list + ee
        qdd = np.array([qdd_list[idx::self.num_joints] for idx in range(self.num_joints)])  # format according to joints
        ee = np.array([ee_list[idx::3] for idx in range(3)])  # format according to joints
        return qdd, ee  # refactor according to joints

    def format_trajectory(self, traj, t):
        '''Function that formats the trajectory'''
        # If the user gave the also the wanted qd
        if isinstance(traj(t)[0], list):
            traj_dot = cs.vertcat(*traj(t)[1])
            traj = cs.vertcat(*traj(t)[0])

        # If the user just gave q desired then we set qd desired to be zero
        else:
            zeros = lambda t: [0]*self.num_joints
            traj_dot = cs.vertcat(*zeros(t))
            traj = cs.vertcat(*traj(t))
        return traj, traj_dot

    def plot_results(self):
        '''Displaying the results with matplotlib'''
        joint_limits = {
            'q': (self.lower_q, self.upper_q),
            'qd': (self.lower_qd, self.upper_qd),
            'u': (self.lower_u, self.upper_u)
        }
        return show(**self.result,
                    q_limits = joint_limits,
                    steps = self.N,
                    cost_func = self.cost_func,
                    final_term = self.final_term_cost,
                    constr = self.constraints,
                    f_constr = self.final_constraints)