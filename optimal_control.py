import casadi as cs
from urdf2casadi import urdfparser as u2c
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


class Urdf2Moon:
    def __init__(self, urdf_path, root, tip):
        self.robot_parser = self.load_urdf(urdf_path)

        # Store inputs
        self.root = root
        self.tip = tip
        
        # Get basic info
        self.num_joints = self.get_joints_n(self.root, self.tip)
        self.q, self.q_dot, self.u_hat = self.define_symbolic_vars(self.num_joints)
        self.M, self.Cq, self.G = self.get_motion_equation_matrix(self.root, self.tip, self.q, self.q_dot)
        self.M_inv = cs.pinv(self.M)
        self.u = self.u_hat + self.G
        self.upper_x, self.lower_x = self.get_limits(self.root, self.tip)

    def solve(self, cost_func, time_horizon, control_steps, initial_cond, trajectory_target, final_term_cost=None, upper_u = None, lower_u=None, rk_interval=4):
        #Store Values
        self.cost_func = cost_func
        self.final_term_cost = final_term_cost
        self.T = time_horizon
        self.N = control_steps
        self.initial_cond = initial_cond
        self.traj = trajectory_target
        self.rk_intervals = rk_interval

        # Fix upper values if not given
        if upper_u == None:
            self.upper_u = [float('inf')] * self.num_joints
            self.lower_u = [-float('inf')] * self.num_joints
        else:
            self.upper_u = upper_u
            self.lower_u = lower_u

        # Working it up
        self.f = self.get_diff_eq(self.cost_func, self.traj)
        self.F = self.rk4(self.f, self.T, self.N, self.rk_intervals)
        self.nlp_solver(self.initial_cond, self.final_term_cost, 
                        self.upper_x, self.lower_x, self.upper_u, self.lower_u, self.traj, self.traj_dot)
        self.u_opt = self.u_function(self.u_hat_opt, self.x_opt[0:self.num_joints])
        return {'q': self.x_opt, 'qd': self.x_dot_opt, 
                'u': self.u_opt, 'u_hat': self.u_hat_opt}

    def define_symbolic_vars(self, num_joints):
        q = cs.SX.sym("q", num_joints)
        q_dot = cs.SX.sym("q_dot", num_joints)
        u_hat = cs.SX.sym("u_hat", num_joints) 
        return q, q_dot, u_hat
    def load_urdf(self, urdf_path):
        robot_parser = u2c.URDFparser()
        robot_parser.from_file(urdf_path)
        return robot_parser
    def get_joints_n(self, root, tip):
        return self.robot_parser.get_n_joints(root, tip) #return the number of actuated joints
    def get_limits(self, root, tip):
        _, _, upper, lower = self.robot_parser.get_joint_info(root, tip)
        return upper, lower
    def get_motion_equation_matrix(self, root, tip, q, q_dot):
        # load inertia terms (function)
        M_sym = self.robot_parser.get_inertia_matrix_crba(root, tip)
        # load gravity terms (function)
        gravity_u2c = [0, 0, -9.81]
        self.G_sym = self.robot_parser.get_gravity_rnea(root, tip, gravity_u2c)
        # load Coriolis terms (function)
        C_sym = self.robot_parser.get_coriolis_rnea(root, tip)
        return M_sym(q), C_sym(q, q_dot), self.G_sym(q)

    def get_diff_eq(self, cost_func, traj):
        self.t = cs.SX.sym("t", 1) #let's define time

        lhs1 = self.q_dot
        lhs2 = -cs.mtimes(self.M_inv, self.Cq) + cs.mtimes(self.M_inv, self.u_hat)
        
        self.tr, self.tr_d, self.traj_dot = self.derive_trajectory(traj, self.t)
        J_dot = cost_func(self.q-self.tr, self.q_dot-self.tr_d, self.u)
        
        self.x = cs.vertcat(self.q, self.q_dot)
        self.x_dot = cs.vertcat(lhs1, lhs2)
        f = cs.Function('f', [self.x, self.u_hat, self.t],    # inputs
                             [self.x_dot, J_dot])  # outputs
        return f
    def rk4(self, f, T, N, m):
        dt = T/N/m

        # variable definition for RK method
        X0 = cs.MX.sym('X0', self.num_joints * 2)
        U = cs.MX.sym('U', self.num_joints)
        t = cs.MX.sym('t', 1)

        # Initial value
        X = X0
        Q = 0

        # Integration
        for j in range(m):
            k1, k1_q = f(X,  U,  t)
            k2, k2_q = f(X + dt/2 * k1, U,  t)
            k3, k3_q = f(X + dt/2 * k2, U,  t)
            k4, k4_q = f(X + dt * k3, U,  t)
            # update the state
            X = X + dt/6*(k1 +2*k2 +2*k3 +k4)    
            # update the cost function 
            Q = Q + dt/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)

        F = cs.Function('F', [X0, U, t], [X, Q],['x0','p', 'time'],['xf','qf'])
        return F
    def nlp_solver(self, initial_cond, final_term_cost, upper_x, lower_x, upper_u, lower_u, traj, traj_dot):
        # Start with an empty NLP
        w       = []    #input vector
        w_g     = []    #initial guess
        lbw     = []    #lower bounds of inputs
        ubw     = []    #upper bounds of inputs 
        J       = 0     #initial value of cost func
        g       = []    #joint state for all timesteps
        lbg     = []    #lower bounds of states
        ubg     = []    #upper bound of states


        Xk = cs.MX.sym('X0', self.num_joints*2) # MUST be coherent with the condition specified abow
        w += [Xk]
        w_g  += initial_cond
        lbw += initial_cond
        ubw += initial_cond
        # Integration!
        dt = self.T/self.N
        for k in range(self.N):
            # New NLP variable for the control
            Uk = cs.MX.sym('U_' + str(k), self.num_joints) # generate the k-th control command, 2x1 dimension
            w += [Uk]        # list of commands [U_0, U_1, ..., U_(N-1)]
            w_g += [0] * self.num_joints  # initial guess
            
            # Add inequality constraint on inputs
            lbw += lower_u       # lower bound on q
            ubw += upper_u       # upper bound on q
            
            # Integrate till the end of the interval
            Fk = self.F(x0=Xk, p=Uk, time=dt*k)     #That's the actual integration!
            Xk_end = Fk['xf']
            J  = J+Fk['qf']
    
            # New NLP variable for state at end of interval
            Xk = cs.MX.sym('X_' + str(k+1), 2*self.num_joints)
            w  += [Xk]
            w_g += [0] * (2*self.num_joints) # initial guess
            
            # Add inequality constraint on inputs
            lbw += lower_x       # lower bound on q
            lbw += [-cs.inf] * self.num_joints #lower bound on q_dot
            ubw += upper_x       # upper bound on q
            ubw += [cs.inf]  * self.num_joints # upper bound on q_dot

            # Add equality constraint
            g   += [Xk_end-Xk]
            lbg += [0] * (2*self.num_joints)
            ubg += [0] * (2*self.num_joints)
    
        # add a final term cost. If not specified is 0.
        if final_term_cost != None:
            J = J+final_term_cost(Xk_end[0:self.num_joints]-traj(self.T), Xk_end[self.num_joints:]-traj_dot(self.T), Uk) # f_t_c(q, qd, u)    
            
        # Define the problem to be solved
        problem = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
        # NLP solver options
        opts = {}
        opts["ipopt"] = {'max_iter': 100}
        # Define the solver and add boundary conditions
        solver = cs.nlpsol('solver', 'ipopt', problem, opts)
        solver = solver(x0=w_g, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # Solve the NLP
        opt = solver['x']
        self.x_opt = [opt[idx::3*self.num_joints]for idx in range(self.num_joints)]
        self.x_dot_opt = [opt[self.num_joints+idx::3*self.num_joints]for idx in range(self.num_joints)]
        self.u_hat_opt = [opt[self.num_joints*2+idx::3*self.num_joints]for idx in range(self.num_joints)]

    
    def derive_trajectory(self, traj, t):
        traj_dot_list = [cs.jacobian(traj(t)[idx],t) for idx in range(self.num_joints)]
        traj_sx = cs.vertcat(*traj(t))
        traj_dot_sx = cs.vertcat(*traj_dot_list)
        traj_dot = cs.Function('traj_dot', [t], [traj_dot_sx], ['t'], ['traj_dot'])
        return traj_sx, traj_dot_sx, traj_dot
    def u_function(self, u_hat_opt, x_opt): # function that returns the value of u = u_hat + G(q)
        g_opt = []
        for k in range(self.N): # for every simulation instant
            x_opt_list = [x_opt[idx][k] for idx in range(self.num_joints)] # list q0, q1, ..., q_n at k-th instant
            x_t = cs.vertcat(*x_opt_list) # 
            gk = self.G_sym(x_t) # evaluate G(q) at k-th instant
            g_opt += [gk.full()] # save G(q) in a list like [G0(0) G1(0) G0(1) G1(1)]
            
        g_opt_flat = [item for sublist in g_opt for item in sublist]
        g_opt_list = [g_opt_flat[idx::self.num_joints]for idx in range(self.num_joints)]
        return g_opt_list + u_hat_opt

    def print_results(self):
        tgrid = [self.T/self.N*k for k in range(self.N+1)]

        fig, axes = plt.subplots(nrows=ceil(self.num_joints/2), ncols=2, figsize=(15, 4*ceil(self.num_joints/2)))
        
        for idx, ax in enumerate(fig.axes):
            self.get_ax(ax, idx, tgrid)

        return fig    
    def get_ax(self, ax, idx, tgrid):
        n = self.num_joints
        ax.plot(tgrid, self.x_opt[idx], '--')
        ax.plot(tgrid, self.x_dot_opt[idx], '--')
        ax.plot(tgrid[1:], self.u_hat_opt[idx], '-.')
        ax.plot(tgrid[1:], self.u_opt[idx], '-.')
        ax.legend(['q'+str(idx),'q' + str(idx) +'_dot', 'u' + str(idx)+'_hat', 'u' + str(idx)])
        return ax
        

if __name__ == '__main__':
    urdf_path = "urdf/rrbot.urdf"
    root = "link1" 
    end = "link3"

    def my_cost_func(q, qd, u):
        return 100*cs.mtimes(q.T,q) + cs.mtimes(qd.T,qd) + cs.mtimes(u.T,u)/10

    def my_final_term_cost(q_f, qd_f, u_f):
        return 10*cs.mtimes(q_f.T,q_f)  #+ cs.mtimes(qd_f.T,qd_f)
    
    def trajectory_target(t):
        q = [0.1, 0.1]
        q = [1*t, -1*t]
        q = [cs.cos(t), cs.sin(t)]
        return q

    time_horizon = 10
    steps = 400
    in_cond = [1.1,0.1,0.1,1.1]

    urdf_2_opt = Urdf2Moon(urdf_path, root, end)
    opt = urdf_2_opt.solve(my_cost_func, time_horizon, steps, in_cond, trajectory_target, my_final_term_cost)
    fig = urdf_2_opt.print_results()
