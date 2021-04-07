import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimal_control import URDFopt
import casadi as cs
from matplotlib import pyplot as plt

class TestCalc(unittest.TestCase):

    def setUp(self):
        # Instantiating optimizer
        self.optimizer = URDFopt()

        # Cost Functions
        def my_cost_func(q, qd, u):
            return 10 * cs.mtimes(q.T,q) + cs.mtimes(u.T,u) / 10
        self.my_cost_func = my_cost_func
        
        # Time control
        self.time_horizon = 1
        self.steps = 50

    def test_rrbot(self):
        urdf_path = "../urdf/rrbot.urdf"
        root = "link1" 
        end = "link3"

        def trajectory_target(t):
            q = [3.14, 0]
            return q
        in_cond = [0]*4

        self.optimizer.load_robot(urdf_path, root, end)
        self.optimizer.load_problem(self.my_cost_func, self.steps, in_cond, trajectory_target, time_horizon = self.time_horizon, max_iter=3)

    def test_simplecube(self):
        urdf_path = "../urdf/simplecube.urdf"
        root = "cube_base" 
        end = "Link1"

        def trajectory_target(t):
            q = [0]
            return q
        in_cond = [0.2]*2
        B = {'J1': 10}

        self.optimizer.load_robot(urdf_path, root, end, motor_inertias=B)
        self.optimizer.load_problem(self.my_cost_func, self.steps, in_cond, trajectory_target, time_horizon = self.time_horizon, max_iter=3)

    def test_simplecube_no_inertia(self):
        urdf_path = "../urdf/simplecube_weak.urdf"
        root = "cube_base" 
        end = "Link1"

        def trajectory_target(t):
            q = [0]
            return q
        in_cond = [0.2]*2

        self.optimizer.load_robot(urdf_path, root, end)
        self.optimizer.load_problem(self.my_cost_func, self.steps, in_cond, trajectory_target, time_horizon = self.time_horizon, max_iter=3)
        fig = self.optimizer.show_result()
        plt.show()


    def _test_panda(self):
        urdf_path = "../urdf/panda_no_friction.urdf"
        root = "panda_link0" 
        end = "panda_link8"

        def trajectory_target(t):
            q = [0]*7
            return q
        in_cond = [0.2]*14

        self.optimizer.load_robot(urdf_path, root, end)
        self.optimizer.load_problem(self.my_cost_func, self.steps, in_cond, trajectory_target, time_horizon = self.time_horizon, max_iter=2)

    def test_constraints(self):
        urdf_path = "../urdf/rrbot.urdf"
        root = "link1" 
        end = "link3"

        def trajectory_target(t):
            q = [3.14, 0]
            return q
        in_cond = [0]*4

        # Additional Constraints
        def my_constraint1(q, q_dot, u, ee_pos):
            return [-10, -10], u, [10, 10]
        def my_constraint2(q, q_dot, u, ee_pos):
            return [-4, -4], q_dot, [4, 4]
        def my_constraint3(q, q_dot, u, ee_pos):
            return 0, ee_pos[0]**2 + ee_pos[1]**2 + ee_pos[2]**2, 20
        my_constraints=[my_constraint1, my_constraint2, my_constraint3]
        
        def my_final_constraint1(q, q_dot, u, ee_pos):
            return [1, 1], q, [1, 1]
        def my_final_constraint2(q, q_dot, u, ee_pos):
            return [0.757324, 0.2, 2.43627], ee_pos, [0.757324, 0.2, 2.43627]
        my_final_constraints = [my_final_constraint1]

        self.optimizer.load_robot(urdf_path, root, end)
        self.optimizer.load_problem(self.my_cost_func,
                                    self.steps,
                                    in_cond,
                                    trajectory_target,
                                    time_horizon = self.time_horizon,
                                    my_constraint = my_constraints,
                                    my_final_constraint=my_final_constraints,
                                    max_iter=3)

    def test_timehorizon(self):
        urdf_path = "../urdf/rrbot.urdf"
        root = "link1" 
        end = "link3"

        def trajectory_target(t):
            q = [3.14, 0]
            return q
        in_cond = [0]*4

        self.optimizer.load_robot(urdf_path, root, end)
        self.optimizer.load_problem(self.my_cost_func, self.steps, in_cond, trajectory_target, max_iter=3)





if __name__ == '__main__':
    unittest.main(buffer=True)
