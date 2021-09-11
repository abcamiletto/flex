from .problem import Problem
from .robot import Robot

class Optimizer:
    '''Helper class for the end user'''

    def load_robot(self, urdf_path, root, tip, **kwargs):
        '''Loading robot info from URDF'''
        self.robot = Robot(urdf_path, root, tip, **kwargs)

    def load_problem(self, cost_func, control_steps, initial_cond, **kwargs):
        '''Loading the problem settings'''
        self.problem = Problem(self.robot, cost_func, control_steps, initial_cond, **kwargs)

    def solve(self):
        '''Actually solving it'''
        result = self.problem.solve_nlp()
        return result

    def plot_result(self):
        '''Display the results with matplotlib'''
        return self.problem.plot_results()


optimizer = Optimizer()
