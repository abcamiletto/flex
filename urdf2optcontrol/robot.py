import casadi as cs
from urdf2casadi import urdfparser as u2c
import numpy as np
import xml.etree.ElementTree as ET

class Robot:
    '''Class that handles the loading process from the URDF'''
    def __init__(self, urdf_path, root, tip, get_motor_dynamics=True):
        '''Takes the minimum info required to start the dynamics matrix calculations.
        Motor inertias and SEA dampings can be given manually, or will be automatically parsed'''

        # Create a parser
        self.robot_parser = self._load_urdf(urdf_path)

        # Store inputs
        self.root = root
        self.tip = tip

        # Retrieve basic info
        self.num_joints = self.get_joints_n()
        self._define_symbolic_vars()
        self.M, self.Cq, self.G = self._get_motion_equation_matrix()
        self.M_inv = cs.pinv(self.M)
        self.upper_q, self.lower_q, self.max_effort, self.max_velocity = self.get_limits()
        self.ee_pos = self._get_forward_kinematics()

        # Fix boundaries if not given
        self.upper_u, self.lower_u = self._fix_boundaries(self.max_effort)
        self.upper_qd, self.lower_qd = self._fix_boundaries(self.max_velocity)

        # SEA Stuff
        self.sea = self.get_joints_plugin(urdf_path, 'spring_k', 0, diag=True).any()  # True if at least one joint is sea
        if self.sea:
            self.get_seaplugin_values(urdf_path, get_motor_dynamics)

    @staticmethod
    def _load_urdf(urdf_path: str):
        '''Function that creates a parser and load the path'''
        robot_parser = u2c.URDFparser()
        robot_parser.from_file(urdf_path)
        return robot_parser

    def get_joints_n(self) -> int:
        '''Return the number of joints'''
        return self.robot_parser.get_n_joints(self.root, self.tip)

    def _define_symbolic_vars(self) -> None:
        '''Instantiating the main symbolic variables'''
        self.q = cs.SX.sym("q", self.num_joints)
        self.q_dot = cs.SX.sym("q_dot", self.num_joints)
        self.u = cs.SX.sym("u", self.num_joints)
        self.ee = cs.SX.sym("ee", 1)
        self.q_ddot = cs.SX.sym("q_ddot", self.num_joints)

    def _get_motion_equation_matrix(self):
        '''Function that returns all the matrix in the motion equation, already linked to our symbolic vars'''
        # Load inertia terms (function)
        self.M_sym = self.robot_parser.get_inertia_matrix_crba(self.root, self.tip)
        # Load gravity terms (function)
        g = [0, 0, -9.81]
        self.G_sym = self.robot_parser.get_gravity_rnea(self.root, self.tip, g)
        # Load Coriolis terms (function)
        self.C_sym = self.robot_parser.get_coriolis_rnea(self.root, self.tip)
        # Load frictional matrices
        self.Ff, self.Fd = self.robot_parser.get_friction_matrices(self.root, self.tip)

        return self.M_sym(self.q), self.C_sym(self.q, self.q_dot), self.G_sym(self.q)

    def get_limits(self) -> list:
        '''Function that returns all the limits stored in the URDF'''
        _, self.actuated_list, upper_q, lower_q = self.robot_parser.get_joint_info(self.root, self.tip)
        max_effort, max_velocity = self.robot_parser.get_other_limits(self.root, self.tip)
        return upper_q, lower_q, max_effort, max_velocity
    
    def get_limits_plugin(self, urdf_path) -> list:
        '''Function that returns all the limits stored in the plugin of URDF'''
        upper_theta = self.get_joints_plugin(urdf_path, 'mot_maxPos', float('inf'))
        lower_theta = self.get_joints_plugin(urdf_path, 'mot_minPos', -float('inf'))
        max_effort_theta = self.get_joints_plugin(urdf_path, 'mot_tauMax', float('inf'))
        max_velocity_theta = self.get_joints_plugin(urdf_path, 'mot_maxVel', float('inf'))
        return upper_theta, lower_theta, max_effort_theta, max_velocity_theta

    def get_seaplugin_values(self, urdf_path, get_motor_dynamics):
        self.K = self.get_joints_plugin(urdf_path, 'spring_k', 0, diag=True)
        if get_motor_dynamics == True:  # if user wants to consider motor inertias
            print('Flex* is loading motor dynamics...')
            self.B = self.get_joints_plugin(urdf_path, 'mot_J', 0, diag=True)
            self.FDsea = self.get_joints_plugin(urdf_path, 'mot_D', 0, diag=True)
            self.FMusea = self.get_joints_plugin(urdf_path, 'mot_tauFric', 0, diag=True)
            self.SEAvars()
            self.SEAdynamics = self.B.any()  # True when we're modeling also motor inertia != 0
        else:
            print('Flex* is not condidering motor dynamics!')
            print('So, in the cost function, the control term is u=(theta-q) !')
            self.SEAdynamics = False
        self.upper_theta, self.lower_theta, self.max_effort_theta, self.max_velocity_theta = self.get_limits_plugin(
            urdf_path)
        if self.SEAdynamics:
            self.upper_u, self.lower_u = self._fix_boundaries(self.max_effort_theta)
            self.upper_thetad, self.lower_thetad = self._fix_boundaries(self.max_velocity_theta)
        else:
            # if not considering motor inertia, theta is system control
            self.upper_u, self.lower_u = self.upper_theta, self.lower_theta

    def _get_forward_kinematics(self):
        '''Return a CASaDi function that takes the joints position as input and returns the end effector position'''
        fk_dict = self.robot_parser.get_forward_kinematics(self.root, self.tip)
        dummy_sym = cs.MX.sym('dummy', self.num_joints)
        FK_sym = fk_dict["T_fk"]  # ee position
        ee = FK_sym(dummy_sym)[0:3, 3]
        ee_pos = cs.Function('ee_pos', [dummy_sym], [ee])
        return ee_pos

    def _fix_boundaries(self, item):
        '''Function that check whether the limits were set in the URDF, and if not, set them properly'''
        if item is None:
            # If they weren't set at all
            upper = [float('inf')] * self.num_joints
            lower = [-float('inf')] * self.num_joints

        elif isinstance(item, list):
            # If we have a list (like max velocities), we mirror it
            if len(item) == self.num_joints:
                upper = item
                lower = [-x for x in item]
            else:
                raise ValueError(
                    'Boundaries lenght does not match the number of joints! It should be long {}'.format(self.num_joints))

        elif isinstance(item, (int, float)):
            # If we get a single number for all the joints
            upper = [item] * self.num_joints
            lower = [-item] * self.num_joints

        else:
            # If we receive a str or anything stranger
            raise ValueError('Input should be a number or a list of numbers')

        return upper, lower

    # SEA Functions
    def SEAvars(self):
        '''Instantiating the SEA additional symbolic variables'''
        self.theta = cs.SX.sym("theta", self.num_joints)
        self.theta_dot = cs.SX.sym("theta_dot", self.num_joints)
        self.tau_sea = self.K @ (self.q - self.theta)

    def get_joints_plugin(self, urdf_path, key_word, value_if_not_found, diag=False):
        '''Manually retriving Gazebo Plugin info from the URDF'''
        tree = ET.parse(urdf_path)
        results = {}
        for gazebo in tree.findall('gazebo/plugin'):
            try:
                results[gazebo.find('joint').text] = float(gazebo.find(key_word).text)
            except:
                pass

        in_list = [results.get(joint, value_if_not_found) for joint in self.actuated_list]
        if diag == True:
            in_list = np.diag(in_list)
        return in_list

    def __str__(self):
        '''Summary of the info loaded into the robot, for an easier debugging'''
        strRobot = ('      ROBOT DESCRIPTION' "\n"
                    f'The number of joints is {self.num_joints}' "\n"
                    f'The first one is {self.root}, the last one {self.tip}' "\n"
                    f'Lower Bounds on q : {self.lower_q}' "\n"
                    f'Upper Bounds on q : {self.upper_q}' "\n"
                    f'Max Velocity : {self.max_velocity}' "\n"
                    f'Max Effort : {self.max_effort}' "\n")
        if self.sea:
            strRobot += f"Stiffness Matrix : {self.K}"
        else:
            strRobot += "There are no elastic joints"

        return strRobot
