"""A torque based stance controller framework using crocoddyl."""

from mpc_controller.centroidal_dynamics_diff_action_model import DifferentialActionModelCentroidal
import crocoddyl
import pybullet as p
import numpy as np
from typing import Any, Sequence, Tuple
import time
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


try:
    from mpc_controller import gait_generator as gait_generator_lib
    from mpc_controller import leg_controller
except:  # pylint: disable=W0702
    print("You need to install motion_imitation")
    print("Either run python3 setup.py install --user in this repo")
    print("or use pip3 install motion_imitation --user")
    sys.exit()

# import mpc_osqp as convex_mpc  # pytype: disable=import-error

_FORCE_DIMENSION = 3
# The QP weights in the convex MPC formulation. See the MIT paper for details:
#   https://ieeexplore.ieee.org/document/8594448/
# Intuitively, this is the weights of each state dimension when tracking a
# desired CoM trajectory. The full CoM state is represented by
# (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder).
# _MPC_WEIGHTS = (5, 5, 0.2, 0, 0, 10, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0)
# This worked well for in-place stepping in the real robot.
# _MPC_WEIGHTS = (5, 5, 0.2, 0, 0, 10, 0., 0., 0.2, 1., 1., 0., 0)
_MPC_WEIGHTS = (5, 5, 0.2, 0, 0, 10, 0., 0., 1., 1., 1., 0.)
# _MPC_WEIGHTS = (50, 50, 2, 0, 0, 100, 0., 0., 10., 10., 10., 0.)
_PLANNING_HORIZON_STEPS = 10
_PLANNING_TIMESTEP = 0.025


class TorqueStanceLegController(leg_controller.LegController):
    """A torque based stance leg controller framework.

    Takes in high level parameters like walking speed and turning speed, and
    generates necessary the torques for stance legs.
    """

    def __init__(
            self,
            robot: Any,
            gait_generator: Any,
            state_estimator: Any,
            desired_speed: Tuple[float, float] = (0, 0),
            desired_twisting_speed: float = 0,
            desired_body_height: float = 0.45,
            body_mass: float = 220 / 9.8,
            body_inertia: Tuple[float, float, float, float, float, float, float,
                                float, float] = (0.07335, 0, 0, 0, 0.25068, 0, 0, 0,
                                                 0.25447),
            num_legs: int = 4,
            friction_coeffs: Sequence[float] = (0.45, 0.45, 0.45, 0.45),
            ddp_solver='DDP'
    ):
        """Initializes the class.

        Tracks the desired position/velocity of the robot by computing proper joint
        torques using MPC module (DDP).

        Args:
          robot: A robot instance.
          gait_generator: Used to query the locomotion phase and leg states.
          state_estimator: Estimate the robot states (e.g. CoM velocity).
          desired_speed: desired CoM speed in x-y plane.
          desired_twisting_speed: desired CoM rotating speed in z direction.
          desired_body_height: The standing height of the robot.
          body_mass: The total mass of the robot.
          body_inertia: The inertia matrix in the body principle frame. We assume
                the body principle coordinate frame has x-forward and z-up.
          num_legs: The number of legs used for force planning.
          friction_coeffs: The friction coeffs on the contact surfaces.
        """
        self._robot = robot
        self._gait_generator = gait_generator
        self._state_estimator = state_estimator
        self.desired_speed = desired_speed
        self.desired_twisting_speed = desired_twisting_speed

        self._desired_body_height = desired_body_height
        self._body_mass = body_mass
        self._num_legs = num_legs
        self._friction_coeffs = np.array(friction_coeffs)
        self._body_inertia_list = list(body_inertia)
        self._weights_list = list(_MPC_WEIGHTS)

        self.ddp_solver = ddp_solver
        # self._cpp_mpc = convex_mpc.ConvexMpc(
        # 	body_mass,
        # 	body_inertia_list,
        # 	self._num_legs,
        # 	_PLANNING_HORIZON_STEPS,
        # 	_PLANNING_TIMESTEP,
        # 	weights_list,
        # 	1e-5,
        # 	qp_solver

        # )

    def reset(self, current_time):
        del current_time

    def update(self, current_time):
        del current_time

    def get_action(self):
        """Computes the torque for stance legs."""
        desired_com_position = np.array((0., 0., self._desired_body_height),
                                        dtype=np.float64)
        desired_com_velocity = np.array(
            (self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)
        desired_com_roll_pitch_yaw = np.array((0., 0., 0.), dtype=np.float64)
        desired_com_angular_velocity = np.array(
            (0., 0., self.desired_twisting_speed), dtype=np.float64)
        foot_contact_state = np.array(
            [(leg_state in (gait_generator_lib.LegState.STANCE,
                            gait_generator_lib.LegState.EARLY_CONTACT))
             for leg_state in self._gait_generator.desired_leg_state],
            dtype=np.int32)

        # We use the body yaw aligned world frame for MPC computation.
        com_roll_pitch_yaw = np.array(self._robot.GetBaseRollPitchYaw(),
                                      dtype=np.float64)
        com_roll_pitch_yaw[2] = 0
        com_position = np.asarray(self._robot.GetTrueBasePosition())
        com_position[0] = 0
        com_position[1] = 0
        com_angular_velocity = np.asarray(self._robot.GetBaseRollPitchYawRate(), dtype=np.float64)
        com_velocity = np.asarray(self._state_estimator.com_velocity_body_frame, dtype=np.float64)
        # com_velocity = np.asarray(self._robot.GetBaseVelocity())

        foot_position = np.array(self._robot.GetFootPositionsInBaseFrame(), dtype=np.float64)

        p.submitProfileTiming("predicted_contact_forces")
        com_quadDAM = DifferentialActionModelCentroidal(
            mass=self._body_mass,
            inertia=np.array(self._body_inertia_list).reshape((3, 3)),
            costWeights=np.array(_MPC_WEIGHTS),
            timeStep=_PLANNING_TIMESTEP,
            horizon=_PLANNING_HORIZON_STEPS)
        com_quadDAM.prepareModel(com_position, com_velocity, com_roll_pitch_yaw, com_angular_velocity, 
            foot_contact_state, foot_position, 
            desired_com_position, desired_com_velocity, desired_com_roll_pitch_yaw, desired_com_angular_velocity)

        # Using NumDiff for computing the derivatives. We specify the
        # withGaussApprox=True to have approximation of the Hessian based on the
        # Jacobian of the cost residuals.
        com_quadND = crocoddyl.DifferentialActionModelNumDiff(
            com_quadDAM, True)
        # Getting the IAM using the simpletic Euler rule
        timeStep = _PLANNING_TIMESTEP
        com_quadIAM = crocoddyl.IntegratedActionModelEuler(
            com_quadND, timeStep)

        T = _PLANNING_HORIZON_STEPS
        problem = crocoddyl.ShootingProblem(com_quadDAM.x0, [com_quadIAM] * T, com_quadIAM)
        
        if self.ddp_solver == "DDP":
            ddp = crocoddyl.SolverDDP(problem)
        else:
            ddp = crocoddyl.SolverFDDP(problem)
        ddp.setCallbacks([crocoddyl.CallbackVerbose()])
        
        ddp.solve([], [], 20)
        predicted_contact_forces = -ddp.us[0]

        print("predicted_contact_forces: {}".format(predicted_contact_forces))

        p.submitProfileTiming()
        # sol = np.array(predicted_contact_forces).reshape((-1, 12))
        # x_dim = np.array([0, 3, 6, 9])
        # y_dim = x_dim + 1
        # z_dim = y_dim + 1
        # print("Y_forces: {}".format(sol[:, y_dim]))

        contact_forces = {}
        for i in range(self._num_legs):
            contact_forces[i] = foot_contact_state[i] * np.array(
                predicted_contact_forces[i * _FORCE_DIMENSION:(i + 1) *
                                         _FORCE_DIMENSION])
        print("contact_forces: {}".format(contact_forces))
        print(com_quadDAM.x0)
        print(com_quadDAM.xtarget)
        # exit()
        action = {}
        for leg_id, force in contact_forces.items():
            # While "Lose Contact" is useful in simulation, in real environment it's
            # susceptible to sensor noise. Disabling for now.
            # if self._gait_generator.leg_state[
            #     leg_id] == gait_generator_lib.LegState.LOSE_CONTACT:
            #   force = (0, 0, 0)
            motor_torques = self._robot.MapContactForceToJointTorques(
                leg_id, force)
            for joint_id, torque in motor_torques.items():
                action[joint_id] = (0, 0, 0, 0, torque)

        return action, contact_forces
