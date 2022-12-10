# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import sys

from typing import Any, Sequence, Tuple

import numpy as np
import pybullet as p  # pytype: disable=import-error
from scipy.spatial.transform import Rotation

try:
  from mpc_controller import gait_generator as gait_generator_lib
  from mpc_controller import leg_controller
except:  #pylint: disable=W0702
  print("You need to install motion_imitation")
  print("Either run python3 setup.py install --user in this repo")
  print("or use pip3 install motion_imitation --user")
  sys.exit()

try:
  import mpc_osqp as convex_mpc  # pytype: disable=import-error
  import stoch
  import qpsolvers
except:  #pylint: disable=W0702
  print("You need to install motion_imitation")
  print("Either run python3 setup.py install --user in this repo")
  print("or use pip3 install motion_imitation --user")
  sys.exit()

_FORCE_DIMENSION = 3
# The QP weights in the convex MPC formulation. See the MIT paper for details:
#   https://ieeexplore.ieee.org/document/8594448/
# Intuitively, this is the weights of each state dimension when tracking a
# desired CoM trajectory. The full CoM state is represented by
# (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder).
# _MPC_WEIGHTS = (5, 5, 0.2, 0, 0, 10, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0)
# This worked well for in-place stepping in the real robot.
# _MPC_WEIGHTS = (5, 5, 0.2, 0, 0, 10, 0., 0., 0.2, 1., 1., 0., 0)
_MPC_WEIGHTS = (5, 5, 0.2, 0, 0, 10, 0., 0., 1., 1., 1., 0., 0)
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
      qp_solver = convex_mpc.QPOASES,
      mpc_method = 'rf'
  ):
    """Initializes the class.

    Tracks the desired position/velocity of the robot by computing proper joint
    torques using MPC module.

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
    body_inertia_list = list(body_inertia)
    weights_list = list(_MPC_WEIGHTS)
    self._cpp_mpc = convex_mpc.ConvexMpc(
        body_mass,
        body_inertia_list,
        self._num_legs,
        _PLANNING_HORIZON_STEPS,
        _PLANNING_TIMESTEP,
        weights_list,
        1e-5,
        qp_solver   
    )
    self._mpc_method = mpc_method
    self.Ut = np.zeros((self._num_legs*3, 1), dtype=np.float64)
    self.Ut[[2,5,8,11], :] = np.array([self._body_mass*9.8/4 for _ in range(self._num_legs)]).reshape(-1, 1)
    self._step_mpc = 0
    self._cvx_freq = 1
    self.Ud = np.zeros((12, _PLANNING_HORIZON_STEPS), dtype=np.float64)

  def reset(self, current_time):
    del current_time

  def update(self, current_time):
    del current_time

  def get_action(self) -> Any:
    if self._mpc_method == 'cvx':
      return self.get_action_cvx()
    elif self._mpc_method == 'rf':
      return self.get_action_rf()
    else:
      print('ERROR: MPC Method invalid')
      raise NotImplementedError

  def get_action_rf(self):
    euler_to_rot_mat = lambda rpy: Rotation.from_euler('xyz', rpy).as_matrix()
    euler_to_rot_flat = lambda rpy: euler_to_rot_mat(rpy).reshape((rpy.shape[0], 9), order='F')

    ############### Current Terms ################
    # We use the body yaw aligned world frame for MPC computation.
    com_roll_pitch_yaw = np.array(self._robot.GetBaseRollPitchYaw(), dtype=np.float64)
    com_roll_pitch_yaw[2] = 0
    com_rotation_matrix = Rotation.from_euler('xyz', com_roll_pitch_yaw).as_matrix()

    # com_position = self._robot.GetBasePosition()

    com_velocity = np.asarray(self._state_estimator.com_velocity_body_frame, dtype=np.float64)
    # Angular velocity in the yaw aligned world frame is actually different
    # from rpy rate. We use it here as a simple approximation.
    com_angular_velocity = np.asarray(self._robot.GetBaseRollPitchYawRate(), dtype=np.float64)
    #####################################################################

    # ############# Desired Terms ################
    desired_com_velocity = np.array(
        (self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)

    desired_com_position = np.array((0., 0., self._desired_body_height), dtype=np.float64)
    desired_com_position[0:2] += desired_com_velocity[0:2]*_PLANNING_HORIZON_STEPS*_PLANNING_TIMESTEP

    desired_com_angular_velocity = np.array(
        (0., 0., self.desired_twisting_speed), dtype=np.float64)
    desired_com_roll_pitch_yaw = np.array((0., 0., com_roll_pitch_yaw[2]), dtype=np.float64)
    desired_com_roll_pitch_yaw[2] += desired_com_angular_velocity[2]*_PLANNING_HORIZON_STEPS*_PLANNING_TIMESTEP
    desired_rotation_matrix = euler_to_rot_mat(desired_com_roll_pitch_yaw)

    foot_contact_state = np.array(
        [(leg_state in (gait_generator_lib.LegState.STANCE,
                        gait_generator_lib.LegState.EARLY_CONTACT))
         for leg_state in self._gait_generator.desired_leg_state],
        dtype=np.int32)
    # ##############################################

    # ######### Reference Trajectory Generation ###########
    Xd = np.zeros((18, _PLANNING_HORIZON_STEPS), dtype=np.float64)
    Xd[0:3, :] = np.linspace((0, 0, self._desired_body_height), desired_com_position, _PLANNING_HORIZON_STEPS).T
    Xd[3:6, :] = np.linspace(desired_com_velocity, desired_com_velocity, _PLANNING_HORIZON_STEPS).T
    Xd[6:15, :] = euler_to_rot_flat(np.linspace((0., 0., com_roll_pitch_yaw[2]), desired_com_roll_pitch_yaw, _PLANNING_HORIZON_STEPS)).T
    Xd[15:18, :] = np.linspace(desired_com_angular_velocity, desired_com_angular_velocity, _PLANNING_HORIZON_STEPS).T
    fz_unit = foot_contact_state*self._body_mass*9.8/(foot_contact_state.sum() + 1e-6)
    self.Ud[[2,5,8,11], :] = np.array([fz_unit for _ in range(_PLANNING_HORIZON_STEPS)]).T

    # if self._step_mpc % self._cvx_freq == 0:
    #   p.submitProfileTiming("predicted_contact_forces")
    #   predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
    #       [0],  #com_position
    #       np.asarray(self._state_estimator.com_velocity_body_frame,
    #                 dtype=np.float64),  #com_velocity
    #       np.array(com_roll_pitch_yaw, dtype=np.float64),  #com_roll_pitch_yaw
    #       # Angular velocity in the yaw aligned world frame is actually different
    #       # from rpy rate. We use it here as a simple approximation.
    #       np.asarray(self._robot.GetBaseRollPitchYawRate(),
    #                 dtype=np.float64),  #com_angular_velocity
    #       foot_contact_state,  #foot_contact_states
    #       np.array(self._robot.GetFootPositionsInBaseFrame().flatten(),
    #               dtype=np.float64),  #foot_positions_base_frame
    #       self._friction_coeffs,  #foot_friction_coeffs
    #       desired_com_position,  #desired_com_position
    #       desired_com_velocity,  #desired_com_velocity
    #       desired_com_roll_pitch_yaw,  #desired_com_roll_pitch_yaw
    #       desired_com_angular_velocity  #desired_com_angular_velocity
    #   )
    #   self.Ud = -np.array(predicted_contact_forces).reshape((12, _PLANNING_HORIZON_STEPS), order='F')
    #######################################################

    r_body = self._robot.GetFootPositionsInBaseFrame().astype(np.float64).T
    # print(r_body)
    com_z = (foot_contact_state*np.abs(r_body[2,:])).flatten().mean()
    # print(com_z)
    Xt = np.array([
            0, # com position x
            0, # com position y
            com_z,
            com_velocity[0],
            com_velocity[1],
            com_velocity[2],
            com_rotation_matrix[0,0], 
            com_rotation_matrix[1,0], 
            com_rotation_matrix[2,0], 
            com_rotation_matrix[0,1], 
            com_rotation_matrix[1,1], 
            com_rotation_matrix[2,1], 
            com_rotation_matrix[0,2], 
            com_rotation_matrix[1,2], 
            com_rotation_matrix[2,2],
            com_angular_velocity[0],
            com_angular_velocity[1],
            com_angular_velocity[2],
        ], dtype=np.float64).reshape(18,1)

    self.Ut = self.Ud[:,0].reshape((-1,1), order='F').astype(np.float64)
    # import time
    # time.sleep(40e-2)
    # print(r_body)
    matrices_dict = {}
    stoch.get_qp_form_eta(Xt, self.Ut, r_body, Xd, self.Ud, matrices_dict)
    res = qpsolvers.solve_qp(
      P=matrices_dict['H'], q=matrices_dict['g'].squeeze(), 
      G=matrices_dict['Aineq'], h=matrices_dict['bineq'].squeeze(), 
      A=matrices_dict['Aeq'], b=matrices_dict['beq'].squeeze(), solver="osqp")
    self.Ut += np.reshape(res[:12], (12,1), order='F')
    predicted_contact_forces = -(self.Ut).flatten(order='F')
    p.submitProfileTiming()

    contact_forces = {}
    for i in range(self._num_legs):
      contact_forces[i] = np.array(
          predicted_contact_forces[i * _FORCE_DIMENSION:(i + 1) * _FORCE_DIMENSION])
    # print(Xt)
    # print(Xd)
    # print(Ud)
    print('dU ', np.reshape(res[:12], (12,1), order='F').flatten())
    # print('X last ', np.reshape(res[-12:], (12,1), order='F').flatten())
    print(contact_forces)
    # exit()
    action = {}
    for leg_id, force in contact_forces.items():
      # While "Lose Contact" is useful in simulation, in real environment it's
      # susceptible to sensor noise. Disabling for now.
      # if self._gait_generator.leg_state[
      #     leg_id] == gait_generator_lib.LegState.LOSE_CONTACT:
      #   force = (0, 0, 0)
      motor_torques = self._robot.MapContactForceToJointTorques(leg_id, com_rotation_matrix @ force)
      for joint_id, torque in motor_torques.items():
        action[joint_id] = (0, 0, 0, 0, torque)

    self._step_mpc += 1

    return action, contact_forces



  def get_action_cvx(self):
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

    #predicted_contact_forces=[0]*self._num_legs*_FORCE_DIMENSION
    # print("Com Vel: {}".format(self._state_estimator.com_velocity_body_frame))
    # print("Com RPY: {}".format(self._robot.GetBaseRollPitchYawRate()))
    # print("Com RPY Rate: {}".format(self._robot.GetBaseRollPitchYawRate()))
    p.submitProfileTiming("predicted_contact_forces")
    # print(np.asarray(self._state_estimator.com_velocity_body_frame,
    #                dtype=np.float64))
    # print(np.array(com_roll_pitch_yaw, dtype=np.float64))
    # print(np.asarray(self._robot.GetBaseRollPitchYawRate(),
    #                dtype=np.float64))
    # print(foot_contact_state)
    # print(np.array(self._robot.GetFootPositionsInBaseFrame().flatten(),
    #              dtype=np.float64))
    # print(self._friction_coeffs)
    # print(desired_com_position)
    # print(desired_com_velocity)
    # print(desired_com_roll_pitch_yaw)
    # print(desired_com_angular_velocity)
    if self._step_mpc % self._cvx_freq == 0:
      predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
          [0],  #com_position
          np.asarray(self._state_estimator.com_velocity_body_frame,
                    dtype=np.float64),  #com_velocity
          np.array(com_roll_pitch_yaw, dtype=np.float64),  #com_roll_pitch_yaw
          # Angular velocity in the yaw aligned world frame is actually different
          # from rpy rate. We use it here as a simple approximation.
          np.asarray(self._robot.GetBaseRollPitchYawRate(),
                    dtype=np.float64),  #com_angular_velocity
          foot_contact_state,  #foot_contact_states
          np.array(self._robot.GetFootPositionsInBaseFrame().flatten(),
                  dtype=np.float64),  #foot_positions_base_frame
          self._friction_coeffs,  #foot_friction_coeffs
          desired_com_position,  #desired_com_position
          desired_com_velocity,  #desired_com_velocity
          desired_com_roll_pitch_yaw,  #desired_com_roll_pitch_yaw
          desired_com_angular_velocity  #desired_com_angular_velocity
      )
      self.Ud = np.array(predicted_contact_forces).reshape((12, _PLANNING_HORIZON_STEPS), order='F')

    # sol = np.array(predicted_contact_forces).reshape((-1, 12))
    # x_dim = np.array([0, 3, 6, 9])
    # y_dim = x_dim + 1
    # z_dim = y_dim + 1
    # print("Y_forces: {}".format(sol[:, y_dim]))
    self.Ut = self.Ud[:,0].reshape((-1,1), order='F').astype(np.float64)
    predicted_contact_forces = (self.Ut).flatten(order='F')
    p.submitProfileTiming()
    contact_forces = {}
    for i in range(self._num_legs):
      contact_forces[i] = np.array(
          predicted_contact_forces[i * _FORCE_DIMENSION:(i + 1) *
                                   _FORCE_DIMENSION])
    # print(contact_forces)
    # exit()
    action = {}
    for leg_id, force in contact_forces.items():
      # While "Lose Contact" is useful in simulation, in real environment it's
      # susceptible to sensor noise. Disabling for now.
      # if self._gait_generator.leg_state[
      #     leg_id] == gait_generator_lib.LegState.LOSE_CONTACT:
      #   force = (0, 0, 0)
      motor_torques = self._robot.MapContactForceToJointTorques(leg_id, force)
      for joint_id, torque in motor_torques.items():
        action[joint_id] = (0, 0, 0, 0, torque)

    self._step_mpc += 1

    return action, contact_forces
