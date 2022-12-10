import crocoddyl
import numpy as np
import pinocchio

def ConvertToSkewSymmetric(x: np.ndarray):
    return np.array([[   0, -x[2],  x[1]],
                    [ x[2],     0, -x[0]],
                    [-x[1],  x[0],    0]])


class DifferentialActionModelCentroidal(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, mass, inertia, costWeights, timeStep, horizon):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(12), 12, 12)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)
        
        self.mass = mass
        self.inertia = inertia
        
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.inv_mass = 1.0 / self.mass

        self.g = -9.81
        self.gravity_vector = np.array([0, 0, self.g])

        self.costWeights = costWeights
        self.timeStep = timeStep
        self.horizon = horizon
        # self.costWeights = [1., 1., 1.0, 0.001, 0.001, 1.0]

    def prepareModel(self, 
        com_position, com_velocity, com_roll_pitch_yaw, com_angular_velocity, 
        foot_contact_states, foot_positions_body_frame, 
        desired_com_position, desired_com_velocity, desired_com_roll_pitch_yaw, desired_com_angular_velocity):
        # Compute the foot positions in the world frame.
        self.R_body2world = pinocchio.rpy.rpyToMatrix(*com_roll_pitch_yaw)
        self.foot_positions_world_frame = np.zeros_like(foot_positions_body_frame)
        for i in range(4):            
            self.foot_positions_world_frame[i] = self.R_body2world @ foot_positions_body_frame[i]
        self.contact_condition = (foot_contact_states > 0).astype(int)
        self.foot_positions_body_frame = foot_positions_body_frame

        com_height = 0
        for i in range(4):
            if self.contact_condition[i]:
                com_height += abs(self.foot_positions_world_frame[i][2])
        if sum(self.contact_condition) > 0:
            com_position[2] = com_height / sum(self.contact_condition)


        self.calc_A_mat(com_roll_pitch_yaw)
        self.calc_B_mat()

        self.x0 = np.array([com_roll_pitch_yaw[0], com_roll_pitch_yaw[1], com_roll_pitch_yaw[2],
            com_position[0], com_position[1], com_position[2], 
            com_angular_velocity[0], com_angular_velocity[1], com_angular_velocity[2], 
            com_velocity[0], com_velocity[1], com_velocity[2]])

        self.xtarget = np.array([desired_com_roll_pitch_yaw[0], desired_com_roll_pitch_yaw[1], desired_com_roll_pitch_yaw[2],
            self.timeStep * 1 * desired_com_velocity[0] + com_position[0], 
            self.timeStep * 1 * desired_com_velocity[1] + com_position[1], 
            desired_com_position[2], 
            desired_com_angular_velocity[0], desired_com_angular_velocity[1], desired_com_angular_velocity[2], 
            desired_com_velocity[0], desired_com_velocity[1], desired_com_velocity[2]])

        # print("x0: ", self.x0)
        # print("xtarget: ", self.xtarget)

    def calc_A_mat(self, com_roll_pitch_yaw):
        A = np.zeros((13, 13))
        cos_yaw = np.cos(com_roll_pitch_yaw[2])
        sin_yaw = np.sin(com_roll_pitch_yaw[2])
        cos_pitch = np.cos(com_roll_pitch_yaw[1])
        tan_pitch = np.tan(com_roll_pitch_yaw[1])
        angular_velocity_to_rpy_rate = np.array([
            [cos_yaw / cos_pitch, sin_yaw / cos_pitch, 0],
            [-sin_yaw, cos_yaw, 0], 
            [cos_yaw* tan_pitch, sin_yaw* tan_pitch, 1]])
        A[0:3, 6:6+3] = angular_velocity_to_rpy_rate
        A[3, 9] = 1
        A[4, 10] = 1
        A[5, 11] = 1
        A[11, 12] = 1
        self.A_mat = A

    def calc_B_mat(self):
        B = np.zeros((13, 12))
        inertia_world = self.R_body2world @ self.inertia @ self.R_body2world.T
        inv_inertia_world = self.R_body2world @ self.inv_inertia @ self.R_body2world.T
        for i in range(4):
            B[6:6+3, i*3:i*3+3] = self.contact_condition[i] * inv_inertia_world * ConvertToSkewSymmetric(self.foot_positions_world_frame[i])
            B[9, i * 3] = self.contact_condition[i] * self.inv_mass
            B[10, i * 3 + 1] = self.contact_condition[i] * self.inv_mass
            B[11, i * 3 + 2] = self.contact_condition[i] * self.inv_mass
        self.B_mat = B


    def calc(self, data, x, u):
        ang_x = x[0:3]
        lin_x = x[3:6]
        ang_dx = x[6:9]
        lin_dx = x[9:12]

        # print('old: x', lin_x[0, :])
        # print('old: dx', lin_dx[0, :])

        ree_FL = self.foot_positions_world_frame[0]
        ree_FR = self.foot_positions_world_frame[1]
        ree_BL = self.foot_positions_world_frame[2]
        ree_BR = self.foot_positions_world_frame[3]

        F_FL = u[0:3]
        F_FR = u[3:6]
        F_BL = u[6:9]
        F_BR = u[9:12]

        lin_ddx = self.gravity_vector + ((F_FL + F_FR + F_BL + F_BR) / self.mass)

        inertia_world = self.R_body2world @ self.inertia @ self.R_body2world.T
        inv_inertia_world = self.R_body2world @ self.inv_inertia @ self.R_body2world.T

        # inertia_world = self.inertia
        # inv_inertia_world = self.inv_inertia

        ang_ddx = inv_inertia_world @ (
            -np.cross(ang_dx, (inertia_world @ ang_dx)) +
            np.cross(ree_FL, F_FL) +
            np.cross(ree_FR, F_FR) +
            np.cross(ree_BL, F_BL) +
            np.cross(ree_BR, F_BR))
        
        data.xout = np.concatenate([ang_ddx, lin_ddx])

        # BUG: Results in zero forces
        # print(self.B_mat)
        # print(u)
        # print(self.B_mat @ u)
        # exit()
        # data.xout = self.A_mat @ x + self.B_mat @ u


        # Computing the cost residual and value
        # TODO: Add cost model for friction, force and state
        data.r = np.matrix(self.costWeights * (self.xtarget-x)).T #+ 0.1 * (u).T
        # print("cost residual: ", data.r)
        data.cost = 0.5 * sum(np.asarray(data.r)**2).item()
        # print("cost: ", data.cost)

    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass

# def run_crocoddyl():
#     # Creating the DAM for the com_quad
#     np.random.seed(0)
#     com_quadDAM = DifferentialActionModelCentroidal(mass=0.5, xcom_0=np.zeros(12), foot_pos_0=np.random.rand(4,3), xtarget=np.random.rand(12), costWeights=np.ones(12), torso_dims=[0.2, 0.2, 0.2])
#     # Using NumDiff for computing the derivatives. We specify the
#     # withGaussApprox=True to have approximation of the Hessian based on the
#     # Jacobian of the cost residuals.
#     com_quadND = crocoddyl.DifferentialActionModelNumDiff(com_quadDAM, True)

#     # Getting the IAM using the simpletic Euler rule
#     timeStep = 1e-2
#     com_quadIAM = crocoddyl.IntegratedActionModelEuler(com_quadND, timeStep)

#     # Creating the shooting problem
#     x0 = np.random.rand(12)
#     T = 250

#     terminalcom_quad = DifferentialActionModelCentroidal(mass=0.5, xcom_0=np.zeros(12), foot_pos_0=np.random.rand(4,3), xtarget=np.random.rand(12), costWeights=np.ones(12), torso_dims=[0.2, 0.2, 0.2])
#     terminalcom_quadDAM = crocoddyl.DifferentialActionModelNumDiff(terminalcom_quad, True)
#     terminalcom_quadIAM = crocoddyl.IntegratedActionModelEuler(terminalcom_quadDAM)

#     # terminalcom_quad.costWeights[0] = 100
#     # terminalcom_quad.costWeights[1] = 100
#     # terminalcom_quad.costWeights[2] = 5.
#     # terminalcom_quad.costWeights[3] = 0.1
#     # terminalcom_quad.costWeights[4] = 0.01
#     # terminalcom_quad.costWeights[5] = 0.0001
#     problem = crocoddyl.ShootingProblem(x0, [com_quadIAM] * T, terminalcom_quadIAM)

#     # Solving it using DDP
#     ddp = crocoddyl.SolverDDP(problem)
#     ddp.setCallbacks([crocoddyl.CallbackVerbose()])
#     ddp.solve([], [], 50)

#     # ui = ddp.us
#     # xi = ddp.xs

if __name__ == '__main__':
    # run_crocoddyl()
    M = pinocchio.rpy.rpyToMatrix(0, 0, 0)
    print(M)