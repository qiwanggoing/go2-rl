#!/usr/bin/env python3
import rclpy
import torch
from config import Config
import numpy as np
import time
from rclpy.node import Node
import argparse
from unitree_go.msg import LowState,LowCmd
from xbox_command import XboxController
from std_msgs.msg import Float32MultiArray

class dataReciever(Node):
    def __init__(self,config:Config):
        super().__init__("data_reciever")
        self.config = config
        self.cmd_sub=XboxController(self)
        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.low_state=LowState()
        # self.kps=self.config.kps
        # self.kds=self.config.kds
        if args.simulation:
            self.low_state_sub=self.create_subscription(LowState,"/mujoco/lowstate",self.low_state_callback,10)
            print("reading data from simuation")
        else:    
            self.low_state_sub=self.create_subscription(LowState,"/lowstate",self.low_state_callback,10) #500 HZ
            print("reading data from reality")


        self.get_logger().info("Waiting for data")
        self.timer = self.create_timer(0.02, self.run)
        self.torque_puber=self.create_publisher(Float32MultiArray,"/rl/target_pos",10)



    def low_state_callback(self,msg:LowState):
        self.low_state=msg
    
    def run(self):
        # self.get_logger().info("running")
        # Get the current joint position and velocity
        for i in range(12):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq
        sequence = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
        # comment these two lines for simulation
        self.qj=self.qj[sequence]
        self.dqj=self.dqj[sequence]
        
        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # create observation
        gravity_orientation = self.get_gravity_orientation(quat)
        qj_obs = (self.qj - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = self.dqj * self.config.dof_vel_scale

        ang_vel = ang_vel * self.config.ang_vel_scale
        self.cmd=np.zeros(3)
        self.left_button,self.right_button=self.cmd_sub.is_pressed()
        if self.left_button and self.right_button:
            linear_x,linear_y=self.cmd_sub.get_left_stick()
            angular_z=self.cmd_sub.get_right_stick()
            self.cmd=0.7*np.array([linear_x,linear_y,angular_z])
            # print(self.cmd)
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale 
        self.obs[9 : 21] = qj_obs
        self.obs[21 : 33] = dqj_obs
        self.obs[33:45] = self.action
        self.obs=np.clip(self.obs,-100,100)
        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        self.action=np.clip(self.action,-100,100)
        
        # transform action to target_dof_pos
        self.target_dof_pos = self.config.default_angles + self.action * self.config.action_scale
        #tau computed by network in the sequence of: 
        ''' 
        FL_hip,FL_thigh,FL_calf
        FR_hip,FR_thigh,FR_calf
        RL_hip,RL_thigh,RL_calf
        RR_hip,RR_thigh,RR_calf
        '''
        # The true order of the actuator in /lowcmd (also in mujoco simulator) should be:
        ''' 
        FR_hip,FR_thigh,FR_calf
        FL_hip,FL_thigh,FL_calf
        RR_hip,RR_thigh,RR_calf
        RL_hip,RL_thigh,RL_calf
        '''
        msg=Float32MultiArray()
        msg.data.extend(self.target_dof_pos.astype(np.float32)[sequence].tolist())
        self.torque_puber.publish(msg)
        
        
        
    @staticmethod
    def get_gravity_orientation(quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        # print(gravity_orientation)
        return gravity_orientation

    # @staticmethod
    # def pd_control(target_q, q, kp, dq, kd):
    #     """Calculates torques from position commands"""
    #     torques=(target_q - q) * kp -  dq * kd
    #     return torques
    
def main():
    rclpy.init()
    reciever_node=dataReciever(config=config)
    rclpy.spin(reciever_node)
    reciever_node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    # Load config
    config_path = f"/home/song/unitree_rl_gym/ros_communication/deploy_real/configs/go2.yaml"
    config = Config(config_path)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--simulation', type=bool, required=True, help='simulation or reality')
    parser.add_argument('--simulation', type=str, choices=["True", "False"], default="False")
    args = parser.parse_args()
    args.simulation = args.simulation == "True"
    main()