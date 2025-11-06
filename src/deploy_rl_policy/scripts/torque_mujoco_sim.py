#!/usr/bin/env python3
import time
import threading
import rclpy
from rclpy.node import Node

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
# import torch
import yaml
from pathlib import Path
# from xbox_command import XboxController
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32MultiArray



class TorqueMujocoSim(Node):
    def __init__(self):
        super().__init__("torque_mujoco_sim") # 修正1: 节点名称 (第22行)
        # self.cmd_sub = XboxController(self)  # 修正2: 删除手柄 (第23行)

        self.load_config()

        # 修正3: 优化发布者
        self.imu_pub=self.create_publisher(Imu,"/imu/data",10)
        self.contact_force_pub=self.create_publisher(Float32MultiArray,"/contact",10)

        # 将两个 Float32MultiArray (/joint_angels, /joint_velocities)
        # 替换为 TorquePolicyNode 期望的单个 JointState 话题
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10) # (替换 第27行)
        # self.joint_vel_pub=... (删除 第28行)
        # self.omega_pub=... (删除 第29行, IMU的角速度将合并到 /imu/data 中)

        self.z_axis_force_pub=self.create_publisher(Float32MultiArray,"/z_axis_force",10)
        self.step_counter = 0
        self.true_vel_pub=self.create_publisher(Float32MultiArray,"/true_velocities",10)

        # Initialize Mujoco
        self.init_mujoco()

        # Load policy
        # self.policy = torch.jit.load(self.policy_path)
        # self.action = np.zeros(self.num_actions, dtype=np.float32)
        # self.target_dof_pos = self.default_angles.copy()
        # self.obs = np.zeros(self.num_obs, dtype=np.float32)

        # 1. 存储目标力矩的变量 (从 /rl/torque_cmd 接收)
        self.target_torques = np.zeros(12, dtype=np.float32)

        # 2. 订阅力矩话题 (这是 "大脑" 发布的)
        self.torque_sub = self.create_subscription(
            Float32MultiArray,
            '/rl/torque_cmd',
            self.torque_callback,
            10)

        # 3. 添加执行器映射 (从 deploy_rl_policy/mujoco_simulator.py 复制)
        # SATA 策略 (FL,FR,RL,RR) 与 MuJoCo XML (FR,FL,RR,RL) 的映射
        #
        self.policy_to_mujoco_actuator_map = np.array([
            3, 4, 5,    # 策略 FL (0-2) -> MuJoCo FL (3-5)
            0, 1, 2,    # 策略 FR (3-5) -> MuJoCo FR (0-2)
            9, 10, 11,  # 策略 RL (6-8) -> MuJoCo RL (9-11)
            6, 7, 8     # 策略 RR (9-11)-> MuJoCo RR (6-8)
        ])
        self.timer = self.create_timer(self.simulation_dt, self.step_simulation)

    def load_config(self):
        """
        加载此仿真节点（身体）所需的配置：
        1. MuJoCo XML 路径。
        2. 仿真时间步长 (dt)。
        """
        current_file = Path(__file__).resolve()
        # parent_dir 应该是 '.../deploy_rl_policy'
        parent_dir = current_file.parent.parent 
        
        # 配置文件路径已更改，指向 'deploy_rl_policy/configs/go2.yaml'
        config_file = parent_dir /'configs'/'go2.yaml' #

        if not config_file.exists():
            # 这是一个关键错误，需要停止
            self.get_logger().fatal(f"配置文件未找到: {config_file}")
            raise FileNotFoundError(f"配置文件未找到: {config_file}")

        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
            # --- 1. (保留) XML 路径 ---
            self.xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR) #
            
            # --- 2. (保留) 仿真步长 ---
            self.simulation_dt = config["simulation_dt"] 

        # --- 4. (新增) 关键检查 ---
        # 确保仿真 dt 必须与 SATA 策略训练的 dt (0.005s) 一致
        sata_dt = 0.005 #
        if self.simulation_dt != sata_dt:
            self.get_logger().warn(f"YAML中的 simulation_dt ({self.simulation_dt}) 与SATA训练 ({sata_dt}) 不符。")
            self.get_logger().warn(f"将强制覆盖为 {sata_dt} 以保证物理一致性。")
            self.simulation_dt = sata_dt

        self.get_logger().info(f"已从 {config_file} 加载仿真配置。")
        self.get_logger().info(f"XML 路径: {self.xml_path}")
        self.get_logger().info(f"仿真步长 (dt): {self.simulation_dt}")

    def torque_callback(self, msg: Float32MultiArray):
        """
        接收来自策略节点的目标力矩并存储。
        """
        if len(msg.data) == 12:
            self.target_torques = np.array(msg.data, dtype=np.float32)
        else:
            self.get_logger().warn(
                f"接收到的力矩消息长度为 {len(msg.data)}，但期望长度为 12。")

    def init_mujoco(self):
        """Initialize Mujoco model and data"""
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        
        # 1. 设置SATA策略的默认站立姿态
        
        # (这是SATA策略训练时的默认关节角度)
        # (来源: deploy_rl_policy/scripts/config.py)
        #
        sata_default_angles_dict = { 
            'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,
            'FL_thigh_joint': 1.45, 'FR_thigh_joint': 1.45,
            'FL_calf_joint': -2.5,  'FR_calf_joint': -2.5,
            
            'RL_hip_joint': 0.1,    'RR_hip_joint': -0.1,
            'RL_thigh_joint': 1.45, 'RR_thigh_joint': 1.45,
            'RL_calf_joint': -2.5,  'RR_calf_joint': -2.5,
        }

        # (这是MuJoCo XML 
        # 中 qpos[7:] 的关节顺序)
        mujoco_joint_order = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        
        # 2. 按照MuJoCo的顺序填充默认角度
        sata_default_qpos = np.zeros(12)
        for i, joint_name in enumerate(mujoco_joint_order):
            sata_default_qpos[i] = sata_default_angles_dict[joint_name]

        # 3. 设置初始姿态
        self.d.qpos[0:7] = [0, 0, 0.42, 1, 0, 0, 0] # 站立的躯干位置
        self.d.qpos[7:] = sata_default_qpos
        self.d.qvel[:] = 0.0 # 初始速度为0

        # 4. (保留) 设置时间步长
        self.m.opt.timestep = self.simulation_dt
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        print("Number of qpos:", self.m.nq)
        print("Joint order:")
        for i in range(self.m.njnt):
            print(f"{i}: {self.m.joint(i).name}")

    def step_simulation(self):
        """
        主仿真步骤 (由 ROS 2 定时器调用)。
        此版本应用来自 /rl/torque_cmd 话题的力矩。
        """
        # 1. (保留) 计数器
        self.step_counter += 1

        # 2. (替换) 应用力矩控制
        # self.target_torques 是由 self.torque_callback 在后台更新的
        
        # 应用SATA策略 (FL,FR,RL,RR) 到 MuJoCo (FR,FL,RR,RL) 的执行器映射
        #
        mapped_torques = self.target_torques[self.policy_to_mujoco_actuator_map]
        
        # 将最终力矩应用到 MuJoCo 执行器
        #
        self.d.ctrl[:] = mapped_torques
        
        # 3. (删除) PD 控制逻辑
        # tau = self.pd_control(...)
        # sequence = [...]
        # tau = [tau[index] for index in sequence]
        
        # 4. (保留) Mujoco 步进
        mujoco.mj_step(self.m, self.d) #
        
        # 5. (保留) 发布传感器数据
        # (这个函数 publish_sensor_data 暂时不需要修改)
        self.publish_sensor_data() #
        
        # 6. (删除) 策略推理
        # if self.step_counter % self.control_decimation == 0:
        #     self.run_policy()
        
        # 7. (保留) 同步 Mujoco 查看器
        self.viewer.sync() 

    # def run_policy(self):
    #     """Run policy inference and update target DOF positions"""
    #     # Build observation vector
    #     self.cmd=np.zeros(3)
    #     # self.get_logger().info("run policy")
    #     self.left_button,self.right_button=self.cmd_sub.is_pressed()
    #     if self.left_button and self.right_button:
    #         linear_x,linear_y=self.cmd_sub.get_left_stick()
    #         angular_z=self.cmd_sub.get_right_stick()
    #         self.cmd=np.array([linear_x,linear_y,angular_z])
    #     # self.get_logger().info(f"FORCE {self.d.sensordata[55:]}")
    #     # print(len(self.d.sensordata))
    #     self.obs[:3] = self.d.sensordata[40:43] * self.ang_vel_scale  # Angular velocity
    #     self.obs[3:6] = self.get_gravity_orientation(self.d.qpos[3:7])  # Gravity vector
    #     self.obs[6:9] = self.cmd * self.cmd_scale  # Scaled command
    #     self.obs[9:21] = (self.d.qpos[7:19] - self.default_angles) * self.dof_pos_scale  # Joint positions
    #     self.obs[21:33] = self.d.qvel[6:18] * self.dof_vel_scale  # Joint velocities
    #     self.obs[33:45] = self.action  # Previous actions
    #     self.grav_acc=9.81*self.obs[3:6]
    #     # Policy inference
    #     obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
    #     self.action = self.policy(obs_tensor).detach().numpy().squeeze()
    #     self.target_dof_pos = self.action * self.action_scale + self.default_angles
    #     print(self.target_dof_pos)
    def publish_sensor_data(self):
        """
        发布传感器数据到 ROS 话题。
        此版本经过修正，可匹配 TorquePolicyNode 的订阅需求。
        """
        
        # 1. (已修复) 发布 IMU 数据 (/imu/data)
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        
        # 从 MuJoCo 传感器读取数据
        #
        quat_wxyz = self.d.sensor("imu_quat").data
        ang_vel = self.d.sensor("imu_gyro").data
        
        # 填充朝向 (w, x, y, z)
        imu_msg.orientation.w = quat_wxyz[0]
        imu_msg.orientation.x = quat_wxyz[1]
        imu_msg.orientation.y = quat_wxyz[2]
        imu_msg.orientation.z = quat_wxyz[3]
        
        # 填充角速度
        imu_msg.angular_velocity.x = ang_vel[0]
        imu_msg.angular_velocity.y = ang_vel[1]
        imu_msg.angular_velocity.z = ang_vel[2]
        
        # (我们不再计算或发布线性加速度，因为它在策略中没用到)
        #
        self.imu_pub.publish(imu_msg)

        # 2. (已修复) 发布关节状态 (/joint_states)
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        # 关节名称的顺序必须与 pos/vel 数据的顺序一致
        joint_msg.name = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        joint_msg.position = list(self.d.qpos[7:]) #
        joint_msg.velocity = list(self.d.qvel[6:]) #
        # (力矩/effort 留空)
        self.joint_pub.publish(joint_msg)

        # 3. (保留) 发布真实线速度 (/true_velocities)
        #
        true_velocity_array=Float32MultiArray()
        true_velocity_array.data=list(self.d.sensor("frame_vel").data) #
        self.true_vel_pub.publish(true_velocity_array)
        
        # 4. (保留) 发布接触力 (可选，策略不使用，但调试有用)
        #
        array=Float32MultiArray()
        # (注意：go2.xml 中没有定义足部接触传感器)
        # (原始的 mujoco_simulator.py 假设传感器索引 55-67 是接触力)
        # (在 scene.xml 中,
        # 'imu_acc' 是 43-45, 之后没有其他传感器了.
        # 啊, 你使用的是 `go2.xml` 还是 `scene.xml`?
        # `base_velocity_estimator/config/go2.yaml` 指向 `scene.xml`。
        # `scene.xml` 包含 `go2.xml`。
        # 在 `go2.xml` 中，传感器 `frame_vel` 是最后一个 (索引 52-54)。
        # 索引 55-67 在 `go2.xml` 中并不存在。
        
        # (结论：原始代码中的力传感器索引 [55-67] 是无效的，
        # 除非你使用了不同的 XML。我们将暂时注释掉这部分以避免崩溃。)
        
        # fl_force_list=np.array([self.d.sensordata[i] for i in range (55,58)])
        # ... (lines 203-214)
        # array.data=[FL_force,FR_force,RL_force,RR_force]
        # self.contact_force_pub.publish(array)





    # @staticmethod
    # def get_gravity_orientation(quaternion):
    #     qw = quaternion[0]
    #     qx = quaternion[1]
    #     qy = quaternion[2]
    #     qz = quaternion[3]

    #     gravity_orientation = np.zeros(3)

    #     gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    #     gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    #     gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    #     return gravity_orientation

    # @staticmethod
    # def pd_control(target_q, q, kp, dq, kd):
    #     """Calculates torques from position commands"""
    #     torques=(target_q - q) * kp -  dq * kd
    #     return torques
    
    # @staticmethod
    # def quat_to_rot_matrix(q):
    #     """ 将四元数 (x, y, z, w) 转换为旋转矩阵 (3x3) """
    #     w,x, y, z = q
    #     R = np.array([
    #         [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
    #         [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
    #         [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    #     ])
    #     return R
    

def main(args=None):
    rclpy.init(args=args)
    node = TorqueMujocoSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

