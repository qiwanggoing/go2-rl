#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import numpy as np
import time
import os


# --- ROS 2 消息 ---
from unitree_go.msg import LowCmd, LowState, MotorState, IMUState
from geometry_msgs.msg import Twist

from rl_policy import RLPolicy
from config import Go2Config, Commands


class MujocoRosBridgeNode(Node):
    def __init__(self, model_path):
        super().__init__('mujoco_ros_simulator')
        self.get_logger().info("Starting MuJoCo ROS 2 Simulator Bridge...")

        # 1. 加载 MuJoCo
        if not os.path.exists(model_path):
            self.get_logger().error(f"MuJoCo model file not found: {model_path}")
            raise FileNotFoundError
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.005 # 200Hz
        self.sim_dt = self.model.opt.timestep
        self.get_logger().info(f"MuJoCo model loaded. Sim timestep: {self.sim_dt}s")

        self.num_actuators = 12
        self.applied_torques = np.zeros(self.num_actuators, dtype=np.float32)

        # 2. 关节顺序和映射 (S24/S26 修正)
        # 1. XML 关节 (qpos) 顺序 (FL, FR, RL, RR)
        # 2. 策略/ROS 顺序 (FL, FR, RL, RR)
        # 3. XML 执行器 (ctrl) 顺序 (FR, FL, RR, RL)
        
        # 映射: 策略/ROS (FL,FR,RL,RR) -> 执行器 (FR,FL,RR,RL)
        # 这与 mujoco_simulator.py 中的映射一致
        self.ros_to_actuator_map = np.array([
            3, 4, 5,    # 策略 FL (0-2) -> 执行器 FL (3-5)
            0, 1, 2,    # 策略 FR (3-5) -> 执行器 FR (0-2)
            9, 10, 11,  # 策略 RL (6-8) -> 执行器 RL (9-11)
            6, 7, 8     # 策略 RR (9-11)-> 执行器 RR (6-8)
        ])

        # 3. ROS 2 发布者
        self.low_state_pub = self.create_publisher(LowState, "/mujoco/lowstate", 10)
        self.vel_pub = self.create_publisher(Twist, "/ekf/base_velocity", 10) # 模拟EKF

        # 4. ROS 2 订阅者
        self.low_cmd_sub = self.create_subscription(
            LowCmd,
            "/mujoco/lowcmd",
            self.low_cmd_callback,
            10
        )

        # 5. 初始化模拟器
        self.reset_simulation()

        # 6. 启动 MuJoCo 查看器
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

        # 7. 启动 ROS 2 主循环
        self.timer = self.create_timer(self.sim_dt, self.timer_callback)
        self.get_logger().info("MuJoCo ROS 2 Simulator is running.")

    def reset_simulation(self):
        """ 重置 MuJoCo 仿真到初始状态 """
        mujoco.mj_resetData(self.model, self.data)
        # 站立姿态的 qpos (来自 go2.xml 的 <key name="home">)
        qpos_home = [
            0, 0, 0.445, 1, 0, 0, 0,  # 基础
            0, 0.9, -1.8, # FL
            0, 0.9, -1.8, # FR
            0, 0.9, -1.8, # RL
            0, 0.9, -1.8  # RR
        ]
        # 我们的 XML 有 19 个 qpos (7 free + 12 joints)
        self.data.qpos[0:19] = qpos_home[0:19]
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info("Simulation reset to home pose.")


    def low_cmd_callback(self, msg: LowCmd):
        """
        接收来自 low_level_ctrl.cpp 的命令 (FL, FR, RL, RR 顺序)
        (S24 修正)
        """
        torques = np.zeros(self.num_actuators, dtype=np.float32)
        
        # MuJoCo XML 关节顺序 (FL, FR, RL, RR)
        q_joint_order = self.data.qpos[7:19]
        dq_joint_order = self.data.qvel[6:18]
        
        # (q_joint_order 和 msg.motor_cmd 都是 FL, FR, RL, RR 顺序)

        for i in range(self.num_actuators):
            cmd = msg.motor_cmd[i] # (数据是 FL, FR, RL, RR 顺序)
            
            if cmd.kp > 0.1:
                # PD 命令 (来自 C++ 状态机)
                tau = cmd.kp * (cmd.q - q_joint_order[i]) + cmd.kd * (cmd.dq - dq_joint_order[i])
                torques[i] = tau
            else:
                # 力矩命令 (来自 Python 策略)
                torques[i] = cmd.tau # (已经是 FL, FR, RL, RR 顺序)
        
        # 限制力矩 (FL, FR, RL, RR 顺序)
        torques[0:3] = np.clip(torques[0:3], -23.7, 23.7) # FL Hip/Thigh
        torques[2] = np.clip(torques[2], -45.43, 45.43) # FL Calf
        torques[3:6] = np.clip(torques[3:6], -23.7, 23.7) # FR Hip/Thigh
        torques[5] = np.clip(torques[5], -45.43, 45.43) # FR Calf
        torques[6:9] = np.clip(torques[6:9], -23.7, 23.7) # RL Hip/Thigh
        torques[8] = np.clip(torques[8], -45.43, 45.43) # RL Calf
        torques[9:12] = np.clip(torques[9:12], -23.7, 23.7) # RR Hip/Thigh
        torques[11] = np.clip(torques[11], -45.43, 45.43) # RR Calf
        
        # 存储 (FL, FR, RL, RR) 顺序的力矩
        self.applied_torques = torques 


    def publish_state(self):
        """
        从 MuJoCo 读取状态并发布 /mujoco/lowstate (FL, FR, RL, RR 顺序)
        (S24 修正)
        """
        # 1. 准备 LowState 消息
        low_state_msg = LowState()
        
        # 1. 读取 MuJoCo XML 关节顺序 (FL, FR, RL, RR)
        # (S16 修正: 不再需要映射)
        q_policy_order = self.data.qpos[7:19].astype(np.float32)
        dq_policy_order = self.data.qvel[6:18].astype(np.float32)
        
        # 2. 填充消息 (FL, FR, RL, RR 顺序)
        for i in range(self.num_actuators):
            motor = MotorState()
            motor.q = float(q_policy_order[i]) # (S10 修正)
            motor.dq = float(dq_policy_order[i]) # (S10 修正)
            motor.tau_est = 0.0 
            low_state_msg.motor_state[i] = motor

        # 3. IMU 数据 (S14 修正)
        imu_msg = IMUState()
        imu_msg.gyroscope = self.data.sensor('imu_gyro').data.copy().astype(np.float32)
        imu_msg.accelerometer = self.data.sensor('imu_acc').data.copy().astype(np.float32) 
        imu_msg.quaternion = self.data.sensor('imu_quat').data.copy().astype(np.float32) 
        low_state_msg.imu_state = imu_msg
        
        self.low_state_pub.publish(low_state_msg)

        # 4. 准备 Twist 消息 (模拟 EKF)
        vel_msg = Twist()
        vel_data = self.data.sensor('frame_vel').data
        vel_msg.linear.x = float(vel_data[0])
        vel_msg.linear.y = float(vel_data[1])
        vel_msg.linear.z = float(vel_data[2])
        vel_msg.angular.x = float(imu_msg.gyroscope[0])
        vel_msg.angular.y = float(imu_msg.gyroscope[1])
        vel_msg.angular.z = float(imu_msg.gyroscope[2])
        self.vel_pub.publish(vel_msg)


    def timer_callback(self):
        """
        ROS 2 主循环, 运行 MuJoCo 步进
        (S24 修正)
        """
        if self.viewer.is_running():
            step_start = time.time()

            # --- 关键：应用映射 ---
            # self.applied_torques 是 (FL, FR, RL, RR)
            # data.ctrl 期望 (FR, FL, RR, RL)
            mapped_torques = self.applied_torques[self.ros_to_actuator_map]
            
            self.data.ctrl[0:self.num_actuators] = mapped_torques[0:self.num_actuators]

            # 步进
            mujoco.mj_step(self.model, self.data)
            self.publish_state()
            self.viewer.sync()

            time_until_next_step = self.sim_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        else:
            self.get_logger().info("MuJoCo viewer closed, shutting down ROS 2 node.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    # 路径来自 mujoco_simulator.py
    # !!! 关键：你必须修改这个路径为你本地的绝对路径
    model_path = '/home/qiwang/sata_mujoco/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/resources/go2/scene.xml'
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"FATAL: Model path does not exist: {model_path}")
        print("Please edit 'mujoco_ros_simulator.py' and fix the 'model_path' variable.")
        return

    try:
        ros_sim_node = MujocoRosBridgeNode(model_path)
        rclpy.spin(ros_sim_node)
    except Exception as e:
        print(f"FATAL error during node execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'ros_sim_node' in locals():
            ros_sim_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
