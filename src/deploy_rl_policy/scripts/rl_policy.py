#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import numpy as np
import argparse 
import time
import traceback # 用于调试

# --- ROS 消息导入 ---
from unitree_go.msg import LowState, MotorState, IMUState
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist 

# --- 修复导入 (S6) ---
try:
    from config import Go2Config, Commands
except ImportError as e:
    print(f"FATAL: Failed to import config.py: {e}")
    print("Ensure 'config.py' is in the same 'scripts' directory and was installed by colcon.")
    exit(1)


# =====================================================================================
# 1. 原始的 RLPolicy 类（保持不变）
# =====================================================================================
class RLPolicy:
    def __init__(self, policy_path):
        print("Loading TorchScript policy from:", policy_path)
        self.actor_critic = torch.jit.load(policy_path, map_location=torch.device('cpu'))
        self.actor_critic.eval()
        print("TorchScript RL Policy loaded successfully!")

    def get_action(self, observations):
        obs_batch = torch.from_numpy(observations).float().unsqueeze(0)
        with torch.no_grad():
            actions_mean = self.actor_critic(obs_batch)
        return actions_mean.detach().numpy().flatten()

# =====================================================================================
# 2. SATA 策略 ROS 2 节点 (S26 最终版)
# =====================================================================================
class SATAPolicyNode(Node):
    def __init__(self, policy_path, is_simulation=False):
        super().__init__('sata_policy_node')
        self.get_logger().info(f"Initializing SATA Policy Node (Simulation: {is_simulation})")

        # 1. 加载策略
        self.policy = RLPolicy(policy_path)
        self.num_actions = 12 

        # 2. 状态变量 (用于存储来自回调的数据)
        self.current_q = np.zeros(self.num_actions, dtype=np.float32)
        self.current_dq = np.zeros(self.num_actions, dtype=np.float32)
        self.current_base_ang_vel = np.zeros(3, dtype=np.float32)
        self.current_base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # w, x, y, z
        self.estimated_base_lin_vel = np.zeros(3, dtype=np.float32)
        self.commands = Commands()      
        self.data_ready = False
        self.is_rl_mode_active = False # (S26) 状态污染修复
        
        # 3. 订阅
        state_topic = "/mujoco/lowstate" if is_simulation else "/lowstate"
        self.state_sub = self.create_subscription(LowState, state_topic, self.state_callback, 10)
        self.joy_sub = self.create_subscription(Joy, "/joy", self.joy_callback, 10)
        
        # 模拟器会发布这个，EKF 节点也会发布
        self.ekf_sub = self.create_subscription(Twist, "/ekf/base_velocity", self.ekf_callback, 10)
        
        self.get_logger().info(f"Subscribing to LowState on: {state_topic}")
        self.get_logger().info("Subscribing to Joy on: /joy")
        self.get_logger().info("Subscribing to EKF velocity on: /ekf/base_velocity")

        # 4. SATA 内部状态和配置
        self.default_dof_pos = self.get_default_dof_pos_ordered()
        self.last_torques = np.zeros(self.num_actions, dtype=np.float32)
        self.motor_fatigue = np.zeros(self.num_actions, dtype=np.float32)
        self.activation_sign = np.zeros(self.num_actions, dtype=np.float32) # (已修正为 12 维)

        # 5. SATA 生物力学模型参数
        self.dof_vel_limits = np.full(self.num_actions, Go2Config.DOF_VEL_LIMITS, dtype=np.float32)
        self.torque_limits = np.full(self.num_actions, Go2Config.TORQUE_LIMITS, dtype=np.float32)
        self.GAMMA = Go2Config.GAMMA 
        self.BETA = Go2Config.BETA   
        self.dt = 1.0 / 200.0 # 目标控制频率 200Hz
        
        # 6. (S24) 移除所有映射，本节点只处理 (FL, FR, RL, RR) 顺序
        self.get_logger().info("SATA internal states initialized (FL, FR, RL, RR order).")
        
        # 7. 力矩发布者
        self.torque_pub = self.create_publisher(Float32MultiArray, "/rl/target_torques", 10)
        self.torque_msg = Float32MultiArray()

        # 8. 主循环 Timer
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info(f"SATA policy node ready. Running at {1.0/self.dt} Hz.")

    def state_callback(self, msg: LowState):
        """
        处理来自 /lowstate 的数据 (FL, FR, RL, RR 顺序)
        (S24 修正)
        """
        # 消息已经是 (FL, FR, RL, RR) 顺序，无需映射
        self.current_q = np.array([motor.q for motor in msg.motor_state[:self.num_actions]], dtype=np.float32)
        self.current_dq = np.array([motor.dq for motor in msg.motor_state[:self.num_actions]], dtype=np.float32)
        
        # IMU 数据 (S14 修正)
        self.current_base_ang_vel = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        self.current_base_quat = np.array(msg.imu_state.quaternion, dtype=np.float32) # w, x, y, z
        
        if not self.data_ready and self.estimated_base_lin_vel is not None:
             self.get_logger().info("LowState and EKF data received. System is ready.")
             self.data_ready = True

    def ekf_callback(self, msg: Twist):
        self.estimated_base_lin_vel = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32)

    def joy_callback(self, msg: Joy):
        """
        处理来自 /joy 的手柄命令，更新期望速度
        并检测 RL 模式切换 (S26 修正)
        """
        # 1. 更新速度命令
        if len(msg.axes) > 4:
            self.commands.lin_vel_x = msg.axes[1] * 1.5 # 假设 左摇杆上下
            self.commands.lin_vel_y = msg.axes[0] * -0.5 # 假设 左摇杆左右
            self.commands.ang_vel_yaw = msg.axes[3] * -1.5 # 假设 右摇杆左右
        
        # 2. 检测模式切换
        if len(msg.buttons) > 5:
            # (A 键)
            if msg.buttons[0]: 
                if self.is_rl_mode_active:
                    self.get_logger().info("RL Mode DEACTIVATED (A pressed).")
                self.is_rl_mode_active = False
            
            # (LB + RB 键)
            elif msg.buttons[4] and msg.buttons[5]:
                if not self.is_rl_mode_active:
                    self.get_logger().warn("RL Mode ACTIVATED. Resetting SATA internal state.")
                    # --- 关键：重置内部状态为 0 ---
                    self.last_torques.fill(0.0)
                    self.motor_fatigue.fill(0.0)
                    self.activation_sign.fill(0.0)
                    # ---
                self.is_rl_mode_active = True
        
    def get_default_dof_pos_ordered(self):
        """
        获取 (FL, FR, RL, RR) 顺序的默认关节角度 (S24 修正)
        """
        default_dof_pos_dict = Go2Config.DEFAULT_JOINT_ANGLES
        
        # 策略顺序 (FL, FR, RL, RR)
        ordered_joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        
        ordered_dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        for i, name in enumerate(ordered_joint_names):
            if name in default_dof_pos_dict:
                ordered_dof_pos[i] = default_dof_pos_dict[name]
            else:
                self.get_logger().warn(f"Joint '{name}' not in Go2Config.DEFAULT_JOINT_ANGLES!")
        return ordered_dof_pos # (返回 FL, FR, RL, RR 顺序)

    def get_observation(self):
        """
        构建SATA观测向量 (60维)
        """
        base_lin_vel = self.estimated_base_lin_vel
        base_ang_vel = self.current_base_ang_vel
        base_quat_wxyz = self.current_base_quat
        
        q_w, q_x, q_y, q_z = base_quat_wxyz
        gravity_vec = np.array([0., 0., -1.], dtype=np.float32)
        q_calc = np.array([q_x, q_y, q_z, q_w], dtype=np.float32)
        w, x, y, z = q_calc[3], q_calc[0], q_calc[1], q_calc[2]
        qv = np.array([x, y, z], dtype=np.float32)
        uv = np.cross(qv, gravity_vec)
        uuv = np.cross(qv, uv)
        projected_gravity = gravity_vec + 2 * (w * uv + uuv)

        joint_pos = self.current_q
        joint_vel = self.current_dq

        obs_lin_vel = base_lin_vel * Go2Config.OBS_SCALES.lin_vel
        obs_ang_vel = base_ang_vel * Go2Config.OBS_SCALES.ang_vel
        obs_dof_pos = (joint_pos - self.default_dof_pos) * Go2Config.OBS_SCALES.dof_pos
        obs_dof_vel = joint_vel * Go2Config.OBS_SCALES.dof_vel
        
        commands_scaled = np.array([
            self.commands.lin_vel_x * Go2Config.COMMANDS_SCALES.lin_vel_x,
            self.commands.lin_vel_y * Go2Config.COMMANDS_SCALES.lin_vel_y,
            self.commands.ang_vel_yaw * Go2Config.COMMANDS_SCALES.ang_vel_yaw
        ], dtype=np.float32)

        observation = np.concatenate([
            obs_lin_vel,        # 3
            obs_ang_vel,        # 3
            projected_gravity,  # 3
            obs_dof_pos,        # 12
            obs_dof_vel,        # 12
            commands_scaled,    # 3
            self.last_torques,  # 12
            self.motor_fatigue  # 12
        ]) 
        
        return observation.astype(np.float32)

    def _compute_sata_torques(self, raw_actions, dof_vel):
        """
        实现 SATA 生物力学模型 (Eq. 1-4)
        """
        actions_scaled = raw_actions * Go2Config.ACTION_SCALE 
        torques_limits = self.torque_limits                 

        current_activation_sign = np.tanh(actions_scaled / torques_limits)
        self.activation_sign = (current_activation_sign - self.activation_sign) * self.GAMMA + self.activation_sign

        torques = self.activation_sign * torques_limits * (
            1 - np.sign(self.activation_sign) * dof_vel[:self.num_actions] / self.dof_vel_limits
        )

        self.motor_fatigue += np.abs(torques) * self.dt
        self.motor_fatigue *= self.BETA
        
        return torques.astype(np.float32)

    def timer_callback(self):
        """
        SATA 策略主循环 (已添加状态重置和保护 - S26)
        """
        # 0. 检查 C++ 节点是否也处于 RL 模式
        if not self.is_rl_mode_active:
            # C++ 节点处于 PD (站立/卧倒) 模式。
            # 我们必须停止计算，以防止“状态污染”。
            return
            
        if not self.data_ready:
            self.get_logger().warn("RL Mode active, but waiting for sensor data...", throttle_duration_sec=2.0)
            return
            
        # 1. 获取观测 (此时内部状态是干净的)
        observation = self.get_observation()
        
        # 2. 策略推理 (输出 FL, FR, RL, RR 顺序)
        raw_action = self.policy.get_action(observation)
        
        # 3. 计算 SATA 力矩 (使用 FL, FR, RL, RR 顺序)
        final_torques = self._compute_sata_torques(raw_action, self.current_dq) 

        # 4. 存储内部状态 (使用 FL, FR, RL, RR 顺序)
        self.last_torques = final_torques.copy()
        
        # 5. 发布力矩 (FL, FR, RL, RR 顺序)
        self.torque_msg.data = final_torques.tolist()
        self.torque_pub.publish(self.torque_msg)

# =====================================================================================
# 3. 主函数
# =====================================================================================
def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--is_simulation', 
        type=lambda x: (str(x).lower() == 'true'), 
        default=False,
        help='Run in simulation mode (default: False)'
    )
    known_args, _ = parser.parse_known_args()
    is_simulation = known_args.is_simulation

    # !!! 关键：更新此路径 !!!
    policy_path = '/home/qiwang/sata_mujoco/legged_gym/logs/SATA/exported/policies/policy_1.pt'
    # (这个路径来自你原始的 mujoco_simulator.py)

    try:
        sata_node = SATAPolicyNode(policy_path, is_simulation)
        rclpy.spin(sata_node)
    except Exception as e:
        print(f"FATAL error during node execution: {e}")
        traceback.print_exc() # 打印详细错误
    finally:
        if 'sata_node' in locals():
            sata_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()