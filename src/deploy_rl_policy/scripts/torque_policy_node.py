#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu, JointState
# from .xbox_command import XboxController # 假设 xbox_command.py 在同目录
import mujoco
import mujoco.viewer
import numpy as np
import time
import torch
import os # 导入 os 模块用于文件检查
# import traceback # 移除错误打印，不再需要 traceback

from rl_policy import RLPolicy
from config import Go2Config, Commands
try:
    from xbox_command import XboxController
except ImportError:
    from xbox_command import XboxController

class TorquePolicyNode(Node): # 重命名类
    def __init__(self, policy_path):
        super().__init__('torque_policy_node')

        # 1. 保留策略加载和SATA状态
        self.policy = RLPolicy(policy_path)
        self.commands = Commands()
        self.num_actions = 12
        self.default_dof_pos = self.get_default_dof_pos_ordered() # 需要修改 get_default_dof_pos_ordered

        # 2. 保留SATA生物力学模型参数和状态
        self.last_torques = np.zeros(self.num_actions, dtype=np.float32)
        self.motor_fatigue = np.zeros(self.num_actions, dtype=np.float32)
        self.activation_sign = np.zeros(self.num_actions, dtype=np.float32)
        self.dof_vel_limits = np.full(self.num_actions, Go2Config.DOF_VEL_LIMITS, dtype=np.float32)
        self.torque_limits = np.full(self.num_actions, Go2Config.TORQUE_LIMITS, dtype=np.float32)
        self.GAMMA = Go2Config.GAMMA
        self.BETA = Go2Config.BETA

        # 3. 移除MuJoCo模型和查看器
        # (删除 self.model, self.data, self.viewer, self.model_path)

        # 4. 添加ROS2接口
        self.cmd_sub = XboxController(self)
        self.torque_pub = self.create_publisher(Float32MultiArray, '/rl/torque_cmd', 10)

        # 5. 添加传感器订阅
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10) # 假设使用 JointState
        # self.lin_vel_sub = self.create_subscription(Float32MultiArray, '/base_lin_vel', self.lin_vel_callback, 10)
        self.lin_vel_sub = self.create_subscription(Float32MultiArray, '/true_velocities', self.lin_vel_callback, 10)
        # 6. 存储传感器数据
        self.current_ang_vel = np.zeros(3, dtype=np.float32)
        self.current_lin_vel = np.zeros(3, dtype=np.float32)
        self.current_dof_pos = np.zeros(12, dtype=np.float32)
        self.current_dof_vel = np.zeros(12, dtype=np.float32)
        self.current_quat_wxyz = np.array([1.0, 0, 0, 0], dtype=np.float32)

        #   8. (新) 定义策略期望的关节顺序
        # 这个顺序必须与 get_default_dof_pos_ordered 中的顺序一致
        self.policy_joint_order = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        # (新) 添加就绪标志
        self.imu_received = False
        self.lin_vel_received = False
        self.get_logger().info("策略节点已启动，正在等待所有传感器数据...")
        # 7. 创建策略定时器 (例如 50Hz, 对应 0.02s)
        # self.policy_timer = self.create_timer(0.005, self.policy_step)
    
    # callbacks for sensor data
    def imu_callback(self, msg):
        self.current_ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.current_quat_wxyz = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        # (新) 添加标志
        self.imu_received = True 

    def joint_callback(self, msg: JointState):
        """
        (健壮的) 存储来自 /joint_states 话题的最新关节数据。
        这个版本会根据 msg.name 重新排序，以匹配策略期望的顺序。
        *** 这现在是策略计算的主触发器 ***
        """
        # 1. (保留) 检查消息是否完整
        if len(msg.name) != len(msg.position) or len(msg.name) != len(msg.velocity):
            self.get_logger().warn("JointState 消息格式错误 (name, pos, vel 长度不匹配)", once=True)
            return

        # 2. (保留) 关节名 -> 索引 的映射
        name_to_index_map = {name: i for i, name in enumerate(msg.name)}

        # 3. (保留) 按照策略期望的顺序填充数据
        for i, expected_name in enumerate(self.policy_joint_order):
            if expected_name in name_to_index_map:
                msg_index = name_to_index_map[expected_name]
                self.current_dof_pos[i] = msg.position[msg_index]
                self.current_dof_vel[i] = msg.velocity[msg_index]
            else:
                self.get_logger().error(f"在 /joint_states 消息中未找到期望的关节: {expected_name}", once=True)
                return 

        # --- (新) 触发策略逻辑 ---

        # 4. 检查所有传感器是否"就绪"
        if not self.imu_received or not self.lin_vel_received:
            # (只警告一次，避免刷屏)
            self.get_logger().warn("正在等待 IMU 和 LinVel 的首次数据...", once=True)
            return

        # 5. (新) 运行策略计算
        # (我们之前重命名的函数)
        self._run_policy_logic()

    def lin_vel_callback(self, msg):
        self.current_lin_vel = np.array(msg.data)
        # (新) 添加标志
        self.lin_vel_received = True


    def get_default_dof_pos_ordered(self):
        """
        获取与SATA策略训练时一致的默认关节角度（FL, FR, RL, RR 顺序）。
        此版本移除了对 self.model 的依赖。
        """
        # 1. 仍然从 config.py 中获取关节角度的字典
        default_dof_pos_dict = Go2Config.DEFAULT_JOINT_ANGLES #
        
        # 2. 我们根据分析 明确定义策略期望的关节顺序
        # (这个顺序必须与 get_observation 中 qpos[7:...] 的顺序一致)
        ordered_joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]

        # 3. 准备一个空的数组，
        # self.num_actions 在 __init__ 中定义为 12
        ordered_dof_pos = np.zeros(self.num_actions, dtype=np.float32) 
        
        # 4. 循环我们手动定义的列表，而不是依赖 self.model
        for i, joint_name in enumerate(ordered_joint_names):
            if joint_name in default_dof_pos_dict:
                # 5. 从字典中取值，填充到数组的正确位置
                ordered_dof_pos[i] = default_dof_pos_dict[joint_name]
            else:
                 # 这个警告现在更重要，因为它意味着 config.py 中可能缺少关键定义
                 print(f"严重警告：在 Go2Config.DEFAULT_JOINT_ANGLES 中未找到关节 '{joint_name}'。将使用 0。")

        # 6. 确保我们总是返回一个12维的数组
        if len(ordered_joint_names) != self.num_actions:
             print(f"严重警告：代码中定义的关节列表长度 ({len(ordered_joint_names)}) 与期望的动作数量 ({self.num_actions}) 不匹配。")

        return ordered_dof_pos


    def get_observation(self):
        """
        构建观测向量。
        数据来源：ROS 话题 (存储在 self.current_... 变量中)
        指令来源：Xbox 手柄 (self.cmd_sub)
        """
        
        # 1. 获取基础状态 (来自ROS存储变量)
        base_lin_vel = self.current_lin_vel       # [替换: `self.data.sensor("frame_vel").data`]
        base_ang_vel = self.current_ang_vel       # [替换: `self.data.sensor("imu_gyro").data`]
        base_quat_wxyz = self.current_quat_wxyz   # [替换: `self.data.sensor("imu_quat").data`]
        
        # 2. 计算重力向量 (这部分逻辑不变)
        q_w, q_x, q_y, q_z = base_quat_wxyz
        gravity_vec = np.array([0., 0., -1.], dtype=np.float32)
        # (注意：SATA 策略使用的四元数顺序是 [x, y, z, w])
        q_calc = np.array([q_x, q_y, q_z, q_w], dtype=np.float32)
        w, x, y, z = q_calc[3], q_calc[0], q_calc[1], q_calc[2]
        qv = np.array([x, y, z], dtype=np.float32)
        uv = np.cross(qv, gravity_vec)
        uuv = np.cross(qv, uv)
        projected_gravity = gravity_vec + 2 * (w * uv + uuv)

        # 3. 获取关节状态 (来自ROS存储变量)
        joint_pos = self.current_dof_pos # [替换: `qpos[7:7+self.num_actions]`]
        joint_vel = self.current_dof_vel # [替换: `qvel[6:6+self.num_actions]`]

        # 4. 应用观测值缩放 (这部分逻辑不变)
        obs_lin_vel = base_lin_vel * Go2Config.OBS_SCALES.lin_vel
        obs_ang_vel = base_ang_vel * Go2Config.OBS_SCALES.ang_vel
        # (self.default_dof_pos 来自我们第3步修改的函数)
        obs_dof_pos = (joint_pos - self.default_dof_pos) * Go2Config.OBS_SCALES.dof_pos
        obs_dof_vel = joint_vel * Go2Config.OBS_SCALES.dof_vel
        
        # 5. (新) 从手柄获取指令
        # 检查手柄是否按下了 LB 和 RB 键 (索引 4 和 5)
        left_pressed, right_pressed = self.cmd_sub.is_pressed(4, 5)
        
        if left_pressed and right_pressed:
            # (Y) 轴, (X) 轴
            lin_x, lin_y = self.cmd_sub.get_left_stick() 
            # (RX) 轴
            ang_z = self.cmd_sub.get_right_stick()
            
            # (注意：SATA 策略使用 x 前进, y 左/右, yaw 旋转)
            # (xbox_command.py 返回 (axes[1], axes[0]))
            # (axes[1] 是左摇杆上下，默认向上为正)
            # (axes[0] 是左摇杆左右，默认向左为正)
            # (axes[3] 是右摇杆左右，默认向左为正)
            
            # 假设SATA指令：+lin_x (前进), +lin_y (向左), +ang_yaw (逆时针)
            self.commands.lin_vel_x = lin_x    # 左摇杆上下 (axes[1])
            self.commands.lin_vel_y = lin_y    # 左摇杆左右 (axes[0])
            self.commands.ang_vel_yaw = ang_z  # 右摇杆左右 (axes[3])
        else:
            # 如果没按住 LB+RB，则指令为 0
            self.commands.lin_vel_x = 0.0
            self.commands.lin_vel_y = 0.0
            self.commands.ang_vel_yaw = 0.0

        # 5.1 (旧) 应用指令缩放 (这部分逻辑不变)
        commands_scaled = np.array([
            self.commands.lin_vel_x * Go2Config.COMMANDS_SCALES.lin_vel_x,
            self.commands.lin_vel_y * Go2Config.COMMANDS_SCALES.lin_vel_y,
            self.commands.ang_vel_yaw * Go2Config.COMMANDS_SCALES.ang_vel_yaw
        ], dtype=np.float32)

        # 6. 拼接最终观测向量 (这部分逻辑不变)
        # (self.last_torques 和 self.motor_fatigue 是SATA状态，保留)
        observation = np.concatenate([
            obs_lin_vel,        # 3
            obs_ang_vel,        # 3
            projected_gravity,  # 3
            obs_dof_pos,        # 12
            obs_dof_vel,        # 12
            commands_scaled,    # 3
            self.last_torques,  # 12 (SATA 特定)
            self.motor_fatigue  # 12 (SATA 特定)
        ]) # 总共 60 维
        
        return observation.astype(np.float32)


    def _compute_sata_torques(self, raw_actions, dof_vel):
        """
        实现 SATA 生物力学模型 (Eq. 1-4)。
        对应 `go2_torque.py` 的 `_compute_torques` 方法。
        """
        
        # 1. 缩放动作 (Eq. 1 - 部分)
        # 对应 `go2_torque_config.py -> Actions.action_scale`
        actions_scaled = raw_actions * Go2Config.ACTION_SCALE # κ_scale
        # 对应 `go2_torque_config.py -> ControlCfg.torque_limits`
        torques_limits = self.torque_limits                 # τ_limit

        # 2. 激活模型 (Eq. 1 & 2)
        # 对应 `go2_torque_config.py -> ControlCfg.activation_smooth_factor` (GAMMA)
        current_activation_sign = np.tanh(actions_scaled / torques_limits)
        self.activation_sign = (current_activation_sign - self.activation_sign) * self.GAMMA + self.activation_sign

        # 3. 肌肉模型 (Eq. 3)
        # 对应 `go2_torque_config.py -> AssetCfg.dof_vel_limit`
        torques = self.activation_sign * torques_limits * (
            1 - np.sign(self.activation_sign) * dof_vel[:self.num_actions] / self.dof_vel_limits
        )

        # 4. 内部状态模型 (疲劳) (Eq. 4)
        # 对应 `go2_torque_config.py -> ControlCfg.fatigue_recovery_factor` (BETA)
        dt = 0.005 # 200Hz 控制周期
        self.motor_fatigue += np.abs(torques) * dt
        self.motor_fatigue *= self.BETA
        
        return torques.astype(np.float32)

    def _run_policy_logic(self):
        """
        策略的主循环，由 ROS2 定时器 (self.policy_timer) 触发。
        """
        
        # 1. 获取观测
        #    (self.get_observation 已经过修改，从ROS话题和手柄获取数据)
        observation = self.get_observation()
        
        # 2. 策略推理 (获取原始动作)
        raw_action = self.policy.get_action(observation)
        
        # 3. 计算 SATA 力矩
        #    (我们使用 self.current_dof_vel 替代 `self.data.qvel`)
        dof_vel = self.current_dof_vel
        final_torques = self._compute_sata_torques(raw_action, dof_vel) #

        # 4. 存储状态 (用于下一次观测)
        #    (self.motor_fatigue 已经在 _compute_sata_torques 中被更新)
        self.last_torques = final_torques.copy() #
        
        # 5. 发布力矩到 ROS 话题
        msg = Float32MultiArray()
        msg.data = final_torques.astype(float).tolist()
        self.torque_pub.publish(msg)
    # def step(self):
    #     """
    #     执行一个模拟步骤（获取观测、推理、计算力矩、应用力矩、步进）。
    #     对应 `legged_robot.py` 的 `step` 和 `go2_torque.py` 的 `step` 方法。
    #     """
    #     # 1. 获取观测
    #     observation = self.get_observation()
        
    #     # 2. 策略推理 (获取原始动作)
    #     raw_action = self.policy.get_action(observation)
        
    #     # 3. 计算 SATA 力矩
    #     dof_vel = self.data.qvel[6:6+self.num_actions].copy()
    #     final_torques = self._compute_sata_torques(raw_action, dof_vel) # 对应 `_compute_torques`

    #     # 4. 应用执行器映射 (Sim2Sim 必需)
    #     mapped_torques = final_torques[self.policy_to_mujoco_actuator_map]
        
    #     # 5. 应用力矩
    #     # 对应 `legged_robot.py` 中的 `self.gym.set_dof_actuation_force_tensor`
    #     self.data.ctrl[:self.num_actions] = mapped_torques

    #     # 6. 步进 MuJoCo 仿真
    #     # 对应 `legged_robot.py` 中的 `self.gym.simulate`
    #     mujoco.mj_step(self.model, self.data)

    #     # 7. 存储状态 (用于下一次观测)
    #     # 对应 `go2_torque.py` `step` 方法中的 `self.last_torques[:] = ...`
    #     self.last_torques = final_torques.copy()
    #     self.step_count += 1

    # def run_simulation(self):
    #     """
    #     运行主仿真循环。
    #     对应 `legged_gym/scripts/play.py` 的主循环。
    #     """
    #     # 1. 初始化并启动查看器
    #     if self.viewer is None:
    #          print("Launching passive viewer for run_simulation.")
    #          try:
    #              self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    #          except Exception as e:
    #              print(f"无法启动 MuJoCo viewer: {e}")
    #              return
        
    #     with self.viewer:
    #         # 2. 重置仿真环境
    #         # 对应 `legged_robot.py` 的 `reset` / `_reset_dofs` / `_reset_root_states`
    #         mujoco.mj_resetData(self.model, self.data)
    #         self.data.qpos[0:7] = [0, 0, 0.42, 1, 0, 0, 0] # 初始位置和姿态
            
    #         # 3. 添加初始状态扰动
    #         # 对应 `legged_robot.py` `_reset_dofs` 中的随机化
    #         noise_scale = 0.05
    #         perturbed_dof_pos = self.default_dof_pos * (1 + noise_scale * (np.random.rand(self.num_actions) * 2 - 1))
    #         self.data.qpos[7:7+self.num_actions] = perturbed_dof_pos
    #         self.data.qvel[:] = 0.0 # 初始速度设为 0
            
    #         # 4. 重置 SATA 内部状态
    #         # 对应 `go2_torque.py` 的 `reset_idx`
    #         self.last_torques.fill(0)
    #         self.motor_fatigue.fill(0)
    #         self.activation_sign.fill(0)
    #         self.step_count = 0

    #         print("Running simulation with initial state perturbation.")

    #         # 5. 仿真主循环
    #         while self.viewer.is_running():
    #             step_start = time.time()
                
    #             # 6. 设置指令
    #             # 对应 `play.py` 中设置固定指令: `env.commands[:, 0] = 0.5`
    #             self.commands.lin_vel_x = 0.0 # 向前走
    #             self.commands.lin_vel_y = 0.0
    #             self.commands.ang_vel_yaw = 0.0

    #             # 7. 执行一步仿真
    #             self.step()

    #             # 8. 同步查看器
    #             self.viewer.sync()

    #             # 9. 控制循环频率 (实时同步)
    #             # 对应 `play.py` 中的 `time.sleep(self.dt)`
    #             time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
    #             if time_until_next_step > 0:
    #                 time.sleep(time_until_next_step)

# if __name__ == '__main__':
#     # 确保路径正确
#     model_path = '/home/qiwang/sata_mujoco/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/resources/go2/scene.xml'
#     policy_path = '/home/qiwang/sata_mujoco/legged_gym/logs/SATA/exported/policies/policy_1.pt'

#     simulator = MujocoSimulator(model_path, policy_path)
#     simulator.run_simulation()

def main(args=None):
    rclpy.init(args=args)
    
    # 1. 定义策略文件路径
    # (确保这个路径对你的 ROS2 环境是正确的)
    policy_path = '/home/qiwang/sata_mujoco/legged_gym/logs/SATA/exported/policies/policy_1.pt' #
    
    try:
        # 2. 实例化我们的新节点
        policy_node = TorquePolicyNode(policy_path)
        
        # 3. 运行节点，等待回调
        rclpy.spin(policy_node)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"节点运行时发生错误: {e}")
    finally:
        # 4. 清理
        if 'policy_node' in locals() and rclpy.ok():
            policy_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
