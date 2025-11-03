import mujoco
import mujoco.viewer
import numpy as np
import time
import torch
import os # 导入 os 模块用于文件检查
# import traceback # 移除错误打印，不再需要 traceback

# 假设 rl_policy 和 config 在同一目录下或 PYTHONPATH 中
from .rl_policy import RLPolicy
from .config import Go2Config, Commands

class MujocoSimulator:
    def __init__(self, model_path, policy_path):
        """
        构造函数：初始化模拟器、模型、数据和策略。
        对应 SATA 中的 `legged_robot.py` 和 `go2_torque.py` 的 `__init__` 方法。
        """
        
        # 1. 加载 MuJoCo 模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"错误：找不到模型文件 {model_path}")
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # 2. 仿真时间步长 (dt)
        # 对应 SATA `legged_robot_config.py` 中的 `cfg.sim.dt = 0.005`
        target_timestep = 0.005 
        print(f"从 XML 加载的原始 timestep: {self.model.opt.timestep}")
        self.model.opt.timestep = target_timestep
        print(f"在 Python 中将 timestep 覆盖为: {self.model.opt.timestep}")

        # 3. 加载策略
        # 对应 SATA `play.py` 中加载策略的部分
        self.policy = RLPolicy(policy_path)
        self.commands = Commands() # 对应 `legged_robot.py` 中的 `self.commands`
        self.num_actions = 12

        # 4. 获取默认关节角度
        # 对应 `legged_robot.py` 的 `_init_dofs` 和 `go2_torque_config.py` 的 `InitState.default_joint_angles`
        self.default_dof_pos = self.get_default_dof_pos_ordered()

        # --- SATA MODIFICATION: 初始化生物力学模型状态和参数 ---
        # 对应 `go2_torque.py` 的 `__init__` 方法中对状态变量的初始化
        self.last_torques = np.zeros(self.num_actions, dtype=np.float32)
        self.motor_fatigue = np.zeros(self.num_actions, dtype=np.float32)
        self.activation_sign = np.zeros(self.num_actions, dtype=np.float32)

        # 从 Go2Config 加载 SATA 特定参数
        # 对应 `go2_torque_config.py` 中的 `AssetCfg.dof_vel_limit`
        self.dof_vel_limits = np.full(self.num_actions, Go2Config.DOF_VEL_LIMITS, dtype=np.float32)
        # 对应 `go2_torque_config.py` 中的 `ControlCfg.torque_limits`
        self.torque_limits = np.full(self.num_actions, Go2Config.TORQUE_LIMITS, dtype=np.float32)
        # 对应 `go2_torque_config.py` 中的 `ControlCfg.activation_smooth_factor`
        self.GAMMA = Go2Config.GAMMA # 激活平滑因子
        # 对应 `go2_torque_config.py` 中的 `ControlCfg.fatigue_recovery_factor`
        self.BETA = Go2Config.BETA   # 疲劳恢复因子
        # --- END SATA MODIFICATION ---

        self.step_count = 0

        # --- 执行器映射 ---
        # SATA 训练时使用的 URDF (FL,FR,RL,RR) 与官方 `go2.xml` 的 <actuator> 顺序不同。
        self.policy_to_mujoco_actuator_map = np.array([
            3, 4, 5,    # 策略 FL (0-2) -> MuJoCo FL (3-5)
            0, 1, 2,    # 策略 FR (3-5) -> MuJoCo FR (0-2)
            9, 10, 11,  # 策略 RL (6-8) -> MuJoCo RL (9-11)
            6, 7, 8     # 策略 RR (9-11)-> MuJoCo RR (6-8)
        ])
        # --- 结束新增代码 ---
        
        self.viewer = None # 初始化查看器
        print("MuJoCo Simulator initialized for SATA TORQUE CONTROL.")
        print(f"Using timestep: {self.model.opt.timestep}")

    def get_default_dof_pos_ordered(self):
        """
        获取与 MuJoCo 模型关节顺序一致的默认关节角度。
        对应 `legged_robot.py` 的 `_init_dofs` 和 `go2_torque_config.py` 的 `InitState.default_joint_angles`。
        """
        default_dof_pos_dict = Go2Config.DEFAULT_JOINT_ANGLES
        # 跳过 freejoint (索引0)
        joint_names = [self.model.joint(i).name for i in range(1, self.model.njnt)]
        ordered_dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        
        num_actuated_joints = min(self.num_actions, len(joint_names))
        for i in range(num_actuated_joints):
            joint_name = joint_names[i]
            if joint_name in default_dof_pos_dict:
                 ordered_dof_pos[i] = default_dof_pos_dict[joint_name]
            else:
                 print(f"警告：在 Go2Config.DEFAULT_JOINT_ANGLES 中未找到关节 '{joint_name}' 的默认位置。将使用 0。")

        if num_actuated_joints < self.num_actions:
            print(f"警告：模型只有 {num_actuated_joints} 个驱动关节，但期望 {self.num_actions} 个动作。")

        return ordered_dof_pos


    def get_observation(self):
        """
        构建观测向量。
        对应 `go2_torque.py` 的 `compute_observations` 方法。
        """
        # 1. 获取基础状态 (对应 `legged_robot.py` compute_observations)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        # 对应 `self.base_lin_vel` (从 `frame_vel` 传感器读取)
        base_lin_vel = self.data.sensor("frame_vel").data.copy()
        # 对应 `self.base_ang_vel` (从 `imu_gyro` 传感器读取)
        base_ang_vel = self.data.sensor("imu_gyro").data.copy()
        # 对应 `self.base_quat` (从 `imu_quat` 传感器读取)
        base_quat_wxyz = self.data.sensor("imu_quat").data.copy()
        
        # 2. 计算重力向量 (对应 `legged_robot.py` compute_observations)
        q_w, q_x, q_y, q_z = base_quat_wxyz
        gravity_vec = np.array([0., 0., -1.], dtype=np.float32)
        q_calc = np.array([q_x, q_y, q_z, q_w], dtype=np.float32)
        w, x, y, z = q_calc[3], q_calc[0], q_calc[1], q_calc[2]
        qv = np.array([x, y, z], dtype=np.float32)
        uv = np.cross(qv, gravity_vec)
        uuv = np.cross(qv, uv)
        # 对应 `self.projected_gravity`
        projected_gravity = gravity_vec + 2 * (w * uv + uuv)

        # 3. 获取关节状态 (对应 `legged_robot.py` compute_observations)
        # 对应 `self.dof_pos`
        joint_pos = qpos[7:7+self.num_actions]
        # 对应 `self.dof_vel`
        joint_vel = qvel[6:6+self.num_actions]

        # 4. 应用观测值缩放 (对应 `go2_torque_config.py -> ObservationScales`)
        obs_lin_vel = base_lin_vel * Go2Config.OBS_SCALES.lin_vel
        obs_ang_vel = base_ang_vel * Go2Config.OBS_SCALES.ang_vel
        # 对应 `(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos`
        obs_dof_pos = (joint_pos - self.default_dof_pos) * Go2Config.OBS_SCALES.dof_pos
        obs_dof_vel = joint_vel * Go2Config.OBS_SCALES.dof_vel
        
        # 5. 应用指令缩放 (对应 `go2_torque_config.py -> Commands.scales`)
        # 对应 `self.commands[:, :3] * self.commands_scale`
        commands_scaled = np.array([
            self.commands.lin_vel_x * Go2Config.COMMANDS_SCALES.lin_vel_x,
            self.commands.lin_vel_y * Go2Config.COMMANDS_SCALES.lin_vel_y,
            self.commands.ang_vel_yaw * Go2Config.COMMANDS_SCALES.ang_vel_yaw
        ], dtype=np.float32)

        # 6. 拼接最终观测向量
        # 对应 `go2_torque.py` `compute_observations` 中的 `self.obs_buf = torch.cat(...)`
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
        
        return observation.astype(np.float32) # 确保是 float32


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
        dt = self.model.opt.timestep
        self.motor_fatigue += np.abs(torques) * dt
        self.motor_fatigue *= self.BETA
        
        return torques.astype(np.float32)

    def step(self):
        """
        执行一个模拟步骤（获取观测、推理、计算力矩、应用力矩、步进）。
        对应 `legged_robot.py` 的 `step` 和 `go2_torque.py` 的 `step` 方法。
        """
        # 1. 获取观测
        observation = self.get_observation()
        
        # 2. 策略推理 (获取原始动作)
        raw_action = self.policy.get_action(observation)
        
        # 3. 计算 SATA 力矩
        dof_vel = self.data.qvel[6:6+self.num_actions].copy()
        final_torques = self._compute_sata_torques(raw_action, dof_vel) # 对应 `_compute_torques`

        # 4. 应用执行器映射 (Sim2Sim 必需)
        mapped_torques = final_torques[self.policy_to_mujoco_actuator_map]
        
        # 5. 应用力矩
        # 对应 `legged_robot.py` 中的 `self.gym.set_dof_actuation_force_tensor`
        self.data.ctrl[:self.num_actions] = mapped_torques

        # 6. 步进 MuJoCo 仿真
        # 对应 `legged_robot.py` 中的 `self.gym.simulate`
        mujoco.mj_step(self.model, self.data)

        # 7. 存储状态 (用于下一次观测)
        # 对应 `go2_torque.py` `step` 方法中的 `self.last_torques[:] = ...`
        self.last_torques = final_torques.copy()
        self.step_count += 1

    def run_simulation(self):
        """
        运行主仿真循环。
        对应 `legged_gym/scripts/play.py` 的主循环。
        """
        # 1. 初始化并启动查看器
        if self.viewer is None:
             print("Launching passive viewer for run_simulation.")
             try:
                 self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
             except Exception as e:
                 print(f"无法启动 MuJoCo viewer: {e}")
                 return
        
        with self.viewer:
            # 2. 重置仿真环境
            # 对应 `legged_robot.py` 的 `reset` / `_reset_dofs` / `_reset_root_states`
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[0:7] = [0, 0, 0.42, 1, 0, 0, 0] # 初始位置和姿态
            
            # 3. 添加初始状态扰动
            # 对应 `legged_robot.py` `_reset_dofs` 中的随机化
            noise_scale = 0.05
            perturbed_dof_pos = self.default_dof_pos * (1 + noise_scale * (np.random.rand(self.num_actions) * 2 - 1))
            self.data.qpos[7:7+self.num_actions] = perturbed_dof_pos
            self.data.qvel[:] = 0.0 # 初始速度设为 0
            
            # 4. 重置 SATA 内部状态
            # 对应 `go2_torque.py` 的 `reset_idx`
            self.last_torques.fill(0)
            self.motor_fatigue.fill(0)
            self.activation_sign.fill(0)
            self.step_count = 0

            print("Running simulation with initial state perturbation.")

            # 5. 仿真主循环
            while self.viewer.is_running():
                step_start = time.time()
                
                # 6. 设置指令
                # 对应 `play.py` 中设置固定指令: `env.commands[:, 0] = 0.5`
                self.commands.lin_vel_x = 0.8 # 向前走
                self.commands.lin_vel_y = 0.0
                self.commands.ang_vel_yaw = 0.0

                # 7. 执行一步仿真
                self.step()

                # 8. 同步查看器
                self.viewer.sync()

                # 9. 控制循环频率 (实时同步)
                # 对应 `play.py` 中的 `time.sleep(self.dt)`
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == '__main__':
    # 确保路径正确
    model_path = '/home/qiwang/sata_mujoco/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/resources/go2/scene.xml'
    policy_path = '/home/qiwang/sata_mujoco/legged_gym/logs/SATA/exported/policies/policy_1.pt'

    simulator = MujocoSimulator(model_path, policy_path)
    simulator.run_simulation()
