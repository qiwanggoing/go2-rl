import mujoco
import mujoco.viewer
import numpy as np
import time
import torch

# 假设 rl_policy 和 config 在同一目录下或 PYTHONPATH 中
# 如果运行时仍有问题，可能需要调整这里的导入
from .rl_policy import RLPolicy
from .config import Go2Config, Commands

class MujocoSimulator:
    def __init__(self, model_path, policy_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.policy = RLPolicy(policy_path)
        self.commands = Commands()
        self.num_actions = 12
        self.default_dof_pos = self.get_default_dof_pos_ordered()

        # --- SATA MODIFICATION: 初始化生物力学模型状态和参数 ---
        # 来源: go2_torque.py (__init__)
        self.last_torques = np.zeros(self.num_actions)
        self.motor_fatigue = np.zeros(self.num_actions)
        self.activation_sign = np.zeros(self.num_actions)

        # 从 Go2Config 加载 SATA 特定参数
        self.dof_vel_limits = np.full(self.num_actions, Go2Config.DOF_VEL_LIMITS)
        self.torque_limits = np.full(self.num_actions, Go2Config.TORQUE_LIMITS)
        self.GAMMA = Go2Config.GAMMA # 激活平滑因子
        self.BETA = Go2Config.BETA   # 疲劳恢复因子
        # --- END SATA MODIFICATION ---

        self.step_count = 0

        # --- 新增代码：定义执行器映射 ---
        # 策略 (legged_gym / SATA) 输出顺序: FL, FR, RL, RR (索引 0-2, 3-5, 6-8, 9-11)
        # MuJoCo (官方 go2.xml) 执行器顺序: FR, FL, RR, RL (索引 0-2, 3-5, 6-8, 9-11)
        # 这个列表将策略索引映射到 MuJoCo 索引
        # 例：策略的第0个输出 (FL_hip) 应该给 MuJoCo 的第3个执行器 (FL_hip)
        # 例：策略的第3个输出 (FR_hip) 应该给 MuJoCo 的第0个执行器 (FR_hip)
        self.policy_to_mujoco_actuator_map = np.array([
            3, 4, 5,    # 策略 FL (0-2) -> MuJoCo FL (3-5)
            0, 1, 2,    # 策略 FR (3-5) -> MuJoCo FR (0-2)
            9, 10, 11,  # 策略 RL (6-8) -> MuJoCo RL (9-11)
            6, 7, 8     # 策略 RR (9-11)-> MuJoCo RR (6-8)
        ])
        # --- 结束新增代码 ---

        print("MuJoCo Simulator initialized for SATA TORQUE CONTROL.")

    def get_default_dof_pos_ordered(self):
        # ... (保持不变)
        default_dof_pos_dict = Go2Config.DEFAULT_JOINT_ANGLES
        # 跳过 freejoint (索引0)
        joint_names = [self.model.joint(i).name for i in range(1, self.model.njnt)]
        ordered_dof_pos = np.zeros(self.num_actions)
        # 确保只处理12个驱动关节
        num_actuated_joints = min(self.num_actions, len(joint_names))
        for i in range(num_actuated_joints):
            joint_name = joint_names[i]
            if joint_name in default_dof_pos_dict:
                 ordered_dof_pos[i] = default_dof_pos_dict[joint_name]
            else:
                 print(f"警告：在 Go2Config.DEFAULT_JOINT_ANGLES 中未找到关节 '{joint_name}' 的默认位置。")
        # 检查是否所有动作都被赋值
        if num_actuated_joints < self.num_actions:
            print(f"警告：模型只有 {num_actuated_joints} 个驱动关节，但期望 {self.num_actions} 个动作。")

        return ordered_dof_pos


    def get_observation(self):
        # ... (这部分已经修改正确，保持不变)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        base_lin_vel = self.data.sensor("frame_vel").data.copy()
        base_ang_vel = self.data.sensor("imu_gyro").data.copy()
        base_quat_wxyz = self.data.sensor("imu_quat").data.copy()
        q_w, q_x, q_y, q_z = base_quat_wxyz
        gravity_vec = np.array([0., 0., -1.])
        # MuJoCo四元数是 [w, x, y, z]
        # 计算旋转后的重力向量 projected_gravity (代码来自 legged_gym math utils)
        # q = [x, y, z, w] for calculation
        q_calc = np.array([q_x, q_y, q_z, q_w])
        # quat_rotate_inverse
        w, x, y, z = q_calc[3], q_calc[0], q_calc[1], q_calc[2]
        qv = np.array([x, y, z])
        uv = np.cross(qv, gravity_vec)
        uuv = np.cross(qv, uv)
        projected_gravity = gravity_vec + 2 * (w * uv + uuv)

        joint_pos = qpos[7:]
        joint_vel = qvel[6:]
        obs_lin_vel = base_lin_vel * Go2Config.OBS_SCALES.lin_vel
        obs_ang_vel = base_ang_vel * Go2Config.OBS_SCALES.ang_vel
        obs_dof_pos = (joint_pos - self.default_dof_pos) * Go2Config.OBS_SCALES.dof_pos
        obs_dof_vel = joint_vel * Go2Config.OBS_SCALES.dof_vel
        commands_scaled = np.array([
            self.commands.lin_vel_x * Go2Config.COMMANDS_SCALES.lin_vel_x,
            self.commands.lin_vel_y * Go2Config.COMMANDS_SCALES.lin_vel_y,
            self.commands.ang_vel_yaw * Go2Config.COMMANDS_SCALES.ang_vel_yaw
        ])

        # 确保所有部分都是 numpy 数组并且形状正确
        # print(f"Shapes: lin_vel={obs_lin_vel.shape}, ang_vel={obs_ang_vel.shape}, grav={projected_gravity.shape}, dof_pos={obs_dof_pos.shape}, dof_vel={obs_dof_vel.shape}, cmd={commands_scaled.shape}, torques={self.last_torques.shape}, fatigue={self.motor_fatigue.shape}")

        observation = np.concatenate([
            obs_lin_vel, obs_ang_vel, projected_gravity,
            obs_dof_pos, obs_dof_vel, commands_scaled,
            self.last_torques, self.motor_fatigue
        ])
        return observation.astype(np.float32) # 确保是 float32


    def _compute_sata_torques(self, raw_actions, dof_vel):
        # ... (这部分保持不变)
        actions_scaled = raw_actions * Go2Config.ACTION_SCALE # κ_scale
        torques_limits = self.torque_limits                 # τ_limit
        current_activation_sign = np.tanh(actions_scaled / torques_limits)
        self.activation_sign = (current_activation_sign - self.activation_sign) * self.GAMMA + self.activation_sign
        torques = self.activation_sign * torques_limits * (
            1 - np.sign(self.activation_sign) * dof_vel / self.dof_vel_limits
        )
        dt = self.model.opt.timestep
        self.motor_fatigue += np.abs(torques) * dt
        self.motor_fatigue *= self.BETA
        return torques

    def step(self):
        observation = self.get_observation()
        # 确保观测值是 float32
        if observation.dtype != np.float32:
            observation = observation.astype(np.float32)

        raw_action = self.policy.get_action(observation)
        dof_vel = self.data.qvel[6:].copy()
        final_torques = self._compute_sata_torques(raw_action, dof_vel)

        # --- 修改：应用执行器映射 ---
        # 重新排序力矩以匹配 go2.xml 的执行器顺序 (FR, FL, RR, RL)
        mapped_torques = final_torques[self.policy_to_mujoco_actuator_map]
        self.data.ctrl[:] = mapped_torques
        # --- 结束修改 ---

        mujoco.mj_step(self.model, self.data)

        # last_torques 应该存储策略的原始输出 (FL, FR, RL, RR 顺序)
        self.last_torques = final_torques.copy()
        self.step_count += 1

    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # ... (重置部分保持不变) ...
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[0:7] = [0, 0, 0.44, 1, 0, 0, 0]
            noise_scale = 0.05
            perturbed_dof_pos = self.default_dof_pos * (1 + noise_scale * (np.random.rand(self.num_actions) * 2 - 1))
            self.data.qpos[7:] = perturbed_dof_pos
            self.data.qvel[:] = 0.0
            self.last_torques.fill(0)
            self.motor_fatigue.fill(0)
            self.activation_sign.fill(0)
            self.step_count = 0

            print("Running simulation with initial state perturbation.")

            while viewer.is_running():
                step_start = time.time()
                # 设置指令
                self.commands.lin_vel_x = 0.5 # 例如，向前走
                self.commands.lin_vel_y = -0.1
                try:
                    self.step()
                except Exception as e:
                    print(f"!!!!!!!!!! SIMULATION ERROR: {e} !!!!!!!!!!")
                    import traceback
                    traceback.print_exc() # 打印更详细的错误信息
                    print("Stopping simulation due to instability.")
                    break
                viewer.sync()
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == '__main__':
    # 确保路径正确
    model_path = '/home/qiwang/unitree_mujoco/unitree_robots/go2/scene.xml'
    policy_path = '/home/qiwang/sata_mujoco/legged_gym/logs/SATA/exported/policies/policy_1.pt'

    # 添加一些检查确保文件存在
    import os
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        exit()
    if not os.path.exists(policy_path):
        print(f"错误：找不到策略文件 {policy_path}")
        exit()

    simulator = MujocoSimulator(model_path, policy_path)
    simulator.run_simulation()