import mujoco
import mujoco.viewer
import numpy as np
import time
import torch

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

        self.last_torques = np.zeros(self.num_actions)
        self.motor_fatigue = np.zeros(self.num_actions)
        self.activation_sign = np.zeros(self.num_actions)
        self.dof_vel_limits = np.full(self.num_actions, 30.0)
        self.step_count = 0

        print("MuJoCo Simulator initialized for SATA TORQUE CONTROL.")

    def get_default_dof_pos_ordered(self):
        # ... (保持不变)
        default_dof_pos_dict = Go2Config.DEFAULT_JOINT_ANGLES
        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        ordered_dof_pos = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            joint_name = joint_names[i+1]
            ordered_dof_pos[i] = default_dof_pos_dict[joint_name]
        return ordered_dof_pos

    def get_observation(self):
        # ... (保持不变)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        base_lin_vel = self.data.sensor("base_lin_vel").data.copy()
        base_ang_vel = self.data.sensor("base_ang_vel").data.copy()
        base_quat_wxyz = self.data.sensor("base_quat").data.copy()
        q_w, q_x, q_y, q_z = base_quat_wxyz
        gravity_vec = np.array([0., 0., -1.])
        v_part = np.array([gravity_vec[0], gravity_vec[1], gravity_vec[2]])
        q_v = np.array([q_x, q_y, q_z])
        t = 2.0 * np.cross(q_v, v_part)
        projected_gravity = v_part + q_w * t + np.cross(q_v, t)
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
        observation = np.concatenate([
            obs_lin_vel, obs_ang_vel, projected_gravity,
            obs_dof_pos, obs_dof_vel, commands_scaled,
            self.last_torques, self.motor_fatigue
        ])
        return observation

    def _compute_sata_torques(self, raw_actions, dof_vel):
        # ... (保持不变)
        actions_scaled = raw_actions * Go2Config.ACTION_SCALE
        torque_limits = Go2Config.TORQUE_LIMITS # 使用统一的力矩限制，除非后续发现膝关节需要特殊处理
        current_activation_sign = np.tanh(actions_scaled / torque_limits)
        gamma = 0.6
        self.activation_sign = (current_activation_sign - self.activation_sign) * gamma + self.activation_sign
        safe_dof_vel_limits = self.dof_vel_limits + 1e-6
        torques = self.activation_sign * torque_limits * (
            1 - np.sign(self.activation_sign) * dof_vel / safe_dof_vel_limits
        )
        beta = 0.9
        dt = self.model.opt.timestep
        self.motor_fatigue += np.abs(torques) * dt
        self.motor_fatigue *= beta
        return torques

    def step(self):
        # ... (保持不变，但移除了诊断打印)
        observation = self.get_observation()
        raw_action = self.policy.get_action(observation)
        dof_vel = self.data.qvel[6:].copy()
        final_torques = self._compute_sata_torques(raw_action, dof_vel)
        self.data.ctrl[:] = final_torques
        mujoco.mj_step(self.model, self.data)
        self.last_torques = final_torques.copy()
        self.step_count += 1 # 计数器仍然保留

    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # ---
            # 关键修改：在重置时加入扰动
            # ---
            mujoco.mj_resetData(self.model, self.data)
            # 初始位置和姿态
            self.data.qpos[0:7] = [0, 0, 0.44, 1, 0, 0, 0]
            # 初始关节角度 + 微小随机扰动 (符合SATA训练中的做法)
            # -> _reset_dofs
            noise_scale = 0.05 # 可以调整这个扰动的大小
            perturbed_dof_pos = self.default_dof_pos * (1 + noise_scale * (np.random.rand(self.num_actions) * 2 - 1))
            self.data.qpos[7:] = perturbed_dof_pos
            # 初始速度设为0 (SATA代码也是这样做的)
            self.data.qvel[:] = 0.0
            # 重置内部状态
            self.last_torques.fill(0)
            self.motor_fatigue.fill(0) # 可以考虑像SATA一样加入初始疲劳随机化
            self.activation_sign.fill(0)
            self.step_count = 0

            print("Running simulation with initial state perturbation.")

            while viewer.is_running():
                step_start = time.time()
                self.commands.lin_vel_x = 0.5
                self.commands.lin_vel_y = 0.0
                self.commands.ang_vel_yaw = 0.0
                try:
                    self.step()
                except Exception as e:
                    print(f"!!!!!!!!!! SIMULATION ERROR: {e} !!!!!!!!!!")
                    print("Stopping simulation due to instability.")
                    break
                viewer.sync()
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == '__main__':
    model_path = '/home/qiwang/sata_mujoco/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/resources/go2/go2.xml'
    policy_path = '/home/qiwang/sata_mujoco/legged_gym/logs/SATA/exported/policies/policy_1.pt'
    simulator = MujocoSimulator(model_path, policy_path)
    simulator.run_simulation()