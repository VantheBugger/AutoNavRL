import os
import gym
from gym import spaces
import numpy as np
import argparse
import torch
from pvo_network import PVONetwork  # 导入PVO架构

from sim import SIM_ENV
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# # 1. 在主程序运行前加载 PVO 模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pvo_predictor = PVONetwork().to(device)
# try:
#     pvo_predictor.load_state_dict(torch.load("pvo_model_weights.pth", map_size=device))
#     pvo_predictor.eval() # 设为评估模式，关闭 Dropout/BatchNorm
#     print("PVO 预训练模型加载成功！")
# except:
#     print("未找到 PVO 权重，将使用随机初始化（不建议）")

class TensorboardCallback(BaseCallback):
    """
    自定义回调函数：实时从环境 info 中提取数据并记录到 TensorBoard
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_successes = []
        self.episode_collisions = []

    def _on_step(self) -> bool:
        # locals['infos'] 包含了当前步所有并行环境返回的 info 字典
        for info in self.locals['infos']:
            # 当一个 episode 结束时，info 会包含终端信息
            if "terminal_observation" in info:
                # 记录成功或失败
                if 'success' in info:
                    self.episode_successes.append(1.0 if info['success'] else 0.0)
                if 'collision' in info:
                    self.episode_collisions.append(1.0 if info['collision'] else 0.0)

        # 每 1000 步计算一次最近 50 个回合的平均值并写入 TensorBoard
        if self.n_calls % 1000 == 0 and len(self.episode_successes) > 0:
            avg_success = np.mean(self.episode_successes[-50:])
            avg_collision = np.mean(self.episode_collisions[-50:])
            
            self.logger.record("metrics/success_rate", avg_success)
            self.logger.record("metrics/collision_rate", avg_collision)
            
        return True

class RobotNavEnv(gym.Env):
    """
    Custom Gym environment that wraps the SIM_ENV simulator.
    
    This environment converts the simulator's outputs into a fixed-size observation,
    defines an action space, and scales actions as required for reinforcement learning.
    """
    
    def __init__(self, render=False, pvo_model_path="pvo_model_weights.pth"):
        """
        Initialize the robot navigation environment.
        
        Args:
            render (bool): Whether to enable visualization
        """
        super(RobotNavEnv, self).__init__()
        
        # 使用绝对路径确保子进程能找到文件
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_pvo_path = os.path.join(base_path, pvo_model_path)

        # Environment configuration
        self.render = render
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- PVO 模型加载 ---
        self.pvo_model = PVONetwork().to(self.device)
        if os.path.exists(full_pvo_path):
            try:
                self.pvo_model.load_state_dict(torch.load(full_pvo_path, map_location=self.device))
                self.pvo_model.eval()
                # 只有第一个环境打印成功信息，避免 7 个进程一起刷屏
                if render: print(f"成功从 {full_pvo_path} 加载 PVO 权重！")
            except Exception as e:
                print(f"加载 PVO 权重出错: {e}")
        else:
            print(f"致命错误: 在 {full_pvo_path} 未找到权重文件！")

        # Add dimensions for the PVO latent representation or probability map
        # e.g., Let's assume the PVO network outputs a 10-dimensional feature vector 
        # representing the collision risk in different velocity sectors.
        self.pvo_feature_dim = 10
        self.state_dim = 49 + 42 + self.pvo_feature_dim # 增加 42 维，用于存储雷达数据的变化率 (dx/dt), 增加10维用于PVO  # Dimension of the observation space
        self.max_steps = 300  # Maximum number of steps per episode
        
        # Define action space (linear and angular velocity)
        self.action_space = spaces.Box(
            low=np.array([-0.6, -1.2]),  # [min_linear_vel, min_angular_vel]
            high=np.array([0.6, 1.2]),   # [max_linear_vel, max_angular_vel]
            dtype=np.float32
        )
        
        # Define observation space (normalized to [-1, 1])
        self.observation_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )

        # 用于存储上一帧归一化后的雷达数据
        self.last_min_values = None

        # Initialize simulator
        self.sim = SIM_ENV(render=render)

        # Initialize episode tracking
        self._reset_episode_tracking()
        
        # Get initial observation
        initial_data = self.sim.reset()
        self.current_obs, _ = self.prepare_state(initial_data, current_action=[0.0, 0.0]) # 初始动作为静止

    def _get_pvo_features(self, current_scan, scan_velocity, action):
        """
        Query the PVO network to get future collision probabilities 
        and state features.
        """
        if self.pvo_model is None:
            # Fallback if no model is loaded
            return [0.0] * self.pvo_feature_dim, 0.0
        
        # --- PSEUDOCODE FOR PVO INFERENCE ---
        # 1. Format inputs for the network (historical scans, current action)
        # 2. pvo_features, collision_prob = self.pvo_model.predict(current_scan, scan_velocity, action)
        # return pvo_features.tolist(), collision_prob
        
        # Placeholder return
        return [0.0] * self.pvo_feature_dim, 0.0

    def _reset_episode_tracking(self):
        """Reset all episode tracking variables."""
        self.time = 0
        self.last_position = None
        self.total_distance = 0
        self.total_velocity = 0
        self.steps = 0

    def _calculate_metrics(self, current_position, action):
        """
        Calculate and update episode metrics.
        
        Args:
            current_position: Current robot position [x, y]
            action: Current action [linear_vel, angular_vel]
        """
        if self.last_position is not None:
            step_distance = np.linalg.norm(current_position - self.last_position)
            self.total_distance += step_distance
            self.total_velocity += np.linalg.norm(action)
        self.last_position = current_position
        self.steps += 1

    def _get_episode_info(self, terminal, reward):
        """
        Generate episode information dictionary.
        
        Args:
            terminal (bool): Whether episode is terminal
            reward (float): Final reward
            
        Returns:
            dict: Episode information
        """
        avg_velocity = self.total_velocity / self.steps if self.steps > 0 else 0
        return {
            'success': terminal and reward > 0,
            'collision': terminal and reward < 0,
            'steps': self.steps,
            'total_distance': self.total_distance,
            'average_velocity': avg_velocity,
            'time_limit_reached': self.time >= self.max_steps
        }

    def prepare_state(self, data, current_action):
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward = data
        latest_scan = np.array(latest_scan)

        # 1. 常规处理：处理 Lidar 无穷远值
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 8

        # 2. 下采样逻辑 (保持你原来的 bins 计算)
        max_bins = 42 # self.state_dim - 7 原来的逻辑
        remainder = len(latest_scan) % max_bins
        if remainder != 0:
            extension = np.full(max_bins - remainder, latest_scan[-1])
            latest_scan = np.concatenate((latest_scan, extension))

        bin_size = int(len(latest_scan) / max_bins)
        current_min_values = []
        for i in range(0, len(latest_scan), bin_size):
            bin_data = latest_scan[i : i + bin_size]
            current_min_values.append(min(bin_data) / 8.0) # 归一化到 [0, 1]

        # 3. 预测逻辑：计算雷达变化率 (Velocity Feature)
        if self.last_min_values is None:
            # 第一帧，变化率为 0
            scan_velocity = [0.0] * max_bins
        else:
            # 计算导数：(当前 - 过去) 
            # 负值表示障碍物靠近，正值表示远离
            scan_velocity = [curr - prev for curr, prev in zip(current_min_values, self.last_min_values)]
        
        self.last_min_values = current_min_values # 更新缓存

        # --- NEW: Fetch PVO logic ---
        pvo_features, self.current_pvo_prob = self._get_pvo_features(current_min_values, scan_velocity, current_action)

        # 4. 其他状态标准化
        distance_norm = distance / 12.0
        lin_vel = (action[0] + 0.6) / 1.2
        ang_vel = (action[1] + 1.2) / 2.4
        rad_cos = np.cos(diff_rad)
        rad_sin = np.sin(diff_rad)

        # 5. 组合最终状态向量
        # 现在的状态包含：[当前雷达值(42), 雷达变化率(42), pvo_features(?) ,距离(1), 目标Cos(1), 目标Sin(1), 线速度(1), 角速度(1), 航向Cos(1), 航向Sin(1)]
        # 总维度：42 + 42 + 7 = 91
        state = current_min_values + scan_velocity + pvo_features + [distance_norm, cos, sin] + \
                [lin_vel, ang_vel] + [rad_cos, rad_sin]
        
        terminal = 1 if collision or goal else 0
        return np.array(state, dtype=np.float32), terminal

    def reset(self):
        """
        Reset the environment and return initial observation.
        
        Returns:
            numpy.ndarray: Initial observation
        """
        sim_data = self.sim.reset()
        obs, _ = self.prepare_state(sim_data, current_action=[0.0, 0.0]) # 初始动作为静止
        self.current_obs = obs
        self._reset_episode_tracking()
        return obs

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: [linear_velocity, angular_velocity]
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Process actions with deadzone
        # 移除死区，直接透传物理动作
        lin_velocity = action[0]
        ang_velocity = action[1]

        # Step simulation
        sim_data = self.sim.step(lin_velocity=lin_velocity, ang_velocity=ang_velocity)
        obs, terminal = self.prepare_state(sim_data, action)
        reward = sim_data[-1]

        # Update metrics
        current_position = self.sim.env.get_robot_state()[:2]
        self._calculate_metrics(current_position, action)

        # Check termination conditions
        done = terminal
        self.time += 1
        if self.time >= self.max_steps:
            done = True
            reward = -100

        # Generate info dictionary
        info = self._get_episode_info(terminal, reward)
        
        self.current_obs = obs
        return obs, reward, done, info

class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq=100, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        # 用于记录每个并行环境当前回合的累积奖励
        self.current_episode_rewards = None 
        # 用于存储已完成回合的总奖励历史
        self.episode_final_rewards = []

    def _on_training_start(self) -> None:
        # 初始化累积奖励数组，大小等于并行环境的数量
        n_envs = self.training_env.num_envs
        self.current_episode_rewards = np.zeros(n_envs)

    def _on_step(self) -> bool:
        # 1. 将当前步的即时奖励累加到各自的环境中
        # self.locals['rewards'] 包含了这一步所有环境给出的奖励
        self.current_episode_rewards += self.locals['rewards']

        # 2. 检查哪些环境的回合结束了
        # self.locals['dones'] 是一个布尔数组
        for i, done in enumerate(self.locals['dones']):
            if done:
                # 记录该环境完成的回合总分
                self.episode_final_rewards.append(self.current_episode_rewards[i])
                # 重置该环境的累加器，准备下一个回合
                self.current_episode_rewards[i] = 0

        # 3. 定期将数据写入 TensorBoard
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_final_rewards) > 0:
                # 计算最近 20 个回合的平均奖励
                mean_reward = np.mean(self.episode_final_rewards[-20:])
                self.logger.record("rollout/custom_ep_rew_mean", mean_reward)
                
                # 顺便记录一下单步奖励的均值，看看波动
                self.logger.record("train/step_reward_raw", np.mean(self.locals['rewards']))

        return True

def make_env(render=False):
    """
    Utility function for creating new instances of RobotNavEnv.
    This is used to create multiple parallel environments.
    
    Args:
        render (bool): Whether to enable visualization
        
    Returns:
        function: Environment initialization function
    """
    def _init():
        env = RobotNavEnv(render=render) 
        return env
    return _init


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train TD3 model for robot navigation')
    parser.add_argument('--num-envs', type=int, default=7,
                       help='Number of parallel training environments')
    parser.add_argument('--total-timesteps', type=int, default=200000,
                       help='Total timesteps to train for')
    parser.add_argument('--model-path', type=str, default="models/td3_robot_nav_model",
                       help='Path to save/load model')
    parser.add_argument('--tensorboard-log', type=str, default="./td3_robot_nav_tensorboard/",
                       help='Tensorboard log directory')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering during training')
    args = parser.parse_args()

    # Create environments
    env_fns = [make_env() for _ in range(args.num_envs)]
    env_fns.append(make_env(render=args.render))  # Add one environment with optional rendering
    try:
        env = SubprocVecEnv(env_fns)
    except Exception as e:
        print(f"SubprocVecEnv 初始化失败，回退到 DummyVecEnv。原因: {e}")
        env = DummyVecEnv(env_fns)

    # Create or load the TD3 model
    try:
        model = TD3.load(args.model_path, env=env)
        print(f"Loaded existing model from {args.model_path}")
    except:
        model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log)
        print("Created new model")

    # 3. 实例化我们刚写的 Callback
    reward_callback = RewardLoggerCallback(check_freq=500) # 每 500 步刷新一次曲线
    nav_callback = TensorboardCallback()

    from stable_baselines3.common.callbacks import CallbackList
    callback_list = CallbackList([reward_callback, nav_callback])

    # Train the model
    # model.learn(total_timesteps=args.total_timesteps)
    model.learn(
        total_timesteps=args.total_timesteps, 
        callback=callback_list,           # 关键：注入回调
        tb_log_name="TD3_PVO_Run_3_23"         # 在 TensorBoard 中显示的运行名称
    )

    # Evaluate the trained model
    eval_env = DummyVecEnv([make_env()])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
    # --- 开始评估逻辑 (模仿 run.py) ---
    print("\nStarting Final Evaluation...")
    eval_env = DummyVecEnv([make_env(render=False)])
    
    success_count = 0
    total_timesteps = 0
    total_distance = 0
    total_velocity = 0
    collision_count = 0
    time_limit_count = 0
    
    for episode in range(args.eval_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

        # 提取当前 episode 的 info (VecEnv 返回的是 list)
        ep_info = info[0]
        if ep_info['success']: success_count += 1
        if ep_info['collision']: collision_count += 1
        if ep_info['time_limit_reached']: time_limit_count += 1
            
        total_timesteps += ep_info['steps']
        total_distance += ep_info['total_distance']
        total_velocity += ep_info['average_velocity']

    # 计算最终指标
    print("\nFinal Evaluation Metrics (After Training):")
    print(f"Success Rate: {(success_count / args.eval_episodes) * 100:.1f}%")
    print(f"Collision Rate: {(collision_count / args.eval_episodes) * 100:.1f}%")
    print(f"Time Limit Rate: {(time_limit_count / args.eval_episodes) * 100:.1f}%")
    print(f"Avg Timesteps: {total_timesteps / args.eval_episodes:.1f}")
    print(f"Avg Distance: {total_distance / args.eval_episodes:.2f}")
    print(f"Avg Velocity: {total_velocity / args.eval_episodes:.2f}")

    # Save the trained model
    model.save(args.model_path)
