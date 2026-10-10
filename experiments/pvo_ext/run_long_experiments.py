import os
import sys
import argparse
import numpy as np
import torch
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ensure pvo_code importable
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PVO_PATH = os.path.join(BASE, 'pvo_code', 'pvo_code')
if PVO_PATH not in sys.path:
    sys.path.insert(0, PVO_PATH)
if BASE not in sys.path:
    sys.path.append(BASE)

from pvo_network import PVONetwork
from sim import SIM_ENV
from ext_pvo import UncertaintyPVONetwork
from fourwis import FourWISActionMapper

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
TB_BASE = os.path.join(RESULTS, 'tb')
MODELS_DIR = os.path.join(RESULTS, 'models')
os.makedirs(TB_BASE, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class TensorboardMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, check_freq=1000):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_successes = []
        self.episode_collisions = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'terminal_observation' in info:
                if 'success' in info:
                    self.episode_successes.append(1.0 if info['success'] else 0.0)
                if 'collision' in info:
                    self.episode_collisions.append(1.0 if info['collision'] else 0.0)
        if self.n_calls % self.check_freq == 0 and len(self.episode_successes) > 0:
            avg_success = np.mean(self.episode_successes[-50:])
            avg_collision = np.mean(self.episode_collisions[-50:])
            self.logger.record('metrics/success_rate', float(avg_success))
            self.logger.record('metrics/collision_rate', float(avg_collision))
        return True

class RobotNavEnvPVO(gym.Env):
    def __init__(self, render=False, pvo_model=None, pvo_type='none', vehicle_model='diff'):
        super().__init__()
        self.render = render
        self.pvo_type = pvo_type
        self.pvo_model = pvo_model
        self.vehicle_model = vehicle_model
        self.is_4wis = vehicle_model == '4wis'
        self.fourwis = FourWISActionMapper() if self.is_4wis else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # diff: 42 lidar + 42 scan_velocity + 10 pvo + 7 other = 101
        # 4WIS: replace 2D previous action with 3D action and add 3D mode one-hot = 105
        self.state_dim = 105 if self.is_4wis else 101
        self.max_steps = 150
        if self.is_4wis:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(low=np.array([-0.6,-1.2]), high=np.array([0.6,1.2]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_dim,), dtype=np.float32)
        world_file = os.path.join(BASE, 'robot_world_4wis.yaml') if self.is_4wis else os.path.join(BASE, 'robot_world.yaml')
        self.sim = SIM_ENV(world_file=world_file, render=render)
        self._reset_episode_tracking()
        initial = self.sim.reset()
        self.current_obs, _ = self.prepare_state(initial, current_action=self._zero_action())

    def _zero_action(self):
        return [0.0, 0.0, 0.0] if self.is_4wis else [0.0, 0.0]

    def _reset_episode_tracking(self):
        self.time = 0
        self.last_position = None
        self.total_distance = 0
        self.total_velocity = 0
        self.steps = 0
        if self.fourwis is not None:
            self.fourwis.reset()

    def _calculate_metrics(self, current_position, action):
        if self.last_position is not None:
            step_dist = np.linalg.norm(current_position - self.last_position)
            self.total_distance += step_dist
            self.total_velocity += np.linalg.norm(action)
        self.last_position = current_position
        self.steps += 1

    def _get_episode_info(self, terminal, reward):
        avg_vel = self.total_velocity / self.steps if self.steps>0 else 0
        info = {'success': terminal and reward>0, 'collision': terminal and reward<0, 'steps': self.steps, 'total_distance': self.total_distance, 'average_velocity': avg_vel}
        if self.fourwis is not None:
            info['mode_switches'] = self.fourwis.mode_switches
            info['motion_mode'] = self.fourwis.current_mode
            info.update(self.fourwis.mode_ratios())
        return info

    def _get_pvo_feature(self, scan_seq, action):
        if self.pvo_type == 'none' or self.pvo_model is None:
            return [0.0]*10
        if self.pvo_type == 'baseline':
            with torch.no_grad():
                xs = torch.tensor(np.tile(np.array(scan_seq)[:42]/8.0,(5,1)), dtype=torch.float32).unsqueeze(0)
                xa = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                p = self.pvo_model(xs, xa).cpu().numpy().ravel()[0]
                return [float(p)]*10
        if self.pvo_type == 'unc':
            # run MC dropout if model supports train mode with dropout; estimate mean
            self.pvo_model.eval()
            preds = []
            xs = torch.tensor(np.tile(np.array(scan_seq)[:42]/8.0,(5,1)), dtype=torch.float32).unsqueeze(0)
            xa = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            self.pvo_model.train()
            for _ in range(6):
                with torch.no_grad():
                    p = self.pvo_model(xs, xa).cpu().numpy().ravel()[0]
                    preds.append(p)
            self.pvo_model.eval()
            mean_p = float(np.mean(preds))
            std_p = float(np.std(preds))
            return [mean_p, std_p] + [0.0]*8

    def prepare_state(self, data, current_action):
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward = data
        latest_scan = np.array(latest_scan)
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 8
        max_bins = 42
        remainder = len(latest_scan) % max_bins
        if remainder != 0:
            extension = np.full(max_bins - remainder, latest_scan[-1])
            latest_scan = np.concatenate((latest_scan, extension))
        bin_size = int(len(latest_scan)/max_bins)
        current_min_values = []
        for i in range(0, len(latest_scan), bin_size):
            bin_data = latest_scan[i:i+bin_size]
            current_min_values.append(min(bin_data)/8.0)
        # scan velocity not tracked here; set zeros
        scan_velocity = [0.0]*max_bins
        pvo_features = self._get_pvo_feature(current_min_values, current_action)
        distance_norm = distance/12.0
        current_action = np.asarray(current_action, dtype=float).reshape(-1)
        if self.is_4wis:
            if current_action.size < 3:
                current_action = np.zeros(3, dtype=float)
            action_features = np.clip(current_action[:3], -1.0, 1.0).tolist()
        else:
            lin_vel = (current_action[0] + 0.6)/1.2
            ang_vel = (current_action[1] + 1.2)/2.4
            action_features = [lin_vel, ang_vel]
        rad_cos = np.cos(diff_rad)
        rad_sin = np.sin(diff_rad)
        state = current_min_values + scan_velocity + list(pvo_features[:10]) + [distance_norm, cos, sin] + action_features + [rad_cos, rad_sin]
        if self.is_4wis:
            state += self.fourwis.mode_one_hot().tolist()
        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0
        return np.array(state, dtype=np.float32), terminal

    def reset(self):
        self._reset_episode_tracking()
        simd = self.sim.reset()
        obs, _ = self.prepare_state(simd, current_action=self._zero_action())
        self.current_obs = obs
        return obs

    def step(self, action):
        if self.is_4wis:
            command = self.fourwis.map(action)
            simd = self.sim.step(
                lin_velocity=command.forward,
                lateral_velocity=command.lateral,
                ang_velocity=command.yaw_rate,
            )
            physical_action = simd[7]
        else:
            lin = action[0]
            ang = action[1]
            simd = self.sim.step(lin, ang)
            physical_action = action
        obs, terminal = self.prepare_state(simd, current_action=action)
        reward = simd[-1]
        if self.is_4wis:
            reward -= command.switch_penalty
        current_pos = self.sim.env.get_robot_state()[:2]
        self._calculate_metrics(current_pos, physical_action)
        done = terminal
        self.time += 1
        if self.time >= self.max_steps:
            done = True
            reward = -100
        info = self._get_episode_info(terminal, reward)
        self.current_obs = obs
        return obs, reward, done, info


def make_env_fn(pvo_model, pvo_type, vehicle_model='diff'):
    def _init():
        return RobotNavEnvPVO(render=False, pvo_model=pvo_model, pvo_type=pvo_type, vehicle_model=vehicle_model)
    return _init


def run_training(method, seed, total_timesteps=200000, num_envs=7, vehicle_model='diff'):
    tb_dir = os.path.join(TB_BASE, f"{vehicle_model}_{method}_seed{seed}")
    os.makedirs(tb_dir, exist_ok=True)
    # load models
    pvo_model = None
    pvo_type = 'none'
    action_dim = 3 if vehicle_model == '4wis' else 2
    if method == 'pvo_baseline':
        pvo_type = 'baseline'
        pvo_model = PVONetwork(action_dim=action_dim)
        baseline_path = os.path.join(PVO_PATH, 'pvo_model_weights.pth')
        if os.path.exists(baseline_path):
            try:
                pvo_model.load_state_dict(torch.load(baseline_path, map_location='cpu'))
                pvo_model.eval()
            except RuntimeError:
                pvo_model = None
                pvo_type = 'none'
        else:
            pvo_model = None
            pvo_type = 'none'
    if method == 'pvo_unc':
        pvo_type = 'unc'
        unc_path = os.path.join(MODELS_DIR, 'unc_pvo.pth')
        if os.path.exists(unc_path):
            pvo_model = UncertaintyPVONetwork(action_dim=action_dim)
            try:
                pvo_model.load_state_dict(torch.load(unc_path, map_location='cpu'))
            except RuntimeError:
                pvo_model = None
                pvo_type = 'none'
        else:
            pvo_model = None
            pvo_type = 'none'
    env_fns = [make_env_fn(pvo_model, pvo_type, vehicle_model=vehicle_model) for _ in range(num_envs)]
    env_fns.append(make_env_fn(pvo_model, pvo_type, vehicle_model=vehicle_model))
    try:
        env = SubprocVecEnv(env_fns)
    except Exception as e:
        env = DummyVecEnv(env_fns)
    model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=TB_BASE)
    cb = TensorboardMetricsCallback(check_freq=1000)
    model.learn(total_timesteps=total_timesteps, callback=cb, tb_log_name=f"{vehicle_model}_{method}_seed{seed}")
    save_path = os.path.join(MODELS_DIR, f"td3_{vehicle_model}_{method}_seed{seed}.zip")
    model.save(save_path)
    # evaluate: mean reward (SB3) and success/collision rates via episode rollouts
    eval_env = DummyVecEnv([make_env_fn(pvo_model, pvo_type, vehicle_model=vehicle_model)])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)

    # compute success/collision rates
    def eval_success_collision(model, pvo_model, pvo_type, n_episodes=50):
        env = DummyVecEnv([make_env_fn(pvo_model, pvo_type, vehicle_model=vehicle_model)])
        successes = 0
        collisions = 0
        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                # ensure action shape matches VecEnv expectation
                action_to_step = action if np.asarray(action).ndim == 2 else np.expand_dims(action, 0)
                obs, reward, dones, infos = env.step(action_to_step)
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                if 'success' in info and info['success']:
                    successes += 1
                if 'collision' in info and info['collision']:
                    collisions += 1
                done = bool(dones[0]) if isinstance(dones, (list, tuple, np.ndarray)) else bool(dones)
        env.close()
        return successes / n_episodes, collisions / n_episodes

    succ_rate, coll_rate = eval_success_collision(model, pvo_model, pvo_type, n_episodes=50)
    return save_path, mean_reward, std_reward, succ_rate, coll_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=200000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--vehicle-model', choices=['diff', '4wis'], default='diff')
    args = parser.parse_args()
    methods = ['no_pvo', 'pvo_baseline', 'pvo_unc']
    summary = {}
    for m in methods:
        means = []
        stds = []
        succ_rates = []
        coll_rates = []
        for s in args.seeds:
            print('Running', m, 'seed', s)
            save, mean, std, succ, coll = run_training(m, s, total_timesteps=args.timesteps, vehicle_model=args.vehicle_model)
            means.append(mean)
            stds.append(std)
            succ_rates.append(succ)
            coll_rates.append(coll)
            print('Finished', m, 'seed', s, 'mean', mean, 'succ', succ, 'coll', coll)
        summary[m] = {'mean_reward_mean': float(np.mean(means)), 'mean_reward_std': float(np.std(means)), 'per_seed_means': means, 'per_seed_stds': stds}
        summary[m]['success_rate_mean'] = float(np.mean(succ_rates))
        summary[m]['success_rate_std'] = float(np.std(succ_rates))
        summary[m]['collision_rate_mean'] = float(np.mean(coll_rates))
        summary[m]['collision_rate_std'] = float(np.std(coll_rates))
        summary[m]['per_seed_success_rates'] = succ_rates
        summary[m]['per_seed_collision_rates'] = coll_rates
    np.savez(os.path.join(RESULTS, 'long_training_summary.npz'), summary=summary)
    print('Saved summary to', os.path.join(RESULTS, 'long_training_summary.npz'))
    # produce comparison bar plot with error bars
    labels = ['No-PVO','PVO','PVO-UQ']
    success_means = [summary['no_pvo']['success_rate_mean'], summary['pvo_baseline']['success_rate_mean'], summary['pvo_unc']['success_rate_mean']]
    success_stds = [summary['no_pvo']['success_rate_std'], summary['pvo_baseline']['success_rate_std'], summary['pvo_unc']['success_rate_std']]
    coll_means = [summary['no_pvo']['collision_rate_mean'], summary['pvo_baseline']['collision_rate_mean'], summary['pvo_unc']['collision_rate_mean']]
    coll_stds = [summary['no_pvo']['collision_rate_std'], summary['pvo_baseline']['collision_rate_std'], summary['pvo_unc']['collision_rate_std']]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, success_means, width, yerr=success_stds, capsize=6, label='success_rate')
    plt.bar(x + width/2, coll_means, width, yerr=coll_stds, capsize=6, label='collision_rate')
    plt.xticks(x, labels)
    plt.ylim(0,1)
    plt.ylabel('Rate')
    plt.title('PVO methods comparison (mean ± std)')
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(RESULTS, 'pvo_comparison_errorbars.png')
    plt.savefig(out_png)
    print('Saved comparison plot with error bars to', out_png)
