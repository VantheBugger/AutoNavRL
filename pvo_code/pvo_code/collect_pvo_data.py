import numpy as np
import random
import os
from collections import deque
from sim import SIM_ENV

def collect_data(num_episodes=500, seq_len=5, horizon=10, save_path="pvo_dataset.npz"):
    """
    收集 PVO 训练数据。
    
    参数:
        num_episodes: 收集的回合数
        seq_len: 输入网络的历史雷达帧数 (比如过去 5 帧)
        horizon: 向未来看多少步来定义“假想碰撞” (比如未来 10 帧)
    """
    env = SIM_ENV(render=False)
    
    # 存储最终的训练数据
    X_scans = []
    X_actions = []
    Y_labels = []
    
    print(f"开始收集离线数据，目标回合数: {num_episodes}...")
    
    for ep in range(num_episodes):
        # 初始化环境和历史队列
        sim_data = env.reset()
        scan, distance, _, _, collision, goal, _, _, _ = sim_data
        
        # 将原始雷达数据下采样到 42 维，并归一化 (复用你 train.py 的逻辑)
        def process_scan(raw_scan):
            raw_scan = np.array(raw_scan)
            raw_scan[np.isinf(raw_scan)] = 8.0
            max_bins = 42
            remainder = len(raw_scan) % max_bins
            if remainder != 0:
                raw_scan = np.concatenate((raw_scan, np.full(max_bins - remainder, raw_scan[-1])))
            bin_size = int(len(raw_scan) / max_bins)
            return [min(raw_scan[i : i + bin_size]) / 8.0 for i in range(0, len(raw_scan), bin_size)]

        current_scan = process_scan(scan)
        
        # 使用双端队列维护过去 seq_len 帧的数据
        scan_history = deque([current_scan for _ in range(seq_len)], maxlen=seq_len)
        
        # 用于记录当前回合轨迹的临时列表
        episode_transitions = []
        
        done = False
        step = 0
        while not done and step < 300:
            # 采用混合策略收集数据：有时随机，有时向前，以保证数据的多样性
            if random.random() < 0.3:
                action = [random.uniform(-0.6, 0.6), random.uniform(-1.2, 1.2)]
            else:
                action = [random.uniform(0.2, 0.6), random.uniform(-0.3, 0.3)] # 倾向于向前走
                
            # 执行动作
            sim_data = env.step(lin_velocity=action[0], ang_velocity=action[1])
            scan, distance, _, _, collision, goal, _, _, _ = sim_data
            
            next_scan = process_scan(scan)
            scan_history.append(next_scan)
            
            # 保存当前步的信息
            # 注意：此时还不知道未来是否会碰撞，先记下来
            episode_transitions.append({
                "scan_seq": np.array(scan_history), # shape: (seq_len, 42)
                "action": action,
                "collision": collision
            })
            
            if collision or goal:
                done = True
            step += 1
            
        # 回合结束后，计算 Horizon 碰撞标签 (Labeling)
        # 如果当前步的未来 Horizon 步内发生了碰撞，当前动作就被标记为 1 (危险)
        for i in range(len(episode_transitions)):
            future_window = episode_transitions[i : i + horizon]
            # 检查未来窗口内是否有碰撞
            is_future_collision = any(trans["collision"] for trans in future_window)
            
            X_scans.append(episode_transitions[i]["scan_seq"])
            X_actions.append(episode_transitions[i]["action"])
            Y_labels.append(1.0 if is_future_collision else 0.0)
            
        if (ep + 1) % 50 == 0:
            print(f"已收集 {ep + 1} 回合，当前样本数: {len(Y_labels)}")

    # 转换为 Numpy 数组并保存
    X_scans = np.array(X_scans, dtype=np.float32)
    X_actions = np.array(X_actions, dtype=np.float32)
    Y_labels = np.array(Y_labels, dtype=np.float32)
    
    np.savez(save_path, scans=X_scans, actions=X_actions, labels=Y_labels)
    print(f"数据收集完毕！已保存至 {save_path}。正样本比例: {np.mean(Y_labels)*100:.2f}%")

if __name__ == "__main__":
    collect_data()
