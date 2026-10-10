import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class PVONetwork(nn.Module):
    def __init__(self, seq_len=5, scan_dim=42, action_dim=2, hidden_dim=64):
        super(PVONetwork, self).__init__()
        
        # 1. 空间特征提取 (处理单帧雷达数据)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(scan_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 2. 时序特征提取 (处理连续帧，学习速度和加速度趋势)
        # batch_first=True 表示输入形状为 (batch, seq, feature)
        self.gru = nn.GRU(input_size=32, hidden_size=hidden_dim, batch_first=True)
        
        # 3. 概率推理多层感知机 (结合 GRU 的隐藏状态和当前的动作)
        self.prob_mlp = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 输出 0 到 1 之间的概率
        )
        
    def forward(self, scan_seq, action):
        """
        参数:
            scan_seq: (batch_size, seq_len, 42)
            action: (batch_size, 2)
        返回:
            collision_prob: (batch_size, 1)
        """
        batch_size, seq_len, _ = scan_seq.size()
        
        # 展平以便通过空间 MLP: (batch * seq, 42)
        scan_flat = scan_seq.view(batch_size * seq_len, -1)
        spatial_features = self.spatial_mlp(scan_flat)
        
        # 恢复时序形状: (batch, seq, 32)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        # 通过 GRU
        gru_out, hidden = self.gru(spatial_features)
        
        # 取 GRU 最后一个时间步的输出作为环境特征: (batch, hidden_dim)
        env_context = gru_out[:, -1, :] 
        
        # 将环境特征与候选动作拼接
        fused_features = torch.cat([env_context, action], dim=1)
        
        # 推理碰撞概率
        collision_prob = self.prob_mlp(fused_features)
        return collision_prob

def train_pvo_model(dataset_path="pvo_dataset.npz", epochs=50, batch_size=128):
    """
    训练 PVO 网络
    """
    # 1. 加载数据
    print(f"加载数据集: {dataset_path}")
    data = np.load(dataset_path)
    X_scans = torch.tensor(data['scans'], dtype=torch.float32)
    X_actions = torch.tensor(data['actions'], dtype=torch.float32)
    Y_labels = torch.tensor(data['labels'], dtype=torch.float32).unsqueeze(1) # 转为列向量
    
    # 构建 DataLoader
    dataset = TensorDataset(X_scans, X_actions, Y_labels)
    # 划分训练集和验证集 (简单起见，这里全量训练，你可以自己加 split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. 初始化网络、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PVONetwork().to(device)
    
    # 使用二元交叉熵损失函数 (Binary Cross Entropy) 预测概率
    # 如果正负样本不平衡，可以在这里加入 weight
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"开始训练，使用设备: {device}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_scans, batch_actions, batch_labels in dataloader:
            batch_scans = batch_scans.to(device)
            batch_actions = batch_actions.to(device)
            batch_labels = batch_labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            preds = model(batch_scans, batch_actions)
            
            # 计算损失并反向传播
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    # 保存训练好的模型参数
    save_path = "pvo_model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型训练完毕并保存至 {save_path}")

if __name__ == "__main__":
    train_pvo_model()
