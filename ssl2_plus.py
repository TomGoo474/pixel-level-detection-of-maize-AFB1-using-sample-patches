import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

# Set random seed for reproducibility
SEED = 45
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 全局参数
INPUT_DIM = 400  # 光谱序列长度
PATCH_SIZE = 40  # 每个小片段长度
NUM_PATCHES = INPUT_DIM // PATCH_SIZE  # 10个片段
ENCODER_DIM = 256  # 编码器输出维度
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIS_EPOCHS = [0,2,50]  # 可视化的epoch


# 自定义数据集类
class SpectralDataset(Dataset):
    def __init__(self, spectra, labels=None):
        self.spectra = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)  # [N, 1, 400]
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        if self.labels is not None:
            return {"spectrum": self.spectra[idx], "label": self.labels[idx]}
        return {"spectrum": self.spectra[idx]}


# 分割光谱为10个小片段
def split_spectrum(spectrum):
    # spectrum: [batch_size, 1, 400]
    batch_size = spectrum.shape[0]
    patches = torch.split(spectrum, PATCH_SIZE, dim=2)  # 10个 [batch_size, 1, 40]
    return torch.stack(patches, dim=1)  # [batch_size, 10, 1, 40]


class SpectrumEncoder(nn.Module):
    """
    改进的 ResNet-18 编码器，结合多尺度卷积和 Transformer 模块。
    Args:
        input_dim (int, optional): 输入光谱序列长度，默认 400。
        output_dim (int, optional): 输出嵌入维度，默认 256。
    """

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out

    class MultiScaleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            base_channels = out_channels // 3
            remaining_channels = out_channels - 2 * base_channels
            self.conv3 = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
            self.conv5 = nn.Conv1d(in_channels, base_channels, kernel_size=5, padding=2, bias=False)
            self.conv7 = nn.Conv1d(in_channels, remaining_channels, kernel_size=7, padding=3, bias=False)
            self.bn = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            out3 = self.conv3(x)
            out5 = self.conv5(x)
            out7 = self.conv7(x)
            out = torch.cat([out3, out5, out7], dim=1)
            out = self.bn(out)
            out = self.relu(out)
            return out

    class TransformerLayer(nn.Module):
        def __init__(self, dim, num_heads=4, dropout=0.1):
            super().__init__()
            self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim)
            )
            self.norm2 = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = x.permute(2, 0, 1)
            attn_output, _ = self.attention(x, x, x)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            return x.permute(1, 2, 0)

    def __init__(self, input_dim=400, output_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.multiscale = self.MultiScaleConv(64, 64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.transformer = self.TransformerLayer(dim=512, num_heads=4, dropout=0.1)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512 * self.BasicBlock.expansion, output_dim)

        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * self.BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.BasicBlock.expansion),
            )
        layers = []
        layers.append(self.BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * self.BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(self.BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.005)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.multiscale(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.transformer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 融合模块（加权平均）
class FusionModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, z_orig):
        # features: [batch_size, NUM_PATCHES, ENCODER_DIM], z_orig: [batch_size, ENCODER_DIM]
        similarities = torch.stack([cosine_similarity(f, z_orig) for f in features.permute(1, 0, 2)],
                                   dim=1)  # [batch_size, NUM_PATCHES]
        weights = torch.softmax(similarities / 0.1, dim=1)  # [batch_size, NUM_PATCHES]
        fused_features = torch.sum(features * weights.unsqueeze(-1), dim=1)  # [batch_size, ENCODER_DIM]
        return fused_features, weights  # 返回融合特征和权重


def cosine_similarity(a, b):
    a = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-8)
    b = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-8)
    return (a * b).sum(dim=-1)


def info_nce_loss(z_avg, z_orig, z_negatives, tau=1):
    pos_sim = cosine_similarity(z_avg, z_orig) / tau
    neg_sim = torch.stack([cosine_similarity(z_avg, z_neg) / tau for z_neg in z_negatives], dim=1)
    return -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim), dim=1)))


def total_loss(z_avg, z_orig, z_negatives, alpha=0.5, beta=0.5):
    cos_loss = 1 - cosine_similarity(z_avg, z_orig).mean()
    nce_loss = info_nce_loss(z_avg, z_orig, z_negatives).mean()
    return alpha * cos_loss + beta * nce_loss


# t-SNE可视化函数，保存数据
def visualize_features(model, fusion, dataloader, epoch, save_dir="tsne_plots"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    features_orig, features_avg = [], []

    with torch.no_grad():
        for batch in dataloader:
            x_orig = batch['spectrum'].to(DEVICE)  # [batch_size, 1, 400]
            x_patches = split_spectrum(x_orig)  # [batch_size, 10, 1, 40]

            z_orig = model(x_orig)  # [batch_size, 256]
            z_patches = model(x_patches.view(-1, 1, PATCH_SIZE)).view(-1, NUM_PATCHES,
                                                                      ENCODER_DIM)  # [batch_size, 10, 256]
            z_avg, _ = fusion(z_patches, z_orig)  # [batch_size, 256]

            features_orig.append(z_orig.cpu().numpy())
            features_avg.append(z_avg.cpu().numpy())

    features_orig = np.concatenate(features_orig, axis=0)
    features_avg = np.concatenate(features_avg, axis=0)

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_combined = np.concatenate([features_orig, features_avg], axis=0)
    features_2d = tsne.fit_transform(features_combined)

    n_samples = features_orig.shape[0]
    orig_2d = features_2d[:n_samples]
    avg_2d = features_2d[n_samples:]

    # 保存t-SNE数据
    np.save(os.path.join(save_dir, f"tsne_orig_epoch_{epoch}.npy"), orig_2d)
    np.save(os.path.join(save_dir, f"tsne_avg_epoch_{epoch}.npy"), avg_2d)

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', label='Original Features', alpha=0.5)
    plt.scatter(avg_2d[:, 0], avg_2d[:, 1], c='red', label='Average Patched Features', alpha=0.5)
    plt.title(f"t-SNE Visualization at Epoch {epoch}")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"tsne_epoch_{epoch}.png"))
    plt.close()


# 绘制patch权重分布图，数据已由train_ssl保存
def plot_patch_weights(patch_weights, epoch, save_dir="patch_weights_plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    patches = np.arange(1, NUM_PATCHES + 1)
    # 绘制柱状图
    bars = plt.bar(patches, patch_weights, color='skyblue', edgecolor='black')
    # 标注关键波长范围（权重最高的3个patch）
    max_indices = np.argsort(patch_weights)[-3:]  # 取权重最高的3个patch

    # 保存t-SNE数据
    np.save(os.path.join(save_dir, f"wavelength_importance_epoch_{epoch}.npy"), patch_weights)



    for idx in max_indices:
        bars[idx].set_color('orange')  # 高亮关键波长范围
        plt.text(patches[idx], patch_weights[idx] + 0.005, f'{patch_weights[idx]:.4f}',
                 ha='center', va='bottom', fontsize=10, color='black')
    plt.xlabel('Waveband Range', fontsize=12)
    plt.ylabel('Average Weight', fontsize=12)
    plt.title(f'Patch Weights at Epoch {epoch} (Highlighted: Key Wavebands)', fontsize=14)
    plt.xticks(patches, [f'{i * PATCH_SIZE + 1}-{(i + 1) * PATCH_SIZE}' for i in range(NUM_PATCHES)], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'patch_weights_epoch_{epoch}.png'), dpi=300)
    plt.close()

    # 打印关键波长范围
    print(f"Epoch {epoch} Key Wavebands:")
    for idx in max_indices:
        print(
            f"  Patch {idx + 1} (Waveband {idx * PATCH_SIZE + 1}-{(idx + 1) * PATCH_SIZE}): Weight = {patch_weights[idx]:.4f}")


# 计算每个波长的梯度重要性
def compute_wavelength_importance(model, fusion, dataloader):
    model.eval()
    importance_scores = []

    with torch.enable_grad():
        for batch in dataloader:
            x_orig = batch['spectrum'].to(DEVICE).requires_grad_(True)
            x_patches = split_spectrum(x_orig)

            z_orig = model(x_orig)
            z_patches = model(x_patches.view(-1, 1, PATCH_SIZE)).view(-1, NUM_PATCHES, ENCODER_DIM)
            z_avg, _ = fusion(z_patches, z_orig)

            output = z_avg.mean()
            grad = torch.autograd.grad(output, x_orig, retain_graph=True)[0]
            grad_abs = torch.abs(grad).squeeze(1)

            importance_scores.append(grad_abs.cpu().numpy())

    avg_importance = np.mean(np.concatenate(importance_scores, axis=0), axis=0)
    avg_importance = avg_importance / np.max(avg_importance)  # 归一化到 [0, 1]
    return avg_importance


# 绘制每个波长的重要性图，数据已由train_ssl保存
def plot_wavelength_importance(importance_scores, epoch, save_dir="wavelength_importance_plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(14, 7))
    wavelengths = np.arange(1, INPUT_DIM + 1)
    plt.plot(wavelengths, importance_scores, color='blue', label='Wavelength Importance')

    # 添加patch边界线
    for i in range(1, NUM_PATCHES):
        plt.axvline(x=i * PATCH_SIZE, color='gray', linestyle='--', alpha=0.5)

    # 突出每个patch中的最大重要性波长
    for patch_idx in range(NUM_PATCHES):
        start = patch_idx * PATCH_SIZE
        end = start + PATCH_SIZE
        patch_scores = importance_scores[start:end]
        max_idx = np.argmax(patch_scores) + start + 1
        max_score = np.max(patch_scores)
        plt.scatter(max_idx, max_score, color='red', marker='*', s=100)
        plt.text(max_idx, max_score + 0.02, f'Wave {max_idx}: {max_score:.4f}', ha='center', va='bottom', fontsize=9,
                 color='red')

    plt.xlabel('Wavelength Index', fontsize=12)
    plt.ylabel('Normalized Importance Score', fontsize=12)
    plt.title(f'Wavelength Importance at Epoch {epoch} (Highlighted: Max in Each Patch)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'wavelength_importance_epoch_{epoch}.png'), dpi=300)
    plt.close()

    # 打印每个patch中的关键波长
    print(f"Epoch {epoch} Key Wavelengths per Patch:")
    for patch_idx in range(NUM_PATCHES):
        start = patch_idx * PATCH_SIZE
        end = start + PATCH_SIZE
        patch_scores = importance_scores[start:end]
        max_idx = np.argmax(patch_scores) + start + 1
        max_score = np.max(patch_scores)
        print(f"  Patch {patch_idx + 1} (Waveband {start + 1}-{end}): Max Wave {max_idx}, Score = {max_score:.4f}")


# 数据加载和预处理
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None, None

    df = pd.read_excel(file_path)
    spectra = df.iloc[:, 2:].values
    labels = df['Class'].map({'Non': 0, 'AF': 1}).values

    global INPUT_DIM
    if spectra.shape[1] != INPUT_DIM:
        print(f"Adjusting input_dim to {spectra.shape[1]}")
        INPUT_DIM = spectra.shape[1]

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        spectra, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    # 无监督学习数据预处理
    scaler = StandardScaler()
    X_train_ssl = scaler.fit_transform(X_train)
    X_val_ssl = scaler.transform(X_val)
    norm_scaler = MinMaxScaler()
    X_train_ssl = norm_scaler.fit_transform(X_train_ssl)
    X_val_ssl = norm_scaler.transform(X_val_ssl)

    # 创建数据集
    ssl_train_dataset = SpectralDataset(X_train_ssl)
    ssl_val_dataset = SpectralDataset(X_val_ssl)

    # 创建数据加载器
    train_loader = DataLoader(ssl_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ssl_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# 训练函数
def train_ssl(model, fusion, train_loader, val_loader, epochs=EPOCHS):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 初始化最优验证损失和对应的epoch
    best_val_loss = float('inf')
    best_epoch = 0
    patch_weights_history = []  # 记录每个epoch的平均权重
    wavelength_importance_history = []  # 记录每个epoch的波长重要性

    # 在训练开始前可视化
    visualize_features(model, fusion, val_loader, epoch=0)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x_orig = batch['spectrum'].to(DEVICE)  # [batch_size, 1, 400]
            x_patches = split_spectrum(x_orig)  # [batch_size, 10, 1, 40]

            z_orig = model(x_orig)  # [batch_size, 256]
            z_patches = model(x_patches.view(-1, 1, PATCH_SIZE)).view(-1, NUM_PATCHES,
                                                                      ENCODER_DIM)  # [batch_size, 10, 256]
            z_avg, _ = fusion(z_patches, z_orig)  # 只取融合特征用于损失计算

            # 负样本
            batch_indices = torch.randperm(x_orig.shape[0])
            z_negatives = [z_orig[idx] for idx in batch_indices[:x_orig.shape[0]]]

            # 计算损失
            loss = total_loss(z_avg, z_orig, z_negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        patch_weights = []
        with torch.no_grad():
            for batch in val_loader:
                x_orig = batch['spectrum'].to(DEVICE)
                x_patches = split_spectrum(x_orig)

                z_orig = model(x_orig)
                z_patches = model(x_patches.view(-1, 1, PATCH_SIZE)).view(-1, NUM_PATCHES, ENCODER_DIM)
                z_avg, weights = fusion(z_patches, z_orig)

                batch_indices = torch.randperm(x_orig.shape[0])
                z_negatives = [z_orig[idx] for idx in batch_indices[:x_orig.shape[0]]]

                loss = total_loss(z_avg, z_orig, z_negatives)
                val_loss += loss.item()

                patch_weights.append(weights.cpu().numpy())  # 收集权重

        avg_val_loss = val_loss / len(val_loader)
        avg_patch_weights = np.mean(np.concatenate(patch_weights, axis=0), axis=0)  # [NUM_PATCHES]
        patch_weights_history.append(avg_patch_weights)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Patch Weights: {avg_patch_weights}")

        # 计算波长重要性
        avg_wavelength_importance = compute_wavelength_importance(model, fusion, val_loader)
        wavelength_importance_history.append(avg_wavelength_importance)

        # 保存最优模型权重
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_ssl_encoder_patches.pth")
            print(f"Saved best model weights at epoch {best_epoch} with val_loss: {best_val_loss:.4f}")

        # 在指定epoch可视化
        if epoch + 1 in VIS_EPOCHS:
            visualize_features(model, fusion, val_loader, epoch=epoch + 1)
            plot_patch_weights(avg_patch_weights, epoch + 1)
            plot_wavelength_importance(avg_wavelength_importance, epoch + 1)

        scheduler.step()

    # 保存权重历史
    np.save("patch_weights_history.npy", np.array(patch_weights_history))
    np.save("wavelength_importance_history.npy", np.array(wavelength_importance_history))
    print(f"Training completed. Best model saved at epoch {best_epoch} with val_loss: {best_val_loss:.4f}")

    # 绘制最终的权重和重要性图
    final_patch_weights = patch_weights_history[-1]
    plot_patch_weights(final_patch_weights, "Final")
    final_wavelength_importance = wavelength_importance_history[-1]
    plot_wavelength_importance(final_wavelength_importance, "Final")

    return patch_weights_history, wavelength_importance_history


# 主函数
def main():
    file_path = "T_Spectr_Maize.xlsx"
    train_loader, val_loader = load_data(file_path)
    if train_loader is None:
        return

    model = SpectrumEncoder(input_dim=INPUT_DIM, output_dim=ENCODER_DIM).to(DEVICE)
    fusion = FusionModule().to(DEVICE)
    patch_weights_history, wavelength_importance_history = train_ssl(model, fusion, train_loader, val_loader)

    # 输出最终的1-400波长重要性分数
    final_importance = wavelength_importance_history[-1]
    print("\nFinal Wavelength Importance Scores (1-400):")
    for i, score in enumerate(final_importance, 1):
        print(f"Wavelength {i}: {score:.6f}")


if __name__ == "__main__":
    main()