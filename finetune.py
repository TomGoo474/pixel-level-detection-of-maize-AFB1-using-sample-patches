import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import math

# 全局参数
INPUT_DIM = 400  # 光谱序列长度
BATCH_SIZE = 32
EPOCHS = 50  # 微调轮数
LEARNING_RATE = 1e-4  # 微调学习率
NUM_CLASSES = 2  # 二分类（Non: 0, AF: 1）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIS_EPOCHS = [0, 45, 50]  # 可视化的epoch
ENCODER_DIM = 256  # 编码器输出维度

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

class Classifier(nn.Module):
    def __init__(self, encoder, num_classes, hidden_dim=256, num_heads=8, gru_layers=1, fc_depth=2):
        super().__init__()
        self.encoder = encoder
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_layers, batch_first=False, bidirectional=False)
        fc_layers = []
        in_dim = hidden_dim
        for i in range(fc_depth):
            out_dim = 256 if i < fc_depth - 1 else 64
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.BatchNorm1d(out_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.1))
            in_dim = out_dim
        self.fc_stack = nn.Sequential(*fc_layers)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        z = z.unsqueeze(0)
        attn_out, _ = self.attention(z, z, z)
        gru_out, _ = self.gru(attn_out)
        z = attn_out + gru_out
        z = z.squeeze(0)
        x = self.fc_stack(z)
        x = self.fc2(x)
        return x, z

# t-SNE可视化函数
def visualize_features(model, dataloader, epoch, seed, save_dir="tsne_plots_finetune"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['spectrum'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            _, z = model(x)
            features.append(z.cpu().numpy())
            labels.append(y.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    tsne = TSNE(n_components=2, random_state=seed)
    features_2d = tsne.fit_transform(features)

    # 保存t-SNE数据
    np.save(os.path.join(save_dir, f"tsne_class1_epoch_{epoch}.npy"), features_2d)
    np.save(os.path.join(save_dir, f"tsne_class2_epoch_{epoch}.npy"), labels)

    plt.figure(figsize=(8, 6))
    for class_id in range(NUM_CLASSES):
        mask = labels == class_id
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=f'Class {class_id}', alpha=0.5)
    plt.title(f"t-SNE Visualization of Features at Epoch {epoch} (Seed {seed})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"tsne_epoch_{epoch}_seed_{seed}.png"))
    plt.close()

# 数据加载和预处理
def load_data(file_path, seed):
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None, None, None, None

    df = pd.read_excel(file_path)
    spectra = df.iloc[:, 2:].values
    labels = df['Class'].map({'Non': 0, 'AF': 1}).values

    global INPUT_DIM
    if spectra.shape[1] != INPUT_DIM:
        print(f"Adjusting input_dim to {spectra.shape[1]}")
        INPUT_DIM = spectra.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        spectra, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    print(f"Seed {seed} - Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    scaler_clf = StandardScaler()
    X_train_clf = scaler_clf.fit_transform(X_train)
    X_val_clf = scaler_clf.transform(X_val)

    train_dataset = SpectralDataset(X_train_clf, y_train)
    val_dataset = SpectralDataset(X_val_clf, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, X_train_clf, y_train

# 评估函数
def evaluate(model, dataloader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['spectrum'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    auc = roc_auc_score(true_labels, predictions) if NUM_CLASSES == 2 else None
    cm = confusion_matrix(true_labels, predictions)
    return accuracy, f1, auc, cm

# 微调训练函数
def finetune(model, train_loader, val_loader, seed, epochs=EPOCHS):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0.0

    # 在训练开始前可视化
    visualize_features(model, val_loader, epoch=0, seed=seed)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch['spectrum'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_acc, val_f1, val_auc, val_cm = evaluate(model, val_loader)
        train_acc, train_f1, train_auc, _ = evaluate(model, train_loader)

        # 更新最佳验证准确率并保存混淆矩阵
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"finetuned_classifier_seed_{seed}.pth")
            np.save(f"confusion_matrix_seed_{seed}.npy", val_cm)
            print(f"Seed {seed}: Saved best model and confusion matrix at epoch {epoch + 1}")

        print(f"Seed {seed}, Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, "
              f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f}")

        if epoch + 1 in VIS_EPOCHS:
            visualize_features(model, val_loader, epoch=epoch + 1, seed=seed)

        scheduler.step()

    print(f"Seed {seed}: Model saved to finetuned_classifier_seed_{seed}.pth")
    print(f"Seed {seed}: Confusion matrix saved to confusion_matrix_seed_{seed}.npy")
    return best_val_acc

# 主函数
def main():
    file_path = "T_Spectr_Maize.xlsx"
    seeds = range(42, 52)  # Seeds from 50 to 51
    best_accuracies = []

    for seed in seeds:
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"\nStarting experiment with seed {seed}")
        train_loader, val_loader, _, _ = load_data(file_path, seed)
        if train_loader is None:
            print(f"Seed {seed}: Failed to load data, skipping.")
            continue

        encoder = SpectrumEncoder().to(DEVICE)
        pretrained_path = "best_ssl_encoder_patch.pth"
        if os.path.exists(pretrained_path):
            try:
                encoder.load_state_dict(torch.load(pretrained_path))
                print(f"Seed {seed}: Loaded pre-trained weights from {pretrained_path}")
            except Exception as e:
                print(f"Seed {seed}: Error loading pre-trained weights: {e}. Starting with random initialization.")
        else:
            print(f"Seed {seed}: Pre-trained weights not found at {pretrained_path}. Starting with random initialization.")

        model = Classifier(encoder, NUM_CLASSES).to(DEVICE)
        best_val_acc = finetune(model, train_loader, val_loader, seed)
        best_accuracies.append(best_val_acc)
        print(f"Seed {seed}: Best Validation Accuracy: {best_val_acc:.4f}")

    # 保存最佳准确率到 .npy 文件
    np.save("best_accuracies.npy", np.array(best_accuracies))
    print(f"Best accuracies for all seeds saved to best_accuracies.npy: {best_accuracies}")

if __name__ == "__main__":
    main()