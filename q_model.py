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
import torch.nn.functional as F

# 全局参数
INPUT_DIM = 400
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIS_EPOCHS = [0, 37,38]
ENCODER_DIM = 256


# 计算模型参数量的函数
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# 自定义数据集类
class SpectralDataset(Dataset):
    def __init__(self, spectra, labels=None):
        self.spectra = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        if self.labels is not None:
            return {"spectrum": self.spectra[idx], "label": self.labels[idx]}
        return {"spectrum": self.spectra[idx]}


# 原始 SpectrumEncoder（未轻量化）
class SpectrumEncoder(nn.Module):
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

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.multiscale(x)
        x = self.maxpool(x)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        transformer_out = self.transformer(layer4_out)

        x = self.avgpool(transformer_out)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        if return_features:
            return x, [layer1_out, layer2_out, layer3_out, layer4_out, transformer_out]
        return x


# 原始 Classifier（未轻量化）
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

    def forward(self, x, return_features=False):
        encoder_out, encoder_features = self.encoder(x, return_features=True) if return_features else (
        self.encoder(x), None)
        z = encoder_out.unsqueeze(0)
        attn_out, _ = self.attention(z, z, z)
        gru_out, _ = self.gru(attn_out)
        z = attn_out + gru_out
        z = z.squeeze(0)
        logits = self.fc_stack(z)
        logits = self.fc2(logits)
        if return_features:
            return logits, z, encoder_features, attn_out, gru_out
        return logits, z


# 轻量化 SpectrumEncoder
class SpectrumEncoderLight(nn.Module):
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
        def __init__(self, dim, num_heads=2, dropout=0.1):
            super().__init__()
            self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim)
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
        self.in_channels = 32

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.multiscale = self.MultiScaleConv(32, 32)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 1, stride=1)
        self.layer2 = self._make_layer(64, 1, stride=2)
        self.layer3 = self._make_layer(128, 1, stride=2)
        self.layer4 = self._make_layer(256, 1, stride=2)

        self.transformer = self.TransformerLayer(dim=256, num_heads=2, dropout=0.1)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(256 * self.BasicBlock.expansion, output_dim)

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

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.multiscale(x)
        x = self.maxpool(x)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        transformer_out = self.transformer(layer4_out)

        x = self.avgpool(transformer_out)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        if return_features:
            return x, [layer1_out, layer2_out, layer3_out, layer4_out, transformer_out]
        return x


# 轻量化 Classifier
class ClassifierLight(nn.Module):
    def __init__(self, encoder, num_classes, hidden_dim=256, num_heads=4):
        super().__init__()
        self.encoder = encoder
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1)
        self.fc_stack = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, return_features=False):
        encoder_out, encoder_features = self.encoder(x, return_features=True) if return_features else (
        self.encoder(x), None)
        z = encoder_out.unsqueeze(0)
        attn_out, _ = self.attention(z, z, z)
        z = attn_out.squeeze(0)
        logits = self.fc_stack(z)
        logits = self.fc2(logits)
        if return_features:
            return logits, z, encoder_features, attn_out
        return logits, z


# 投影模块，用于对齐特征尺寸
class Projection(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(student_dim, teacher_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(teacher_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.proj(x)


# 线性投影模块，用于对齐分类器的注意力输出和GRU输出
class LinearProjection(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.BatchNorm1d(teacher_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.proj(x)


# 部分加载预训练权重函数
def load_pretrained_weights(student_encoder, teacher_state_dict):
    student_dict = student_encoder.state_dict()
    pretrained_dict = {k: v for k, v in teacher_state_dict.items() if k.startswith('encoder.') and k in student_dict}
    pretrained_dict = {k.replace('encoder.', ''): v for k, v in pretrained_dict.items()}
    for k, v in list(pretrained_dict.items()):
        if 'conv' in k or 'bn' in k:
            if len(v.shape) > 1:
                if v.shape[0] > student_dict[k].shape[0]:
                    pretrained_dict[k] = v[:student_dict[k].shape[0]]
                if v.shape[1] > student_dict[k].shape[1]:
                    pretrained_dict[k] = v[:, :student_dict[k].shape[1]]
            else:
                if v.shape[0] > student_dict[k].shape[0]:
                    pretrained_dict[k] = v[:student_dict[k].shape[0]]
        elif 'transformer' in k and 'attention' in k:
            if v.shape[0] > student_dict[k].shape[0]:
                pretrained_dict[k] = v[:student_dict[k].shape[0]]
    student_dict.update(pretrained_dict)
    student_encoder.load_state_dict(student_dict)
    print(f"Loaded compatible weights from teacher model to student encoder")


# 修改后的t-SNE可视化函数，用于可视化教师和学生的Transformer层特征
def visualize_transformer_features(student_model, teacher_model, dataloader, epoch, seed, projection,
                                   save_dir="tsne_plots_finetune_transformer"):
    os.makedirs(save_dir, exist_ok=True)
    student_model.eval()
    teacher_model.eval()
    student_features, teacher_features, labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['spectrum'].to(DEVICE)
            y = batch['label'].to(DEVICE)

            # 学生模型的Transformer输出
            _, _, student_encoder_features, _ = student_model(x, return_features=True)
            student_trans_out = student_encoder_features[-1]  # transformer_out 是 encoder_features[-1]
            student_trans_proj = projection(student_trans_out)  # 投影到教师维度
            student_features.append(
                student_trans_proj.cpu().contiguous().reshape(student_trans_proj.size(0), -1).numpy())

            # 教师模型的Transformer输出
            _, _, teacher_encoder_features, _, _ = teacher_model(x, return_features=True)
            teacher_trans_out = teacher_encoder_features[-1]  # transformer_out 是 encoder_features[-1]
            teacher_features.append(teacher_trans_out.cpu().contiguous().reshape(teacher_trans_out.size(0), -1).numpy())

            labels.append(y.cpu().numpy())

    student_features = np.concatenate(student_features, axis=0)
    teacher_features = np.concatenate(teacher_features, axis=0)
    labels = np.concatenate(labels, axis=0)

    tsne = TSNE(n_components=2, random_state=seed)

    # 学生t-SNE
    student_2d = tsne.fit_transform(student_features)
    plt.figure(figsize=(8, 6))
    for class_id in range(NUM_CLASSES):
        mask = labels == class_id
        plt.scatter(student_2d[mask, 0], student_2d[mask, 1], label=f'Class {class_id}', alpha=0.5)
    plt.title(f"t-SNE Student Transformer Features at Epoch {epoch} (Seed {seed})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"tsne_student_epoch_{epoch}_seed_{seed}.png"))
    plt.close()
    np.save(os.path.join(save_dir, f"tsne_student_2d_epoch_{epoch}_seed_{seed}.npy"), student_2d)

    # 教师t-SNE
    teacher_2d = tsne.fit_transform(teacher_features)
    plt.figure(figsize=(8, 6))
    for class_id in range(NUM_CLASSES):
        mask = labels == class_id
        plt.scatter(teacher_2d[mask, 0], teacher_2d[mask, 1], label=f'Class {class_id}', alpha=0.5)
    plt.title(f"t-SNE Teacher Transformer Features at Epoch {epoch} (Seed {seed})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"tsne_teacher_epoch_{epoch}_seed_{seed}.png"))
    plt.close()
    np.save(os.path.join(save_dir, f"tsne_teacher_2d_epoch_{epoch}_seed_{seed}.npy"), teacher_2d)

    # 保存标签（相同于学生和教师）
    np.save(os.path.join(save_dir, f"tsne_labels_epoch_{epoch}_seed_{seed}.npy"), labels)


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


# 知识蒸馏损失函数
def distill_loss(student_logits, teacher_logits, student_encoder_features, teacher_encoder_features,
                 student_attn_out, teacher_attn_out, teacher_gru_out, labels, criterion, encoder_projections,
                 attn_projection, gru_projection, temperature=1.0, alpha=0.95, beta=1, gamma=1, delta=1, epoch=0,
                 total_epochs=EPOCHS):
    # logits 蒸馏
    logits_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logits / temperature, dim=1),
                                                      F.softmax(teacher_logits / temperature, dim=1)) * (
                              temperature ** 2)
    class_loss = criterion(student_logits, labels)
    distill_logits_loss = alpha * logits_loss + (1 - alpha) * class_loss

    # 编码器中间特征蒸馏（MSE + Cosine Similarity）
    encoder_feature_loss = 0.0
    encoder_weights = [0.2, 0.2, 0.2, 0.2, 0.4]  # 更重视Transformer层
    for i, (s_feat, t_feat, proj, w) in enumerate(
            zip(student_encoder_features, teacher_encoder_features, encoder_projections, encoder_weights)):
        s_feat_proj = proj(s_feat)
        mse = F.mse_loss(s_feat_proj, t_feat)
        cosine = 1 - F.cosine_similarity(s_feat_proj.contiguous().view(s_feat_proj.size(0), -1),
                                         t_feat.contiguous().view(t_feat.size(0), -1), dim=1).mean()
        mse_weight = 0.7 if i < 3 else 0.5  # 浅层更重视MSE
        cosine_weight = 0.3 if i < 3 else 0.5  # 深层更重视Cosine
        encoder_feature_loss += w * (mse_weight * mse + cosine_weight * cosine)
    encoder_feature_loss /= sum(encoder_weights)

    # 分类器注意力特征蒸馏
    attn_feature_loss = F.mse_loss(attn_projection(student_attn_out.squeeze(0)), teacher_attn_out.squeeze(0))

    # GRU 输出蒸馏
    gru_feature_loss = F.mse_loss(gru_projection(student_attn_out.squeeze(0)), teacher_gru_out.squeeze(0))

    # 总损失
    total_loss = distill_logits_loss + beta * encoder_feature_loss + gamma * attn_feature_loss + delta * gru_feature_loss

    return total_loss, {
        'logits_loss': logits_loss.item(),
        'class_loss': class_loss.item(),
        'encoder_feature_loss': encoder_feature_loss.item(),
        'attn_feature_loss': attn_feature_loss.item(),
        'gru_feature_loss': gru_feature_loss.item(),
        'total_loss': total_loss.item()
    }


# 微调训练函数
def finetune(student_model, teacher_model, encoder_projections, attn_projection, gru_projection, train_loader,
             val_loader, seed, epochs=EPOCHS):
    optimizer = optim.AdamW(list(student_model.parameters()) +
                            [p for proj in encoder_projections for p in proj.parameters()] +
                            list(attn_projection.parameters()) +
                            list(gru_projection.parameters()),
                            lr=LEARNING_RATE, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0.0

    # 在训练开始时可视化
    visualize_transformer_features(student_model, teacher_model, val_loader, epoch=0, seed=seed,
                                   projection=encoder_projections[-1])

    for epoch in range(epochs):
        student_model.train()
        teacher_model.eval()
        train_loss = 0
        loss_components = {
            'logits_loss': 0.0,
            'class_loss': 0.0,
            'encoder_feature_loss': 0.0,
            'attn_feature_loss': 0.0,
            'gru_feature_loss': 0.0,
            'total_loss': 0.0
        }
        for batch in train_loader:
            x = batch['spectrum'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            with torch.no_grad():
                teacher_logits, _, teacher_encoder_features, teacher_attn_out, teacher_gru_out = teacher_model(x,
                                                                                                               return_features=True)
            student_logits, _, student_encoder_features, student_attn_out = student_model(x, return_features=True)
            loss, components = distill_loss(student_logits, teacher_logits, student_encoder_features,
                                            teacher_encoder_features,
                                            student_attn_out, teacher_attn_out, teacher_gru_out, y, criterion,
                                            encoder_projections, attn_projection, gru_projection,
                                            epoch=epoch, total_epochs=epochs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v

        # 打印平均损失分量
        print(f"Seed {seed}, Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Logits Loss: {loss_components['logits_loss'] / len(train_loader):.4f}, "
              f"Class Loss: {loss_components['class_loss'] / len(train_loader):.4f}, "
              f"Encoder Feature Loss: {loss_components['encoder_feature_loss'] / len(train_loader):.4f}, "
              f"Attn Feature Loss: {loss_components['attn_feature_loss'] / len(train_loader):.4f}, "
              f"GRU Feature Loss: {loss_components['gru_feature_loss'] / len(train_loader):.4f}, "
              f"Total Loss: {loss_components['total_loss'] / len(train_loader):.4f}")

        val_acc, val_f1, val_auc, val_cm = evaluate(student_model, val_loader)
        train_acc, train_f1, train_auc, _ = evaluate(student_model, train_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), f"light_finetuned_classifier_seed_{seed}.pth")
            np.save(f"light_confusion_matrix_seed_{seed}.npy", val_cm)
            print(f"Seed {seed}: Saved best light model and confusion matrix at epoch {epoch + 1}")

        print(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, "
              f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f}")

        if epoch + 1 in VIS_EPOCHS:
            visualize_transformer_features(student_model, teacher_model, val_loader, epoch=epoch + 1, seed=seed,
                                           projection=encoder_projections[-1])

        scheduler.step()

    print(f"Seed {seed}: Light model saved to light_finetuned_classifier_seed_{seed}.pth")
    print(f"Seed {seed}: Light confusion matrix saved to light_confusion_matrix_seed_{seed}.npy")
    return best_val_acc


# 冻结部分层函数
def freeze_encoder_layers(model):
    for name, param in model.encoder.named_parameters():
        if 'conv1' in name or 'bn1' in name or 'multiscale' in name or 'layer1' in name:
            param.requires_grad = False


# 主函数
def main():
    file_path = "T_Spectr_Maize.xlsx"
    seeds = range(42, 52)
    best_accuracies = []

    for seed in seeds:
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

        teacher_encoder = SpectrumEncoder().to(DEVICE)
        teacher_model = Classifier(teacher_encoder, NUM_CLASSES).to(DEVICE)
        finetuned_path = "finetuned_classifier_seed_50.pth"
        if os.path.exists(finetuned_path):
            try:
                teacher_model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))
                print(f"Seed {seed}: Loaded finetuned teacher weights from {finetuned_path}")
            except Exception as e:
                print(f"Seed {seed}: Error loading finetuned teacher weights: {e}. Skipping.")
                continue
        else:
            print(f"Seed {seed}: Finetuned teacher weights not found at {finetuned_path}. Skipping.")
            continue

        student_encoder = SpectrumEncoderLight().to(DEVICE)
        try:
            load_pretrained_weights(student_encoder, teacher_model.state_dict())
            print(f"Seed {seed}: Loaded encoder weights from teacher model to student encoder")
        except Exception as e:
            print(
                f"Seed {seed}: Error loading encoder weights from teacher model: {e}. Starting with random initialization.")

        student_model = ClassifierLight(student_encoder, NUM_CLASSES).to(DEVICE)
        freeze_encoder_layers(student_model)

        # 计算并打印教师和学生模型的参数量
        teacher_total_params, teacher_trainable_params = count_parameters(teacher_model)
        student_total_params, student_trainable_params = count_parameters(student_model)
        print(
            f"Seed {seed}: Teacher Model - Total Parameters: {teacher_total_params:,}, Trainable Parameters: {teacher_trainable_params:,}")
        print(
            f"Seed {seed}: Student Model - Total Parameters: {student_total_params:,}, Trainable Parameters: {student_trainable_params:,}")
        print(
            f"Seed {seed}: Student model size is {student_total_params / teacher_total_params:.2%} of teacher model size")

        encoder_projections = [
            Projection(32, 64).to(DEVICE),
            Projection(64, 128).to(DEVICE),
            Projection(128, 256).to(DEVICE),
            Projection(256, 512).to(DEVICE),
            Projection(256, 512).to(DEVICE)
        ]
        attn_projection = LinearProjection(256, 256).to(DEVICE)
        gru_projection = LinearProjection(256, 256).to(DEVICE)

        best_val_acc = finetune(student_model, teacher_model, encoder_projections, attn_projection, gru_projection,
                                train_loader, val_loader, seed)
        best_accuracies.append(best_val_acc)
        print(f"Seed {seed}: Best Validation Accuracy for light model: {best_val_acc:.4f}")

    np.save("light_best_accuracies.npy", np.array(best_accuracies))
    print(f"Best accuracies for all seeds saved to light_best_accuracies.npy: {best_accuracies}")


if __name__ == "__main__":
    main()