import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, vgg11
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import json

# ==============================================================================
# 1. 模型定义 (Model Definitions)
# ==============================================================================

# --- 为CIFAR-10修改的ResNet-18 ---
class ModifiedResNet18(nn.Module):
    """
    目标模型/影子模型的定义。
    这里使用一个为CIFAR-10数据集修改过的ResNet-18。
    """
    def __init__(self, num_classes=10, **kwargs):
        # 备注：kwargs用于接收通用参数，如此处不需要的channel
        super(ModifiedResNet18, self).__init__()
        # weights=None 表示我们从头开始训练
        self.model = resnet18(weights=None, num_classes=num_classes)
        # 针对CIFAR-10 (32x32图像) 的修改
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # 移除最大池化层
        
    def forward(self, x):
        return self.model(x)

# --- 经典的LeNet模型 ---
class LeNet(nn.Module):
    def __init__(self, channel=3, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2), # 针对32x32输入调整padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120), # 针对32x32输入调整全连接层
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- 攻击模型 ---
class AttackModel(nn.Module):
    """
    攻击模型的定义。
    一个简单的多层感知机 (MLP)，用于区分成员和非成员。
    """
    def __init__(self, input_dim, n_hidden=50):
        super(AttackModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# ==============================================================================
# 1.5 模型工厂 (Model Factory)
# ==============================================================================
def create_model_from_name(model_name, num_classes=10, channel=3):
    """
    根据名称创建模型实例 (模型工厂)。
    """
    name = model_name.lower()
    print(f"正在创建模型: {name}")

    if name == "resnet18":
        # 返回为CIFAR-10修改过的ResNet18
        return ModifiedResNet18(num_classes=num_classes, channel=channel)
    elif name == "vgg11":
        # 从torchvision加载vgg11，并修改分类层以适应类别数
        model = vgg11(weights=None, num_classes=num_classes)
        return model
    elif name == 'lenet':
        return LeNet(channel=channel, num_classes=num_classes)
    
    # --- 在此添加您自己的模型定义 ---
    # 例如:
    # elif name == "resnet9":
    #     # 需要您自己提供 ResNet9 的类定义
    #     return ResNet9(channel, num_classes)
    # elif name == 'wrn28-2':
    #     # 需要您自己提供 WideResNet 的类定义
    #     return WideResNet(28, 2, channel, num_classes, 0.3)
    
    else:
        raise NotImplementedError(f"模型 '{name}' 未实现或未在工厂函数中注册。")

# ==============================================================================
# 2. 辅助函数 (Helper Functions)
# ==============================================================================

def load_cifar10_data(data_root='./data'):
    """
    加载并预处理CIFAR-10数据集。
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset

def train_generic_model(model, train_loader, device, epochs, learning_rate, l2_ratio, save_path=None, is_attack_model=False):
    """
    一个通用的模型训练函数。
    """
    model.to(device)
    model.train()
    criterion = nn.BCELoss() if is_attack_model else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_ratio)
    print(f"开始训练模型, 共 {epochs} 个 epochs...")
    for epoch in range(epochs):
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"模型权重已保存到: {save_path}")
    return model

def load_model_weights(model, weights_path, device):
    """
    加载已保存的模型权重。
    """
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"成功从 {weights_path} 加载模型权重。")
        return model
    except Exception as e:
        print(f"加载模型权重失败: {e}。")
        return None

def prepare_attack_data(model, member_loader, non_member_loader, device):
    """
    使用给定的模型（目标或影子）生成攻击数据。
    """
    model.eval()
    model.to(device)
    attack_data, attack_labels, attack_classes = [], [], []
    with torch.no_grad():
        for images, y in member_loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            attack_data.append(probs.cpu().numpy())
            attack_labels.extend([1] * len(y))
            attack_classes.extend(y.numpy())
    with torch.no_grad():
        for images, y in non_member_loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            attack_data.append(probs.cpu().numpy())
            attack_labels.extend([0] * len(y))
            attack_classes.extend(y.numpy())
    return np.vstack(attack_data), np.array(attack_labels), np.array(attack_classes)

# ==============================================================================
# 3. 主攻击流程函数 (Main Attack Function)
# ==============================================================================

def run_membership_inference_attack(
    # --- 模型和数据参数 ---
    target_model_name='resnet18',
    target_model_instance=None,
    dataset_name='CIFAR10',
    data_root='./data',
    weights_dir='./mia_weights',
    
    # --- 目标模型训练参数 (仅当 target_model_instance is None 时生效) ---
    target_train_size=15000,
    target_epochs=50,
    target_batch_size=128,
    target_learning_rate=0.001,
    target_l2_ratio=1e-6,
    
    # --- 影子模型参数 ---
    n_shadow_models=4,
    
    # --- 攻击模型参数 ---
    attack_model_class=AttackModel,
    attack_epochs=50,
    attack_batch_size=128,
    attack_learning_rate=0.001,
    attack_l2_ratio=1e-6,
    attack_n_hidden=50,

    # --- 其他参数 ---
    force_retrain=False
):
    """
    执行完整的成员推理攻击流程。
    """
    # --- 0. 初始化设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    shadow_models_dir = os.path.join(weights_dir, target_model_name, 'shadow')
    attack_models_dir = os.path.join(weights_dir, target_model_name, 'attack')

    # --- 1. 加载数据 ---
    print("\n" + "="*10 + " 1. 加载数据 " + "="*10)
    if dataset_name == 'CIFAR10':
        train_dataset, test_dataset = load_cifar10_data(data_root)
        num_classes = 10
        channel = 3
    else:
        raise ValueError("目前只支持 'CIFAR10' 数据集。")

    all_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_indices)
    target_member_indices = all_indices[:target_train_size]
    shadow_dataset_indices = all_indices[target_train_size:]
    target_member_loader = DataLoader(Subset(train_dataset, target_member_indices), batch_size=target_batch_size, shuffle=True)
    target_non_member_loader = DataLoader(test_dataset, batch_size=target_batch_size, shuffle=False)
    
    # --- 2. 准备目标模型 ---
    print("\n" + "="*10 + " 2. 准备目标模型 " + "="*10)
    if target_model_instance:
        print("检测到已传入目标模型实例，将跳过训练过程。")
        target_model = target_model_instance.to(device)
    else:
        target_model_path = os.path.join(weights_dir, target_model_name, 'target', 'target_model.pth')
        target_model = create_model_from_name(target_model_name, num_classes=num_classes, channel=channel)
        if not force_retrain and os.path.exists(target_model_path):
            target_model = load_model_weights(target_model, target_model_path, device)
            if target_model is None: # 如果加载失败，则重新训练
                 force_retrain = True
        
        if force_retrain or not os.path.exists(target_model_path):
            print("未提供目标模型实例或需要强制重训，开始训练新的目标模型。")
            target_model = train_generic_model(
                model=target_model, train_loader=target_member_loader, device=device, epochs=target_epochs,
                learning_rate=target_learning_rate, l2_ratio=target_l2_ratio, save_path=target_model_path
            )

    # --- 3. 准备攻击模型的 *测试* 数据 ---
    print("\n" + "="*10 + " 3. 准备攻击测试数据 " + "="*10)
    attack_test_data, attack_test_labels, attack_test_classes = prepare_attack_data(
        target_model, target_member_loader, target_non_member_loader, device
    )
    print(f"攻击测试数据已生成，形状: {attack_test_data.shape}")

    # --- 4. 训练/加载影子模型并生成攻击模型的 *训练* 数据 ---
    print("\n" + "="*10 + " 4. 训练影子模型并生成攻击训练数据 " + "="*10)
    all_shadow_attack_data, all_shadow_attack_labels, all_shadow_attack_classes = [], [], []
    shadow_train_size = target_train_size 
    for i in range(n_shadow_models):
        print(f"\n--- 处理影子模型 {i+1}/{n_shadow_models} ---")
        shadow_model_path = os.path.join(shadow_models_dir, f'shadow_model_{i}.pth')
        np.random.shuffle(shadow_dataset_indices)
        current_shadow_indices = shadow_dataset_indices[:shadow_train_size * 2]
        shadow_member_indices = current_shadow_indices[:shadow_train_size]
        shadow_non_member_indices = current_shadow_indices[shadow_train_size:]
        shadow_member_loader = DataLoader(Subset(train_dataset, shadow_member_indices), batch_size=target_batch_size, shuffle=True)
        shadow_non_member_loader = DataLoader(Subset(train_dataset, shadow_non_member_indices), batch_size=target_batch_size, shuffle=False)
        
        shadow_model = create_model_from_name(target_model_name, num_classes=num_classes, channel=channel)
        if not force_retrain and os.path.exists(shadow_model_path):
            shadow_model = load_model_weights(shadow_model, shadow_model_path, device)
            if shadow_model is None:
                force_retrain = True # 临时强制重训当前影子模型
        
        if force_retrain or not os.path.exists(shadow_model_path):
            shadow_model = train_generic_model(
                model=shadow_model, train_loader=shadow_member_loader, device=device, epochs=target_epochs,
                learning_rate=target_learning_rate, l2_ratio=target_l2_ratio, save_path=shadow_model_path
            )
        
        s_attack_data, s_attack_labels, s_attack_classes = prepare_attack_data(
            shadow_model, shadow_member_loader, shadow_non_member_loader, device
        )
        all_shadow_attack_data.append(s_attack_data)
        all_shadow_attack_labels.append(s_attack_labels)
        all_shadow_attack_classes.append(s_attack_classes)

    attack_train_data = np.concatenate(all_shadow_attack_data, axis=0)
    attack_train_labels = np.concatenate(all_shadow_attack_labels, axis=0)
    attack_train_classes = np.concatenate(all_shadow_attack_classes, axis=0)
    print(f"攻击训练数据已生成，形状: {attack_train_data.shape}")

    # --- 5. 训练/加载并评估攻击模型 ---
    print("\n" + "="*10 + " 5. 训练并评估攻击模型 " + "="*10)
    true_all_labels, pred_all_labels = [], []
    unique_classes = np.unique(attack_train_classes)
    for c in unique_classes:
        print(f"\n--- 训练类别 {c} 的攻击模型 ---")
        attack_model_path = os.path.join(attack_models_dir, f'attack_model_class_{c}.pth')
        train_indices = np.where(attack_train_classes == c)[0]
        test_indices = np.where(attack_test_classes == c)[0]
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue
        c_train_x = torch.FloatTensor(attack_train_data[train_indices])
        c_train_y = torch.FloatTensor(attack_train_labels[train_indices]).unsqueeze(1)
        c_test_x = torch.FloatTensor(attack_test_data[test_indices])
        c_test_y = torch.FloatTensor(attack_test_labels[test_indices]).unsqueeze(1)
        train_loader = DataLoader(TensorDataset(c_train_x, c_train_y), batch_size=attack_batch_size, shuffle=True)
        
        attack_model = attack_model_class(input_dim=num_classes, n_hidden=attack_n_hidden)
        if not force_retrain and os.path.exists(attack_model_path):
            attack_model = load_model_weights(attack_model, attack_model_path, device)
            if attack_model is None:
                 force_retrain = True
        
        if force_retrain or not os.path.exists(attack_model_path):
            attack_model = train_generic_model(
                model=attack_model, train_loader=train_loader, device=device, epochs=attack_epochs,
                learning_rate=attack_learning_rate, l2_ratio=attack_l2_ratio, save_path=attack_model_path, is_attack_model=True
            )
        attack_model.eval()
        with torch.no_grad():
            outputs = attack_model(c_test_x.to(device))
            preds = (outputs > 0.5).float().cpu().numpy()
            true_all_labels.extend(c_test_y.numpy().flatten())
            pred_all_labels.extend(preds.flatten())

    # --- 6. 生成最终评估结果 ---
    print("\n" + "="*20 + " 最终评估结果 " + "="*20)
    if not true_all_labels:
        return {"error": "没有可评估的数据，攻击失败。"}
    accuracy = accuracy_score(true_all_labels, pred_all_labels)
    report = classification_report(true_all_labels, pred_all_labels, target_names=['Non-Member', 'Member'], output_dict=True)
    print(f"模型 [{target_model_name}] 的整体攻击准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(true_all_labels, pred_all_labels, target_names=['Non-Member', 'Member']))
    return {'overall_accuracy': accuracy, 'classification_report': report}

# ==============================================================================
# 4. 示例调用 (Example Usage)
# ==============================================================================
if __name__ == '__main__':
    # 为了快速演示，减少epochs
    demo_target_epochs = 5
    demo_attack_epochs = 5

    # --- 示例 1: 评测 ResNet-18 ---
    print("\n" + "#"*20 + " 示例 1: 评测 ResNet-18 " + "#"*20)
    results_resnet = run_membership_inference_attack(
        target_model_name='resnet18',
        target_epochs=demo_target_epochs,
        attack_epochs=demo_attack_epochs,
        force_retrain=True
    )
    with open('mia_results_resnet18.json', 'w', encoding='utf-8') as f:
        json.dump(results_resnet, f, indent=4, ensure_ascii=False)
    print("\nResNet-18 攻击评测完成，结果已保存到 mia_results_resnet18.json")

    # --- 示例 2: 评测 LeNet ---
    print("\n" + "#"*20 + " 示例 2: 评测 LeNet " + "#"*20)
    results_lenet = run_membership_inference_attack(
        target_model_name='lenet',
        target_epochs=demo_target_epochs,
        attack_epochs=demo_attack_epochs,
        force_retrain=True
    )
    with open('mia_results_lenet.json', 'w', encoding='utf-8') as f:
        json.dump(results_lenet, f, indent=4, ensure_ascii=False)
    print("\nLeNet 攻击评测完成，结果已保存到 mia_results_lenet.json")
