#coding:utf8
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np

"""
=== 多分类任务完整演示 ===
任务：随机向量，哪一维数字最大就属于第几类
"""

print("=" * 60)
print("多分类任务训练：随机向量最大值识别")
print("=" * 60)

# 1. 生成数据集
input_dim = 5
num_samples = 1000

X = np.random.randn(num_samples, input_dim)
y = np.argmax(X, axis=1)  # 关键：哪一维最大就属于第几类！

print(f"\n1. 数据集生成")
print(f"   - 样本数量: {num_samples}")
print(f"   - 输入维度: {input_dim} (同时也是类别数)")
print(f"   - 标签定义: np.argmax(X, axis=1)")
print(f"   - 示例样本: {X[0].round(3)}")
print(f"   - 示例标签: {y[0]} (最大值在第 {y[0]} 维)")
print(f"   - 类别分布: {np.bincount(y)}")

# 2. 划分训练集和验证集
split_idx = int(num_samples * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"\n2. 数据集划分")
print(f"   - 训练集: {len(y_train)} 样本")
print(f"   - 验证集: {len(y_val)} 样本")

# 3. 定义模型
class MultiClassModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MultiClassModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        y_pred = self.layer2(x)
        return y_pred

model = MultiClassModel(input_dim=5, hidden_dim=10, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"\n3. 模型创建")
print(f"   - 模型结构: {model}")
print(f"   - 损失函数: CrossEntropyLoss (交叉熵)")
print(f"   - 优化器: Adam (学习率=0.01)")

# 4. 训练模型
print(f"\n4. 开始训练 (100 epochs)...")
print("-" * 60)

for epoch in range(100):
    model.train()
    
    # 前向传播
    inputs = torch.FloatTensor(X_train)
    labels = torch.LongTensor(y_train)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 计算准确率
    _, preds = torch.max(outputs, 1)
    train_acc = (preds == labels).float().mean().item()
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_inputs = torch.FloatTensor(X_val)
        val_labels = torch.LongTensor(y_val)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)
        _, val_preds = torch.max(val_outputs, 1)
        val_acc = (val_preds == val_labels).float().mean().item()
    
    # 每10轮打印一次
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/100] | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")

print("-" * 60)
print("训练完成！")

# 5. 测试模型
print(f"\n5. 模型测试")
print("-" * 60)

test_samples = np.random.randn(8, input_dim)
test_labels = np.argmax(test_samples, axis=1)

model.eval()
with torch.no_grad():
    outputs = model(torch.FloatTensor(test_samples))
    _, preds = torch.max(outputs, 1)
    probs = torch.softmax(outputs, dim=1)

correct = 0
for i in range(8):
    is_correct = (test_labels[i] == preds[i].item())
    correct += 1 if is_correct else 0
    print(f"测试 {i+1}:")
    print(f"  输入: {test_samples[i].round(3)}")
    print(f"  真实: 维度 {test_labels[i]} (值: {test_samples[i][test_labels[i]].round(3)})")
    print(f"  预测: 维度 {preds[i].item()}")
    print(f"  概率: {probs[i].numpy().round(3)}")
    print(f"  结果: {'✓ 正确' if is_correct else '✗ 错误'}")
    if i < 7:
        print()

print("-" * 60)
print(f"测试准确率: {correct/8*100:.1f}% ({correct}/{8} 正确)")
print("=" * 60)

# 保存模型
torch.save(model.state_dict(), 'multi_class_model.pth')
print("\n模型已保存: multi_class_model.pth")
