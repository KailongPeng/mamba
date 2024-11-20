# 下面是一个简单的PyTorch实现的Transformer模型的例子，包括网络的搭建、训练和测试。这个例子会使用PyTorch的 `nn`
# 模块来构建模型，使用 `optim` 模块来定义优化器，以及使用 `torch.utils.data` 来处理数据。


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


# 定义一个简单的数据集
class SimpleDataset(Dataset):
    def __init__(self, size, seq_length):
        # 初始化数据集，这里只是随机生成一些数据作为示例
        self.data = torch.randn(size, seq_length)
        self.targets = torch.randint(0, 2, (size,))

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, index):
        # 根据索引返回数据和标签
        return self.data[index], self.targets[index]


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dim_feedforward, num_layers):
        super(TransformerModel, self).__init__()
        # 编码器层的数量
        self.num_layers = num_layers
        # 多头自注意力机制
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        # 前馈网络
        self.feed_forward = nn.Linear(input_dim, dim_feedforward)
        # LayerNorm
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        # 最后的全连接层
        self.final_FC = nn.Linear(dim_feedforward, output_dim)

    def forward(self, x):
        # 多头自注意力 + 前馈网络 + LayerNorm
        attn_output = self.multi_head_attn(x, x)[0]
        attn_output = self.layer_norm1(attn_output + x)
        # 前馈网络 + LayerNorm
        ff_output = self.feed_forward(attn_output)
        ff_output = F.relu(ff_output)
        ff_output = self.layer_norm2(ff_output + attn_output)
        # 经过多次编码器层
        for _ in range(self.num_layers - 1):
            attn_output = self.multi_head_attn(attn_output, attn_output)[0]
            attn_output = self.layer_norm1(attn_output + attn_output)
            ff_output = self.feed_forward(attn_output)
            ff_output = F.relu(ff_output)
            ff_output = self.layer_norm2(ff_output + attn_output)
        # 最后的全连接层
        output = self.final_FC(ff_output)
        return output


# 实例化数据集和数据加载器
dataset = SimpleDataset(size=100, seq_length=10)  # 100个样本，序列长度为10
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # 批量大小为4

# 实例化模型
model = TransformerModel(input_dim=10, output_dim=2, num_heads=2, dim_feedforward=20, num_layers=2)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(5):  # 训练5个epoch
    for batch, (data, targets) in enumerate(dataloader):
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失信息
        if batch % 10 == 0:
            print(f'Epoch [{epoch + 1}/5] Batch [{batch + 1}/{len(dataloader)}] Loss: {loss.item():.4f}')

# 测试模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 不计算梯度
    for data, targets in dataloader:
        outputs = model(data)
        # 这里可以计算测试的准确率等指标
        # ...

print("测试完成！")

# 这个例子中，我们首先定义了一个简单的数据集
# `SimpleDataset`，它随机生成了一些数据作为输入和目标。然后，我们定义了
# `TransformerModel` 类，它包含了多头自注意力机制、前馈网络和LayerNorm。在模型的 `forward`
# 方法中，我们实现了数据通过Transformer模型的流程。
#
# 接着，我们实例化了数据集和数据加载器，并设置了模型、损失函数和优化器。在训练循环中，我们进行了前向传播、计算损失、反向传播和参数更新。最后，我们设置了模型为评估模式，并进行了测试。
#
# 请注意，这个例子非常基础，实际应用中Transformer模型会更复杂，包括位置编码、掩码、更复杂的优化策略等。此外，为了真正运行这个代码，你需要确保你的环境已经安装了PyTorch。
