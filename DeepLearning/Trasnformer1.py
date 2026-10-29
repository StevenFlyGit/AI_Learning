
import torch
import math
import torch.nn as nn
from dataclasses import dataclass







class SeftAttentionWPF(nn.Module):

    '''
    hidden_dim：隐藏层维度，表示模型中间层的特征维度
    '''
    def __init__(self, hidden_dim) -> None: # ‘-> None’ 表示该函数没有返回值，符合 Python 的类型提示规范
        super().__init__()
        self.hidden_dim = hidden_dim
        # 定义三个线性层，分别用于生成查询（query）、键（key）和值（value）
        # # 初始化三个线性层，输入和输出的维度都是 hidden_dim，得到Q、K、V的投影矩阵

        self.query = nn.Linear(hidden_dim, hidden_dim) 
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)


    '''
    X: 输入张量，形状为 (batch_size, seq_len, hidden_dim)，其中 batch_size 是批次大小，seq_len 是序列长度，hidden_dim 是隐藏层维度
    为什么形状是 (batch_size, seq_len, hidden_dim)？因为在自注意力机制中，我们通常处理的是一个批次的序列数据，每个序列由多个位置组成，每个位置有一个表示向量，表示向量的维度就是隐藏层维度。
    该函数实现了自注意力机制的前向传播过程，计算输入序列中每个位置的表示向量，考虑到序列中其他位置的信息。具体步骤如下：
    1. 生成查询、键和值：通过线性变换将输入张量 X 映射到查询（query）、键（key）和值（value）空间，得到三个新的张量 Q、K、V。
    2. 计算注意力分数：通过矩阵乘法计算查询 Q 与键 K 的点积，得到注意力分数矩阵 scores。然后将 scores 除以根号下隐藏层维度的平方根，以稳定梯度。
    3. 归一化注意力分数：对注意力分数矩阵 scores 进行 softmax 操作，得到归一化的注意力权重矩阵 attn_weights。
    4. 计算加权和：将注意力权重矩阵 attn_weights 与值 V 相乘，得到加权和的输出张量 output。
    5. 返回输出张量 output，形状为 (batch_size, seq_len, hidden_dim)，表示每个位置的表示向量，考虑了序列中其他位置的信息。
    该函数实现了自注意力机制的核心计算过程，使模型能够在处理序列数据时捕捉到不同位置之间的依赖关系，从而提高模型的表示能力。
    '''
    def forward(self, X):
        # X shpape: (batch_size, seq_len, hidden_dim)
        '''
        Q = self.query(X)实际调用的是nn.Linear 的前向计算:
        self.query = nn.Linear(hidden_dim, hidden_dim)
        调用链为：
        self.query(X)
        → nn.Linear.__call__(X)
        → nn.Linear.forward(X)
        → F.linear(X, self.query.weight, self.query.bias)
        数学形式是：
        Q = X @ self.query.weight.T + self.query.bias
        '''
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        # 计算注意力分数
        # K.transpose(-2, -1) 表示对 K 的最后两个维度进行转置，即将形状为 (batch_size, seq_len, hidden_dim) 的 K 转换为 (batch_size, hidden_dim, seq_len)，以便与 Q 进行矩阵乘法。
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim) # 按照公式计算
        # 对分数进行softmax归一化
        attention_weights = torch.softmax(scores, dim=-1) # dim=-1 表示对最后一个维度进行softmax归一化，即对每个查询位置的注意力分数进行归一化，使其和为1。
        # 计算加权和
        output = torch.matmul(attention_weights, V)
        return output

 