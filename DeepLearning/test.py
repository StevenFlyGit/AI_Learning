import torch.nn as nn
import torch


# nn.Embedding 本质上是一个查找表，它内部维护了一个形状为 [num_embeddings, embedding_dim] 的权重矩阵。
embedding = nn.Embedding(
	num_embeddings=10,       # 必填：词表大小
	embedding_dim=3,         # 必填：每个词向量的维度
	padding_idx=0,           # 填充索引：输入中的 0 表示补齐位置；其向量默认全 0，且不参与梯度更新
	max_norm=None,            # 向量范数上限，None 表示不限制
	norm_type=2.0,            # 范数类型
	scale_grad_by_freq=False, # 是否按出现频率缩放梯度
	sparse=False,             # 是否使用稀疏梯度
	_weight=None,             # 可选的预训练权重矩阵
)


# 对时间步 t，输入为 x_t，上一时刻隐藏状态为 h_{t-1}，GRU 的计算过程为：
#
#   r_t = sigmoid(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)  # 重置门
#   z_t = sigmoid(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)  # 更新门
#   n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))  # 候选状态
#   h_t = (1 - z_t) * n_t + z_t * h_{t-1}  # 当前隐藏状态
# 其中 sigmoid(a)=1/(1+exp(-a))，* 表示逐元素相乘，W 和 b 是可学习参数。# 重置门决定生成候选状态时保留多少历史信息；更新门决定保留多少旧状态、写入多少候选状态。
# z_t 接近 1 时主要保留 h_{t-1}，接近 0 时主要使用 n_t。
# 双向 GRU 分别按正向和反向处理序列，并拼接两个方向的输出；多层 GRU 将上一层的输出作为下一层的输入。

# nn.GRU 是门控循环单元，用于处理序列数据；它通过更新门和重置门控制历史信息的保留与遗忘。
# 这里 input_size 规定输入 x_t 的特征维度（不是权重 W 的维度）：x_t 的形状为 [input_size]。
# hidden_size 规定隐藏状态 h_t 的特征维度：h_t 的形状为 [hidden_size]。
# 因此，输入到隐藏状态的权重矩阵形状为 [hidden_size, input_size]，
# 隐藏状态到隐藏状态的权重矩阵形状为 [hidden_size, hidden_size]（每个门分别有对应参数）
gru = nn.GRU(
	input_size=3,       # 必填：每个时间步输入特征的维度
	hidden_size=5,      # 必填：隐藏状态的特征维度
	num_layers=2,       # 循环层数；多层 GRU 会将上一层的输出作为下一层的输入
	batch_first=True,   # 输入和输出的形状是否为 [batch, sequence, feature]；否则为 [sequence, batch, feature]
	bidirectional=True,  # 是否使用双向 GRU；为 True 时会同时处理正向和反向序列
)


def main():
    tensor = torch.randint(0, 10, (2, 4))  # 随机生成一个形状为 [2, 4] 的整数张量，表示两个序列，每个序列长度为 4
    print("Input tensor:\n", tensor)
    # view(-1, 4) 表示变形成“每行 4 个元素”的二维张量。
	# -1 表示让 PyTorch 根据元素总数自动推断这一维：2 * 4 / 4 = 2。
	# 一次 view 或 reshape 中通常只能有一个 -1；其余维度确定后，PyTorch 才能唯一计算它的大小。
	# 张量原本有几维不影响这一规则；也可以不使用 -1，或在不同操作中分别使用它。
	# 所以结果形状为 [2, 4]。注意 view 不会原地修改 tensor，需要接收返回值。
    # view 和 reshape 的区别：
    #   view 只能在张量是连续的情况下使用；reshape 可以在张量不连续的情况下使用。
    # 张量连续的意思是它在内存中是按顺序排列的，没有被切片或转置等操作打乱。
    #   view 返回的是原张量的视图（共享内存）， 而 reshape 返回的是原张量的副本（不共享内存）。
    tensor2 = tensor.view(-1, 2)
    tensor[0, 0] = 100  # 修改 tensor 的第一个元素
    print("Reshaped tensor2:\n", tensor2)
    print("Reshaped tensor:\n", tensor)



    tensor3 = torch.randint(0, 10, (2, 1, 1))  # 随机生成一个形状为 [2, 1, 1] 的整数张量
    print("Input tensor3:\n", tensor3)
    tensor3 = tensor3.expand(2, 2, 3) # expand 会返回一个新的张量，形状为 [2, 2, 3]，但不复制数据；它只是改变了张量的视图。
    # expnad 和 view 的区别：
    #   expand 可以在张量的某些维度上扩展，而 view 只能改变张量的形状。
    # expand 和 repeat 的区别：
    #   expand 不会复制数据，而 repeat 会复制数据。
    print("Expanded tensor3:\n", tensor3)


    tensor4 = torch.randint(0, 10, (3, 2, 3))  # 随机生成一个形状为 [3, 2, 3] 的整数张量
    print("Input tensor4:\n", tensor4)
    tensor4 = tensor4.gather(1, torch.tensor( [[[1], [0]], [[0], [1]]])) # 沿第 1 维收集；index 形状为 [2, 2, 1]，所以结果也是 [2, 2, 1]。
    # gather() 的参数：
    #   dim：指定沿着哪个维度收集数据。从0开始计数，0表示第一个维度，1表示第二个维度，以此类推。
    #   index：指定收集数据的索引张量。除 dim 外，各维大小不能超过输入对应维度；输出形状与 index 相同。
    # 举例说明 gather() 的作用：
    #   假设有一个张量 tensor4，形状为 [3, 2, 3]，其内容如下：
    #   tensor4 = [[[1, 2, 3], [4, 5, 6]],
    #              [[7, 8, 9], [10, 11, 12]],
    #              [[13, 14, 15], [16, 17, 18]]]
    #   如果我们调用 tensor4.gather(1, torch.tensor([[[1], [0]], [[0], [1]]]))，则会沿第 1 维收集数据。
    #   对于第一个样本（tensor4[0]），我们收集 tensor4[0][1] 和 tensor4[0][0]，得到 [[4], [1]]。
    #   对于第二个样本（tensor4[1]），我们收集 tensor4[1][0] 和 tensor4[1][1]，得到 [[7], [10]]。
    #   因此，最终结果为：
    #   tensor4 = [[[4], [1]], [[7], [10]]]
    print("Gathered tensor4:\n", tensor4)



    tensor5 = torch.randint(0, 10, (2, 1, 3))  # 随机生成一个形状为 [2, 1, 3] 的整数张量
    print("Input tensor5:\n", tensor5)
    tensor5 = tensor5.squeeze()  # squeeze() 会移除所有维度为 1 的维度；如果没有维度为 1，则返回原张量。
    # 什么叫维度为1的维度？
    #   维度为1的维度是指在张量中，某个维度的大小为1。例如，在张量 [2, 1, 3] 中，第1个维度（从0开始计数）的大小为1，因此它是维度为1的维度。
    print("Squeezed tensor5:\n", tensor5)
    tensor5 = tensor5.unsqueeze(0)  # unsqueeze() 会在指定位置插入一个维度为 1 的维度；如果指定位置已经有维度，则返回原张量。
    print("Unsqueezed tensor5:\n", tensor5)
main()