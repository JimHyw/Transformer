class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        # 手动初始化权重矩阵（替代框架的Dense层）
        self.WQ = np.random.randn(d_model, d_model)
        self.WK = np.random.randn(d_model, d_model)
        self.WV = np.random.randn(d_model, d_model)
        self.WO = np.random.randn(d_model, d_model)

    def split_heads(self, x):
        """分割多头"""
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.num_heads, self.depth).transpose(0, 2, 1, 3)

    def call(self, Q, K, V, mask=None):
        """前向计算"""
        Q = np.matmul(Q, self.WQ)  # (batch, seq_len, d_model)
        K = np.matmul(K, self.WK)
        V = np.matmul(V, self.WV)

        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, depth)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 计算每个头的注意力
        attention_output = scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(-1, -1, self.d_model)
        return np.matmul(attention_output, self.WO)  # 合并多头输出
