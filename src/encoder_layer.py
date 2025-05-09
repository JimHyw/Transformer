class EncoderLayer:
    def __init__(self, d_model, num_heads, dff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)

    def call(self, x):
        attn_output = self.mha.call(x, x, x)  # 自注意力
        x = self.layernorm1.call(x + attn_output)  # 残差连接 + LayerNorm
        ffn_output = self.ffn.call(x)
        return self.layernorm2.call(x + ffn_output)
