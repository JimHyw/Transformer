class FeedForwardNetwork:
    def __init__(self, d_model, dff):
        self.W1 = np.random.randn(d_model, dff)
        self.b1 = np.zeros(dff)
        self.W2 = np.random.randn(dff, d_model)
        self.b2 = np.zeros(d_model)

    def call(self, x):
        """GELU激活函数近似实现"""
        x = np.matmul(x, self.W1) + self.b1
        x = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))  # GELU近似
        return np.matmul(x, self.W2) + self.b2
