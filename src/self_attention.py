import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """计算缩放点积注意力"""
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)  # (batch, seq_len, seq_len)
    if mask is not None:
        scores += mask * -1e9  # 掩码处理（如解码器的上三角掩码）
    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)  # (batch, seq_len, d_model)
    return output

def softmax(x, axis=-1):
    """手动实现softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
