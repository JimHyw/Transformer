def cross_entropy_loss(y_true, y_pred):
    """手动实现交叉熵损失"""
    m = y_true.shape[0]
    logits = y_pred - np.max(y_pred, axis=-1, keepdims=True)  # 数值稳定
    exp_logits = np.exp(logits)
    softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    loss = -np.sum(y_true * np.log(softmax + 1e-9)) / m
    return loss
