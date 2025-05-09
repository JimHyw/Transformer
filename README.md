Python与NumPy完成Transformer关键组件开发，不包括如梯度裁剪、学习率调度，其中:
1.权重初始化‌：使用np.random.randn模拟框架的随机初始化；
2.‌模块化设计‌：每个组件（如MultiHeadAttention）独立实现，便于调试；
3.‌性能优化‌：对大规模数据需用Cython或Numba加速计算。