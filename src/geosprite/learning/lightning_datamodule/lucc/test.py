import numpy as np
from einops import rearrange, repeat

# 假设 arr 是你的一维 bool 数组
arr = [np.random.randn(36) for i in range(2)]
# 找到等于False的所有元素的索引
b = repeat(arr, pattern="b (h w) -> c b (h d) w", h=6, c=2, d=2)

pass