import numpy as np
import torch

arr = np.array((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16),dtype=np.float32)
img_new = torch.from_numpy(arr) # 转化为tensor
print("------------第一次开始-------------")
print(img_new)
print(img_new.shape)
print("------------第一次结束-------------")

img_new = img_new.reshape(2, 8)
print(img_new)
print(img_new.shape)
print("------------第二次结束-------------")

img_new = img_new.reshape(4, 4)
print(img_new)
print(img_new.shape)
print("------------第三次结束-------------")

img_new = img_new.reshape(1, 16)
print(img_new)
print(img_new.shape)
print("------------第四次结束-------------")

img_new = img_new.reshape(4, 4)
print(img_new)
print(img_new.shape)
print("------------第五次结束-------------")

img_new = img_new.reshape(-1, 16)
print(img_new)
print(img_new.shape)
print("------------第六次结束-------------")