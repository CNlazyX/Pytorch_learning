from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot
import torchvision.transforms as transforms

def convert_to_float32(value):
     return np.float32(value)

# 预处理 读取本地图片 并灰度化
img6 = Image.open("6.png")
img6 = img6.convert("L")
img6 = np.array(img6) # 转化为numpy
print(img6)

#img6 = img6.astype(float)
img6 = convert_to_float32(img6)
print(img6)

print("===============================================")
print(img6.dtype)
print(img6.shape)

# 图片格式转化
img_new = torch.from_numpy(img6) # 转化为tensor

print(img_new.dtype)
print(img_new.shape)

img_new = img_new.reshape(1, 784) # 维度变化（降维）
print(img_new)
print(img_new.dtype)
print(img_new.shape)

# show一下图片看看成功没
pyplot.imshow(img_new[0].reshape((28, 28)), cmap="gray")

#my_img = map(torch.tensor,img6)
#print(my_img.shape)