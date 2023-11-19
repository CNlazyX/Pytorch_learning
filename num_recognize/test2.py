from PIL import Image
import numpy as np

img6 = Image.open("6.png")
img6 = img6.convert("L")
img6 = np.array(img6)
print(img6.shape)


img_nsh = Image.open("nsh.png") #读取图像
img_nsh.show()
img_nsh = img_nsh.convert("L") #以灰度图像读取
img_nsh.show()

img_nsh = np.array(img_nsh)
print(img_nsh.shape)

#img_nsh.show()  #将原始图像展示出来
#img_nsh.show()  #将改变后的灰度图像展示出来

#image1.save("new_image adress") #保存图像到新的地址
