from PIL import Image

# 打开图像文件
image = Image.open("image.jpg")

# 调整图像大小
new_size = (800, 600)
resized_image = image.resize(new_size)

# 裁剪图像
box = (100, 100, 500, 400)
cropped_image = image.crop(box)

# 旋转图像
rotated_image = image.rotate(45)

# 生成缩略图
thumbnail_size = (200, 200)
image.thumbnail(thumbnail_size)

# 保存调整后的图像
resized_image.save("resized_image.jpg")



from PIL import Image, ImageDraw, ImageFont
# 创建绘图对象
draw = ImageDraw.Draw(image)

# 添加水印文本
text = "Watermark"
font = ImageFont.truetype("arial.ttf", 36)
text_size = draw.textsize(text, font)
text_position = (image.width - text_size[0], image.height - text_size[1])
draw.text(text_position, text, fill=(255, 255, 255), font=font)

# 保存带水印的图像
image.save("watermarked_image.jpg")