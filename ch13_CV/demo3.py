
import torch
from d2l import torch as d2l
# ⽬标检测和边界框
# 在图像分类任务中，我们假设图像中只有⼀个主要物体对象，我们只关注如何识别其类别。
# 然⽽，很多时候图像⾥有多个我们感兴趣的⽬标，我们不仅想知道它们的类别，还想得到它们在图像中的具体位置。
# 在计算机视觉⾥，我们将这类任务称为⽬标检测（object detection）或⽬标识别（object recognition）
d2l.set_figsize()
img = d2l.plt.imread('image/catdog.png')
# print(img.shape)
# d2l.plt.imshow(img)
# d2l.plt.show()

# 边界框
# box_corner_to_center从两⻆表⽰法转换为中⼼宽度表⽰法，⽽box_center_to_corner反之亦然。
# 输⼊参数boxes可以是⻓度为4的张量，也可以是形状为（n，4）的⼆维张量，其中n是边界框的数量。
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，⾼度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，⾼度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# 我们将根据坐标信息定义图像中狗和猫的边界框。图像中坐标的原点是图像的左上⻆，向右的⽅向为x轴的
# 正⽅向，向下的⽅向为y轴的正⽅向
x_scale = float(294.0 / 700.0)
y_scale = float(225.0 / 580.0)
dog_bbox, cat_bbox = [60.0 * x_scale, 45.0*y_scale, 378.0*x_scale, 516.0*y_scale], [400.0*x_scale, 112.0*y_scale, 655.0*x_scale, 493.0*y_scale]
print(dog_bbox)
print(cat_bbox)

# 可以通过转换两次来验证边界框转换函数的正确性。
# boxes = torch.tensor((dog_bbox, cat_bbox))
# print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

# 我们可以将边界框在图中画出，以检查其是否准确。 画之前，我们定义一个辅助函数bbox_to_rect。 它将边界框表示成matplotlib的边界框格式。
def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    #   ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3]-bbox[1], fill=False, edgecolor=color, linewidth=2
    )
# 在图像上添加边界框之后，我们可以看到两个物体的主要轮廓基本上在两个框内
fig = d2l.plt.imshow(img) # AxesImage
print(fig)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show()


