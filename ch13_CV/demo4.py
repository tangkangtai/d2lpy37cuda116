
import torch
from d2l import torch as d2l
# 以每个像素为中⼼，⽣成多个缩放⽐和宽⾼⽐（aspect ratio）
# 不同的边界框。这些边界框被称为锚框（anchor box）
torch.set_printoptions(2) # 精简输出精度
# 假设输⼊图像的⾼度为h，宽度为w。我们以图像的每个像素为中⼼⽣成不同形状的锚框：缩放⽐为s ∈ (0, 1]，
# 宽⾼⽐为r > 0。那么锚框的宽度和⾼度分别是ws√r和hs/√r。
# 请注意，当中⼼位置给定时，已知宽和⾼的锚框是确定的。
def multibox_prior(data, sizes, ratios):
    """⽣成以每个像素为中⼼具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # 为了将锚点移动到像素的中⼼，需要设置偏移量。
    # 因为⼀个像素的⾼为1且宽为1，我们选择偏移我们的中⼼0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height # 在y轴上缩放步⻓
    steps_w = 1.0 / in_width # 在x轴上缩放步⻓
    # # ⽣成锚框的所有中⼼点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    # # ⽣成“boxes_per_pixel”个⾼和宽，
    # # 之后⽤于创建锚框的四⻆坐标(xmin,xmax,ymin,ymax)
    w = torch.cat( (size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:])) ) * in_height / in_width # 处理矩形输⼊

    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # 除以2来获得半⾼和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    # 每个中⼼点都将有“boxes_per_pixel”个锚框，
    # 所以⽣成含所有锚框中⼼的⽹格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

img = d2l.plt.imread('image/catdog.png')
h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)

# 将锚框变量Y的形状更改为(图像⾼度,图像宽度,以同⼀像素为中⼼的锚框的数量,4)后，我们可以获得以指定
# 像素的位置为中⼼的所有锚框
# 我们访问以（250,250）为中⼼的第⼀个锚框。
# 它有四个元素：锚框左上⻆的(x, y)轴坐标和右下⻆的(x, y)轴坐标。
# 将两个轴的坐标各分别除以图像的宽度和⾼度后，
# 所得的值介于0和1之间。



