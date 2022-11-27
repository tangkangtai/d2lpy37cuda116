
import torch


def slice_test():
    x = torch.arange(24)
    print(x)
    for i in range(6):
        idx = slice(i * 4, (i + 1) * 4)
        print(x[idx])


def sort_rand_test():
    n_train = 10
    p = torch.rand(n_train)
    print(p)
    p5 = p * 5
    print(p5)
    p5_s = torch.sort(p5)  # torch.sort排序后，返回排序后的tensor,和排序好了的元素对应之前的下标
    print(p5_s)


def repeat_interleave_test():
    a = torch.arange(0, 5, 0.1)
    print(a)
    print(a.shape)
    a_r = a.repeat_interleave(10)
    print(a_r)
    a_r_r = a_r.reshape((-1, 10))
    print(a_r_r)
    print(a_r_r.shape)


def eye_test():
    e = torch.eye(10)
    print(e)
    e1 = 1 - e
    print(e1)
    e2 = e1.type(torch.bool)  # 当不指定dtype时,返回类型.当指定dtype时,返回类型转换后的数据,如果类型已经符合要求
    print(e2)


def tensor_test():
    x_train, _ = torch.sort(torch.rand(10) * 5)
    print(x_train)
    print(x_train.shape)
    x_tile = x_train.repeat((10, 1))
    print(x_tile)
    print(x_tile.shape)
    k = x_tile[(1 - torch.eye(10)).type(torch.bool)]
    print(k)
    print(k.shape)
    ks = k.reshape((10, -1))
    print(ks)
    print(ks.shape)


if __name__ == '__main__':
    # slice_test()
    # sort_rand_test()
    # repeat_interleave_test()
    # eye_test()
    # tensor_test()
    x = torch.rand(2, 3)
    print(x)
    print(x.shape)
    xr0 = x.repeat_interleave(4, 0)
    print(xr0)
    print(xr0.shape)
    xr1 = x.repeat_interleave(4, 1)
    print(xr1)
    print(xr1.shape)