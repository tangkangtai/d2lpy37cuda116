
import torch

if __name__ == '__main__':
    x = torch.arange(24)
    print(x)
    for i in range(6):
        idx = slice(i * 4, (i+1)*4)
        print(x[idx])