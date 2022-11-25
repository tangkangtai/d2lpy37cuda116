
import hashlib
import os
import tarfile
import zipfile
import requests
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 下⾯的download函数⽤来下载数据集，将数据集缓存在本地⽬录（默认情况下为../data）中，并返回下
# 载⽂件的名称。如果缓存⽬录中已经存在此数据集⽂件，并且其sha-1与存储在DATA_HUB中的相匹配，我们
# 将使⽤缓存的⽂件，以避免重复的下载

def download(name, cache_dir=os.path.join('..','data')):
    """下载⼀个DATA_HUB中的⽂件，返回本地⽂件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True) # os.makedirs() 方法用于递归创建目录
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()  # sha1生成一个160bit的结果，通常用40位的16进制字符串表示
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)

        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True) # 向服务器请求数据，服务器返回的结果是个Response对象
    with open(fname, 'wb') as f:
        f.write(r.content)  #response.content能把Response对象的内容以二进制数据的形式返回，适用于图片、音频、视频的下载
    return fname

# ⼀个将下载并解压缩⼀个zip或tar⽂件，
# 另⼀个是将本书中使⽤的所有数据集从DATA_HUB下载到缓存⽬录中。
def download_extract(name, folder=None):
    """下载并解压zip/tar⽂件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)  # os.path.dirname(path) 去掉文件名，返回目录
    data_dir, ext = os.path.splitext(fname) # os.path.splitext(apath)的作用是分离文件名与扩展名，结果以元组返回
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')  # 解压为r,压缩为w
    elif ext in ('.tar', '.gz'):
        fp = tarfile.TarFile(fname, 'r')
    else:
        assert False, '只有zip/tar⽂件可以被解压缩'
    fp.extractall(base_dir) 
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有⽂件"""
    for name in DATA_HUB:
        download(name)

