"""文件下载模块解析"""
# urlretrieve(url, filename=None, reporthook=None, data=None)
"""
参数 finename 指定了保存本地路径（如果参数未指定，urllib会生成一个临时文件保存数据。）
参数 reporthook 是一个回调函数，当连接上服务器、以及相应的数据块传输完毕时会触发该回调，我们可以利用这个回调函数来显示当前的下载进度。
参数 data 指 post 到服务器的数据，该方法返回一个包含两个元素的(filename, headers)元组，filename 表示保存到本地的路径，header 表示服务器的响应头。
example:
下面通过例子来演示一下这个方法的使用，这个例子将一张图片抓取到本地，保存在此文件夹中，同时显示下载的进度。
"""
from six.moves import urllib


def Schedule(a, b, c):
    """
    a:已经下载的数据块
    b:数据块的大小
    c:远程文件的大小
    """
    per = 100.0*float(a*b)/float(c)
    if per > 100:
        per = 100
    print("a", a)
    print("b", b)
    print("c", c)
    print('{:.2f}%'.format(per))


url = 'https://avatars1.githubusercontent.com/u/14261323?s=400&u=150449ce27748c3b23b5175f8c8342c918ae6aa8&v=4'
local = 'mylogo.png'
filename, _ = urllib.request.urlretrieve(url, local, Schedule)
# ('mylogo.png', <http.client.HTTPMessage object at 0x000001FD6491D6D8>)
print(filename)
# mylogo.png

# a 0
# b 8192
# c 38225
# 0.00%
# a 1
# b 8192
# c 38225
# 21.43%
# a 2
# b 8192
# c 38225
# 42.86%
# a 3
# b 8192
# c 38225
# 64.29%
# a 4
# b 8192
# c 38225
# 85.72%
# a 5
# b 8192
# c 38225
# 100.00%
