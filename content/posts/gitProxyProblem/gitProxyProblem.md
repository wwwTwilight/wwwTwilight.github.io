---
date: '2025-05-15T13:02:13+08:00'
draft: false
title: '关于git的代理问题'
---

这是题外话：鉴于很多时候app不认中文路由，从这篇文章开始，后续所有的文章都采用英文路由，之前的就不管了，反正除了我，应该也没什么人看😭

下面是正文

---

# git 代理问题

很多人应该都遇到了在使用git的时候出现的代理问题，明明可以直接登上github，但是git clone和git push之类的操作却总是提示超时，这是怎么个事。

# 解决方案

原因其实是你的git工具没有设置代理，Git 是独立的命令行工具，它不会自动使用系统代理，其使用的底层库（如 libcurl）默认不走任何代理。

通过以下的命令可以使用代理

```bash
git config --global http.proxy 127.0.0.1:7890
git config --global https.proxy 127.0.0.1:7890
```

关于最后的端口以及ip，127.0.0.1是本地地址的一个特殊地址，就是每一台电脑都有这个地址，其作用主要是用于在网络中表示自己（类似localhost），而7890则是代理的端口号，具体可以看clash的端口号，默认是7890。

# 其他

除了添加git代理的命令，删除和查看同样是比较重要的，这里也给出命令

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
git config --global --get http.proxy
git config --global --get https.proxy
```

# 针对linux虚拟机的额外说明

如果你使用的是linux虚拟机，那么你可能会发现，你添加了代理之后，git还是无法使用代理，这是为什么呢？

~~这个具体原因我也不是很清楚，但是可以记录一下解决方案，我是直接取消了git的代理设置，在物理机上开启了clash的allow LAN模式，然后虚拟机在设置里直接连接代理，这样就可以使用了。~~

这里讲一下正确的解决方案，在物理机上开启clash的allow LAN模式，然后在虚拟机的设置里设置对应的代理，是可以做到使用代理，访问外网，但是此时git并没有走代理，我们需要手动设置git的代理，命令如下：

```bash
git config --global http.proxy xxx.xxx.xxx.xxx:7890
git config --global https.proxy xxx.xxx.xxx.xxx:7890
```

其中xxx.xxx.xxx.xxx是你的物理机的ip地址，7890是端口号，这里ip不填127.0.0.1，是因为这个ip是本机的ip，也就是说虚拟机的ip，但是虚拟机并没有在7890端口上开启代理，所以设置为127.0.0.1:7890是无效的，而使用物理机ip就能够正常使用，**但是这样子有个缺陷，当物理机的网络环境发生变化，这个代理的地址也要发生变化，不然不能正常使用**

# 如果还不行

那我建议你换一架🪜，有些时候是这样的