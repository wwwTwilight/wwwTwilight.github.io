---
date: '2026-05-24T13:58:54+08:00'
draft: False
title: '曲线&曲面积分解法及例题'
---

本文内容完全由ChatGPT生成，但是个人认为很有价值，改了一下格式就放了上来

# 曲线积分与曲面积分典型题单（含详细解法）

我把题目按“方法”分类，而不是按“章节”分类。
这样你会更容易形成：

* “看到题 → 判断用什么方法”的能力。

内容包括：

* 第一类曲线积分
* 第二类曲线积分
* 第一类曲面积分
* 第二类曲面积分
* 直接法
* 格林公式
* 高斯公式
* 补线法
* 路径无关
* 参数化
* 对称性技巧

---

# 一、第一类曲线积分（弧长型）

形式：

$$
\int_L f(x,y),\mathrm ds
$$

本质：

> 沿着曲线对“长度”加权。

---

# 题1（直接参数法）

计算：

$$
\int_L (x+y),\mathrm ds
$$

其中 (L) 为圆

$$
x^2+y^2=1
$$

逆时针一周。

---

## 解

圆参数化：

$$
x=\cos t,\quad y=\sin t,\quad 0\le t\le 2\pi
$$

弧长元：

$$
\mathrm ds=\sqrt{\left(\frac{dx}{dt}\right)^2+\left(\frac{dy}{dt}\right)^2},dt
$$

因为：

$$
\frac{dx}{dt}=-\sin t,\quad \frac{dy}{dt}=\cos t
$$

所以：

$$
\mathrm ds=dt
$$

于是：

$$
\int_0^{2\pi}(\cos t+\sin t),dt
$$

计算：

$$
=\left[\sin t-\cos t\right]_0^{2\pi}
$$

$$
=(0-1)-(0-1)=0
$$

---

## 结论

$$
\boxed{0}
$$

---

# 题2（第一类曲线积分中的对称性）

计算：

$$
\int_L x^2,ds
$$

其中 (L) 为半径 (a) 的圆周。

---

## 解

参数化：

$$
x=a\cos t,\quad y=a\sin t
$$

$$
ds=a,dt
$$

于是：

$$
\int_0^{2\pi} a^2\cos^2 t\cdot a,dt
$$

$$
=a^3\int_0^{2\pi}\cos^2 t,dt
$$

利用：

$$
\cos^2 t=\frac{1+\cos2t}{2}
$$

得到：

$$
=a^3\cdot \pi
$$

---

## 结论

$$
\boxed{\pi a^3}
$$

---

# 二、第二类曲线积分（做功型）

形式：

$$
\int_L P,dx+Q,dy
$$

本质：

> 向量场沿路径做功。

---

# 题3（直接法）

计算：

$$
\int_L y,dx+x,dy
$$

其中：

$$
L:\ y=x^2,\quad (0,0)\to(1,1)
$$

---

## 解

参数化：

$$
x=t,\quad y=t^2,\quad 0\le t\le1
$$

于是：

$$
dx=dt,\quad dy=2t,dt
$$

代入：

$$
\int_0^1 (t^2),dt+t(2t),dt
$$

$$
=\int_0^1 3t^2,dt
$$

$$
[t^3]_0^1
$$

$$
=1
$$

---

## 结论

$$
\boxed{1}
$$

---

# 题4（路径无关）

计算：

$$
\int_L (2xy+y^2),dx+(x^2+2xy),dy
$$

其中 (L) 为任意连接 $((0,0)\to(1,1))$ 的光滑曲线。

---

## 第一步：判断路径无关

设：

$$
P=2xy+y^2,\quad Q=x^2+2xy
$$

检查：

$$
\frac{\partial P}{\partial y}=2x+2y
$$

$$
\frac{\partial Q}{\partial x}=2x+2y
$$

相等。

因此：

> 积分与路径无关。

---

## 第二步：找势函数

求：

$$
\phi_x=P
$$

即：

$$
\phi_x=2xy+y^2
$$

对 (x) 积分：

$$
\phi=x^2y+xy^2+C(y)
$$

对 (y) 求导：

$$
\phi_y=x^2+2xy+C'(y)
$$

与 (Q) 比较：

$$
C'(y)=0
$$

故：

$$
\phi=x^2y+xy^2
$$

---

## 第三步：代端点

$$
\int_L =\phi(1,1)-\phi(0,0)
$$

$$
=1+1
$$

---

## 结论

$$
\boxed{2}
$$

---

# 三、补线 + 格林公式

格林公式：

$$
\oint_L P\,dx+Q\,dy = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right)\,dA
$$

---

# 题5（补线经典题）

计算：

$$
\int_L (-y,dx+x,dy)
$$

其中 (L) 为上半圆：

$$
x^2+y^2=1
$$

从 $((-1,0)\to(1,0))$。

---

## 思路

不是闭合曲线。

所以：

> 补上线段。

---

## 第一步：构造闭曲线

补：

$$
C:\ (1,0)\to(-1,0)
$$

沿 (x) 轴。

闭曲线记作：

$$
\Gamma=L+C
$$

---

## 第二步：用格林公式

这里：

$$
P=-y,\quad Q=x
$$

所以：

$$
\frac{\partial Q}{\partial x}=1
$$

$$
\frac{\partial P}{\partial y}=-1
$$

于是：

$$
\oint_\Gamma =\iint_D 2,dA
$$

区域是半圆：

$$
\frac{\pi}{2}
$$

故：

$$
=2\cdot\frac{\pi}{2}=\pi
$$

---

## 第三步：减去补线积分

补线：

$$
y=0,\quad dy=0
$$

故：

$$
\int_C (-y,dx+x,dy)=0
$$

因此：

$$
\int_L=\pi
$$

---

## 结论

$$
\boxed{\pi}
$$

---

# 四、格林公式求面积

---

# 题6（格林公式求面积）

利用曲线积分求椭圆面积：

$$
\frac{x^2}{a^2}+\frac{y^2}{b^2}=1
$$

---

## 面积公式

由格林公式：

$$
A=\oint_L x,dy
$$

---

## 参数化

$$
x=a\cos t,\quad y=b\sin t
$$

$$
dy=b\cos t,dt
$$

于是：

$$
A=\int_0^{2\pi} a\cos t\cdot b\cos t,dt
$$

$$
=ab\int_0^{2\pi}\cos^2 t,dt
$$

$$
=\pi ab
$$

---

## 结论

$$
\boxed{\pi ab}
$$

---

# 五、第一类曲面积分

形式：

$$
\iint_\Sigma f(x,y,z),dS
$$

本质：

> 在曲面上按面积加权。

---

# 题7（投影公式）

当曲面可以写成

$$
z=z(x,y)
$$

并且在 $xOy$ 平面上的投影区域为 $D$ 时，第一类曲面积分可以化成二重积分：

<div>
$$
\iint_\Sigma f(x,y,z),dS 
= 
\iint_D f(x,y,z(x,y))\sqrt{1+\left(\frac{\partial z}{\partial x}\right)^2+\left(\frac{\partial z}{\partial y}\right)^2},dxdy
$$
</div>

这里的

$$
dS=\sqrt{1+z_x^2+z_y^2},dxdy
$$

就是把曲面投影到 $xOy$ 平面后得到的面积元公式。

---

## 例子

计算抛物面上半部分

$$
z=x^2+y^2,\quad z\le1
$$

上的积分：

$$
\iint_\Sigma z,dS
$$

---

## 投影到 $xOy$ 平面

因为曲面是

$$
z=x^2+y^2
$$

所以：

$$
z_x=2x,\quad z_y=2y
$$

因此：

$$
dS=\sqrt{1+4x^2+4y^2},dxdy
$$

投影区域为单位圆盘：

$$
D:\ x^2+y^2\le1
$$

于是：

<div>
$$
\iint_\Sigma z,dS
=
\iint_D (x^2+y^2)\sqrt{1+4x^2+4y^2},dxdy
$$
</div>

改为极坐标：

$$
x=r\cos\theta,\quad y=r\sin\theta
$$

则：

$$
x^2+y^2=r^2,\quad dxdy=r,drd\theta
$$

所以：

$$
\int_0^{2\pi}\int_0^1 r^2\sqrt{1+4r^2}\,r,drd\theta
$$

即：

$$
=2\pi\int_0^1 r^3\sqrt{1+4r^2},dr
$$

---

## 结论

$$
\boxed{2\pi\int_0^1 r^3\sqrt{1+4r^2},dr}
$$

---

## 对称性利用

计算球面：

$$
x^2+y^2+z^2=a^2
$$

上的积分：

$$
\iint_\Sigma z^2,dS
$$

---

## 对称性

球面对称：

$$
\iint x^2\,dS = \iint y^2\,dS = \iint z^2\,dS
$$

而：

$$
x^2+y^2+z^2=a^2
$$

所以：

$$
3\iint z^2\,dS = \iint a^2\,dS
$$

球面积：

$$
4\pi a^2
$$

故：

$$
3I=a^2\cdot4\pi a^2
$$

$$
I=\frac{4\pi a^4}{3}
$$

---

## 结论

$$
\boxed{\frac{4\pi a^4}{3}}
$$

---

# 六、第二类曲面积分（通量）

形式：

$$
\iint_\Sigma
P,dy,dz
+
Q,dz,dx
+
R,dx,dy
$$

本质：

> 向量场穿过曲面的流量。

---

# 题8（高斯公式）

计算：

$$
\iint_\Sigma
(x,dy,dz+y,dz,dx+z,dx,dy)
$$

其中 $(\Sigma)$ 为球面：

$$
x^2+y^2+z^2=a^2
$$

外侧。

---

# 使用高斯公式

高斯公式：

$$
\iint_\Sigma \vec F\cdot \vec n\,dS = \iiint_\Omega \nabla\cdot\vec F\,dV
$$

这里：

$$
\vec F=(x,y,z)
$$

散度：

$$
\nabla\cdot\vec F = 1+1+1 = 3
$$

于是：

$$
=3\iiint_\Omega dV
$$

球体积：

$$
\frac43\pi a^3
$$

故：

$$
=4\pi a^3
$$

---

## 结论

$$
\boxed{4\pi a^3}
$$

---

# 七、第二类曲面积分直接法

---

# 题9

计算：

$$
\iint_\Sigma z,dx,dy
$$

其中：

$$
z=x^2+y^2,\quad z\le1
$$

取上侧。

---

## 化为二重积分

因为：

$$
\iint_\Sigma R\,dx\,dy = \iint_D R(x,y,z)\,dx\,dy
$$

这里：

$$
R=z=x^2+y^2
$$

投影区域：

$$
x^2+y^2\le1
$$

于是：

$$
\iint_D (x^2+y^2),dxdy
$$

改极坐标：

$$
=\int_0^{2\pi}\int_0^1 r^2\cdot r,dr,d\theta
$$

$$
=2\pi\int_0^1 r^3,dr
$$

$$
=2\pi\cdot\frac14
$$

---

## 结论

$$
\boxed{\frac{\pi}{2}}
$$

---

# 八、路径相关 vs 路径无关

---

# 题10（路径相关）

计算：

$$
\int_L (-y,dx+x,dy)
$$

分别沿：

1. 直线 (y=x)
2. 抛物线 (y=x^2)

从 $((0,0)\to(1,1))$。

---

# 路径1：直线

参数：

$$
x=t,\ y=t
$$

$$
dx=dy=dt
$$

积分：

$$
\int_0^1 (-t+t),dt=0
$$

---

# 路径2：抛物线

$$
x=t,\ y=t^2
$$

$$
dx=dt,\ dy=2t,dt
$$

积分：

$$
\int_0^1 (-t^2+2t^2),dt
$$

$$
=\int_0^1 t^2,dt
$$

$$
=\frac13
$$

---

## 结果不同

因此：

> 与路径有关。

---

# 九、方法总结（最重要）

---

# 1. 第一类曲线积分

看到：

$$
\int f,ds
$$

优先想：

* 参数化
* 弧长元

---

# 2. 第二类曲线积分

看到：

$$
Pdx+Qdy
$$

优先判断：

## 是否路径无关

检查：

$$
P_y=Q_x
$$

如果成立：

> 找势函数。

---

## 是否闭合曲线

若闭合：
直接想格林公式。

---

# 3. 不是闭合怎么办

想到：

> 补线。

然后：

* 闭合
* 格林
* 减补线

---

# 4. 第二类曲面积分

看到：

$$
P,dy,dz+Q,dz,dx+R,dx,dy
$$

优先判断：

## 是否闭曲面

若闭曲面：
直接高斯公式。

---

# 5. 球面/圆面常用技巧

看到：

$$
x^2+y^2+z^2=a^2
$$

优先考虑：

* 对称性
* 均分思想

例如：

$$
\iint x^2 dS=\iint y^2 dS=\iint z^2 dS
$$

这是高频技巧。

