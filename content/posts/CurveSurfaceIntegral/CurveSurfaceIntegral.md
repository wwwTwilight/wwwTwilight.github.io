---
date: '2026-11-05T10:21:32+08:00'
draft: false
title: "曲线积分与曲面积分解法及例题"
categories: ["考研数学"]
tags: ["高等数学", "曲线积分", "曲面积分", "解题方法"]
---

# 曲线积分与曲面积分解法及例题

曲线积分与曲面积分主要分为四类：

- **第一类曲线积分**：对弧长进行积分，与曲线方向无关。
- **第二类曲线积分**：对坐标微分进行积分，与曲线方向有关。
- **第一类曲面积分**：对曲面面积进行积分，与曲面侧无关。
- **第二类曲面积分**：对有向曲面进行积分，与法向方向有关。

---

# 一、第一类曲线积分

## 1. 基本定义

第一类曲线积分又称对弧长的曲线积分，形式为：

$$
\int_L f(x,y)\,ds
$$

其中，弧长微元为：

$$
ds=\sqrt{dx^2+dy^2}
$$

若曲线由参数方程给出：

$$
\begin{cases}
x=x(t)\\
y=y(t)
\end{cases}
\qquad \alpha\leq t\leq\beta
$$

则：

$$
ds=\sqrt{[x'(t)]^2+[y'(t)]^2}\,dt
$$

因此：

$$
\int_L f(x,y)\,ds
=
\int_\alpha^\beta
f(x(t),y(t))
\sqrt{[x'(t)]^2+[y'(t)]^2}\,dt
$$

**核心特点：第一类曲线积分与曲线的方向无关。**

无论从起点走向终点，还是从终点走向起点，积分值都不变。

## 2. 常见计算方法

### 方法一：参数方程

当曲线容易写成参数方程时，优先使用参数方程。

例如，计算第一象限内圆弧的积分：

$$
\int_L(x+y)\,ds
$$

其中圆弧为：

$$
x^2+y^2=1
$$

参数化：

$$
\begin{cases}
x=\cos t\\
y=\sin t
\end{cases}
\qquad 0\leq t\leq\frac{\pi}{2}
$$

计算弧长微元：

$$
ds=\sqrt{\sin^2t+\cos^2t}\,dt=dt
$$

代入：

$$
\begin{aligned}
\int_L(x+y)\,ds
&=\int_0^{\pi/2}(\cos t+\sin t)\,dt\\
&=\left[\sin t-\cos t\right]_0^{\pi/2}\\
&=2
\end{aligned}
$$

### 方法二：直接利用曲线方程

当曲线可以写成显函数时，可以直接利用弧长微元。

若：

$$
y=y(x)
$$

则：

$$
ds=\sqrt{1+[y'(x)]^2}\,dx
$$

因此：

$$
\int_L f(x,y)\,ds
=
\int_a^b f(x,y(x))
\sqrt{1+[y'(x)]^2}\,dx
$$

这种方法适合曲线容易表示成显函数的情况。

## 3. 对称性

若积分曲线关于原点对称，且被积函数满足：

$$
f(-x,-y)=-f(x,y)
$$

则：

$$
\int_L f(x,y)\,ds=0
$$

原因是：对称点处的弧长微元相同，而被积函数的值互为相反数。

---

# 二、第二类曲线积分

## 1. 基本定义

第二类曲线积分的形式为：

$$
\int_L P(x,y)\,dx+Q(x,y)\,dy
$$

它可以理解为沿着曲线对向量场进行积分。

若曲线由参数方程给出：

$$
\begin{cases}
x=x(t)\\
y=y(t)
\end{cases}
\qquad \alpha\leq t\leq\beta
$$

则：

$$
dx=x'(t)\,dt
$$

$$
dy=y'(t)\,dt
$$

代入可得：

$$
\int_L P\,dx+Q\,dy
=
\int_\alpha^\beta
\left[
P(x(t),y(t))x'(t)
+
Q(x(t),y(t))y'(t)
\right]dt
$$

**核心特点：第二类曲线积分与曲线的方向有关。**

如果将积分方向反过来，则：

$$
\int_{-L}P\,dx+Q\,dy
=
-\int_LP\,dx+Q\,dy
$$

## 2. 参数方程法

例如，计算：

$$
\int_L x\,dy-y\,dx
$$

其中，曲线为单位圆的上半圆，从左端点走向右端点。

参数方程：

$$
\begin{cases}
x=\cos t\\
y=\sin t
\end{cases}
\qquad \pi\geq t\geq 0
$$

注意，这里的参数从较大的值变化到较小的值，表示从左端点走向右端点。

有：

$$
dx=-\sin t\,dt
$$

$$
dy=\cos t\,dt
$$

代入：

$$
\begin{aligned}
\int_Lx\,dy-y\,dx
&=\int_\pi^0
(\cos^2t+\sin^2t)\,dt\\
&=\int_\pi^0 1\,dt\\
&=-\pi
\end{aligned}
$$

也可以采用其他参数区间，但必须保证参数变化方向与题目要求的曲线方向一致。

## 3. 利用曲线方程消去一个变量

如果曲线容易表示成：

$$
y=y(x)
$$

则：

$$
dy=y'(x)\,dx
$$

原积分可以化为：

$$
\int_LP\,dx+Q\,dy
=
\int_a^b
\left[P(x,y(x))+Q(x,y(x))y'(x)\right]dx
$$

这种方法适用于曲线可以方便地表示为显函数的情况。

---

# 三、第二类曲线积分的路径无关性

这是第二类曲线积分的重要考点。

## 1. 什么是路径无关？

如果曲线积分只与起点和终点有关，而与具体经过的路径无关，则称该积分与路径无关。

例如，从点 A 到点 B，可以经过不同的曲线：

$$
\int_{L_1}P\,dx+Q\,dy
=
\int_{L_2}P\,dx+Q\,dy
$$

只要起点和终点相同，积分值就相同。

## 2. 全微分与原函数

如果存在函数：

$$
u=u(x,y)
$$

使得：

$$
du=P(x,y)\,dx+Q(x,y)\,dy
$$

那么：

$$
\int_LP\,dx+Q\,dy=u(B)-u(A)
$$

也就是说，只需要求出原函数，再代入终点和起点即可。

## 3. 如何判断是否为全微分？

若：

$$
P_y=Q_x
$$

则在适当的区域条件下，可以判断该微分形式为全微分。

其中：

$$
P_y=\frac{\partial P}{\partial y}
$$

$$
Q_x=\frac{\partial Q}{\partial x}
$$

**注意：仅仅满足偏导数相等还不够，通常还需要积分区域是单连通区域，并且相关偏导数连续。**

在考研题目中，如果定义域是整个平面、矩形、圆形内部等常见单连通区域，通常可以直接使用这一判定。

## 4. 如何求原函数？

假设：

$$
du=P\,dx+Q\,dy
$$

那么：

$$
u_x=P
$$

$$
u_y=Q
$$

首先对 P 关于 x 积分：

$$
u(x,y)=\int P(x,y)\,dx+\varphi(y)
$$

其中，积分时将 y 看作常数，且需要额外添加一个只与 y 有关的函数。

然后对上式关于 y 求偏导：

$$
u_y=
\frac{\partial}{\partial y}
\left[\int P(x,y)\,dx\right]
+\varphi'(y)
$$

再令其等于 Q：

$$
\frac{\partial}{\partial y}
\left[\int P(x,y)\,dx\right]
+\varphi'(y)
=Q(x,y)
$$

由此求出：

$$
\varphi'(y)
$$

最后积分得到：

$$
\varphi(y)
$$

即可求出原函数。

### 例题

计算：

$$
\int_L 2xy\,dx+(x^2+2y)\,dy
$$

其中，曲线从点 A 走到点 B：

$$
A=(0,0),\qquad B=(1,1)
$$

首先：

$$
P=2xy,\qquad Q=x^2+2y
$$

判断：

$$
P_y=2x
$$

$$
Q_x=2x
$$

因此，在整个平面内该微分形式为全微分。

求原函数：

$$
u=\int 2xy\,dx=x^2y+\varphi(y)
$$

对 y 求偏导：

$$
u_y=x^2+\varphi'(y)
$$

令其等于 Q：

$$
x^2+\varphi'(y)=x^2+2y
$$

所以：

$$
\varphi'(y)=2y
$$

得到：

$$
\varphi(y)=y^2+C
$$

因此：

$$
u=x^2y+y^2
$$

最终：

$$
\begin{aligned}
\int_L2xy\,dx+(x^2+2y)\,dy
&=u(1,1)-u(0,0)\\
&=(1+1)-0\\
&=2
\end{aligned}
$$

---

# 四、格林公式

格林公式是第二类曲线积分最重要的工具之一。

## 1. 基本公式

设平面闭区域 D 的正向边界为 L，则：

$$
\oint_LP\,dx+Q\,dy
=
\iint_D
\left(
\frac{\partial Q}{\partial x}
-
\frac{\partial P}{\partial y}
\right)dx\,dy
$$

其中，正向边界通常指逆时针方向。

也就是说：

- 沿着边界行走时，区域始终在左手边。
- 逆时针方向为正向。
- 顺时针方向为负向。

如果边界方向是顺时针，则：

$$
\oint_LP\,dx+Q\,dy
=
-\iint_D(Q_x-P_y)\,dx\,dy
$$

## 2. 格林公式的使用条件

通常需要满足：

1. 曲线 L 是分段光滑的简单闭曲线。
2. 区域 D 是由 L 围成的平面区域。
3. P、Q 在区域内及边界上具有适当的连续偏导数。

如果区域存在孔洞，则需要正确处理内边界的方向，不能直接忽略。

## 3. 典型例题：利用格林公式计算圆周积分

计算：

$$
\oint_L-y\,dx+x\,dy
$$

其中，L 是单位圆，方向为逆时针。

令：

$$
P=-y,\qquad Q=x
$$

则：

$$
Q_x=1
$$

$$
P_y=-1
$$

因此：

$$
Q_x-P_y=2
$$

根据格林公式：

$$
\begin{aligned}
\oint_L-y\,dx+x\,dy
&=\iint_D2\,dx\,dy\\
&=2\pi
\end{aligned}
$$

这里利用了单位圆的面积：

$$
S_D=\pi
$$

所以积分结果为：

$$
2\pi
$$

如果方向改为顺时针，则结果变为：

$$
-2\pi
$$

## 4. 非闭曲线如何使用格林公式？

格林公式要求积分路径是闭曲线。

如果题目给出的曲线不是闭合的，可以考虑添加一段辅助曲线，使其构成闭合曲线。

例如，单位圆上半圆弧从左端点走向右端点，再沿 x 轴从右端点返回左端点。

此时整个闭合曲线的方向为顺时针。

因此：

$$
\oint_LP\,dx+Q\,dy
=
-\iint_D(Q_x-P_y)\,dx\,dy
$$

如果辅助线上的积分容易计算，就可以通过：

$$
\int_{\text{原曲线}}
=
\int_{\text{闭合曲线}}
-
\int_{\text{辅助曲线}}
$$

求出原曲线上的积分。

**注意：添加辅助线后，一定要先判断闭合曲线的方向，不能默认就是逆时针。**

## 5. 利用格林公式求平面区域面积

格林公式还可以用于计算平面区域面积。

令：

$$
P=-\frac{y}{2},\qquad Q=\frac{x}{2}
$$

则：

$$
Q_x-P_y=1
$$

因此：

$$
S_D=\iint_D1\,dx\,dy
$$

可以转化为：

$$
S_D=
\frac{1}{2}
\oint_L(x\,dy-y\,dx)
$$

也可以选择其他形式：

$$
S_D=\oint_Lx\,dy
$$

或者：

$$
S_D=-\oint_Ly\,dx
$$

以上公式均要求边界为逆时针正向。如果边界方向相反，则需要改变符号。

---

# 五、第一类曲面积分

## 1. 基本定义

第一类曲面积分是对曲面面积进行积分：

$$
\iint_\Sigma f(x,y,z)\,dS
$$

其中，面积微元为 dS。

第一类曲面积分与曲面的侧无关。无论选择曲面的上侧还是下侧，积分值都不会改变。

## 2. 曲面投影到 xOy 平面

这是最常见的计算方法。

如果曲面可以表示为：

$$
z=z(x,y)
$$

则面积微元为：

$$
dS=\sqrt{1+z_x^2+z_y^2}\,dx\,dy
$$

其中：

$$
z_x=\frac{\partial z}{\partial x}
$$

$$
z_y=\frac{\partial z}{\partial y}
$$

因此：

$$
\iint_\Sigma f(x,y,z)\,dS
=
\iint_D
f(x,y,z(x,y))
\sqrt{1+z_x^2+z_y^2}\,dx\,dy
$$

这里的 D 是曲面在 xOy 平面上的投影区域。

### 例题：抛物面上的第一类曲面积分

计算：

$$
\iint_\Sigma z\,dS
$$

其中曲面为：

$$
z=x^2+y^2
$$

且：

$$
0\leq z\leq 1
$$

首先求偏导：

$$
z_x=2x
$$

$$
z_y=2y
$$

因此：

$$
dS=\sqrt{1+4x^2+4y^2}\,dx\,dy
$$

将曲面投影到 xOy 平面，得到：

$$
x^2+y^2\leq 1
$$

原积分化为：

$$
\iint_D
(x^2+y^2)
\sqrt{1+4x^2+4y^2}\,dx\,dy
$$

由于积分区域是圆形，适合使用极坐标：

$$
\begin{cases}
x=r\cos\theta\\
y=r\sin\theta
\end{cases}
$$

$$
dx\,dy=r\,dr\,d\theta
$$

因此：

$$
\begin{aligned}
I
&=\int_0^{2\pi}\int_0^1
r^2\sqrt{1+4r^2}\cdot r\,dr\,d\theta\\
&=2\pi\int_0^1r^3\sqrt{1+4r^2}\,dr
\end{aligned}
$$

令：

$$
u=1+4r^2
$$

则：

$$
r^3\,dr=\frac{u-1}{32}\,du
$$

因此：

$$
\begin{aligned}
I
&=\frac{\pi}{16}
\int_1^5(u-1)\sqrt{u}\,du\\
&=\frac{\pi}{16}
\left[
\frac{2}{5}u^{5/2}
-\frac{2}{3}u^{3/2}
\right]_1^5\\
&=\frac{\pi(25\sqrt{5}+1)}{60}
\end{aligned}
$$

## 3. 曲面投影到其他坐标平面

如果曲面不方便表示为：

$$
z=z(x,y)
$$

也可以选择其他投影平面。

若曲面表示为：

$$
x=x(y,z)
$$

则：

$$
dS=\sqrt{1+x_y^2+x_z^2}\,dy\,dz
$$

若曲面表示为：

$$
y=y(x,z)
$$

则：

$$
dS=\sqrt{1+y_x^2+y_z^2}\,dx\,dz
$$

**选择投影平面的原则：尽量选择投影区域简单、曲面方程容易处理的坐标平面。**

## 4. 利用对称性简化

对于球面、圆柱面、旋转曲面等具有明显对称性的曲面，可以优先考虑对称性。

例如，球面：

$$
x^2+y^2+z^2=a^2
$$

计算：

$$
\iint_\Sigma z^2\,dS
$$

由于球面对三个坐标轴完全对称，因此：

$$
\iint_\Sigma x^2\,dS
=
\iint_\Sigma y^2\,dS
=
\iint_\Sigma z^2\,dS
$$

将三式相加：

$$
3\iint_\Sigma z^2\,dS
=
\iint_\Sigma(x^2+y^2+z^2)\,dS
$$

而球面上：

$$
x^2+y^2+z^2=a^2
$$

因此：

$$
3\iint_\Sigma z^2\,dS
=
a^2\iint_\Sigma dS
$$

球面面积为：

$$
4\pi a^2
$$

最终：

$$
\iint_\Sigma z^2\,dS
=
\frac{4\pi a^4}{3}
$$

这种方法比直接计算曲面积分简单得多。

---

# 六、第二类曲面积分

第二类曲面积分是曲面积分中最需要注意方向的一类，也是最容易因为符号问题出错的部分。

## 1. 基本定义

第二类曲面积分的形式为：

$$
\iint_\Sigma P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
$$

它可以理解为向量场：

$$
\mathbf F=(P,Q,R)
$$

通过有向曲面的通量。

如果曲面的单位法向量为：

$$
\mathbf n=(\cos\alpha,\cos\beta,\cos\gamma)
$$

则：

$$
\iint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iint_\Sigma
\mathbf F\cdot\mathbf n\,dS
$$

**核心特点：第二类曲面积分与曲面的法向方向有关。**

如果将曲面的法向方向反过来，则积分变号。

## 2. 最重要的公式：投影到 xOy 平面

假设曲面可以表示为：

$$
z=z(x,y)
$$

将曲面参数化：

$$
\mathbf r(x,y)=(x,y,z(x,y))
$$

分别对 x、y 求偏导：

$$
\mathbf r_x=(1,0,z_x)
$$

$$
\mathbf r_y=(0,1,z_y)
$$

计算叉乘：

$$
\mathbf r_x\times\mathbf r_y
=
(-z_x,-z_y,1)
$$

这个向量的 z 分量为正，因此对应法向量朝向 z 轴正方向的曲面侧，通常称为上侧。

反过来：

$$
\mathbf r_y\times\mathbf r_x
=
(z_x,z_y,-1)
$$

对应法向量朝向 z 轴负方向的曲面侧，通常称为下侧。

### 上侧公式

当曲面取上侧时：

$$
\boxed{
\iint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iint_D
(-Pz_x-Qz_y+R)\,dx\,dy
}
$$

### 下侧公式

当曲面取下侧时：

$$
\boxed{
\iint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iint_D
(Pz_x+Qz_y-R)\,dx\,dy
}
$$

这里的 D 是曲面在 xOy 平面上的投影区域。

**特别注意：**

- 上侧对应法向量 z 分量为正。
- 下侧对应法向量 z 分量为负。
- 上侧公式与下侧公式正好互为相反数。
- 不要把第一类曲面积分的面积微元公式直接套到第二类曲面积分中。

## 3. 为什么上侧公式是这个形式？

从参数化出发：

$$
\mathbf r_x\times\mathbf r_y
=
(-z_x,-z_y,1)
$$

第二类曲面积分可以写成：

$$
\iint_D
\mathbf F\cdot
(\mathbf r_x\times\mathbf r_y)
\,dx\,dy
$$

将向量场代入：

$$
\mathbf F=(P,Q,R)
$$

得到：

$$
\begin{aligned}
\mathbf F\cdot
(\mathbf r_x\times\mathbf r_y)
&=(P,Q,R)\cdot(-z_x,-z_y,1)\\
&=-Pz_x-Qz_y+R
\end{aligned}
$$

因此：

$$
\iint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iint_D(-Pz_x-Qz_y+R)\,dx\,dy
$$

这就是上侧公式的来源。

理解了叉乘，就不需要死记符号。

## 4. 典型例题：抛物面上的第二类曲面积分

计算：

$$
\iint_\Sigma
x\,dy\,dz+y\,dz\,dx+z\,dx\,dy
$$

其中曲面为：

$$
z=x^2+y^2
$$

且：

$$
z\leq 1
$$

曲面取上侧。

首先确定：

$$
P=x,\qquad Q=y,\qquad R=z
$$

求偏导：

$$
z_x=2x
$$

$$
z_y=2y
$$

根据上侧公式：

$$
\begin{aligned}
I
&=\iint_D(-Pz_x-Qz_y+R)\,dx\,dy\\
&=\iint_D(-2x^2-2y^2+x^2+y^2)\,dx\,dy\\
&=\iint_D-(x^2+y^2)\,dx\,dy
\end{aligned}
$$

投影区域为：

$$
x^2+y^2\leq 1
$$

使用极坐标：

$$
\begin{cases}
x=r\cos\theta\\
y=r\sin\theta
\end{cases}
$$

得到：

$$
\begin{aligned}
I
&=-\int_0^{2\pi}\int_0^1r^2\cdot r\,dr\,d\theta\\
&=-2\pi\int_0^1r^3\,dr\\
&=-\frac{\pi}{2}
\end{aligned}
$$

如果曲面改为下侧，则结果为：

$$
\frac{\pi}{2}
$$

## 5. 投影到 yOz 平面

如果曲面可以表示为：

$$
x=x(y,z)
$$

则可以投影到 yOz 平面。

参数化：

$$
\mathbf r(y,z)=(x(y,z),y,z)
$$

计算偏导：

$$
\mathbf r_y=(x_y,1,0)
$$

$$
\mathbf r_z=(x_z,0,1)
$$

叉乘：

$$
\mathbf r_y\times\mathbf r_z
=
(1,-x_y,-x_z)
$$

其 x 分量为正，对应法向量朝向 x 轴正方向的曲面侧。

### x 轴正方向侧

$$
\boxed{
\iint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iint_D(P-Qx_y-Rx_z)\,dy\,dz
}
$$

### x 轴负方向侧

$$
\boxed{
\iint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iint_D(-P+Qx_y+Rx_z)\,dy\,dz
}
$$

## 6. 投影到 xOz 平面

如果曲面可以表示为：

$$
y=y(x,z)
$$

则可以投影到 xOz 平面。

参数化：

$$
\mathbf r(x,z)=(x,y(x,z),z)
$$

计算偏导：

$$
\mathbf r_x=(1,y_x,0)
$$

$$
\mathbf r_z=(0,y_z,1)
$$

叉乘：

$$
\mathbf r_z\times\mathbf r_x
=
(-y_x,1,-y_z)
$$

其 y 分量为正，对应法向量朝向 y 轴正方向的曲面侧。

### y 轴正方向侧

$$
\boxed{
\iint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iint_D(-Py_x+Q-Ry_z)\,dx\,dz
}
$$

### y 轴负方向侧

$$
\boxed{
\iint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iint_D(Py_x-Q+Ry_z)\,dx\,dz
}
$$

## 7. 三种投影方式总结

实际做题时，首先判断曲面能够方便地表示成哪一个变量关于另外两个变量的函数。

**投影到 xOy 平面**

曲面方程：

$$
z=z(x,y)
$$

法向量 z 分量为正时：

$$
(-z_x,-z_y,1)
$$

积分表达式：

$$
-Pz_x-Qz_y+R
$$

**投影到 yOz 平面**

曲面方程：

$$
x=x(y,z)
$$

法向量 x 分量为正时：

$$
(1,-x_y,-x_z)
$$

积分表达式：

$$
P-Qx_y-Rx_z
$$

**投影到 xOz 平面**

曲面方程：

$$
y=y(x,z)
$$

法向量 y 分量为正时：

$$
(-y_x,1,-y_z)
$$

积分表达式：

$$
-Py_x+Q-Ry_z
$$

**记忆建议：不要单纯背三个公式。先写参数方程，再计算两个切向量的叉乘，最后与向量场做点乘。这样不容易弄错符号。**

---

# 七、高斯公式

高斯公式是第二类曲面积分的重要计算工具。

## 1. 基本公式

设空间闭区域 Ω 的边界曲面为 Σ，取外侧为正方向，则：

$$
\boxed{
\oiint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iiint_\Omega
\left(
\frac{\partial P}{\partial x}
+
\frac{\partial Q}{\partial y}
+
\frac{\partial R}{\partial z}
\right)dx\,dy\,dz
}
$$

其中，Σ 必须是闭曲面，且法向量取外法向。

高斯公式的核心是：

**将闭曲面上的第二类曲面积分转化为其内部空间区域上的三重积分。**

## 2. 典型例题

计算：

$$
\oiint_\Sigma
x\,dy\,dz+y\,dz\,dx+z\,dx\,dy
$$

其中，Σ 是球面：

$$
x^2+y^2+z^2=a^2
$$

取外侧。

由高斯公式：

$$
P=x,\qquad Q=y,\qquad R=z
$$

因此：

$$
P_x=1
$$

$$
Q_y=1
$$

$$
R_z=1
$$

所以：

$$
\begin{aligned}
I
&=\iiint_\Omega(1+1+1)\,dx\,dy\,dz\\
&=3V_\Omega
\end{aligned}
$$

球体体积为：

$$
V_\Omega=\frac{4}{3}\pi a^3
$$

因此：

$$
I=4\pi a^3
$$

## 3. 非闭曲面如何使用高斯公式？

高斯公式要求曲面是闭曲面。

如果题目给出的是一个开放曲面，可以考虑添加辅助曲面，将其补成闭曲面。

设原曲面为 Σ，添加的辅助曲面为 Σ₁，则二者共同构成闭曲面。

于是：

$$
\iint_\Sigma
=
\oiint_{\Sigma+\Sigma_1}
-
\iint_{\Sigma_1}
$$

具体计算时：

1. 先选择一个容易计算的辅助曲面。
2. 将原曲面与辅助曲面组成闭曲面。
3. 对闭曲面使用高斯公式。
4. 单独计算辅助曲面的积分。
5. 两者相减得到原曲面的积分。

**特别注意：补面后，原曲面与辅助曲面的法向方向必须共同构成闭曲面的外法向方向。**

例如，向上开口的曲面，如果通过底面补成闭曲面，那么底面作为辅助曲面时，外法向通常朝下。

不能因为原曲面取上侧，就默认辅助底面也取上侧。

---

# 八、曲线积分与曲面积分的对称性

对称性是考研数学中非常实用的简化技巧。

## 1. 第一类曲线积分

对于关于原点对称的曲线，如果被积函数满足：

$$
f(-x,-y)=-f(x,y)
$$

则：

$$
\int_Lf(x,y)\,ds=0
$$

因为弧长微元在对称点处相同，而函数值相反。

## 2. 第一类曲面积分

对于关于某个坐标平面对称的曲面，如果被积函数关于该对称变换为奇函数，则积分为零。

例如，球面关于 xOy 平面对称，若被积函数为 z，则：

$$
\iint_\Sigma z\,dS=0
$$

因为对称点的 z 坐标互为相反数，而面积微元相同。

## 3. 利用对称性拆分积分

对于球面：

$$
x^2+y^2+z^2=a^2
$$

有：

$$
\iint_\Sigma x^2\,dS
=
\iint_\Sigma y^2\,dS
=
\iint_\Sigma z^2\,dS
$$

因此：

$$
\iint_\Sigma(x^2+y^2+z^2)\,dS
=
3\iint_\Sigma z^2\,dS
$$

这种方法可以将复杂的积分转化为简单的面积积分。

## 4. 第二类曲面积分的对称性需要格外注意

第二类曲面积分不仅取决于被积函数，还取决于曲面的法向方向。

因此，不能直接照搬第一类曲面积分的对称性结论。

例如，曲面关于某个坐标平面对称，并不意味着第二类曲面积分一定为零。

必须同时考虑：

- 对称点处向量场的变化；
- 对称点处法向量的变化；
- 曲面方向是否保持一致。

最稳妥的方法是先写出投影公式，再判断投影区域上的被积函数是否为奇函数。

---

# 九、解题方法选择

曲线积分与曲面积分的计算，关键不是一味地套公式，而是根据题目特点选择最合适的方法。

## 1. 第一类曲线积分

优先考虑：

- 参数方程；
- 直接利用曲线方程；
- 对称性简化。

核心公式：

$$
\int_Lf(x,y)\,ds
=
\int_\alpha^\beta
f(x(t),y(t))
\sqrt{[x'(t)]^2+[y'(t)]^2}\,dt
$$

## 2. 第二类曲线积分

优先考虑：

- 参数方程；
- 判断是否为全微分；
- 判断是否满足格林公式；
- 非闭曲线考虑添加辅助线。

全微分判定的常用条件：

$$
P_y=Q_x
$$

全微分计算：

$$
\int_LP\,dx+Q\,dy=u(B)-u(A)
$$

格林公式：

$$
\oint_LP\,dx+Q\,dy
=
\iint_D(Q_x-P_y)\,dx\,dy
$$

## 3. 第一类曲面积分

优先考虑：

- 曲面投影；
- 极坐标；
- 对称性。

投影到 xOy 平面时：

$$
dS=\sqrt{1+z_x^2+z_y^2}\,dx\,dy
$$

## 4. 第二类曲面积分

优先考虑：

- 直接投影到最简单的坐标平面；
- 判断曲面侧；
- 判断是否可以补成闭曲面后使用高斯公式。

投影到 xOy 平面，上侧：

$$
\iint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iint_D(-Pz_x-Qz_y+R)\,dx\,dy
$$

高斯公式：

$$
\oiint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iiint_\Omega(P_x+Q_y+R_z)\,dx\,dy\,dz
$$

---

# 十、核心公式汇总

## 1. 第一类曲线积分

$$
\int_Lf(x,y)\,ds
$$

参数方程：

$$
ds=\sqrt{[x'(t)]^2+[y'(t)]^2}\,dt
$$

## 2. 第二类曲线积分

$$
\int_LP\,dx+Q\,dy
$$

路径无关的常用判定：

$$
P_y=Q_x
$$

全微分计算：

$$
\int_Ldu=u(B)-u(A)
$$

格林公式：

$$
\oint_LP\,dx+Q\,dy
=
\iint_D(Q_x-P_y)\,dx\,dy
$$

## 3. 第一类曲面积分

$$
\iint_\Sigma f(x,y,z)\,dS
$$

投影到 xOy 平面：

$$
dS=\sqrt{1+z_x^2+z_y^2}\,dx\,dy
$$

## 4. 第二类曲面积分

$$
\iint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
$$

投影到 xOy 平面，上侧：

$$
\iint_D(-Pz_x-Qz_y+R)\,dx\,dy
$$

投影到 xOy 平面，下侧：

$$
\iint_D(Pz_x+Qz_y-R)\,dx\,dy
$$

## 5. 高斯公式

$$
\oiint_\Sigma
P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy
=
\iiint_\Omega(P_x+Q_y+R_z)\,dx\,dy\,dz
$$

---

# 十一、最容易出错的地方

1. **混淆第一类与第二类积分。** 第一类对弧长或面积积分，与方向无关；第二类对坐标微分积分，与方向有关。

2. **第二类曲线积分忘记方向。** 参数区间的变化方向必须与题目给出的曲线方向一致。

3. **全微分判定忽略区域条件。** 偏导数相等通常还需要单连通区域等条件。

4. **格林公式忘记判断正向。** 逆时针为正向，顺时针需要增加负号。

5. **第一类曲面积分漏掉面积微元中的根号。** 投影后不能直接把面积微元当作平面面积微元。

6. **第二类曲面积分混淆上侧与下侧。** 投影到 xOy 平面时，上侧对应法向量的 z 分量为正，下侧对应 z 分量为负。

7. **高斯公式没有补成闭曲面。** 开放曲面不能直接使用高斯公式，需要先添加辅助曲面。

8. **补面时法向量方向错误。** 闭曲面使用外法向，辅助曲面的方向必须与原曲面共同构成外法向。

**最终记忆：**

- 第一类曲线积分：弧长微元。
- 第二类曲线积分：方向、全微分、格林公式。
- 第一类曲面积分：面积微元。
- 第二类曲面积分：法向量、直接投影、高斯公式。

尤其是第二类曲面积分，建议真正理解参数化之后的叉乘，而不是单纯背诵投影公式。这是避免符号错误最可靠的方法。