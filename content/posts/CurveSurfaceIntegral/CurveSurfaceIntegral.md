---
date: '2026-11-02'
draft: false
title: '曲线积分与曲面积分解题方法总结及典型例题'
categories: ["学习"]
tags: ["高等数学", "曲线积分", "曲面积分", "考研数学一"]
---

# 曲线积分与曲面积分解题方法总结及典型例题

本文按照解题方法而非教材章节进行分类，重点训练：

- 看到积分形式，能够快速判断积分类型。
- 能够根据曲线或曲面方程选择参数化、投影等方法。
- 能够正确判断积分方向和法向量方向。
- 能够灵活使用格林公式、高斯公式、补线法、对称性等技巧。
- 掌握考研数学一中曲线积分与曲面积分的典型题型。

主要内容：

1. 第一类曲线积分：弧长型积分
2. 第二类曲线积分：对坐标的曲线积分
3. 路径无关与全微分
4. 格林公式与补线法
5. 第一类曲面积分：对面积的曲面积分
6. 第二类曲面积分：对坐标的曲面积分（重点）
7. 高斯公式
8. 对称性与其他常用技巧
9. 解题方法总结

---

# 一、第一类曲线积分（弧长型）

## 1. 基本形式

\[
\int_L f(x,y)\,\mathrm ds
\]

或者在空间曲线中：

\[
\int_L f(x,y,z)\,\mathrm ds
\]

本质：沿着曲线，对函数值按照弧长进行加权积分。

**核心特点：第一类曲线积分与曲线的方向无关。**

无论沿曲线从 A 到 B，还是从 B 到 A，积分结果都相同。

## 2. 参数化计算

设曲线参数方程为：

\[
\begin{cases}
x=x(t)\\
y=y(t)
\end{cases}
\qquad \alpha\leq t\leq\beta
\]

则：

\[
\mathrm ds=
\sqrt{(x'(t))^2+(y'(t))^2}\,\mathrm dt
\]

因此：

\[
\boxed{
\int_L f(x,y)\,\mathrm ds
=
\int_\alpha^\beta
f(x(t),y(t))
\sqrt{(x')^2+(y')^2}\,\mathrm dt
}
\]

空间曲线只需要增加一个坐标：

\[
\mathrm ds=
\sqrt{(x')^2+(y')^2+(z')^2}\,\mathrm dt
\]

## 3. 典型例题：圆周上的第一类曲线积分

计算：

\[
\int_L (x+y)\,\mathrm ds
\]

其中 \(L\) 为单位圆：

\[
x^2+y^2=1
\]

**解：**

参数化：

\[
x=\cos t,\quad y=\sin t
\]

\[
0\leq t\leq 2\pi
\]

计算弧长元：

\[
\mathrm ds=
\sqrt{\sin^2t+\cos^2t}\,\mathrm dt
=\mathrm dt
\]

代入：

\[
\begin{aligned}
I
&=\int_0^{2\pi}(\cos t+\sin t)\,\mathrm dt\\
&=[\sin t-\cos t]_0^{2\pi}\\
&=0
\end{aligned}
\]

答案：

\[
\boxed{I=0}
\]

## 4. 典型例题：对称性

计算：

\[
\int_L x^2\,\mathrm ds
\]

其中 \(L\) 为半径为 \(a\) 的圆周。

**解：**

参数化：

\[
x=a\cos t,\quad y=a\sin t
\]

\[
\mathrm ds=a\,\mathrm dt
\]

于是：

\[
\begin{aligned}
I
&=\int_0^{2\pi}a^2\cos^2t\cdot a\,\mathrm dt\\
&=a^3\int_0^{2\pi}\cos^2t\,\mathrm dt\\
&=\pi a^3
\end{aligned}
\]

答案：

\[
\boxed{I=\pi a^3}
\]

**方法总结：**

第一类曲线积分优先考虑：

- 参数化曲线。
- 求弧长元 \(\mathrm ds\)。
- 代入一元定积分。
- 如果曲线和被积函数具有对称性，可以利用对称性简化计算。

---

# 二、第二类曲线积分（对坐标的曲线积分）

## 1. 基本形式

平面曲线：

\[
\boxed{
\int_L P(x,y)\,\mathrm dx+Q(x,y)\,\mathrm dy
}
\]

空间曲线：

\[
\boxed{
\int_L P\,\mathrm dx+Q\,\mathrm dy+R\,\mathrm dz
}
\]

本质：沿着有方向的曲线，对向量场进行线积分。

与第一类曲线积分最重要的区别：

**第二类曲线积分与积分路径的方向有关。**

如果将积分方向反过来：

\[
\boxed{
\int_{-L}P\,\mathrm dx+Q\,\mathrm dy
=
-\int_LP\,\mathrm dx+Q\,\mathrm dy
}
\]

## 2. 参数化直接计算

设：

\[
x=x(t),\quad y=y(t)
\]

则：

\[
\mathrm dx=x'(t)\,\mathrm dt
\]

\[
\mathrm dy=y'(t)\,\mathrm dt
\]

所以：

\[
\boxed{
\int_LP\,\mathrm dx+Q\,\mathrm dy
=
\int_\alpha^\beta
\left(Px'(t)+Qy'(t)\right)\mathrm dt
}
\]

其中 \(P,Q\) 都需要代入参数方程。

## 3. 典型例题：直接参数法

计算：

\[
\int_L y\,\mathrm dx+x\,\mathrm dy
\]

其中：

\[
L:y=x^2,\quad (0,0)\rightarrow(1,1)
\]

**解：**

参数化：

\[
x=t,\quad y=t^2,\quad 0\leq t\leq1
\]

得到：

\[
\mathrm dx=\mathrm dt,\qquad
\mathrm dy=2t\,\mathrm dt
\]

代入：

\[
\begin{aligned}
I
&=\int_0^1(t^2+2t^2)\,\mathrm dt\\
&=\int_0^1 3t^2\,\mathrm dt\\
&=1
\end{aligned}
\]

答案：

\[
\boxed{I=1}
\]

---

# 三、路径无关与全微分

这是第二类曲线积分中非常重要的技巧。

## 1. 什么是路径无关？

如果对于连接相同两个端点的任意两条曲线，积分结果都相同，那么积分与路径无关。

此时：

\[
\int_LP\,\mathrm dx+Q\,\mathrm dy
\]

只与起点和终点有关。

如果存在一个二元函数 \(\varphi(x,y)\)，使得：

\[
\boxed{
\begin{cases}
\varphi_x=P\\
\varphi_y=Q
\end{cases}
}
\]

那么：

\[
P\,\mathrm dx+Q\,\mathrm dy
=\mathrm d\varphi
\]

积分直接变为：

\[
\boxed{
\int_LP\,\mathrm dx+Q\,\mathrm dy
=
\varphi(B)-\varphi(A)
}
\]

这种情况称为全微分形式。

## 2. 如何判断路径无关？

对于单连通区域内具有连续一阶偏导数的 \(P,Q\)，如果：

\[
\boxed{
\frac{\partial P}{\partial y}
=
\frac{\partial Q}{\partial x}
}
\]

那么该曲线积分与路径无关。

注意：这里的单连通条件很重要。如果积分区域存在孔洞，仅有偏导数相等并不足以保证全局路径无关。

## 3. 典型例题：求势函数

计算：

\[
\int_L(2xy+y^2)\,\mathrm dx
+(x^2+2xy)\,\mathrm dy
\]

其中 \(L\) 为任意连接：

\[
(0,0)\rightarrow(1,1)
\]

的光滑曲线。

**第一步：判断路径无关**

设：

\[
P=2xy+y^2
\]

\[
Q=x^2+2xy
\]

计算：

\[
P_y=2x+2y
\]

\[
Q_x=2x+2y
\]

由于：

\[
P_y=Q_x
\]

且积分区域为整个平面，因此积分与路径无关。

**第二步：求势函数**

设：

\[
\varphi_x=2xy+y^2
\]

对 \(x\) 积分：

\[
\varphi=x^2y+xy^2+C(y)
\]

再对 \(y\) 求偏导：

\[
\varphi_y=x^2+2xy+C'(y)
\]

与 \(Q\) 比较：

\[
C'(y)=0
\]

因此：

\[
\varphi=x^2y+xy^2
\]

**第三步：代入端点**

\[
\begin{aligned}
I
&=\varphi(1,1)-\varphi(0,0)\\
&=2-0\\
&=2
\end{aligned}
\]

答案：

\[
\boxed{I=2}
\]

**方法总结：**

看到第二类曲线积分，首先考虑：

1. 能否直接参数化？
2. 是否满足路径无关条件？
3. 是否能够找到势函数？

如果能够找到势函数，通常不需要计算具体路径。

---

# 四、格林公式与补线法

## 1. 格林公式

对于平面正向（逆时针）闭曲线 \(L\)，以及它所围成的区域 \(D\)，有：

\[
\boxed{
\oint_LP\,\mathrm dx+Q\,\mathrm dy
=
\iint_D
\left(
\frac{\partial Q}{\partial x}
-
\frac{\partial P}{\partial y}
\right)\mathrm dA
}
\]

这里最重要的是：

**正向闭曲线指逆时针方向，区域始终位于运动方向的左侧。**

如果曲线方向为顺时针，则：

\[
\oint_LP\,\mathrm dx+Q\,\mathrm dy
=
-\iint_D(Q_x-P_y)\,\mathrm dA
\]

## 2. 典型例题：直接使用格林公式

计算：

\[
\oint_L(-y\,\mathrm dx+x\,\mathrm dy)
\]

其中 \(L\) 为单位圆，方向为逆时针。

**解：**

设：

\[
P=-y,\quad Q=x
\]

则：

\[
Q_x=1,\qquad P_y=-1
\]

所以：

\[
Q_x-P_y=2
\]

利用格林公式：

\[
\begin{aligned}
I
&=\iint_D2\,\mathrm dA\\
&=2\pi
\end{aligned}
\]

答案：

\[
\boxed{I=2\pi}
\]

## 3. 补线法

如果积分曲线不是闭合曲线，但能够通过补充一段曲线构成闭合曲线，就可以使用格林公式。

核心关系：

\[
\boxed{
\int_L+\int_C
=
\oint_{L+C}
}
\]

因此：

\[
\boxed{
\int_L
=
\oint_{L+C}-\int_C
}
\]

注意：补线方向必须保证构成的闭曲线方向正确。

## 4. 典型例题：上半圆补线

计算：

\[
\int_L(-y\,\mathrm dx+x\,\mathrm dy)
\]

其中 \(L\) 为上半圆：

\[
x^2+y^2=1
\]

方向：

\[
(-1,0)\rightarrow(1,0)
\]

**第一步：构造闭曲线**

补充下半部分的直径：

\[
C:(1,0)\rightarrow(-1,0)
\]

沿 \(x\) 轴运动。

此时上半圆从左向右，下方直径从右向左，构成的是顺时针闭曲线。

**第二步：使用格林公式**

仍然有：

\[
Q_x-P_y=2
\]

由于方向为顺时针：

\[
\oint_{L+C}
=-\iint_D2\,\mathrm dA
\]

区域为单位半圆：

\[
S_D=\frac{\pi}{2}
\]

所以：

\[
\oint_{L+C}=-2\cdot\frac{\pi}{2}=-\pi
\]

**第三步：计算补线积分**

在 \(x\) 轴上：

\[
y=0,\qquad \mathrm dy=0
\]

所以：

\[
\int_C(-y\,\mathrm dx+x\,\mathrm dy)=0
\]

因此：

\[
\int_L=-\pi-0
\]

答案：

\[
\boxed{-\pi}
\]

**特别注意：补线法最容易出错的地方就是方向。不能只计算区域面积而忽略正负号。**

---

# 五、利用格林公式求平面图形面积

格林公式还有一个非常重要的用途：计算封闭曲线围成的面积。

由格林公式：

\[
A=\iint_D1\,\mathrm dA
\]

可以得到：

\[
\boxed{
A=\oint_Lx\,\mathrm dy
}
\]

同样可以得到：

\[
\boxed{
A=-\oint_Ly\,\mathrm dx
}
\]

以及：

\[
\boxed{
A=\frac12\oint_L(x\,\mathrm dy-y\,\mathrm dx)
}
\]

以上公式均要求 \(L\) 取逆时针正向。

## 典型例题：椭圆面积

利用曲线积分计算椭圆：

\[
\frac{x^2}{a^2}+\frac{y^2}{b^2}=1
\]

的面积。

**解：**

参数化：

\[
x=a\cos t,\quad y=b\sin t
\]

\[
0\leq t\leq2\pi
\]

因此：

\[
\mathrm dy=b\cos t\,\mathrm dt
\]

使用面积公式：

\[
\begin{aligned}
A
&=\oint_Lx\,\mathrm dy\\
&=\int_0^{2\pi}ab\cos^2t\,\mathrm dt\\
&=\pi ab
\end{aligned}
\]

答案：

\[
\boxed{A=\pi ab}
\]

---

# 六、第一类曲面积分（对面积的曲面积分）

## 1. 基本形式

\[
\boxed{
\iint_\Sigma f(x,y,z)\,\mathrm dS
}
\]

本质：在曲面上按照面积对函数进行积分。

**核心特点：第一类曲面积分与曲面的侧无关。**

也就是说，取上侧还是下侧，结果都一样。

## 2. 曲面投影公式

这是第一类曲面积分最基本的计算方法。

假设曲面可以表示为：

\[
z=z(x,y)
\]

在 \(xOy\) 平面上的投影区域为 \(D\)。

则：

\[
\boxed{
\mathrm dS=
\sqrt{1+z_x^2+z_y^2}\,\mathrm dx\,\mathrm dy
}
\]

所以：

\[
\boxed{
\iint_\Sigma f(x,y,z)\,\mathrm dS
=
\iint_D
f(x,y,z(x,y))
\sqrt{1+z_x^2+z_y^2}
\,\mathrm dx\,\mathrm dy
}
\]

注意：这里的根号因子不能漏掉。

## 3. 典型例题：抛物面第一类曲面积分

计算：

\[
\iint_\Sigma z\,\mathrm dS
\]

其中：

\[
z=x^2+y^2,\qquad z\leq1
\]

**第一步：求偏导数**

\[
z_x=2x,\qquad z_y=2y
\]

所以：

\[
\mathrm dS=
\sqrt{1+4x^2+4y^2}\,\mathrm dx\,\mathrm dy
\]

**第二步：确定投影区域**

由：

\[
z=x^2+y^2\leq1
\]

得到：

\[
D:x^2+y^2\leq1
\]

**第三步：建立二重积分**

\[
I=
\iint_D
(x^2+y^2)
\sqrt{1+4x^2+4y^2}
\,\mathrm dx\,\mathrm dy
\]

使用极坐标：

\[
x=r\cos\theta,\quad y=r\sin\theta
\]

\[
\mathrm dx\,\mathrm dy=r\,\mathrm dr\,\mathrm d\theta
\]

得到：

\[
I=
2\pi\int_0^1r^3\sqrt{1+4r^2}\,\mathrm dr
\]

如果继续计算：

\[
\boxed{
I=\frac{\pi(25\sqrt5+1)}{60}
}
\]

## 4. 利用对称性计算球面积分

计算：

\[
\iint_\Sigma z^2\,\mathrm dS
\]

其中：

\[
x^2+y^2+z^2=a^2
\]

为整个球面。

由于球面对称：

\[
\iint_\Sigma x^2\,\mathrm dS
=
\iint_\Sigma y^2\,\mathrm dS
=
\iint_\Sigma z^2\,\mathrm dS
\]

设：

\[
I=\iint_\Sigma z^2\,\mathrm dS
\]

则：

\[
3I=
\iint_\Sigma(x^2+y^2+z^2)\,\mathrm dS
\]

利用球面方程：

\[
3I=a^2\iint_\Sigma\mathrm dS
\]

球面积：

\[
S=4\pi a^2
\]

因此：

\[
3I=4\pi a^4
\]

最终：

\[
\boxed{
I=\frac{4\pi a^4}{3}
}
\]

**方法总结：**

第一类曲面积分主要考虑：

- 曲面投影公式。
- 选择合适的投影坐标平面。
- 求出面积元。
- 使用极坐标或其他变量代换。
- 利用曲面的对称性。

---

# 七、第二类曲面积分（对坐标的曲面积分）——重点

这一部分是整篇文章最重要的内容之一。

第二类曲面积分的直接投影法必须熟练掌握，尤其是曲面表示为 \(z=z(x,y)\) 时，如何处理：

\[
(z_x,z_y,-1)
\]

以及它与曲面方向之间的关系。

## 1. 基本形式

\[
\boxed{
\iint_\Sigma
P\,\mathrm dy\,\mathrm dz
+
Q\,\mathrm dz\,\mathrm dx
+
R\,\mathrm dx\,\mathrm dy
}
\]

它也可以写成向量形式：

\[
\boxed{
\iint_\Sigma
\vec F\cdot\vec n\,\mathrm dS
}
\]

其中：

\[
\vec F=(P,Q,R)
\]

本质：向量场穿过有向曲面的通量。

**核心特点：第二类曲面积分与曲面的方向有关。**

如果将曲面的法向方向反过来：

\[
\boxed{
\iint_{-\Sigma}
=
-\iint_\Sigma
}
\]

## 2. 最基本的投影公式：曲面 \(z=z(x,y)\)

设曲面：

\[
z=z(x,y)
\]

在 \(xOy\) 平面上的投影区域为 \(D\)。

首先参数化曲面：

\[
\vec r(x,y)=(x,y,z(x,y))
\]

求两个切向量：

\[
\vec r_x=(1,0,z_x)
\]

\[
\vec r_y=(0,1,z_y)
\]

叉乘：

\[
\vec r_x\times\vec r_y
=
\begin{vmatrix}
\vec i&\vec j&\vec k\\
1&0&z_x\\
0&1&z_y
\end{vmatrix}
\]

得到：

\[
\boxed{
\vec r_x\times\vec r_y=(-z_x,-z_y,1)
}
\]

这个向量的 \(z\) 分量为正，因此对应**上侧（朝向 \(z\) 正方向）**。

如果取下侧，则法向量反向：

\[
\boxed{
(z_x,z_y,-1)
}
\]

这就是直接投影法中最重要的两个向量。

### 上侧公式

对于上侧：

\[
\boxed{
\begin{aligned}
&\iint_\Sigma
P\,\mathrm dy\,\mathrm dz+
Q\,\mathrm dz\,\mathrm dx+
R\,\mathrm dx\,\mathrm dy\\
&=
\iint_D
(-Pz_x-Qz_y+R)\,\mathrm dx\,\mathrm dy
\end{aligned}
}
\]

### 下侧公式

对于下侧：

\[
\boxed{
\begin{aligned}
&\iint_\Sigma
P\,\mathrm dy\,\mathrm dz+
Q\,\mathrm dz\,\mathrm dx+
R\,\mathrm dx\,\mathrm dy\\
&=
\iint_D
(Pz_x+Qz_y-R)\,\mathrm dx\,\mathrm dy
\end{aligned}
}
\]

注意：以上 \(P,Q,R\) 都要先代入曲面方程 \(z=z(x,y)\)。

### 为什么会出现这些正负号？

关键是三个有向面积元：

\[
\mathrm dy\,\mathrm dz
=
-z_x\,\mathrm dx\,\mathrm dy
\]

\[
\mathrm dz\,\mathrm dx
=
-z_y\,\mathrm dx\,\mathrm dy
\]

\[
\mathrm dx\,\mathrm dy
=
\mathrm dx\,\mathrm dy
\]

因此：

\[
P\,\mathrm dy\,\mathrm dz+
Q\,\mathrm dz\,\mathrm dx+
R\,\mathrm dx\,\mathrm dy
\]

直接变为：

\[
(-Pz_x-Qz_y+R)\,\mathrm dx\,\mathrm dy
\]

这就是上侧公式。

下侧只需要整体变号。

## 3. 典型例题：第二类曲面积分直接投影

计算：

\[
\iint_\Sigma
(x\,\mathrm dy\,\mathrm dz+
y\,\mathrm dz\,\mathrm dx+
z\,\mathrm dx\,\mathrm dy)
\]

其中曲面：

\[
z=x^2+y^2,\qquad z\leq1
\]

取上侧。

**第一步：确定被积函数**

由题目可知：

\[
P=x,\quad Q=y,\quad R=z
\]

**第二步：求偏导数**

\[
z_x=2x,\qquad z_y=2y
\]

**第三步：确定积分方向**

题目要求上侧，因此使用：

\[
(-z_x,-z_y,1)
\]

对应的投影公式：

\[
I=
\iint_D(-Pz_x-Qz_y+R)\,\mathrm dx\,\mathrm dy
\]

代入：

\[
\begin{aligned}
I
&=\iint_D(-2x^2-2y^2+x^2+y^2)
\,\mathrm dx\,\mathrm dy\\
&=-\iint_D(x^2+y^2)\,\mathrm dx\,\mathrm dy
\end{aligned}
\]

**第四步：确定投影区域**

由：

\[
z=x^2+y^2\leq1
\]

可得：

\[
D:x^2+y^2\leq1
\]

**第五步：极坐标计算**

\[
\begin{aligned}
I
&=-\int_0^{2\pi}\int_0^1r^2\cdot r
\,\mathrm dr\,\mathrm d\theta\\
&=-2\pi\int_0^1r^3\,\mathrm dr\\
&=-\frac{\pi}{2}
\end{aligned}
\]

答案：

\[
\boxed{I=-\frac{\pi}{2}}
\]

如果改为下侧，结果整体变号：

\[
\boxed{I=\frac{\pi}{2}}
\]

**这道题需要掌握的核心就是：求偏导数 → 判断上下侧 → 代入投影公式 → 确定投影区域 → 二重积分。**

## 4. 其他两种投影方式

上面介绍的是投影到 \(xOy\) 平面。

实际上，曲面也可以表示为：

\[
x=x(y,z)
\]

或者：

\[
y=y(x,z)
\]

这时需要投影到其他坐标平面。

三个坐标平面的公式可以统一整理如下：

| 曲面方程 | 投影平面 | 上侧/正向对应法向量 | 下侧/反向对应法向量 |
|---|---|---|---|
| \(z=z(x,y)\) | \(xOy\) | \((-z_x,-z_y,1)\) | \((z_x,z_y,-1)\) |
| \(x=x(y,z)\) | \(yOz\) | \((1,-x_y,-x_z)\) | \((-1,x_y,x_z)\) |
| \(y=y(x,z)\) | \(xOz\) | \((-y_x,1,-y_z)\) | \((y_x,-1,y_z)\) |

这里的正向和反向分别指法向量对应坐标轴正方向和负方向。

相应的第二类曲面积分公式如下。

### （1）投影到 \(xOy\) 平面

曲面：

\[
z=z(x,y)
\]

上侧：

\[
\boxed{
\iint_D(-Pz_x-Qz_y+R)\,\mathrm dx\,\mathrm dy
}
\]

下侧：

\[
\boxed{
\iint_D(Pz_x+Qz_y-R)\,\mathrm dx\,\mathrm dy
}
\]

### （2）投影到 \(yOz\) 平面

曲面：

\[
x=x(y,z)
\]

取法向量 \(x\) 分量为正的方向：

\[
\boxed{
\iint_D(P-Qx_y-Rx_z)\,\mathrm dy\,\mathrm dz
}
\]

反向：

\[
\boxed{
\iint_D(-P+Qx_y+Rx_z)\,\mathrm dy\,\mathrm dz
}
\]

### （3）投影到 \(xOz\) 平面

曲面：

\[
y=y(x,z)
\]

取法向量 \(y\) 分量为正的方向：

\[
\boxed{
\iint_D(-Py_x+Q-Ry_z)\,\mathrm dx\,\mathrm dz
}
\]

反向：

\[
\boxed{
\iint_D(Py_x-Q+Ry_z)\,\mathrm dx\,\mathrm dz
}
\]

注意：这里 \(xOz\) 平面的面积元使用的是 \(\mathrm dx\,\mathrm dz\)，不是 \(\mathrm dz\,\mathrm dx\)。上述公式已经将法向量方向统一处理好了。

## 5. 如何选择投影平面？

实际做题时，不一定要投影到 \(xOy\) 平面。

应该优先选择：

- 曲面方程容易表示的平面。
- 投影区域边界简单的平面。
- 能够让偏导数计算尽量简单的平面。
- 投影后容易使用极坐标的平面。

例如：

如果曲面方程是：

\[
z=x^2+y^2
\]

显然投影到 \(xOy\) 平面最方便。

如果曲面方程是：

\[
x=y^2+z^2
\]

显然投影到 \(yOz\) 平面更方便。

如果曲面方程是：

\[
y=x^2+z^2
\]

则投影到 \(xOz\) 平面更方便。

## 6. 特殊情况：被积函数只有一个分量

例如：

\[
\iint_\Sigma R\,\mathrm dx\,\mathrm dy
\]

如果曲面取上侧，则：

\[
\boxed{
\iint_\Sigma R\,\mathrm dx\,\mathrm dy
=
\iint_D R(x,y,z(x,y))\,\mathrm dx\,\mathrm dy
}
\]

因为上侧的法向量 \(z\) 分量为正。

如果取下侧：

\[
\boxed{
\iint_\Sigma R\,\mathrm dx\,\mathrm dy
=
-\iint_D R(x,y,z(x,y))\,\mathrm dx\,\mathrm dy
}
\]

这也解释了为什么有些题目只需要将曲面方程代入，然后直接计算二重积分。

---

# 八、高斯公式

## 1. 基本形式

高斯公式用于将闭曲面的第二类曲面积分转化为三重积分。

设 \(\Sigma\) 为包围空间区域 \(\Omega\) 的封闭曲面，取外侧，则：

\[
\boxed{
\iint_\Sigma
P\,\mathrm dy\,\mathrm dz+
Q\,\mathrm dz\,\mathrm dx+
R\,\mathrm dx\,\mathrm dy
=
\iiint_\Omega
\left(
P_x+Q_y+R_z
\right)\mathrm dV
}
\]

其中：

\[
P_x+Q_y+R_z
\]

称为向量场：

\[
\vec F=(P,Q,R)
\]

的散度：

\[
\boxed{
\nabla\cdot\vec F=P_x+Q_y+R_z
}
\]

**使用条件：闭曲面，且积分方向为外侧。**

如果取内侧，则整体变号。

## 2. 典型例题：球面的通量

计算：

\[
\iint_\Sigma
(x\,\mathrm dy\,\mathrm dz+
y\,\mathrm dz\,\mathrm dx+
z\,\mathrm dx\,\mathrm dy)
\]

其中 \(\Sigma\) 为球面：

\[
x^2+y^2+z^2=a^2
\]

取外侧。

**第一步：构造向量场**

\[
\vec F=(x,y,z)
\]

**第二步：求散度**

\[
\nabla\cdot\vec F=1+1+1=3
\]

**第三步：使用高斯公式**

\[
\begin{aligned}
I
&=\iiint_\Omega3\,\mathrm dV\\
&=3\cdot\frac43\pi a^3\\
&=4\pi a^3
\end{aligned}
\]

答案：

\[
\boxed{I=4\pi a^3}
\]

## 3. 非闭曲面如何使用高斯公式？

有些题目给出的曲面并不是封闭曲面，但是可以通过补充一个曲面，使其构成封闭曲面。

例如：上半球面。

可以补充底面：

\[
z=0
\]

构成封闭曲面。

设：

\[
\Sigma=\Sigma_1+\Sigma_2
\]

其中：

- \(\Sigma_1\)：原来的上半球面。
- \(\Sigma_2\)：补充的圆盘底面。

则：

\[
\iint_{\Sigma_1}+\iint_{\Sigma_2}
=
\iiint_\Omega\nabla\cdot\vec F\,\mathrm dV
\]

因此：

\[
\boxed{
\iint_{\Sigma_1}
=
\iiint_\Omega\nabla\cdot\vec F\,\mathrm dV
-
\iint_{\Sigma_2}
}
\]

这实际上就是曲面积分中的补面法。

**注意：补面以后，原曲面和补充曲面都必须按照封闭曲面的外侧确定方向。**

---

# 九、对称性与其他常用技巧

## 1. 对称性

当积分区域、曲面或者被积函数具有对称性时，可以考虑利用对称性简化积分。

例如，单位球面：

\[
x^2+y^2+z^2=a^2
\]

由于球面对称：

\[
\iint_\Sigma x^2\,\mathrm dS
=
\iint_\Sigma y^2\,\mathrm dS
=
\iint_\Sigma z^2\,\mathrm dS
\]

而：

\[
x^2+y^2+z^2=a^2
\]

因此：

\[
3\iint_\Sigma x^2\,\mathrm dS
=
4\pi a^4
\]

从而：

\[
\boxed{
\iint_\Sigma x^2\,\mathrm dS
=
\frac{4\pi a^4}{3}
}
\]

## 2. 奇偶对称

如果积分区域关于某个坐标轴对称，而被积函数关于该坐标为奇函数，则积分为零。

例如：

\[
\iint_Dx\,\mathrm dA=0
\]

其中 \(D\) 关于 \(y\) 轴对称。

类似地：

\[
\iint_Dy\,\mathrm dA=0
\]

其中 \(D\) 关于 \(x\) 轴对称。

但是需要注意：必须同时满足积分区域对称和被积函数的奇偶性要求。

## 3. 第一类与第二类积分的关系

对于空间曲线：

\[
\int_LP\,\mathrm dx+Q\,\mathrm dy+R\,\mathrm dz
\]

可以理解为：

\[
\int_L\vec F\cdot\vec\tau\,\mathrm ds
\]

其中：

\[
\vec\tau
\]

为单位切向量。

对于曲面：

\[
\iint_\Sigma
P\,\mathrm dy\,\mathrm dz+
Q\,\mathrm dz\,\mathrm dx+
R\,\mathrm dx\,\mathrm dy
\]

可以理解为：

\[
\iint_\Sigma\vec F\cdot\vec n\,\mathrm dS
\]

其中：

\[
\vec n
\]

为单位法向量。

这样就能理解：

- 曲线积分考虑向量场沿切线方向的分量。
- 曲面积分考虑向量场沿法线方向的分量。

---

# 十、考研解题方法总结

这一部分建议重点记忆。

## 1. 第一类曲线积分

看到：

\[
\int_Lf(x,y)\,\mathrm ds
\]

优先考虑：

1. 参数化曲线。
2. 求弧长元。
3. 代入一元积分。
4. 利用对称性。

核心公式：

\[
\mathrm ds=
\sqrt{(x')^2+(y')^2}\,\mathrm dt
\]

## 2. 第二类曲线积分

看到：

\[
\int_LP\,\mathrm dx+Q\,\mathrm dy
\]

优先考虑：

1. 能否直接参数化？
2. 是否满足路径无关条件？
3. 是否可以找到势函数？
4. 是否为闭合曲线？
5. 如果不是闭合曲线，能否补线使用格林公式？

核心公式：

\[
P_y=Q_x
\]

以及：

\[
\int_L\mathrm d\varphi=\varphi(B)-\varphi(A)
\]

注意路径无关的判定需要结合积分区域的条件。

## 3. 格林公式

看到：

\[
\oint_LP\,\mathrm dx+Q\,\mathrm dy
\]

优先考虑格林公式：

\[
\boxed{
\oint_LP\,\mathrm dx+Q\,\mathrm dy
=
\iint_D(Q_x-P_y)\,\mathrm dA
}
\]

注意：

- 逆时针为正向。
- 顺时针整体变号。
- 非闭合曲线可以考虑补线。
- 求平面区域面积也可以考虑格林公式。

## 4. 第一类曲面积分

看到：

\[
\iint_\Sigma f(x,y,z)\,\mathrm dS
\]

优先考虑曲面投影：

\[
\boxed{
\mathrm dS=
\sqrt{1+z_x^2+z_y^2}\,\mathrm dx\,\mathrm dy
}
\]

还可以考虑：

- 极坐标。
- 球坐标。
- 曲面的对称性。

注意：第一类曲面积分不需要考虑曲面的方向。

## 5. 第二类曲面积分

看到：

\[
\iint_\Sigma
P\,\mathrm dy\,\mathrm dz+
Q\,\mathrm dz\,\mathrm dx+
R\,\mathrm dx\,\mathrm dy
\]

首先判断：

**曲面是否封闭？**

如果是封闭曲面：

优先考虑高斯公式。

如果不是封闭曲面：

优先考虑直接投影法。

对于：

\[
z=z(x,y)
\]

上侧：

\[
\boxed{
\iint_D(-Pz_x-Qz_y+R)\,\mathrm dx\,\mathrm dy
}
\]

下侧：

\[
\boxed{
\iint_D(Pz_x+Qz_y-R)\,\mathrm dx\,\mathrm dy
}
\]

对于其他曲面：

- \(x=x(y,z)\)：投影到 \(yOz\) 平面。
- \(y=y(x,z)\)：投影到 \(xOz\) 平面。

如果非闭曲面能够补充曲面构成封闭曲面，也可以考虑高斯公式。

---

# 十一、最终方法选择表

| 题目特征 | 优先考虑的方法 |
|---|---|
| 第一类曲线积分 | 参数化、弧长元 |
| 第二类曲线积分 | 参数化直接计算 |
| \(P_y=Q_x\) 且区域满足条件 | 路径无关、求势函数 |
| 第二类曲线积分为闭合曲线 | 格林公式 |
| 第二类曲线积分不是闭合曲线 | 补线法 |
| 利用曲线积分求平面面积 | 格林公式 |
| 第一类曲面积分 | 投影公式、面积元 |
| 第二类曲面积分为非闭曲面 | 直接投影法 |
| 第二类曲面积分为闭曲面 | 高斯公式 |
| 非闭曲面可以方便补面 | 补面法、高斯公式 |
| 球面或具有对称性的曲面 | 对称性 |
| 被积函数具有奇偶性 | 奇偶对称 |

---

# 十二、特别需要掌握的公式

最后把最重要的公式集中整理，方便复习时快速回顾。

**第一类曲线积分：**

\[
\int_Lf\,\mathrm ds
=
\int_\alpha^\beta
f(x(t),y(t))
\sqrt{(x')^2+(y')^2}\,\mathrm dt
\]

**第二类曲线积分：**

\[
\int_LP\,\mathrm dx+Q\,\mathrm dy
=
\int_\alpha^\beta(Px'+Qy')\,\mathrm dt
\]

**格林公式：**

\[
\oint_LP\,\mathrm dx+Q\,\mathrm dy
=
\iint_D(Q_x-P_y)\,\mathrm dA
\]

**第一类曲面积分：**

\[
\mathrm dS=
\sqrt{1+z_x^2+z_y^2}\,\mathrm dx\,\mathrm dy
\]

**第二类曲面积分（上侧）：**

\[
\boxed{
\iint_\Sigma
P\,\mathrm dy\,\mathrm dz+
Q\,\mathrm dz\,\mathrm dx+
R\,\mathrm dx\,\mathrm dy
=
\iint_D(-Pz_x-Qz_y+R)\,\mathrm dx\,\mathrm dy
}
\]

**第二类曲面积分（下侧）：**

\[
\boxed{
\iint_D(Pz_x+Qz_y-R)\,\mathrm dx\,\mathrm dy
}
\]

**高斯公式：**

\[
\boxed{
\iint_\Sigma\vec F\cdot\vec n\,\mathrm dS
=
\iiint_\Omega\nabla\cdot\vec F\,\mathrm dV
}
\]

---

**学习建议：**

曲线积分和曲面积分最重要的不是记住所有计算过程，而是掌握积分类型、积分方向、参数化与投影之间的关系。

特别是第二类曲面积分，建议把三个投影平面的法向量公式真正推导一遍，而不是单纯背诵。只要理解了叉乘和有向面积元，遇到不同的曲面方程也能够自行推出正确公式。