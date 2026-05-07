# 01 RoPE Deep Dive：旋转位置编码推导

## 核心直觉

RoPE 的一句话解释是：

```text
不要把位置向量加到 token 上，而是按照位置把 Q 和 K 旋转一下。
```

旋转之后，两个 token 的 attention score 不只取决于它们本身的语义向量，还会自然带上相对位置 `m-n`。这就是 RoPE 比普通绝对位置编码更优雅的地方。

你可以先抓住这个结论：

$$
(R_m q_m)^\top(R_n k_n)
=q_m^\top R_{n-m}k_n
$$

左边看起来用了两个绝对位置 `m` 和 `n`，右边却只剩相对位置 `n-m`。这条式子就是 RoPE 的核心。

## 必要数学

### 1. 二维旋转矩阵

二维向量绕原点旋转角度 `theta`，可以写成矩阵乘法：

$$
R_\theta=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

对向量 `x = [x_1, x_2]^T`：

$$
R_\theta x=
\begin{bmatrix}
x_1\cos\theta-x_2\sin\theta \\
x_1\sin\theta+x_2\cos\theta
\end{bmatrix}
$$

旋转矩阵有两个关键性质：

$$
R_\theta^\top=R_{-\theta}
$$

$$
R_aR_b=R_{a+b}
$$

所以：

$$
R_m^\top R_n=R_{-m}R_n=R_{n-m}
$$

这一步非常关键：两个绝对旋转相乘后，会变成相对角度。

### 2. 复数乘法直觉

二维向量 `[x_1, x_2]` 也可以看成复数：

$$
z=x_1+ix_2
$$

乘上单位复数 `e^{i\theta}` 就是在复平面旋转：

$$
ze^{i\theta}
$$

欧拉公式告诉我们：

$$
e^{i\theta}=\cos\theta+i\sin\theta
$$

所以 RoPE 的“旋转”既可以用二维旋转矩阵理解，也可以用复数乘法理解。工程实现里通常不用真的创建复数，而是用 sin / cos 对向量维度做成对变换。

## RoPE 推导

### 1. Attention 原始分数

普通 self-attention 的 score 是：

$$
\operatorname{score}_{mn}=q_m^\top k_n
$$

其中：

```text
m: query token 的位置
n: key token 的位置
q_m: 第 m 个 token 的 query 向量
k_n: 第 n 个 token 的 key 向量
```

如果 `q_m` 和 `k_n` 不含位置，score 就主要表达语义匹配，不知道 `m` 和 `n` 的距离。

### 2. 给 Q/K 加旋转

RoPE 对不同位置使用不同旋转：

$$
\tilde{q}_m=R_mq_m
$$

$$
\tilde{k}_n=R_nk_n
$$

然后用旋转后的 Q/K 做 attention：

$$
\tilde{\operatorname{score}}_{mn}
=\tilde{q}_m^\top\tilde{k}_n
=(R_mq_m)^\top(R_nk_n)
$$

展开转置：

$$
(R_mq_m)^\top(R_nk_n)
=q_m^\top R_m^\top R_n k_n
$$

利用旋转矩阵性质：

$$
R_m^\top R_n=R_{n-m}
$$

得到：

$$
\tilde{\operatorname{score}}_{mn}
=q_m^\top R_{n-m}k_n
$$

这说明 attention score 依赖的是相对位置 `n-m`，不是单独依赖绝对位置 `m` 或 `n`。

## 多维向量怎么旋转

真实模型里的 head dimension 通常不是 2，而是 `D_h`，例如 64、80、128。RoPE 的做法是把最后一维两两分组：

```text
[x0, x1, x2, x3, x4, x5, ...]
 -> (x0, x1), (x2, x3), (x4, x5), ...
```

每一对二维子向量都用一个频率旋转：

$$
\theta_i=base^{-2i/D_h}
$$

位置 `m` 上第 `i` 对维度的旋转角度是：

$$
m\theta_i
$$

因此第 `i` 对维度用：

$$
R_{m\theta_i}
$$

这和 sinusoidal PE 很像：低维/高维对应不同频率，让模型同时感知短距离和长距离。

## 工程实现形态

### 1. rotate_half

很多代码里会看到 `rotate_half(x)`。它本质上是在做二维旋转里的这一步：

$$
(x_1,x_2)\rightarrow(-x_2,x_1)
$$

因为：

$$
R_\theta x
=x\cos\theta+\operatorname{rotate\_half}(x)\sin\theta
$$

所以实现常写成：

```text
x_rotated = x * cos + rotate_half(x) * sin
```

### 2. shape

常见 RoPE 输入：

```text
q, k:          (B, H, T, D_h)
position_ids:  (B, T) 或 (T,)
cos, sin:      (T, D_h)
q_rot, k_rot:  (B, H, T, D_h)
```

注意 `cos` / `sin` 经常会 unsqueeze 到能 broadcast 的形状：

```text
cos: (1, 1, T, D_h)
sin: (1, 1, T, D_h)
```

### 3. 为什么不旋转 V

Attention score 由 Q 和 K 决定：

$$
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_h}}\right)
$$

V 是被加权汇总的信息内容。RoPE 只需要让“谁关注谁”的分数带上位置信息，所以旋转 Q / K 就够了。旋转 V 会改变被聚合的内容表示，通常不是 RoPE 的目标。

## KV Cache 下的关键点

推理 decode 阶段，每次只生成一个新 token，但历史 K/V 会缓存下来：

```text
prefill: 处理完整 prompt，缓存历史 K/V
decode: 只计算新 token 的 Q/K/V，再拼到 cache
```

RoPE 下最容易出错的是 `position_id`。新 token 的位置不能每一步都从 0 开始，而应该接在历史长度后面：

```text
prompt 长度是 128
第一个 decode token 的 position_id 应该是 128
下一个应该是 129
```

如果 position_id 错了，Q/K 的旋转角度就错了，attention 看到的相对位置关系也会乱。

## 长上下文外推直觉

RoPE 的位置是通过旋转角度编码的。训练时如果最大长度是 2048，模型主要见过这些位置范围内的角度组合。直接推到 8192 时，角度可能进入训练时很少见的区域，模型可能不稳定。

常见长上下文改造会围绕两个方向：

```text
调整频率：让位置增长时旋转不要太快
缩放位置：把更长序列压回模型熟悉的角度范围
```

所以你会看到一些 RoPE scaling 方法。它们不是改变 Attention 的主体结构，而是调整 `position -> rotation angle` 的映射。

## 面试高频问法

1. RoPE 和 learned absolute position embedding 的区别是什么？
2. 为什么 RoPE 可以表达相对位置？
3. 推导一下 `(R_m q)^T (R_n k)` 为什么只依赖 `n-m`。
4. RoPE 为什么要把 hidden dimension 两两分组？
5. `rotate_half` 在做什么？
6. RoPE 为什么只旋转 Q/K，不旋转 V？
7. KV Cache 场景下 RoPE 最容易出什么 bug？
8. 长上下文扩展为什么经常要做 RoPE scaling？

## 常见陷阱

- RoPE 不是给 embedding 加一个位置向量，而是旋转 Q/K。
- “绝对位置旋转”经过点积后会变成“相对位置差”，这是 RoPE 的核心。
- `R_m^\top R_n = R_{n-m}` 的方向别写反；不同约定下可能写成 `m-n`，但核心是相对差。
- `cos` / `sin` 的 shape 要能 broadcast 到 `(B, H, T, D_h)`。
- KV Cache 时 position_id 必须延续历史长度。
- RoPE scaling 改的是位置到角度的映射，不是简单把上下文窗口数字调大。

## 自测题

1. 写出二维旋转矩阵 `R_theta`。
2. 为什么 `R_theta^T = R_-theta`？
3. 用一行公式推导 `R_m^T R_n = R_{n-m}`。
4. 从 `(R_mq_m)^T(R_nk_n)` 推到 `q_m^T R_{n-m} k_n`。
5. RoPE 为什么把最后一维两两分组？
6. `rotate_half([x1, x2])` 应该得到什么？
7. 为什么 RoPE 不旋转 V？
8. `q.shape = (B, H, T, D_h)`，`cos.shape` 通常要整理成什么形状方便 broadcast？
9. decode 时 prompt 长度是 512，新生成第 3 个 token 的 position_id 应该是多少？
10. 为什么长上下文外推和 RoPE 的频率/角度有关？
