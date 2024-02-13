---
title: Transformer
author: Fangzheng
date: 2023-12-28 23:06:00 +0800
categories: [Transformer, Artificial Intelligence]
tags: [algorithm ]
# pin: true
mermaid: true  #code模块
comments: true
mermaid: true

math: true
# img_cdn: https://github.com/MysteriousL2019/mysteriousl2019.github.io/tree/master/assets/img/
---
## Before Transformer
* 深度学习做 NLP 的方法，基本上都是先将句子分词，然后每个词转化为对应的词向量序列。这样一来，每个句子都对应的是一个矩阵 $X = (x_1,x_2,...,x_t)$, 其中$x_i$都代表着第i个词的向量（行向量），纬度（d），$X\in R^{n*d}$问题就变成了编码这些序列了。
* 第一个基本的思路是 RNN 层，RNN 的方案很简单，递归式进行：
* $y_t = f(y_{t-1},x_t)$不管是已经被广泛使用的 LSTM、GRU 还是最近的 SRU，都并未脱离这个递归框架。RNN 结构本身比较简单，也很适合序列建模，但 RNN 的明显缺点之一就是无法并行，因此速度较慢，这是递归的天然缺陷。它本质是一个马尔科夫决策过程,RNN 无法很好地学习到全局的结构信息.
* 第二个思路是 CNN 层，其实 CNN 的方案也是很自然的，窗口式遍历，比如尺寸为 3 的卷积，就是
* $y_t = f(x_{t-1},x_t,x_{t+1})$
* 在 FaceBook 的论文中，纯粹使用卷积也完成了 Seq2Seq 的学习，是卷积的一个精致且极致的使用案例，热衷卷积的读者必须得好好读读这篇文论。CNN 方便并行，而且容易捕捉到一些全局的结构信息。但是我认为 CNN 还是有一些问题，那就是需要叠一定量的层数之后才可以获取到较为全局的西信息。
* Google 的大作 《Attention Is All You Need》 提供了第三个思路：**纯 Attention！单靠注意力就可以！**RNN 要逐步递归才能获得全局信息，因此一般要双向 RNN 才比较好；CNN事实上只能获取局部信息，是通过层叠来增大感受野；Attention 的思路最为粗暴，它一步到位获取了全局信息！它的解决方案是
* $y_t = f(x_t,A,B)$ 其中 A, B 是另外一个序列（矩阵）。如果都取 A = B = X ，那么就称为Self Attention，它的意思是直接将 $x_t$与原来的每个词进行比较，最后算出$y_t$
## Serval Basic Units
### Bath Normalize & Layer Normalize
* 加速网络收敛速度，提升训练稳定性，Batchnorm本质上是解决反向传播过程中的梯度问题。batchnorm全名是batch normalization，简称BN，即批规范化，通过规范化操作将输出信号x规范化到均值为0，方差为1保证网络的稳定性。
* $\hat x_i = \frac{x_i-\mu_b}{\sigma^2_b+\varepsilon }$
* 此部分大概讲一下batchnorm解决梯度的问题上。具体来说就是反向传播中，经过每一层的梯度会乘以该层的权重，举个简单例子： 正向传播中
     * $f_3=f_2(w^Tx+b)$, 那么反向传播中,$\frac{\partial f_2}{\partial x}=\frac{\partial f_2}{\partial f_1}w $
     * 反向传播式子中有w的存在，所以的大小影响了梯度的消失和爆炸，batchnorm就是通过对每一层的输出做scale和shift的方法，通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到接近均值为0方差为1的标准正太分布，即严重偏离的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入比较敏感的区域，这样输入的小变化就会导致损失函数较大的变化，使得让梯度变大，避免梯度消失问题产生，而且梯度变大意味着学习收敛速度快，能大大加快训练速度。
* 来自操作： BN 对同一批次中的所有数据的同一特征数据进行操作；而 LN 对同一样本进行操作。
* BN 不适合 RNN、变换器等序列网络，也不适合文本长度不确定和批量较小的情况，它适合 CV 中的 CNN 等网络。
* 而 LN 适用于 NLP 中的 RNN 和变换器等网络，因为序列的长度可能不一致。
<!-- * ![Alt text](image.png) -->
### Code 
```
     torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     num_features: the number of features from the desired input, the size of the desired input is 'batch_size x num_features [x width]'.
     eps: the value to be added to the denominator to ensure numerical stability (the denominator cannot converge to or take 0). Default is 1e-5.
     momentum: the momentum used for dynamic mean and dynamic variance. The default is 0.1.
     affine: boolean value, when set to true, to add the layer can be learned affine transformation parameters.
     track_running_stats: boolean, when set to true, records the mean and variance during training;
```

### Why Normalization
* 在神经网络中，数据的分布会对训练产生影响。例如，如果一个神经元 x 的值为 1，而 Weights 的初始值为 0.1，那么下一层神经元将计算 Wx = 0.1；或者如果 x = 20，那么 Wx 的值将为 2。我们还没发现问题，但当我们添加一层激励来激活 Wx 的这个值时，问题就出现了。如果我们使用 tanh 这样的激励函数，Wx 的激活值就会变成 ~0.1 和 ~1，而接近 1 的部分已经处于激励函数的饱和阶段，也就是说，如果 x 无论扩大多少，tanh 激励函数的输出仍然接近 1。换句话说，神经网络不再对初始阶段的大范围 x 特征敏感。换句话说，神经网络不再对初始阶段那些大范围的 x 特征敏感。这就糟糕了，想象一下，拍打自己和撞击自己的感觉没有任何区别，这就证明我的感觉系统失灵了。当然，我们可以使用前面提到的归一化预处理，让输入 x 范围不要太大，这样输入值就能通过激励函数的敏感部分。但这种不敏感问题不仅出现在神经网络的输入层，也会出现在隐藏层。但是，当 x 被隐藏层取代后，我们还能像以前那样对隐藏层的输入进行归一化处理吗？答案是肯定的，因为大人物们已经发明了一种叫做批量归一化的技术，可以解决这种情况。
### BN Add Location
* 批量归一化是将一批数据分成若干小批，用于随机梯度下降。此外，在对每批数据进行前向传播时，会对每一层进行归一化处理。
### BN Result
* * 批量归一化也可以看作是一层。在逐层添加神经网络时，我们从数据 X 开始，添加一个全连接层，全连接层的结果通过激励函数成为下一层的输入，然后重复前面的操作。在每个全连接层和激励函数之间添加批量归一化（BN）。正如我之前所说，结果在进入激励函数之前的值很重要，如果我们不只看一个值，我们可以说结果值的分布对激励函数很重要。如果我们不只看一个值，我们就可以说计算值的分布对激发函数很重要，如果数值大多分布在这个区间内，数据的传递效率就会更高。比较一下激活前这两个值的分布。
* 上面的数据没有经过归一化处理，而下面的数据经过了归一化处理，这当然意味着下面的数据能够更有效地利用 tanh 进行非线性化处理。在对未归一化的数据进行 tanh 激活后，大部分激活值都分布到了饱和阶段，即大部分激活值不是-1 就是 1，而在归一化后，大部分激活值仍然存在于分布的每个区间。通过将这种激活分布传递给神经网络的下一层进行后续计算，每个区间中存在的分布对神经网络更有价值。批量归一化不只是对数据进行归一化，它还会对数据进行去归一化。
### Flaws of BN
* BN是按照样本数计算归一化统计量的，当样本数很少时，比如说只有4个。这四个样本的均值和方差便不能反映全局的统计分布息，所以基于少量样本的BN的效果会变得很差。在一些场景中，比如说硬件资源受限，在线学习等场景，BN是非常不适用的。
<!-- * ![Alt text](image-1.png) -->
## Resnet
*  x --> F(x) --> x+F(x)
* 所以结果来看 $\frac{\partial loss}{\partial x_l} =\frac{\partial loss}{\partial x_L}*\frac{\partial x_L}{\partial x_l}=\frac{\partial loss}{\partial x_L}(1+\frac{\partial \sum_{i=l}^{L-1}F(x_i,W_i)}{\partial x_L} )   $
* 式子的第一个因子 $\frac{\partial loss}{\partial x_L} $ 表示的损失函数到达 L 的梯度，小括号中的1表明短路机制可以无损地传播梯度，而另外一项残差梯度则需要经过带有weights的层，梯度不是直接传递过来的。残差梯度不会那么巧全为-1，而且就算其比较小，有1的存在也不会导致梯度消失。所以残差学习会更容易。
## Positional Encoding
* 假设有一个长度为的L输入序列，token在句子中的位置计作POS，那么token的位置编码PE = POS = 0, 1, 2, ..., T-1。显然有一个问题，若很长序列为（10,000）。那么左右一个token的POS值将会很大，导致:
     * 和word embedding向量做加法之后将会使得网络把后面的编码较大的token当作主要信息（这是没有道理的）
     * 所以PE需要有一定的值域范围
* 那我们如果直接对第一种方法直接归一化呢？ $ PE = \frac{POS}{T-1}$.这样会使得所有PE都落到[0,1] 但问题是：
     * 不同长度的序列位置编码步长不同（相邻token的数值差会收到序列长度影响），短序列的相邻token的位置编码差异会小于长序列的位置编码差异。————会导致长序列中相对关系被“稀释”。
     * 我们关注的位置信息主要是相对次序关系，————上下文次序（POS1 和POS2 要比POS3 和10更近）所以被稀释是致命的。
### 总结一下，position encoding的定义要满足下列需求：
* 每个位置有一个唯一的positional encoding；
* 最好具有一定的值域范围，否则它比一般的字嵌入的数值要大，难免会抢了字嵌入的「风头」，对模型可能有一定的干扰；
* 需要体现一定的相对次序关系，并且在一定范围内的编码差异不应该依赖于文本长度，具有一定translation invariant平移不变性。
* *  假设有一个长度为的L输入序列，并且需要$POS^{th}$对象在此序列中的位置。位置编码由不同频率的正弦和余弦函数给出：
* $$P(POS,2i) = sin(\frac{POS}{n^{\frac{2i}{d_{model}}}})$$
* $$
     P(POS,2i+1) = cos(\frac{POS}{n^{\frac{2i}{d_{model}}}})$$
* POS: 对象在输入序列中的位置, $0<=POS<\frac{L}{2}$
* $d_{model}$: 输出嵌入空间的尺寸
* n 用户定义的标量，由 Attention Is All You Need 的作者设置为 10,000。
* i 用于映射到列索引 $0<=i<\frac{d_{model}}{2}$ 具有正弦和余弦函数的i单个映射值.
* P(k,j) 位置函数，用于将输入序列中的位置 k 映射到位置矩阵的索引 (k,j) 

### 总结
* Transformer中涉及繁琐的矩阵计算，本质是用矩阵乘法衡量特征向量之间的相似度，理解了计算过程有助于加深理解网络设计原理。
     * 比如 向量 $a*b=|a||b|, a*b=-|a||b|, $向量乘法表示相似度
* Encoder 过程$(word2vec + PE)* \frac {softmax(Q,K)*(V)}{\sqrt d_k}$, 其中QKV都是原始的Word Embedding分别经过三个不同的全联接层得到的。
     * 为什么除以$\sqrt d_k$因为$d_k$过大时候内积也会增加，softmax函数的梯度很小
     * 先解释：为什么当$d_k$较大时，向量内积容易取很大的值（借用原论文的注释）
     * 假设 query 和 key 向量中的元素都是相互独立的均值为 0，方差为 1 的随机变量，那么这两个向量的内积$q^Tk=\sum_{i=1}^{d_k}q_ik_i$的均值是0，方差是$d_k$
     * Prove: 
          * $E[q_i]=E[k_i]=0$, $Var(q_i)=Var(k_i)=1$
          * 由于$q_i,k_i独立,Cov(q_i,k_i)=E[(q_i-E[q_i])(k_i-E[k_i])]=E[q_ik_i]-E[q_i]E[k_i]=0 ==> E[q_ik_i]=E[q_i]E[k_i]=0$
          * $Var(q_ik_i)=E[(q_ik_i)^2]-(E[q_ik_i])^2=E[q_i^2]E[k_i^2]-(E[q_i]E[k_i])^2=Var(q_i)Var(k_i)=1$ Since $Var(X+Y)=Var(X)+Var(Y)+2Cov(X,Y)=Var(X)+Var(Y) $for two independent variables
          * $E[q^Tk]=\sum_{i=1}^{d_k}=0$
          * $Var(q^Tk)=\sum_{i=1}^{d_k}Var(q_ik_i)=d_k$
          * 所以$d_k$较大时，$q^Tk$的方差较大，不同的 key 与同一个 query 算出的对齐分数可能会相差很大，有的远大于 0，有的则远小于 0.
     * 因此，向量内积的值（对齐分数）较大时==> $d_k$较大，方差较大，很多值落在softmax横坐标较大位置，softmax的函数梯度很小
     * 此时softmax函数梯度过低（趋于零），使得模型误差反向传播（back-propagation）经过 s o f t m a x softmaxsoftmax 函数后无法继续传播到模型前面部分的参数上，造成这些参数无法得到更新，最终影响模型的训练效率。
     * 因此score = $\frac{q^Tk}{\sqrt d_k}$, 由$Var(kx)=k^2Var(x)得到$此时的score是1，这样就消除了Var随着$d_k$增大而变动
* Decoder 过程 
     * AT auto-regression: 由于musk，不可以并行。类似英文汉译英的时候读入一整句话，翻译却是一点点写出来的
          * CrossAttention: from BEGIN(word2vec + PE)经过全联接层后，输出作为Query, 与Encoder 中的Key, Value 做Multi-head attention
          * 因此如果是AT的话，只有encoder可以并行。
     * NAT 非auto-regression: 可以并行化。
### Reference
* [图解 Transformers](https://zhuanlan.zhihu.com/p/654051912)
* [Dive deepl into Transformer](https://www.linkedin.com/pulse/deep-dive-positional-encodings-transformer-neural-network-ajay-taneja#:~:text=Positional%20Encodings%20can%20be%20looked,vector%20representation%20of%20the%20input.)
* [why trigonometric function PE?](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)
* [Positional Encoding](https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6)
* [大白话讲解 Transformer](https://zhuanlan.zhihu.com/p/264468193?utm_medium=social&utm_oi=1123713438710718464&utm_psn=1737408413846585345&utm_source=wechat_session)
* [从反向传播推导到梯度消失and爆炸的原因及解决](https://zhuanlan.zhihu.com/p/76772734)