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
* 因为通常来说BN在图片中使用，(B,C,H,W) for Batch, Channel, Height, Weight. (Batch_size,Sequence_length, Embedding_length),
* 图片在做可将(N, C, H*W)分别理解成(B, S, E)
* BatchNorm是对一个batch-size样本内的每个特征做归一化，LayerNorm是对每个样本的所有特征做归一化。BN 的转换是针对单个神经元可训练的：不同神经元的输入经过再平移和再缩放后分布在不同的区间；而 LN 对于一整层的神经元训练得到同一个转换：所有的输入都在同一个区间范围内。如果不同输入特征不属于相似的类别（比如颜色和大小），那么 LN 的处理可能会降低模型的表达能力。

* BN抹杀了不同特征之间的大小关系，但是保留了不同样本间的大小关系；LN抹杀了不同样本间的大小关系，但是保留了一个样本内不同特征之间的大小关系。（理解：BN对batch数据的同一特征进行标准化，变换之后，纵向来看，不同样本的同一特征仍然保留了之前的大小关系，但是横向对比样本内部的各个特征之间的大小关系不一定和变换之前一样了，因此抹杀或破坏了不同特征之间的大小关系，保留了不同样本之间的大小关系；LN对单一样本进行标准化，样本内的特征处理后原来数值大的还是相对较大，原来数值小的还是相对较小，不同特征之间的大小关系还是保留了下来，但是不同样本在各自标准化处理之后，两个样本对应位置的特征之间的大小关系将不再确定，可能和处理之前就不一样了，所以破坏了不同样本间的大小关系）

###  使用场景上
* 在BN和LN都能使用的场景中，BN的效果一般优于LN，原因是基于不同数据，同一特征得到的归一化特征更不容易损失信息。但是有些场景是不能使用BN的，例如batch size较小或者序列问题中可以使用LN。这也就解答了RNN 或Transformer为什么用Layer Normalization？

* 首先RNN或Transformer解决的是序列问题，一个存在的问题是不同样本的序列长度不一致，而Batch Normalization需要对不同样本的同一位置特征进行标准化处理，所以无法应用；当然，输入的序列都要做padding补齐操作，但是补齐的位置填充的都是0，这些位置都是无意义的，此时的标准化也就没有意义了。

* 其次上面说到，BN抹杀了不同特征之间的大小关系；LN是保留了一个样本内不同特征之间的大小关系，这对NLP任务是至关重要的。对于NLP或者序列任务来说，一条样本的不同特征，其实就是时序上的变化，这正是需要学习的东西自然不能做归一化抹杀，所以要用LN。
* 总的来说，**BatchNorm是对一个batch-size样本内的每个特征[分别]做归一化，LayerNorm是[分别]对每个样本的所有特征做归一化** 
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
### Gradient Vanishing & Gradient Explode
* Gradient Vanishing: 假设（假设每一层只有一个神经元且对于每一层 $y_i=\sigma (z_i)=\sigma(w_ix_i+b_i)$，则通过链式法则知，网络越深每一层网络求导后都会加上一个系数 $w_i$ 以及每次 $\sigma ^{'}(z_i)$ ，sigmoid函数求导后最大最大也只能是0.25。再来看W，一般我们初始化权重参数W时，通常都小于1，用的最多的是0，1正态分布。
     * 所以 $\|\sigma^{'}(z)W\|<=0.25$ 多个小于1的数连乘之后，那将会越来越小，导致靠近输入层的层的权重的偏导几乎为0，也就是说几乎不更新，这就是梯度消失的根本原因。
     * 再来看看梯度爆炸的原因，也就是说如果 $\|\sigma^{'}(z)W\|>=1$ ，则连乘下来就会导致梯度过大，导致梯度更新幅度特别大，可能会溢出，导致模型无法收敛。sigmoid的函数是不可能大于1了，那只能是w过大，故初始权重过大，会导致梯度爆炸。
     * 其实梯度爆炸和梯度消失问题都是因为网络太深，网络权值更新不稳定造成的，本质上是因为梯度反向传播中的连乘效应。
     ### How to deal with Gradient Vanishing & Gradient Explode
     * 1. pre-trained & Fine-tuning: in Deep Belief Networks firstly used
     * 2. 梯度剪切、正则： 梯度剪切这个方案主要是针对梯度爆炸提出的，其思想是设置一个梯度剪切阈值，然后更新梯度的时候，如果梯度超过这个阈值，那么就将其强制限制在这个范围之内。这可以防止梯度爆炸。正则化是通过对网络权重做正则限制过拟合，仔细看正则项在损失函数的形式： $LOSS=(y-W^Tx)^2+\alpha ||W||^2$ 事实上，在深度神经网络中，往往是梯度消失出现的更多一些
     * 3. Activation Function: (idea 最好是求导后值都等于一，这样网络深度加大后求导的值仅仅是连乘1，那么就不存在梯度消失爆炸的问题了)
          * relu:(解决了梯度消失、爆炸的问题,计算方便；计算速度快,加速了网络的训练； 但由于负数部分恒为0，会导致一些神经元无法激活（可通过设置小学习率部分解决；输出不是以0为中心的——会导致数据分布改变)
     * 4. Batchnorm(见上文)
     * 5. Resnet module(见下文)


### Compare Sigmoid with tanh
* 双曲线正切其实也是sigmoid函数的线性组合 
* $sigmoid = \frac{1}{1+e^{-x}}$
* $tanh = \frac{sinhx}{coshx}=\frac{e^x-e^{-x}}{e^x+e^{-x}}=1-\frac{2}{1+e^{-2x}}=1-2sigmoid(2x)$
* 他们函数图像的主要区别就是tanhx是均值为零的，而sigmoid由于均值不为0，会在梯度下降中出现zigzag现象，导致收敛变慢
* $\frac{d_o}{d_w}=\frac{d_o}{d_h}\frac{d_h}{d_w}=a(1-a)x$ 
* where $a=sigmoid(h),h=wx,a*(1-a)>0$ ,因此上式的梯度主要取决于x的符号，而x在输入层可能有正有负，但是在隐藏层中经过sigmoid激活函数后全为正数，因此，隐藏层w的更新要么全部往负向走，要么全部正向走（取决于最后一层 $\frac{d_{loss}}{d_y}$ 传递上来的值),不妨假设一共两个参数.
* 由于参数的梯度方向一致，要么同正，要么同负，因此更新方向只能为第三象限角度或第一象限角度，而若梯度的最优方向为第四象限角度，在这种情况下，每更新一次梯度，不管是同时变小(第三象限角度)还是同时变大(第四象限角度)，总是一个参数更接近最优状态，另一个参数远离最优状态，因此为了使参数尽快收敛到最优状态，出现交替向最优状态更新的现象，也就是zigzag现象。

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
### Transformer 参数量计算
* 我们来分析下 Transformer 模型的参数量，先假设序列中每个token 的向量维度 $d_{model}=D$,表大小是V,Multi-head个数是$H$,每个 head 中向量维度 $d_q=d_k=d_v=\frac{D}{H}$
* $softmax$前的Linear层不属于 Encoder 和 Decoder，它的参数量和层数、维度相关，并且线性层的参数量很容易计算，这里就忽略掉。我们重点看 Encoder 和 Decoder 的参数量。
* 先看下 Positional Encoding 和 Embedding，由于前者使用的是三角函数直接计算得到的，参数量为 0，后者出现在 Transformer 中的 3 个位置（Input Embedding、Output Embedding、Pre-softmax），但是参数共享，所以只有一份参数，也就是 $VD$
* Transformer 中有三处地方用到了 self-attention: Encoder 中的Multi-Head Attention、Decoder 中的Masked Multi-Head Attention、Decoder 中的Multi-Head Attention(Cross-Attention)。其实三者的模型结构完全相同，差别仅在于 $Q(query), K(Key), V(Value)$来自同一个序列还是两个序列，是否用 mask。因此参数量计算方法相同，并且向量维度都是$d_{model}，三者的参数量也相同。
* 同样的，Encoder 和 Decoder 中的FFNN 结构相同，参数量也相同。
* 分别看下 self-attention 和 FFNN 的参数量，先来看 self-attention，假设序列 $X\in R^{L*d_{model}}$其中L是序列长度，Q=K=V=X,回顾下scaled dot-product attention公式：$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d}})V $$
* 再回顾下论文中给出的 MHSA的计算公式，:$$MultiHead(Q,K,V)=Concat(head_1,head_2,...,head_H)W^O$$,$$head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)$$
* 其中 $W_i^Q\in R^{d_{model}d_q},W_i^K\in R^{d_{model}d_k},W_i^V\in R^{d_{model}d_v}$
* 现在来计算下参数量: $$H(W_i^Q+W_i^K+W_i^V)+W^O$$ , $$=H(3d_{model}d_k)+d^2_{model}$$, $$=3d_{model}(Hd_k)+d^2_{model}$$,$$=4d^2_{model}$$
* 具体实现的时候是，创建 $W^Q\in R^{d_{model}d_{model}}$... 这样可以清楚看到参数量是 $ 4d_{model}^2$
* 再来看下 FFNN，它对序列$X$中每个元素向量进行转换，$$FFNN(x)=max(0,xW_1+b_1)W_2+b_2$$

### 总结
* Transformer中涉及繁琐的矩阵计算，本质是用矩阵乘法衡量特征向量之间的相似度，理解了计算过程有助于加深理解网络设计原理。 比如 向量 $a*b=ab, a*b=-ab, $向量乘法表示相似度
* Encoder 过程$(word2vec + PE)* \frac {softmax(Q,K)*(V)}{\sqrt d_k}$, 其中QKV都是原始的Word Embedding分别经过三个不同的全联接层得到的。
     * 为什么除以$\sqrt d_k$因为$d_k$过大时候内积也会增加，softmax函数的梯度很小
     * 先解释：为什么当$d_k$较大时，向量内积容易取很大的值（借用原论文的注释）
     * 假设 query 和 key 向量中的元素都是相互独立的均值为 0，方差为 1 的随机变量，那么这两个向量的内积$q^Tk=\sum_{i=1}^{d_k}q_ik_i$的均值是0，方差是$d_k$
     * Prove: 
          * $E[q_i]=E[k_i]=0$, $Var(q_i)=Var(k_i)=1$
          * 由于$q_i,k_i$独立,$Cov(q_i,k_i)=E[(q_i-E[q_i])(k_i-E[k_i])]=E[q_ik_i]-E[q_i]E[k_i]=0 $
          * ==> $E[q_ik_i]=E[q_i]E[k_i]=0$
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
### Self-Attention 时间和空间复杂度
* 在讨论 self-attention 的时间和空间复杂度时，都会提到是 $O(N^2)$, $N$ 为序列长度
* 其实略高在 [AAAI 2021的Best paper, Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/17325) 中这个问题得到解决
* 我们来看下 scaled dot-product attention的时间和空间复杂度
* $$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d}})V $$
* 我们把上面的计算过程拆分 , $S=QK^T$, $S=\frac{S}{\sqrt d}$, $S=softmax(S)$ , $SV$ 只要我们分析出每个计算的复杂度，就可以得到整体计算的复杂度。
* $QK^T$ , where $Q\in R^{N*d}$, $K\in R^{N*d}$, 矩阵乘法的朴素算法时间复杂度是 $O(NdN)=O(N^2d)$, 至于空间复杂度，只看存储$QK^T$ 计算结果，复杂度是 $O(N^2)$, 但是也不要觉得这个数字很大，如果, $N<d$.其实存储 $Q$和 $V$ 比 $QV^T$更占显(内)存.除非是序列很长$N>>d$空间复杂度 $O(N^2)$才是瓶颈。
* 简单回顾下矩阵乘法 $C=AB, A\in R^{m*n}, B\in R^{n*l},C\in R^{m*l}$, 显而易见，3 个 for loop, 因此矩阵乘法时间复杂度 $O(mnl)$
'''python
C = np.zeros((m, l))

for i in range(m):
  for j in range(l):
    for k in range(n):
      C[i][j] += A[m][k] * B[k][n]
'''
* 由于 $QK^T\in R^{N*N}$, 因此 $\frac{QK^T}{\sqrt d}$时间复杂度是 $O(N^2)$ 
* 至于softmax 时间复杂度， 假设x是一维向量，
* $$softmax(x)=\frac{e^{x_i}}{\sum_i e^{x_i}}$$
'''python
def softmax(x):
    m_val = max(x)
    x = [i-m_val for i in x]
    x = [math.exp(i) for i in x]
    deno = sum(x)
    return [item / deno for item in x]

softmax([1,2,3])  # [0.0900, 0.2447, 0.6652]
'''
* 可以看到，上述计算过程时间复杂度是线性的$O(N)$, 因为$QK^T\in R^{N*N}$,softmax是按照行计算的，所以$softmax(R^{N*N})$的时间复杂度是$O(N^2)$
* 假设 $S=softmax(\frac{QK^T}{\sqrt d})\in R^{N*N}, V\in R^{N*d}$, 则SV的时间复杂度是 $O(N^2d)$
* 因此整个attention时间复杂度是 $O(N^2d+N^2+N^2+N^2d)=O(N^2d)$ 此时如果把向量维度d看作常数，则可以说 self-attention 的时间复杂度是序列长度的平方。
* 至于空间复杂度 存储 $QK^T$,$\frac{QK^T}{\sqrt d}$, 还是softmax(S),都是 $O(N^2)$, 最后存储 $SV\in R^{N*d}$的空间复杂度是 $O(Nd)$这样，整个Attention 空间复杂度可以看作 $O(N^2+Nd)$如果把向量维度d看作常数，则可以说 self-attention 的空间复杂度是序列长度的平方。P.S.(如果是Multi-head self-attention, 空间复杂度需要乘上head的个数)
## Transformer变种
### VIT(Vision Transformer)
* 处理数据的模式：
     * 加入直接把224*224像素的一张图片以像素作为基本token输入呢？ 那么一个序列长度会是224*224=50176(计算机很难处理长序列)
     * 因此[采用如下](https://en.wikipedia.org/wiki/Vision_transformer#/media/File:Vision_Transformer.gif)--将原图切割为16x16 pixels，以此作为基本单元输入网络。此时16*16的方块类似句子里的单词，整张图片类似句子。
### BERT(Bidirectional Encoder Representations from Transformers)
* BERT 主要是利用了Transformer的Encoder并作出相应改进
* Embedding
     * Embedding由三种Embedding求和而成：
     * Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务
          * 通过建立字向量表将每个字转换成一个一维向量，作为模型输入。特别的，英文词汇会做更细粒度的切分，比如playing 或切割成 play 和 ##ing，中文目前尚未对输入文本进行分词，直接对单子构成为本的输入单位。将词切割成更细粒度的 Word Piece 是为了解决未登录词的常见方法。
          * 假如输入文本 ”I like dog“。下图则为 Token Embeddings 层实现过程。输入文本在送入 Token Embeddings 层之前要先进性 tokenization 处理，且两个特殊的 Token 会插入在文本开头 [CLS] 和结尾 [SEP]。[CLS]表示该特征用于分类模型，对非分类模型，该符号可以省去。[SEP]表示分句符号，用于断开输入语料中的两个句子。
          * Bert 在处理英文文本时只需要 30522 个词，Token Embeddings 层会将每个词转换成 768 维向量，例子中 5 个Token 会被转换成一个 (6, 768) 的矩阵或 (1, 6, 768) 的张量。
     * Segment Embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务
          * Bert 能够处理句子对的分类任务，这类任务就是判断两个文本是否是语义相似的。句子对中的两个句子被简单的拼接在一起后送入模型中，Bert 如何区分一个句子对是两个句子呢？答案就是 Segment Embeddings。
          * Segement Embeddings 层有两种向量表示，前一个向量是把 0 赋值给第一个句子的各个 Token，后一个向量是把1赋值给各个 Token，问答系统等任务要预测下一句，因此输入是有关联的句子。而文本分类只有一个句子，那么 Segement embeddings 就全部是 0。
     * Position Embeddings和之前文章中的Transformer不一样，不是三角函数而是学习出来的
          * 由于出现在文本不同位置的字/词所携带的语义信息存在差异(如 ”你爱我“ 和 ”我爱你“)，你和我虽然都和爱字很接近，但是位置不同，表示的含义不同。
          * 在 RNN 中，第二个 ”I“ 和 第一个 ”I“ 表达的意义不一样，因为它们的隐状态不一样。对第二个 ”I“ 来说，隐状态经过 ”I think therefore“ 三个词，包含了前面三个词的信息，而第一个 ”I“ 只是一个初始值。因此，RNN 的隐状态保证在不同位置上相同的词有不同的输出向量表示。
          * RNN 能够让模型隐式的编码序列的顺序信息，相比之下，Transformer 的自注意力层 (Self-Attention) 对不同位置出现相同词给出的是同样的输出向量表示。尽管 Transformer 中两个 ”I“ 在不同的位置上，但是表示的向量是相同的。
          * Transformer 中通过植入关于 Token 的相对位置或者绝对位置信息来表示序列的顺序信息。作者测试用学习的方法来得到 Position Embeddings，最终发现固定位置和相对位置效果差不多，所以最后用的是固定位置的，而正弦可以处理更长的 Sequence，且可以用前面位置的值线性表示后面的位置。
          * BERT 中处理的最长序列是 512 个 Token，长度超过 512 会被截取，BERT 在各个位置上学习一个向量来表示序列顺序的信息编码进来，这意味着 Position Embeddings 实际上是一个 (512, 768) 的 lookup 表，表第一行是代表第一个序列的每个位置，第二行代表序列第二个位置。
          * 最后，BERT 模型将 Token Embeddings (1, n, 768) + Segment Embeddings(1, n, 768) + Position Embeddings(1, n, 768) 求和的方式得到一个 Embedding(1, n, 768) 作为模型的输入。
     * [CLS]的作用
          * BERT在第一句前会加一个[CLS]标志，最后一层该位对应向量可以作为整句话的语义表示，从而用于下游的分类任务等。因为与文本中已有的其它词相比，这个无明显语义信息的符号会更“公平”地融合文本中各个词的语义信息，从而更好的表示整句话的语义。 具体来说，self-attention是用文本中的其它词来增强目标词的语义表示，但是目标词本身的语义还是会占主要部分的，因此，经过BERT的12层（BERT-base为例），每次词的embedding融合了所有词的信息，可以去更好的表示自己的语义。而[CLS]位本身没有语义，经过12层，句子级别的向量，相比其他正常词，可以更好的表征句子语义。
* BERT的训练包含pre-train和fine-tune两个阶段。pre-train阶段模型是在无标注的标签数据上进行训练，fine-tune阶段，BERT模型首先是被pre-train模型参数初始化，然后所有的参数会用下游的有标注的数据进行训。
     * 1. 预训练
     * BERT是一个多任务模型，它的预训练（Pre-training）任务是由两个自监督任务组成，即MLM和NSP
     * MLM是指在训练的时候随即从输入语料上mask掉一些单词，然后通过的上下文预测该单词，该任务非常像我们在中学时期经常做的完形填空。正如传统的语言模型算法和RNN匹配那样，MLM的这个性质和Transformer的结构是非常匹配的。在BERT的实验中，15%的WordPiece Token会被随机Mask掉。在训练模型时，一个句子会被多次喂到模型中用于参数学习，但是Google并没有在每次都mask掉这些单词，而是在确定要Mask掉的单词之后，做以下处理。
          * 80%的时候会直接替换为[Mask]，将句子 "my dog is cute" 转换为句子 "my dog is [Mask]"。
          * 10%的时候将其替换为其它任意单词，将单词 "cute" 替换成另一个随机词，例如 "apple"。将句子 "my dog is cute" 转换为句子 "my dog is apple"。
          * 10%的时候会保留原始Token，例如保持句子为 "my dog is cute" 不变。
          * 这么做的原因是如果句子中的某个Token 100%都会被mask掉，那么在fine-tuning的时候模型就会有一些没有见过的单词。加入随机Token的原因是因为Transformer要保持对每个输入token的分布式表征，否则模型就会记住这个[mask]是token ’cute‘。至于单词带来的负面影响，因为一个单词被随机替换掉的概率只有15%*10% =1.5%，这个负面影响其实是可以忽略不计的。 另外文章指出每次只预测15%的单词，因此模型收敛的比较慢。
     * Next Sentence Prediction（NSP）的任务是判断句子B是否是句子A的下文。如果是的话输出’IsNext‘，否则输出’NotNext‘。训练数据的生成方式是从平行语料中随机抽取的连续两句话，其中50%保留抽取的两句话，它们符合IsNext关系，另外50%的第二句话是随机从预料中提取的，它们的关系是NotNext的。这个关系保存中的[CLS]符号中。
          * 输入 = [CLS] 我 喜欢 玩 [Mask] 联盟 [SEP] 我 最 擅长 的 [Mask] 是 亚索 [SEP]
          * 类别 = IsNext
          * 输入 = [CLS] 我 喜欢 玩 [Mask] 联盟 [SEP] 今天 天气 很 [Mask] [SEP]
          * 类别 = NotNext
          * (P.S.)在此后的研究（论文《Crosslingual language model pretraining》等）中发现，NSP任务可能并不是必要的，消除NSP损失在下游任务的性能上能够与原始BERT持平或略有提高。这可能是由于BERT以单句子为单位输入，模型无法学习到词之间的远程依赖关系。针对这一点，后续的RoBERTa、ALBERT、spanBERT都移去了NSP任务。
          * BERT预训练模型最多只能输入512个词，这是因为在BERT中，Token，Position，Segment Embeddings 都是通过学习来得到的。在直接使用Google 的BERT预训练模型时，输入最多512个词（还要除掉[CLS]和[SEP]），最多两个句子合成一句。这之外的词和句子会没有对应的embedding。
          * 如果有足够的硬件资源自己重新训练BERT，可以更改 BERT config，设置更大max_position_embeddings 和 type_vocab_size值去满足自己的需求。
     * 2. 微调
     * 在海量的语料上训练完BERT之后，便可以将其应用到NLP的各个任务中了。 微调(Fine-Tuning)的任务包括：基于句子对的分类任务，基于单个句子的分类任务，问答任务，命名实体识别等。
     * 基于句子对的分类任务：
          * MNLI：给定一个前提 (Premise) ，根据这个前提去推断假设 (Hypothesis) 与前提的关系。该任务的关系分为三种，蕴含关系 (Entailment)、矛盾关系 (Contradiction) 以及中立关系 (Neutral)。所以这个问题本质上是一个分类问题，我们需要做的是去发掘前提和假设这两个句子对之间的交互信息。
          * QQP：基于Quora，判断 Quora 上的两个问题句是否表示的是一样的意思。
          * QNLI：用于判断文本是否包含问题的答案，类似于我们做阅读理解定位问题所在的段落。
          * STS-B：预测两个句子的相似性，包括5个级别。
          * MRPC：也是判断两个句子是否是等价的。
          * RTE：类似于MNLI，但是只是对蕴含关系的二分类判断，而且数据集更小。
          * SWAG：从四个句子中选择为可能为前句下文的那个。
     * 基于单个句子的分类任务
          * SST-2：电影评价的情感分析。
          * CoLA：句子语义判断，是否是可接受的（Acceptable）。
     * 问答任务
          * SQuAD v1.1：给定一个句子（通常是一个问题）和一段描述文本，输出这个问题的答案，类似于做阅读理解的简答题。
     * 命名实体识别
          * CoNLL-2003 NER：判断一个句子中的单词是不是Person，Organization，Location，Miscellaneous或者other（无命名实体）
* BERT的优缺点
     * 优点
          * BERT 相较于原来的 RNN、LSTM 可以做到并发执行，同时提取词在句子中的关系特征，并且能在多个不同层次提取关系特征，进而更全面反映句子语义。
          * 相较于 word2vec，其又能根据句子上下文获取词义，从而避免歧义出现。
     * 缺点
          * 模型参数太多，而且模型太大，少量数据训练时，容易过拟合。
          * BERT的NSP任务效果不明显，MLM存在和下游任务mismathch的情况。
          * BERT对生成式任务和长序列建模支持不好。

## Attentions
* 自注意力机制在处理单个序列时表现出色，但对于多个输入序列时效果不佳;
* 交叉注意力机制适用于多个输入序列之间的关联建模，但需要额外的计算资源和参数数量;
* 多头注意力机制可以同时捕捉多种特征信息，提高模型性能和泛化能力，但需要更多的计算和存储资源。


## GPT(Generative pre-trained transformer)
* text to text: GPT 利用decoder是接收到问题之后。根据问题生成第一个字的概率分布，sample最高概率的之后拿取第一个字，根据第一个字生成第二个字的概率分布....
* 但是图像不可以这样
     * 并且256*256的图片切割成序列输入，再像素化输出的话速度很慢
     * 文字基本由正确答案的，但是生成图片的时候，如：”画一只正在奔跑的狗“，如果一个pixel by pixel 生成的话 难免会出现错乱。所以伴随文字会有一个normal distribution 一起输入（所有生成模型需要攻克这个问题VAE,Diffusion model, Flow-based Generative Model, GAN）

## Difference between GPT and BERT
* 1. 任务类型：GPT和BERT的主要区别在于它们所执行的任务类型。GPT是一种生成式模型，专注于生成类似人类写作的文本。这意味着它可以应用于诸如机器翻译、文本摘要、问答等任务，在这些任务中，模型需要生成与目标语言匹配的文本。 相比之下，BERT是一种预训练模型，专注于理解文本中的语义关系。这意味着它适用于诸如情感分析、实体识别、关系提取等任务，在这些任务中，模型需要理解并提取文本中的结构化信息。
* 2. 输入顺序: GPT是一个从左到右的单向模型，这意味着它只能利用当前位置之前的上下文信息。这种单向性使得GPT在处理某些任务时可能会遇到上下文信息的限制。而BERT是一种双向模型，可以同时查看输入文本的前后部分。这意味着BERT在处理需要理解整个句子或段落的任务时具有优势，因为它能够同时分析输入文本的前后关系。
* 3. 训练数据: GPT使用更广泛的训练数据，包括维基百科和网页文本。这种广泛的训练数据使得GPT在处理各种不同主题和风格的文本时具有优势。然而，这并不意味着GPT在特定领域的任务中表现不佳，它可以通过在特定数据集上进行微调来适应特定任务。相比之下，BERT使用更具体的语言任务作为训练数据，如问答和阅读理解。这种针对性训练使得BERT在处理特定类型的任务时具有优势，因为它的训练目标是理解并解决这些特定任务。
* 4. 预训练方式: GPT采用自回归预训练方法，即从左到右生成下一个单词或句子。这种自回归方法使得GPT在文本生成任务中表现出色，因为它能够逐步生成与目标语言匹配的文本。 而BERT则采用双向预训练方法，即同时预测文本中的上下文信息。这种双向方法使得BERT在理解文本的语义关系方面具有优势，因为它能够在分析文本时同时考虑前后文信息。
* 5. 应用场景: 由于GPT专注于生成文本任务，它在机器翻译、摘要生成和问答等任务中表现良好。这些任务需要模型能够生成与目标语言匹配的文本，而GPT正是通过自回归预训练方法来实现这一目标。相比之下，BERT在情感分析、实体识别和关系提取等任务中表现出色。这些任务需要模型理解文本中的语义关系，而BERT通过双向预训练方法来同时分析前后文信息，从而在这些任务中取得了显著成果。
* 总结：GPT和BERT是两大强大的预训练语言模型，各自拥有独特的特点和应用场景。GPT专注于生成文本任务，适用于机器翻译、摘要生成和问答等任务；而BERT专注于理解文本中的语义关系，适用于情感分析、实体识别和关系提取等任务。在选择模型时，应考虑特定任务的性质和需求，以便选择最适合的模型并优化其性能。


## MAE( Masked Autoencoders Are Scalable Vision Learners)
* 得益于硬件发展与算力的支持，模型越来越大，大模型由于参数量众多，因此也很容易过拟合一般规模的数据集。于是，就需要更大量的数据，而这么大量的标注数据人工成本很高
* 开辟出了新的玩法：自监督预训练。其中，较为常见的一种模式就是 masked autoencoding，这种在 NLP 尤为火热，大名鼎鼎的 BERT 在预训练中就是这么玩的：以一定比例 mask 掉输入文本中的一些部分，让模型去预测这批被 mask 掉的内容。这样，利用数据本身就可以作为监督(模型要预测的目标来源于数据本身，并非人工构造)，无需复杂的人工标注。同时，使用大量的数据让拥有大规模参数量的模型能够学到通用的知识，从而拥有良好的泛化能力。
* KaiMing在论文中也谈到 Why Masked Autoencoding In CV Lags Behind NLP?
* * progress of autoencoding methods in vision lags behind NLP.
We ask: what makes masked autoencoding different between vision and language ?
* i). 架构(architecture)差异
* CV 和 NLP 的网络架构不一致，前者在过去一直被 CNN 统治，它基于方正的局部窗口来操作，不方便集成像 mask token 以及 position embedding 这类带有指示性的可学习因子。不过，这个 gap 现在看来应该可以解决了，因为 ViT(Vision Transformer) 已经在 CV 界大肆虐杀，风头很猛..

* ii). 信息密度(information density)不同
* 图像和语言的信息密度是不一样的。语言是人类创造的，本身就是高度语义和信息密集的，于是将句子中的少量词语抹去再让模型去预测这些被抹去的词本身就已经是比较困难的任务了；而对于图像则相反，它在空间上是高度冗余的，对于图片中的某个部分，模型很容易由其相邻的图像块推断出来(你想想看插值的道理)，不需要大量的高级语义信息。
因此，在 CV 中，如果要使用 mask 这种玩法，就应该要 mask 掉图片中的较多的部分，这样才能使任务本身具有足够的挑战性，从而使模型学到良好的潜在特征表示。

* iii). 解码的目标不一致
* CV 和 NLP 在解码器的设计上应该有不一样的考虑：NLP 解码输出的是对应被 mask 掉的词语，本身包含了丰富的语义信息；而 CV 要重建的是被 mask 掉的图像块(像素值)，是低语义的。
* 因此，NLP 的解码器可以很简单，比如 BERT，严格来说它并没有解码器，最后用 MLP 也可以搞定。因为来自编码器的特征也是高度语义的，与需要解码的目标之间的 gap 较小；而 CV 的解码器设计则需要“谨慎”考虑了，因为它要将来自编码器的高级语义特征解码至低级语义层级。


### Reference
* [图解 Transformers](https://zhuanlan.zhihu.com/p/654051912)
* [Dive deepl into Transformer](https://www.linkedin.com/pulse/deep-dive-positional-encodings-transformer-neural-network-ajay-taneja#:~:text=Positional%20Encodings%20can%20be%20looked,vector%20representation%20of%20the%20input.)
* [why trigonometric function PE?](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)
* [Positional Encoding](https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6)
* [大白话讲解 Transformer](https://zhuanlan.zhihu.com/p/264468193?utm_medium=social&utm_oi=1123713438710718464&utm_psn=1737408413846585345&utm_source=wechat_session)
* [从反向传播推导到梯度消失and爆炸的原因及解决](https://zhuanlan.zhihu.com/p/76772734)
* [BERT](https://zhuanlan.zhihu.com/p/403495863)
* [参数量和时间空间复杂度](https://zhuanlan.zhihu.com/p/661804092)
* [PE](https://blog.csdn.net/weixin_44012382/article/details/113059423)
* [Lost of Gradient RNN vs DNN](https://zhuanlan.zhihu.com/p/673787298)
* [LSTM reduce lost of gradient](https://zhuanlan.zhihu.com/p/109519044)

* [Bert 理论](https://blog.csdn.net/yjw123456/article/details/120211601)