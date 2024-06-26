---
title: High frequency models in Data Science in Company
author: Fangzheng
date: 2023-6-10 17:19:00 +0800
categories: [Data Science]
tags: [machine learning]
# pin: false
mermaid: true  #code模块
# comments: true
math: true
---
# What is Machine Learning
* Joshua Gans和Avi Goldfarb在《预测机器》一书中所说，“人工智能的新浪潮实际上并没有给我们带来智能，而是智能的关键组成部分 - 预测”。你可以用机器学习做各种美好的事情。唯一的要求是将你的问题框定为预测问题。想从英语翻译成葡萄牙语吗？然后构建一个 ML 模型，在给定英语句子时预测葡萄牙语句子。想要识别人脸？然后创建一个 ML 模型，用于预测图片子部分中是否存在人脸。
* 然而，ML不是灵丹妙药。它可以在严格的边界下创造奇迹，但如果它的数据稍微偏离模型习惯的东西，它仍然会惨败。举另一个来自预测机器的例子，“在许多行业中，低价格与低销售额有关。例如，在酒店业，旅游旺季以外的价格较低，需求最高且酒店客满时价格较高。鉴于这些数据，一个天真的预测可能表明，提高价格将导致更多的房间售出。
* 机器学习使用变量之间的关联来预测变量。只要您不更改用于进行预测的变量，它就会非常出色地工作。这完全违背了将预测ML用于大多数涉及干预的决策的目的。
* 事实上，大多数数据科学家对ML了解很多，但对因果推理知之甚少，这导致大量的ML模型被部署在对手头的任务没有用的地方。公司的主要目标之一是增加销售或使用量。然而，仅预测销售的 ML 模型通常无用 - 如果不是有害的话 - 为此目的。这个模型甚至可能得出一些无稽之谈的结论，例如在高销量与高价格相关的示例中。然而，你会惊讶于有多少公司实施预测ML模型，而他们心中的目标与预测无关。

# What is Causal Inference
* 干预 Intervention
* 反事实 Counterfactual
* 传统的深度学习和机器学习 是寻找association.  注意区别相关关系与因果关系。相关关系：看到街上人们带伞，于是预测今天要下雨，这只是相关关系。人们带伞并不会导致下雨。根据与流感相关的海量词条搜索记录，Google通过big data可以很快的预测流行病的地域传播，这也只是相关关系。如果仅仅预测感兴趣，相关关系就足够了。如果需要推断变量之间的因果关系，则计量分析必须建立在经济学理论的基础上。但实际情况更为复杂。1.存在逆向因果（双向因果）：即X导致Y，但Y也会反过来变化X。
* 经济萧条引发内战-》内战也会进一步促进经济萧条。

# 统计学知识
## 描述统计
* 均值,中位数, 方差, 百分位数
## 假设检验
* 原假设, T test , Z test , ANOVA 等
## 概率论
* 正态分布, 中心极限定, 条件概率, 贝叶斯概率等
## 数据实验
* AB test( 原理, 样本量计算, MDE)
* DID实验(平行检验, PSM等等)
# Machine Learning part
## 1.正则化为什么可以避免过拟合？
* 正规化是防止过拟合的一种重要技巧。正则化通过降低模型的复杂性， 达到避免过拟合的问题。
* 这里的降低模型的复杂性可以理解为：
* L1将很多权重变成0，这样起到作用的因素就会减少。
* L2使权重都趋于0，这样就不会有某个权重占比特别大。
* 那为什么模型的复杂性是用权重来调节的？
* 因为过拟合的时候，拟合函数的系数往往非常大。越是复杂的模型，越是尝试对所有样本进行拟合，需要顾忌每一个点，包括异常点。这就会造成在较小的区间中产生较大的波动，这个较大的波动也会反映在这个区间的导数比较大。
* 只有越大的参数才可能产生较大的导数。因此参数越小，模型就越简单。
* 而加入正则化项就是在原来目标函数的基础上加入了约束。当目标函数的等高线和L1，L2范数函数第一次相交时，得到最优解。
## 2、 L1和L2公式原理
* L1和L2都是在求loss的时候，加入到loss的求解过程中，得到被惩罚后的loss，再让loss对权重求导，得到更新值，最后原权重减去学习率和更新值的乘积，最终得到权重的更新值，完成反向传播过程，进行新一轮的训练。
* 按上述思路，可得
* L1正则求Loss
* L1正则化是指权值向量ww中各个元素的绝对值之和。其中L是带有绝对值符号的函数，因此L是不完全可微的。
* $ L = L(w) + \lambda \sum_{1}^{n}|w_i| $
* $ \frac{\partial L}{\partial w_i}  = \frac{\partial L(w)}{\partial w_i} + \lambda sign(w_i) $
* 更新权重，其中 $\lambda$ 表示正则化系数， $ \eta$ 表示学习率，也是可以动态变化的，这里不做过多展开。
* $w_i = w_i - \eta (\frac{\partial L(w_i)}{\partial w_i}+\lambda sign(w_i))$
* 因为是权重相加，会比较大，最后到$w_i$就会更加趋近于0，从而达到惩罚的效果。
* L2正则化是模型各个参数的平方和的开方值。
* $L = L(w)+ \lambda \sum_{1}^{n}w_i^2$
* $\frac{L}{w_i}=\frac{\partial L(w)}{\partial w_i} +2 \lambda w_i$
* $w_i = w_i - \eta (\frac{\partial L(w)}{\partial w_i} +2 \lambda w_i)$
* 而这也会让$w_i$越来越小，这里称之为权重衰减。
* L1正则化和L2正则化的作用：
* 1. L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择；
* 2. L2正则化可以防止模型过拟合，在一定程度上，L1也可以防止过拟合，提升模型的泛化能力；
* 3. L1（拉格朗日）正则假设参数的先验分布是Laplace分布，可以保证模型的稀疏性，也就是某些参数等于0；
* 4. L2（岭回归）正则假设参数的先验分布是Gaussian分布，可以保证模型的稳定性，也就是参数的值不会太大或太小。
* 在实际使用中，如果特征是高维稀疏的，则使用L1正则；如果特征是低维稠密的，则使用L2正则
* 5、L1和L2正则先验分别服从什么分布 ？
* L1和L2正则先验分别服从什么分布，L1是拉普拉斯分布，L2是高斯分布。