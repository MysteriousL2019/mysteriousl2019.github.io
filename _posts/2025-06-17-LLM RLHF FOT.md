---
title: 「Paper Reading」 LLM RLHF 2024论文（三十九）FoT
author: Fangzheng
date: 2025-06-17 19:06:00 +0800
categories: [Paper Reading, RLHF]
tags: [Algorithm,Paper Reading]
# pin: true
mermaid: true  #code模块
comments: true

math: true
# img_cdn: https://github.com/MysteriousL2019/mysteriousl2019.github.io/tree/master/assets/img/
---
# LLM RLHF 2024论文（三十九）FoT

论文标题[Forest-of-Thought]: Scaling Test-Time Compute for Enhancing LLM Reasoning，[原文](https://arxiv.org/abs/2412.09078)，发表于ICML 2025。

LLM reasoning经常使用思维链（CoT）或思维树（ToT），来分解问题，增强推理，这种方法通常只进行一次推理过程，可能无法重新处理有缺陷的路径，从而影响准确性。为了解决这一限制，本文提出了思维森林（Forest-of-Thought, FoT）的学习框架，整合了多个推理树，利用集体决策来解决复杂的推理问题。FoT采用[稀疏激活策略]来选择最相关的推理路径，此外，还引入了一种动态自我修正策略，实现实时的错误修正，来优化资源的使用。实验结果表明，FoT框架显著增强了LLM的推理能力，能够以更高的精度和效率解决复杂任务。

**背景**

大模型应用中的数学计算、逻辑推理之类的任务可以用CoT的方式，将数学推理任务step by step地拆解，并且进一步将其建模成树的结构（ToT），并且使用MCTS的方式来增强inference的效果。

![](https://pic4.zhimg.com/v2-a16af4c087c9b6e728d48e00817302cd_1440w.jpg)

其中，ToT结构经常结合[MCTS算法](https://zhida.zhihu.com/search?content_id=254191916&content_type=Article&match_order=1&q=MCTS%E7%AE%97%E6%B3%95&zhida_source=entity)使用，MCTS是一种搜索算法，它利用蒙特卡洛模拟来进行搜索MCTS 维护了一个搜索树，这个搜索树记录了之前的搜索过的（状态动作序列）轨迹和相关统计信息（平均回报与访问次数），从而可以兼顾探索（explore）和利用（exploit），以较低的开销搜索到较好的结果。

![](https://pic3.zhimg.com/v2-402c2dc986aecdab18d30ec55d7e8034_1440w.jpg)

**算法**

现有的ToT类方法将复杂问题分解为更简单的子问题来进行推理。然而，在分解问题的过程中，可能会由于中间步骤出错，从而导致最终答案不正确。但这类方法一旦完成了一条推理路径，如果初始路径存在缺陷，往往会过早放弃掉路径而没有充分地探索，从而损害了解答的准确性。

因此，本文提出FoT算法，通过引入多个推理树进行独立决策，并采用稀疏激活策略来过滤关键树的结果，构建思维森林来增强LLM的推理能力。FoT利用集体智能来弥补个体的不足，从而提高模型从多个角度进行推理的能力。如图所示：

![](https://picx.zhimg.com/v2-9fa45e06860008577ba7936c9c80a76f_1440w.jpg)

FoT算法会使用n棵推理树进行决策，每棵树的根节点使用了数据增强，并且在推理过程中设置了稀疏激活函数，决定每棵树是否被激活：

![](https://pic4.zhimg.com/v2-def526f5af7a79d9a53a745604e832ef_1440w.jpg)

对于每棵推理树，每一层选择评估分数最大的节点，进行后续的推理，扩展后续的子节点。如果一层的节点都不能输出有效值，会停止扩展这一棵树，即稀疏激活值设置为0:

![](https://picx.zhimg.com/v2-e1925169239ae7e4fb9fdd7de589a02b_1440w.jpg)

后续过程中，只有激活的树会参与最终的决策。

此外，FoT还使用了数据增强的方式，从公开可用的数据集中收集和构建了一个知识库，以支持模型的推理过程：

![](https://pic1.zhimg.com/v2-1de47a0eb1f344c2bc5f03804ff3b856_1440w.jpg)

FoT还采用了提前终止的搜索方式，一旦找到解决方案，搜索就会立即停止，避免冗余计算，一旦某个分支与ground truth匹配，也会停止对无关路径的进一步探索，提高整体效率。

此外，FoT还使用了[动态自我纠正]的策略，通过动态评估每个推理步骤，特别是通过监控预测的logits分数来评估推理结果的质量。当模型的分数低于预定义的阈值时，会触发校正机制，以及时检测和修正错误。FoT还融入了预定义的数学规则，通过将这些规则嵌入到推理框架中，模型可以在检测到错误时立即进行修正。例如，在[24点游戏]中，模型可以验证输出中的剩余数字是否来源于输入数字，从而实现快速的错误检测和修正。

动态自我纠正的具体流程如下：

![](https://pica.zhimg.com/v2-05371882cb00a29299f190c23420e194_1440w.jpg)

在每棵树都给出结果后，FoT算法会经过majority voting和专家评估，以确定最佳答案。对于复杂的推理任务，如果大多数树产生的结果不一致，会有一个LLM专家比较不同树的推理过程和结果，并基于其专业知识和经验做出最终决策。

FoT的完整算法流程如下：

![](https://pic1.zhimg.com/v2-1c32ce2d6e62add3b90f81d97908e452_1440w.jpg)

**实验**

在24点游戏、[GSM8K](https://zhida.zhihu.com/search?content_id=254191916&content_type=Article&match_order=1&q=GSM8K&zhida_source=entity)和[MATH](https://zhida.zhihu.com/search?content_id=254191916&content_type=Article&match_order=1&q=MATH&zhida_source=entity)等任务上进行实验，使用Llama3-8B-Instruct、Mistral-7B和GLM-4-9B等模型作为基础模型，对比CoT，ToT等算法。

24点上的实验结果：

![](https://pica.zhimg.com/v2-3e6337fc9aa5a1f9248657ef5b93fe26_1440w.jpg)

FoT对比ToT：

![](https://pica.zhimg.com/v2-9929c747151eff22c65bb0e725c7bc5e_1440w.jpg)

消融实验的结果表明了各个添加模块的有效性：

![](https://picx.zhimg.com/v2-5e2f27ebab62a529980e96bd48232c8d_1440w.jpg)

GSM8K上的实验结果对比：

![](https://pic3.zhimg.com/v2-d3b21b32fb94e4dc9974de75de2c024e_1440w.jpg)

动态自我纠正阈值的对比：

![](https://pica.zhimg.com/v2-9083d6798041bb227c19403a2a08b3c0_1440w.jpg)

Scaling law的实验，随着FoT中激活的子树数量增加，模型准确率显著提高：

![](https://pic2.zhimg.com/v2-8329f260fccdd39407dbde6d1a5424ab_1440w.jpg)

Math任务上的实验结果：

![](https://pica.zhimg.com/v2-7b2240ebc141f4e81aae0aa4d22ff424_1440w.jpg)

对比不同决策策略的效果，多数投票，专家LLM，和FoT使用的二者结合的方式：

![](https://pic1.zhimg.com/v2-70124fc66a0a602d2f83565350d83700_1440w.jpg)