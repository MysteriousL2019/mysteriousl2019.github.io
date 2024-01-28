---
title: How to Read Paper in New Area I
author: Fangzheng
date: 2023-01-19 00:14:00 +0800
categories: [Artificial Intelligence ]
tags: [algorithm ]
# pin: true
mermaid: true  #code模块
# comments: true
math: true
# img_cdn: https://github.com/MysteriousL2019/mysteriousl2019.github.io/tree/master/assets/img/
---

# 高质量的论文期刊
* CVPR, ECCV, ICCV, AAAI, NIPS, ICLR, ICML

# 高引用量的论文
* 同行间认可，参考，借鉴的论文

# 知名团队
* Yoshua Bengio, Yann LeCun, Geoffrey Hinton, Andrew Ng

# 重视有代码的论文
* paper with code

# 在新领域的时候要阅读综述性的文章

# 如何寻找论文
* 知网：用于寻找优质的综述，快速入门
* 百度学术，Google Scholar
* arXiv：https://arxiv.org/
* 顶会（AI）：CVPR, ECCV, ICCV, AAAI, NIPS, ICLR, ICML

# 关于是否优质，看IF
* IF(impact factor，影响因子): 
* JCR(Journal Citation Reports 期刊引证报告)
* SCI分区:1. JCR 方式，一、二、三、四各区各占25%，因为搜索引擎(国际公认方式) 2.中科院方式（中国认可），一区为前5%， 二为5-20%， 三为20%-50%，四为50-100%

# 绕过科研论文收费的网站
* sci-hub,使用方式是搜索url，DOI（文献数字身份），PMID（唯一标识码）
* https://sci-hub.tw
* https://sci-hub.si
* https://sci-hub.se

# 整理论文方式
* 时间-作者-题目  或者  时间-关键词-题目
* 在校学生以学习知识为主，要求读论文面面俱到
* 工程师则注重算法实现，借鉴为主

# 读论文方法
* 读三遍
* 1.泛读：快速浏览，把握要点。 读标题、摘要、所有小标题、图表  ** 效果自测  1.论文要解决什么问题  2.论文使用了什么方法  3.达到了什么效果**
* 2.精读 ：找出关键内容
* 3.总结：找出创新点，启发点等重要信息

# 终极问题
* 是谁：论文提出什么方法
* 从哪来： 要解决什么问题
* 到哪去：达到什么效果
* 可借鉴的地方

# Short Paper: 
* 短文是被正会录用的论文，长度为 4 页（录用后可以增加一页），整体给人感受就是一个比较 focus 的贡献，或者是对某个现象/问题的分析以促进未来进一步的工作，比较经典的像 ACL 的 Energy and Policy Considerations for Deep Learning in NLP，讨论了预训练模型带来的对环境的影响，也后续催化了一系列 Green AI / NLP 的研究，目前已经 1100 + 引用了。和长文相比，短文的长度会限制其系统性。
# Findings Paper:
* Findings 是在 EMNLP 2020 提出，接收略微 Miss 正会 bar 的 paper 的一项类别，经过 Peer Review 并且算正式出版物被 ACL anthology 收录，所以质量还是有保障的。唯一遗憾的一点是无法在正会上进行 Poster 或者 Oral 的展示，但这两年各大会议也为 Findings 提供了 Poster 的展示环节。官方 Blog 对 Findings 的解释如下：
* Papers that extend the state of the art on a particular focused task, but have few novel insights or Findings of broader applicability to the wider EMNLP community;
* Papers that have well-executed, novel experiments and present thorough analyses and Findings, but using methods that are not thought to be sufficiently “novel”;   
* 简而言之，大多是审稿人认为 Novelty 不足但是 well-written and solid 的 paper，考虑到 novelty 很多时候取决于审稿人的 taste 和对 novelty 的理解，所以被 Findings 收录同样可以看做是对 paper 质量的一种肯定。就我个人而言， Findings Paper 可能是更为简洁有效的对某个问题的解决方案，经典的 Paper 像发表于 Findings of EMNLP 2020 的 TinyBERT，同样有着很高的影响力（500+ citations, GitHub 2.1k stars）。不过对于有毕业要求的同学来说，目前短文和 Findings Paper 不算 CCF 的分类要求（必须是主会长文），所以如果需要满足学位的要求的话，在 ARR 的情况下可以考虑 revise 后再投一轮 ARR 试试。
# Workshop Paper:
* Workshop 我个人会倾向于把它翻译成研讨会，一般会是收录一个特定 topic 的论文，因而也可以和很多小同行进行比较深入的交流，这是主会有些时候都不太遇得到的，比较经典的 Workshop 像 WMT、Rep4NLP 等。另外 ACL 和 NAACL 这几年都会办 Student Research Workshop（SRW），并且会设置相应的 mentoring program，来指导比较 junior 的同学投稿，这个 workshop 的 reviewer 多半会比较 constructive。第一次投稿且组里没有比较资深的老师或者同学的，可以尝试投稿类似的 workshop 获取审稿建议，并且可以选择不收录，再充分吸取建议修改之后再投稿到正式的会议，能够大幅提升中稿率。同时，被 workshop 接收也是一种正反馈，对 reward sparse 的研究生来说，是有很大促进作用滴！

# baseline & pipeline
* baseline，基线，表示数据预处理: 特征工程，模型训练，评估预测，已经拉通，完成了一个比较基础的实现。后面的过程都是在尝试提示性能和效果。
后面可以以此baseline为基准，逐步调整预处理方法，特征，模型选择和参数调优，以及一些如正负样本不平衡，loss等的修改和尝试。一个机器学习的工程问题被实现。不少比赛中，选手们都会抛出一个baseline，后面大家各凭本事，，也有一些选手抛出一个更高分的baseline，好吧，那学习一波new baseline再上分。baseline的构建也表示着，业务方或者出题方的业务建模，好的业务建模是对工程问题的有效理解，baseline则表示对工程问题的有效解决方案，但不是最有效方案。

* pipeline ,流水线，在深度学习中表示 -数据读取，-数据预处理 -创建模型 -评估模型结果 -模型调参