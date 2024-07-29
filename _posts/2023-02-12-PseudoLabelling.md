---
title: Pseudo-Labelling
author: Fangzheng
date: 2023-02-12 17:06:00 +0800
categories: [Semi-supervised Learning, Artificial Intelligence]
tags: [Algorithm ]
# pin: true
mermaid: true  #code模块
comments: true

math: true
# img_cdn: https://github.com/MysteriousL2019/mysteriousl2019.github.io/tree/master/assets/img/
---
### background
* 大数据时代中，在推荐、广告领域样本的获取从来都不是问题，似乎适用于小样本学习的伪标签技术渐渐淡出了人们的视野，但实际上在样本及其珍贵的金融、医疗图像、安全等领域，伪标签学习是一把锋利的匕首，简单而有效。
## definition of Pseudo-Labelling
* 伪标签的定义来自于半监督学习，半监督学习的核心思想是通过借助无标签的数据来提升有监督过程中的模型性能。
* 举个简单的半监督学习例子，训练一个通过胸片图像来诊断是否患有某种疾病的模型，专家标注一张胸片图像要收费，经费仅可标注10张胸片，10张图片又要划分训练集测试集，要面临过拟合。 因此为了节约成本，就要结合大量没有标注的数据
## Take use of Pseudo-Labelling
* 伪标签技术的使用自由度非常高，在这里我们介绍最常用的也是最有效的三种，对于某些特殊场景，可能有更花哨的方法。
### Method one
* 1. 使用标记数据训练有监督模型M
* 2. 使用有监督模型M对无标签数据进行预测，得出预测概率P
* 3. 通过预测概率P筛选高置信度样本
* 4. 使用有标记数据以及伪标签数据训练新模型M’
### Method two
* 1. 使用标记数据训练有监督模型M
* 2. 使用有监督模型M对无标签数据进行预测，得出预测概率P
* 3. 通过预测概率P筛选高置信度样本
* 4. 使用有标记数据以及伪标签数据训练新模型M’
* 5. 将M替换为M’，重复以上步骤直至模型效果不出现提升
### Method three
* 1. 使用标记数据训练有监督模型M
* 2. 使用有监督模型M对无标签数据进行预测，得出预测概率P
* 3. 将模型损失函数改为Loss = loss(labeled_data) + alpha*loss(unlabeled_data)
* 4. 使用有标记数据以及伪标签数据训练新模型M’
## Why Pseudo-labelling work
* [Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks](https://scholar.google.com.sg/scholar?q=Pseudo-Label+:+The+Simple+and+Efficient+Semi-Supervised+Learning+Method+for+Deep+Neural+Networks&hl=zh-CN&as_sdt=0&as_vis=1&oi=scholart) 论文中解释了伪标签学习为何有效，它的有效性可以在两个方面进行考虑
