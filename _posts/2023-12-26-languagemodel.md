---
title: Language Model
author: Fangzheng
date: 2023-12-26 12:06:00 +0800
categories: [Language Model, Artificial Intelligence]
tags: [algorithm ]
# pin: true
mermaid: true  #code模块
comments: true
mermaid: true

math: true
# img_cdn: https://github.com/MysteriousL2019/mysteriousl2019.github.io/tree/master/assets/img/
---
### Language Usage for NLP task
* Speech Recognition 语音识别 P("We built this city on rock and roll") > P("We built this city on sausage rolls")
* Spelling correction 拼写更正 P("... has no mistakes") > P("... has no mistaeks")
* Grammar correction 语法更正 P("... has imporoved") > P("... has improve")
* Machine Translation 机器翻译 P("I went home") > P("I went to home")
## 引言
在机器学习领域，Embedding模型是一种常见的技术，用于将离散型数据（如单词、商品ID等）转换为固定大小的向量表示，以便在神经网络中进行处理。当我们面对大规模数据集时，一个性能出色的Embedding模型能够极大地提升模型的召回效果。那么，如何对Embedding模型进行微调以实现召回效果的飞跃呢？接下来，我们将一探究竟。
### 1. 理解Embedding模型
* 首先，我们要了解Embedding模型是如何工作的。在Embedding层中，每个离散型数据都被映射到一个低维向量空间。这个向量空间能够捕捉到数据之间的语义和相关性，使得相似的数据在向量空间中彼此靠近。通过训练，Embedding模型可以学习到数据的有效表示，从而提高模型的性能。
### 2. 微调Embedding模型
* 微调（Fine-tuning）是指对已经训练好的模型进行进一步的训练，以适应特定的任务或数据集。对于Embedding模型来说，微调意味着对已经学习到的向量表示进行调整，以提高模型在目标数据集上的性能。
* 在微调过程中，我们通常会对Embedding层的参数进行更新。这可以通过反向传播算法和梯度下降等优化方法来实现。通过调整这些参数，我们可以让模型更好地适应目标数据集，从而提高召回效果。
### 3. 实战技巧：提升召回效果
* 在进行Embedding模型微调时，以下几个技巧可以帮助你提升召回效果：

## 3.1 选择合适的数据集
* 选择一个与目标任务相关且质量较高的数据集进行微调至关重要。一个好的数据集应该包含足够的样本和丰富的多样性，以便让模型学习到更多的信息。


## 3.2 初始化策略
* 在微调过程中，我们可以选择使用预训练的Embedding模型作为初始化。这样做可以充分利用已经学到的知识，加速模型的收敛速度。同时，也可以尝试使用不同的初始化策略，如随机初始化、使用其他任务的Embedding模型等，以找到最佳的性能表现。
## 3.3 学习率设置
* 学习率是微调过程中的一个重要超参数。过大的学习率可能导致模型在微调过程中偏离最优解，而过小的学习率则可能导致模型收敛速度过慢。因此，我们需要根据实际情况调整学习率，以找到最佳的平衡点。
## 3.4 正则化技巧
* 为了防止模型在微调过程中出现过拟合现象，我们可以使用正则化技巧，如L1正则化、L2正则化等。这些技巧可以帮助我们在优化目标函数中引入额外的约束项，从而防止模型过度拟合训练数据。
## 3.5 监控和调整
* 在微调过程中，我们需要时刻关注模型的性能表现。通过监控验证集上的召回率、准确率等指标，我们可以及时调整模型的参数和超参数，以找到最佳的模型表现。同时，也可以尝试使用不同的优化算法和技巧，如早停法（Early Stopping）、模型集成等，以提高模型的性能。
## 4. 结语
* 通过对Embedding模型进行微调，我们可以有效地提升大模型的召回效果。在实际应用中，我们需要结合具体任务和数据集特点，选择合适的微调策略和优化技巧。相信通过不断尝试和实践，你一定能够掌握这一关键技术，实现召回效果的飞跃之旅！
### Embedding and Word Embedding

* 现有的机器学习方法往往无法直接处理文本数据，因此需要找到合适的方法，将文本数据转换为数值型数据，由此引出了Word Embedding（词嵌入）的概念。
* 词嵌入是自然语言处理（NLP）中语言模型与表征学习技术的统称，它是NLP里的早期预训练技术。它是指把一个维数为所有词的数量的高维空间嵌入到一个维数低得多的连续向量空间中，每个单词或词组被映射为实数域上的向量，这也是分布式表示：向量的每一维度都没有实际意义，而整体代表一个具体概念。

* 分布式表示相较于传统的独热编码（one-hot）表示具备更强的表示能力，而独热编码存在维度灾难和语义鸿沟（不能进行相似度计算）等问题。传统的分布式表示方法，如矩阵分解（SVD/LSA）、LDA等均是根据全局语料进行训练，是机器学习时代的产物。
* Word Embedding的输入是原始文本中的一组不重叠的词汇，假设有句子：apple on a apple tree。那么为了便于处理，我们可以将这些词汇放置到一个dictionary里，例如：[“apple”, “on”, “a”, “tree”]，这个dictionary就可以看作是Word Embedding的一个输入。
* Word Embedding的输出就是每个word的向量表示。对于上文中的原始输入，假设使用最简单的one hot编码方式，那么每个word都对应了一种数值表示。例如，apple对应的vector就是[1, 0, 0, 0]，a对应的vector就是[0, 0, 1, 0]，各种机器学习应用可以基于这种word的数值表示来构建各自的模型。当然，这是一种最简单的映射方法，但却足以阐述Word Embedding的意义。
* 文本表示的类型：
    * 基于one-hot、tf-idf、textrank等的bag-of-words；
    * 主题模型：LSA（SVD）、pLSA、LDA；
    * 基于词向量的固定表征：word2vec、fastText、glove
    * 基于词向量的动态表征：ELMO、GPT、bert
* 上面给出的4个类型也是nlp领域最为常用的文本表示了，文本是由每个单词构成的，而谈起词向量，one-hot是可认为是最为简单的词向量，但存在维度灾难和语义鸿沟等问题；通过构建共现矩阵并利用SVD求解构建词向量，则计算复杂度高；而早期词向量的研究通常来源于语言模型，比如NNLM和RNNLM，其主要目的是语言模型，而词向量只是一个副产物。
### 使得Embedding流行的Word2Vec
* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781). Google的Tomas Mikolov提出word2vec的两篇文章之一，这篇文章更具有综述性质，列举了NNLM、RNNLM等诸多词向量模型，但最重要的还是提出了CBOW和Skip-gram两种word2vec的模型结构。虽然词向量的研究早已有之，但不得不说还是Google的word2vec的提出让词向量重归主流，拉开了整个embedding技术发展的序幕。
* [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546) Tomas Mikolov的另一篇word2vec奠基性的文章。相比上一篇的综述，本文更详细的阐述了Skip-gram模型的细节，包括模型的具体形式和 Hierarchical Softmax和 Negative Sampling两种可行的训练方法。
* [Word2vec Parameter Learning Explained](https://arxiv.org/pdf/1411.2738). 虽然Mikolov的两篇代表作标志的word2vec的诞生，但其中忽略了大量技术细节，如果希望完全读懂word2vec的原理和实现方法，比如词向量具体如何抽取，具体的训练过程等，强烈建议大家阅读UMich Xin Rong博士的这篇针对word2vec的解释性文章。
    * Word2Vec算法原理：
        * skip-gram: 用一个词语作为输入，来预测它周围的上下文
        * cbow: 拿一个词语的上下文作为输入，来预测这个词语本身
### Word Embedding 缺点
* 多义词是自然语言中经常出现的现象，也是语言灵活性和高效性的一种体现。然而，Word Embedding针对多义词问题没有得到很好的解决。
* 比如多义词Bank，有两个常用含义，但是Word Embedding在对bank这个单词进行编码的时候，是区分不开这两个含义的，因为它们尽管上下文环境中出现的单词不同，但是在用语言模型训练的时候，不论什么上下文的句子经过word2vec，都是预测相同的单词bank，而同一个单词占的是同一行的参数空间，这导致两种不同的上下文信息都会编码到相同的word embedding空间里去。所以word embedding无法区分多义词的不同语义，这就是它的一个比较严重的问题。

## 演进和发展
* word embedding得到的词向量是固定表征的，无法解决一词多义等问题，因此引入基于语言模型的动态表征方法：ELMO、GPT、bert，以ELMO为例：
* 针对多义词问题，ELMO提供了一种简洁优雅的解决方案，ELMO是“Embedding from Language Models”的简称（论文：Deep contextualized word representation）。ELMO的本质思想是：事先用语言模型学好一个单词的Word Embedding，此时多义词无法区分，但在实际使用Word Embedding的时候，单词已经具备了特定的上下文了，这个时候可以根据上下文单词的语义去调整单词的Word Embedding表示，这样经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。所以ELMO本身是个根据当前上下文对Word Embedding动态调整的思路。

## Word2vec 的训练trick
* Word2vec 本质上是一个语言模型，它的输出节点数是 V 个，对应了 V 个词语，本质上是一个多分类问题，但实际当中，词语的个数非常非常多，会给计算造成很大困难，所以需要用技巧来加速训练。
* 为了更新输出向量的参数，我们需要先计算误差，然后通过反向传播更新参数。在计算误差是我们需要遍历词向量的所有维度，这相当于遍历了一遍单词表，碰到大型语料库时计算代价非常昂贵。要解决这个问题，有三种方式：
* Hierarchical Softmax：通过 Hierarchical Softmax 将复杂度从 O(n) 降为 O(log n)；
* Sub-Sampling Frequent Words：通过采样函数一定概率过滤高频单词；
* Negative Sampling：直接通过采样的方式减少负样本。

### Application
* Word2vec 主要原理是根据上下文来预测单词，一个词的意义往往可以从其前后的句子中抽取出来。
* 而用户的行为也是一种相似的时间序列，可以通过上下文进行推断。当用户浏览并与内容进行交互时，我们可以从用户前后的交互过程中判断行为的抽象特征，这就使得我们可以用词向量模型应用到推荐、广告领域当中。
* Word2vec 已经应用于多个领域，并取得了巨大成功：
* Airbnb 将用户的浏览行为组成 List，通过 Word2Vec 方法学习 item 的向量，其点击率提升了 21%，且带动了 99% 的预定转化率；
* Yahoo 邮箱从发送到用户的购物凭证中抽取商品并组成 List，通过 Word2Vec 学习并为用户推荐潜在的商品；
* 将用户的搜索查询和广告组成 List，并为其学习特征向量，以便对于给定的搜索查询可以匹配适合的广告。

### 区别
BERT模型与OpenAI GPT的区别就在于采用了Transformer Encoder，也就是每个时刻的Attention计算都能够得到全部时刻的输入，而OpenAI GPT采用了Transformer Decoder，每个时刻的Attention计算只能依赖于该时刻前的所有时刻的输入，因为OpenAI GPT是采用了单向语言模型。

### Large language model
* GPT，全称Generative Pre-training Transformer，是OpenAI开发的一种基于Transformer的大规模自然语言生成模型。GPT模型采用了自监督学习的方式，首先在大量的无标签数据上进行预训练，然后在特定任务的数据上进行微调。
    * 在预训练（Pre-training）阶段，GPT模型使用了一个被称为“Masked Language Model”（MLM）的任务，要求预测一个句子中被遮盖住的部分。预训练的目标是最大化句子中每个位置的单词的条件概率，这个概率由模型生成的分布和真实单词的分布之间的交叉熵来计算。
    * 在微调（fine-tuning）阶段，GPT模型训练在特定任务的数据上进行，例如情感分类、问答等。最大的目标是最小化特定任务的损失函数，例如分类任务的交叉熵损失函数。
    * GPT模型的优点在于，由于其预训练-强度的训练策略，它可以有效地利用大量无标签数据进行学习，并且可以轻松地适应各种不同的任务。此外，由于其基于Transformer的结构，它可以处理输入序列中的所有单词，比基于回圈神经网络的模型更高效。
* GPT模型的主要结构是一个多层的Transformer解码器，但它只使用了Transformer解码器的部分，没有使用编码器-解码器的结构。另外，为了保证生成的文本在语法和语义上的连贯性， GPT模型采用了因果掩码（causal mask）或者称为自回归掩码（auto-regressive mask），这使得每个单词只能看到其前面的单词，而不能看到后面的单词。
* GPT演进了三个版本：
    * （1）GPT-1用的是自监督预训练+有监督参数，5G文档，1亿参数，这两段式的语言模型，其能力还是比较单一，即翻译模型只能翻译，填空模型而已能填空，抽象模型只能抽象等等，要在实际任务中使用，需要分别在各自的数据上做参数训练，这显然很不智慧。
    * （2）GPT-2用的是纯自监督训练预训练，相对于GPT-1，它可以无监督学习，即可以从大量未标记的文本中学习语言模式，而无需人工标记的数据。这使得GPT-2在训练时更加灵活和。训练它引入了更多的任务进行高效预，40G文档，15亿参数，能够在没有针对下游任务进行训练的条件下，就在下游任务上有很好的表现。
    * （3）GPT-3沿用了GPT-2的纯自监督预训练，但数据大了好几个量级，570G文档，模型参数量为1750亿，GPT-3表现出强大的零样本（零-这意味着它可以在没有或只有极少例子的情况下，理解并完成新的任务，它能够生成更连贯、自然和人性化的文本，理解文本、获取常识以及理解复杂概念等方面也比 GPT-2 表现得更好。
* InstructGPT
* GPT-3虽然在伟大的NLP任务以及文本生成的能力上令人惊叹，但模型在实际应用中时长会暴露以下缺陷，很多时候，他并不是按人类的表达方式去说话：
    * （1）提供无效回答：没有遵循用户的明确指示，回答非所问。
    * （2）内容胡编乱造：部分根据文字概率分布虚构出不合理的内容。
    * （3）缺乏可解释性：人们很难理解模型是如何做出特定决策的，难以相信答案的准确性。
    * （4）内容偏差：模型从数据中获取偏差，导致预测不公平或不准确。
    * （5）连续交易能力弱：长文本生成较弱，上下文无法实现连续。
* 在此背景下，OpenAI 提出了一个概念“对齐”，意思是模型输出与人类真实意图一致，符合人类偏好。因此，为了让模型输出与用户意图更加“对齐”，需要指导 GPT 这个工作。
* InstructGPT相对于GPT的主要改进是采用了来自人类回馈的强化学习——RLHF（Reinforcement Learning with human Feedback）来配置GPT-3，这种技术将人类的偏好作为预警信号来调控模型。如上所示，以抽象生成任务为例，详细展示了如何基于人类回馈进行强化学习，最终训练完成得到InstructGPT模型。主要分为三步：
    * 1. 收集人类回馈：使用初始化模型对一个样本生成多个不同的摘要，人工对多个摘要按效果进行排序，得到一批排好序的摘要样本；
    * 2. 训练奖励模型：使用第1步得到的样本集，训练一个模型，该模型输入为一篇文章和回答的一个摘要，模型输出为该摘要的得分；
    * 3. 训练策略模型：使用初始化的策略模型生成一篇文章的摘要，然后使用奖励模型进行摘要打分，再使用打分值借助PPO演算法重新优化策略模型
* InstructGPT可以更好地理解用户的意图，通过指令-回答对的数据集和指令-评价对的数据集。此外，它可以学习如何根据不同的指令生成更有用、更真实、更友好的输出。

* In the Application of Large language model, there are basically two parts. One is Fine-tuning 指将预训练的大型模型在特定任务或数据集上进行进一步调整，以提高性能。这种调整可以是在已有模型的基础上进行少量的参数更新，使模型适应特定的任务或数据，从而避免了从头开始训练模型。
    * Fine-tuning 的过程通常包括选择合适的预训练模型、定义适当的目标任务和损失函数、调整学习率等超参数，并在目标任务的数据上进行模型的训练和优化。
    * When to use Fine-tuning
        * 1. firstly try: Prompt engineering, or Prompt chaining (breaking complex tasks into multiple prompts), and [function calling](https://platform.openai.com/docs/guides/function-calling?lang=python) with the key reasons being:
            * There are many tasks at which our models may not initially appear to perform well, but results can be improved with the right prompts - thus fine-tuning may not be necessary
            * Iterating over prompts and other tactics has a much faster feedback loop than iterating with fine-tuning, which requires creating datasets and running training jobs
            * In cases where fine-tuning is still necessary, initial prompt engineering work is not wasted - we typically see best results when using a good prompt in the fine-tuning data (or combining prompt chaining / tool use with fine-tuning)
        * [References OpenAI Prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering/strategy-write-clear-instructions)
        * Here are some common issues that Fine-tune can solve
            * Setting the style, tone, format, or other qualitative aspects
            * Improving reliability at producing a desired output
            * Correcting failures to follow complex prompts
            * Handling many edge cases in specific ways

* RAG(Retrieval-Augmented Generation)
    * 是一种结合了检索和生成的模型架构，旨在解决生成式任务中的信息检索和生成之间的问题。它由两个主要组件组成：一个检索器和一个生成器。
    * 检索器负责从大型文本语料库中检索相关的上下文信息或答案，并将其提供给生成器。生成器则使用检索到的信息作为输入，生成与上下文相一致的自然语言文本。
    * RAG 可以用于各种任务，包括问答、摘要、对话生成等，它在信息检索和生成之间实现了良好的平衡，提供了更准确和连贯的生成结果。
### Langchain
* LangChain开源项目最近很火，其是一个工具包，帮助把LLM和其他资源（比如你自己的领域资料）、计算能力结合起来，实现本地化知识库搜索与智慧答案生成。

### LangChain的准备工作包括：
* 1、海量的本地领域知识库，知识库是由一段时期的文本构成的。
* 2、基于问题搜索知识库中文本的功能语言模型。
* 3、基于问题与问题相关的知识库文本进行问答式的对话式大语言模型，比如开源的chatglm、LLama、Bloom等。
* 其主要工作思路如下：
* 1、把领域内容拆成一块块的小块、对块进行了嵌入后放入索引库（为后面提供语义搜索做准备）。
* 2、搜索的时候把Query进行Embedding后通过语义检索找到最相似的K个Docs。
* 3、把相关的文档构建成提示的语境，基于相关内容进行QA，让chatglm等进行语境学习，用人话回答问题。

## Reference
* [Transformer、GPT、ChatGPT、LLM、AIGC和LangChain的区别](https://zhuanlan.zhihu.com/p/647391226)
* [一文读懂NLP](https://zhuanlan.zhihu.com/p/384452959)
* [embedding的原理及实践](https://qiankunli.github.io/2022/03/02/embedding.html)
* [大模型微调embedding](https://blog.csdn.net/asd8705/article/details/135586686)
* [Prompt工程还是SFT微调？剖析企业应用中优化大语言模型输出的两种方案](https://developer.volcengine.com/articles/7370375425945993226)