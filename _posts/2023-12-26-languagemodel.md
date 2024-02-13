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
* Speech Recognition 语音识别 $P("We built this city on rock and roll") > P("We built this city on sausage rolls")$
* Spelling correction 拼写更正 $P("... has no mistakes") > P("... has no mistaeks")$
* Grammar correction 语法更正 $P("... has imporoved") > P("... has improve")$ 
* Machine Translation 机器翻译 $P("I went home") > P("I went to home")$

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
## Reference
* [Transformer、GPT、ChatGPT、LLM、AIGC和LangChain的区别
](https://zhuanlan.zhihu.com/p/647391226)