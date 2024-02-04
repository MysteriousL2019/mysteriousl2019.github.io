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
