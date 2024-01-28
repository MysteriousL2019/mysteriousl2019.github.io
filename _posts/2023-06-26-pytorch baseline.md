---
title: Pytorch Baseline 
author: Fangzheng
date: 2023-06-26 13:16:00 +0800
categories: [machine learning]
tags: [hint]
# pin: true
mermaid: true  #code模块
# comments: true
math: true
# img_cdn: https://github.com/MysteriousL2019/mysteriousl2019.github.io/tree/master/assets/img/
---
## 核心流程
# 数据获取
* 神经网络输入比较灵活，这里需要手动确定输入形式（图片，文本）
# 模型定义
* 根据input来定义网络结构
# 训练
* 定义loss, optimizer, Train&Valid, 保存模型权重
# 测试
* 根据训练结果测试数据

### 海华中文阅读理解挑战赛
* 基于bert的baseline
# bert 基础
* BERT 的输入可以包含一个句子对 (句子 A 和句子 B)，也可以是单个句子。此外还增加了一些有特殊作用的标志位：
* [CLS] 标志放在第一个句子的首位，经过 BERT 得到的的表征向量 C 可以用于后续的分类任务。
* [SEP] 标志用于分开两个输入句子，例如输入句子 A 和 B，要在句子 A，B 后面增加 [SEP] 标志。
* [UNK]标志指的是未知字符
* [MASK] 标志用于遮盖句子中的一些单词，将单词用 [MASK] 遮盖之后，再利用 BERT 输出的 [MASK] 向量预测单词是什么。
* GPT uses a sentence separator ([SEP]) and classifier token ([CLS]) which are only introduced at fine-tuning time; BERT learns [SEP], [CLS] and sentence A/B embeddings during pre-training. 简单来讲 [CLS]：classification，[SEP]：separator 

# 讲述模型的时候的trick
* 模型结构（结合diagram）, 使用场景, 输入输出
* 数据集介绍, Train(train&valid), Test 标出其中的mean,min(len),25% , 50%, 75%, max
* 五折交叉验证，使用多折模型提升训练效果，充分利用数据集合
* [【赛事分享】海华中文阅读理解挑战赛 | 北京大学朱政烨：基于BERT的Baseline分享 ](https://www.bilibili.com/video/BV1FK4y1J719/?share_source=copy_web&vd_source=bb8b77a3ec54981b27dd91bb401ded0b){:target="_blank"}

# Bert Question Demo 
* colab version
```console
$ !pip install transformers
$ import torch
$ from transformers import BertTokenizer, BertForQuestionAnswering
```


* [使用的是](https://huggingface.co/bert-base-uncased){:target="_blank"}

```console
$ bert_path = 'bert-base-uncased'
$ bert_path = 'bert-large-uncased-whole-word-masking-finetuned-squad' # This is the path for pre-trained BERT model available on $ Hugging Face's model hub.
$ model = BertForQuestionAnswering.from_pretrained(bert_path)
$ tokenizer = BertTokenizer.from_pretrained(bert_path)


$ question, doc = "Who is Lyon" , "Lyon is a killer"
$ encoding = tokenizer.encode_plus(text = question,text_pair = doc,  verbose=False)
$ inputs = encoding['input_ids']  #Token embeddings
$ sentence_embedding = encoding['token_type_ids']  #Segment embeddings
$ tokens = tokenizer.convert_ids_to_tokens(inputs)  #input tokens
$ print("tokens: ", tokens)
$ start_scores, end_scores = model(input_ids=torch.tensor([inputs]),
                                token_type_ids=torch.tensor([sentence_embedding]))
$ print("encoding: ", encoding)
$ print("inputs: ", inputs)
$ print("sentence_embedding: ",sentence_embedding)
$ print("start_scores: ",start_scores)
$ print("end_scores: ", end_scores)


$ outputs = model(input_ids=torch.tensor([inputs]),
                                token_type_ids=torch.tensor([sentence_embedding]))
$ start_scores = torch.tensor(outputs.start_logits)
$ end_scores = torch.tensor(outputs.end_logits)
$ start_index = torch.argmax(start_scores)
$ end_index = torch.argmax(end_scores)
$ #print("start_index:%d, end_index %d"%(start_index, end_index))
$ answer = ' '.join(tokens[start_index:end_index + 1])
$ print(answer) 
$ # 每次执行的结果不一致，这里因为模型没有经过训练，所以效果不好，输出结果不佳
```
* answer是通过BERT模型预测得出的答案，它表示输入文本中与给定问题相关的最佳答案。在这里，answer是从tokens列表中选择的一部分，该部分被认为是回答给定问题的最佳表示方式，其开始和结束位置由start_index和end_index确定。

## 知识图谱问答项目
* 数据集对比，找几个开源数据集，之后对比内部词典，可视化，选定最后的选择。
## 模块一：命名实体识别（Named Entity Recognition，简称NER）：
* 其作用在于从文本中自动识别出命名实体并进行分类。
* 命名实体是指在文本中表示具有特定意义的实体，例如人名、地名、组织机构名称、时间、日期等。通过对文本中的命名实体进行识别和分类，可以帮助我们更好地理解文本的含义和结构，并为其他自然语言处理任务提供帮助，例如信息抽取、问答系统、机器翻译等。
* 命名实体识别通常需要使用到机器学习或深度学习技术，例如基于统计模型的CRF、基于深度学习的神经网络模型等。这些模型可以通过大量已标注好的数据进行训练，并根据预先设定的规则来对文本中的命名实体进行识别和分类。
## 模块二：句子相似度计算
* 句子相似度计算在问答系统中起着重要的作用。问答系统的目的是将用户的自然语言问句转化为机器可以理解的形式，并从知识库或文本库中检索出与该问题相关的答案。
* 在这个过程中，需要衡量用户的输入与知识库或文本库中的内容之间的相似度，以便确定可能的答案。如果问题与库中的某些内容非常相似，则可以推断库中的一些条目可能与该问题有关。
* 例如，在基于检索的问答系统中，通常使用句子相似度计算来评估用户问题和文本库中存储的问题之间的相似度。在使用情况下，当用户提出问题时，将会把问题转换成向量空间表示，并计算它们与数据库中前n个最相似问题的相似度，然后返回与这些最相似问题相关的答案。
* 帮助系统快速地找到与用户问题相关的答案，提升问答系统的准确性和效率。