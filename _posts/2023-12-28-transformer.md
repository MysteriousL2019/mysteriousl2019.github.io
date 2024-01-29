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
img_path: /_posts/transformer/
math: true
# img_cdn: https://github.com/MysteriousL2019/mysteriousl2019.github.io/tree/master/assets/img/
---
## Serval Basic Units
### Bath Normalize & Layer Normalize
* ![Alt text](image.png)
* N for batchsize, C for seqlen, H, W for embedding dim
* From Operation: BN operates on the same feature data for all data within the same batch; while LN operates on the same sample. 
* Both BN and LN can better suppress the gradient disappearance and gradient explosion.BN is not suitable for sequence networks such as RNN, transformer, etc., and is not suitable for the case of indeterminate text length and small batchsize, it is suitable for networks such as CNN in CV.
* And LN is suitable for networks such as RNN and transformer in NLP, because the length of sequence may be inconsistent.
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
