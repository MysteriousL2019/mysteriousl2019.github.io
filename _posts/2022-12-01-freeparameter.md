---
title: Freeze parameters in Pytorch
author: Fangzheng
date: 2022-12-01 13:06:00 +0800
categories: [Large Language Model, Artificial Intelligence]
tags: [Algorithm ]
# pin: true
mermaid: true  #code模块
comments: true

math: true
# img_cdn: https://github.com/MysteriousL2019/mysteriousl2019.github.io/tree/master/assets/img/
---
## 冻结预训练模型参数
* Pytorch 如何精确的冻结我想冻结的预训练模型的某一层？
* 四种方法，假设目前有模型如下
```python
class Char3SeqModel(nn.Module):
    
    def __init__(self, char_sz, n_fac, n_h):
        super().__init__()
        self.em = nn.Embedding(char_sz, n_fac)
        self.fc1 = nn.Linear(n_fac, n_h)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, char_sz)
        
    def forward(self, ch1, ch2, ch3):
        # do something
        out = #....
        return out

model = Char3SeqModel(10000, 50, 25)
```
* 假设需要冻结fc1，有如下几个方法
* 1.
```
# 冻结
model.fc1.weight.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
# 
# compute loss 
# loss.backward()
# optmizer.step()

# 解冻
model.fc1.weight.requires_grad = True
optimizer.add_param_group({'params': model.fc1.parameters()})
```
* 2.
```python
# 冻结
optimizer = optim.Adam([{'params':[ param for name, param in model.named_parameters() if 'fc1' not in name]}], lr=0.1)
# compute loss
# loss.backward()
# optimizer.step()

# 解冻
optimizer.add_param_group({'params': model.fc1.parameters()})
```
* 3.思路：将原来的layer的weight缓存下来，每次反向传播之后，再将原来的weight赋值给相应的layer。
```python
fc1_old_weights = Variable(model.fc1.weight.data.clone())
# compute loss
# loss.backward()
# optimizer.step()
model.fc1.weight.data = fc1_old_weights.data
```
* 4.思路：在每次进行反向传播更新权重之前将相应layer的gradient手动置为0。缺点也很明显，会浪费计算资源。
```python
# compute loss
# loss.backward()
# set fc1 gradients to 0
# optimizer.step()
```
