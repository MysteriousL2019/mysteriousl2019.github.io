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
* 方便的组件
```python
作者：肥波喇齐
链接：https://www.zhihu.com/question/311095447/answer/589307812
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

from collections.abc import Iterable

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)
```
```python
# 冻结第一层
freeze_by_idxs(model, 0)
# 冻结第一、二层
freeze_by_idxs(model, [0, 1])
#冻结倒数第一层
freeze_by_idxs(model, -1)
# 解冻第一层
unfreeze_by_idxs(model, 0)
# 解冻倒数第一层
unfreeze_by_idxs(model, -1)


# 冻结 em层
freeze_by_names(model, 'em')
# 冻结 fc1, fc3层
freeze_by_names(model, ('fc1', 'fc3'))
# 解冻em, fc1, fc3层
unfreeze_by_names(model, ('em', 'fc1', 'fc3'))
```