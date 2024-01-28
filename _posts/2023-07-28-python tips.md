---
title: Python Language Trick
author: Fangzheng
date: 2023-07-28 20:25:00 +0800
categories: [python, Language Trick]
tags: [python ,language]
pin: false
mermaid: true  #code模块
# comments: true
math: true
---
# 使用python创建二维数组的时候
* dp = [[0]*n] * m
* dp = [[0] * n for _ in range(m)]
* 第一种方式使用了乘法操作符，创建了一个由相同引用对象组成的列表。意味着如果你修改了其中一个元素，在其它行中具有相同索引的元素也会被修改，因为它们引用同一个对象。

* 第二种方式使用了列表推导式和循环来创建二维数组。在每一行中都创建了一个新的列表对象，所以每个元素都是不同的对象。因此，修改其中一个元素不会影响到其他行。

```console
m = 3
n = 2

# 第一种方式
dp1 = [[0]*n] * m
dp1[0][0] = 1

print(dp1)
# 输出：[[1, 0], [1, 0], [1, 0]]

# 第二种方式
dp2 = [[0] * n for _ in range(m)]
dp2[0][0] = 1

print(dp2)
# 输出：[[1, 0], [0, 0], [0, 0]]
```