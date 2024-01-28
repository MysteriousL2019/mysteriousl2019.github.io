---
title: Block Chain
author: Fangzheng
date: 2022-11-14 00:48:00 +0800
categories: [block chain]
tags: [algorithm notebook]
# pin: true
mermaid: true  #code模块
# comments: true
math: true
# img_cdn: https://github.com/MysteriousL2019/mysteriousl2019.github.io/tree/master/assets/img/
---
北京大学区块链安全与应用

## 结构
* 链表的形式
* 使用hash指针代替传统的指针,指针中保存对应数据的内存地址,hash 指针保存前一个block得hash value
* 每个block仅仅用存相邻的块中的信息，利用hash 指针中的值判断前方的信息的真实性

## Merkle tree && binary tree
* 利用hash 指针代替普通指针
* 即，某节点的左指针指向的就是左子树的hash value，同理，某节点的父亲节点存储的值就是这个节点的左右指针的hash值一起取一个hash值放在里面
* 这样的数据结构好处就是只要记住了root hash（root node的两个hash再取hash），就能判读整个树的状态
* tree 的最子叶节点: data blocks是一个transaction

* 作用：提供Merkle Proof，向轻节点证明交易已经完成

* 遍历连接链表中的任意一点的哈希指针就能确保数据没有被篡改。如果对某一 data 进行修改，则会造成哈希指针与上一级不匹配，从而不得不向上修改，直至修改到顶部—— root hash（根哈希值）。所以 只需要记住 root hash 就可以防篡改，并能检测出 Merkle tree 中任何部位的修改。
* 如果要证明某个区块在 Merkle tree 上，需要展示此区块，验证哈希值是否匹配，一直验证到 root hash ，若正确就可以证明该区块在 Merkle tree 上，时间复杂度为 log（n）。
* 如果要证明一个区块不在 Merkle tree 上，我们需要给Merkle tree进行排序，找到一条可能的路径，指向该项可能的所在位置之前以及之后，通过证明前后两项之间没有空间来试图证明该区块不可能在 Merkle tree 上。


<!-- ![The flower](block_chain_structure.png) -->
![superproxy-dashboard](/block_chain_structure.png){: width="1210" height="694"}
<!-- 
![img-description](/block_chain_structure.png)
_Block Chain Structure_ -->

## 总结构
* 所有的block通过hash vector组成的block chain
* 每个chain中的一个block都是一个Merkle tree

* block header 只有一个root hash value；以及使用的哪个版本的协议：version；以及指向前一个区块的指针；难度目标阈：target（使得H（block header）<=target，难度）；随机数nonce
* block body 存有交易列表（transaction list）
* 全节点fully validation node：有区块链的所有信息
* 轻节点 light weight node 只保存block header的信息

## Hash pointer
* 只要是数据结构是无环的情况，都可以使用hash pointer代替普通指针

# 一致性的考虑
* 普通分布式系统中满足CAP原理，即consistency，availability，partition tolerance.不能被同时满足
* 所以为了保证block chain一致性，可以利用其中多数节点是好的原理进行少数服从多数的投票，但是为了确定membership是另一个问题
* 因为block chain的建立是生成一个public，private key pair就可以在本地进行工作。所以会产生————一个超级计算机不断地产生公私对，直到超过一半，就有了绝对投票权（sybil attack），为了避免这个问题，使用H(block Header)中的nonce 随机数作为工作的证明。
* PoW顾名思义就是用工作成果来证明完成的工作量。在区块链採用 POW 工作量证明的共识机制中，比特币 Bitcoin 是其中最具有代表性的，其共识协议则主要是由工作量证明和最长链机制两部分组成。


# 工作量证明
* 英国的密码学专家亚当.贝克（Adam Back）1997年发明了哈希现金（Hashcash），就是用工作量证明系统（Proof Of Work）这个方法解决了互联网垃圾邮件问题。它要求计算机在获得发送信息权限之前做一定的计算工作，这对正常的信息传播来讲，几乎很难察觉，但是对向全网大量散布垃圾信息的计算机来说，就成为了巨大的工作量和负担。在比特币之前，哈希现金被用于垃圾邮件的过滤，也被微软用于Hotmail、Outlook等产品中。
* 工作量证明，简单说就是一份证明，用来确认你做过一定量的工作。你拿到了大学的毕业证，能够客观证明你进行过大学四年的学习，具备一个大学生的学习能力。通过对工作的结果进行认证来证明完成了相应的工作量，这样的方式是一种非常高效的方式。工作量证明，虽然工作量证明很公平，然而对它大家也有一些批评。一个常见的指责就是消耗能源，因为节点进行算力竞赛是要消耗电力的。我们在前面讲到比特币的挖矿的时候，我们了解过电力占挖矿成本中很大的比重，就可以实际感受到工作量证明的对于能量消耗是很大的。
# 最长链机制
* 一般的区块链网络为了长久发展下去，都会要求所有节点遵守一个公式，就是所有保存到本地的区块链，都必须是被本地节点验证通过的最长链。由于区块链的每个区块必须引用它的上一个区块，所以最长链是最难推翻的。
* 那么，怎么来保证最长链呢？理论上，矿工可以在任意区块的基础上开始计算下一个区块的。但只有最长区块链上的区块才能获得系统的承认并得到挖矿奖励。打包区块获得的奖励只有在该区块上被增加之后才能获得使用。也就是说，如果你是矿工，你挖出来了新的区块，你获得了新生的比特币奖励，只有往后诞生了99个区块之后，你才能动用这个区块里的奖励。这是保证区块链不发生分裂的重要机制。

* 在比特币区块链上，每隔 10 - 15 分钟就会把这期间的所有交易讯息打包成一个区块并加到最长的区块链上，帮助成功验证新区块链的节点会被区块链上的奖励机制给予帮助新区块生成的营运奖励金，这整串打包区块纪录帐本与获得奖励的过程即俗称挖矿 mining，而所有参与写入帐本竞争的节点则称为矿工 miner。
* 比特币区块链上的节点，需付出工作量证明来获得记帐权。意味在比特币区块链上的所有节点，只要提出工作量证明都可以参与写帐本打包新区块的竞争。优先胜出的节点会将新区块的帐本讯息广播至比特币全网路上，其馀的节点验证后即从竞争改为接受新的区块并同步新区块的帐本讯息，大家全部完成后再一同参与新的交易讯息区块打包。
* 比特币挖矿竞争的实际行为，是所有的参与节点使用电脑算力来执行数学运算，其第一个运算结果满足下列算式的节点有权利写入帐本并获得奖励。
* SHA256 ( Block data + Nonce) < Difficulty
* 比特币设计的挖矿难度会在维持 10-15 分钟区块生成速率的前提下，依据整个网路算力大小来动态调整难度值 Difficulty (难度值为使用杂凑函数 Hash Algorithm 算出的无法预测数值)，全网越多人挖矿算力月高而难度越高。
* 难度值 Difficulty 的调整是在每个完整节点中独立自动发生，每 2016 个区块，所有节点都会按统一的公式自动调整难度值，这个公式是由最新 2016 个区块的花费时长与期望时长（期望时长为 20160 分钟即 2周，是按每10分钟一个区块的产生速率计算出的总时长）比较得出的，根据实际时长与期望时长的比值，进行相应调整，其公式如下

