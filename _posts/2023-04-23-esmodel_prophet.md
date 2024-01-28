---
title: ES model with prophet in Series forecasting
author: Fangzheng
date: 2023-04-23 19:00:00 +0800
categories: [machine learning]
tags: [machine learning]
pin: false
mermaid: true  #code模块
# comments: true
math: true
---
# Exponential Smothing Model
* 一种时序预测算法，可用于所有时序
* 平稳时间序列 (Stationary Time Series)
* * 1)任意t，其均值为常数
* * 2)任意t,s.自相关系数以及自方差仅依赖于时间差t-s
* * 平稳又包括：严平稳、宽平稳。
* * 非平稳时间序列 (Non-stationary Time Series)是指包含趋势、季节性或周期性的序列，它
* * 可能只含有其中的一种成分，也可能是几种成分的组合。
* * 基于加权平均，最近的观测值应该具有更高的权重。在中短期预测
* 在做时序预测时，一个显然的思路是：认为离着预测点越近的点，作用越大。比如我这个月体重100斤，去年某个月120斤，显然对于预测下个月体重而言，这个月的数据影响力更大些。假设随着时间变化权重以指数方式下降——最近为0.8，然后0.8**2，0.8**3…，最终年代久远的数据权重将接近于0。将权重按照指数级进行衰减，这就是指数平滑法的基本思想。特点是：短期预测能力较好。
## 假设序列由三部分:trend、seasonality、random noise
* 指数平滑法有几种不同形式：一次指数平滑法针对没有趋势和季节性的序列，二次指数平滑法针对有趋势但没有季节性的序列，三次指数平滑法针对有趋势也有季节性的序列。“Holt-Winters”有时特指三次指数平滑法。
* 单指数模型（simple/single exponential model）拟合的是只有常数水平项和时间点i处随机项的时间序列，这时认为时间序列不存在趋势项和季节效应；
* 双指数模型（double exponential model；也叫Holt指数平滑，Holt exponential smoothing）拟合的是有水平项和趋势项的时序
* 三指数模型（triple exponential model；也叫Holt-Winters指数平滑，Holt-Winters exponential smoothing）拟合的是有水平项、趋势项以及季节效应的时序。

* [官方网站](https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html){:target="_blank"}

# Prophet
* [论文 Forecast At Scale](https://peerj.com/preprints/3190/){:target="_blank"}
* [官方网站](https://facebook.github.io/prophet/docs/quick_start.html){:target="_blank"}
* g(t)。它代表了趋势，目的是为了捕捉系列的一般趋势。例如，随着越来越多的人加入网络，Facebook上的广告浏览量可能会随着时间而增加。但是，增加的确切函数会是什么？
* s(t): 它是 季节性 成分。广告浏览量也可能取决于季节。例如，在北半球的夏季，人们可能会花更多的时间在户外，而减少在电脑前的时间。这种季节性的波动对于不同的商业时间序列来说可能是非常不同的。因此，第二部分是一个模拟季节性趋势的函数。
* h(t)。 假期部分。我们使用对大多数商业时间序列有明显影响的假期信息。需要注意的是，不同年份、不同国家的假期都不一样，因此需要明确提供给模型的信息。
* 误差项εt代表模型无法解释的随机波动。像往常一样，假设εt遵循正态分布N （0，σ2），均值为零，未知方差为σ，必须从数据中得出。

# Compare Prophet with Exponential Smothing Model
* Prophet是一种更复杂、更灵活的时间序列预测模型，由Facebook开发，与Exponential Smoothing有许多不同之处。

* 处理缺失值：Exponential Smoothing对于缺失值和异常值非常敏感，而Prophet能够处理缺失值和异常值，并对其进行自动插值和调整。

* 考虑节假日因素：Prophet能够自动识别和建模节假日效应，例如每周的周末，国定假日等，而Exponential Smoothing则需要手动输入这些信息。

* 考虑季节性：Prophet能够处理多种季节性，例如日度、周度、年度等，而Exponential Smoothing则只能处理单一季节性。

* 灵活的趋势建模：Prophet允许用户通过添加可选的调节项来更好地建模趋势，例如变化点、非线性趋势等。

* 总体而言，Prophet比Exponential Smoothing更加灵活和全面，能够处理更多的时间序列情况。然而，Prophet需要更多的计算资源和更长的运行时间，对于较小的数据集，Exponential Smoothing可能更为适合。因此，选择哪种方法取决于数据的特点和预测的要求。
