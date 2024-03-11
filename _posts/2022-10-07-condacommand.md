---
title: Basic conda command
author: Fangzheng
date: 2022-10-07 13:06:00 +0800
categories: [command, Artificial Intelligence]
tags: [command ]
# pin: true
mermaid: true  #code模块
comments: true

math: true
# img_cdn: https://github.com/MysteriousL2019/mysteriousl2019.github.io/tree/master/assets/img/
---
## Basic command
### get conda version
```python 
conda --version
```
```python 
conda -V
```
### get help
```
conda --help
conda -h
```
### get help for a specific command
```
conda update --help
conda remove --help
```
## Environment management
### get all command related to environment management
```
conda env -h
```
### create envirnment 
```
conda create --name your_env_name
```
### with specific python version
```
conda create --name your_env_name python=2.7
conda create --name your_env_name python=3
conda create --name your_env_name python=3.5
```
### list all envirnment
```
conda info --envs
conda env list
```
### enter a specific envirnment
```
activate your_env_name
```
### exit 
```
deactivate 
```
### clone 
```
conda create --name new_env_name --clone old_env_name 
```
### delete an envirnment
```
conda remove --name your_env_name --all
```
## Share an envirnment 
* share your current configuration to others
```
activate target_env
conda env export > environment.yml
```
* then others can use environment.yml
```
conda env create -f environment.yml
```
## Package mangement 
### list all packages in current envirnment
```
conda list
```
### list other envirnment
```
conda list -n your_env_name
```