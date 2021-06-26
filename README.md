# pytorch_ner_v1
中文命名实体识别的三种架构实现:
- 普通的bert用于命名实体识别：bertNerNorExample.py
- 基于机器阅读理解的命名实体识别：bertNerMrcExample.py
- 基于span的命名实体识别：bertNerSpanExample.py

# 整体结构说明
--checkpoins：存放保存的模型<br>
--config：一些配置文件<br>
--data：存放数据<br>
--datasets：将数据转换为数据集<br>
--example：使用的样例<br>
--layers：使用的一些公共层<br>
--logs：存放日志文件<br>
--models：存放模型结构<br>
--preprocess：处理数据相关<br>
--utils：存放辅助的一些内容<br>
--test：用于测试一些函数<br>
--README.md：说明文档<br>
--requirements.txt：一些依赖的包的版本<br>

# 运行流程
1. 在data中存放这数据，以cner数据为例：<br>
--raw_data：存放原始数据，数据可能是BMOES、BIO格式的，我们先在该目录下创建一个process.py文件，转换成相应的格式保存到mid_data中<br>
--mid_data:里面存放着转换后保存的一些数据<br>
--nor_data：普通bert命名实体识别数据<br>
--mrc_data：基于机器阅读理解的命名实体识别数据<br>
--span_data：基于span的命名实体识别数据<br>
2. 在第一步得到json格式的数据之后，我们在preprocess文件夹下有三种不同方式的处理过程，分别对应不同架构，分别处理之后会在data下生成对应的数据。
3. 在example中有三种架构运行的样例，在main中会分别对应有训练、验证和预测。
3. 相关的结果会保存在logs文件夹下

# 结果
bertNerNorExample.py
在测试集上：
```python
          precision    recall  f1-score   support

     PRO       0.70      0.20      0.31        35
     ORG       0.77      0.46      0.58       571
    CONT       1.00      1.00      1.00        28
    RACE       1.00      0.14      0.25        14
    NAME       1.00      0.97      0.99       112
     EDU       0.85      0.81      0.83       115
     LOC       0.00      0.00      0.00         6
   TITLE       0.82      0.59      0.69       854

micro-f1       0.83      0.58      0.68      1735
```
预测：<br>
raw_text = "虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。"<br>
{'NAME': [('虞兔良', 0)], 'CONT': [('中国国籍', 20)], 'EDU': [('浙江绍兴人', 34), ('MBA', 45)], 'TITLE': [('中共党员', 40), ('经济师', 49)]}<br>
bertNerMrcExample.py
```python
-           precision    recall  f1-score   support

     PRO       0.42      0.31      0.36        35
     ORG       0.67      0.54      0.60       571
    CONT       1.00      1.00      1.00        28
    RACE       0.82      1.00      0.90        14
    NAME       0.96      0.98      0.97       112
     EDU       0.89      0.83      0.86       115
     LOC       1.00      0.67      0.80         6
   TITLE       0.79      0.60      0.69       854

micro-f1       0.77      0.63      0.69      1735
```
raw_text = "1954年10月出生，大专学历，中共党员，高级经济师，汉商集团董事长、党委副书记。"<br>
预测：<br>
[('中共党员', 16), ('高级经济师', 21), ('汉商集团董事长', 27), ('党委副书记', 35)]<br>
bertNerSpanExample.py
```python
-           precision    recall  f1-score   support

     PRO       0.11      0.03      0.05        35
     ORG       0.68      0.49      0.57       571
    CONT       0.96      0.96      0.96        28
    RACE       0.82      1.00      0.90        14
    NAME       0.88      0.88      0.88       112
     EDU       0.84      0.82      0.83       115
     LOC       0.00      0.00      0.00         6
   TITLE       0.78      0.58      0.66       854

micro-f1       0.76      0.58      0.66      1735
```
预测：<br>
raw_text = "顾建国先生：研究生学历，正高级工程师，现任本公司董事长、马钢(集团)控股有限公司总经理。"<br>
{'NAME': [('顾建国', 0)], 'EDU': [('研究生学历', 6)], 'TITLE': [('正高级工程师', 12), ('董事长', 24), ('马钢(集团)控股有限公司总经理', 28), ('总经理', 40)], 'ORG': [('本公司', 21)]}