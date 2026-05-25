# 基于代价敏感集成学习的工业设备预测性维护与故障风险识别研究

## 1. 项目简介

本项目为研究生《高级机器学习理论》课程报告配套实验程序，选题来源于 AI-Cases 中的 Manufacturing / Predictive Maintenance 场景，研究对象为工业设备预测性维护问题。

工业设备在运行过程中会产生温度、转速、扭矩、刀具磨损等状态数据。预测性维护的目标是在设备发生故障前，根据运行状态数据识别潜在故障风险，从而减少非计划停机、降低维护成本，并提高生产系统的可靠性。

本项目以 AI4I 2020 Predictive Maintenance Dataset 为实验数据，构建工业设备故障二分类模型。实验对比 Logistic Regression、Random Forest、XGBoost 或 GradientBoosting、MLP 等多种机器学习方法，并进一步引入代价敏感学习和阈值优化策略，以提高故障样本的识别能力。

## 2. 项目信息

- 课程名称：高级机器学习理论
- 报告题目：基于代价敏感集成学习的工业设备预测性维护与故障风险识别研究
- 姓名：董艺玮
- 院系：社会学院
- 任务类型：二分类任务
- 预测目标：Machine failure
- 主要方法：机器学习分类模型、集成学习、类别不均衡处理、阈值优化

## 3. 数据来源

本项目使用公开数据集：

AI4I 2020 Predictive Maintenance Dataset

数据来源：

UCI Machine Learning Repository

数据下载地址：

https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv

该数据集包含工业预测性维护场景下的设备运行状态数据。每条样本对应一个设备运行状态记录，包含空气温度、工艺温度、转速、扭矩、刀具磨损时间等特征，并给出设备是否发生故障的标签。

## 4. 数据字段说明

本项目使用的主要输入特征包括：

| 字段名 | 含义 |
|---|---|
| Type | 产品类型 |
| Air temperature [K] | 空气温度 |
| Process temperature [K] | 工艺温度 |
| Rotational speed [rpm] | 转速 |
| Torque [Nm] | 扭矩 |
| Tool wear [min] | 刀具磨损时间 |

预测目标为：

| 字段名 | 含义 |
|---|---|
| Machine failure | 设备是否发生故障 |

其中，`Machine failure = 0` 表示设备未发生故障，`Machine failure = 1` 表示设备发生故障。

注意：`UID` 和 `Product ID` 仅为样本编号或产品编号，不作为模型输入。`TWF`、`HDF`、`PWF`、`OSF`、`RNF` 是故障模式标签，它们与最终故障标签存在直接关系，因此本项目不将这些字段作为模型输入，以避免标签泄漏。

## 5. 项目结构

```text
course_dyw/
├── data/
│   ├── raw/                 # 原始数据
│   └── processed/           # 预处理后的数据
│
├── src/
│   ├── download_data.py     # 数据下载
│   ├── preprocess.py        # 数据清洗、编码、标准化和数据集划分
│   ├── train_models.py      # 模型训练
│   ├── evaluate.py          # 模型评价
│   ├── plot_results.py      # 实验图表绘制
│   └── utils.py             # 通用路径和工具函数
│
├── results/
│   ├── figures/             # 实验图表
│   ├── tables/              # 实验结果表格
│   └── models/              # 训练后的模型文件
│
├── report/
│   └── 高级机器学习理论课程报告_董艺玮_预测性维护.docx
│
├── requirements.txt         # Python 依赖
├── run_all.py               # 一键运行实验流程
└── README.md                # 项目说明文件