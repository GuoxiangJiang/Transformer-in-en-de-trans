# Transformer 机器翻译项目

基于 Transformer 架构的英德机器翻译系统，使用 IWSLT2017 数据集训练。

##  项目结构

```
mid-term/
├── src/
│   ├── models/
│   │   └── model.py          # Transformer 模型实现
│   ├── utils/
│   │   └── tokenizer.py      # 分词器工具
│   ├── data/
│   │   └── iwslt2017_dataset/  # 数据集
│   ├── train.py              # 训练脚本
│   └── test.py               # 测试评估脚本
├── scripts/
│   ├── test.sh               # 测试脚本
│   └── train.sh              # 训练脚本
├── results/
│   └── best_model.pt         # 训练好的模型
├── runs/                     # TensorBoard 日志
├── requirements.txt          
└── README.md                 
```

##  快速开始

### 1. 安装依赖

```
pip install -r requirements.txt
```

### 2. 下载数据集

在 `src/data/` 目录下运行数据下载文件：

```bash
python download.py

```

### 3. 训练模型

```bash
source scripts/train.sh
```


### 4. 测试评估


```bash
source scripts/test.sh
```

### 5. 查看训练过程（TensorBoard）

```bash
tensorboard --logdir=./runs --port=6006
```




