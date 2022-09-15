# Pytorch | 深入剖析torch.nn.Module方法及源码

torch.nn是一个专门为神经网络设计的模块化接口，包含卷积、池化、线性等计算，以及其他如loss等，可以将torch.nn中的每一个模块看做神经网络中的每一层。

torch.nn.Module是网络模型的一个基类，大部分自定义的子模型（卷积、池化甚至整个网络）是这个基类的子类。

首先我们看看如何定义一个Module：

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

## 一、`class torch.nn.Parameter`

功能：`torch.nn.Parameter`是

exp: `torch.nn.Parameter(torch.tensor[3.14159],requore_grad=True)`