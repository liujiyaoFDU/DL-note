# Pytorch | 深入剖析torch.nn.Module方法及源码

torch.nn是一个专门为神经网络设计的模块化接口，包含卷积、池化、线性等计算，以及其他如loss等，可以将torch.nn中的每一个模块看做神经网络中的每一层。

torch.nn.Module是网络模型的一个基类，大部分自定义的子模型（卷积、池化甚至整个网络）是这个基类的子类。

## 一、`class torch.nn.Parameter`

功能：`torch.nn.Parameter`是继承至`torch.tensor`的子类，Parameter类型会自动被认为是module的可训练参数，即加入`.parameter()`迭代器中。

exp: `torch.nn.Parameter(torch.tensor[3.14159],requore_grad=True)`

## 二、`class torch.nn.Module`

### 2.1 构建模型

首先我们看看如何定义一个Module：

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```
Module类包含48个方法，下面我们来看一下这些方法如何使用。
### 2.2 子模型操作

> register_module()

add_module方法的封装，用于将新的`name:module`键值对加入module中。

> add_module(name, module)

将子模块添加到当前模块。example：

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 下面是两种等价方式
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.add_module("conv2", nn.Conv2d(1, 20, 5))

model = Model()
print(model)
```

输出：

Model(

  (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

  (conv2): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

)

> 访问子模型：`model.conv1`

输出：

Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

> children()：

返回网络模型里的组成元素的迭代器。类似`modules()`方法。二者对比可参考：[https://blog.csdn.net/u013066730/article/details/94600978](https://blog.csdn.net/u013066730/article/details/94600978)

举例：

```python 
for i in model.children():
    print(i)
```
结果：

Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

> named_children()

返回直接子模块的迭代器，产生模块的名称以及模块本身。

```python
for name, module in model.named_children():
    print(name,module)
```
> named_children()

返回直接子模块的迭代器，产生模块的名称以及模块本身。

```python
for name, module in model.named_children():
    print(name,module)
```

输出：

conv1 Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

conv2 Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

> `modules()`

返回当前模型所有模型的迭代器，重复的模型只被返回一次。与`children()`方法不同，`modules()`方法还会返回当前模型module

```python
import torch

net = nn.Sequential(nn.Linear(2, 2), 
                    nn.Linear(2, 2),
                    nn.ReLU(),
                    nn.Sequential(nn.Linear(2, 2),
                                    nn.ReLU())
                    )
for module in net.modules():
    print(module)
```

结果：

![nxRY1p](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/nxRY1p.png)

> named_modules

返回网络中所有模块的迭代器，产生模块的名称以及模块本身。

```python
for name, module in model.named_modules():
    print(name, module)
```

输出：

 GaussianModel()

> get_submodule(target: str) -> 'Module'

从Module中获取子module，example：

### 2.3 模型参数（parameter）与缓冲区（buffer）

> register_parameter(self, name: str, param: Optional[Parameter])

用于在当前模块中添加一个parameter变量，其中参数param是一个Parameter类型（继承至tensor类型，nn.parameter.Parameter）。

Example:

```python 
import torch
import torch.nn as nn

class GaussianModel(nn.Module):

    def __init__(self):
        super(GaussianModel, self).__init__()

        self.register_parameter('mean', nn.Parameter(torch.zeros(1),
                                                     requires_grad=True))
        
        self.pdf = torch.distributions.Normal(self.state_dict()['mean'],
                                              torch.tensor([1.0]))
    def forward(self, x):
        return -self.pdf.log_prob(x)

model = GaussianModel()
for name, param in model.named_parameters():
    print(name,param.size())
```

结果：

mean torch.Size([1])

> parameters(recurse=True)

返回模型参数的迭代器

```python 
for param in model.parameters():
    print(type(param), param.size())
```

> named_parameters(prefix='', recurse=True)

返回模块参数的迭代器，产生参数的名称以及参数本身。

```python 
for name, param in net.named_parameters():
    print(name,param.size())
```

结果：

0.weight torch.Size([2, 2])

0.bias torch.Size([2])

1.weight torch.Size([2, 2])

1.bias torch.Size([2])

3.0.weight torch.Size([2, 2])

3.0.bias torch.Size([2])

> get_parameter(target: str)

根据参数名得到参数，exp：

```python 
net.get_parameter('1.weight')
```

结果：

Parameter containing:

tensor([[-0.1466, -0.1264],

        [ 0.2812,  0.1436]], requires_grad=True)

> buffers(recurse=True)

模型中需要保存下来的参数包括两种：

+ 一种是反向传播需要更新：parameter，可以通过parameter()返回
+ 一种是反向传播不需要更新的：buffer，可以通过buffer()返回  

> named_buffers(prefix='', recurse=True)

返回module buffers' name的迭代器，example：

```python 
for name, buf in net.named_buffers():
    print(buf.size())
```

> register_buffer(name: str, tensor: Optional[Tensor], persistent: bool = True)

在当前模块中添加一个buffer变量，例如，现在需要手写一个BatchNorm，那么其`running_mean`并不是一个parameter，这就需要用下述方式注册一个buffer：

```python
class BatchNorm(nn.Module):
    def __init__(self,..):
        self.register_buffer('running_mean',torch.zeros(num_features))
        self.register_buffer('running_variance',torch.ones(num_features))
```

> get_buffer(target: str)

根据buffer名得到buffer值，用法同get_parameter。

### 2.4 数据格式及转换

> float() / double() / half() / bfloat16() 

将所有的parameters和buffers转化为指定的数据类型。

> type(dst_type)

将所有的parameters和buffers转化为目标数据类型。

### 2.5 模型移动

> to_empty()

把模型parameter和buffers移动到指定device上（不保存其具体数值）。

> cpu() / cuda() / xpu(）

将模型的parameters和buffers移动到CPU/GPU

### 2.5 模型模式调整

> train(mode=True)
将该模块设置为train训练模式。默认值：True。

> eval()

将module设置为验证模式，会影响一些特定modules，如：Dropout，BatchNorm等

### 2.6 其他

> zero_grad(set_to_none=False)

将所有模型参数的梯度设置为零。set_to_none=True会让内存分配器来处理梯度，而不是主动将它们设置为0，这样会适度加速。

> forward(*input)

方法定义了神经网络每次调用时都需要执行的前向传播计算，所有的子类都必须要重写这个方法。

> apply(fn)

+ 递归地将函数应用于所有子模块。
+ apply方法可以用于任何submodule（通过.children()或者self.获取到的）
+ 常用来初始化模型参数（同torch.nn.init）

example:

```python
import torch
@torch.no_grad()  # 不计算梯度，不反向传播
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2),nn.ReLU(),nn.Sequential(nn.Linear(2, 2),nn.ReLU()))
net.apply(init_weights)
```

![bZWe9D](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/bZWe9D.png)

> load_state_dict(state_dict, strict=True)

将 state_dict 中的参数(parameters)和缓冲区(buffers)复制到此模块及其子模块中。如果 strict 为 True，则 state_dict 的键必须与该模块的 state_dict() 函数返回的键完全匹配。

> state_dict()

返回包含模块整个状态的字典。 包括参数和持久缓冲区（例如运行平均值）。键是对应的参数和缓冲区名称。不包括设置为 None 的参数和缓冲区。常用于保存模型参数。

保存模型例子：

```python
# Additional information
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
```

加载模型例子：
```python
model = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```
更多详情参考：[https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)

> _apply(fn)

+ 对所有的module、parameter、buffer都进行一个fn

Example：.cpu / .cuda()源码

```python
class Module:
    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cuda(device))
```

## 三、hook方法

为了节省内存，pytorch在计算过程中不保存中间变量，包括中间层的特征图和非叶子张量的梯度。为了访问网络的中间变量，我们需要注册`hook`来导出中间变量。利用它，我们可以不必改变网络输入输出的结构，方便地获取、改变网络中间层变量的值和梯度。

`hook`方法有四种：

+ torch.Tensor.register_hook()
+ torch.nn.Module.register_forward_hook()
+ torch.nn.Module.register_backward_hook()
+ torch.nn.Module.register_forward_pre_hook()

### 3.1 torch.Tensor.register_hook(hook_fn)

注册一个反向传播hook函数hook_fn，针对tensor的register_hook函数接收一个输入参数hook_fn，为自定义函数名称。在每次调用backward函数之前都会先调用hook_fn函数。hook_fn函数同样接收一个输入参数，为torch.Tensor张量的梯度。

例子：

```python
import torch

# x,y 为leaf节点，也就是说，在计算的时候，PyTorch只会保留此节点的梯度值
x = torch.tensor([3.], requires_grad=True)
y = torch.tensor([5.], requires_grad=True)

# a,b,c 均为中间变量，在计算梯度时，此部分会被释放掉
a = x + y
b = x * y
c = a * b
# 新建列表，用于存储hook函数保存的中间梯度值
a_grad = []
def hook_grad(grad):
    a_grad.append(grad)

# register_hook的参数为一个函数
handle = a.register_hook(hook_grad)
c.backward()

# 只有leaf节点才会有梯度值
print('gradient:', x.grad, y.grad, a.grad, b.grad, c.grad)
# hook函数保留中间变量a的梯度值
print('hook函数保留中间变量a的梯度值:', a_grad[0])
# 移除hook函数
handle.remove()
```

输出：

gradient: tensor([55.]) tensor([39.]) None None None

hook函数保留中间变量a的梯度值: tensor([15.])

### 3.2 torch.nn.Module.register_forward_hook(hook_fn)

用法：在神经网络模型module上注册一个forward_hook函数hook_fn，register_forward_hook函数接收一个输入参数hook_fn，为自定义函数名称。注：在调用hook_fn函数的那个模型（层）进行前向传播并计算得到结果之后才会执行hook_fn函数，因此修改output值仅会对后续操作产生影响。hook_fn函数接收三个输入参数：module，input，output，其中module为当前网络层，input为当前网络层输入数据，output为当前网络层输出数据。例子：

```python
import timm
import torch
from torch import nn

def print_shape(model, input, output):
    print(model)
    print(input[0].shape, '=>', output.shape)
    print('====================================')


def get_children(model: nn.Module):
    # get children from model(取出所有子model及嵌套model)
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
        for child in children:
            try: 
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

model_name = 'vgg11'
model = timm.create_model(model_name, pretrained=True)
flatt_children = get_children(model)
for layer in flatt_children:
    layer.register_forward_hook(print_shape)

batch_input = torch.randn(4,3,299,299)
model(batch_input)
```

### 3.3 torch.nn.Module.register_forward_pre_hook(hook_fn)

功能：用来导出或修改指定子模型的输入张量，需要使用return返回修改后的output值使操作生效。

用法：在神经网络模型module上注册一个forward_pre_hook函数hook_fn，register_forward_pre_hook函数接收一个输入参数hook_fn，为自定义函数名称。注：在调用hook_fn函数的那个模型（层）进行前向传播操作之前会先执行hook_fn函数，因此修改input值会对该层的操作产生影响，该层的运算结果被继续向前传递。hook_fn函数接收两个输入参数：module，input，其中module为当前网络层，input为当前网络层输入数据。下面代码执行的功能是 3 × 3 3 \times 33×3 的卷积和 2 × 2 2 \times 22×2 的池化。我们使用register_forward_pre_hook函数修改中间卷积层输入的张量。

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.pool1 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        print("-------------执行forward函数-------------")
        print("卷积层输入：",x)
        x = self.conv1(x)
        print("卷积层输出：",x)
        x = self.pool1(x)
        print("池化层输出：",x)
        print("-------------结束forward函数-------------")
        return x

# module为net.conv1
# data_input为net.conv1层输入
def forward_pre_hook(module, data_input):
    print("-------------执行forward_pre_hook函数-------------")
    input_block.append(data_input)
    #print("修改前的卷积层输入：{}".format(data_input))
    #data_input = torch.rand((1, 1, 4, 4))
    #print("修改后的卷积层输入：{}".format(data_input))
    print("-------------结束forward_pre_hook函数-------------")
    #return data_input

# 初始化网络
net = Net()
net.conv1.weight[0].detach().fill_(1)
net.conv1.weight[1].detach().fill_(2)
net.conv1.bias.data.detach().zero_()

# 注册hook
input_block = list()
handle = net.conv1.register_forward_pre_hook(forward_pre_hook)

# inference
fake_img = torch.ones((1, 1, 4, 4))  # batch size * channel * H * W
output = net(fake_img)
handle.remove()

# 观察
print("神经网络模型输出：\noutput shape: {}\noutput value: {}\n".format(output.shape, output))
```

![tv2Bhb](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/tv2Bhb.png)

### 3.4 torch.nn.Module.register_backward_hook(hook)

网络在进行反向传播时，可以通过register_backward_hook来获取中间层的梯度输入和输出，常用来实现特征图梯度的提取。

