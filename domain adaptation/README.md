# Domain Adaption

> 参考资料：https://youtu.be/Mnk_oUrgppM

Domain Shift可能出现以下三种情况：
![HO5NmF](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/HO5NmF.png)

针对Target domain 样本数和标签情况可以使用以下方法进行处理：
1. 少量含标签数据：source data训练后微调（小学习率训练几个epoch），*注意：防止在target data 过拟合*
2. 大量不含标签的数据：学习将样本映射到相近的空间分布 ——*本节主要探究的方法，见下文Domain Adversarial Training*
![zXTUtz](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/zXTUtz.png)
3. ff

## Domain Adversarial Training

> 参考文献：![[Unsupervised Domain Adaptation by Backpropagation.pdf]]

**主要思想：**将未标记的target data 通过Feature Extract映射至source data同一空间，使得Domain Classifier无法区分图像来自哪个域
![77caOc](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/77caOc.png)
![hpk4NA](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/hpk4NA.png)
![um8sXH](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/um8sXH.png)

**算法缺陷：**
对于无标签的target data而言，仅仅使得Feature Embedding映射到source data同样的空间而不考虑boundaries可能会出现下图左所示的情况。
![vELTII](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/vELTII.png)

考虑到决策边界，我们希望对于Target unlabeled data，分类得到的标签能够更加聚集，也就意味着离决策边界更远，我们更想得到下图中上方的图所示：
![CkHz1F](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/CkHz1F.png)
可参考文献：
> Used in Decision-boundary Iterative Refinement Training with a Teacher (DIRT-T)   [https://arxiv.org/abs/1802.08735](https://arxiv.org/abs/1802.08735)
> Maximum Classifier Discrepancy [https://arxiv.org/abs/1712.02560](https://arxiv.org/abs/1712.02560)



## Open&Close Set问题
在实际情况中，source domain和target domain的label set可能不是全等关系，而是包含关系，如下图：
![4larsA](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/4larsA.png)
> 来源：[Universal domain adaptation.pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/You_Universal_Domain_Adaptation_CVPR_2019_paper.pdf)
