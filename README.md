# SignCapTrainTools
----------
归一化操作只能将不同数值量的数据线性变换到一个统一的区间内（如 0-1）,
不能拿来进行数据平移缩放操作，以消除相同动作因为采集时的力度不同等导致数值统一发生的偏移
归一化还是应该从全局数据的角度出发 为每种特征生成一个统一的scale、min，

要想进行归一化操作最好还是考虑使用pytorch的BatchNorm层，使用一个能够被训练数据进行调整的层
根据数据的规律，将数据进行缩放变换，以提高模型的健壮性
####课程视频：https://www.bilibili.com/video/av17204303/?p=15
视频中有一篇2015年的论文 可以看看了解下具体原理