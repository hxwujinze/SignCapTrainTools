def train(model,
          epoch_num,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          print_inter=50,
          val_inter=100
          ):
    """
    通用训练函数
    将训练的过程与模型以及训练所需的内容解耦合
    只需给出所需的一系列东西即可训练
    :param model: 模型
    :param epoch_num: epoch总数
    :param optimizer: 优化器
    :param criterion: 损失函数 nn.XX Loss
    :param exp_lr_scheduler: LR 规划器
    :param data_set: 数据集对象 继承自torch中DateSet对象，
                    Dataloader从中load数据并在训练中进行feed
    :param data_loader: torch的data loader
    :param save_dir: log及model param保存位置
    :param print_inter: 输出loss的epoch间隔
    :param val_inter: 进行测试的epoch间隔
    :return:
    """
