import os
import time

import torch
import torch.nn as nn
from torch.autograd import Variable


def train(model: nn.Module,
          model_name: str,
          EPOCH,
          optimizer: torch.optim.Optimizer,
          loss_func: nn.Module,
          exp_lr_scheduler,
          data_set: dict,
          data_loader: dict,
          save_dir,
          cuda_mode=False,
          print_inter=50,
          val_inter=100
          ):
    """
        通用训练函数
    将训练的过程与模型以及训练所需的内容解耦合
    只需给出所需的一系列东西即可训练
    :param model: 模型
    :param model_name:
    :param EPOCH:  epoch总数
    :param optimizer: 优化器
    :param loss_func: 损失函数 nn.XX Loss
    :param exp_lr_scheduler: LR 规划器
    :param data_set: 数据集对象 继承自torch中DateSet对象，
                    Dataloader从中load数据并在训练中进行feed
    :param data_loader: torch的data loader
    :param save_dir: log及model param保存位置
    :param print_inter: 输出loss的epoch间隔
    :param val_inter: 进行测试的epoch间隔

    :return:
    """
    if cuda_mode:
        model.cuda()
    else:
        model.cpu()
    model.train()
    start_time_raw = time.time()
    start_time = time.strftime('%H:%M:%S', time.localtime(start_time_raw))
    print('start_at: %s' % start_time)

    # start training
    # epoch: 用所有训练数据跑一遍称为一次epoch
    accuracy_res = ""
    for epoch in range(EPOCH + 1):
        if epoch % 50 == 0 and epoch != 0:
            exp_lr_scheduler.step()

        for batch_x, batch_y in data_loader['train']:
            batch_x = Variable(batch_x).cuda()
            batch_y = Variable(batch_y).cuda()
            if not cuda_mode:
                batch_x = batch_x.cpu()
                batch_y = batch_y.cpu()

            batch_out = model(batch_x)
            batch_out = torch.squeeze(batch_out)
            loss = loss_func(batch_out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % print_inter == 0:
            print("epoch %d epoch %s" % (epoch, loss.data.float()[0]))

        if epoch % val_inter == 0:

            # start testing
            model.eval()
            model.cpu()
            # 转换为求值模式

            test_result = {}
            all_t_cnt = 0
            all_f_cnt = 0
            for test_x, test_y in data_loader['test']:

                test_input = Variable(test_x).cpu()  # 转换在gpu内跑识别
                # 转换为可读取的输入 Variable
                # 如下进行nn的正向使用 分类
                test_output = model(test_input).cpu()  # 从gpu中取回cpu算准确度
                # 需要从gpu的显存中取回内存进行计算正误率
                test_output = get_max_index(test_output)
                # softmax是14个概率的输出
                # test数据是连续的100个输入 于是输出也是一个 100 * 14 的矩阵
                target_y = test_y.item()
                test_output = test_output.item()

                if test_result.get(target_y) is None:
                    test_result[target_y] = {
                        't': 0,
                        'f': 0
                    }
                if test_output == target_y:
                    all_t_cnt += 1
                    test_result[target_y]['t'] += 1
                else:
                    all_f_cnt += 1
                    test_result[target_y]['f'] += 1
            accuracy_res = "accuracy of each sign:\n"

            for each_sign in sorted(test_result.keys()):
                t_cnt = test_result[each_sign]['t']
                f_cnt = test_result[each_sign]['f']
                accuracy_rate = t_cnt / (t_cnt + f_cnt)
                accuracy_res += "sign %d, accuracy %f (%d / %d)\n" % \
                                (each_sign, accuracy_rate, t_cnt, t_cnt + f_cnt)
            accuracy_res += "overall accuracy: %.5f\n" % (all_t_cnt / (all_f_cnt + all_t_cnt))

            print("**************************************")
            print("epoch: %s\nloss: %s\nprogress: %.2f" %
                  (epoch, loss.data.float()[0], 100 * epoch / EPOCH,))
            print(accuracy_res)
            model.train()
            model.cuda()

    end_time_raw = time.time()
    end_time = time.strftime('%H:%M:%S', time.localtime(end_time_raw))
    print('end_at: %s' % end_time)

    cost_time = end_time_raw - start_time_raw
    cost_time = time.strftime('%H:%M:%S', time.gmtime(cost_time, ))
    print('cost time: %s' % cost_time)

    end_time = time.strftime('%m-%d,%H-%M', time.localtime(end_time_raw))
    model = model.cpu()
    torch.save(model.state_dict(), os.path.join(save_dir, '%s_model%s.pkl' % (model_name, end_time)))

    file = open(os.path.join(save_dir, 'cnn_models_info_%s' % end_time), 'w')
    info = 'data_set_size:%d\n' % len(data_set['train']) + \
           str(accuracy_res) + \
           'loss: %f\n' % loss.data.float()[0] + \
           'Epoch: %d\n' % EPOCH
    info += str(model)

    file.writelines(info)
    file.close()

def get_max_index(tensor):
    # print('置信度')
    # print(tensor.data.float()[0])
    tensor = torch.max(tensor, dim=1)[1]
    # 对矩阵延一个固定方向取最大值
    return torch.squeeze(tensor).data.int()
