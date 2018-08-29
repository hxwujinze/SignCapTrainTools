import os
import time

import numpy as np
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
          test_result_output_func,
          save_dir,
          cuda_mode=None,
          print_inter=2,
          val_inter=50,
          scheduler_step_inter=100,
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
    :param test_result_output_func: how to print the test result after test
    :param save_dir: log及model param保存位置
    :param cuda_mode: use which GPU for train ?
                      give GPU ID, if is None use CPU
    :param print_inter: 输出loss的epoch间隔
    :param val_inter: 进行测试的epoch间隔
    :param scheduler_step_inter how many epoch passed let scheduler step once

    :return:
    """
    if cuda_mode is not None:
        torch.cuda.set_device(cuda_mode)
        model.cuda(cuda_mode)
    else:
        model.cpu()
    start_time_raw = time.time()
    start_time = time.strftime('%H:%M:%S', time.localtime(start_time_raw))
    print('start_at: %s' % start_time)

    # start training
    # epoch: 用所有训练数据跑一遍称为一次epoch
    accuracy_res = ""
    try:
        for epoch in range(EPOCH + 1):
            loss_his = []
            # if epoch % scheduler_step_inter == 0 and epoch != 0:
            #     optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

            for batch_x, batch_y in data_loader['train']:
                if model_name.startswith("cnn"):
                    batch_x = batch_x.cpu()
                else:
                    batch_x = [each.cpu() for each in batch_x]
                batch_y = batch_y.cpu()

                if cuda_mode is not None:
                    if model_name.startswith("cnn"):
                        batch_x = batch_x.cuda()
                    else:
                        batch_x = [Variable(each).cuda() for each in batch_x]
                    batch_y = Variable(batch_y).float().cuda()

                # in the siamese train mode ,
                # it may come two output, two input, so need to wrap them in tuple/list
                if model_name.startswith("cnn"):
                    batch_out = model(batch_x)
                    batch_out = torch.squeeze(batch_out)
                    loss = loss_func(batch_out, batch_y)
                else:
                    batch_out = model(*batch_x)

                    loss = loss_func(batch_out[0], batch_out[1], batch_y)

                loss_his.append(loss.data.float()[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_val = np.mean(np.array(loss_his))

            if epoch % print_inter == 0:
                print("epoch %d loss %s" % (epoch, loss_val))

            if epoch % val_inter == 0:

                # start testing
                model.eval()
                model.cpu()
                # 转换为求值模式
                test_result_list = []
                for test_x, target_y in data_loader['test']:
                    if model_name.startswith("cnn"):
                        test_x = test_x.cpu()
                    else:
                        test_x = [Variable(each).cpu() for each in test_x]
                    target_y = Variable(target_y).cpu()
                    if model_name.startswith("cnn"):
                        test_output = model(test_x).cpu()
                    else:
                        test_output = model(*test_x)

                    # in cnn model get max probability category label and ground-truth label
                    # in verify model get dissimilarities ground-truth result

                    # only classify mode need max index

                    if model_name.startswith("cnn"):
                        test_output = get_max_index(test_output)
                        test_output = test_output.item()

                    target_y = target_y.data.float()[0]  # new style of get value in tensor
                    test_result_list.append((target_y, test_output))

                accuracy_res = test_result_output_func(test_result_list, epoch=epoch, loss=loss_val)
                model.train()
                model.cuda()

    except KeyboardInterrupt:
        print("stop train\n save model ?")
        res = input()
        if res != 'y':
            return

    end_time_raw = time.time()
    end_time = time.strftime('%H:%M:%S', time.localtime(end_time_raw))
    print('end_at: %s' % end_time)

    cost_time = end_time_raw - start_time_raw
    cost_time = time.strftime('%H:%M:%S', time.gmtime(cost_time, ))
    print('cost time: %s' % cost_time)

    end_time = time.strftime('%m-%d,%H-%M', time.localtime(end_time_raw))
    model = model.cpu()
    torch.save(model.state_dict(), os.path.join(save_dir, '%s_model%s.pkl' % (model_name, end_time)))

    file = open(os.path.join(save_dir, '%s_models_info_%s.txt' % (model_name, end_time)), 'w')
    info = 'data_set_size:%d\n' % len(data_set['train']) + \
           str(accuracy_res) + \
           'loss: %f\n' % loss_val + \
           'Epoch: %d\n' % EPOCH + accuracy_res
    info += str(model)

    file.writelines(info)
    file.close()

def get_max_index(tensor):
    # print('置信度')
    # print(tensor.data.float()[0])
    tensor = torch.max(tensor, dim=1)[1]
    # 对矩阵延一个固定方向取最大值
    return torch.squeeze(tensor).data.int()
