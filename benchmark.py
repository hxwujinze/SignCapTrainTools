import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from algorithm_models.CNN_model import CNN
from algorithm_models.verify_model import SiameseNetwork
from process_data_dev import DATA_DIR_PATH


class Benchmark:
    def __init__(self):
        self.classify_m = CNN()
        load_model_param(self.classify_m, 'cnn')
        self.verify_m = SiameseNetwork(train=False)
        load_model_param(self.verify_m, 'verify')
        self.verify_m.single_output()
        self.reference_vectors = os.path.join(DATA_DIR_PATH, 'reference_verify_vector')
        with open(self.reference_vectors, 'rb') as f:
            self.reference_vectors = pickle.load(f)
        with open(os.path.join(DATA_DIR_PATH, 'new_train_data'), 'rb') as f:
            offline_test_data = pickle.load(f)

        data_dict = {}
        for each in offline_test_data:
            if data_dict.get(each[1]) is None:
                data_dict[each[1]] = [each[0]]
            else:
                data_dict[each[1]].append(each[0])
        self.offline_test_data = data_dict

    def offline_test(self):
        test_label = random.choice(list(self.offline_test_data.keys()))
        test_data = random.choice(self.offline_test_data[test_label])
        print('target label %s' % str(test_label))
        test_data = np.where(test_data > 0.00000000001, test_data, 0)
        test_data = np.array([test_data.T])
        test_data = torch.from_numpy(test_data)
        test_data = test_data.double()
        inference_res = self.classify_m(test_data)
        inference_res = torch.nn.functional.softmax(inference_res, dim=1)
        # print('raw output %s' % str(inference_res))
        inference_res = get_max_index(inference_res).item()
        print('inference label %s' % str(inference_res))
        verify_vector = self.verify_m(test_data)
        refer_vector = torch.from_numpy(self.reference_vectors[inference_res])
        dis = F.pairwise_distance(verify_vector, refer_vector)
        print('dis: %f\n' % dis.item())



def get_max_index(tensor):
    # print('置信度')
    # tensor = F.softmax(tensor, dim=1)
    # print (tensor)
    tensor = torch.max(tensor, dim=1)[1]
    # 对矩阵延一个固定方向取最大值
    return torch.squeeze(tensor).data.int()


def load_model_param(model, model_name):
    for root, dirs, files in os.walk(DATA_DIR_PATH):
        for file_ in files:
            file_name_split = os.path.splitext(file_)
            if file_name_split[1] == '.pkl' and file_name_split[0].startswith(model_name):
                print('load model params %s' % file_name_split[0])
                file_ = os.path.join(DATA_DIR_PATH, file_)
                model.load_state_dict(torch.load(file_))
                model.double()
                model.eval()
                return model


def main():
    b = Benchmark()
    for i in range(2):
        start = time.clock()
        b.offline_test()
        print('cost time %f' % (time.clock() - start))


if __name__ == '__main__':
    main()
