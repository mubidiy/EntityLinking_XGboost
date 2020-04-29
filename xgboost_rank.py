# coding=utf-8

import xgboost as xgb
import numpy as np
from set_config import Config


class XgboostRank(object):
    def __init__(self, model_path):
        self.model = None
        self.param = {'booster': 'gbtree', 'max_depth': 5, 'eta': 0.01, 'silent': 1, 'objective': 'rank:pairwise',
                      'gamma': 0.2, 'lambda': 2, 'subsample': 0.8, 'seed': 1}
        self.num_round = 1000
        self.model_path = model_path

    def read_data_from_file(self, file_name):
        """ 读取rank数据对应的feature文件 """
        y_list = []
        x_list = []
        with open(file_name) as fp:
            for line in fp:
                features = line.strip()
                ulineList = features.split('\t')
                _y = np.int(ulineList[0])
                _x = [np.float(x.split(':')[1]) for x in ulineList[2:]]
                y_list.append(_y)
                x_list.append(_x)
        return np.array(y_list), np.array(x_list)

    def read_group(self, file_name):
        """  读取rank数据对应的group文件 """
        group_list = []
        for line in open(file_name):
            uline = line.strip()
            group_count = np.int(uline)
            group_list.append(group_count)
        return np.array(group_list)

    def train_models(self, feature_path, group_path):
        """  训练模型  """
        y, x = self.read_data_from_file(feature_path)
        group_list = self.read_group(group_path)
        dtrain = xgb.DMatrix(x, label=y)
        dtrain.set_group(group_list)
        self.model = xgb.train(self.param, dtrain, self.num_round)
        self.model.save_model(self.model_path)
        self.model.dump_model(self.model_path + '.dump.txt')
        return

    def load_rank_model(self, model_path=None):
        """  加载保存的模型  """
        _model_path = model_path if model_path else self.model_path
        self.model = xgb.Booster()
        self.model.load_model(_model_path)
        return self.model

    def compute_precision(self, y, preds, group_list):
        num = len(group_list)
        correct = 0
        i = 0
        j = group_list[0]
        group_index = 0
        while 1:
            group_index += 1
            if group_index >= num:
                break
            _y_list = y[i:j]
            _preds_list = preds[i:j]
            _y_index = _y_list.index(min(_y_list))
            _preds_index = _preds_list.index(min(_preds_list))
            if _y_index == _preds_index:
                correct += 1
            i = j
            j = i + group_list[group_index]
        preci = float(correct) / num
        return preci

    def predict_from_file(self, feature_path, group_path):
        """  计算预测准确率  """
        y, x = self.read_data_from_file(feature_path)
        dtest = xgb.DMatrix(x, label=y)
        group_list = self.read_group(group_path)
        dtest.set_group(group_list)
        preds = self.model.predict(dtest)

        # 计算group准确率
        precision = self.compute_precision(list(y), list(preds), list(group_list))
        return precision

    ''' 训练xgboost模型'''
    def controller_train(self):

        feature_path = config.feature_train_path
        group_path = config.group_train_path

        self.train_models(feature_path, group_path)
        print("在训练集上的准确率：" )
        print(self.predict_from_file(feature_path, group_path))

    '''测试模型'''
    def controller_test(self):

        feature_path = config.feature_test_path
        group_path = config.group_test_path

        self.load_rank_model()
        print("在测试集上的准确率：")
        print(self.predict_from_file(feature_path, group_path))


if __name__ == "__main__":
    config = Config()
    model_path = config.model_path

    xgboost_rank = XgboostRank(model_path)
    # 训练模型
    xgboost_rank.controller_train()  # 0.906

    # 测试模型
    # xgboost_rank.controller_test()  # 0.865


