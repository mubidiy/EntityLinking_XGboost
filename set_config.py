# coding=utf-8

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))
# print(pwd_path)


class Config(object):
    embedding_path = os.path.join(pwd_path + "word_vec_300.bin")

    origin_data_path = os.path.join(pwd_path + '/data' + "origin_data.txt")
    manual_data_path = os.path.join(pwd_path + '/data' + "manual_data.txt")

    combine_filter_path = os.path.join(pwd_path + '/data' + "combine_filter.txt")
    target_combine_path = os.path.join(pwd_path + '/data' + "target_combine.txt")

    # 训练数据集
    target_train_path = os.path.join(pwd_path + '/data' + "target_combine_train.txt")
    feature_train_path = os.path.join(pwd_path + '/data' + "feature_combine_train.txt")
    group_train_path = os.path.join(pwd_path + '/data' + "group_combine_train.txt")
    # 测试数据集
    target_test_path = os.path.join(pwd_path + '/data' + "target_combine_test.txt")
    feature_test_path = os.path.join(pwd_path + '/data' + "feature_combine_test.txt")
    group_test_path = os.path.join(pwd_path + '/data' + "group_combine_test.txt")

    model_path =os.path.join(pwd_path + '/model' + "xgboost_model")