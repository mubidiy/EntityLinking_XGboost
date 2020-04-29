# coding=utf-8

import re
from set_config import Config


class PreData(object):

    def extract_data_onefile(self, origin_data_path, manual_data_path, combine_filter_path):
        """从初始分词结果和人工确认的分词结果抽取出共同的数据集"""
        with open(origin_data_path, "r", encoding="utf-8") as data_file:
            with open(combine_filter_path, "w") as filter_file:
                id = 0
                for item in data_file:  # item  是 str类型
                    p1 = re.compile(r'〖(.*)〗', re.S)
                    sentence = re.findall(p1, item)   # sentence 是 list 类型
                    sentence = "〖" + str(sentence[0]) + "〗"  # 取出list中的字符串

                    with open(manual_data_path, "r", encoding="utf-8") as data_anno_file:
                        for item_anno in data_anno_file:
                            if sentence in item_anno:
                                id += 1
                                filter_file.write(str(id)+ "\t" + item + "\n" + str(id)+ "\t" + item_anno + "\n")
                                print(id)
                                break
                print("pre data finish")


if __name__ == "__main__":

    config = Config()
    origin_data_path = config.origin_data_path
    manual_data_path = config.manual_data_path
    combine_filter_path = config.combine_filter_path

    pre_data = PreData()
    pre_data.extract_data_onefile(origin_data_path, manual_data_path, combine_filter_path)


