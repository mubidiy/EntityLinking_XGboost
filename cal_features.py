# coding=utf-8

import numpy as np
import re
from set_config import Config


class CalFeatures(object):

    def __init__(self, embedding_path,
                 combine_filter_path, combine_target_path,
                 feature_path, group_path):
        self.embedding_path = embedding_path
        self.embedding_size = 300
        self.embdding_dict = self.load_embedding()

        self.combine_filter_path = combine_filter_path
        self.combine_target_path = combine_target_path
        self.feature_path = feature_path
        self.group_path = group_path

    def process_source_data_find_true_entity(self):
        """
        target_path 文件格式：mention \t sentence \t sent_to_words \t candidate entities
        """
        with open(self.combine_filter_path, "r", encoding='utf-8') as filter_file:
            with open(self.combine_target_path, "w") as target_file:

                qid = 0
                origin_data = ""
                manual_data = ""
                sentence = ""
                sent_to_word = []  # 存储 sentence中的分词结果
                mention_entity_dic = {}
                for doc in filter_file:
                    qid += 1
                    if doc == "\n":  # 每4次一个循环
                        if origin_data != "" and manual_data != "":
                            origin_data = manual_data = ""
                            sentence = ""
                            sent_to_word = []
                            mention_entity_dic = {}
                        continue
                    if origin_data == "":
                        origin_data = doc  # 存储初始分词结果数据
                    elif manual_data == "":
                        manual_data = doc  # 存储人工确认的分词结果数据
                    # print("origin_data: " + origin_data)
                    # print("manual_data: " + manual_data)

                    if origin_data != "" and manual_data == "":  # 处理初始分词结果中的数据
                        p1 = re.compile(r'〖(.*)〗', re.S)
                        p2 = re.compile(r'(\d+ words)')
                        sentence = re.findall(p1, origin_data)
                        num_words = re.findall(p2, origin_data)
                        sentence = str(sentence[0])  # 取出句子 ,不带〖〗符号
                        num_words = str(num_words[0])  # 取出结尾的\d+ words
                        # print(sentence)
                        # print(num_words)

                        index_start = origin_data.find("分词结果") + 5
                        index_end = origin_data.find(num_words)
                        items = origin_data[index_start: index_end - 1]
                        items = items.split()  # 存储mention及其候选实体

                        for item in items:
                            mention = item[:item.find("(")]
                            sent_to_word.append(mention)  # 存储句子的mention

                        for item in items:
                            cand_entities = []
                            if "(OOV)" in item: continue   # 忽略没有候选实体的mention
                            mention = item[:item.find("(")]
                            entities_items = item[item.find("(") + 1: item.find(")")].split("|")  # 存储每个mention的候选实体

                            for entity_item in entities_items:
                                for ch in ["!", "近类", "词类", "父类"]:
                                    if ch in entity_item:
                                        entity_item = entity_item.replace(ch, "")
                                cand_entity = entity_item
                                cand_entities.append(cand_entity)
                            mention_entity_dic[mention] = cand_entities  # mention_entity_dic = {mention, cand_entities, ...}
                        # print(mention_entity_dic)

                    if origin_data !="" and manual_data !="":  # # 处理人工确认的分词结果中的数据
                        p2 = re.compile(r'(wordpat: \d+)')
                        num_words = re.findall(p2, manual_data)
                        # print(num_words)
                        num_words = str(num_words[0])  # 取出结尾的\d+ words
                        index_start = manual_data.find(num_words) + len(num_words) + 1
                        items = manual_data[index_start:]  # 存储 标签数据

                        true_entity_list = self.extract_true_entity(items)  # 抽取正确实体列表

                        for true_entity in true_entity_list:  # 遍历正确的实体，找到对应的mention
                            cand_entities = []
                            for mention in mention_entity_dic:
                                if true_entity in mention_entity_dic[mention]:
                                    cand_entities.append(true_entity)   # 将mention的正确实体放在第一的位置
                                    for entity in mention_entity_dic[mention]:
                                        if entity != true_entity and entity not in true_entity_list:
                                            cand_entities.append(entity)
                                    target_file.write( mention + "\t" + sentence + "\t" + " ".join(sent_to_word) + "\t"
                                                       + " ".join(cand_entities) + "\n")
                                    break
                    print(qid)
                print(" process source data to find true entity finish")

    def extract_true_entity(self, string):
        """
        抽取正确的实体列表
        :param string:   标签数据字符串
        :return:
        """
        i = 0
        str_list = []
        true_entity_list = []
        while i < (len(string) - 2):
            if string[i] == "[":  # 处理 [<!反馈近类>]
                i += 3
            elif string[i] == "<" and string[i + 1] == "!":  # 处理<!杆路近类>*<!问题近类>
                str = ""
                i += 2
                while string[i] != ">":
                    str += string[i]
                    i += 1
                str_list.append((str))
            elif string[i] == "|" and string[i + 1] == "!":  # 处理 <怎么|!怎么近类|!怎样近类|!这么近类>
                str = ""
                i += 2
                while True:
                    if string[i] == "|":
                        i -= 1
                        break
                    if string[i] == ">":
                        break
                    str += string[i]
                    i += 1
                str_list.append(str)
            i += 1

        for str in str_list:
            for ch in ["!", "近类", "词类", "父类"]:
                if ch in str:
                    str = str.replace(ch, "")
            true_entity = str
            true_entity_list.append(true_entity)

        return true_entity_list

    def cal_feature(self):
        """
        feature_path 文件格式：flag \t qid:value \t name_dis:value \t pv:value \t summary_cos:value
        """
        with open(self.combine_target_path, "r", encoding='utf-8') as target_file:
            with open(self.feature_path, "w") as feature_file:
                qid = 0
                for line in target_file:
                    qid += 1
                    line = line.split("\t")
                    mention = line[0]
                    sentence = line[1]
                    sent_to_words = line[2]
                    cand_entities = line[3].split()

                    mention_vec = self.get_phrasevector(mention)
                    sent_vec = self.rep_sentencevector(sent_to_words)
                    num = 0
                    for entity in cand_entities:
                        if num==0: flag = 1
                        else: flag = 2

                        # 计算features
                        entity_vec = self.get_phrasevector(entity)
                        name_dis = round(self.similarity_cosine(mention_vec, entity_vec), 6)
                        pv = 0.1 if num < 3 else 0.0
                        summary_cos = round(self.similarity_cosine(sent_vec, entity_vec), 6)
                        num += 1
                        feature_file.write(str(flag) + "\t" +
                                           "qid:" + str(qid) + "\t" +
                                           "name_dis:" + str(name_dis) + "\t" +
                                           "pv:" + str(pv) + "\t" +
                                           "summary_cos:" + str(summary_cos) + "\n")
                    print(qid)
                print("calculate feature finish")

    def build_group_file(self):
        """
        group_path 文件格式：count \n
        """
        with open(self.combine_target_path, "r", encoding='utf-8') as target_file:
            with open(self.group_path, "w") as group_file:
                for line in target_file:
                    line = line.split("\t")
                    cand_entities = line[3].split()
                    # print(cand_entities)
                    group_file.write(str(len(cand_entities)) + "\n")
                print("build group file finish")

    def load_embedding(self):
        """ 加载词向量 """
        embedding_dict = {}
        count = 0
        with open(self.embedding_path, "r") as embedding_file:
            for line in embedding_file:
                line = line.strip().split(' ')
                if len(line) < 300:
                    continue
                wd = line[0]
                vector = np.array([float(i) for i in line[1:]])
                embedding_dict[wd] = vector
                count += 1
                if count%10000 == 0:
                    print(count, 'loaded')
        print('loaded %s word embedding, finished'%count)
        return embedding_dict

    def rep_sentencevector(self, sent_to_words):
        """基于wordvector，通过lookup table的方式找到句子的wordvector的表示"""
        word_list = sent_to_words.split()
        embedding = np.zeros(self.embedding_size)
        sent_len = 0
        for index, word in enumerate(word_list):
            embedding += self.get_phrasevector(word)
            sent_len += 1
        return embedding/sent_len

    def get_wordvector(self, word):
        """ 获取单个词的词向量 """
        return np.array(self.embdding_dict.get(word, [0]*self.embedding_size))

    def get_phrasevector(self, phrase):
        """ 获取短语的词向量 """
        phrase_vec = 0.0
        phrase_len = 0
        for word in phrase:
            if word in self.embdding_dict:
                phrase_vec += self.get_wordvector(word)
                phrase_len += 1
        phrase_vec += self.get_wordvector(phrase)
        phrase_len += 1
        return phrase_vec / phrase_len

    def similarity_cosine(self, vector1, vector2):
        """ 计算余弦距离 """
        cos = 0.0
        vector1_norm = np.linalg.norm(vector1)
        vector2_norm = np.linalg.norm(vector2)
        if len(vector1) == len(vector2) and len(vector1) > 0 \
                and vector1_norm != 0 and vector2_norm != 0:
            cos = np.dot(vector1, vector2) / (vector1_norm * vector2_norm)

        return cos

    def distance_words(self, sent1_to_words, sent2_to_words):
        """基于词语相似度计算句子相似度"""
        wds1 = sent1_to_words
        wds2 = sent2_to_words
        score_wds1 = []
        score_wds2 = []
        for word1 in wds1:
            score = max([self.similarity_cosine(self.get_phrasevector(word1), self.get_phrasevector(word2)) for word2 in wds2])
            score_wds1.append(score)
        for word2 in wds2:
            score = max([self.similarity_cosine(self.get_phrasevector(word2), self.get_phrasevector(word1)) for word1 in wds1])
            score_wds2.append(score)
        sim_score = max(sum(score_wds1)/len(wds1), sum(score_wds2)/len(wds2))
        return sim_score


if __name__ == "__main__":
    config = Config()
    embedding_path = config.embedding_path

    combine_filter_path = config.combine_filter_path
    combine_target_path = config.target_combine_path

    # 训练数据集
    # target_path 是 combine_target_path 划分为训练集(80%)和测试集(20%)， 可以调用函数进行划分
    target_path = config.target_train_path
    feature_path = config.feature_train_path
    group_path = config.group_train_path
    # 测试数据集
    # target_path = config.target_test_path
    # group_path = config.feature_test_path
    # group_path = config.group_test_path

    cal_fea = CalFeatures(embedding_path,
                          combine_filter_path, combine_target_path,
                          feature_path, group_path)

    # 第一步 处理数据（找到mention的正确实体）
    cal_fea.process_source_data_find_true_entity()

    # 第二步 计算数据特征
    # cal_fea.cal_feature()

    # 第三步 创建xgboost需要的group
    # cal_fea.build_group_file()







