# coding:utf-8

import pickle
import numpy
import utiles
from keras.preprocessing.sequence import pad_sequences


# 由train_X 通过word2id得到原来的word, 无需使用
def get_word_from_id(train_x, word2id):
    sentences = []
    id2word = {}
    for v, k in enumerate(word2id):
        id2word[v] = k

    for para in train_x:
        for sentence in para:
            sent_str = ""
            for d in sentence:
                if d != 0 and d <= 3940:
                    sent_str = sent_str + id2word[d]
            sentences.append(sent_str)
    return sentences


# 输入一个句子列表，输出对应的部首id或拼音id并padding,maxlen=特征最长长度
def get_char2id(train_x, id2id, maxlen):
    char_l = []
    lose = [0]*maxlen

    for sentence in train_x:
        sent_l = []
        for word_id in sentence:
            try:
                sent_l.append(id2id[word_id])
            except Exception, e:
                sent_l.append(lose)
        char_l.append(sent_l)
    print char_l[0][0]
    return char_l


with open('document/msra-ner.pk') as f:
    train_x, train_y, val_x, val_y, word2id, tags, image = pickle.load(f)
f.close()

# with open('char/char_train_val_seg.pkl') as f:
#     radical_train, radical_val, pinyin_train, pinyin_val, rad_train,rad_val = pickle.load(f)
# f.close()

with open('char/all_char_dic.pkl') as f:
    radical2id, pinyin2id, id2id_radical, id2id_pinyin, rad2id, id2id_rad = pickle.load(f)
f.close()


# step1: 建立特征和id对应的词典及 字id到特征id的词典  只需要跑一次
# id2id_radical, radical2id = utiles.get_id2radical(word2id)
# id2id_rad, rad2id = utiles.get_id2rad(word2id)
# id2id_pinyin, pinyin2id = utiles.get_id2pinyin(word2id)
#
# pickle.dump((radical2id, pinyin2id, rad2id, id2id_radical, id2id_pinyin, id2id_rad), open('all_char_dic.pkl', 'wb'))
# print 'ok'
# step 1 结束
# step 2: 求出train_x等对应的特征train
radical_max = 18
rad_max = 3
pinyin_max = 8

radical_train = get_char2id(train_x, id2id_radical, radical_max)
pinyin_train = get_char2id(train_x, id2id_pinyin, pinyin_max)
rad_train = get_char2id(train_x, id2id_rad, rad_max)

radical_val = get_char2id(val_x, id2id_radical, radical_max)
pinyin_val = get_char2id(val_x, id2id_pinyin, pinyin_max)
rad_val = get_char2id(val_x, id2id_rad, rad_max)
# step2 结束


# step3 radical_train list -> numpy.ndarry
rad_train = numpy.array(rad_train).reshape(len(train_x), -1)
rad_val = numpy.array(rad_val).reshape(len(val_x), -1)
radical_train = numpy.array(radical_train).reshape(len(train_x), -1)
radical_val = numpy.array(radical_val).reshape(len(val_x), -1)
pinyin_train = numpy.array(pinyin_train).reshape(len(train_x), -1)
pinyin_val = numpy.array(pinyin_val).reshape(len(val_x), -1)

# pos 才有五份
# for i in range(0, 5):
#     print i
    # radical_train[i] = numpy.array(radical_train[i]).reshape(8000, -1)
    # radical_val[i] = numpy.array(radical_val[i]).reshape(2000, -1)

    # pinyin_train[i] = numpy.array(pinyin_train[i]).reshape(8000, -1)
    # pinyin_val[i] = numpy.array(pinyin_val[i]).reshape(2000, -1)

    # rad_train[i] = numpy.array(rad_train[i]).reshape(14525, -1)
    # rad_val[i] = numpy.array(rad_val[i]).reshape(14525, -1)


pickle.dump((radical_train, radical_val, pinyin_train, pinyin_val, rad_train, rad_val), open('char/char_train_val_ner.pkl', 'wb'))

