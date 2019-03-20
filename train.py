# coding:utf-8

import model_both
import pickle
import test

train_x, train_y, val_x, val_y, word2id, tags, image_emb = pickle.load(open('document/msra-ner.pk', 'rb'))
radical_train, radical_val, pinyin_train, pinyin_val, rad_train, rad_val = pickle.load(open('char/char_train_val_ner.pkl', 'rb'))
radical2id, pinyin2id, rad2id, id2id_radical, id2id_pinyin, id2id_rad = pickle.load(open('char/all_char_dic.pkl', 'rb'))

para = {}
para['max_words'] = len(train_x[0])
para['word_vocab_size'] = len(word2id.keys()) + 1
para['tag_label_size'] = len(tags)
para['rad_vocab_size'] = len(rad2id.keys()) + 1
para['radical_vocab_size'] = len(radical2id.keys()) + 1
para['pinyin_vocab_size'] = len(pinyin2id.keys()) + 1


model_path = "weights-rad-ner"
char_train = rad_train
char_val = rad_val
model = model_both.train_model(model_path, train_x, train_y, char_train, val_x, val_y, char_val, para, options='Rad')

# test
pred_y, val_y = test.prdict(model_path=model_path, val_x=val_x, val_y=val_y, tags=tags, para=para, options='Rad')
test.char_seg_acc(pred_y, val_y)
P, R, F = test.word_pos_F1(pred_y, val_y)
print("P:"+str(P))
print("R:"+str(R))
print("F1:"+str(F))
