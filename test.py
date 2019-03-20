import pickle
import numpy
import model_both


def prdict(model_path, val_x, val_y, tags, para, options):
    model = model_both.build_model(para, options=options)
    model.load_weights(filepath=model_path)
    lengths = get_lengths(val_x)
    pred_y = model.predict(val_x)
    tag_pred_y = []
    tag_val_y = []
    for i, y in enumerate(pred_y):
        y = [numpy.argmax(dim) for dim in y]
        p_y = y[-lengths[i]:]
        v_y = val_y[i][-lengths[i]:].flatten()
        p_y = [tags[dim] for dim in p_y]
        v_y = [tags[dim] for dim in v_y]
        tag_pred_y.append(p_y)
        tag_val_y.append(v_y)
    return tag_pred_y, tag_val_y


def char_seg_acc(tag_pred_y, tag_val_y):
    acc = 0.0
    num = 0.0
    for j in range(len(tag_pred_y)):
        for z in range(len(tag_pred_y[j])):
            if tag_pred_y[j][z] == tag_val_y[j][z]:
                acc += 1
            num += 1
    print("test acc:"+str(acc/num))


def word_pos_F1(y_pred, y):
    c = 0
    true = 0
    pos = 0
    for i in xrange(len(y)):
        start = 0
        for j in xrange(len(y[i])):
            if y_pred[i][j][0] == 'E' or y_pred[i][j][0] == 'S':
                pos += 1
            if y[i][j][0] == 'E' or y[i][j][0] == 'S':
                flag = True
                if y_pred[i][j] != y[i][j]:
                    flag = False
                if flag:
                    for k in range(start, j):
                        if y_pred[i][k] != y[i][k]:
                            flag = False
                            break
                    if flag:
                        c += 1
                true += 1
                start = j+1

    P = c/float(pos)
    R = c/float(true)
    F = 2*P*R/(P+R)
    return P, R, F


def get_lengths(X):
    pad_length = X.shape[1]
    lengths = []
    for i in range(len(X)):
        length = 0
        for dim in X[i]:
            if dim == 0:
                length += 1
            else:
                break
        lengths.append(pad_length-length)
    return lengths

#
# if __name__ == "__main__":
#     train_x, train_y, val_x, val_y, word2id, tags, image = pickle.load(open('document/pku.pk', 'rb'))
#     radical_train, radical_val, pinyin_train, pinyin_val, rad_train, rad_val = pickle.load(open('char/char_train_val_pku.pkl', 'rb'))
#     radical2id, pinyin2id, rad2id, id2id_radical, id2id_pinyin, id2id_rad = pickle.load(open('char/all_char_dic.pkl', 'rb'))
#
#     para = {}
#     para['max_words'] = len(train_x[0])
#     para['word_vocab_size'] = len(word2id.keys()) + 1
#     para['rad_vocab_size'] = len(rad2id.keys()) + 1
#     para['radical_vocab_size'] = len(radical2id.keys()) + 1
#     para['pinyin_vocab_size'] = len(pinyin2id.keys())
#     para['tag_label_size'] = len(tags)
#
#     pred_y, val_y = prdict(model_path="weights-rad-seg", val_x=val_x, val_y=val_y, tags=tags, para=para)
#
#     char_seg_acc(pred_y, val_y)
#     P, R, F = word_pos_F1(pred_y, val_y)
#     print("P:"+str(P))
#     print("R:"+str(R))
#     print("F1:"+str(F))
