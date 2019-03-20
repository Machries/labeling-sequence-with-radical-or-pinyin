# coding:utf-8

from keras.layers.core import Dense, Dropout, Reshape, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional, Input
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras_contrib.layers import CRF
from keras.layers import Concatenate
import keras.backend as K
from keras.layers.convolutional import Convolution1D, MaxPooling1D


def char_layer(max_words, max_char_size, char_vocab_size, char_embedding_dim, lstm_dim, dropout):
	# radical
	char_input = Input(shape=(max_char_size * max_words,), dtype='int32', name='char_input')
	char_emb = Embedding(char_vocab_size, char_embedding_dim, input_length=max_words * max_char_size, dropout=dropout, name='char_emb')(char_input)
	bilstm_char = Bidirectional(LSTM(lstm_dim, inner_init='uniform', forget_bias_init='one', return_sequences=True))(char_emb)
	bilstm_char_d = Dropout(dropout)(bilstm_char)
	char_re = Reshape((max_words, -1))(bilstm_char_d)
	char_mask = Masking(mask_value=0, )(char_re)
	return char_input, char_mask


def build_model(para, options='Both'):
	word_vocab_size = para['word_vocab_size'] # len(word2id.keys())
	max_words = para['max_words']  # train_x[0] 长度
	tag_label_size = para['tag_label_size']  # len(tags)
	radical_vocab_size = para['radical_vocab_size']
	pinyin_vocab_size = para['pinyin_vocab_size']
	rad_vocab_size = para['rad_vocab_size']

	max_radical_size = 18
	max_pinyin_size = 8
	max_rad_size = 3
	char_embedding_dim = 25
	word_embedding_dim = 100
	dropout = 0.3
	lstm_dim = 100

	word_input = Input(shape=(max_words,), dtype='int32', name='word_input')
	word_emb = Embedding(word_vocab_size, word_embedding_dim, input_length=max_words, dropout=0, name='word_emb',
						 mask_zero=True)(word_input)
	if options == 'None':
		total_emb = word_emb
	elif options == 'Radical':
		char_input, radical_mask = char_layer(max_words, max_radical_size, radical_vocab_size, char_embedding_dim,
													lstm_dim, dropout)
		total_emb = Concatenate()([word_emb, radical_mask])
	elif options == 'Pinyin':
		char_input, pinyin_mask = char_layer(max_words, max_pinyin_size, pinyin_vocab_size, char_embedding_dim,
												 lstm_dim, dropout)
		total_emb = Concatenate()([word_emb, pinyin_mask])
	elif options == 'Rad':
		char_input, rad_mask = char_layer(max_words, max_rad_size, rad_vocab_size, char_embedding_dim,
													lstm_dim, dropout)
		total_emb = Concatenate()([word_emb, rad_mask])
	elif options == "Both":
		radical_input, radical_mask = char_layer(max_words, max_radical_size, radical_vocab_size, char_embedding_dim,
												lstm_dim, dropout)
		pinyin_input, pinyin_mask = char_layer(max_words, max_pinyin_size, pinyin_vocab_size, char_embedding_dim,
										 lstm_dim, dropout)
		total_emb = Concatenate()([word_emb, radical_mask, pinyin_mask])

	emb_droput = Dropout(dropout)(total_emb)
	bilstm_word = Bidirectional(LSTM(lstm_dim, inner_init='uniform', forget_bias_init='one', return_sequences=True))(emb_droput)
	bilstm_word_d = Dropout(dropout)(bilstm_word)
	crf = CRF(tag_label_size, sparse_target=True)
	dense = TimeDistributed(Dense(tag_label_size))(bilstm_word_d)
	crf_output = crf(dense)

	if options == 'None':
		model = Model(input=word_input, output=crf_output)
	elif options == "Both":
		model = Model(input=[word_input, radical_input, pinyin_input], output=crf_output)
	else:
		model = Model(input=[word_input, char_input], output=crf_output)

	# model.compile(loss="mean_squared_error", optimizer=sgd, metrics=['sparse_categorical_accuracy'])
	model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
	model.summary()
	return model

# def train_model(filepath, model, word_x, radical_x, pinyin_x, train_y):
# 	# define the checkpoint
# 	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# 	model.fit([word_x, radical_x, pinyin_x], train_y, batch_size=64, validation_split=0.2, epochs=50, callbacks=[checkpoint])
# 	return model


def train_model(model_path, train_x, train_y, char_train, val_x, val_y, char_val, para, options):
	fold = 4
	print train_x.__len__()
	step = int(train_x.__len__() / fold)

	model = build_model(para, options=options)
	checkpoint = ModelCheckpoint(model_path, monitor='val_crf_viterbi_accuracy', verbose=1,
								 save_best_only=True, mode='max')
	model.fit([train_x, char_train], train_y, batch_size=128, epochs=30, callbacks=[checkpoint], validation_data=([val_x, char_val], val_y), shuffle=True)

	K.set_value(model.optimizer.lr, 0.0005)
	for epoch in range(15):
		for i in range(fold-1):
			print("EPOCH:"+str(15))
			model.fit([train_x[i*step:(i+1)*step], char_train[i*step:(i+1)*step]], train_y[i*step:(i+1)*step],
					  batch_size=128, epochs=1, callbacks=[checkpoint], validation_data=([val_x, char_val], val_y),
					  shuffle=True)
		model.fit([train_x[(fold-1) * step:], char_train[(fold-1) * step:]], train_y[(fold-1) * step:], batch_size=128, epochs=1, callbacks=[checkpoint],
				  validation_data=([val_x, char_val], val_y), shuffle=True)
	print("ok")


# def train_model(model_path, train_x, radical_train, pinyin_train, rad_train, train_y, val_x, radical_val, pinyin_val, rad_val, val_y, para, word2id):
# 	fold = 4
# 	print train_x.__len__()
# 	step = int(train_x.__len__() / fold)
#
# 	model = build_model(para, options='Rad')
# 	checkpoint = ModelCheckpoint(model_path, monitor='val_crf_viterbi_accuracy', verbose=1,
# 								 save_best_only=True, mode='max')
# 	# print train_x.shape
# 	# print radical_train.shape
# 	model.fit([train_x, pinyin_train], train_y, batch_size=128, epochs=30, callbacks=[checkpoint],validation_data=([val_x, rad_val],val_y),shuffle=True)
#
# 	K.set_value(model.optimizer.lr, 0.0005)
# 	for epoch in range(15):
# 		for i in range(fold-1):
# 			print("EPOCH:"+str(15))
# 			model.fit([train_x[i*step:(i+1)*step], rad_train[i*step:(i+1)*step]], train_y[i*step:(i+1)*step],
# 					  batch_size=128, epochs=1, callbacks=[checkpoint], validation_data=([val_x, rad_val], val_y),
# 					  shuffle=True)
# 		model.fit([train_x[(fold-1) * step:], rad_train[(fold-1) * step:]], train_y[(fold-1) * step:], batch_size=128, epochs=1, callbacks=[checkpoint],
# 				  validation_data=([val_x, rad_val], val_y), shuffle=True)
# 	print("ok")
