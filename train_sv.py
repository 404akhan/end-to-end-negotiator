"""
This file have
1) base Model class
2) code to train it from data provided at https://github.com/facebookresearch/end-to-end-negotiator
3) saves model parameters after training
"""

import tensorflow as tf 
import numpy as np
import data
import domain

class Model():
	def __init__(self, corpus, name):
		# parameters
		self.ctx_size = 11
		self.ctx_emb_size = 64
		self.ctx_hid_size = 64

		self.inpt_size = 463
		self.inpt_emb_size = 256
		self.lang_hid_size = 128
		
		self.bsize = 16
		self.attn_hid_size = 256
		self.sel_hid_size = 256
		self.item_num_class = len(corpus.item_dict)
		self.item_out_length = corpus.output_length 
		self.corpus = corpus

		# build the graph
		with tf.variable_scope(name) as scope:
			self._build_pl_val()
			self._build_ctx_graph()
			self._build_lang_graph()
			self._build_attn_graph()

			self._build_predictions()
			self._prepare_loss()

	def _build_pl_val(self):
		# placeholders
		self.ctx_count = tf.placeholder(shape=(None, 3), dtype=tf.int32)
		self.ctx_val = tf.placeholder(shape=(None, 3), dtype=tf.int32)

		self.inpt = tf.placeholder(shape=(None, None), dtype=tf.int32)
		self.tgt = tf.placeholder(shape=(None, None), dtype=tf.int32)
		self.init_lang_state = tf.placeholder(tf.float32, [None, self.lang_hid_size])

		self.item_target = tf.placeholder(shape=(self.item_out_length, None), dtype=tf.int32)
		
		# state of gru cell, during sampling size 1, during training size 16
		self.lang_state_one = np.zeros([1, self.lang_hid_size])
		self.lang_state_batch = np.zeros([self.bsize, self.lang_hid_size])

	def _build_ctx_graph(self):
		# non linearites to get from context -> ctx_h representation
		matrix_count = tf.Variable(tf.random_uniform([self.ctx_size, self.ctx_emb_size], -1.0, 1.0), dtype=tf.float32)
		matrix_val = tf.Variable(tf.random_uniform([self.ctx_size, self.ctx_emb_size], -1.0, 1.0), dtype=tf.float32)

		count_embedded = tf.nn.embedding_lookup(matrix_count, self.ctx_count)
		val_embedded = tf.nn.embedding_lookup(matrix_val, self.ctx_val)

		h_tmp = tf.multiply(count_embedded, val_embedded)
		h_tmp = tf.tanh(h_tmp)
		h_tmp = tf.reshape(h_tmp, [-1, 3 * self.ctx_emb_size])
		self.ctx_h = tf.contrib.layers.fully_connected(h_tmp, self.ctx_hid_size, activation_fn=None)

	def _build_lang_graph(self):
		# build rnn for language
		matrix_inpt = tf.Variable(tf.random_uniform([self.inpt_size, self.inpt_emb_size], -1.0, 1.0), dtype=tf.float32)
		self.inpt_embedded = tf.nn.embedding_lookup(matrix_inpt, self.inpt)

		ctx_h_time = tf.tile(tf.expand_dims(self.ctx_h, 1), [1, tf.shape(self.inpt_embedded)[1], 1])
		lang_input = tf.concat([self.inpt_embedded, ctx_h_time], 2)
		lang_input = tf.layers.dropout(lang_input)

		lang_cell = tf.contrib.rnn.GRUCell(self.lang_hid_size)
		self.lang_out, self.lang_final_state = tf.nn.dynamic_rnn(lang_cell, lang_input, initial_state=self.init_lang_state, dtype=tf.float32)

		lang_decoded = tf.contrib.layers.fully_connected(self.lang_out, self.inpt_emb_size)
		self.lang_logit = tf.contrib.layers.fully_connected(lang_decoded, self.inpt_size, activation_fn=None)

	def _get_attn_probs(self, attn_out):
		# used in building attn_graph
		attn_bsize, seq_size, _ = tf.unstack(tf.shape(attn_out))

		tmp = tf.reshape(attn_out, (attn_bsize * seq_size, 2 * self.attn_hid_size))
		tmp = tf.contrib.layers.fully_connected(tmp, self.attn_hid_size, activation_fn=tf.tanh)
		tmp = tf.contrib.layers.fully_connected(tmp, 1, activation_fn=None)
		logit = tf.reshape(tmp, (attn_bsize, seq_size))
		prob = tf.nn.softmax(logit)
		prob = tf.tile(tf.expand_dims(prob, 2), [1, 1, 2*self.attn_hid_size])

		return prob

	def _build_attn_graph(self):
		# build attn_graph to get output choices
		attn_input = tf.concat([self.lang_out, self.inpt_embedded], 2)
		attn_input = tf.layers.dropout(attn_input)

		attn_cell = tf.contrib.rnn.GRUCell(self.attn_hid_size)
		attn_out, _ = tf.nn.bidirectional_dynamic_rnn(
			cell_fw=attn_cell,
			cell_bw=attn_cell,
			dtype=tf.float32,
			inputs=attn_input)
		attn_out = tf.concat(attn_out, 2)
		
		prob = self._get_attn_probs(attn_out)
		attn = tf.reduce_sum(tf.multiply(prob, attn_out), 1)

		h_tmp = tf.concat([attn, self.ctx_h], 1)
		h_tmp = tf.layers.dropout(h_tmp)
		h_tmp = tf.contrib.layers.fully_connected(h_tmp, self.sel_hid_size, activation_fn=tf.tanh)

		self.item_out = [tf.contrib.layers.fully_connected(h_tmp, self.item_num_class, activation_fn=None) for _ in range(self.item_out_length)]

	def _build_predictions(self):
		# used for sampling next word and for sampling output choices
		self.lang_softmax = tf.nn.softmax(self.lang_logit * 2)
		self.item_softmax = [tf.nn.softmax(out) for out in self.item_out]

	def _prepare_loss(self):
		# loss preparation
		lang_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(self.tgt, depth=self.inpt_size, dtype=tf.float32),
			logits=self.lang_logit
		)

		losses_item = []
		for i in range(self.item_out_length):
			loss_tmp = tf.nn.softmax_cross_entropy_with_logits(
				labels=tf.one_hot(self.item_target[i], depth=self.item_num_class, dtype=tf.float32),
				logits=self.item_out[i]
			)

			bad_tokens = self.corpus.item_dict.w2i(['<disconnect>', '<disagree>'])
			mask_item = 1. - tf.to_float(tf.logical_or(tf.equal(self.item_target[i], bad_tokens[0]), 
				tf.equal(self.item_target[i], bad_tokens[1])))

			loss_tmp = loss_tmp * mask_item
			losses_item.append(loss_tmp)

		self.loss = tf.reduce_mean(lang_cross_entropy)
		self.loss += tf.reduce_mean(losses_item) * 0.5

		self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


def print_ctx(model, count, val):
	ctx = model.corpus.context_dict.i2w(np.concatenate((count[0], val[0])))
	print('book=(count:%s value:%s) hat=(count:%s value:%s) ball=(count:%s value:%s)' % 
		(ctx[0], ctx[3], ctx[1], ctx[4], ctx[2], ctx[5]))


def print_words(words):
	for word in words: 
		print(corpus.word_dict.get_word(word), end=" ")
	print("")


def print_items(sess, model, count, val, words):
	# to print output choices, we need to find max probable that is in valid set
	my_domain = domain.get_domain('object_division')
	count_w, val_w = model.corpus.context_dict.i2w(count[0]), model.corpus.context_dict.i2w(val[0])

	ctx = [str(count_w[0]), str(val_w[0]), str(count_w[1]), str(val_w[1]), str(count_w[2]), str(val_w[2])]
	choices = my_domain.generate_choices(ctx)

	idxs = [model.corpus.item_dict.w2i(c) for c in choices]
	probs = sess.run(model.item_softmax, {
		model.ctx_count: count, 
		model.ctx_val: val, 
		model.inpt: words, 
		model.init_lang_state: model.lang_state_one, 
	})

	probs = np.array(probs)[:, 0, :]
	probs_arr = []

	for i in range(len(idxs)):
		choice = idxs[i]
		prob_of_choice = 1.
		for j in range(6):
			prob_of_choice *= probs[j][choice[j]]

		probs_arr.append(prob_of_choice)

	best_choce = idxs[np.argmax(probs_arr)]
	items = model.corpus.item_dict.i2w(best_choce) 

	for item in items: 
		print(item, end=" ")
	print("\n")


def sample(words_dstr):
	dstr = words_dstr[0][0]
	word = np.random.choice(np.arange(len(dstr)), p=dstr)

	return [[word]]


def sample_seq(sess, model, count, val):
	# sample words, form initial given word
	for start in ['YOU:', 'THEM:']:
		words = [[model.corpus.word_dict.get_idx(start)]]
		hidden = model.lang_state_one

		words_collect = []
		
		for i in range(100):
			words_collect.append(words[0][0])
			words_dstr, hidden = sess.run([model.lang_softmax, model.lang_final_state], {
				model.inpt: words, 
				model.init_lang_state: hidden, 
				model.ctx_count: count, 
				model.ctx_val: val
			})
			words = sample(words_dstr)
			if words[0][0] == model.corpus.word_dict.get_idx('<selection>'):
				break
		words_collect.append(words[0][0])

		print_ctx(model, count, val)
		print_words(words_collect)
		print_items(sess, model, count, val, [words_collect])


# data loading
corpus = data.WordCorpus('end-to-end-negotiator/src/data/negotiate', freq_cutoff=20, verbose=True)
# model
model = Model(corpus, 'SV')
traindata = corpus.train_dataset(model.bsize)
trainset, _ = traindata

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for epoch in range(50):
		print('\n----- started epoch %d' % epoch)	
		losses_epoch = []

		for counter, batch in enumerate(trainset):
			count, val, inputs, targets, items, _ = batch

			# minibatch update
			loss_val, _ = sess.run([model.loss, model.train_op], {
				model.ctx_count: count, 
				model.ctx_val: val, 
				model.inpt: inputs, 
				model.tgt: targets, 
				model.item_target: items,
				model.init_lang_state: model.lang_state_batch, 
			})

			losses_epoch.append(loss_val)

		print('----- loss is %f\n' % np.mean(losses_epoch))	
		sample_seq(sess, model, [count[1]], [val[1]])
		sample_seq(sess, model, [count[3]], [val[3]])
		sample_seq(sess, model, [count[12]], [val[12]])

		saver.save(sess, 'model/sv-named', global_step=epoch)
