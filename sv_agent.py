"""
This file have
1) base SVModel class
	explanatory comments can find in Model class at train_sv.py
2) SVAgent that uses SVModel to read, write (normal, rollout) dialogue
	used to train reinforcement counterpart
"""

import tensorflow as tf 
import numpy as np
import data
import domain

class SVModel():
	def __init__(self, corpus, name):
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

		with tf.variable_scope(name) as scope:
			self._build_pl_val()
			self._build_ctx_graph()
			self._build_lang_graph()
			self._build_attn_graph()

			self._build_predictions()
			self._prepare_loss()

	def _build_pl_val(self):
		self.ctx_count = tf.placeholder(shape=(None, 3), dtype=tf.int32)
		self.ctx_val = tf.placeholder(shape=(None, 3), dtype=tf.int32)

		self.inpt = tf.placeholder(shape=(None, None), dtype=tf.int32)
		self.tgt = tf.placeholder(shape=(None, None), dtype=tf.int32)
		self.init_lang_state = tf.placeholder(tf.float32, [None, self.lang_hid_size])

		self.item_target = tf.placeholder(shape=(self.item_out_length, None), dtype=tf.int32)
		
		# use for init_lang_state, for sampling and training
		self.lang_state_one = np.zeros([1, self.lang_hid_size])
		self.lang_state_batch = np.zeros([self.bsize, self.lang_hid_size])

	def _build_ctx_graph(self):
		matrix_count = tf.Variable(tf.random_uniform([self.ctx_size, self.ctx_emb_size], -1.0, 1.0), dtype=tf.float32)
		matrix_val = tf.Variable(tf.random_uniform([self.ctx_size, self.ctx_emb_size], -1.0, 1.0), dtype=tf.float32)

		count_embedded = tf.nn.embedding_lookup(matrix_count, self.ctx_count)
		val_embedded = tf.nn.embedding_lookup(matrix_val, self.ctx_val)

		h_tmp = tf.multiply(count_embedded, val_embedded)
		h_tmp = tf.tanh(h_tmp)
		h_tmp = tf.reshape(h_tmp, [-1, 3 * self.ctx_emb_size])
		self.ctx_h = tf.contrib.layers.fully_connected(h_tmp, self.ctx_hid_size, activation_fn=None)

	def _build_lang_graph(self):
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
		attn_bsize, seq_size, _ = tf.unstack(tf.shape(attn_out))

		tmp = tf.reshape(attn_out, (attn_bsize * seq_size, 2 * self.attn_hid_size))
		tmp = tf.contrib.layers.fully_connected(tmp, self.attn_hid_size, activation_fn=tf.tanh)
		tmp = tf.contrib.layers.fully_connected(tmp, 1, activation_fn=None)
		logit = tf.reshape(tmp, (attn_bsize, seq_size))
		prob = tf.nn.softmax(logit)
		prob = tf.tile(tf.expand_dims(prob, 2), [1, 1, 2*self.attn_hid_size])

		return prob

	def _build_attn_graph(self):
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
		self.lang_softmax = tf.nn.softmax(self.lang_logit * 2)
		self.item_softmax = [tf.nn.softmax(out) for out in self.item_out]

	def _prepare_loss(self):
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


class SVAgent(object):
	def __init__(self, sess, model, use_rollouts=False):
		self.sess = sess
		self.model = model
		self.all_rewards = []
		self.domain = domain.get_domain('object_division')
		self.use_rollouts = use_rollouts # only used for generation
		# params for rollouts
		self.ncandidate = 10
		self.nrollout = 5
		self.rollout_len = 100

	def feed_context(self, count, val):
		self.words = []
		self.mask = []

		self.ctx_h = self.sess.run(self.model.ctx_h, {
			self.model.ctx_count: count, 
			self.model.ctx_val: val
		})
		self.lang_h = self.model.lang_state_one

		self.count = count
		self.val = val
		
	def read(self, inpt, prefix_token="THEM:"):
		assert inpt[0] == self.model.corpus.word_dict.get_idx("YOU:") and inpt[-1] in self.model.corpus.word_dict.w2i(["<eos>", "<selection>"])
		inpt = np.concatenate(([self.model.corpus.word_dict.get_idx(prefix_token)], inpt[1:])) 

		self.lang_h = self.sess.run(self.model.lang_final_state, {
			self.model.inpt: [inpt], 
			self.model.init_lang_state: self.lang_h,
			self.model.ctx_h: self.ctx_h
		})
		
		self.words.extend(inpt)

	def write(self):
		if self.use_rollouts:
			return self.write_rollout()
		else:
			return self.write_normal()

	def write_normal(self, start="YOU:"):
		words = [[self.model.corpus.word_dict.get_idx(start)]]
		words_collect = [words[0][0]]
		not_finished = False

		for i in range(100):
			words_dstr, self.lang_h = self.sess.run([self.model.lang_softmax, self.model.lang_final_state], {
				self.model.inpt: words, 
				self.model.init_lang_state: self.lang_h, 
				self.model.ctx_h: self.ctx_h
			})
			distr = np.array(words_dstr[0][0])
			bad_tokens = self.model.corpus.word_dict.w2i(['<unk>', 'YOU:', 'THEM:', '<pad>'])
			distr[bad_tokens] = 0
			distr /= np.sum(distr)

			words = [[ np.random.choice(np.arange(len(distr)), p=distr) ]]
			
			words_collect.append(words[0][0])
			if words[0][0] in self.model.corpus.word_dict.w2i(['<eos>', '<selection>']):
				break
			if i == 99: not_finished = True

		self.words.extend(words_collect)

		if not_finished:
			eos = self.model.corpus.word_dict.get_idx('<eos>')
			words_collect.append(eos)
			self.words.append(eos)

		return words_collect

	def write_rollout(self, start="YOU:", opp_start="THEM:"):
		best_score = -1
		res = None

		for _ in range(self.ncandidate):

			words = [[self.model.corpus.word_dict.get_idx(start)]]
			words_collect = [words[0][0]]
			lang_h_locel1 = self.lang_h
			not_finished = False

			for i in range(100):
				words_dstr, lang_h_locel1 = self.sess.run([self.model.lang_softmax, self.model.lang_final_state], {
					self.model.inpt: words, 
					self.model.init_lang_state: lang_h_locel1, 
					self.model.ctx_h: self.ctx_h
				})
				distr = np.array(words_dstr[0][0])
				bad_tokens = self.model.corpus.word_dict.w2i(['<unk>', 'YOU:', 'THEM:', '<pad>'])
				distr[bad_tokens] = 0
				distr /= np.sum(distr)

				words = [[ np.random.choice(np.arange(len(distr)), p=distr) ]]
				
				words_collect.append(words[0][0])
				if words[0][0] in self.model.corpus.word_dict.w2i(['<eos>', '<selection>']):
					break
				if i == 99: not_finished = True

			if not_finished:
				eos = self.model.corpus.word_dict.get_idx('<eos>')
				words_collect.append(eos)

			is_selection = words_collect[-1] == self.model.corpus.word_dict.get_idx('<selection>')

			# try nrollout rollouts to estimate the reward
			score = 0
			for _ in range(self.nrollout):
				combined_words = self.words + words_collect
				lang_h_local2 = lang_h_locel1

				if not is_selection:
					# complete the conversation with rollout_length samples
					words = [[self.model.corpus.word_dict.get_idx(opp_start)]]
					words_collect_local = [words[0][0]]

					for i in range(100):
						words_dstr, lang_h_local2 = self.sess.run([self.model.lang_softmax, self.model.lang_final_state], {
							self.model.inpt: words, 
							self.model.init_lang_state: lang_h_local2, 
							self.model.ctx_h: self.ctx_h
						})
						distr = np.array(words_dstr[0][0])
						bad_tokens = self.model.corpus.word_dict.w2i(['<unk>', '<pad>'])
						distr[bad_tokens] = 0
						distr /= np.sum(distr)
						
						words = [[ np.random.choice(np.arange(len(distr)), p=distr) ]]

						words_collect_local.append(words[0][0])
						if words[0][0] in self.model.corpus.word_dict.w2i(['<selection>']):
							break
					combined_words.extend(words_collect_local)

				# make choice
				ctx = self.get_ctx()
				choices = self.domain.generate_choices(ctx)

				idxs = [self.model.corpus.item_dict.w2i(c) for c in choices]
				probs = self.sess.run(self.model.item_softmax, {
					self.model.ctx_h: self.ctx_h,
					self.model.inpt: [combined_words], 
					self.model.init_lang_state: self.model.lang_state_one, 
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
				items = self.model.corpus.item_dict.i2w(best_choce) 

				# calculate reward
				choice = items[:3]
				ctx = self.get_ctx() 
				if choice[0] in ['<disconnect>', '<no_agreement>']: 
					self_reward = 0.
				else:
					self_reward = int(choice[0][-1]) * int(ctx[1]) + int(choice[1][-1]) * int(ctx[3]) + int(choice[2][-1]) * int(ctx[5])
				# multiply by probability
				score += np.max(probs_arr) * self_reward

			# take the candidate with the max expected reward
			if score > best_score:
				res = (lang_h_locel1, words_collect)
				best_score = score

		lang_h_locel1, words_collect = res
		self.lang_h = lang_h_locel1
		self.words.extend(words_collect)
		
		return words_collect

	def get_ctx(self):
		count_w, val_w = self.model.corpus.context_dict.i2w(self.count[0]), self.model.corpus.context_dict.i2w(self.val[0])

		ctx = [str(count_w[0]), str(val_w[0]), str(count_w[1]), str(val_w[1]), str(count_w[2]), str(val_w[2])]
		return ctx

	def choose(self):
		ctx = self.get_ctx()
		choices = self.domain.generate_choices(ctx)

		idxs = [self.model.corpus.item_dict.w2i(c) for c in choices]
		probs = self.sess.run(self.model.item_softmax, {
			self.model.ctx_h: self.ctx_h,
			self.model.inpt: [self.words], 
			self.model.init_lang_state: self.model.lang_state_one, 
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
		items = self.model.corpus.item_dict.i2w(best_choce) 

		return items
