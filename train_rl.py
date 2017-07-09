"""
This file have
1) trains RLAgent by making it play with SVAgent 
2) saves rl_model after training
"""

import tensorflow as tf 
import numpy as np
import data
import domain
import sys
import random

from rl_agent import *
from sv_agent import *
from helpers import *
from utils import ContextGenerator

# load data same way as original repository
corpus = data.WordCorpus('end-to-end-negotiator/src/data/negotiate', freq_cutoff=20, verbose=True)
traindata = corpus.train_dataset(16)
trainset, _ = traindata

# initialize models
rl_model = RLModel(corpus, 'RL')
sv_model = SVModel(corpus, 'SV')


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# load pretrained SuperVised model
	var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "SV")
	saver = tf.train.Saver(var_list=var_list)
	saver.restore(sess, 'model/sv-named99/sv-named-99')
	print('sv restored')

	# copy it to Reinforcement Learning model
	copy_params_op = make_copy_params_op(
		tf.contrib.slim.get_variables(scope="SV", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
		tf.contrib.slim.get_variables(scope="RL", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

	sess.run(copy_params_op)
	print('rl copied')

	# get contexts, initial counts and values of objects
	ctx_gen = ContextGenerator('end-to-end-negotiator/src/data/negotiate/selfplay.txt')
	ctxs_all = ctx_gen.get_ctxs()

	# agents that read and write to each other
	sv_agent = SVAgent(sess, sv_model)
	rl_agent = RlAgent(sess, rl_model)

	# train rl here
	aver_sv, aver_rl, aver_sv_upd = 5., 5., None
	for counter_epoch in range(5):
		for counter, ctxs in enumerate(ctxs_all):
			# get formatted counts and values of objects
			count_sv, count_rl, val_sv, val_rl = get_counts_vals(corpus, ctxs)

			rl_agent.feed_context(count_rl, val_rl)
			sv_agent.feed_context(count_sv, val_sv)

			# randomly each agent starts first
			sv_first = np.random.randint(2) == 0
			if sv_first:
				for said_time in range(10):
					# sv writes, then rl reads
					wrote_sv = sv_agent.write()
					rl_agent.read(wrote_sv)

					if wrote_sv[-1] == corpus.word_dict.get_idx('<selection>'):
						break

					# rl writes, then sv reads
					wrote_rl = rl_agent.write()
					sv_agent.read(wrote_rl)

					if wrote_rl[-1] == corpus.word_dict.get_idx('<selection>'):
						break
			else:
				for said_time in range(10):
					wrote_rl = rl_agent.write()
					sv_agent.read(wrote_rl)

					if wrote_rl[-1] == corpus.word_dict.get_idx('<selection>'):
						break

					wrote_sv = sv_agent.write()
					rl_agent.read(wrote_sv)

					if wrote_sv[-1] == corpus.word_dict.get_idx('<selection>'):
						break

			# if after the loop not selection, rewards are 0
			if corpus.word_dict.get_word(rl_agent.words[-1]) == '<eos>':
				sv_choice = ['<no_agreement>'] * 6
				agree, rewards = True, [0, 0]
			# get rewards, depending on their agreement, where sv_choice is how much each object was assigned to him and opponent
			# sv_choice values were pretrained from real data
			else:
				sv_choice = sv_agent.choose()

				choices = [sv_choice[:3], sv_choice[3:]]
				ctxs = [sv_agent.get_ctx(), rl_agent.get_ctx()]
				agree, rewards = get_rewards(choices, ctxs)

			# update rl_agent with policy gradient
			loss_rl = rl_agent.update(agree, rewards[1])

			aver_sv, aver_rl = get_aver(rewards[0], aver_sv), get_aver(rewards[1], aver_rl)
			print('counter %d\%d, loss_rl %f, dial len %d, sv reward %d, rl reward %d, sv aver %.2f, rl aver %.2f' % \
				(counter, len(ctxs_all), loss_rl, len(sv_agent.words), rewards[0], rewards[1], aver_sv, aver_rl))

			if counter % 10 == 0:
				print_words(corpus.word_dict.i2w(sv_agent.words))
				print_items(sv_choice)

			# supervised update on rl_agent every 4 time
			if counter % 4 == 0:
				batch = random.choice(trainset)
				count, val, inputs, targets, items, _ = batch

				loss_val, _ = sess.run([rl_model.loss, rl_model.train_op], {
					rl_model.ctx_count: count, 
					rl_model.ctx_val: val, 
					rl_model.inpt: inputs, 
					rl_model.tgt: targets, 
					rl_model.item_target: items,
					rl_model.init_lang_state: rl_model.lang_state_batch, 
				})
				aver_sv_upd = get_aver(loss_val, aver_sv_upd, alpha=0.95)
				print('supervised loss %f, aver_sv_upd %f' % (loss_val, aver_sv_upd))

		# save rl_model
		var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "RL")
		saver = tf.train.Saver(var_list=var_list)
		saver.save(sess, 'model/rl-saved4-99/model', global_step=counter_epoch)
