"""
This file have
1) runs two pretrained agent: SVAgent (normal, with rollouts), RlAgent (normal, with rollouts)
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

# find sample logs at sv_versus_rl-normal.log and sv_versus_rl-rollout.log
USE_ROLLOUTS = False
total_games = 100

corpus = data.WordCorpus('end-to-end-negotiator/src/data/negotiate', freq_cutoff=20, verbose=True)
traindata = corpus.train_dataset(16)
trainset, _ = traindata

rl_model = RLModel(corpus, 'RL')
sv_model = SVModel(corpus, 'SV')


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "RL")
	saver = tf.train.Saver(var_list=var_list)
	saver.restore(sess, 'model/rl-saved4-99-server/model-3')
	print('rl restored')

	var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "SV")
	saver = tf.train.Saver(var_list=var_list)
	saver.restore(sess, 'model/sv-named99/sv-named-99')
	print('sv restored')

	ctx_gen = ContextGenerator('end-to-end-negotiator/src/data/negotiate/selfplay.txt')
	ctxs_all = ctx_gen.get_ctxs()

	rl_agent = RlAgent(sess, rl_model, use_rollouts=USE_ROLLOUTS)
	sv_agent = SVAgent(sess, sv_model)

	sv_rewards, rl_rewards = [], []
	for counter in range(total_games):
		ctxs = random.choice(ctxs_all)
		print_ctxs(ctxs)
		count_sv, count_rl, val_sv, val_rl = get_counts_vals(corpus, ctxs)

		rl_agent.feed_context(count_rl, val_rl)
		sv_agent.feed_context(count_sv, val_sv)

		sv_first = np.random.randint(2) == 0
		if sv_first:
			for said_time in range(10):
				wrote_sv = sv_agent.write()
				rl_agent.read(wrote_sv)

				if wrote_sv[-1] == corpus.word_dict.get_idx('<selection>'):
					break

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


		if corpus.word_dict.get_word(rl_agent.words[-1]) == '<eos>':
			rl_choice = ['<no_agreement>'] * 6
			agree, rewards = True, [0, 0]
		else:
			rl_choice = rl_agent.choose()

			choices = [rl_choice[:3], rl_choice[3:]]
			ctxs = [rl_agent.get_ctx(), sv_agent.get_ctx()]
			agree, rewards = get_rewards(choices, ctxs)

		rl_rewards.append(rewards[0]) 
		sv_rewards.append(rewards[1])

		print_words2(corpus.word_dict.i2w(rl_agent.words))
		print_items_explained(rl_choice)
		print('-----\ncounter %d, rl reward %d, sv reward %d, rl aver %.2f, sv aver %.2f\n' % \
			(counter, rewards[0], rewards[1], np.mean(rl_rewards), np.mean(sv_rewards)))
