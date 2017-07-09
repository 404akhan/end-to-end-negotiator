def print_words(words):
	print('\n\n-' * 5)
	for counter, word in enumerate(words):
		if counter != 0 and (word == 'YOU:' or word == 'THEM:'): 
			print('')

		print(word, end=" ")
	print('')

def print_words2(words):
	print('-' * 5)
	for counter, word in enumerate(words):
		if counter != 0 and (word == 'YOU:' or word == 'THEM:'): 
			print('')

		print(word, end=" ")
	print('')

def get_rewards(choices, ctxs):
	if choices[0][0] in ['<disconnect>', '<no_agreement>']:
		return True, [0, 0]

	rewards = []
	for choice, ctx in zip(choices, ctxs):
		reward = int(choice[0][-1]) * int(ctx[1]) + int(choice[1][-1]) * int(ctx[3]) + int(choice[2][-1]) * int(ctx[5])
		rewards.append(reward)
	return True, rewards

def get_counts_vals(corpus, ctxs):
	counts, vals = [], []
	for ctx in ctxs:
		count = [corpus.context_dict.w2i([ 
			ctx[0], ctx[2], ctx[4] 
		])]
		val = [corpus.context_dict.w2i([ 
			ctx[1], ctx[3], ctx[5] 
		])]

		counts.append(count)
		vals.append(val)
	return counts[0], counts[1], vals[0], vals[1]

def make_copy_params_op(v1_list, v2_list):
	v1_list = list(sorted(v1_list, key=lambda v: v.name))
	v2_list = list(sorted(v2_list, key=lambda v: v.name))

	update_ops = []
	for v1, v2 in zip(v1_list, v2_list):
		op = v2.assign(v1)
		update_ops.append(op)

	return update_ops

def get_aver(r, aver, alpha=0.99):
	if aver is None:
		return r
	else:
		return r * (1-alpha) + aver * alpha

def print_items(items):
	print('-' * 5)
	for item in items: 
		print(item, end=" ")
	print("\n\n\n")

def print_items_explained(items):
	arr = ['book=', 'hat=', 'ball=', 'book=', 'hat=', 'ball=']
	print('-' * 5)
	for i, item in enumerate(items): 
		print(arr[i] + item[-1], end=" ")
	print("")

def print_ctxs(ctxs):
	ctx = ctxs[1]
	print('\n\nRlAgent (YOU) : book=(count:%s value:%s) hat=(count:%s value:%s) ball=(count:%s value:%s)' % 
		(ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5]))
	ctx = ctxs[0]
	print('SVAgent (THEM) : book=(count:%s value:%s) hat=(count:%s value:%s) ball=(count:%s value:%s)' % 
		(ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5]))
