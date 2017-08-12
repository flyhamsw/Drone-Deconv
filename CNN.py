import tensorflow as tf

phase_train = tf.Variable(True)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.05)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.05, shape=shape)
	return tf.Variable(initial)

def conv_layer(x, W_shape, b_shape, name, padding='SAME'):
	W = weight_variable(W_shape)
	b = bias_variable([b_shape])
	return tf.nn.relu(batch_norm(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding) + b, b_shape, phase_train))

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv_layer(x, W_shape, b_shape, name, padding='SAME'):
	W = weight_variable(W_shape)
	b = bias_variable([b_shape])

	x_shape = tf.shape(x)
	out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

	return batch_norm(tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b, b_shape, phase_train)

def pool_layer(x):
	return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def unravel_argmax(argmax, shape):
	output_list = []
	output_list.append(argmax // (shape[2] * shape[3]))
	output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
	return tf.stack(output_list)

def unpool_layer2x2(x, raveled_argmax, out_shape):
	argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
	output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

	height = tf.shape(output)[0]
	width = tf.shape(output)[1]
	channels = tf.shape(output)[2]

	t1 = tf.to_int64(tf.range(channels))
	t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
	t1 = tf.reshape(t1, [-1, channels])
	t1 = tf.transpose(t1, perm=[1, 0])
	t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

	t2 = tf.squeeze(argmax)
	t2 = tf.stack([t2[0], t2[1]], axis=0)
	t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

	t = tf.concat([t2, t1], 3)
	indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

	x1 = tf.squeeze(x)
	x1 = tf.reshape(x1, [-1, channels])
	x1 = tf.transpose(x1, perm=[1, 0])
	values = tf.reshape(x1, [-1])

	delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
	return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

def unpool_layer2x2_batch(bottom, argmax):
	'''
	x_shape = tf.shape(x)
	out_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]]

	batch_size = out_shape[0]
	height = out_shape[1]
	width = out_shape[2]
	channels = out_shape[3]

	argmax_shape = tf.to_int64([batch_size, height, width, channels])
	argmax = unravel_argmax(argmax, argmax_shape)

	t1 = tf.to_int64(tf.range(channels))
	t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
	t1 = tf.reshape(t1, [-1, channels])
	t1 = tf.transpose(t1, perm=[1, 0])
	t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
	t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

	t2 = tf.to_int64(tf.range(batch_size))
	t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
	t2 = tf.reshape(t2, [-1, batch_size])
	t2 = tf.transpose(t2, perm=[1, 0])
	t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

	t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

	t = tf.concat([t2, t3, t1], 4)
	indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

	x1 = tf.transpose(x, perm=[0, 3, 1, 2])
	values = tf.reshape(x1, [-1])

	delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))
	return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))
	'''
	bottom_shape = tf.shape(bottom)
	top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

	batch_size = top_shape[0]
	height = top_shape[1]
	width = top_shape[2]
	channels = top_shape[3]

	argmax_shape = tf.to_int64([batch_size, height, width, channels])
	argmax = unravel_argmax(argmax, argmax_shape)

	t1 = tf.to_int64(tf.range(channels))
	t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
	t1 = tf.reshape(t1, [-1, channels])
	t1 = tf.transpose(t1, perm=[1, 0])
	t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
	t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

	t2 = tf.to_int64(tf.range(batch_size))
	t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
	t2 = tf.reshape(t2, [-1, batch_size])
	t2 = tf.transpose(t2, perm=[1, 0])
	t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

	t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

	t = tf.concat(4, [t2, t3, t1])
	indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

	x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
	values = tf.reshape(x1, [-1])
	return tf.scatter_nd(indices, values, tf.to_int64(top_shape))

def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
	with tf.variable_scope(scope):
		input_shape =  tf.shape(pool)
		output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

		flat_input_size = tf.cumprod(input_shape)[-1]
		flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

		pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
		batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
		                                  shape=tf.stack([input_shape[0], 1, 1, 1]))
		b = tf.ones_like(ind) * batch_range
		b = tf.reshape(b, tf.stack([flat_input_size, 1]))
		ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
		ind_ = tf.concat([b, ind_], 1)

		ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
		ret = tf.reshape(ret, tf.stack(output_shape))
		return ret

def batch_norm(x, n_out, phase_train):
	with tf.variable_scope('bn'):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
									 name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
									  name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train,
							mean_var_with_update,
							lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed
