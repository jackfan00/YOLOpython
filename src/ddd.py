from keras import backend as K
import tensorflow as tf

def yololoss(y_true, y_pred):
	cofidloss = y_pred**2
	value_if_true = 10*(cofidloss)
	value_if_false = (cofidloss)
	t = K.greater(y_true, 0.5) #tf.constant(0.5))
	#print K.ndim(t)
	#print K.int_shape(t)
	loss1 = tf.select(t, value_if_true, value_if_false)
	loss = K.sum(loss1)

	#sess = tf.Session()

	#if sess.run(t) :
	#if t is not None:
	#if y_true[0] == 1:
	#	loss = 10*K.sum(cofidloss)
	#else:
	#	loss = K.sum(cofidloss)
	return loss1, t

x =K.placeholder(ndim=2)
y =K.placeholder(ndim=2)
loss,t = yololoss(y,x)

f = K.function([y,x], [loss,t])

print f([[[1,0.5,0.1],[0,1,0]], [[1,1,1],[2,2,2]]])

