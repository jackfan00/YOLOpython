import darknet
import keras.layers.advanced_activations as a_a
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization as BNOR
from keras.layers.convolutional import Convolution2D as C2D
from keras.layers.pooling import MaxPooling2D as MP2D
from keras.layers.pooling import GlobalAveragePooling2D as GAP2D
import tensorflow as tf
import keras.optimizers as opt

# loss really use this
import testyololoss
#
# loss according to yolo paper
#
# pred tabel is num boundingBox for 1 grid cell
# detection_layer.side**2*((1+detection_layer.coords)*detection_layer.num+detection_layer.classes)
# truth tabel is 1 boundingBox for 1 grid cell
# detection_layer.truths = detection_layer.side**2*(1+detection_layer.coords+detection_layer.classes)
#
def yolo_loss11111(y_true, y_pred, detection_layer):
	totloss = 0
	bnum = 1+detection_layer.coords
	truthsofcell = 1+detection_layer.coords+detection_layer.classes
	predsofcell = (1+detection_layer.coords)*detection_layer.num+detection_layer.classes
	for cell in range(detection_layer.side**2):
		for i in range(detection_layer.num):
			xyloss = (y_true[cell*truthsofcell:cell*truthsofcell+2] - y_pred[(cell*predsofcell)+(i*bnum):(cell*predsofcell)+(i*bnum)]+2)**2
			whloss = (tf.sqrt(y_true[cell*truthsofcell+2:cell*truthsofcell+4]) - tf.sqrt(y_pred[(cell*predsofcell+2)+(i*bnum):(cell*predsofcell+4)+(i*bnum)]))**2
			confidloss = (y_true[cell*truthsofcell+4:cell*truthsofcell+5] - y_pred[(cell*predsofcell+4)+(i*bnum):(cell*predsofcell+5)+(i*bnum)])**2
			porbloss = (y_true[cell*truthsofcell+bnum:(cell+1)*truthsofcell] - y_pred[cell*predsofcell+detection_layer.num*bnum:(cell+1)*predsofcell])**2

# weighting loss according object exist state, reference to yolo paper
		if y_true[cell*truthsofcell+4] == 0:  # I OBJ ij, reference yolo paper
			totloss += detection_layer.coord_scale*(tf.reduce_sum(xyloss)+tf.reduce_sum(whloss))+tf.reduce_sum(confidloss)+tf.reduce_sum(porbloss)
		else:
			totloss += detection_layer.noobject_scale*tf.reduce_sum(confidloss)

	return totloss

def printmodel(model):
	#print len(model.layers)
	#print model.layers[31].output_shape[1]
	for l in model.layers:
		#print l.activation
		try:
			print l.name+':'+str(l.input_shape)+'-->'+str(l.output_shape)+' act = '+l.activation.name
		except:
			print l.name+':'+str(l.input_shape)+'-->'+str(l.output_shape)



def makenetwork(net):
	model = Sequential()
	index =0
	for l in net.layers:
		try:
			if l.activation_s == 'leaky' or l.activation_s == 'relu':
				#act = a_a.LeakyReLU(alpha=0.1)
				act = 'relu'
			elif l.activation_s == 'logistic' or l.activation_s == 'sigmoid':
				act = 'sigmoid'
			else:
				act = 'linear'
			print 'activation='+act
		except:
			print 'no activation at index '+str(index)

		if l.type == '[crop]':
			#model.add(Input(shape=(l.outh*l.outw*l.outc,), name='input'+str(index)))
			crop_shape = (l.outh,l.outw,l.outc,)
			insert_input = True
		elif l.type == '[convolutional]':
			if l.pad == 1:
				pad = 'same'
			else:
				pad = 'valid'
			if insert_input:
				model.add(C2D( l.n, l.size, l.size, activation=act, border_mode=pad, subsample=(l.stride,l.stride), input_shape=crop_shape, name='convol'+str(index)))
			else:
				model.add(C2D( l.n, l.size, l.size, activation=act, border_mode=pad, subsample=(l.stride,l.stride), name='convol'+str(index)))
			if l.batch_normalize == 1:
				model.add(BNOR(name='bnor'+str(index)))
			insert_input = False
		elif l.type == '[maxpool]':
			model.add(MP2D( pool_size=(l.size,l.size),strides=(l.stride,l.stride),name='maxpool'+str(index) ))
		elif l.type == '[connected]':
			try:
				model.add(Flatten(name='flattern'+str(index)))
			except:
				print 'no need to flattern'
			model.add(Dense( l.outputs, activation=act, name='conted'+str(index)))
		elif l.type == '[dropout]':
			model.add(Dropout(l.probability, name='dropout'+str(index) ))
		elif l.type == '[detection]':
			testyololoss.check(l,model)
			sgd = opt.SGD(lr=net.learning_rate, decay=net.decay, momentum=net.momentum, nesterov=True)
			model.compile(loss = testyololoss.yololoss, optimizer=sgd)
		print l.type + str(index)
		index = index+1

	printmodel(model)
	return model
