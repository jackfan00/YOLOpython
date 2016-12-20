import sys
import os
sys.path.append(os.path.abspath("/home/jack/ML2016/darknet_keras/src"))
import darknet
import utils
import parse
import kerasmodel
import yolodata
import testyololoss
from keras.models import load_model
from PIL import Image
import numpy as np
from keras import backend as K

# define constant
CLASSNUM = 2
voc_names = ["stopsign", "skis"]

# run_yolo

if len(sys.argv) < 3:
	print ('usage: python %s [train/test/valid] [cfg] [model (optional)]\n' %(sys.argv[0]))
	exit()

voc_labels= []
for i in range(CLASSNUM):
	voc_labels.append("ui_data/labels/"+voc_names[i]+".png")
	if  not os.path.isfile(voc_labels[i]):
		print ('can not load image %s' %(voc_labels[i]))
		exit()


import utils
thresh = utils.find_float_arg(sys.argv, "-thresh", .2)
cam_index = utils.find_int_arg(sys.argv, "-c", 0)
cfg_path = sys.argv[2]
model_weights_path = sys.argv[3] if len(sys.argv) > 3 else 'noweight'
filename = sys.argv[4] if len(sys.argv) > 4 else 'nofilename'


def train_yolo(cfg_path, weights_path):


	net = parse.parse_network_cfg(cfg_path)
	train_images = "train_data/train.txt"
	backup_directory = "backup/"

	# load pretrained model 
	if os.path.isfile(model_weights_path):
		print 'Loading '+model_weights_path
		model=load_model(model_weights_path, custom_objects={'yololoss': testyololoss.yololoss})
	else:
	
		# base is cfg name
		#base = utils.basecfg(cfg_path)

		# construct network
		#net = parse.parse_network_cfg(cfg_path)

		print ('Learning Rate: %f, Momentum: %f, Decay: %f\n' %(net.learning_rate, net.momentum, net.decay));
		model = kerasmodel.makenetwork(net)

	(X_train, Y_train) = yolodata.load_data(train_images,net.h,net.w,net.c, net)

	print ('max_batches : %d, X_train: %d, batch: %d\n' %(net.max_batches, len(X_train), net.batch));
	print str(net.max_batches/(len(X_train)/net.batch))

	#datagen = ImageDataGenerator(
	#	featurewise_center=True,
	#	featurewise_std_normalization=True,
	#	rotation_range=0,
	#	width_shift_range=0.,
	#	height_shift_range=0.,
	#	horizontal_flip=True)

	#datagen.fit(X_train)

	#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=net.batch),
        #            samples_per_epoch=len(X_train), nb_epoch=net.max_batches/(len(X_train)/net.batch))
	#model.fit(X_train, Y_train, batch_size=net.batch, nb_epoch=net.max_batches/(len(X_train)/net.batch))
	batchesPerdataset = max(1,len(X_train)/net.batch)
	model.fit(X_train, Y_train, nb_epoch=net.max_batches/(batchesPerdataset), batch_size=net.batch, verbose=1 )

	model.save_weights('yolo_jack_weight.h5')
	model.save('yolo_jack_kerasmodel.h5')

def debug_yolo( cfg_path, model_weights_path='yolo_jack_kerasmodel.h5' ):
	net = parse.parse_network_cfg(cfg_path)
	testmodel = load_model(model_weights_path, custom_objects={'yololoss': testyololoss.yololoss})
	(s,w,h,c) = testmodel.layers[0].input_shape
	x_test,y_test = yolodata.load_data('train_data/test.txt', h, w, c, net)
	testloss = testmodel.evaluate(x_test,y_test)
	print y_test
	print 'testloss= '+str(testloss)

def test_yolo(img_path, model_weights_path='yolo_jack_kerasmodel.h5', confid_thresh=0.5):
	print 'test_yolo'
	# custom objective function
	testmodel = load_model(model_weights_path, custom_objects={'yololoss': testyololoss.yololoss})
	(s,w,h,c) = testmodel.layers[0].input_shape
	#print (s,w,h,c)
	#exit()
	X_test = []
	if os.path.isfile(img_path):
		img = Image.open(img_path.strip())
		(orgw,orgh) = img.size
		nim = img.resize( (w, h), Image.BILINEAR )
		X_test.append(np.asarray(nim))

	pred = testmodel.predict(np.asarray(X_test))
	for p in pred:
		for i in range(7):
			for j in range(7):
				sys.stdout.write( str(p[i*7+j])+', ' )
			print '-'
	#print pred

def demo_yolo(cfg_path,weights_path,thresh,cam_index,filename):
	print 'demo_yolo'

if sys.argv[1]=='train':
        train_yolo(cfg_path,model_weights_path)
elif sys.argv[1]=='test':
	if os.path.isfile(model_weights_path):
        	test_yolo(filename, model_weights_path, confid_thresh=thresh)
	else:
		test_yolo(filename, confid_thresh=thresh)
elif sys.argv[1]=='demo_video':
        demo_yolo(cfg_path,weights_path,thresh,-1,filename)
elif sys.argv[1]=='debug':
        debug_yolo( cfg_path, model_weights_path )

