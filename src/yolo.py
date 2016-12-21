import sys
import os
sys.path.append(os.path.abspath("/home/jack/ML2016/darknet_keras/src"))
import darknet
import utils
import parse
import kerasmodel
import yolodata
import ddd
from keras.models import load_model
from PIL import Image, ImageDraw
import numpy as np
from keras import backend as K
import keras.optimizers as opt
import cfgconst

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


	# construct network
	net = cfgconst.net  #parse.parse_network_cfg(cfg_path)
	train_images = "train_data/train.txt"
	backup_directory = "backup/"

	# load pretrained model 
	if os.path.isfile(model_weights_path):
		print 'Loading '+model_weights_path
		model=load_model(model_weights_path, custom_objects={'yololoss': ddd.yololoss})
		sgd = opt.SGD(lr=net.learning_rate, decay=net.decay, momentum=net.momentum, nesterov=True)
		model.compile(loss=ddd.yololoss, optimizer=sgd)

	else:
	
		# base is cfg name
		#base = utils.basecfg(cfg_path)

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
	net = cfgconst.net ##parse.parse_network_cfg(cfg_path)
	testmodel = load_model(model_weights_path, custom_objects={'yololoss': ddd.yololoss})
	(s,w,h,c) = testmodel.layers[0].input_shape
	x_test,y_test = yolodata.load_data('train_data/test.txt', h, w, c, net)
	testloss = testmodel.evaluate(x_test,y_test)
	print y_test
	print 'testloss= '+str(testloss)

def test_yolo(img_path, model_weights_path='yolo_jack_kerasmodel.h5', confid_thresh=0.5):
	print 'test_yolo'
	# custom objective function
	testmodel = load_model(model_weights_path, custom_objects={'yololoss': ddd.yololoss})
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
	
	# find confidence value > 0.5
	confid_index =-1
	confid_value =-1
	x_value =-1
	y_value =-1
	w_value =-1
	h_value =-1
	class_id =-1
	classprob =-1
	det_l = cfgconst.net.layers[len(cfgconst.net.layers)-1]
        side = det_l.side
	classes = det_l.classes
	for p in pred:
		foundindex = False
		for k in range(5+classes):
			print 'L'+str(k)
			for i in range(side):
				for j in range(side):
					sys.stdout.write( str(p[k*49+i*7+j])+', ' )
					if confid_index ==-1 and k==0 and p[k*49+i*7+j]>0.5:
						confid_index = i*7+j
						foundindex = True
						break
				print '-'
		#
		confid_value = p[0*49+confid_index]
		x_value = p[1*49+confid_index]
		y_value = p[2*49+confid_index]
		w_value = p[3*49+confid_index]
		h_value = p[4*49+confid_index]
		for i in range(classes):
			if p[(5+i)*49+confid_index] > classprob:
				classprob = p[(5+i)*49+confid_index]
				class_id = i
		print 'c='+str(confid_value)+',x='+str(x_value)+',y='+str(y_value)+',w='+str(w_value)+',h='+str(h_value)+',cid='+str(class_id)+',prob='+str(classprob)
		#
		draw = ImageDraw.Draw(nim)
		draw.rectangle([(x_value-w_value/2)*w,(y_value-h_value/2)*h,(x_value+w_value/2)*w,(y_value+h_value/2)*h])
		del draw
		nim.save('predbox.png')

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

