from PIL import Image, ImageDraw
import numpy as np
import sys

class yolobbox():
	def __init__(self):
		self

def readlabel(fn):
	#print 'readlabel '+ fn
	f = open(fn)
	box = yolobbox()
	for l in f:
		try:
			ss= l.split(' ')
			box.id = int(ss[0])
			box.x = float(ss[1]) 
			box.y = float(ss[2]) 
			box.w = float(ss[3]) 
			box.h = float(ss[4]) 
		except:
			box.id = -1
	return box
		
def load_data(train_images, h, w, c, net):
	f = open(train_images)
	paths = []
	for l in f:
		paths.append(l)

	bckptsPercell = net.layers[len(net.layers)-1].coords + 1
	gridcells = net.layers[len(net.layers)-1].side 
	bnumPercell = net.layers[len(net.layers)-1].num
	classes = net.layers[len(net.layers)-1].classes

	X_train = []
	Y_train = []
	count = 1
	for fn in paths:
		img = Image.open( fn.strip())
		(orgw,orgh) = img.size
		nim = img.resize( (w, h), Image.BILINEAR )
		data = np.asarray( nim )
		if data.shape != (w, h, c):
			continue
		X_train.append(data)

		# replace to label path
		fn=fn.replace("images","labels")
		fn=fn.replace(".JPEG",".txt")
		fn=fn.replace(".jpg",".txt")
		fn=fn.replace(".JPG",".txt")
		#print fn

		box = readlabel(fn.strip())
		if box.id == -1:
			print 'read bbox fail'
			continue


		#
		# let truth size == pred size, different from yolo.c 
		# trurh data arrangement is (confid,x,y,w,h)(..)(classes)
		#
		truth = np.zeros(gridcells**2*(bckptsPercell*bnumPercell+classes))
		col = int(box.x * gridcells)
		row = int(box.y * gridcells)
		x = box.x * gridcells - col
		y = box.y * gridcells - row
		for i in range(bnumPercell):
			index = (col+row*gridcells)*(bckptsPercell*bnumPercell+classes) + bckptsPercell*i
			truth[index] = 1
			truth[index+1] = x
			truth[index+2] = y
			truth[index+3] = box.w
			truth[index+4] = box.h
			#print 'index='+str(index)+' '+str(box.x)+' '+str(box.y)+' '+str(box.w)+' '+str(box.h)
		truth[index+bckptsPercell*bnumPercell+box.id] =1
		Y_train.append(truth)

		#print 'draw rect bounding box'
		#draw = ImageDraw.Draw(nim)
		#draw.rectangle([(box.x-box.w/2)*w,(box.y-box.y/2)*h,(box.x+box.w/2)*w,(box.y+box.y/2)*h])
		#del draw
		#nim.save('ttt.png')
		#exit()
		#for row_cell in range(7):
		#	for col_cell in range(7):
		#		sys.stdout.write( str(truth[col_cell*7+row_cell*(7*7)])+', ' )
		#	print '-'

		#print truth[720:740]
		#exit()
		if count > 10:
			break
		else:
			count = count + 1

	#print len(X_train)
	XX_train = np.asarray(X_train)
	YY_train = np.asarray(Y_train)
	print XX_train.shape
	print YY_train.shape
	#exit()

	return XX_train, YY_train
		

