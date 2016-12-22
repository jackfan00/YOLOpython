# YOLOpython

Introduction
---------------------------------------------------------------------------------------------------

This is a Keras implementation of the YOLO:Real-Time Object Detection

        YOLO paper please reference to http://pjreddie.com/darknet/yolo/

        also reference http://guanghan.info/blog/en/my-works/train-yolo/

Usage
---------------------------------------------------------------------------------------

Training

    Python src/yolo.py train workingcfg.txt [saved_Keras_model.h5]
    
If saved_Keras_model.h5 option used, it will read in pretrained model and do incrmentally training
Otherwise train from scratch


Predict

    Python src/yolo.py test cfg/your.cfg saved_Keras_model.h5 predicted_image.jpg


Need to know about code
---------------------------------------------------------------------------------------------

Code explanation:

workingcfg.txt : put your cfg file in this file

yolo.py : main function

yolodata.py : read train_data/train.txt, then generate resized X_train and proper Y_train numpy matrix

ddd.py : create custom YOLO loss function (the equation reference to http://pjreddie.com/media/files/papers/yolo_1.pdf)

kerasmodel.py : create Keras model according to cfg file

parse.py : parse cfg file

cfgconst.py : read workingcfg.txt then call parse.py to parse the correcponding cfg file

utils.py, darknet.py : misc


Running enviroment 
--------------------------------------------------------------------------------------------

train_data/train.txt : contain training image data file list 

train_data/[images/labels] folder : contain training image files and corresponding label text files


Loss function code explanation
--------------------------------------------------------------------------------------

Truth table format arrangement as follow
  for one training sample : truth atble format is [confidence,x,y,w,h,classes], and [confidence,x,y,w,h,classes] each element 
  size is gridcell^2 
  
  Given a example : gridcell =7, classes =2, then truth table dimension is (7^2) * (5 + 2) = 343.
  In this truth vector, index 0~48 represent confidence, 49~97 represent x, 98~146 represent y, 
  147~195 represent w, 196~244 represent h, 245~293 represent class 0, 294~342 represent class 1
  
  
  The reason for the trurh table arrangement is for YOLO custom loss implementation.
  
  
OPENCV enviroment setup (ubuntu 16.04 + CUDA 8.0)
--------------------------------------------------------------------------------------------------

Please reference to http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/

follow its steps will fail if you use CUDA 8.0 , need to do some modification.
First following the STEP 1 ~ 9 , then at STEP 10, use follows instead 
                
                1.cd ~
                2.git clone https://github.com/daveselinger/opencv
                3.cd opencv
                4.git checkout 3.1.0-with-cuda8
                
                at opencv_contrib part also use "git checkout 3.1.0"
                
                please reference to https://github.com/opencv/opencv/issues/6677
                
modify cmake command to add "-D WITH_GTK=ON"

		cmake -D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
		-D INSTALL_C_EXAMPLES=OFF \
		-D INSTALL_PYTHON_EXAMPLES=ON \
		-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
		-D WITH_GTK=ON \
		-D BUILD_EXAMPLES=ON ..
                
then follow 3.1.0 setup is OK.


Be caution at STEP 11, "ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so" may not work
because of the wrong cv2.so path. If something wrong, check out the path where your cv2.so be installed. 
In my case cv2.so at /usr/local/lib/python2.7/dist-packages/cv2/cv2.so



