# YOLOpython

Introduction
---------------------------------------------------------------------------------------------------

This is a Keras implementation of the YOLO:Real-Time Object Detection

        YOLO paper please reference to http://pjreddie.com/darknet/yolo/

Usage
---------------------------------------------------------------------------------------

Training

    Python src/yolo.py train cfg/your.cfg [saved_Keras_model.h5]
If saved_Keras_model.h5 option used, it will read in pretrained model and do incrmentally training
Otherwise train from scratch


Predict

    Python src/yolo.py test cfg/your.cfg saved_Keras_model.h5 predicted_image.jpg


Need to know about code
---------------------------------------------------------------------------------------------

The diffcult part to implement is loss function (the equation reference to http://pjreddie.com/media/files/papers/yolo_1.pdf)
When we change the bounding box number for one cell or classes number, we need to modify the loss code
  
    Can not just modify cfg file then GO, if we change bounding box number for one cell or classes number

Loss function code explanation
--------------------------------------------------------------------------------------

Truth table format arrangement as follow
  one sample : confidence,x,y,w,h,classes, and confidence,x,y,w,h each size is gridcell^2 , 
  classes size is (gridcell^2) * classes
  
  Ex : gridcell =7, classes =2, then truth dimension is (7^2) * (5 + 2) = 343
  and for this 343 vector, index 0~47 is confidence, 48~97 is x, 98~146 is y, 147~195 is w, 
  196~244 is h, 245~342 is classes
  
  This arrangement's purpose is for custom loss implementation
