# Convolutional-Neural-Network
This is a convolutional neural network made by using pytorch.


data folder has dataset for training the convolutional neural network.
when training the model, parameters are saved at every predefine epoches and those are saved in to the save
folder. train.py has codes for training the convolutional neural network. In the eval.py, model had load 
from the saved file and then get images batches from the testloader and then those are give to the load model to predict.
deployment folder has deploy.py and that codes are show how load model is used to predict any kind of the image
and that is the format that can be used to when making a software.
