# Convolutional-Neural-Network
This is a convolutional neural network made by using pytorch.

Followings are the steps for to use above source codes.

1) When run train.py in the first time, it will create a folder named 'data' and then it will    download dataset to that folder.

2) Set 'None' for the load_model variable in the train.py 
3) Create a folder named save for the save model.
4) Then run the train.py 
5) It will automatically create a folder named run to save tensorboard graphs.
6) After training is done, set path of the last checkpoint in save folder to the load_model variable.
7) Then test trained model using eval.py.
8) Then give image path that want to predict to the IMAGE_PATH variable in the deploy.py in the deployment folder. then run deploy.py to predict that image.


when training the model, parameters are saved at every predefine epoches and those are saved in to the save
folder. train.py has codes for training the convolutional neural network. In the eval.py, model had load 
from the saved file and then get images batches from the testloader and then those are give to the load model to predict.
deployment folder has deploy.py and it used to predict images.

