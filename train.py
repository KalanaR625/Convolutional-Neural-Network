import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# Device.
CUDA_USE = torch.cuda.is_available()
device = torch.device("cuda" if CUDA_USE else "cpu")
print(f"Training device: {device}")



# Output of the torchvision datasets are PIL images and there are in [0,1] range. 
# we normalize it in to [-1,1] range of the tensor.
transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


batch_size = 4

trainset = torchvision.datasets.CIFAR10(root= './data', train= True, download= True, transform= transforms)
trainloader= DataLoader(trainset, batch_size = batch_size, shuffle= True)

testset = torchvision.datasets.CIFAR10(root='./data', train= False, download= True, transform= transforms)
testloader = DataLoader(testset, batch_size= batch_size, shuffle= True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




class ConvNet(nn.Module):  # class is inherit from the nn.Module
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 3)
        self.conv2 = nn.Conv2d(96, 64, 3)
        self.maxp1 = nn.MaxPool2d(kernel_size= 2, stride= 2)
        
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.maxp2 = nn.MaxPool2d(kernel_size= 3, stride= 1)

        self.conv4 = nn.Conv2d(64, 16, 3)

        self.dropout1 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(16*8*8, 64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxp1(F.relu(self.conv2(x)))
        x = self.maxp2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.dropout1(x)
        x = self.fc3(x) # softmax function is apply for us by CrossEntropyLoss()
                        # so we don't want to put softmax activation in last layer.
     
        return x

# Initialize the network
convnet = ConvNet().to(device) 


# path to the folder that model want to save.
PATH = './save'

load_model = './save/10_num_epoches.tar' # give the path of the saved model. if don't have a model then give None.


if load_model:

    # if loading machine is same to the machine that used to train the model, then use following code.
    #checkpoint = torch.load(load_model)

    # if model is train in GPU and loading on to CPU, then use following code.
    checkpoint = torch.load(load_model, map_location= torch.device('cpu'))
    
    convnet.load_state_dict(checkpoint['model_state_dict'])


learning_rate = 0.001
num_epoches = 4
momentum =0.9

# crossentropyloss() apply softmax activation for last layer of feed forward layer.
# also this is the loss
criterion = nn.CrossEntropyLoss() 

# momentum method in the SGD is help to converging faster.
# This is the optimizer.
optimizer = optim.SGD(convnet.parameters(), lr= learning_rate, momentum= momentum)



"""------Tensorbord------"""

# following will create runs/conv_experiment_1 folder.
writer = SummaryWriter('runs/conv_experiment_1')


# get random training images.
dataiter = iter(trainloader)
images, labels = dataiter.next()


# creating grid of images.
img_gird = torchvision.utils.make_grid(images)

# images write to tensorboard.
writer.add_image('images_of_cifar10', img_gird)

# write Convnet structure in tensorboard.
#writer.add_graph(convnet, images)
#writer.close()

# used following code in terminal to run tensorboard 
# tensorboard --logdir=runs  
START_ITERATION = 1
save_every = 1

if load_model:
    START_ITERATION = checkpoint['num_epoches'] + 1



def train_loop(train_loader, convnet, optimizer, criterion):

    # set back to training mode after setting evaluation mode.
    convnet.train()

    # This is size of the training loader. Number of images in the training dataset.
    size = len(train_loader.dataset)

    
    for i, data in enumerate(train_loader, 0):
        # i in the for loop is the index number that start from 0. 
        # because 0 defined in the enumerate().

        # data are lists that contain [images, labels]
        images, labels = data[0].to(device),data[1].to(device) # send images and labels to GPU at every step
        
     
        # forward pass
        outputs = convnet(images)
        
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()  # First need to zero the gradient.
        loss.backward() # This do the backpropagation
        optimizer.step() # Updates parameters

        # print every 100 steps.
        if (i+1) % 100 == 1:
            loss, current = loss.item(), i * len(images)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


        # write training loss in the tensorboard after avery single epoches.
    writer.add_scalar('training loss', loss, current)

        

    print("Finished Training")
   

def test_loop(test_loader, convnet, criterion, epoches):

    # set to evaluation mode.
    convnet.eval()

    # This is size of the test loader. Number of images in the test dataset
    size = len(test_loader.dataset) # this has 10000 images 
    
    num_batches = len(test_loader) # testloader has 2500 batches. 4*2500 =1000. 4 is batch size.
    test_loss, correct =0,0

    # Don't want to compute gradient in the testing.
    # So use with torch.no_grad().
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device) # send to GPU
            labels = labels.to(device) # send to GPU

            output = convnet(images)
            test_loss += criterion(output, labels).item()
            correct += (output.argmax(1)== labels).type(torch.float).sum().item()

            """ argmax() return indices of maximum value of the all elements in the input(tensor).output.argmax(1) give index of maximum value for 4 images in batch (because batch size is 4). ex:-tensor([2, 2, 6, 2])
                this output is a tensor. 2 is the tensor is predicted class (ex:- bird). when output.argmax(1) == labels it gives output
                as True or False. if predicted class same to the label then True. if predicted class is not same to the label
                then False. output is give as tensor([ True, False,  True, False]). Then True or False convert to 0.0 or 1.0 (float numbers)
                by using (outputs.argmax(1) == labels).type(torch.float). Then output is:- tensor([0., 0., 1., 1.]). 
                Then get sum of that tensor by adding .sum(). ex:- (output.argmax(1)== labels).type(torch.float).sum(). Then output
                is:- tensor(2.). Then add .item() and then code is like (output.argmax(1)== labels).type(torch.float).sum().item().
                it return value of the tensor as a standard python number.ex:- 1.0. 
                In the final all the test_loss and correct for the all the images in the testloader had added.
                If final correct is 5000,then model had predicted 5000 images correctly.
                Final average test loss and avarage correct is calculate as follows.  """

    test_loss /= num_batches  # test_loss = test_loss/num_batches
    correct /= size   # correct = correct/ size. 

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")   
    
    # write test loss in the tensorboard.
    writer.add_scalar('test loss', test_loss, epoches)


if __name__ == "__main__":
  
    for t in range(START_ITERATION, num_epoches +1):
        print(f"Epoch {t}\n---------------------")
        train_loop(trainloader, convnet, optimizer, criterion,t)
        test_loop(testloader, convnet, criterion,t)

        if (t% save_every)== 0:

      
            torch.save({'num_epoches': num_epoches,
                 'model_state_dict':convnet.state_dict()
                }, os.path.join(PATH, '{}_{}.tar'.format(t, 'num_epoches')))

    print("Done !")



