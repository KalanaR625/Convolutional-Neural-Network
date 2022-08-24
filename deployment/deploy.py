import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

IMAGE_PATH = './Predict_images/download (1).jpg'

# open the image.
image = Image.open(IMAGE_PATH)

transform = transforms.Compose(
        [transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((32,32)),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )


''' unsqueeze(0) used because in the training its input shape is [4, 3, 32, 32]. Because it has batch that 
    size is 4. But this time there is only 1 image and then input image shape is [3, 32, 32]. So then neural network 
    doen't work. So we need to add another dimention. So add one dimention by using unsqueeze(0) and then shape is [1, 3, 32, 32]'''

image= transform(image).unsqueeze(0)



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
convnet = ConvNet()

# load saved model.
load_model = './save/10_num_epoches.tar'

checkpoint = torch.load(load_model, map_location= torch.device('cpu'))
    
convnet.load_state_dict(checkpoint['model_state_dict'])

# set to eval mode.
convnet.eval()

pred = convnet(image)

_, prediction = torch.max(pred, 1)

print('Predicted: ', ''.join(f'{classes[prediction]:5s}'))

