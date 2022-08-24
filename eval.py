from train import torch, torchvision, testloader, convnet, checkpoint, device
import matplotlib.pyplot as plt
import numpy as np

def imshow(image):
    image = image/2 + 0.5 # unnormalizing image
    image = image.cpu()
    npimage = image.numpy() # convert tensor to numpy
    plt.imshow(np.transpose(npimage, (1,2,0)))
    plt.show()


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



dataiter = iter(testloader)
images, labels = dataiter.next()
images = images.to(device)

# plot images in the one batch
imshow(torchvision.utils.make_grid(images))
print('Ground truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


# load save model
convnet.load_state_dict(checkpoint['model_state_dict'])

# set module to evaluation mode.
convnet.eval()

# predict images that gives from testiter. 
outputs = convnet(images)

# torch.max return value, index, dim =1 so indexes along in dimension 1.
# we only need index. so first value ignore by using _.
# index is give as tensor([5, 5, 5, 5]) when four images are dogs.

_, prediction = torch.max(outputs, 1)



print('Predicted: ', ' '.join(f'{classes[prediction[j]]:5s}' for j in range(4)))

