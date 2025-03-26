# -*- coding: utf-8 -*-
"""
Training a Hyperbolic Classifier
================================

This is an adaptation of torchvision's tutorial "Training a Classifier" to 
hyperbolic space. The original tutorial can be found here:

- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Training a Hyperbolic Image Classifier
--------------------------------------

We will do the following steps in order:

1. Define a hyperbolic manifold
2. Load and normalize the CIFAR10 training and test datasets using ``torchvision``
3. Define a hyperbolic Convolutional Neural Network
4. Define a loss function and optimizer
5. Train the network on the training data
6. Test the network on the test data

"""

########################################################################
# 1. Define a hyperbolic manifold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We use the Poincaré ball model for the purposes of this tutorial.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.manifolds.euclidean import Euclidean

import torch
from tqdm import tqdm  # Add this to use progress bar

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Making the curvature a learnable parameter is usually suboptimal but can
# make training smoother.
manifold = PoincareBall(c=Curvature(value=1.0, requires_grad=False))
# manifold=Euclidean()

########################################################################
# 2. Load and normalize CIFAR10
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


batch_size = 256

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=4
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=4
)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


########################################################################
# 3. Define a hyperbolic Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's rebuild the convolutional neural network from torchvision's tutorial
# using hyperbolic modules.

from torch import nn

from hypll import nn as hnn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = hnn.HConvolution2d(
            in_channels=3, out_channels=6, kernel_size=5, manifold=manifold
        )
        self.pool = hnn.HMaxPool2d(kernel_size=2, manifold=manifold, stride=2)
        self.conv2 = hnn.HConvolution2d(
            in_channels=6, out_channels=16, kernel_size=5, manifold=manifold
        )
        self.fc1 = hnn.HLinear(in_features=16 * 5 * 5, out_features=120, manifold=manifold)
        self.fc2 = hnn.HLinear(in_features=120, out_features=84, manifold=manifold)
        self.fc3 = hnn.HLinear(in_features=84, out_features=10, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net = Net().to(device)  # Send model to GPU (if available)

########################################################################
# 4. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and RiemannianAdam optimizer.
# Adam is preferred because hyperbolic linear layers can sometimes have training
# difficulties early on due to poor initialization.

criterion = nn.CrossEntropyLoss()
# net.parameters() includes the learnable curvature "c" of the manifold.
from hypll.optim import RiemannianAdam

optimizer = RiemannianAdam(net.parameters(), lr=0.001)


########################################################################
# 5. Train the network
# ^^^^^^^^^^^^^^^^^^^^
# This is when things start to get interesting.
# We simply have to loop over our data iterator, project the inputs onto the
# manifold, and feed them to the network and optimize.

from hypll.tensors import TangentTensor

from tqdm import tqdm

for epoch in range(20):  # Loop over the dataset multiple times
    running_loss = 0.0
    loop = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}")

    for i, data in loop:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Move inputs to manifold
        tangents = TangentTensor(data=inputs, man_dim=1, manifold=manifold).cuda(device)
        manifold_inputs = manifold.expmap(tangents).cuda(device)

        optimizer.zero_grad()
        outputs = net(manifold_inputs)
        loss = criterion(outputs.tensor, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (i + 1))

print("Finished Training")


########################################################################
# Let's quickly save our trained model:

PATH = "./cifar_net.pth"
torch.save(net.state_dict(), PATH)


########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

net = Net().to(device)
net.load_state_dict(torch.load(PATH))

########################################################################
# 6. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # move the images to the manifold
        tangents = TangentTensor(data=images, man_dim=1, manifold=manifold).cuda(device)
        manifold_images = manifold.expmap(tangents).cuda(device)

        # calculate outputs by running images through the network
        outputs = net(manifold_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.tensor, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")


########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data

        # move the images to the manifold
        tangents = TangentTensor(data=images, man_dim=1, manifold=manifold).cuda(device)
        manifold_images = manifold.expmap(tangents).cuda(device)

        outputs = net(manifold_images)
        _, predictions = torch.max(outputs.tensor, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")

