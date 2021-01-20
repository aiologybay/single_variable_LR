#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

'''
function: fitting the equation: y = a * x + b
get the value a, b
'''

# set the hyperparameters
a = 3
b = 4
input_size = 1
output_size = 1
num_epochs = 1000
learning_rate = 1e-3
loss_list = []

# create the dataset
x_train = torch.randn(500, 1)
y_train = a * x_train + b + torch.randn(500,1) # add random interference

# create a single layer linear regression network and show its
net = nn.Sequential(
    nn.Linear(input_size, output_size),
)
print(net)
summary(net, input_size=(1, 1))

# select the suitable cross function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# train the model
for epoch in range(1, num_epochs):
    # Forward propagation
    outputs = net(x_train)
    loss = criterion(outputs, y_train)
    # Back propagation and optimize
    optimizer.zero_grad()  # Set the gradient with zero
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))
    loss_list.append(loss.item())

# show parameters a, b
for name, param in net.named_parameters():
    if name == '0.weight':
        print('a={:.4f}'.format(param.detach().numpy().item()))
    elif name == '0.bias':
        print('b={:.4f}'.format(param.detach().numpy().item()))

# save model weight
torch.save(net.state_dict(), 'weight.pt')

# test
model = net
model.load_state_dict(torch.load('weight.pt'))
model.eval()
with torch.no_grad():
    test_y = model(torch.tensor([6.]))  # define test data
    print('Test result:{:.2f}'.format(test_y.item()))

# visual the process of training
predicted = net(x_train).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Train Data')
plt.plot(x_train, predicted, label='Fitted Line')
plt.legend()
plt.title('Fitting Curve')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.savefig('fitting.png')
plt.pause(2)
plt.close()

# draw loss curve
epoch_list = np.arange(1, len(loss_list) + 1, 1).tolist()
plt.plot(epoch_list, loss_list, label='loss')
plt.legend()
plt.title('Loss Curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.pause(2)
plt.close()

