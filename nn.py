import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os.path
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

imageDir = Path('../shape')
filepaths = list(imageDir.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split((os.path.split(x)[0]))[1],filepaths))
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
labels = labels.map({'circle':0, 'square': 1, 'star':2, 'triangle':3})
#labels_numpy = pd.Series(labels, dtype=np.float32)
labels_numpy = np.array(labels)
#print(labels_numpy)
#print(type(labels_numpy))

images = []
for filename in filepaths:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(-1)
    images.append(img)
print('Finished image conversion.')
images = np.array(images)
features_numpy = images/255

#print(features_numpy)
features_train, features_test, labels_train, labels_test = train_test_split(features_numpy, labels_numpy, test_size=0.2, random_state=15)
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(labels_train).type(torch.LongTensor)
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(labels_test).type(torch.LongTensor)
batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

class NNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(NNModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.elu3 = nn.ELU()

        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.elu3(out)
        out = self.fc4(out)
        return out

input_dim = 200*200
hidden_dim = 400
output_dim = 4

model = NNModel(input_dim, hidden_dim, output_dim)
error = nn.CrossEntropyLoss()
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        train = Variable(images.view(-1, 200*200))
        train = train.float()
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(train)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()
        count += 1

        if count % 50 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                test = Variable(images.view(-1, 200*200))
                test = test.float()
                outputs = model(test)
                predicted = torch.max(outputs.data, 1)[1]
                total += len(labels)
                correct += (predicted ==labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            print('Iteration: {} Loss: {} Accuracy: {} %'.format(count, loss.data, accuracy))

plt.plot(iteration_list, loss_list)
plt.xlabel('Number of iteration')
plt.ylabel('Loss')
plt.title('NN: Loss vs Number of iteration')
plt.show()

plt.plot(iteration_list, accuracy_list, color='red')
plt.xlabel('Number of iteration')
plt.ylabel('Accuracy')
plt.title('NN: Accuracy vs Number of iteration')
plt.show()
