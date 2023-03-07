import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

#print(os.listdir('../Tutorial'))
car_prices_array = [3, 4, 5, 6, 7, 8, 9]
car_price_np = np.array(car_prices_array, dtype=np.float32)
car_price_np = car_price_np.reshape(-1, 1)
car_price_tensor = torch.tensor(car_price_np, requires_grad=True)

number_of_car_sell_array = [7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
number_of_car_sell_np = np.array(number_of_car_sell_array, dtype=np.float32)
number_of_car_sell_np = number_of_car_sell_np.reshape(-1, 1)
number_of_car_sell_tensor = torch.tensor(number_of_car_sell_np)

#linear

class LinearRegression(nn.Module):
    def __init__(self, input_size, outsize_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)
mse = nn.MSELoss()

learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

loss_list = []
iteration_number = 1001
for iteration in range(iteration_number):
    optimizer.zero_grad()
    results = model(car_price_tensor)
    loss = mse(results, number_of_car_sell_tensor)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.data)
    #if (iteration % 50 == 0):
        #print('epoch {}, loss {}'.format(iteration, loss.data))

plt.plot(range(iteration_number), loss_list)
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.show()


predicted = model(car_price_tensor).data.numpy()
plt.scatter(car_prices_array, number_of_car_sell_array, label = 'original data', color = 'red')
plt.scatter(car_prices_array, predicted, label = 'predicted data', color = 'blue')
plt.legend()
plt.xlabel('Car Price $')
plt.ylabel('Number of Car Sell')
plt.title('Original vs Predicted values')
plt.show()
