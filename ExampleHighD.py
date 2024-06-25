import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time,os
import mmnn


# torch.set_default_dtype(torch.float64)
mydtype = torch.get_default_dtype()
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
##############################
dim = 4
def func(X, dim=dim): # X size:  BS * dim
    # Covariance matrix
    cov = np.array([[min(i+1, j+1) - abs(i-j)*0.1 for j in range(dim)] for i in range(dim)])
    cov *= 20
    # print('Covariance matrix:\n\n', cov, "\n")
    # print('Eig vals:\n\n', scipy.linalg.eigvalsh(cov),"\n\n")
    Y = np.matmul(X, cov)
    Y = Y*X
    Y = np.sum(Y, axis=1)
    Y = np.exp(-0.5*Y)
    const=(2*np.pi)**(dim/2) / scipy.linalg.det(cov)**0.5
    Y = Y/const
    return Y


num_epochs = 150
batch_size = 35**2
training_samples_gridsize = [35]*dim # uniform grid samples
num_test_samples = 66666 # random samples
  
# learning rate in epoch k is 
# lr_init*lr_gamma**floor(k/lr_step_size)
lr_init=0.001
lr_gamma=0.9
lr_step_size= 3


# Set this to False if running the code on a remote server.
# Set this to True if running the code on a local PC 
# to monitor the training process.
show_plot = True 

interval=[-1,1]
ranks = [dim] + [36]*7 + [1]
widths = [666]*8
model = mmnn.MMNN(ranks = ranks, 
                 widths = widths,
                 device = device,
                 ResNet = False)



x_list=[np.linspace(*interval, training_samples_gridsize[i]) for i in range(dim)]
X=np.meshgrid(*x_list)
X=[X[i].reshape([-1,1]) for i in range(dim)]
x_train =np.concatenate(X,axis=1)
y_train = func(x_train).reshape([-1,1])
x_train = torch.tensor(x_train, device=device, dtype=mydtype)
y_train = torch.tensor(y_train, device=device, dtype=mydtype)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=batch_size, shuffle=True)


time1=time.time()
errors_train=[]
errors_test=[]
errors_test_max=[]

optimizer = optim.Adam(model.parameters(), lr=lr_init)
scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
criterion = nn.MSELoss()

for epoch in range(1,1+num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
              
    if epoch % 1 == 0:
        training_error = loss.item()
        print(f"\nEpoch {epoch} / {num_epochs}" + 
              f"  ( {epoch/num_epochs*100:.2f}% )" +
              f"\nTraining error (MSE): { training_error :.2e}" + 
              f"\nTime used: { time.time() - time1 :.2f}s")
        errors_train.append(training_error)
    
        def learned_nn(x): # input and output are numpy.ndarray  
            x=x.reshape([-1, dim]) 
            input_data = torch.tensor(x, dtype=mydtype).to(device)
            y = model(input_data)
            y = y.cpu().detach().numpy().reshape([-1])
            return y     
        
        
        x = np.random.rand(num_test_samples, dim) * 2 - 1
        y_nn = learned_nn(x)
        y_true = func(x)
        
        # Calculate errors
        e = y_nn - y_true
        e_max = np.max(np.abs(e))
        e_mse = np.mean(e**2)
        errors_test.append(e_mse)
        errors_test_max.append(e_max)
        
        print("Test errors (MAX and MSE): " + 
              f"{e_max:.2e} and {e_mse:.2e}")
        

torch.save(model.state_dict(), f'model_parameters{dim}D.pth')
np.savez(f"errors{dim}D", 
         test=np.array(errors_test), 
         testmax=np.array(errors_test_max), 
         train = np.array(errors_train), 
         time=time.time()-time1
         )
fig=plt.figure(figsize=(6,4))
n=len(errors_test) 
m=len(errors_train)
plt.plot(np.linspace(1,m,m), np.log10(errors_train), 
         label="log of training error")
plt.plot(np.linspace(1,n,n), np.log10(errors_test), 
         label="test error")
plt.legend()
