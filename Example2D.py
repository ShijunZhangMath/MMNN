import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time,os
import mmnn

## 2D function Example


# torch.set_default_dtype(torch.float64)
mydtype = torch.get_default_dtype()
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
##############################
def func(x):
    def cart2pol(x, y):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return(r, theta)
    r, theta = cart2pol(x[:,0], x[:,1])
    m=np.pi**2
    r1 = 0.5+ 0.2*np.cos(m*theta**2)
    z1 = 0.5 - 5*(r-r1)
    def g(z):        
        z = np.maximum(z, 0)
        z = np.minimum(z, 1)
        return(z)
    y = g(z1)
    return y

num_epochs = 1000
batch_size = 1000
training_samples_gridsize = [300, 300] # uniform grid samples
num_test_samples = 66666 # random samples
  
# learning rate in epoch k is 
# lr_init*lr_gamma**floor(k/lr_step_size)
lr_init=0.001
lr_gamma=0.9
lr_step_size= 20

# Set this to False if running the code on a remote server.
# Set this to True if running the code on a local PC 
# to monitor the training process.
show_plot = True 

interval=[-1,1]
ranks = [2] + [36]*7 + [1]
widths = [666]*8
model = mmnn.MMNN(ranks = ranks, 
                 widths = widths,
                 device = device,
                 ResNet = False)

   
x1 = np.linspace(*interval, training_samples_gridsize[0])
x2 = np.linspace(*interval, training_samples_gridsize[1])
X1, X2 = np.meshgrid(x1, x2)
X = np.concatenate([np.reshape(X1,[-1,1]),
                          np.reshape(X2,[-1,1])],axis=1)
Y = func(X).reshape([-1,1])
x_train = torch.tensor(X, device=device, dtype=mydtype)
y_train = torch.tensor(Y, device=device, dtype=mydtype)
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
              
    if epoch % 2 == 0:
        training_error = loss.item()
        print(f"\nEpoch {epoch} / {num_epochs}" + 
              f"  ( {epoch/num_epochs*100:.2f}% )" +
              f"\nTraining error (MSE): { training_error :.2e}" + 
              f"\nTime used: { time.time() - time1 :.2f}s")
        errors_train.append(training_error)
    
        def learned_nn(x): # input and output are numpy.ndarray
            x=x.reshape([-1, 2])            
            input_data = torch.tensor(x, dtype=mydtype).to(device)
            y = model(input_data)
            y = y.cpu().detach().numpy().reshape([-1])
            return y     
        
        x = np.random.rand(num_test_samples, 2) * 2 - 1
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
        
        if epoch % 1 == 0:
            # Plot the results
            gridsize=[181, 181]
            x1 = np.linspace(*interval, gridsize[0])
            x2 = np.linspace(*interval, gridsize[1])
            X1, X2 = np.meshgrid(x1, x2)
            X = np.concatenate([np.reshape(X1,[-1,1]),
                                      np.reshape(X2,[-1,1])],axis=1)
            Y_true = func(X).reshape(gridsize[::-1])
            Y_nn = learned_nn(X).reshape(gridsize[::-1])
            fig=plt.figure(figsize=(12, 4.8))
            plt.subplot(1, 2, 1)
            ax=plt.gca()
            ctf = ax.contourf(X1, X2, Y_true, 100,
                    alpha=0.8, cmap="coolwarm")
            cbar = fig.colorbar(ctf, shrink=0.99, aspect=6)
            plt.title(f'true function (Epoch {epoch}')
            plt.subplot(1, 2, 2)
            ax=plt.gca()
            ctf = ax.contourf(X1, X2, Y_nn, 100,
                    alpha=0.8, cmap="coolwarm")
            cbar = fig.colorbar(ctf, shrink=0.99, aspect=6)
            plt.title(f'learned network (Epoch {epoch})')
            plt.tight_layout()
    
            FPN="./figures/"
            if not os.path.exists(FPN):
                os.makedirs(FPN)
            plt.savefig(f"{FPN}mmnn_epoch{epoch}2D.png", dpi=50)
            if show_plot:
                plt.show()

torch.save(model.state_dict(), 'model_parameters2D.pth')

fig=plt.figure(figsize=(6,4))
n=len(errors_test) 
m=len(errors_train)
k=round(m/n)
np.savez("errors2D", 
         test=np.array(errors_test), 
         testmax=np.array(errors_test_max), 
         train = np.array(errors_train), 
         time=time.time()-time1
         )
t=np.linspace(1,n,n)   
plt.plot(t, np.log10(errors_train[::k]), label="log of training error")
plt.plot(t, np.log10(errors_test), label="test error")
plt.legend()   

