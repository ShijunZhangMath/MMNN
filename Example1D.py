import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time,os
import mmnn

# torch.set_default_dtype(torch.float64)
mydtype = torch.get_default_dtype()
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
##############################
def func(x):
    y = np.cos(36*np.pi* x**2) - 0.8*np.cos(12*np.pi* x**2)
    return y


num_epochs = 20000
batch_size = 100
num_training_samples = 1000 # uniform grid samples
num_test_samples = 1234 # random samples
  
# learning rate in epoch k is 
# lr_init*lr_gamma**floor(k/lr_step_size)
lr_init=0.001
lr_gamma=0.9
lr_step_size= 400


# Set this to False if running the code on a remote server.
# Set this to True if running the code on a local PC 
# to monitor the training process.
show_plot = True 

interval=[-1,1]
ranks = [1] + [36]*5 + [1]
widths = [666]*6
model = mmnn.MMNN(ranks = ranks, 
                 widths = widths,
                 device = device,
                 ResNet = False)


x_train = np.linspace(*interval, num_training_samples).reshape([-1, 1])
y_train = func(x_train)
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
              
    if epoch % 50 == 0:
        training_error = loss.item()
        print(f"\nEpoch {epoch} / {num_epochs}" + 
              f"  ( {epoch/num_epochs*100:.2f}% )" +
              f"\nTraining error (MSE): { training_error :.2e}" + 
              f"\nTime used: { time.time() - time1 :.2f}s")
        errors_train.append(training_error)
    
        def learned_nn(x): # input and output are numpy.ndarray  
            x=x.reshape([-1, 1]) 
            input_data = torch.tensor(x, dtype=mydtype).to(device)
            y = model(input_data)
            y = y.cpu().detach().numpy().reshape([-1])
            return y     
        
        
        x = np.random.rand(num_test_samples) * 2 - 1
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
        
        if epoch % 100 == 0:
            # Plot the results
            x = np.linspace(-1, 1, 1000)
            y_nn = learned_nn(x)
            y_true = func(x)
            fig=plt.figure(figsize=(6,4))
            plt.plot(x, y_true, label='true function')
            plt.plot(x, y_nn, label='learned network')
            plt.xticks(np.linspace(*interval,5))
            plt.tick_params(axis='both', 
                            which='major', labelsize=12)
            plt.grid(True, axis='both', color='#AAAAAA', 
                      linestyle='--', linewidth=1.4)
            plt.title(f'true function and learned network in (Epoch {epoch})')
            plt.tight_layout()
            plt.legend(loc="upper center" , fontsize=13,  ncol=2,
                )
    
            FPN="./figures/"
            if not os.path.exists(FPN):
                os.makedirs(FPN)
            plt.savefig(f"{FPN}mmnn_epoch{epoch}1D.png", dpi=50)
            if show_plot:
                plt.show()

torch.save(model.state_dict(), 'model_parameters1D.pth')
np.savez("errors1D", 
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



