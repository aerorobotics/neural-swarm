from utils import interpolation_cubic, data_extraction, Merge, Fa, get_data, hist, set_generate, vis, validation
from nns import phi_Net, rho_Net
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from random import shuffle
import matplotlib.pyplot as plt
torch.set_default_tensor_type('torch.DoubleTensor')
torch.multiprocessing.set_sharing_strategy('file_system')


##### Part I: Data generation and interpolation #####
print('***** Data generation and interpolation! *****')
Data_SS_S1_list = []
Data_SS_S2_list = []
Data_LL_L1_list = []
Data_LL_L2_list = []
Data_LS_L_list = []
Data_LS_S_list = []

# L-L random walk; cf101 & cf102; total: 127.55s
# ./random_walk_large_large/
# 00     01
# 63.73s 63.82s
TF = np.array([63.73, 63.82])
S = ['00', '01']
name = '../../datacollection19_12_11_2019_2/random_walk_large_large/'
for i in range(len(TF)):
    Data_LL_L1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf101_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_LL_L2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf102_'+S[i]+'.csv'), ss=0, ee=-1))

# L-S random walk; cf102 & cf50; total: 127.18s
# ./random_walk_large_small/
# 02     03
# 63.74s 63.44s
TF = np.array([63.74, 63.44])
S = ['02', '03']
name = '../../datacollection19_12_11_2019_2/random_walk_large_small/'
for i in range(len(TF)):
    Data_LS_L_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf102_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_LS_S_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))

# S-S random walk; cf50 & cf51; total: 127.45s
# ./random_walk_small_small/
# 04     05
# 63.62s 63.83s
TF = np.array([63.62, 63.83])
S = ['04', '05']
name = '../../datacollection19_12_11_2019_2/random_walk_small_small/'
for i in range(len(TF)):
    Data_SS_S1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SS_S2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf51_'+S[i]+'.csv'), ss=0, ee=-1))
    
# L-L swap; cf101 & cf102; total: 59.77s
# ./swap_large_large/
# 02
# 59.77s 
TF = np.array([59.77])
S = ['02']
name = '../../datacollection19_12_11_2019_2/swap_large_large/'
for i in range(len(TF)):
    Data_LL_L1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf101_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_LL_L2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf102_'+S[i]+'.csv'), ss=0, ee=-1))

# S-S swap; cf50 & cf51; total: 59.92s
# ./swap_small_small/
# 02
# 59.92s
TF = np.array([59.92])
S = ['02']
name = '../../datacollection19_12_11_2019_2/swap_small_small/'
for i in range(len(TF)):
    Data_SS_S1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SS_S2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf51_'+S[i]+'.csv'), ss=0, ee=-1))

# L-S swap; cf102 & cf50; total: 119.59s
# ./swap_large_small/
# 00(LS) 01(SL)
# 59.79s 59.80s
TF = np.array([59.79, 59.80])
S = ['00', '01']
name = '../../datacollection19_12_11_2019_2/swap_large_small/'
for i in range(len(TF)):
    Data_LS_L_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf102_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_LS_S_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))


##### Part II: Data merge #####
print('***** Data merge! *****')
Data_LL_L1 = Merge(Data_LL_L1_list)
Data_LL_L2 = Merge(Data_LL_L2_list)
Data_SS_S1 = Merge(Data_SS_S1_list)
Data_SS_S2 = Merge(Data_SS_S2_list)
Data_LS_L = Merge(Data_LS_L_list)
Data_LS_S = Merge(Data_LS_S_list)


##### Part III: Fa computation #####
print('***** Fa computation! *****')
# small CF
m = 67
g = 9.81
C_00 = 44.10386631845999
C_10 = -122.51151800146272
C_01 = -36.18484254283743
C_20 = 53.10772568607133
C_11 = 107.6819263349139
Data_LL_L1 = Fa(Data_LL_L1, m, g, C_00, C_10, C_01, C_20, C_11)
Data_LL_L2 = Fa(Data_LL_L2, m, g, C_00, C_10, C_01, C_20, C_11)
Data_LS_L = Fa(Data_LS_L, m, g, C_00, C_10, C_01, C_20, C_11)

# big CF
m = 32
g = 9.81
C_00 = 11.093358483549203
C_10 = -39.08104165843915
C_01 = -9.525647087583181
C_20 = 20.573302305476638
C_11 = 38.42885066644033
Data_SS_S1 = Fa(Data_SS_S1, m, g, C_00, C_10, C_01, C_20, C_11)
Data_SS_S2 = Fa(Data_SS_S2, m, g, C_00, C_10, C_01, C_20, C_11)
Data_LS_S = Fa(Data_LS_S, m, g, C_00, C_10, C_01, C_20, C_11)


##### Part IV: Generate input-output pair #####
print('***** Input-output pair generation! *****')
data_input_L2L_a, data_output_L2L_a = get_data(D1=Data_LL_L1, D2=Data_LL_L2, s=0)
data_input_L2L_b, data_output_L2L_b = get_data(D1=Data_LL_L2, D2=Data_LL_L1, s=0)
data_input_L2L = np.vstack((data_input_L2L_a, data_input_L2L_b))
data_output_L2L = np.vstack((data_output_L2L_a, data_output_L2L_b))
print('L2L:', data_input_L2L.shape, data_output_L2L.shape)

data_input_S2S_a, data_output_S2S_a = get_data(D1=Data_SS_S1, D2=Data_SS_S2, s=1)
data_input_S2S_b, data_output_S2S_b = get_data(D1=Data_SS_S2, D2=Data_SS_S1, s=1)
data_input_S2S = np.vstack((data_input_S2S_a, data_input_S2S_b))
data_output_S2S = np.vstack((data_output_S2S_a, data_output_S2S_b))
print('S2S:', data_input_S2S.shape, data_output_S2S.shape)

data_input_L2S, data_output_L2S = get_data(D1=Data_LS_S, D2=Data_LS_L, s=2)
data_input_S2L, data_output_S2L = get_data(D1=Data_LS_L, D2=Data_LS_S, s=3)
print('L2S:', data_input_L2S.shape, data_output_L2S.shape)
print('S2L:', data_input_S2L.shape, data_output_S2L.shape)

if False:
    # visualization of data distribution
    hist(data_input_L2L, data_output_L2L, name='L2L')
    hist(data_input_S2S, data_output_S2S, name='S2S')
    hist(data_input_L2S, data_output_L2S, name='L2S')
    hist(data_input_S2L, data_output_S2L, name='S2L')

# generate torch trainset and trainloader
trainset_L2L, trainloader_L2L = set_generate(data_input_L2L, data_output_L2L)
trainset_S2S, trainloader_S2S = set_generate(data_input_S2S, data_output_S2S)
trainset_L2S, trainloader_L2S = set_generate(data_input_L2S, data_output_L2S)
trainset_S2L, trainloader_S2L = set_generate(data_input_S2L, data_output_S2L)


##### Part V: Training #####
print('***** Training! *****')
phi_L_net = phi_Net(H=20)
phi_S_net = phi_Net(H=20)
rho_L_net = rho_Net(H=20)
rho_S_net = rho_Net(H=20)

criterion = nn.MSELoss()
optimizer_phi_L = optim.Adam(phi_L_net.parameters(), lr=1e-3)
optimizer_rho_L = optim.Adam(rho_L_net.parameters(), lr=1e-3)
optimizer_phi_S = optim.Adam(phi_S_net.parameters(), lr=1e-3)
optimizer_rho_S = optim.Adam(rho_S_net.parameters(), lr=1e-3)

def set_loss(set, criterion, rho_net, phi_net):
    with torch.no_grad():
        inputs = set[:]['input'] 
        label = set[:]['output']
        loss = criterion(rho_net(phi_net(inputs[:, :6])), label)
    return loss.item()

# Loss before training
print('L2L loss b4 training', set_loss(trainset_L2L, criterion, rho_L_net, phi_L_net))
print('S2S loss b4 training', set_loss(trainset_S2S, criterion, rho_S_net, phi_S_net))
print('L2S loss b4 training', set_loss(trainset_L2S, criterion, rho_S_net, phi_L_net))
print('S2L loss b4 training', set_loss(trainset_S2L, criterion, rho_L_net, phi_S_net))

# training
Loss_sn = []
B = 64 # batch size
# mix all the data
mixed = []
Count_L2L = 0
Count_S2S = 0
Count_L2S = 0
Count_S2L = 0
for i, data in enumerate(trainloader_L2L, 0):
    Count_L2L += 1
    mixed.append(data)
for i, data in enumerate(trainloader_S2S, 0):
    Count_S2S += 1
    mixed.append(data)
for i, data in enumerate(trainloader_L2S, 0):
    Count_L2S += 1
    mixed.append(data)
for i, data in enumerate(trainloader_S2L, 0):
    Count_S2L += 1
    mixed.append(data)      

for epoch in range(150):  # loop over the dataset multiple times
    running_loss = 0.0
    count_L2L = Count_L2L
    count_S2S = Count_S2S
    count_L2S = Count_L2S
    count_S2L = Count_S2L
    shuffle(mixed)
    for data in mixed:
        # get the inputs
        inputs = data['input']
        labels = data['output']

        # zero the parameter gradients
        optimizer_phi_L.zero_grad()
        optimizer_rho_L.zero_grad()
        optimizer_phi_S.zero_grad()
        optimizer_rho_S.zero_grad()

        # forward + backward + optimize
        # L2L:0 S2S:1 L2S:2 S2L:3
        temp = torch.mean(inputs[:, 6]).item()
        temp = int(temp)
        if temp == 0:
            outputs = rho_L_net(phi_L_net(inputs[:, :6]))
            count_L2L -= 1
        elif temp == 1:
            outputs = rho_S_net(phi_S_net(inputs[:, :6]))
            count_S2S -= 1
        elif temp == 2:
            outputs = rho_S_net(phi_L_net(inputs[:, :6]))
            count_L2S -= 1
        elif temp == 3:
            outputs = rho_L_net(phi_S_net(inputs[:, :6]))
            count_S2L -= 1
        else:
            print('wrong class', temp)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_phi_L.step()
        optimizer_phi_S.step()
        optimizer_rho_L.step()
        optimizer_rho_S.step()
        
        # Lip
        for param in phi_L_net.parameters():
            M = param.detach().numpy()
            if M.ndim > 1:
                s = np.linalg.norm(M, 2)
                if s > 3.5:
                    param.data = param / s * 3.5
        for param in phi_S_net.parameters():
            M = param.detach().numpy()
            if M.ndim > 1:
                s = np.linalg.norm(M, 2)
                if s > 3.5:
                    param.data = param / s * 3.5
        for param in rho_L_net.parameters():
            M = param.detach().numpy()
            if M.ndim > 1:
                s = np.linalg.norm(M, 2)
                if s > 3.5:
                    param.data = param / s * 3.5
        for param in rho_S_net.parameters():
            M = param.detach().numpy()
            if M.ndim > 1:
                s = np.linalg.norm(M, 2)
                if s > 3.5:
                    param.data = param / s * 3.5
                    
        running_loss += loss.item()

    if count_L2L != 0 or count_S2S != 0 or count_L2S != 0 or count_S2S != 0:
        print('something goes wrong!')
        print(count_L2L, len(trainloader_L2L))
        print(count_S2S, len(trainloader_S2S))
        print(count_L2S, len(trainloader_L2S))
        print(count_S2L, len(trainloader_S2L))
        break
    Loss_sn.append(running_loss)
    if epoch % 5 == 0:
        print('[%d] loss: %.3f' % (epoch + 1, running_loss))

print('Training finished!')
plt.plot(Loss_sn)
plt.show()

# Loss after training
print('L2L loss after training', set_loss(trainset_L2L, criterion, rho_L_net, phi_L_net))
print('S2S loss after training', set_loss(trainset_S2S, criterion, rho_S_net, phi_S_net))
print('L2S loss after training', set_loss(trainset_L2S, criterion, rho_S_net, phi_L_net))
print('S2L loss after training', set_loss(trainset_S2L, criterion, rho_L_net, phi_S_net))

if True:
    torch.save(rho_L_net.state_dict(), 'rho_L.pth')
    torch.save(phi_L_net.state_dict(), 'phi_L.pth')
    torch.save(rho_S_net.state_dict(), 'rho_S.pth')
    torch.save(phi_S_net.state_dict(), 'phi_S.pth')
    print('Models saved!')


##### Part VI: Visualization and validation #####
print('***** Visualization and validation! *****')
vis(phi_L_net, rho_L_net, phi_S_net, rho_S_net)
validation(phi_L_net, rho_L_net, data_input_L2L, data_output_L2L, ss=0, ee=-1, name='L2L')
validation(phi_S_net, rho_S_net, data_input_S2S, data_output_S2S, ss=0, ee=-1, name='S2S')
validation(phi_L_net, rho_S_net, data_input_L2S, data_output_L2S, ss=0, ee=-1, name='L2S')
validation(phi_S_net, rho_L_net, data_input_S2L, data_output_S2L, ss=0, ee=-1, name='S2L')