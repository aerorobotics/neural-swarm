from utils import interpolation_cubic, data_extraction, Merge, Fa, get_data, hist, hist_all, set_generate, data_filter
from vis_validation import vis, vis_paper
from nns import phi_Net, rho_Net
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from collections import defaultdict

# output will be written to ../data/models/<output_name> folder
# output_name = "val_onlyL/epoch20_lip3_h20_f0d4_B256"
output_name = "test"
lip = 3
num_epochs = 5
hidden_dim = 20
batch_size = 256
rasterized = True # set to True, to rasterize the pictures in the PDF
fa_type = 'fa_delay' # 'fa_imu', fa_num', 'fa_delay'
x_threshold = 0.4 # threshold for data filtering
y_threshold = 0.4 # threshold for data filtering
g_threshold = [0.07, 0.085] # threshold for ground touching
Filter = True
always_GE = True

# 0:Ge2L 1:Ge2S 2:L2L  3:S2S  4:L2S 5:S2L
# 6:SS2L 7:SL2L 8:LL2S 9:SL2S 10:SS2S 11:LL2L
encoder = {'Ge2L':0, 'Ge2S':1, 'L2L':2, 'S2S':3, 'L2S':4, 'S2L':5, \
           'SS2L':6, 'SL2L':7, 'LL2S':8, 'SL2S':9, 'SS2S':10, 'LL2L': 11}

# This might throw an exception as a safety measure to avoid
# that previously learned files are overwritten
os.makedirs('../data/models/{}'.format(output_name))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('Using device:', device)

torch.set_default_tensor_type('torch.DoubleTensor')
torch.multiprocessing.set_sharing_strategy('file_system')
pp = PdfPages('../data/models/{}/output.pdf'.format(output_name))

##### Part I: Data generation and interpolation #####
print('***** Data generation and interpolation! *****')
# From datacollection19
Data_LL_L1_list = []
Data_LL_L2_list = []
# From datacollection20
Data_LGe_list = []
# From datacollection21
Data_LLL_L1_list = []
Data_LLL_L2_list = []
Data_LLL_L3_list = []

### Data collection 19 ###
# L-L random walk; cf101 & cf102; total: 127.55s
# ./random_walk_large_large/
# 00     01
# 63.73s 63.82s
TF = np.array([63.73, 63.82])
S = ['00', '01']
name = '../data/training/datacollection19_12_11_2019/random_walk_large_large/'
for i in range(len(TF)):
    Data_LL_L1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf101_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_LL_L2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf102_'+S[i]+'.csv'), ss=0, ee=-1))

# L-L swap; cf101 & cf102; total: 59.77s
# ./swap_large_large/
# 02
# 59.77s 
TF = np.array([59.77])
S = ['02']
name = '../data/training/datacollection19_12_11_2019/swap_large_large/'
for i in range(len(TF)):
    Data_LL_L1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf101_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_LL_L2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf102_'+S[i]+'.csv'), ss=0, ee=-1))

### Data collection 20 ###
# L-Ground random walk; cf101; 126.51s
# ./random_walk_large_ground/
# 01     02
# 63.62s 62.89s
TF = np.array([63.62, 62.89])
S = ['01', '02']
name = '../data/training/datacollection20_12_20_2019/random_walk_large_ground/'
for i in range(len(TF)):
    Data_LGe_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf101_'+S[i]+'.csv'), ss=0, ee=-1))

##### Data Collection 21 #####
# (1) ./randomwalk_l; cf 102
# A few light touches but looks Okay
name = '../data/training/datacollection21_06_16_2020/randomwalk_l/'
S = ['01', '02', '03', '04', '05', '06', '07']
for s in S:
    data = data_extraction(name+'cf102_'+s+'.csv')
    TF = np.floor(data['time'][-1]*100)/100.0 - 0.01
    Data_LGe_list.append(interpolation_cubic(0, TF, data, ss=0, ee=-1))

# (2) ./randomwalk_ll; cf 101 & 102
# ['00', '01', '02', '03', '04', '05']
# A few touches -> need filtering for sure!
name = '../data/training/datacollection21_06_16_2020/randomwalk_ll/'
S = ['00', '01', '02', '03', '04', '05']
for s in S:
    data_1 = data_extraction(name+'cf101_'+s+'.csv')
    data_2 = data_extraction(name+'cf102_'+s+'.csv')
    TF_1 = np.floor(data_1['time'][-1]*100)/100.0 - 0.01
    TF_2 = np.floor(data_2['time'][-1]*100)/100.0 - 0.01
    TF = min(TF_1, TF_2)
    Data_LL_L1_list.append(interpolation_cubic(0, TF, data_1, ss=0, ee=-1))
    Data_LL_L2_list.append(interpolation_cubic(0, TF, data_2, ss=0, ee=-1))

# (9) ./swap_ll; cf 101 & 102
# ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
name = '../data/training/datacollection21_06_16_2020/swap_ll/'
S = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
for s in S:
    data_1 = data_extraction(name+'cf101_'+s+'.csv')
    data_2 = data_extraction(name+'cf102_'+s+'.csv')
    TF_1 = np.floor(data_1['time'][-1]*100)/100.0 - 0.01
    TF_2 = np.floor(data_2['time'][-1]*100)/100.0 - 0.01
    TF = min(TF_1, TF_2)
    Data_LL_L1_list.append(interpolation_cubic(0, TF, data_1, ss=0, ee=-1))
    Data_LL_L2_list.append(interpolation_cubic(0, TF, data_2, ss=0, ee=-1))

# (10) ./swap_lll; cf 100 & 101 & 102
# ['01', '03']
# WARNING: DON'T USE IMU FOR CF 100
name = '../data/training/datacollection21_06_16_2020/swap_lll/'
S = ['01', '03']
for s in S:
    data_1 = data_extraction(name+'cf100_'+s+'.csv')
    data_2 = data_extraction(name+'cf101_'+s+'.csv')
    data_3 = data_extraction(name+'cf102_'+s+'.csv')
    TF_1 = np.floor(data_1['time'][-1]*100)/100.0 - 0.01
    TF_2 = np.floor(data_2['time'][-1]*100)/100.0 - 0.01
    TF_3 = np.floor(data_3['time'][-1]*100)/100.0 - 0.01
    TF = min(TF_1, TF_2, TF_3)
    Data_LLL_L1_list.append(interpolation_cubic(0, TF, data_1, ss=0, ee=-1))
    Data_LLL_L2_list.append(interpolation_cubic(0, TF, data_2, ss=0, ee=-1))
    Data_LLL_L3_list.append(interpolation_cubic(0, TF, data_3, ss=0, ee=-1))

# (16) ./takeoff_l; cf 102
# ['00']
name = '../data/training/datacollection21_06_16_2020/takeoff_l/'
S = ['00']
for s in S:
    data = data_extraction(name+'cf102_'+s+'.csv')
    TF = np.floor(data['time'][-1]*100)/100.0 - 0.01
    Data_LGe_list.append(interpolation_cubic(0, TF, data, ss=0, ee=-1))

##### Data Collection 22 #####
# (1) ./randomwalk_lll; cf 100 & 101 & 102
name = '../data/training/datacollection22_06_19_2020/randomwalk_lll/'
S = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
for s in S:
    data_1 = data_extraction(name+'cf100_'+s+'.csv')
    data_2 = data_extraction(name+'cf101_'+s+'.csv')
    data_3 = data_extraction(name+'cf102_'+s+'.csv')
    TF_1 = np.floor(data_1['time'][-1]*100)/100.0 - 0.01
    TF_2 = np.floor(data_2['time'][-1]*100)/100.0 - 0.01
    TF_3 = np.floor(data_3['time'][-1]*100)/100.0 - 0.01
    TF = min(TF_1, TF_2, TF_3)
    Data_LLL_L1_list.append(interpolation_cubic(0, TF, data_1, ss=0, ee=-1))
    Data_LLL_L2_list.append(interpolation_cubic(0, TF, data_2, ss=0, ee=-1))
    Data_LLL_L3_list.append(interpolation_cubic(0, TF, data_3, ss=0, ee=-1))

# (5) ./swap_lll; cf 100 & 101 & 102
name = '../data/training/datacollection22_06_19_2020/swap_lll/'
S = ['00', '01', '02', '04', '05', '08']
for s in S:
    data_1 = data_extraction(name+'cf100_'+s+'.csv')
    data_2 = data_extraction(name+'cf101_'+s+'.csv')
    data_3 = data_extraction(name+'cf102_'+s+'.csv')
    TF_1 = np.floor(data_1['time'][-1]*100)/100.0 - 0.02
    TF_2 = np.floor(data_2['time'][-1]*100)/100.0 - 0.02
    TF_3 = np.floor(data_3['time'][-1]*100)/100.0 - 0.02
    TF = min(TF_1, TF_2, TF_3)
    Data_LLL_L1_list.append(interpolation_cubic(0, TF, data_1, ss=0, ee=-1))
    Data_LLL_L2_list.append(interpolation_cubic(0, TF, data_2, ss=0, ee=-1))
    Data_LLL_L3_list.append(interpolation_cubic(0, TF, data_3, ss=0, ee=-1))

##### Part II: Data merge #####
print('***** Data merge! *****')
Data_LL_L1 = Merge(Data_LL_L1_list)
Data_LL_L2 = Merge(Data_LL_L2_list)
Data_LGe = Merge(Data_LGe_list)
Data_LLL_L1 = Merge(Data_LLL_L1_list)
Data_LLL_L2 = Merge(Data_LLL_L2_list)
Data_LLL_L3 = Merge(Data_LLL_L3_list)


##### Part III: Fa computation #####
print('***** Fa computation! *****')
# big CF
m = 67
g = 9.81
C_00 = 44.10386631845999
C_10 = -122.51151800146272
C_01 = -36.18484254283743
C_20 = 53.10772568607133
C_11 = 107.6819263349139
Data_LL_L1 = Fa(Data_LL_L1, m, g, C_00, C_10, C_01, C_20, C_11)
Data_LL_L2 = Fa(Data_LL_L2, m, g, C_00, C_10, C_01, C_20, C_11)

Data_LGe = Fa(Data_LGe, m, g, C_00, C_10, C_01, C_20, C_11)

Data_LLL_L1 = Fa(Data_LLL_L1, m, g, C_00, C_10, C_01, C_20, C_11)
Data_LLL_L2 = Fa(Data_LLL_L2, m, g, C_00, C_10, C_01, C_20, C_11)
Data_LLL_L3 = Fa(Data_LLL_L3, m, g, C_00, C_10, C_01, C_20, C_11)


##### Part IV: Generate input-output pair #####
print('***** Input-output pair generation! *****')
data_input_Ge2L, data_output_Ge2L = get_data(D1=Data_LGe, D2=None, s=encoder['Ge2L'], typ=fa_type, always_GE=always_GE)
print('Ge2L:', data_input_Ge2L.shape, data_output_Ge2L.shape)

data_input_L2L_a, data_output_L2L_a = get_data(D1=Data_LL_L1, D2=Data_LL_L2, s=encoder['L2L'], typ=fa_type, always_GE=always_GE)
data_input_L2L_b, data_output_L2L_b = get_data(D1=Data_LL_L2, D2=Data_LL_L1, s=encoder['L2L'], typ=fa_type, always_GE=always_GE)
data_input_L2L = np.vstack((data_input_L2L_a, data_input_L2L_b))
data_output_L2L = np.vstack((data_output_L2L_a, data_output_L2L_b))
print('L2L:', data_input_L2L.shape, data_output_L2L.shape)

# CF 100 HAS BAD FA!
# data_input_LL2L_a, data_output_LL2L_a = get_data(D1=Data_LLL_L1, D2=Data_LLL_L2, D3=Data_LLL_L3, s=encoder['LL2L'], typ=fa_type, always_GE=always_GE)
data_input_LL2L_b, data_output_LL2L_b = get_data(D1=Data_LLL_L2, D2=Data_LLL_L1, D3=Data_LLL_L3, s=encoder['LL2L'], typ=fa_type, always_GE=always_GE)
data_input_LL2L_c, data_output_LL2L_c = get_data(D1=Data_LLL_L3, D2=Data_LLL_L1, D3=Data_LLL_L2, s=encoder['LL2L'], typ=fa_type, always_GE=always_GE)
# data_input_LL2L = np.vstack((data_input_LL2L_a, data_input_LL2L_b, data_input_LL2L_c))
# data_output_LL2L = np.vstack((data_output_LL2L_a, data_output_LL2L_b, data_output_LL2L_c))
data_input_LL2L = np.vstack((data_input_LL2L_b, data_input_LL2L_c))
data_output_LL2L = np.vstack((data_output_LL2L_b, data_output_LL2L_c))
print('LL2L:', data_input_LL2L.shape, data_output_LL2L.shape)

Data_input_all = [data_input_Ge2L, data_input_L2L, data_input_LL2L]
Data_output_all = [data_output_Ge2L, data_output_L2L, data_output_LL2L]
Name = ['Ge2L', 'L2L', 'LL2L']

if False:
    # visualization of data distribution
    for i in range(len(Name)):
        hist(pp, Data_input_all[i], Data_output_all[i], Name[i], rasterized)

hist_all(pp, Data_output_all, Name, rasterized, note='before filter')

# Data filter 
if Filter:
    for i in range(len(Name)):
        Data_input_all[i], Data_output_all[i], ratio_g, ratio_xy = data_filter(Data_input_all[i], Data_output_all[i], x_threshold=x_threshold, y_threshold=y_threshold, g_threshold=g_threshold)
        print(Name[i]+': Filter ratio for ground touching and xy are,', ratio_g, ratio_xy)

if False:
    # visualization of data distribution
    for i in range(len(Name)):
        hist(pp, Data_input_all[i], Data_output_all[i], Name[i], rasterized)

hist_all(pp, Data_output_all, Name, rasterized, note='after filter')

# generate torch trainset and trainloader
trainset_Ge2L, trainloader_Ge2L, valset_Ge2L, val_input_Ge2L, val_output_Ge2L = set_generate(Data_input_all[0], Data_output_all[0], 'Ge2L', device, batch_size)
trainset_L2L, trainloader_L2L, valset_L2L, val_input_L2L, val_output_L2L = set_generate(Data_input_all[1], Data_output_all[1], 'L2L', device, batch_size)
trainset_LL2L, trainloader_LL2L, valset_LL2L, val_input_LL2L, val_output_LL2L = set_generate(Data_input_all[2], Data_output_all[2], 'LL2L', device, batch_size)


##### Part V: Training #####
print('***** Training! *****')
# ground effect doesn't consider x and y
phi_G_net = phi_Net(inputdim=4,hiddendim=hidden_dim).to(device, dtype=torch.float32)
phi_L_net = phi_Net(inputdim=6,hiddendim=hidden_dim).to(device, dtype=torch.float32)
rho_L_net = rho_Net(hiddendim=hidden_dim).to(device, dtype=torch.float32)
phi_S_net = phi_Net(inputdim=6,hiddendim=hidden_dim).to(device, dtype=torch.float32)
rho_S_net = rho_Net(hiddendim=hidden_dim).to(device, dtype=torch.float32)

criterion = nn.MSELoss()
optimizer_phi_G = optim.Adam(phi_G_net.parameters(), lr=1e-3)
optimizer_phi_L = optim.Adam(phi_L_net.parameters(), lr=1e-3)
optimizer_rho_L = optim.Adam(rho_L_net.parameters(), lr=1e-3)

def set_loss(set, criterion, rho_net, phi_1_net, phi_2_net=None, phi_3_net=None):
    with torch.no_grad():
        inputs = set[:]['input'] 
        label = set[:]['output']
        if phi_2_net is None and phi_3_net is None:
            loss = criterion(rho_net(phi_1_net(inputs[:, 2:6])), label)
        else:
            if phi_3_net is None:
                loss = criterion(rho_net(phi_1_net(inputs[:, 2:6]) + phi_2_net(inputs[:, 6:12])), label)
            else:
                loss = criterion(rho_net(phi_1_net(inputs[:, 2:6]) + phi_2_net(inputs[:, 6:12]) + phi_2_net(inputs[:, 12:18])), label)
        # if phi_2_net is None:
        #     if GE:
        #         loss = criterion(rho_net(phi_1_net(inputs[:, 2:6])), label)
        #     else:
        #         loss = criterion(rho_net(phi_1_net(inputs[:, :6])), label)
        # else:
        #     loss = criterion(rho_net(phi_1_net(inputs[:, :6]) + phi_2_net(inputs[:, 6:12])), label)
    return loss.item()

# Loss before training
# 0:Ge2L 1:Ge2S 2:L2L  3:S2S  4:L2S 5:S2L
# 6:SS2L 7:SL2L 8:LL2S 9:SL2S 10:SS2S 11:LL2L
print('Ge2L loss b4 training', set_loss(trainset_Ge2L, criterion, rho_L_net, phi_G_net))
print('L2L loss b4 training', set_loss(trainset_L2L, criterion, rho_L_net, phi_G_net, phi_2_net=phi_L_net))
print('LL2L loss b4 training', set_loss(trainset_LL2L, criterion, rho_L_net, phi_G_net, phi_2_net=phi_L_net, phi_3_net=phi_L_net))

# training
Loss_sn = []
# mix all the data
mixed = []
Count = defaultdict(int)
for data in trainloader_Ge2L:
    Count['Ge2L'] += 1
    mixed.append(data)
for data in trainloader_L2L:
    Count['L2L'] += 1
    mixed.append(data)
for data in trainloader_LL2L:
    Count['LL2L'] += 1
    mixed.append(data)

# Spectral normalization
def Lip(net, lip):
    net.cpu()
    for param in net.parameters():
        M = param.detach().numpy()
        if M.ndim > 1:
            s = np.linalg.norm(M, 2)
            if s > lip:
                param.data = param / s * lip
    net.to(device, dtype=torch.float32)

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    count = Count.copy()
    shuffle(mixed)
    for data in mixed:
        # get the inputs
        inputs = data['input']
        labels = data['output']
        datatype = data['type'][0]

        # zero the parameter gradients
        optimizer_phi_G.zero_grad()
        optimizer_phi_L.zero_grad()
        optimizer_rho_L.zero_grad()

        # forward + backward + optimize
        if datatype == 'Ge2L':
            outputs = rho_L_net(phi_G_net(inputs[:, 2:6]))
        elif datatype == 'L2L':
            outputs = rho_L_net(phi_G_net(inputs[:, 2:6]) + phi_L_net(inputs[:, 6:12]))
        elif datatype == 'LL2L':
            outputs = rho_L_net(phi_G_net(inputs[:, 2:6]) + phi_L_net(inputs[:, 6:12]) + phi_L_net(inputs[:, 12:18]))
        else:
            print('wrong class', datatype)
        
        count[datatype] -= 1

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_phi_G.step()
        optimizer_phi_L.step()
        optimizer_rho_L.step()
        
        # Lip
        Lip(phi_G_net, lip)
        Lip(phi_L_net, lip)
        Lip(rho_L_net, lip)
                    
        running_loss += loss.item()

    if np.max(np.abs(np.array(list(count.values())))) != 0:
        print('something goes wrong!')
        print(count)
        break
    Loss_sn.append(running_loss)
    if epoch % 5 == 0:
        print('[%d] loss: %.3f' % (epoch + 1, running_loss))

print('Training finished!')
plt.figure()
plt.plot(Loss_sn, rasterized=rasterized)
plt.title('Training loss')
pp.savefig()
plt.close()

# Loss after training
print('Ge2L loss after training', set_loss(trainset_Ge2L, criterion, rho_L_net, phi_G_net))
print('L2L loss after training', set_loss(trainset_L2L, criterion, rho_L_net, phi_G_net, phi_2_net=phi_L_net))
print('LL2L loss after training', set_loss(trainset_LL2L, criterion, rho_L_net, phi_G_net, phi_2_net=phi_L_net, phi_3_net=phi_L_net))

Error = []
Error.append(set_loss(valset_Ge2L, criterion, rho_L_net, phi_G_net))
Error.append(set_loss(valset_L2L, criterion, rho_L_net, phi_G_net, phi_2_net=phi_L_net))
Error.append(set_loss(valset_LL2L, criterion, rho_L_net, phi_G_net, phi_2_net=phi_L_net, phi_3_net=phi_L_net))
Error = np.array(Error)
weight = np.array([len(valset_Ge2L), len(valset_L2L), len(valset_LL2L)])
mean_error = np.sum(Error*weight) / np.sum(weight)
print('Validation error:')
print(Error)
print('mean:')
print(mean_error)

phi_G_net.cpu()
phi_L_net.cpu()
rho_L_net.cpu()
rho_S_net.cpu()
phi_S_net.cpu()

if True:
    torch.save(phi_G_net.state_dict(), '../data/models/{}/phi_G.pth'.format(output_name))
    torch.save(rho_L_net.state_dict(), '../data/models/{}/rho_L.pth'.format(output_name))
    torch.save(phi_L_net.state_dict(), '../data/models/{}/phi_L.pth'.format(output_name))
    torch.save(rho_S_net.state_dict(), '../data/models/{}/rho_S.pth'.format(output_name))
    torch.save(phi_S_net.state_dict(), '../data/models/{}/phi_S.pth'.format(output_name))
    print('Models saved!')


##### Part VI: Visualization and validation #####
print('***** Visualization and validation! *****')
phi_G_net.load_state_dict(torch.load('../data/models/{}/phi_G.pth'.format(output_name)))
rho_L_net.load_state_dict(torch.load('../data/models/{}/rho_L.pth'.format(output_name)))
rho_S_net.load_state_dict(torch.load('../data/models/{}/rho_S.pth'.format(output_name)))
phi_L_net.load_state_dict(torch.load('../data/models/{}/phi_L.pth'.format(output_name)))
phi_S_net.load_state_dict(torch.load('../data/models/{}/phi_S.pth'.format(output_name)))
vis(pp, phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net, rasterized)
vis_paper(pp, phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net, rasterized)

# Val of NNs
# 0:Ge2L 1:Ge2S 2:L2L  3:S2S  4:L2S 5:S2L
# 6:SS2L 7:SL2L 8:LL2S 9:SL2S 10:SS2S 11:LL2L
def Fa_prediction(data_input, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net):
    L = len(data_input)
    Fa = np.zeros(L)
    for i in range(L):
        with torch.no_grad():
            inputs = torch.from_numpy(data_input[[i], :])
            temp = inputs[:, -1].item()
            temp = int(temp)
            if temp == encoder['Ge2L']:
                outputs = rho_L_net(phi_G_net(inputs[:, 2:6]))
            elif temp == encoder['Ge2S']:
                outputs = rho_S_net(phi_G_net(inputs[:, 2:6]))
            elif temp == encoder['L2L']:
                outputs = rho_L_net(phi_G_net(inputs[:, 2:6]) + phi_L_net(inputs[:, 6:12]))
            elif temp == encoder['S2S']:
                outputs = rho_S_net(phi_G_net(inputs[:, 2:6]) + phi_S_net(inputs[:, 6:12]))
            elif temp == encoder['L2S']:
                outputs = rho_S_net(phi_G_net(inputs[:, 2:6]) + phi_L_net(inputs[:, 6:12]))
            elif temp == encoder['S2L']:
                outputs = rho_L_net(phi_G_net(inputs[:, 2:6]) + phi_S_net(inputs[:, 6:12]))
            elif temp == encoder['SS2L']:
                outputs = rho_L_net(phi_G_net(inputs[:, 2:6]) + phi_S_net(inputs[:, 6:12]) + phi_S_net(inputs[:, 12:18]))
            elif temp == encoder['SL2L']:
                outputs = rho_L_net(phi_G_net(inputs[:, 2:6]) + phi_S_net(inputs[:, 6:12]) + phi_L_net(inputs[:, 12:18]))
            elif temp == encoder['LL2S']:
                outputs = rho_S_net(phi_G_net(inputs[:, 2:6]) + phi_L_net(inputs[:, 6:12]) + phi_L_net(inputs[:, 12:18]))
            elif temp == encoder['SL2S']:
                outputs = rho_S_net(phi_G_net(inputs[:, 2:6]) + phi_S_net(inputs[:, 6:12]) + phi_L_net(inputs[:, 12:18]))
            elif temp == encoder['SS2S']:
                outputs = rho_S_net(phi_G_net(inputs[:, 2:6]) + phi_S_net(inputs[:, 6:12]) + phi_S_net(inputs[:, 12:18]))
            elif temp == encoder['LL2L']:
                outputs = rho_L_net(phi_G_net(inputs[:, 2:6]) + phi_L_net(inputs[:, 6:12]) + phi_L_net(inputs[:, 12:18]))
            else:
                print('wrong class', temp)
            Fa[i] = outputs[0, 0].item()
    return Fa

# Val of NNs
def validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, data_input, data_output, ss, ee, name):
    Fa_pred = Fa_prediction(data_input, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net)
    plt.figure(figsize=(12, 15))
    plt.subplot(4, 1, 1, rasterized=rasterized)
    plt.plot(-data_input[:, :3])
    plt.legend(['x', 'y', 'z'])
    plt.grid()
    plt.title('Validation: '+name)
    plt.subplot(4, 1, 2, rasterized=rasterized)
    plt.plot(data_input[:, 6:9])
    plt.legend(['x21', 'y21', 'z21'])
    plt.grid()
    plt.subplot(4, 1, 3, rasterized=rasterized)
    plt.plot(data_input[:, 12:15])
    plt.legend(['x31', 'y31', 'z31'])
    plt.grid()
    plt.subplot(4, 1, 4, rasterized=rasterized)
    plt.plot(data_output[:, 2])
    plt.hlines(y=0, xmin=0, xmax=ee-ss, colors='r')
    plt.plot(Fa_pred)
    plt.legend(['fa_gt', 'fa_pred'])
    plt.grid()
    pp.savefig()
    plt.close()
    #plt.show()

validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_Ge2L, val_output_Ge2L, ss=0, ee=-1, name='Ge2L')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_L2L, val_output_L2L, ss=0, ee=-1, name='L2L')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_LL2L, val_output_LL2L, ss=0, ee=-1, name='LL2L')

pp.close()
print('***** Output PDF saved! *****')