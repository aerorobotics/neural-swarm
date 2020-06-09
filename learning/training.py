from utils import interpolation_cubic, data_extraction, Merge, Fa, get_data, hist, set_generate
from vis_validation import vis
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
output_name = "val_new_split/epoch100_lip2_h20"
lip = 2
num_epochs = 100
hidden_dim = 20
batch_size = 128
rasterized = True # set to True, to rasterize the pictures in the PDF
fa_type = 'fa_delay' # 'fa_imu', fa_num', 'fa_delay'

# 0:Ge2L 1:Ge2S 2:L2L  3:S2S  4:L2S 5:S2L
# 6:SS2L 7:SL2L 8:LL2S 9:SL2S 10:SS2S
encoder = {'Ge2L':0, 'Ge2S':1, 'L2L':2, 'S2S':3, 'L2S':4, 'S2L':5, \
           'SS2L':6, 'SL2L':7, 'LL2S':8, 'SL2S':9, 'SS2S':10}

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
Data_SS_S1_list = []
Data_SS_S2_list = []
Data_LL_L1_list = []
Data_LL_L2_list = []
Data_LS_L_list = []
Data_LS_S_list = []
# From datacollection20
Data_LGe_list = []
Data_SGe_list = []
Data_SLL_S_list = []
Data_SLL_L1_list = []
Data_SLL_L2_list = []
Data_SSL_L_list = []
Data_SSL_S1_list = []
Data_SSL_S2_list = []
Data_SSS_S1_list = []
Data_SSS_S2_list = []
Data_SSS_S3_list = []

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

# L-S random walk; cf102 & cf50; total: 127.18s
# ./random_walk_large_small/
# 02     03
# 63.74s 63.44s
TF = np.array([63.74, 63.44])
S = ['02', '03']
name = '../data/training/datacollection19_12_11_2019/random_walk_large_small/'
for i in range(len(TF)):
    Data_LS_L_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf102_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_LS_S_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))

# S-S random walk; cf50 & cf51; total: 127.45s
# ./random_walk_small_small/
# 04     05
# 63.62s 63.83s
TF = np.array([63.62, 63.83])
S = ['04', '05']
name = '../data/training/datacollection19_12_11_2019/random_walk_small_small/'
for i in range(len(TF)):
    Data_SS_S1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SS_S2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf51_'+S[i]+'.csv'), ss=0, ee=-1))
    
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

# S-S swap; cf50 & cf51; total: 59.92s
# ./swap_small_small/
# 02
# 59.92s
TF = np.array([59.92])
S = ['02']
name = '../data/training/datacollection19_12_11_2019/swap_small_small/'
for i in range(len(TF)):
    Data_SS_S1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SS_S2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf51_'+S[i]+'.csv'), ss=0, ee=-1))

# L-S swap; cf102 & cf50; total: 119.59s
# ./swap_large_small/
# 00(LS) 01(SL)
# 59.79s 59.80s
TF = np.array([59.79, 59.80])
S = ['00', '01']
name = '../data/training/datacollection19_12_11_2019/swap_large_small/'
for i in range(len(TF)):
    Data_LS_L_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf102_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_LS_S_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))

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

# S-Ground random walk; cf50; 127.33s
# ./random_walk_small_ground/
# 01     02
# 63.71s 63.62s
TF = np.array([63.71, 63.62])
S = ['01', '02']
name = '../data/training/datacollection20_12_20_2019/random_walk_small_ground/'
for i in range(len(TF)):
    Data_SGe_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))

# SLL random walk; cf50 & cf101 & cf102; 127.58s
# ./random_walk_sll/
# 09     10
# 63.74s 63.84s
TF = np.array([63.74, 63.84])
S = ['09', '10']
name = '../data/training/datacollection20_12_20_2019/random_walk_sll/'
for i in range(len(TF)):
    Data_SLL_S_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SLL_L1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf101_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SLL_L2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf102_'+S[i]+'.csv'), ss=0, ee=-1))

# LSL swap; cf50 & cf101 & cf102; 15.98s
# ./swap_lsl/
# 06
# 15.98s
TF = np.array([15.98])
S = ['06']
name = '../data/training/datacollection20_12_20_2019/swap_lsl/'
for i in range(len(TF)):
    Data_SLL_S_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SLL_L1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf101_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SLL_L2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf102_'+S[i]+'.csv'), ss=0, ee=-1))

# SSL random walk; cf50 & cf52 & cf101; 127.66s
# ./random_walk_ssl/
# 04     13
# 63.83s 63.83s
TF = np.array([63.83, 63.83])
S = ['04', '13']
name = '../data/training/datacollection20_12_20_2019/random_walk_ssl/'
for i in range(len(TF)):
    Data_SSL_L_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf101_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SSL_S1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SSL_S2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf52_'+S[i]+'.csv'), ss=0, ee=-1))

# SLS swap; cf50 & cf52 & cf101; 16.01s
# ./swap_sls/
# 03
# 16.01s
TF = np.array([16.01])
S = ['03']
name = '../data/training/datacollection20_12_20_2019/swap_sls/'
for i in range(len(TF)):
    Data_SSL_L_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf101_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SSL_S1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SSL_S2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf52_'+S[i]+'.csv'), ss=0, ee=-1))

# SSS random walk; cf50 & cf51 & cf52; 127.59s
# ./random_walk_sss/
# 00     01
# 63.83s 63.76s
TF = np.array([63.83, 63.76])
S = ['00', '01']
name = '../data/training/datacollection20_12_20_2019/random_walk_sss/'
for i in range(len(TF)):
    Data_SSS_S1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SSS_S2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf51_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SSS_S3_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf52_'+S[i]+'.csv'), ss=0, ee=-1))

# SSS swap; cf50 & cf51 & cf52; 17.98s
# ./swap_sss/
# 00
# 17.98s
TF = np.array([17.98])
S = ['00']
name = '../data/training/datacollection20_12_20_2019/swap_sss/'
for i in range(len(TF)):
    Data_SSS_S1_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf50_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SSS_S2_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf51_'+S[i]+'.csv'), ss=0, ee=-1))
    Data_SSS_S3_list.append(interpolation_cubic(0, TF[i], data_extraction(name+'cf52_'+S[i]+'.csv'), ss=0, ee=-1))


##### Part II: Data merge #####
print('***** Data merge! *****')
Data_LL_L1 = Merge(Data_LL_L1_list)
Data_LL_L2 = Merge(Data_LL_L2_list)
Data_SS_S1 = Merge(Data_SS_S1_list)
Data_SS_S2 = Merge(Data_SS_S2_list)
Data_LS_L = Merge(Data_LS_L_list)
Data_LS_S = Merge(Data_LS_S_list)

Data_LGe = Merge(Data_LGe_list)
Data_SGe = Merge(Data_SGe_list)
Data_SLL_S = Merge(Data_SLL_S_list)
Data_SLL_L1 = Merge(Data_SLL_L1_list)
Data_SLL_L2 = Merge(Data_SLL_L2_list)
Data_SSL_L = Merge(Data_SSL_L_list)
Data_SSL_S1 = Merge(Data_SSL_S1_list)
Data_SSL_S2 = Merge(Data_SSL_S2_list)
Data_SSS_S1 = Merge(Data_SSS_S1_list)
Data_SSS_S2 = Merge(Data_SSS_S2_list)
Data_SSS_S3 = Merge(Data_SSS_S3_list)


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
Data_LS_L = Fa(Data_LS_L, m, g, C_00, C_10, C_01, C_20, C_11)

Data_LGe = Fa(Data_LGe, m, g, C_00, C_10, C_01, C_20, C_11)
Data_SLL_L1 = Fa(Data_SLL_L1, m, g, C_00, C_10, C_01, C_20, C_11)
Data_SLL_L2 = Fa(Data_SLL_L2, m, g, C_00, C_10, C_01, C_20, C_11)
Data_SSL_L = Fa(Data_SSL_L, m, g, C_00, C_10, C_01, C_20, C_11)

# small CF
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

Data_SGe = Fa(Data_SGe, m, g, C_00, C_10, C_01, C_20, C_11)
Data_SLL_S = Fa(Data_SLL_S, m, g, C_00, C_10, C_01, C_20, C_11)
Data_SSL_S1 = Fa(Data_SSL_S1, m, g, C_00, C_10, C_01, C_20, C_11)
Data_SSL_S2 = Fa(Data_SSL_S2, m, g, C_00, C_10, C_01, C_20, C_11)
Data_SSS_S1 = Fa(Data_SSS_S1, m, g, C_00, C_10, C_01, C_20, C_11)
Data_SSS_S2 = Fa(Data_SSS_S2, m, g, C_00, C_10, C_01, C_20, C_11)
Data_SSS_S3 = Fa(Data_SSS_S3, m, g, C_00, C_10, C_01, C_20, C_11)


##### Part IV: Generate input-output pair #####
print('***** Input-output pair generation! *****')
data_input_Ge2L, data_output_Ge2L = get_data(D1=Data_LGe, D2=None, s=encoder['Ge2L'], typ=fa_type)
data_input_Ge2S, data_output_Ge2S = get_data(D1=Data_SGe, D2=None, s=encoder['Ge2S'], typ=fa_type)
print('Ge2L:', data_input_Ge2L.shape, data_output_Ge2L.shape)
print('Ge2S:', data_input_Ge2S.shape, data_output_Ge2S.shape)

data_input_L2L_a, data_output_L2L_a = get_data(D1=Data_LL_L1, D2=Data_LL_L2, s=encoder['L2L'], typ=fa_type)
data_input_L2L_b, data_output_L2L_b = get_data(D1=Data_LL_L2, D2=Data_LL_L1, s=encoder['L2L'], typ=fa_type)
data_input_L2L = np.vstack((data_input_L2L_a, data_input_L2L_b))
data_output_L2L = np.vstack((data_output_L2L_a, data_output_L2L_b))
print('L2L:', data_input_L2L.shape, data_output_L2L.shape)

data_input_S2S_a, data_output_S2S_a = get_data(D1=Data_SS_S1, D2=Data_SS_S2, s=encoder['S2S'], typ=fa_type)
data_input_S2S_b, data_output_S2S_b = get_data(D1=Data_SS_S2, D2=Data_SS_S1, s=encoder['S2S'], typ=fa_type)
data_input_S2S = np.vstack((data_input_S2S_a, data_input_S2S_b))
data_output_S2S = np.vstack((data_output_S2S_a, data_output_S2S_b))
print('S2S:', data_input_S2S.shape, data_output_S2S.shape)

data_input_L2S, data_output_L2S = get_data(D1=Data_LS_S, D2=Data_LS_L, s=encoder['L2S'], typ=fa_type)
data_input_S2L, data_output_S2L = get_data(D1=Data_LS_L, D2=Data_LS_S, s=encoder['S2L'], typ=fa_type)
print('L2S:', data_input_L2S.shape, data_output_L2S.shape)
print('S2L:', data_input_S2L.shape, data_output_S2L.shape)

data_input_SS2L, data_output_SS2L = get_data(D1=Data_SSL_L, D2=Data_SSL_S1, D3=Data_SSL_S2, s=encoder['SS2L'], typ=fa_type)
print('SS2L:', data_input_SS2L.shape, data_output_SS2L.shape)

data_input_SL2L_a, data_output_SL2L_a = get_data(D1=Data_SLL_L1, D2=Data_SLL_S, D3=Data_SLL_L2, s=encoder['SL2L'], typ=fa_type)
data_input_SL2L_b, data_output_SL2L_b = get_data(D1=Data_SLL_L2, D2=Data_SLL_S, D3=Data_SLL_L1, s=encoder['SL2L'], typ=fa_type)
data_input_SL2L = np.vstack((data_input_SL2L_a, data_input_SL2L_b))
data_output_SL2L = np.vstack((data_output_SL2L_a, data_output_SL2L_b))
print('SL2L:', data_input_SL2L.shape, data_output_SL2L.shape)

data_input_LL2S, data_output_LL2S = get_data(D1=Data_SLL_S, D2=Data_SLL_L1, D3=Data_SLL_L2, s=encoder['LL2S'], typ=fa_type)
print('LL2S:', data_input_LL2S.shape, data_output_LL2S.shape)

data_input_SL2S_a, data_output_SL2S_a = get_data(D1=Data_SSL_S1, D2=Data_SSL_S2, D3=Data_SSL_L, s=encoder['SL2S'], typ=fa_type)
data_input_SL2S_b, data_output_SL2S_b = get_data(D1=Data_SSL_S2, D2=Data_SSL_S1, D3=Data_SSL_L, s=encoder['SL2S'], typ=fa_type)
data_input_SL2S = np.vstack((data_input_SL2S_a, data_input_SL2S_b))
data_output_SL2S = np.vstack((data_output_SL2S_a, data_output_SL2S_b))
print('SL2S:', data_input_SL2S.shape, data_output_SL2S.shape)

data_input_SS2S_a, data_output_SS2S_a = get_data(D1=Data_SSS_S1, D2=Data_SSS_S2, D3=Data_SSS_S3, s=encoder['SS2S'], typ=fa_type)
data_input_SS2S_b, data_output_SS2S_b = get_data(D1=Data_SSS_S2, D2=Data_SSS_S1, D3=Data_SSS_S3, s=encoder['SS2S'], typ=fa_type)
data_input_SS2S_c, data_output_SS2S_c = get_data(D1=Data_SSS_S3, D2=Data_SSS_S1, D3=Data_SSS_S2, s=encoder['SS2S'], typ=fa_type)
data_input_SS2S = np.vstack((data_input_SS2S_a, data_input_SS2S_b, data_input_SS2S_c))
data_output_SS2S = np.vstack((data_output_SS2S_a, data_output_SS2S_b, data_output_SS2S_c))
print('SS2S:', data_input_SS2S.shape, data_output_SS2S.shape)

if False:
    # visualization of data distribution
    hist(pp, data_input_Ge2L, data_output_Ge2L, 'Ge2L', rasterized)
    hist(pp, data_input_Ge2S, data_output_Ge2S, 'Ge2S', rasterized)
    hist(pp, data_input_L2L, data_output_L2L, 'L2L', rasterized)
    hist(pp, data_input_S2S, data_output_S2S, 'S2S', rasterized)
    hist(pp, data_input_L2S, data_output_L2S, 'L2S', rasterized)
    hist(pp, data_input_S2L, data_output_S2L, 'S2L', rasterized)
    hist(pp, data_input_SS2L, data_output_SS2L, 'SS2L', rasterized)
    hist(pp, data_input_SL2L, data_output_SL2L, 'SL2L', rasterized)
    hist(pp, data_input_LL2S, data_output_LL2S, 'LL2S', rasterized)
    hist(pp, data_input_SL2S, data_output_SL2S, 'SL2S', rasterized)
    hist(pp, data_input_SS2S, data_output_SS2S, 'SS2S', rasterized)

# generate torch trainset and trainloader
trainset_Ge2L, trainloader_Ge2L, valset_Ge2L, val_input_Ge2L, val_output_Ge2L = set_generate(data_input_Ge2L, data_output_Ge2L, 'Ge2L', device, batch_size)
trainset_Ge2S, trainloader_Ge2S, valset_Ge2S, val_input_Ge2S, val_output_Ge2S = set_generate(data_input_Ge2S, data_output_Ge2S, 'Ge2S', device, batch_size)
trainset_L2L, trainloader_L2L, valset_L2L, val_input_L2L, val_output_L2L = set_generate(data_input_L2L, data_output_L2L, 'L2L', device, batch_size)
trainset_S2S, trainloader_S2S, valset_S2S, val_input_S2S, val_output_S2S = set_generate(data_input_S2S, data_output_S2S, 'S2S', device, batch_size)
trainset_L2S, trainloader_L2S, valset_L2S, val_input_L2S, val_output_L2S = set_generate(data_input_L2S, data_output_L2S, 'L2S', device, batch_size)
trainset_S2L, trainloader_S2L, valset_S2L, val_input_S2L, val_output_S2L = set_generate(data_input_S2L, data_output_S2L, 'S2L', device, batch_size)
trainset_SS2L, trainloader_SS2L, valset_SS2L, val_input_SS2L, val_output_SS2L = set_generate(data_input_SS2L, data_output_SS2L, 'SS2L', device, batch_size)
trainset_SL2L, trainloader_SL2L, valset_SL2L, val_input_SL2L, val_output_SL2L = set_generate(data_input_SL2L, data_output_SL2L, 'SL2L', device, batch_size)
trainset_LL2S, trainloader_LL2S, valset_LL2S, val_input_LL2S, val_output_LL2S = set_generate(data_input_LL2S, data_output_LL2S, 'LL2S', device, batch_size)
trainset_SL2S, trainloader_SL2S, valset_SL2S, val_input_SL2S, val_output_SL2S = set_generate(data_input_SL2S, data_output_SL2S, 'SL2S', device, batch_size)
trainset_SS2S, trainloader_SS2S, valset_SS2S, val_input_SS2S, val_output_SS2S = set_generate(data_input_SS2S, data_output_SS2S, 'SS2S', device, batch_size)


##### Part V: Training #####
print('***** Training! *****')
# ground effect doesn't consider x and y
phi_G_net = phi_Net(inputdim=4,hiddendim=hidden_dim).to(device, dtype=torch.float32)
phi_L_net = phi_Net(inputdim=6,hiddendim=hidden_dim).to(device, dtype=torch.float32)
phi_S_net = phi_Net(inputdim=6,hiddendim=hidden_dim).to(device, dtype=torch.float32)
rho_L_net = rho_Net(hiddendim=hidden_dim).to(device, dtype=torch.float32)
rho_S_net = rho_Net(hiddendim=hidden_dim).to(device, dtype=torch.float32)

criterion = nn.MSELoss()
optimizer_phi_G = optim.Adam(phi_G_net.parameters(), lr=1e-3)
optimizer_phi_L = optim.Adam(phi_L_net.parameters(), lr=1e-3)
optimizer_rho_L = optim.Adam(rho_L_net.parameters(), lr=1e-3)
optimizer_phi_S = optim.Adam(phi_S_net.parameters(), lr=1e-3)
optimizer_rho_S = optim.Adam(rho_S_net.parameters(), lr=1e-3)

def set_loss(set, criterion, rho_net, phi_1_net, phi_2_net=None, GE=False):
    with torch.no_grad():
        inputs = set[:]['input'] 
        label = set[:]['output']
        if phi_2_net is None:
            if GE:
                loss = criterion(rho_net(phi_1_net(inputs[:, 2:6])), label)
            else:
                loss = criterion(rho_net(phi_1_net(inputs[:, :6])), label)
        else:
            loss = criterion(rho_net(phi_1_net(inputs[:, :6]) + phi_2_net(inputs[:, 6:12])), label)
    return loss.item()

# Loss before training
# 0:Ge2L 1:Ge2S 2:L2L  3:S2S  4:L2S 5:S2L
# 6:SS2L 7:SL2L 8:LL2S 9:SL2S 10:SS2S
print('Ge2L loss b4 training', set_loss(trainset_Ge2L, criterion, rho_L_net, phi_G_net, GE=True))
print('Ge2S loss b4 training', set_loss(trainset_Ge2S, criterion, rho_S_net, phi_G_net, GE=True))
print('L2L loss b4 training', set_loss(trainset_L2L, criterion, rho_L_net, phi_L_net))
print('S2S loss b4 training', set_loss(trainset_S2S, criterion, rho_S_net, phi_S_net))
print('L2S loss b4 training', set_loss(trainset_L2S, criterion, rho_S_net, phi_L_net))
print('S2L loss b4 training', set_loss(trainset_S2L, criterion, rho_L_net, phi_S_net))
print('SS2L loss b4 training', set_loss(trainset_SS2L, criterion, rho_L_net, phi_S_net, phi_2_net=phi_S_net))
print('SL2L loss b4 training', set_loss(trainset_SL2L, criterion, rho_L_net, phi_S_net, phi_2_net=phi_L_net))
print('LL2S loss b4 training', set_loss(trainset_LL2S, criterion, rho_S_net, phi_L_net, phi_2_net=phi_L_net))
print('SL2S loss b4 training', set_loss(trainset_SL2S, criterion, rho_S_net, phi_S_net, phi_2_net=phi_L_net))
print('SS2S loss b4 training', set_loss(trainset_SS2S, criterion, rho_S_net, phi_S_net, phi_2_net=phi_S_net))

# training
Loss_sn = []
# mix all the data
mixed = []
Count = defaultdict(int)
for data in trainloader_Ge2L:
    Count['Ge2L'] += 1
    mixed.append(data)
for data in trainloader_Ge2S:
    Count['Ge2S'] += 1
    mixed.append(data)
for data in trainloader_L2L:
    Count['L2L'] += 1
    mixed.append(data)
for data in trainloader_S2S:
    Count['S2S'] += 1
    mixed.append(data)
for data in trainloader_L2S:
    Count['L2S'] += 1
    mixed.append(data)
for data in trainloader_S2L:
    Count['S2L'] += 1
    mixed.append(data)
for data in trainloader_SS2L:
    Count['SS2L'] += 1
    mixed.append(data)
for data in trainloader_SL2L:
    Count['SL2L'] += 1
    mixed.append(data)
for data in trainloader_LL2S:
    Count['LL2S'] += 1
    mixed.append(data)
for data in trainloader_SL2S:
    Count['SL2S'] += 1
    mixed.append(data)
for data in trainloader_SS2S:
    Count['SS2S'] += 1
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
        optimizer_phi_S.zero_grad()
        optimizer_rho_S.zero_grad()

        # forward + backward + optimize
        if datatype == 'Ge2L':
            outputs = rho_L_net(phi_G_net(inputs[:, 2:6]))
        elif datatype == 'Ge2S':
            outputs = rho_S_net(phi_G_net(inputs[:, 2:6]))
        elif datatype == 'L2L':
            outputs = rho_L_net(phi_L_net(inputs[:, :6]))
        elif datatype == 'S2S':
            outputs = rho_S_net(phi_S_net(inputs[:, :6]))
        elif datatype == 'L2S':
            outputs = rho_S_net(phi_L_net(inputs[:, :6]))
        elif datatype == 'S2L':
            outputs = rho_L_net(phi_S_net(inputs[:, :6]))
        elif datatype == 'SS2L':
            outputs = rho_L_net(phi_S_net(inputs[:, :6]) + phi_S_net(inputs[:, 6:12]))
        elif datatype == 'SL2L':
            outputs = rho_L_net(phi_S_net(inputs[:, :6]) + phi_L_net(inputs[:, 6:12]))
        elif datatype == 'LL2S':
            outputs = rho_S_net(phi_L_net(inputs[:, :6]) + phi_L_net(inputs[:, 6:12]))
        elif datatype == 'SL2S':
            outputs = rho_S_net(phi_S_net(inputs[:, :6]) + phi_L_net(inputs[:, 6:12]))
        elif datatype == 'SS2S':
            outputs = rho_S_net(phi_S_net(inputs[:, :6]) + phi_S_net(inputs[:, 6:12]))
        else:
            print('wrong class', datatype)
        

        count[datatype] -= 1

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_phi_G.step()
        optimizer_phi_L.step()
        optimizer_phi_S.step()
        optimizer_rho_L.step()
        optimizer_rho_S.step()
        
        # Lip
        Lip(phi_G_net, lip)
        Lip(phi_L_net, lip)
        Lip(phi_S_net, lip)
        Lip(rho_L_net, lip)
        Lip(rho_S_net, lip)
                    
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
print('Ge2L loss after training', set_loss(trainset_Ge2L, criterion, rho_L_net, phi_G_net, GE=True))
print('Ge2S loss after training', set_loss(trainset_Ge2S, criterion, rho_S_net, phi_G_net, GE=True))
print('L2L loss after training', set_loss(trainset_L2L, criterion, rho_L_net, phi_L_net))
print('S2S loss after training', set_loss(trainset_S2S, criterion, rho_S_net, phi_S_net))
print('L2S loss after training', set_loss(trainset_L2S, criterion, rho_S_net, phi_L_net))
print('S2L loss after training', set_loss(trainset_S2L, criterion, rho_L_net, phi_S_net))
print('SS2L loss after training', set_loss(trainset_SS2L, criterion, rho_L_net, phi_S_net, phi_2_net=phi_S_net))
print('SL2L loss after training', set_loss(trainset_SL2L, criterion, rho_L_net, phi_S_net, phi_2_net=phi_L_net))
print('LL2S loss after training', set_loss(trainset_LL2S, criterion, rho_S_net, phi_L_net, phi_2_net=phi_L_net))
print('SL2S loss after training', set_loss(trainset_SL2S, criterion, rho_S_net, phi_S_net, phi_2_net=phi_L_net))
print('SS2S loss after training', set_loss(trainset_SS2S, criterion, rho_S_net, phi_S_net, phi_2_net=phi_S_net))

Error = []
Error.append(set_loss(valset_Ge2L, criterion, rho_L_net, phi_G_net, GE=True))
Error.append(set_loss(valset_Ge2S, criterion, rho_S_net, phi_G_net, GE=True))
Error.append(set_loss(valset_L2L, criterion, rho_L_net, phi_L_net))
Error.append(set_loss(valset_S2S, criterion, rho_S_net, phi_S_net))
Error.append(set_loss(valset_L2S, criterion, rho_S_net, phi_L_net))
Error.append(set_loss(valset_S2L, criterion, rho_L_net, phi_S_net))
Error.append(set_loss(valset_SS2L, criterion, rho_L_net, phi_S_net, phi_2_net=phi_S_net))
Error.append(set_loss(valset_SL2L, criterion, rho_L_net, phi_S_net, phi_2_net=phi_L_net))
Error.append(set_loss(valset_LL2S, criterion, rho_S_net, phi_L_net, phi_2_net=phi_L_net))
Error.append(set_loss(valset_SL2S, criterion, rho_S_net, phi_S_net, phi_2_net=phi_L_net))
Error.append(set_loss(valset_SS2S, criterion, rho_S_net, phi_S_net, phi_2_net=phi_S_net))
Error = np.array(Error)
weight = np.array([len(valset_Ge2L), len(valset_Ge2S), len(valset_L2L), len(valset_S2S), len(valset_L2S), len(valset_S2L), len(valset_SS2L), len(valset_SL2L), len(valset_LL2S), len(valset_SL2S), len(valset_SS2S)])
mean_error = np.sum(Error*weight) / np.sum(weight)
print('Validation error:')
print(Error)
print('mean:')
print(mean_error)


phi_G_net.cpu()
phi_L_net.cpu()
phi_S_net.cpu()
rho_L_net.cpu()
rho_S_net.cpu()

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

# Val of NNs
# 0:Ge2L 1:Ge2S 2:L2L  3:S2S  4:L2S 5:S2L
# 6:SS2L 7:SL2L 8:LL2S 9:SL2S 10:SS2S
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
                outputs = rho_L_net(phi_L_net(inputs[:, :6]))
            elif temp == encoder['S2S']:
                outputs = rho_S_net(phi_S_net(inputs[:, :6]))
            elif temp == encoder['L2S']:
                outputs = rho_S_net(phi_L_net(inputs[:, :6]))
            elif temp == encoder['S2L']:
                outputs = rho_L_net(phi_S_net(inputs[:, :6]))
            elif temp == encoder['SS2L']:
                outputs = rho_L_net(phi_S_net(inputs[:, :6]) + phi_S_net(inputs[:, 6:12]))
            elif temp == encoder['SL2L']:
                outputs = rho_L_net(phi_S_net(inputs[:, :6]) + phi_L_net(inputs[:, 6:12]))
            elif temp == encoder['LL2S']:
                outputs = rho_S_net(phi_L_net(inputs[:, :6]) + phi_L_net(inputs[:, 6:12]))
            elif temp == encoder['SL2S']:
                outputs = rho_S_net(phi_S_net(inputs[:, :6]) + phi_L_net(inputs[:, 6:12]))
            elif temp == encoder['SS2S']:
                outputs = rho_S_net(phi_S_net(inputs[:, :6]) + phi_S_net(inputs[:, 6:12]))
            else:
                print('wrong class', temp)
            Fa[i] = outputs[0, 0].item()
    return Fa

# Val of NNs
def validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, data_input, data_output, ss, ee, name):
    Fa_pred = Fa_prediction(data_input, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net)
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1, rasterized=rasterized)
    plt.plot(data_input[:, :3])
    plt.legend(['dx', 'dy', 'dz'])
    plt.grid()
    plt.title('Validation: '+name)
    plt.subplot(2, 1, 2, rasterized=rasterized)
    plt.plot(data_output[:, 2])
    plt.hlines(y=0, xmin=0, xmax=ee-ss, colors='r')
    plt.plot(Fa_pred)
    plt.legend(['fa_gt', 'fa_pred'])
    plt.grid()
    pp.savefig()
    plt.close()
    #plt.show()

validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_Ge2L, val_output_Ge2L, ss=0, ee=-1, name='Ge2L')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_Ge2S, val_output_Ge2S, ss=0, ee=-1, name='Ge2S')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_L2L, val_output_L2L, ss=0, ee=-1, name='L2L')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_S2S, val_output_S2S, ss=0, ee=-1, name='S2S')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_L2S, val_output_L2S, ss=0, ee=-1, name='L2S')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_S2L, val_output_S2L, ss=0, ee=-1, name='S2L')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_SS2L, val_output_SS2L, ss=0, ee=-1, name='SS2L')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_SL2L, val_output_SL2L, ss=0, ee=-1, name='SL2L')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_LL2S, val_output_LL2S, ss=0, ee=-1, name='LL2L')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_SL2S, val_output_SL2S, ss=0, ee=-1, name='SL2S')
validation(pp, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, val_input_SS2S, val_output_SS2S, ss=0, ee=-1, name='SS2S')

pp.close()
print('***** Output PDF saved! *****')