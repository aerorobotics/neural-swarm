import numpy as np
import pandas as pd
import random
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import Dataset, DataLoader

# 0:Ge2L 1:Ge2S 2:L2L  3:S2S  4:L2S 5:S2L
# 6:SS2L 7:SL2L 8:LL2S 9:SL2S 10:SS2S
encoder = {'Ge2L':0, 'Ge2S':1, 'L2L':2, 'S2S':3, 'L2S':4, 'S2L':5, \
           'SS2L':6, 'SL2L':7, 'LL2S':8, 'SL2S':9, 'SS2S':10}

# Convert quaternion to rotation matrix
def rotation_matrix(quat):
    rot_mat = np.ones([3,3])
    a = quat[0]**2
    b = quat[1]**2
    c = quat[2]**2
    d = quat[3]**2
    e = quat[0]*quat[1]
    f = quat[0]*quat[2]
    g = quat[0]*quat[3]
    h = quat[1]*quat[2]
    i = quat[1]*quat[3]
    j = quat[2]*quat[3]
    rot_mat[0,0] = a - b - c + d
    rot_mat[0,1] = 2 * (e - j)
    rot_mat[0,2] = 2 * (f + i)
    rot_mat[1,0] = 2 * (e + j)
    rot_mat[1,1] = -a + b - c + d
    rot_mat[1,2] = 2 * (h - g)
    rot_mat[2,0] = 2 * (f - i)
    rot_mat[2,1] = 2 * (h + g)
    rot_mat[2,2] = -a - b + c + d
     
    return rot_mat

# Convert quaternion to Euler angle
def qua2euler(qua):
    euler = np.zeros(3)
    q0 = qua[3]
    q1 = qua[0]
    q2 = qua[1]
    q3 = qua[2]
    euler[0] = np.degrees(np.arctan2(2*(q0*q1+q2*q3), 1-2*(q1**2+q2**2)))
    euler[1] = np.degrees(np.arcsin(2*(q0*q2-q3*q1)))
    euler[2] = np.degrees(np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2)))
    return euler

# Extract data as a dictionary from .csv file
def data_extraction(filename):    
    raw = pd.read_csv(filename)
    secs = np.copy(raw.iloc[:, 0])
    bias = np.copy(secs[0])
    time = (secs - bias) / 1e6
    pos = np.copy(raw.iloc[:, 1:4])
    vel = np.copy(raw.iloc[:, 4:7])
    acc = np.copy(raw.iloc[:, 7:10])
    qua = np.copy(raw.iloc[:, 10:14])
    thr = np.copy(raw.iloc[:, 14:18])
    pwm = np.copy(raw.iloc[:, 18:22])
    vol = np.copy(raw.iloc[:, 22]) / 4.2
    thr_des = np.copy(raw.iloc[:, 23])
    tau_u = np.copy(raw.iloc[:, 24:27])
    omega = np.copy(raw.iloc[:, 27:])
    
    Data = {'time': time, 'pos': pos, 'vel': vel, 'acc': acc, 'qua': qua, 'thr': thr, \
            'pwm': pwm, 'vol': vol, 'thr_des': thr_des, 'tau_u': tau_u, 'omega': omega}
    
    return Data

# cubic (or linear) interpolation
def cubic(x, y, xnew, kind='linear'):
    f = interp1d(x, y, kind=kind)
    return f(xnew)

# data interpolation
def interpolation_cubic(t0, t1, Data, ss, ee):
    time = np.linspace(t0, t1, int(100*(t1-t0)+1))
    pos = np.zeros((time.shape[0], 3))
    qua = np.zeros((time.shape[0], 4))
    pwm = np.zeros((time.shape[0], 4))
    vel = np.zeros((time.shape[0], 3))
    vel_num = np.zeros((time.shape[0], 3))
    acc_num = np.zeros((time.shape[0], 3))
    vol = np.zeros((time.shape[0]))
    acc_imu = np.zeros((time.shape[0], 3))
    tau_u = np.zeros((time.shape[0], 3))
    omega = np.zeros((time.shape[0], 3))
    omega_dot = np.zeros((time.shape[0], 3))
    euler = np.zeros((time.shape[0], 3))
    acc_filter = np.zeros((time.shape[0], 3))
    acc_smooth = np.zeros((time.shape[0], 3))
    
    x = Data['time'][ss:ee]
    for i in range(3):
        pos[:, i] = cubic(x, Data['pos'][ss:ee, i], time)
        vel[:, i] = cubic(x, Data['vel'][ss:ee, i], time)
        acc_imu[:, i] = cubic(x, Data['acc'][ss:ee, i], time)
        tau_u[:, i] = cubic(x, Data['tau_u'][ss:ee, i], time)
        omega[:, i] = cubic(x, Data['omega'][ss:ee, i], time)
 
    for i in range(3):
        acc_num[2:-2,i] = (-vel[4:,i] + 8 * vel[3:-1,i] - 8 * vel[1:-3,i] + vel[:-4,i]) / 12 * 100
        vel_num[2:-2,i] = (-pos[4:,i] + 8 * pos[3:-1,i] - 8 * pos[1:-3,i] + pos[:-4,i]) / 12 * 100        
        omega_dot[2:-2,i] = (-omega[4:,i] + 8 * omega[3:-1,i] - 8 * omega[1:-3,i] + omega[:-4,i]) / 12 * 100
    
    for i in range(4):
        qua[:, i] = cubic(x, Data['qua'][ss:ee, i], time)
        pwm[:, i] = cubic(x, Data['pwm'][ss:ee, i], time)
    vol[:] = cubic(x, Data['vol'][ss:ee], time)
    
    for j in range(time.shape[0]):
        euler[j, :] = qua2euler(qua[j, :])
    
    # Filter on acc
    b, a = signal.butter(1, 0.1)
    for i in range(3):
        acc_filter[:, i] = signal.filtfilt(b, a, acc_num[:, i])
    
    # Moving average smoothing
    n = 5
    l = int((n-1) / 2)
    for i in range(3):
        for j in range(n):
            if j == n-1:
                temp = acc_num[j:, i]
            else:
                temp = acc_num[j:-(n-1-j), i]
            acc_smooth[l:-l, i] = acc_smooth[l:-l, i] + temp
        acc_smooth[l:-l, i] = acc_smooth[l:-l, i] / n
        
    Data_int = {'time': time, 'pos': pos, 'vel': vel, 'acc_imu': acc_imu, 'vel_num': vel_num, \
                'qua': qua, 'pwm': pwm, 'vol': vol, 'acc_num': acc_num, 'euler': euler, \
               'tau_u': tau_u, 'omega': omega, 'omega_dot': omega_dot, 'acc_filter': acc_filter, 'acc_smooth': acc_smooth}
    return Data_int

# Data merge
def merge(Data1, Data2):
    Data = {'time': np.concatenate((Data1['time'], Data2['time'] + Data1['time'][-1] + 0.01))}
    for topic in Data1:
        if topic != 'time':
            Data[topic] = np.concatenate((Data1[topic], Data2[topic]))
    return Data
def Merge(Data_list):
    if len(Data_list) == 1:
        return Data_list[0]
    Data = merge(Data_list[0], Data_list[1])
    if len(Data_list) > 2:
        for i in range(2, len(Data_list)):
            Data = merge(Data, Data_list[i])
    return Data

# Compute Fa
def Fa(Data, m, g, p_00, p_10, p_01, p_20, p_11):
    R = np.zeros([Data['time'].shape[0], 3, 3])
    for i in range(Data['time'].shape[0]):
        R[i, :, :] = rotation_matrix(Data['qua'][i, :])
        
    force_pwm_1 = p_00 + p_10 * Data['pwm'][:, 0] + p_01 * Data['vol'] + p_20 * Data['pwm'][:, 0]**2 + p_11 * Data['vol'] * Data['pwm'][:, 0]
    force_pwm_2 = p_00 + p_10 * Data['pwm'][:, 1] + p_01 * Data['vol'] + p_20 * Data['pwm'][:, 1]**2 + p_11 * Data['vol'] * Data['pwm'][:, 1]
    force_pwm_3 = p_00 + p_10 * Data['pwm'][:, 2] + p_01 * Data['vol'] + p_20 * Data['pwm'][:, 2]**2 + p_11 * Data['vol'] * Data['pwm'][:, 2]
    force_pwm_4 = p_00 + p_10 * Data['pwm'][:, 3] + p_01 * Data['vol'] + p_20 * Data['pwm'][:, 3]**2 + p_11 * Data['vol'] * Data['pwm'][:, 3]
    thrust_pwm = force_pwm_1 + force_pwm_2 + force_pwm_3 + force_pwm_4 # gram
    # consider delay
    thrust_pwm_delay = np.zeros(len(thrust_pwm))
    thrust_pwm_delay[0] = thrust_pwm[0]
    for i in range(len(thrust_pwm)-1):
        thrust_pwm_delay[i+1] = (1-0.16)*thrust_pwm_delay[i] + 0.16*thrust_pwm[i] 

    Fa = np.zeros([Data['time'].shape[0], 3])
    Fa_delay = np.zeros([Data['time'].shape[0], 3])
    Fa_num = np.zeros([Data['time'].shape[0], 3])
    Fa_filter = np.zeros([Data['time'].shape[0], 3])
    Fa_smooth = np.zeros([Data['time'].shape[0], 3])
    force_world = np.zeros([Data['time'].shape[0], 3])
    for i in range(Data['time'].shape[0]):
        Fa[i, :] = m * Data['acc_imu'][i, :] / 1000 - thrust_pwm[i] / 1000 * g * R[i, :, 2] # Newton
        Fa_delay[i, :] = m * Data['acc_imu'][i, :] / 1000 - thrust_pwm_delay[i] / 1000 * g * R[i, :, 2] # Newton
        Fa_num[i, :] = m * np.array([0, 0, g]) / 1000 + m * Data['acc_num'][i, :] / 1000 - thrust_pwm[i] / 1000 * g * R[i, :, 2] # Newton
        Fa_filter[i, :] = m * np.array([0, 0, g]) / 1000 + m * Data['acc_filter'][i, :] / 1000 - thrust_pwm[i] / 1000 * g * R[i, :, 2] # Newton
        Fa_smooth[i, :] = m * np.array([0, 0, g]) / 1000 + m * Data['acc_smooth'][i, :] / 1000 - thrust_pwm[i] / 1000 * g * R[i, :, 2] # Newton
        force_world[i, :] = thrust_pwm[i] / 1000 * g * R[i, :, 2] # Newton
    
    Data['fa_imu'] = Fa
    Data['fa_delay'] = Fa_delay
    Data['fa_num'] = Fa_num
    Data['fa_filter'] = Fa_filter
    Data['fa_smooth'] = Fa_smooth
    
    return Data

# Get numpy data input and output pair
# s can be used for data type
def get_data(D1, D2, D3=None, s=0, typ='fa_imu'):
    g = 9.81
    L = D1['time'].shape[0]
    data_input = np.zeros([L, 13], dtype=np.float32)
    data_output = np.zeros([L, 3], dtype=np.float32)
    if s == encoder['Ge2L'] or s == encoder['Ge2S']:
        # ground effect
        data_input[:, :3] = 0 - D1['pos']
        data_input[:, 3:6] = 0 - D1['vel']
    else:
        data_input[:, :3] = D2['pos'] - D1['pos']
        data_input[:, 3:6] = D2['vel'] - D1['vel']
    if D3 is not None:
        data_input[:, 6:9] = D3['pos'] - D1['pos']
        data_input[:, 9:12] = D3['vel'] - D1['vel']
    data_input[:, -1] = s
    data_output[:, :] = D1[typ] / g * 1000 # Newton -> gram
    # print(data_input.shape, data_output.shape)
    
    return data_input, data_output

# Histogram visualization of numpy data input and output pair
def hist(pp, data_input, data_output, name, rasterized):
    plt.figure(figsize=(12,12))
    plt.subplot(5, 4, 1, rasterized=rasterized)
    plt.hist(data_input[:, 0], 50, density=True)
    plt.title(name+': x1')
    plt.subplot(5, 4, 2, rasterized=rasterized)
    plt.hist(data_input[:, 1], 50, density=True)
    plt.title('y1')
    plt.subplot(5, 4, 3, rasterized=rasterized)
    plt.hist(data_input[:, 2], 50, density=True)
    plt.title('z1')
    plt.subplot(5, 4, 4, rasterized=rasterized)
    plt.scatter(data_input[:, 1], data_input[:, 2], s=0.1)
    plt.title('y1-z1')
    plt.xlabel('y1')
    plt.ylabel('z1')
    
    plt.subplot(5, 4, 5, rasterized=rasterized)
    plt.hist(data_input[:, 3], 50, density=True)
    plt.title('vx1')
    plt.subplot(5, 4, 6, rasterized=rasterized)
    plt.hist(data_input[:, 4], 50, density=True)
    plt.title('vy1')
    plt.subplot(5, 4, 7, rasterized=rasterized)
    plt.hist(data_input[:, 5], 50, density=True)
    plt.title('vz1')
    plt.subplot(5, 4, 8, rasterized=rasterized)
    plt.scatter(data_input[:, 4], data_input[:, 5], s=0.1)
    plt.title('vy1-vz1')
    plt.xlabel('vy1')
    plt.ylabel('vz1')
    
    plt.subplot(5, 4, 9, rasterized=rasterized)
    plt.hist(data_input[:, 6], 50, density=True)
    plt.title(name+': x2')
    plt.subplot(5, 4, 10, rasterized=rasterized)
    plt.hist(data_input[:, 7], 50, density=True)
    plt.title('y2')
    plt.subplot(5, 4, 11, rasterized=rasterized)
    plt.hist(data_input[:, 8], 50, density=True)
    plt.title('z2')
    plt.subplot(5, 4, 12, rasterized=rasterized)
    plt.scatter(data_input[:, 7], data_input[:, 8], s=0.1)
    plt.title('y2-z2')
    plt.xlabel('y2')
    plt.ylabel('z2')
    
    plt.subplot(5, 4, 13, rasterized=rasterized)
    plt.hist(data_input[:, 9], 50, density=True)
    plt.title('vx2')
    plt.subplot(5, 4, 14, rasterized=rasterized)
    plt.hist(data_input[:, 10], 50, density=True)
    plt.title('vy2')
    plt.subplot(5, 4, 15, rasterized=rasterized)
    plt.hist(data_input[:, 11], 50, density=True)
    plt.title('vz2')
    plt.subplot(5, 4, 16, rasterized=rasterized)
    plt.scatter(data_input[:, 10], data_input[:, 11], s=0.1)
    plt.title('vy2-vz2')
    plt.xlabel('vy2')
    plt.ylabel('vz2')

    plt.subplot(5, 4, 17, rasterized=rasterized)
    plt.hist(data_output[:, 0], 50, density=True)
    plt.title('fa_x')
    plt.subplot(5, 4, 18, rasterized=rasterized)
    plt.hist(data_output[:, 1], 50, density=True)
    plt.title('fa_y')
    plt.subplot(5, 4, 19, rasterized=rasterized)
    plt.hist(data_output[:, 2], 50, density=True)
    plt.title('fa_z')
    plt.subplot(5, 4, 20, rasterized=rasterized)
    plt.plot(data_input[:, -1])
    plt.title('s')
    plt.tight_layout()
    pp.savefig()
    plt.close()
    #plt.show()

# Dataset in torch
class MyDataset(Dataset):

    def __init__(self, inputs, outputs, type):
        self.inputs = inputs
        self.outputs = outputs
        self.type = type

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        Input = self.inputs[idx,]
        output = self.outputs[idx,]
        sample = {'input': Input, 'output': output, 'type': self.type}

        return sample


def split(data, type, val_num=10, num=50):
    # 50 pieces together, and 10 pieces for validation
    L = len(data)
    temp = np.split(data[:L-np.mod(L,num),:], num, axis=0)
    code = np.sum([ord(s) for s in type])
    random.seed(code) # fix the seed for each scenario
    val_index = random.sample([i for i in range(num)], val_num)
    val_index_sorted = sorted(val_index)
    train_index_sorted = sorted([i for i in range(num) if i not in val_index])
    val_data = np.concatenate([temp[i] for i in val_index_sorted], axis=0)
    train_data = np.concatenate([temp[i] for i in train_index_sorted], axis=0)
    return val_data, train_data

# Input numpy data_input (7x1) and data_output (3x1)
# Output trainset and trainloader in torch
def set_generate(data_input, data_output, type, device, batch_size):
    # 20% for validation
    val_input, data_input = split(data_input, type=type)
    val_output, data_output = split(data_output, type=type)

    Data_input = torch.from_numpy(data_input[:, :]).float().to(device) # 7x1
    Data_output = torch.from_numpy(data_output[:, 2:]).float().to(device) # 1x1
    Val_input = torch.from_numpy(val_input[:, :]).float().to(device)
    Val_output = torch.from_numpy(val_output[:, 2:]).float().to(device)
    trainset = MyDataset(Data_input, Data_output, type)
    valset = MyDataset(Val_input, Val_output, type)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainset, trainloader, valset, val_input, val_output


# Vis of NNs
def heatmap(phi_net, rho_net, x_2=0, y_2=0, z_2=0.5, vx_2=0, vy_2=0, vz_2=0):
    z_min = 0.0
    z_max = 1.0
    y_min = -0.5
    y_max = 0.5
    
    z_length = int((z_max-z_min)*50) + 1
    y_length = int((y_max-y_min)*50) + 1
    
    z = np.linspace(z_max, z_min, z_length)
    y = np.linspace(y_min, y_max, y_length)
    
    fa_heatmap = np.zeros([1, z_length, y_length])
    
    for j in range(y_length):
        for i in range(z_length):
            c = np.zeros([1, 6])
            c[0, 0] = x_2 - 0
            c[0, 1] = y_2 - y[j]
            c[0, 2] = z_2 - z[i]
            c[0, 3] = vx_2
            c[0, 4] = vy_2
            c[0, 5] = vz_2
            cc = torch.from_numpy(c)
            with torch.no_grad():
                fa_heatmap[0, i, j] = rho_net(phi_net(cc[:, :6]))[0, 0].item() # f_a_z
    
    return y, z, y_2, z_2, fa_heatmap

# Vis of NNs
def vis(phi_L_net, rho_L_net, phi_S_net, rho_S_net):
    # visualization
    plt.figure(figsize=(16,4))
    plt.subplot(1, 4, 1)
    y, z, y_2, z_2, fa_heatmap = heatmap(phi_L_net, rho_L_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (L2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap[0, :, :], cmap='Reds_r')
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([y_2], [z_2], marker='*', markersize=10, color="black")

    plt.subplot(1, 4, 2)
    y, z, y_2, z_2, fa_heatmap = heatmap(phi_S_net, rho_S_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (S2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap[0, :, :], cmap='Reds_r')
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([y_2], [z_2], marker='*', markersize=10, color="black")

    plt.subplot(1, 4, 3)
    y, z, y_2, z_2, fa_heatmap = heatmap(phi_L_net, rho_S_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (L2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap[0, :, :], cmap='Reds_r')
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([y_2], [z_2], marker='*', markersize=10, color="black")

    plt.subplot(1, 4, 4)
    y, z, y_2, z_2, fa_heatmap = heatmap(phi_S_net, rho_L_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (S2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap[0, :, :], cmap='Reds_r')
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([y_2], [z_2], marker='*', markersize=10, color="black")
    plt.tight_layout()
    plt.show()

# Val of NNs
def Fa_prediction(data_input, phi_net, rho_net):
    L = len(data_input)
    Fa = np.zeros(L)
    for i in range(L):
        inputs = data_input[i, :6]
        Fa[i] = rho_net(phi_net(torch.from_numpy(inputs))).item()
    return Fa

# Val of NNs
def validation(phi_net, rho_net, data_input, data_output, ss, ee, name):
    Fa_pred = Fa_prediction(data_input, phi_net, rho_net)
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)
    plt.plot(data_input[:, :3])
    plt.legend(['dx', 'dy', 'dz'])
    plt.grid()
    plt.title('Validation: '+name)
    plt.subplot(2, 1, 2)
    plt.plot(data_output[:, 2])
    plt.hlines(y=0, xmin=0, xmax=ee-ss, colors='r')
    plt.plot(Fa_pred)
    plt.legend(['fa_gt', 'fa_pred'])
    plt.grid()
    plt.show()