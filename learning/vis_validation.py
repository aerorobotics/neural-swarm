from nns import phi_Net, rho_Net
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_default_tensor_type('torch.DoubleTensor')

def heatmap(phi_1_net, rho_net, phi_2_net=None, pos1=[0,0,0.5], vel1=[0,0,0], pos2=None, vel2=None, GE=False):
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
            c[0, 0] = pos1[0] - 0
            c[0, 1] = pos1[1] - y[j]
            c[0, 2] = pos1[2] - z[i]
            c[0, 3] = vel1[0]
            c[0, 4] = vel1[1]
            c[0, 5] = vel1[2]
            cc1 = torch.from_numpy(c)
            if GE:
            	cc1 = torch.from_numpy(c[:, 2:])
            if pos2 is not None:
	            c = np.zeros([1, 6])
	            c[0, 0] = pos2[0] - 0
	            c[0, 1] = pos2[1] - y[j]
	            c[0, 2] = pos2[2] - z[i]
	            c[0, 3] = vel2[0]
	            c[0, 4] = vel2[1]
	            c[0, 5] = vel2[2]
	            cc2 = torch.from_numpy(c)
            with torch.no_grad():
            	if pos2 is None:
                	fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]))[0, 0].item() # f_a_z
            	else:
                	fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]) + phi_2_net(cc2[:, :]))[0, 0].item() # f_a_z
    
    return y, z, fa_heatmap[0, :, :]

def vis(pp, phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net, rasterized):
    # visualization
    vmin = -20
    vmax = 5
    plt.figure(figsize=(20,16))
    plt.subplot(5, 4, 1, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_L_net, pos1=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (Ge2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)

    plt.subplot(5, 4, 2, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, pos1=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (Ge2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)

    plt.subplot(5, 4, 5, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_L_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (L2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 6, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_S_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (S2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 7, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_S_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (L2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 8, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_L_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (S2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")
    
    plt.subplot(5, 4, 9, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_L_net, phi_2_net=phi_L_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (GeL2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=20, color="black")

    plt.subplot(5, 4, 10, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_L_net, phi_2_net=phi_S_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (GeS2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 11, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_L_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (GeL2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=20, color="black")

    plt.subplot(5, 4, 12, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_S_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (GeS2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")

    pos1 = [0, -0.1, 0.5]
    pos2 = [0, 0.1, 0.5]
    vel1 = [0, 0, 0]
    vel2 = [0 ,0, 0]
    plt.subplot(5, 4, 13, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_L_net, phi_2_net=phi_S_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (SS2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=10, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 14, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_L_net, phi_2_net=phi_L_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (LL2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=20, color="black")

    plt.subplot(5, 4, 15, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_L_net, phi_2_net=phi_S_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (LS2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 16, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_L_net, phi_2_net=phi_L_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    # plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (SL2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=10, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=20, color="black")

    plt.subplot(5, 4, 17, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_S_net, phi_2_net=phi_S_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (SS2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=10, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 18, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_S_net, phi_2_net=phi_L_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (LL2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=20, color="black")

    plt.subplot(5, 4, 19, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_S_net, phi_2_net=phi_S_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (LS2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 20, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_S_net, phi_2_net=phi_L_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (SL2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=10, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=20, color="black")
    plt.tight_layout()
    pp.savefig()
    plt.close()
    #plt.show()

'''
phi_G_net = phi_Net(inputdim=4,hiddendim=20)
phi_L_net = phi_Net(inputdim=6,hiddendim=20)
phi_S_net = phi_Net(inputdim=6,hiddendim=20)
rho_L_net = rho_Net(hiddendim=20)
rho_S_net = rho_Net(hiddendim=20)

phi_G_net.load_state_dict(torch.load('./models_19and20/phi_G.pth'))
rho_L_net.load_state_dict(torch.load('./models_19and20/rho_L.pth'))
rho_S_net.load_state_dict(torch.load('./models_19and20/rho_S.pth'))
phi_L_net.load_state_dict(torch.load('./models_19and20/phi_L.pth'))
phi_S_net.load_state_dict(torch.load('./models_19and20/phi_S.pth'))

vis(phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net)
'''