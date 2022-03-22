from nns import phi_Net, rho_Net
import matplotlib.pyplot as plt
import numpy as np
import torch
# torch.set_default_tensor_type('torch.DoubleTensor')
encoder = {'Ge2L':0, 'Ge2S':1, 'L2L':2, 'S2S':3, 'L2S':4, 'S2L':5, \
           'SS2L':6, 'SL2L':7, 'LL2S':8, 'SL2S':9, 'SS2S':10, 'LL2L':11}

def heatmap(phi_1_net, rho_net, phi_2_net=None, pos1=[0,0,0.5], vel1=[0,0,0], pos2=None, vel2=None, GE=False, pos3=None, phi_3_net=None, vel3=None, pos4=None, phi_4_net=None, vel4=None):
    z_min = 0.0
    z_max = 0.7
    y_min = -0.4
    y_max = 0.4
    
    z_length = int((z_max-z_min)*200) + 1
    y_length = int((y_max-y_min)*200) + 1
    
    z = np.linspace(z_max, z_min, z_length)
    y = np.linspace(y_min, y_max, y_length)
    
    fa_heatmap = np.zeros([1, z_length, y_length], dtype=np.float32)
    
    for j in range(y_length):
        for i in range(z_length):
            c = np.zeros([1, 6], dtype=np.float32)
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
	            c = np.zeros([1, 6], dtype=np.float32)
	            c[0, 0] = pos2[0] - 0
	            c[0, 1] = pos2[1] - y[j]
	            c[0, 2] = pos2[2] - z[i]
	            c[0, 3] = vel2[0]
	            c[0, 4] = vel2[1]
	            c[0, 5] = vel2[2]
	            cc2 = torch.from_numpy(c)
            if pos3 is not None:
                c = np.zeros([1, 6], dtype=np.float32)
                c[0, 0] = pos3[0] - 0
                c[0, 1] = pos3[1] - y[j]
                c[0, 2] = pos3[2] - z[i]
                c[0, 3] = vel3[0]
                c[0, 4] = vel3[1]
                c[0, 5] = vel3[2]
                cc3 = torch.from_numpy(c)
            if pos4 is not None:
                c = np.zeros([1, 6], dtype=np.float32)
                c[0, 0] = pos4[0] - 0
                c[0, 1] = pos4[1] - y[j]
                c[0, 2] = pos4[2] - z[i]
                c[0, 3] = vel4[0]
                c[0, 4] = vel4[1]
                c[0, 5] = vel4[2]
                cc4 = torch.from_numpy(c)
            with torch.no_grad():
            	if pos2 is None:
                	fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]))[0, 0].item() # f_a_z
            	else:
                    if pos3 is None:
                        fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]) + phi_2_net(cc2[:, :]))[0, 0].item() # f_a_z
                    else:
                        if pos4 is None:
                            fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]) + phi_2_net(cc2[:, :]) + phi_3_net(cc3[:, :]))[0, 0].item() # f_a_z
                        else:
                            fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]) + phi_2_net(cc2[:, :]) + phi_3_net(cc3[:, :]) + phi_4_net(cc4[:, :]))[0, 0].item() # f_a_z

    return y, z, fa_heatmap[0, :, :]

def heatmap_superposition(phi_1_net, rho_net, phi_2_net, pos1, vel1, pos2, vel2, pos3, phi_3_net, vel3, pos4, phi_4_net, vel4):
    z_min = 0.0
    z_max = 0.7
    y_min = -0.4
    y_max = 0.4
    
    z_length = int((z_max-z_min)*200) + 1
    y_length = int((y_max-y_min)*200) + 1
    
    z = np.linspace(z_max, z_min, z_length)
    y = np.linspace(y_min, y_max, y_length)
    
    fa_heatmap = np.zeros([1, z_length, y_length], dtype=np.float32)
    
    for j in range(y_length):
        for i in range(z_length):
            c = np.zeros([1, 6], dtype=np.float32)
            c[0, 0] = pos1[0] - 0
            c[0, 1] = pos1[1] - y[j]
            c[0, 2] = pos1[2] - z[i]
            c[0, 3] = vel1[0]
            c[0, 4] = vel1[1]
            c[0, 5] = vel1[2]
            cc1 = torch.from_numpy(c[:, 2:])
            
            c = np.zeros([1, 6], dtype=np.float32)
            c[0, 0] = pos2[0] - 0
            c[0, 1] = pos2[1] - y[j]
            c[0, 2] = pos2[2] - z[i]
            c[0, 3] = vel2[0]
            c[0, 4] = vel2[1]
            c[0, 5] = vel2[2]
            cc2 = torch.from_numpy(c)

            c = np.zeros([1, 6], dtype=np.float32)
            c[0, 0] = pos3[0] - 0
            c[0, 1] = pos3[1] - y[j]
            c[0, 2] = pos3[2] - z[i]
            c[0, 3] = vel3[0]
            c[0, 4] = vel3[1]
            c[0, 5] = vel3[2]
            cc3 = torch.from_numpy(c)

            c = np.zeros([1, 6], dtype=np.float32)
            c[0, 0] = pos4[0] - 0
            c[0, 1] = pos4[1] - y[j]
            c[0, 2] = pos4[2] - z[i]
            c[0, 3] = vel4[0]
            c[0, 4] = vel4[1]
            c[0, 5] = vel4[2]
            cc4 = torch.from_numpy(c)

            with torch.no_grad():
                fa_heatmap[0, i, j] =   rho_net(phi_1_net(cc1[:, :]))[0, 0].item() \
                                      + rho_net(phi_2_net(cc2[:, :]))[0, 0].item() \
                                      + rho_net(phi_3_net(cc3[:, :]))[0, 0].item() \
                                      + rho_net(phi_4_net(cc4[:, :]))[0, 0].item()

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

def vis_paper(pp, phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net, rasterized):
    # visualization
    vmin = -20
    vmax = 5

    # small
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)

    # Ge2S
    ax = axs[0, 0]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, pos1=[0,0,0], GE=True)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.set_ylabel(r'$z$ [m]', fontsize = 15)
    ax.set_title(r'Ground $\rightarrow$ Small', fontsize = 15)
    ax.set_xticks([])
    ax.axhline(y=0.01, linewidth=4, color='black', label='Ground')
    ax.plot([0], [-1], marker='*', markersize=20, color="white", markerfacecolor='black', label='Large Robot')
    ax.plot([0], [-1], marker='*', markersize=10, color="white", markerfacecolor='black', label='Small Robot')
    #ax.scatter([0], [-1], marker='*', s=[20], c="black", label='Large Robot')
    #ax.scatter([0], [-1], marker='*', s=[10], c="black", label='Small Robot')
    ax.legend(fontsize=13, facecolor='white', framealpha=1)
    ax.set_ylim([0, 1])
    ax.text(-0.48, 0.9, '(a)', fontsize=15, color='black')

    # S2S
    ax = axs[0, 1]
    y, z, fa_heatmap = heatmap(phi_S_net, rho_S_net)
    ax.set_title(r'Small $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([0], [0.5], marker='*', markersize=10, color="black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.48, 0.9, '(b)', fontsize=15, color='black')

    # L2S
    ax = axs[0, 2]
    y, z, fa_heatmap = heatmap(phi_L_net, rho_S_net)
    ax.set_title(r'Large $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([0], [0.5], marker='*', markersize=20, color="black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.48, 0.9, '(c)', fontsize=15, color='black')

    # L(moving)2S
    ax = axs[0, 3]
    vel = [0, 0.8, 0]
    y, z, fa_heatmap = heatmap(phi_L_net, rho_S_net, vel1=vel)
    ax.set_title(r'Large (Moving) $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([0], [0.5], marker='*', markersize=20, color="black")
    ax.arrow(0, 0.5, 0.5*vel[1], 0.5*vel[2], length_includes_head=True, head_width=0.03, head_length=0.05, fc='b', ec='b')
    ax.text(0+0.1*vel[1], 0.5+vel[2]+0.08, str(vel[1])+' m/s', fontsize=15, color='b')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.48, 0.9, '(d)', fontsize=15, color='black')

    # GeL2S
    ax = axs[1, 0]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_L_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    ax.set_ylabel(r'$z$ [m]', fontsize = 15)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Ground, Large} $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    #plt.colorbar()
    #plt.tick_params(labelsize = 13)
    ax.plot([0], [0.5], marker='*', markersize=20, color="black")
    ax.axhline(y=0.01, linewidth=4, color='black')
    ax.text(-0.48, 0.9, '(e)', fontsize=15, color='black')

    pos1 = [0, -0.1, 0.5]
    pos2 = [0, 0.1, 0.5]
    vel1 = [0, 0, 0]
    vel2 = [0 ,0, 0]

    # LL2S
    ax = axs[1, 1]
    y, z, fa_heatmap = heatmap(phi_L_net, rho_S_net, phi_2_net=phi_L_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Large, Large} $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker='*', markersize=20, color="black")
    ax.set_yticks([])
    ax.text(-0.48, 0.9, '(f)', fontsize=15, color='black')

    # LS2S
    ax = axs[1, 2]
    y, z, fa_heatmap = heatmap(phi_L_net, rho_S_net, phi_2_net=phi_S_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Large, Small} $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")
    ax.set_yticks([])
    ax.text(-0.48, 0.9, '(g)', fontsize=15, color='black')

    # GeLS2S
    ax = axs[1, 3]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_L_net, phi_3_net=phi_S_net, pos1=[0,0,0], pos2=pos1, vel2=vel1, pos3=pos2, vel3=vel2, GE=True)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Ground, Large, Small} $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")
    ax.set_yticks([])
    ax.axhline(y=0.01, linewidth=4, color='black')
    ax.text(-0.48, 0.9, '(h)', fontsize=15, color='black')

    # Colorbar
    cbar = fig.colorbar(pcm, ax=axs, location='right', shrink=0.7)
    cbar.set_label(r'$f_{a,z}$ [g]', fontsize=15)

    # plt.tight_layout()
    # plt.savefig('../data/vis.pdf')
    # plt.show()
    pp.savefig()
    plt.close()

    # large
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)

    # Ge2S
    ax = axs[0, 0]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_L_net, pos1=[0,0,0], GE=True)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.set_ylabel(r'$z$ [m]', fontsize = 15)
    ax.set_title(r'Ground $\rightarrow$ Large', fontsize = 15)
    ax.set_xticks([])
    ax.axhline(y=0.01, linewidth=4, color='black', label='Ground')
    ax.plot([0], [-1], marker='*', markersize=20, color="white", markerfacecolor='black', label='Large Robot')
    ax.plot([0], [-1], marker='*', markersize=10, color="white", markerfacecolor='black', label='Small Robot')
    #ax.scatter([0], [-1], marker='*', s=[20], c="black", label='Large Robot')
    #ax.scatter([0], [-1], marker='*', s=[10], c="black", label='Small Robot')
    ax.legend(fontsize=13, facecolor='white', framealpha=1)
    ax.set_ylim([0, 1])
    ax.text(-0.48, 0.9, '(a)', fontsize=15, color='black')

    # S2S
    ax = axs[0, 1]
    y, z, fa_heatmap = heatmap(phi_S_net, rho_L_net)
    ax.set_title(r'Small $\rightarrow$ Large', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([0], [0.5], marker='*', markersize=10, color="black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.48, 0.9, '(b)', fontsize=15, color='black')

    # L2S
    ax = axs[0, 2]
    y, z, fa_heatmap = heatmap(phi_L_net, rho_L_net)
    ax.set_title(r'Large $\rightarrow$ Large', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([0], [0.5], marker='*', markersize=20, color="black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.48, 0.9, '(c)', fontsize=15, color='black')

    # L(moving)2S
    ax = axs[0, 3]
    vel = [0, 0.8, 0]
    y, z, fa_heatmap = heatmap(phi_L_net, rho_L_net, vel1=vel)
    ax.set_title(r'Large (Moving) $\rightarrow$ Large', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([0], [0.5], marker='*', markersize=20, color="black")
    ax.arrow(0, 0.5, 0.5*vel[1], 0.5*vel[2], length_includes_head=True, head_width=0.03, head_length=0.05, fc='b', ec='b')
    ax.text(0+0.1*vel[1], 0.5+vel[2]+0.08, str(vel[1])+' m/s', fontsize=15, color='b')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.48, 0.9, '(d)', fontsize=15, color='black')

    # GeL2S
    ax = axs[1, 0]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_L_net, phi_2_net=phi_L_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    ax.set_ylabel(r'$z$ [m]', fontsize = 15)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Ground, Large} $\rightarrow$ Large', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    #plt.colorbar()
    #plt.tick_params(labelsize = 13)
    ax.plot([0], [0.5], marker='*', markersize=20, color="black")
    ax.axhline(y=0.01, linewidth=4, color='black')
    ax.text(-0.48, 0.9, '(e)', fontsize=15, color='black')

    pos1 = [0, -0.1, 0.5]
    pos2 = [0, 0.1, 0.5]
    vel1 = [0, 0, 0]
    vel2 = [0 ,0, 0]

    # LL2S
    ax = axs[1, 1]
    y, z, fa_heatmap = heatmap(phi_L_net, rho_L_net, phi_2_net=phi_L_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Large, Large} $\rightarrow$ Large', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker='*', markersize=20, color="black")
    ax.set_yticks([])
    ax.text(-0.48, 0.9, '(f)', fontsize=15, color='black')

    # LS2S
    ax = axs[1, 2]
    y, z, fa_heatmap = heatmap(phi_L_net, rho_L_net, phi_2_net=phi_S_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Large, Small} $\rightarrow$ Large', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")
    ax.set_yticks([])
    ax.text(-0.48, 0.9, '(g)', fontsize=15, color='black')

    # GeLS2S
    ax = axs[1, 3]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_L_net, phi_2_net=phi_L_net, phi_3_net=phi_S_net, pos1=[0,0,0], pos2=pos1, vel2=vel1, pos3=pos2, vel3=vel2, GE=True)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Ground, Large, Small} $\rightarrow$ Large', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")
    ax.set_yticks([])
    ax.axhline(y=0.01, linewidth=4, color='black')
    ax.text(-0.48, 0.9, '(h)', fontsize=15, color='black')

    # Colorbar
    cbar = fig.colorbar(pcm, ax=axs, location='right', shrink=0.7)
    cbar.set_label(r'$f_{a,z}$ [g]', fontsize=15)

    # plt.tight_layout()
    # plt.savefig('../data/vis.pdf')
    # plt.show()
    pp.savefig()
    plt.close()

def vis_paper_2(phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net):
    # visualization
    vmin = -20
    vmax = 5
    tx = -0.38
    ty = 0.62
    gr = 0.007
    gr2 = 0.007

    # small
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.3), constrained_layout=True)

    # marker = '*'
    # lsize = 20
    # ssize = 10

    marker = '_'
    lsize = 10
    ssize = 10

    # S
    ax = axs[0, 0]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, pos1=[0,0,0], GE=True)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.set_ylabel(r'$z$ [m]', fontsize = 15)
    ax.set_title('Small', fontsize = 15)
    ax.set_xticks([])
    #ax.axis('equal')
    ax.axhline(y=gr, linewidth=4, linestyle='--', color='black', label='Ground')
    #ax.scatter([0], [-1], marker=marker, markersize=lsize/3, color="black", markerfacecolor='black', label='Large Robot')
    #ax.scatter([0], [-1], marker=marker, markersize=ssize/3, color="black", markerfacecolor='black', label='Small Robot')
    ax.scatter([0], [-1], marker=marker, s=[lsize], c="black", label='Large Robot')
    ax.scatter([0], [-1], marker=marker, s=[ssize], c="black", label='Small Robot')
    ax.legend(fontsize=13, facecolor='black', framealpha=0)
    ax.set_ylim([0, 0.7])
    ax.text(tx, ty, '(a)', fontsize=15, color='black')

    # S2S
    ax = axs[0, 1]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_S_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    ax.set_title(r'Small $\rightarrow$ Small', fontsize = 15)
    ax.set_yticks([])
    ax.set_xticks([])
    #ax.axis('equal')
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    #plt.colorbar()
    #plt.tick_params(labelsize = 13)
    ax.plot([0], [0.5], marker=marker, markersize=ssize, color="black")
    ax.axhline(y=gr, linewidth=4, linestyle='--', color='black')
    ax.text(tx, ty, '(b)', fontsize=15, color='black')
    ax.set_ylim([0, 0.7])

    # L2S
    ax = axs[0, 2]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_L_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    ax.set_title(r'Large $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.set_yticks([])
    ax.set_xticks([])
    #ax.axis('equal')
    #plt.colorbar()
    #plt.tick_params(labelsize = 13)
    ax.plot([0], [0.5], marker=marker, markersize=lsize, color="black")
    ax.axhline(y=gr, linewidth=4, linestyle='--', color='black')
    ax.text(tx, ty, '(c)', fontsize=15, color='black')
    ax.set_ylim([0, 0.7])

    # L(moving)2S
    ax = axs[0, 3]
    vel = [0, 0.8, 0]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_L_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=vel, GE=True)
    ax.set_title(r'Large (Moving) $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([0], [0.5], marker=marker, markersize=lsize, color="black")
    ax.arrow(0, 0.5, 0.4*vel[1], 0.4*vel[2], length_includes_head=True, head_width=0.03, head_length=0.05, fc='b', ec='b')
    ax.text(0+0.1*vel[1], 0.5+vel[2]+0.05, str(vel[1])+' m/s', fontsize=15, color='b')
    ax.axhline(y=gr, linewidth=4, linestyle='--', color='black')
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.axis('equal')
    ax.set_ylim([0, 0.7])
    ax.text(tx, ty, '(d)', fontsize=15, color='black')

    pos1 = [0, -0.1, 0.5]
    pos2 = [0, 0.1, 0.5]
    vel1 = [0, 0, 0]
    vel2 = [0 ,0, 0]

    # SL2S
    ax = axs[1, 0]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_L_net, phi_3_net=phi_S_net, pos1=[0,0,0], pos2=pos1, vel2=vel1, pos3=pos2, vel3=vel2, GE=True)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_ylabel(r'$z$ [m]', fontsize = 15)
    #ax.axis('equal')
    ax.set_title(r'{Small, Large} $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker=marker, markersize=lsize, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker=marker, markersize=ssize, color="black")
    ax.axhline(y=gr2, linewidth=4, linestyle='--', color='black')
    ax.set_ylim([0, 0.7])
    ax.text(tx, ty, '(e)', fontsize=15, color='black')

    # LL2S
    ax = axs[1, 1]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_L_net, phi_3_net=phi_L_net, pos1=[0,0,0], pos2=pos1, vel2=vel1, pos3=pos2, vel3=vel2, GE=True)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Large, Large} $\rightarrow$ Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker=marker, markersize=lsize, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker=marker, markersize=lsize, color="black")
    ax.set_yticks([])
    ax.set_ylim([0, 0.7])
    #ax.axis('equal')
    ax.axhline(y=gr2, linewidth=4, linestyle='--', color='black')
    ax.text(tx, ty, '(f)', fontsize=15, color='black')

    pos3 = [0, 0, 0.6]
    vel3 = [0, 0, 0]

    # SSS2S
    ax = axs[1, 2]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_S_net, phi_3_net=phi_S_net, pos1=[0,0,0], pos2=pos1, vel2=vel1, pos3=pos2, vel3=vel2, GE=True, phi_4_net=phi_S_net, pos4=pos3, vel4=vel3)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Small,Small,Small}$\rightarrow$Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker=marker, markersize=ssize, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker=marker, markersize=ssize, color="black")
    ax.plot([pos3[1]], [pos3[2]], marker=marker, markersize=ssize, color="black")
    ax.set_yticks([])
    ax.set_ylim([0, 0.7])
    #ax.axis('equal')
    ax.axhline(y=gr2, linewidth=4, linestyle='--', color='black')
    ax.text(tx, ty, '(g)', fontsize=15, color='black')

    # SSL2S
    ax = axs[1, 3]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_S_net, phi_3_net=phi_S_net, pos1=[0,0,0], pos2=pos1, vel2=vel1, pos3=pos2, vel3=vel2, GE=True, phi_4_net=phi_L_net, pos4=pos3, vel4=vel3)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Small,Small,Large}$\rightarrow$Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker=marker, markersize=ssize, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker=marker, markersize=ssize, color="black")
    ax.plot([pos3[1]], [pos3[2]], marker=marker, markersize=lsize, color="black")
    ax.set_yticks([])
    ax.set_ylim([0, 0.7])
    #ax.axis('equal')
    ax.axhline(y=gr2, linewidth=4, linestyle='--', color='black')
    ax.text(tx, ty, '(h)', fontsize=15, color='black')

    # Colorbar
    cbar = fig.colorbar(pcm, ax=axs, location='right', shrink=0.7)
    cbar.set_label(r'$f_{a,z}$ [g]', fontsize=15)

    #plt.tight_layout()
    plt.savefig('../data/vis_paper.pdf')
    plt.savefig('../data/vis_paper.png')
    plt.savefig('../data/vis_paper_2.png', dpi=400)
    plt.savefig('../data/vis_paper_3.png', dpi=800)
    # plt.show()
    plt.close()


def Fa_prediction(data_input, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, typ):
    L = len(data_input)
    Fa = np.zeros(L)
    for i in range(L):
        with torch.no_grad():
            inputs = torch.from_numpy(data_input[[i], :])
            temp = encoder[typ]
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

def validation(phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, data_input, data_output, ss, ee, name, typ, rasterized):
    Fa_pred = Fa_prediction(data_input, phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, typ)
    plt.figure(figsize=(12, 15))
    plt.subplot(4, 1, 1, rasterized=rasterized)
    # plt.plot(-data_input[ss:ee, :3])
    # plt.legend(['x', 'y', 'z'])
    plt.plot(-data_input[ss:ee, 2])
    plt.legend('z')    
    plt.grid()
    plt.title('Validation: '+name)
    plt.subplot(4, 1, 2, rasterized=rasterized)
    plt.plot(data_input[ss:ee, 6:9])
    plt.legend(['x21', 'y21', 'z21'])
    plt.grid()
    plt.subplot(4, 1, 3, rasterized=rasterized)
    plt.plot(data_input[ss:ee, 12:15])
    plt.legend(['x31', 'y31', 'z31'])
    plt.grid()
    plt.subplot(4, 1, 4, rasterized=rasterized)
    plt.plot(data_output[ss:ee, 2])
    plt.hlines(y=0, xmin=0, xmax=ee-ss, colors='r')
    plt.plot(Fa_pred[ss:ee])
    plt.legend(['fa_gt', 'fa_pred'])
    plt.grid()
    plt.show()

def superposition_compare(phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net):
    # visualization
    vmin = -20
    vmax = 5
    tx = -0.38
    ty = 0.62
    gr = 0.007
    gr2 = 0.007

    # small
    fig, axs = plt.subplots(1, 3, figsize=(10, 6.3), constrained_layout=True)
    marker = '*'
    lsize = 20
    ssize = 10

    # S2S
    ax = axs[0]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_S_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    ax.set_title(r'Small $\rightarrow$ Small', fontsize = 15)
    #ax.axis('equal')
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    #plt.colorbar()
    #plt.tick_params(labelsize = 13)
    ax.plot([0], [0.5], marker=marker, markersize=ssize, color="black")
    ax.axhline(y=gr, linewidth=4, linestyle='--', color='black')
    ax.text(tx, ty, '(b)', fontsize=15, color='black')
    ax.set_ylim([0, 0.7])

    pos1 = [0, -0.1, 0.5]
    pos2 = [0, 0.1, 0.5]
    vel1 = [0, 0, 0]
    vel2 = [0 ,0, 0]
    pos3 = [0, 0, 0.6]
    vel3 = [0, 0, 0]

    # SSS2S
    ax = axs[1]
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_S_net, phi_3_net=phi_S_net, pos1=[0,0,0], pos2=pos1, vel2=vel1, pos3=pos2, vel3=vel2, GE=True, phi_4_net=phi_S_net, pos4=pos3, vel4=vel3)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Small,Small,Small}$\rightarrow$Small', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker=marker, markersize=ssize, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker=marker, markersize=ssize, color="black")
    ax.plot([pos3[1]], [pos3[2]], marker=marker, markersize=ssize, color="black")
    ax.set_yticks([])
    ax.set_ylim([0, 0.7])
    #ax.axis('equal')
    ax.axhline(y=gr2, linewidth=4, linestyle='--', color='black')
    ax.text(tx, ty, '(g)', fontsize=15, color='black')

    # SSS2S (superposition)
    ax = axs[2]
    y, z, fa_heatmap_s = heatmap_superposition(phi_1_net=phi_G_net, rho_net=rho_S_net, phi_2_net=phi_S_net, phi_3_net=phi_S_net, pos1=[0,0,0], vel1=[0,0,0], pos2=pos1, vel2=vel1, pos3=pos2, vel3=vel2, phi_4_net=phi_S_net, pos4=pos3, vel4=vel3)
    ax.set_xlabel(r'$y$ [m]', fontsize = 15)
    ax.set_title(r'{Small,Small,Small}$\rightarrow$Small (superposition)', fontsize = 15)
    pcm = ax.pcolor(y, z, fa_heatmap_s, cmap='Reds_r', vmin=vmin, vmax=vmax)
    ax.plot([pos1[1]], [pos1[2]], marker=marker, markersize=ssize, color="black")
    ax.plot([pos2[1]], [pos2[2]], marker=marker, markersize=ssize, color="black")
    ax.plot([pos3[1]], [pos3[2]], marker=marker, markersize=ssize, color="black")
    ax.set_yticks([])
    ax.set_ylim([0, 0.7])
    #ax.axis('equal')
    ax.axhline(y=gr2, linewidth=4, linestyle='--', color='black')
    ax.text(tx, ty, '(g*)', fontsize=15, color='black')

    # Colorbar
    cbar = fig.colorbar(pcm, ax=axs, location='bottom', shrink=0.7)
    cbar.set_label(r'$f_{a,z}$ [g]', fontsize=15)

    # statistics
    print('Mean gap: ', np.mean(np.abs(fa_heatmap-fa_heatmap_s)))
    print('Max gap: ', np.max(np.abs(fa_heatmap-fa_heatmap_s)))

    plt.show()

# load NNs
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    h = 20
    phi_G_net = phi_Net(inputdim=4,hiddendim=h).to(device, dtype=torch.float32)
    phi_L_net = phi_Net(inputdim=6,hiddendim=h).to(device, dtype=torch.float32)
    phi_S_net = phi_Net(inputdim=6,hiddendim=h).to(device, dtype=torch.float32)
    rho_L_net = rho_Net(hiddendim=h).to(device, dtype=torch.float32)
    rho_S_net = rho_Net(hiddendim=h).to(device, dtype=torch.float32)

    path = '../data/models/val_with23_wdelay/epoch20_lip3_h20_f0d35_B256/'
    phi_G_net.load_state_dict(torch.load(path + 'phi_G.pth'))
    rho_L_net.load_state_dict(torch.load(path + 'rho_L.pth'))
    rho_S_net.load_state_dict(torch.load(path + 'rho_S.pth'))
    phi_L_net.load_state_dict(torch.load(path + 'phi_L.pth'))
    phi_S_net.load_state_dict(torch.load(path + 'phi_S.pth'))

    phi_G_net.cpu()
    phi_L_net.cpu()
    phi_S_net.cpu()
    rho_L_net.cpu()
    rho_S_net.cpu()

    superposition_compare(phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net)
    # vis_paper_2(phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net)

    '''
    from utils import interpolation_cubic, data_extraction, Fa, get_data

    path = '../data/validation/epoch40_lip3_h20_f0d35_B256_3cfs/'
    data_1 = data_extraction(path + 'cf52_00.csv')
    data_2 = data_extraction(path + 'cf51_00.csv')
    data_3 = data_extraction(path + 'cf50_00.csv')
    typ = 'SS2S'

    TF_1 = np.floor(data_1['time'][-1]*100)/100.0 - 0.01
    TF_2 = np.floor(data_2['time'][-1]*100)/100.0 - 0.01
    TF_3 = np.floor(data_3['time'][-1]*100)/100.0 - 0.01
    TF = min(TF_1, TF_2, TF_3)
    data_1 = interpolation_cubic(0, TF, data_1, ss=0, ee=-1)
    data_2 = interpolation_cubic(0, TF, data_2, ss=0, ee=-1)
    data_3 = interpolation_cubic(0, TF, data_3, ss=0, ee=-1)

    # big CF
    m = 67
    g = 9.81
    C_00 = 44.10386631845999
    C_10 = -122.51151800146272
    C_01 = -36.18484254283743
    C_20 = 53.10772568607133
    C_11 = 107.6819263349139

    # small CF
    m = 32
    g = 9.81
    C_00 = 11.093358483549203
    C_10 = -39.08104165843915
    C_01 = -9.525647087583181
    C_20 = 20.573302305476638
    C_11 = 38.42885066644033
    data_1 = Fa(data_1, m, g, C_00, C_10, C_01, C_20, C_11)
    data_2 = Fa(data_2, m, g, C_00, C_10, C_01, C_20, C_11)
    data_3 = Fa(data_3, m, g, C_00, C_10, C_01, C_20, C_11)

    data_input, data_output = get_data(D1=data_1, D2=data_2, D3=data_3, s=encoder[typ], typ='fa_delay', always_GE=True)
    
    validation(phi_G_net, phi_S_net, phi_L_net, rho_S_net, rho_L_net, data_input, data_output, ss=0, ee=-1, name=typ, typ=typ, rasterized=False)
    '''