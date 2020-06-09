from nns import phi_Net, rho_Net
import matplotlib.pyplot as plt
import numpy as np
import torch
# torch.set_default_tensor_type('torch.DoubleTensor')

def heatmap(phi_1_net, rho_net, phi_2_net=None, pos1=[0,0,0.5], vel1=[0,0,0], pos2=None, vel2=None, GE=False, pos3=None, phi_3_net=None, vel3=None):
    z_min = 0.0
    z_max = 1.0
    y_min = -0.5
    y_max = 0.5
    
    z_length = int((z_max-z_min)*100) + 1
    y_length = int((y_max-y_min)*100) + 1
    
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
            with torch.no_grad():
            	if pos2 is None:
                	fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]))[0, 0].item() # f_a_z
            	else:
                    if pos3 is None:
                        fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]) + phi_2_net(cc2[:, :]))[0, 0].item() # f_a_z
                    else:
                        fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]) + phi_2_net(cc2[:, :]) + phi_3_net(cc3[:, :]))[0, 0].item() # f_a_z
    
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

def vis_paper(phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net, rasterized):
    # visualization
    vmin = -18
    vmax = 5
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
    plt.savefig('../data/vis.pdf')
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

h = 20
phi_G_net = phi_Net(inputdim=4,hiddendim=h).to(device, dtype=torch.float32)
phi_L_net = phi_Net(inputdim=6,hiddendim=h).to(device, dtype=torch.float32)
phi_S_net = phi_Net(inputdim=6,hiddendim=h).to(device, dtype=torch.float32)
rho_L_net = rho_Net(hiddendim=h).to(device, dtype=torch.float32)
rho_S_net = rho_Net(hiddendim=h).to(device, dtype=torch.float32)

path = '../data/models/val_new_split/epoch5_lip3_h20'
phi_G_net.load_state_dict(torch.load(path + '/phi_G.pth'))
rho_L_net.load_state_dict(torch.load(path + '/rho_L.pth'))
rho_S_net.load_state_dict(torch.load(path + '/rho_S.pth'))
phi_L_net.load_state_dict(torch.load(path + '/phi_L.pth'))
phi_S_net.load_state_dict(torch.load(path + '/phi_S.pth'))

phi_G_net.cpu()
phi_L_net.cpu()
phi_S_net.cpu()
rho_L_net.cpu()
rho_S_net.cpu()

vis_paper(phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net, rasterized=False)