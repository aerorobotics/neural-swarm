import nnexport
import numpy as np
import matplotlib.pyplot as plt

def heatmap(ax, title, neighbors, cftype):
	z_min = 0.0
	z_max = 1.0
	y_min = -0.5
	y_max = 0.5
	vmin = -20
	vmax = 5
	
	z_length = int((z_max-z_min)*50) + 1
	y_length = int((y_max-y_min)*50) + 1
	
	z = np.linspace(z_max, z_min, z_length)
	y = np.linspace(y_min, y_max, y_length)
	
	fa_heatmap = np.zeros([z_length, y_length], dtype=np.float32)
	
	for j in range(y_length):
		for i in range(z_length):
			nnexport.nn_reset()
			for neighbor in neighbors:
				pos = neighbor['pos']
				if 'vel' in neighbor:
					vel = neighbor['vel']
				else:
					vel = np.zeros(3)
				c = np.zeros(6, dtype=np.float32)
				c[0] = pos[0] - 0
				c[1] = pos[1] - y[j]
				c[2] = pos[2] - z[i]
				c[3:6] = vel
				if neighbor['type'] == 'ground':
					nnexport.nn_add_neighbor_ground(c[2:])
				elif neighbor['type'] == 'small':
					nnexport.nn_add_neighbor(c, nnexport.NN_ROBOT_SMALL)
					ax.plot([pos[1]], [pos[2]], marker='*', markersize=10, color="black")
				elif neighbor['type'] == 'large':
					nnexport.nn_add_neighbor(c, nnexport.NN_ROBOT_LARGE)
					ax.plot([pos[1]], [pos[2]], marker='*', markersize=20, color="black")
				else:
					print("ERROR")
			if cftype == 'small':
				fa_heatmap[i, j] = nnexport.nn_eval(nnexport.NN_ROBOT_SMALL)[0]
			elif cftype == 'large':
				fa_heatmap[i, j] = nnexport.nn_eval(nnexport.NN_ROBOT_LARGE)[0]

	ax.set_aspect('equal')
	ax.set_ylabel(r'$z$ (m)', fontsize = 15)
	ax.set_title(title, fontsize = 15)
	img = ax.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
	plt.colorbar(img, ax=ax)
	ax.tick_params(labelsize = 13)


# visualization
fig, axs = plt.subplots(5, 4, figsize=(20,16), gridspec_kw={'hspace': 0.5})
# heatmap(axs[0,0], r'${F_a}_z$ (Ge2L)', [{'type': 'ground', 'pos': [0,0,0]}], 'large')
# heatmap(axs[0,1], r'${F_a}_z$ (Ge2S)', [{'type': 'ground', 'pos': [0,0,0]}], 'small')

# heatmap(axs[1,0], r'${F_a}_z$ (L2L)', [{'type': 'large', 'pos': [0,0,0.5]}], 'large')
heatmap(axs[1,1], r'${F_a}_z$ (S2S)', [{'type': 'small', 'pos': [0,0,0.5]}], 'small')
# heatmap(axs[1,2], r'${F_a}_z$ (L2S)', [{'type': 'large', 'pos': [0,0,0.5]}], 'small')
# heatmap(axs[1,3], r'${F_a}_z$ (S2L)', [{'type': 'small', 'pos': [0,0,0.5]}], 'large')

# heatmap(axs[2,0], r'${F_a}_z$ (GeL2L)', [{'type': 'ground', 'pos': [0,0,0]}, {'type': 'large', 'pos': [0,0,0.5]}], 'large')
heatmap(axs[2,1], r'${F_a}_z$ (GeS2S)', [{'type': 'ground', 'pos': [0,0,0]}, {'type': 'small', 'pos': [0,0,0.5]}], 'small')
# heatmap(axs[2,2], r'${F_a}_z$ (GeL2S)', [{'type': 'ground', 'pos': [0,0,0]}, {'type': 'large', 'pos': [0,0,0.5]}], 'small')
# heatmap(axs[2,3], r'${F_a}_z$ (GeS2L)', [{'type': 'ground', 'pos': [0,0,0]}, {'type': 'small', 'pos': [0,0,0.5]}], 'large')

# heatmap(axs[3,0], r'${F_a}_z$ (SS2L)', [{'type': 'small', 'pos': [0,-0.1,0.5]}, {'type': 'small', 'pos': [0,0.1,0.5]}], 'large')
# heatmap(axs[3,1], r'${F_a}_z$ (LL2L)', [{'type': 'large', 'pos': [0,-0.1,0.5]}, {'type': 'large', 'pos': [0,0.1,0.5]}], 'large')
# heatmap(axs[3,2], r'${F_a}_z$ (LS2L)', [{'type': 'large', 'pos': [0,-0.1,0.5]}, {'type': 'small', 'pos': [0,0.1,0.5]}], 'large')
# heatmap(axs[3,3], r'${F_a}_z$ (SL2L)', [{'type': 'small', 'pos': [0,-0.1,0.5]}, {'type': 'large', 'pos': [0,0.1,0.5]}], 'large')

# heatmap(axs[4,0], r'${F_a}_z$ (SS2S)', [{'type': 'small', 'pos': [0,-0.1,0.5]}, {'type': 'small', 'pos': [0,0.1,0.5]}], 'small')
# heatmap(axs[4,1], r'${F_a}_z$ (LL2S)', [{'type': 'large', 'pos': [0,-0.1,0.5]}, {'type': 'large', 'pos': [0,0.1,0.5]}], 'small')
# heatmap(axs[4,2], r'${F_a}_z$ (LS2S)', [{'type': 'large', 'pos': [0,-0.1,0.5]}, {'type': 'small', 'pos': [0,0.1,0.5]}], 'small')
# heatmap(axs[4,3], r'${F_a}_z$ (SL2S)', [{'type': 'small', 'pos': [0,-0.1,0.5]}, {'type': 'large', 'pos': [0,0.1,0.5]}], 'small')


plt.show()
