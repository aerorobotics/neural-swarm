import argparse
import torch
import os
import numpy as np

# converts a numpy array to a C-style string
def arr2str(a):
	return np.array2string(a,
		separator=',',
		floatmode='unique',
		threshold = 1e6,
		max_line_width = 1e6).replace("[","{").replace("]","}").replace("\n","")

def exportNet(gen_file, filename):
	exportname = os.path.splitext(os.path.basename(filename))[0]
	state_dict = torch.load(filename)

	# export C struct definition
	result  = "struct neuralNetworkFF_" + exportname + "\n"
	result +="{\n"
	for key, data in state_dict.items():
		name = key.replace(".", "_")
		a = data.numpy().T

		result += "	float " + name
		for s in a.shape:
			result += "[" + str(s) + "]"
		result += ";\n"
	result += "};\n"

	# export actual variable holding the data from the network
	result += "static const struct neuralNetworkFF_{} weights_{} = {{\n".format(exportname, exportname)
	for key, data in state_dict.items():
		name = key.replace(".", "_")
		a = data.numpy().T
		result += "." + name
		result += " = " + arr2str(a) + ",\n"
	result += "};\n\n"

	gen_file.write(result)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="input folder")
	args = parser.parse_args()

	with open('nn_generated_weights.c', 'w') as gen_file:
		gen_file.write("/* GENERATED FILE - DO NOT EDIT */\n")

		state_dict = torch.load('{}/phi_G.pth'.format(args.input))
		NN_H = state_dict['fc4.bias'].shape[0]
		gen_file.write("#define NN_H ({})\n".format(NN_H))

		exportNet(gen_file, '{}/rho_L.pth'.format(args.input))
		exportNet(gen_file, '{}/rho_S.pth'.format(args.input))
		exportNet(gen_file, '{}/phi_L.pth'.format(args.input))
		exportNet(gen_file, '{}/phi_S.pth'.format(args.input))
		exportNet(gen_file, '{}/phi_G.pth'.format(args.input))
