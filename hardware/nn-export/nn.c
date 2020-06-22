#include <string.h> // memset, memcpy
#include <math.h> //tanhf

#include "nn.h"

// unconventional: include generated c-file in here
#include "nn_generated_weights.c"

static float temp1[40];
static float temp2[40];

static float deepset_sum[NN_H];

static float relu(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

static void layer(int rows, int cols, const float in[], const float layer_weight[][cols], const float layer_bias[],
			float * output, int use_activation) {
	for(int ii = 0; ii < cols; ii++) {
		output[ii] = 0;
		for (int jj = 0; jj < rows; jj++) {
			output[ii] += in[jj] * layer_weight[jj][ii];
		}
		output[ii] += layer_bias[ii];
		if (use_activation == 1) {
			output[ii] = relu(output[ii]);
		}
	}
}

static const float* phi_L(const float input[]) {
	layer(6, 25, input, weights_phi_L.fc1_weight, weights_phi_L.fc1_bias, temp1, 1);
	layer(25, 40, temp1, weights_phi_L.fc2_weight, weights_phi_L.fc2_bias, temp2, 1);
	layer(40, 40, temp2, weights_phi_L.fc3_weight, weights_phi_L.fc3_bias, temp1, 1);
	layer(40, NN_H, temp1, weights_phi_L.fc4_weight, weights_phi_L.fc4_bias, temp2, 0);

	return temp2;
}

static const float* phi_S(const float input[]) {
	layer(6, 25, input, weights_phi_S.fc1_weight, weights_phi_S.fc1_bias, temp1, 1);
	layer(25, 40, temp1, weights_phi_S.fc2_weight, weights_phi_S.fc2_bias, temp2, 1);
	layer(40, 40, temp2, weights_phi_S.fc3_weight, weights_phi_S.fc3_bias, temp1, 1);
	layer(40, NN_H, temp1, weights_phi_S.fc4_weight, weights_phi_S.fc4_bias, temp2, 0);

	return temp2;
}

static const float* phi_G(const float input[]) {
	layer(4, 25, input, weights_phi_G.fc1_weight, weights_phi_G.fc1_bias, temp1, 1);
	layer(25, 40, temp1, weights_phi_G.fc2_weight, weights_phi_G.fc2_bias, temp2, 1);
	layer(40, 40, temp2, weights_phi_G.fc3_weight, weights_phi_G.fc3_bias, temp1, 1);
	layer(40, NN_H, temp1, weights_phi_G.fc4_weight, weights_phi_G.fc4_bias, temp2, 0);

	return temp2;
}

static const float* rho_S(const float input[]) {
	layer(NN_H, 40, input, weights_rho_S.fc1_weight, weights_rho_S.fc1_bias, temp1, 1);
	layer(40, 40, temp1, weights_rho_S.fc2_weight, weights_rho_S.fc2_bias, temp2, 1);
	layer(40, 40, temp2, weights_rho_S.fc3_weight, weights_rho_S.fc3_bias, temp1, 1);
	layer(40, 1, temp1, weights_rho_S.fc4_weight, weights_rho_S.fc4_bias, temp2, 0);

	return temp2;
}

static const float* rho_L(const float input[]) {
	layer(NN_H, 40, input, weights_rho_L.fc1_weight, weights_rho_L.fc1_bias, temp1, 1);
	layer(40, 40, temp1, weights_rho_L.fc2_weight, weights_rho_L.fc2_bias, temp2, 1);
	layer(40, 40, temp2, weights_rho_L.fc3_weight, weights_rho_L.fc3_bias, temp1, 1);
	layer(40, 1, temp1, weights_rho_L.fc4_weight, weights_rho_L.fc4_bias, temp2, 0);

	return temp2;
}

void nn_reset()
{
	memset(deepset_sum, 0, sizeof(deepset_sum));
}

void nn_add_neighbor(const float input[6], enum nn_robot_type type)
{
	const float* phi;
	switch(type)
	{
		case NN_ROBOT_SMALL:
			phi = phi_S(input);
			break;
		case NN_ROBOT_LARGE:
			phi = phi_L(input);
			break;
	}
	for (int i = 0; i < NN_H; ++i) {
		deepset_sum[i] += phi[i];
	}
}

void nn_add_neighbor_ground(const float input[4])
{
	const float* phi = phi_G(input);
	for (int i = 0; i < NN_H; ++i) {
		deepset_sum[i] += phi[i];
	}
}

const float* nn_eval(enum nn_robot_type type)
{
	const float* rho;
	switch(type)
	{
		case NN_ROBOT_SMALL:
			rho = rho_S(deepset_sum);
			break;
		case NN_ROBOT_LARGE:
			rho = rho_L(deepset_sum);
			break;
	}
	return rho;
}
