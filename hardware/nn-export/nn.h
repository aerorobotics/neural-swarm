#ifndef __NN_H__
#define __NN_H__

enum nn_robot_type
{
	NN_ROBOT_SMALL = 0,
	NN_ROBOT_LARGE = 1,
};


void nn_reset(void);

void nn_add_neighbor(const float input[6], enum nn_robot_type type);

void nn_add_neighbor_ground(const float input[4]);

const float* nn_eval(enum nn_robot_type type);

#endif