#include <stdio.h>

#include "nn.h"

int main()
{

	nn_reset();

	const float neighbor1[] = {0,0,0.5,0,0,0};
	nn_add_neighbor(neighbor1, NN_ROBOT_SMALL);
	// const float neighbor2[] = {5,6,7,8};
	// nn_add_neighbor(neighbor2);

	// const float obstacle1[] = {1,2};
	// nn_add_obstacle(obstacle1);

	// float goal[4] = {0,0,3,4};

	const float* result = nn_eval(NN_ROBOT_SMALL);
	printf("%f\n", result[0]);

	return 0;
}