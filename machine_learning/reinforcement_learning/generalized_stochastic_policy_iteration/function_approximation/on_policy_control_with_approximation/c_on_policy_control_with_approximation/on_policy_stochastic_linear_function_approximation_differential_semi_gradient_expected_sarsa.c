#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define MAX_NUM_VARS 20 // maximum number of variables in a grid-tiling

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function gets the tile indicies for each tiling */
void GetTileIndices(unsigned int num_tilings, unsigned int memory_size, double* doubles, unsigned int num_doubles, int* ints, unsigned int num_ints, unsigned int* tile_indices);

/* This function takes the modulo of n by k even when n is negative */
int ModuloNegativeSafe(int n, int k);

/* This function takes an array of integers and returns the corresponding tile after hashing */
int HashTiles(int* ints, unsigned int num_ints, long m, int increment);

/* This function initializes episodes */
unsigned int InitializeEpisode(unsigned int number_of_states);

/* This function selects a policy with using epsilon-greedy from the state-action-value function */
void EpsilonGreedyPolicyFromApproximateStateActionFunction(unsigned int max_number_of_actions, unsigned int number_of_state_tilings, unsigned int number_of_state_tiles, unsigned int* state_tile_indices, double* weights, double* approximate_state_action_value_function, double* policy, double* policy_cumulative_sum, double epsilon);

/* This function loops forever through steps and updates the policy */
void LoopForeverThroughSteps(unsigned int number_of_states, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, unsigned int max_number_of_actions, unsigned int number_of_state_tilings, unsigned int number_of_state_tiles, double** state_double_variables, unsigned int number_of_state_double_variables, int** state_int_variables, unsigned int number_of_state_int_variables, unsigned int* state_tile_indices, unsigned int* next_state_tile_indices, double* weights, double* approximate_state_action_value_function, double* policy, double* policy_cumulative_sum, double alpha, double beta, double epsilon, double average_reward_estimate, unsigned int maximum_episode_length, unsigned int state_index);

/* This function calculates the approximate state action value function w^T * x */
double ApproximateStateActionValueFunction(unsigned int number_of_state_tilings, unsigned int number_of_state_tiles, unsigned int* state_tile_indices, unsigned int action_index, double* weights);

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void);

/*********************************************************************************************************/
/************************************************* MAIN **************************************************/
/*********************************************************************************************************/

int main(int argc, char* argv[])
{
	unsigned int i, j, k;
	int system_return;
	
	/*********************************************************************************************************/
	/**************************************** READ IN THE ENVIRONMENT ****************************************/
	/*********************************************************************************************************/
	
	/* Get the number of states */
	unsigned int number_of_states = 0;
	
	FILE* infile_number_of_states = fopen("inputs/number_of_states.txt", "r");
	system_return = fscanf(infile_number_of_states, "%u", &number_of_states);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/number_of_states.txt\n");
	}
	fclose(infile_number_of_states);
	
	/* Get the number of actions per all states */
	unsigned int* number_of_actions_per_state;
	
	FILE* infile_number_of_actions_per_state = fopen("inputs/number_of_actions_per_state.txt", "r");
	number_of_actions_per_state = malloc(sizeof(int) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		system_return = fscanf(infile_number_of_actions_per_state, "%u", &number_of_actions_per_state[i]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/number_of_actions_per_state.txt\n");
		}
	} // end of i loop
	fclose(infile_number_of_actions_per_state);
	
	/* Get the number of state-action successor states */
	unsigned int** number_of_state_action_successor_states;
	
	FILE* infile_number_of_state_action_successor_states = fopen("inputs/number_of_state_action_successor_states.txt", "r");
	number_of_state_action_successor_states = malloc(sizeof(int*) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		number_of_state_action_successor_states[i] = malloc(sizeof(int) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			system_return = fscanf(infile_number_of_state_action_successor_states, "%u\t", &number_of_state_action_successor_states[i][j]);
			if (system_return == -1)
			{
				printf("Failed reading file inputs/number_of_state_action_successor_states.txt\n");
			}	
		} // end of j loop
	} // end of i loop		
	fclose(infile_number_of_state_action_successor_states);
	
	/* Get the state-action-successor state indices */
	unsigned int*** state_action_successor_state_indices;
	
	FILE* infile_state_action_successor_state_indices = fopen("inputs/state_action_successor_state_indices.txt", "r");
	state_action_successor_state_indices = malloc(sizeof(unsigned int**) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_action_successor_state_indices[i] = malloc(sizeof(unsigned int*) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			state_action_successor_state_indices[i][j] = malloc(sizeof(unsigned int*) * number_of_state_action_successor_states[i][j]);
			for (k = 0; k < number_of_state_action_successor_states[i][j]; k++)
			{
				system_return = fscanf(infile_state_action_successor_state_indices, "%u\t", &state_action_successor_state_indices[i][j][k]);
				if (system_return == -1)
				{
					printf("Failed reading file inputs/state_action_successor_state_indices.txt\n");
				}
			} // end of k loop
			
			system_return = fscanf(infile_state_action_successor_state_indices, "\n");
			if (system_return == -1)
			{
				printf("Failed reading file inputs/state_action_successor_state_indices.txt\n");
			}
		} // end of j loop
	} // end of i loop
	fclose(infile_state_action_successor_state_indices);
	
	/* Get the state-action-successor state transition probabilities */
	double*** state_action_successor_state_transition_probabilities;
	
	FILE* infile_state_action_successor_state_transition_probabilities = fopen("inputs/state_action_successor_state_transition_probabilities.txt", "r");
	state_action_successor_state_transition_probabilities = malloc(sizeof(double**) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_action_successor_state_transition_probabilities[i] = malloc(sizeof(double*) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			state_action_successor_state_transition_probabilities[i][j] = malloc(sizeof(double*) * number_of_state_action_successor_states[i][j]);
			for (k = 0; k < number_of_state_action_successor_states[i][j]; k++)
			{
				system_return = fscanf(infile_state_action_successor_state_transition_probabilities, "%lf\t", &state_action_successor_state_transition_probabilities[i][j][k]);
				if (system_return == -1)
				{
					printf("Failed reading file inputs/state_action_successor_state_transition_probabilities.txt\n");
				}
			} // end of k loop
			
			system_return = fscanf(infile_state_action_successor_state_transition_probabilities, "\n");
			if (system_return == -1)
			{
				printf("Failed reading file inputs/state_action_successor_state_transition_probabilities.txt\n");
			}
		} // end of j loop
	} // end of i loop
	fclose(infile_state_action_successor_state_transition_probabilities);
	
	/* Create the state-action-successor state transition probability cumulative sum array */
	double*** state_action_successor_state_transition_probabilities_cumulative_sum;
	
	state_action_successor_state_transition_probabilities_cumulative_sum = malloc(sizeof(double**) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_action_successor_state_transition_probabilities_cumulative_sum[i] = malloc(sizeof(double*) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			state_action_successor_state_transition_probabilities_cumulative_sum[i][j] = malloc(sizeof(double*) * number_of_state_action_successor_states[i][j]);
			
			if (number_of_state_action_successor_states[i][j] > 0)
			{
				state_action_successor_state_transition_probabilities_cumulative_sum[i][j][0] = state_action_successor_state_transition_probabilities[i][j][0];
				
				for (k = 1; k < number_of_state_action_successor_states[i][j]; k++)
				{
					state_action_successor_state_transition_probabilities_cumulative_sum[i][j][k] = state_action_successor_state_transition_probabilities_cumulative_sum[i][j][k - 1] + state_action_successor_state_transition_probabilities[i][j][k];
				} // end of k loop
			}
		} // end of j loop
	} // end of i loop
	
	/* Get the state-action-successor state rewards */
    double*** state_action_successor_state_rewards;
	
	FILE* infile_state_action_successor_state_rewards = fopen("inputs/state_action_successor_state_rewards.txt", "r");
	state_action_successor_state_rewards = malloc(sizeof(double**) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_action_successor_state_rewards[i] = malloc(sizeof(double*) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			state_action_successor_state_rewards[i][j] = malloc(sizeof(double) * number_of_state_action_successor_states[i][j]);
			for (k = 0; k < number_of_state_action_successor_states[i][j]; k++)
			{
				system_return = fscanf(infile_state_action_successor_state_rewards, "%lf\t", &state_action_successor_state_rewards[i][j][k]);
				if (system_return == -1)
				{
					printf("Failed reading file inputs/state_action_successor_state_rewards.txt\n");
				}
			} // end of k loop
			
			system_return = fscanf(infile_state_action_successor_state_rewards, "\n");
			if (system_return == -1)
			{
				printf("Failed reading file inputs/state_action_successor_state_rewards.txt\n");
			}
		} // end of j loop
	} // end of i loop
	fclose(infile_state_action_successor_state_rewards);
	
	/*********************************************************************************************************/
	/********************************************* CREATE TILING *********************************************/
	/*********************************************************************************************************/
	
	/* Get the number of state double variables */
	unsigned int number_of_state_double_variables = 0;
	
	FILE* infile_number_of_state_double_variables = fopen("inputs/number_of_state_double_variables.txt", "r");
	system_return = fscanf(infile_number_of_state_double_variables, "%u", &number_of_state_double_variables);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/number_of_state_double_variables.txt\n");
	}
	fclose(infile_number_of_state_double_variables);
	
	/* Get the number of state int variables */
	unsigned int number_of_state_int_variables = 0;
	
	FILE* infile_number_of_state_int_variables = fopen("inputs/number_of_state_int_variables.txt", "r");
	system_return = fscanf(infile_number_of_state_int_variables, "%u", &number_of_state_int_variables);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/number_of_state_int_variables.txt\n");
	}
	fclose(infile_number_of_state_int_variables);
	
	/* Get state double variables */
	double** state_double_variables;
	
	FILE* infile_state_double_variables = fopen("inputs/state_double_variables.txt", "r");
	state_double_variables = malloc(sizeof(double*) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_double_variables[i] = malloc(sizeof(double) * number_of_state_double_variables);
		for (j = 0; j < number_of_state_double_variables; j++)
		{
			system_return = fscanf(infile_state_double_variables, "%lf\t", &state_double_variables[i][j]);
			if (system_return == -1)
			{
				printf("Failed reading file inputs/state_double_variables.txt\n");
			}	
		} // end of j loop		
	} // end of i loop
	fclose(infile_state_double_variables);
	
	/* Get state int variables */
	int** state_int_variables;
	
	FILE* infile_state_int_variables = fopen("inputs/state_int_variables.txt", "r");
	state_int_variables = malloc(sizeof(int*) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_int_variables[i] = malloc(sizeof(int) * number_of_state_int_variables);
		for (j = 0; j < number_of_state_int_variables; j++)
		{
			system_return = fscanf(infile_state_int_variables, "%d\t", &state_int_variables[i][j]);
			if (system_return == -1)
			{
				printf("Failed reading file inputs/state_int_variables.txt\n");
			}	
		} // end of j loop		
	} // end of i loop
	fclose(infile_state_int_variables);
	
	/* Set the number of tilings */
	unsigned int number_of_state_tilings = 4;
	
	/* Set the number of tiles per tiling */
	unsigned int number_of_state_tiles_per_state_tiling = 4 * 4;
	
	/* Calculate the number of tiles */
	unsigned int number_of_state_tiles = number_of_state_tilings * number_of_state_tiles_per_state_tiling;
	
	/* Create array to store state tile indicies */
	unsigned int* state_tile_indices;
	state_tile_indices = malloc(sizeof(unsigned int) * number_of_state_tilings);
	for (i = 0; i < number_of_state_tilings; i++)
	{
		state_tile_indices[i] = 0;
	} // end of i loop
	
	/* Create array to store next state tile indicies */
	unsigned int* next_state_tile_indices;
	next_state_tile_indices = malloc(sizeof(unsigned int) * number_of_state_tilings);
	for (i = 0; i < number_of_state_tilings; i++)
	{
		next_state_tile_indices[i] = 0;
	} // end of i loop
	
	/*********************************************************************************************************/
	/********************************************* SETUP WEIGHTS *********************************************/
	/*********************************************************************************************************/
	
	/* Get max number of actions */
	unsigned int max_number_of_actions = 0;
	for (i = 0; i < number_of_states; i++)
	{
		if (number_of_actions_per_state[i] > max_number_of_actions)
		{
			max_number_of_actions = number_of_actions_per_state[i];
		}
	} // end of i loop
	
	/* Set the number of features */
	unsigned int number_of_weights = number_of_state_tiles * max_number_of_actions;
	
	/* Create our weights */
	double* weights;
	weights = malloc(sizeof(double) * number_of_weights);
	
	/* Initialize weights to zero */
	for (i = 0; i < number_of_weights; i++)
	{
		weights[i] = 0.0;
	} // end of i loop
	
	/*********************************************************************************************************/
	/**************************************** SETUP POLICY ITERATION *****************************************/
	/*********************************************************************************************************/
	
	/* Set the maximum episode length */
	unsigned int maximum_episode_length = 1000000;
	
	/* Create state-action-value function array */
	double* approximate_state_action_value_function;
	approximate_state_action_value_function = malloc(sizeof(double) * max_number_of_actions);
	for (i = 0; i < max_number_of_actions; i++)
	{
		approximate_state_action_value_function[i] = 0.0;
	} // end of i loop
	
	/* Create policy array */
	double* policy;
	policy = malloc(sizeof(double) * max_number_of_actions);
	for (i = 0; i < max_number_of_actions; i++)
	{
		policy[i] = 1.0 / max_number_of_actions;
	} // end of i loop
	
	/* Create policy cumulative sum array */
	double* policy_cumulative_sum;
	policy_cumulative_sum = malloc(sizeof(double) * max_number_of_actions);
	policy_cumulative_sum[0] = policy[0];
	for (i = 1; i < max_number_of_actions; i++)
	{
		policy_cumulative_sum[i] = policy_cumulative_sum[i - 1] + policy[i];
	} // end of i loop
	
	/* Set learning rate alpha */
	double alpha = 0.01;
	
	/* Set learning rate beta */
	double beta = 0.01;
	
	/* Set epsilon for our epsilon level of exploration */
	double epsilon = 0.1;
	
	/* Create average reward estimate in place of the discounting factor gamma */
	double average_reward_estimate = 0.0;
	
	/* Set random seed */
	srand(0);
	
	/*********************************************************************************************************/
	/****************************************** RUN POLICY CONTROL *****************************************/
	/*********************************************************************************************************/
	
	printf("\nInitial weights:\n");
	for (i = 0; i < number_of_weights; i++)
	{
		printf("%lf\t", weights[i]);
	} // end of i loop
	printf("\n");
	
	printf("\nInitial state-action-value function:\n");
	for (i = 0; i < number_of_states; i++)
	{
		printf("%u", i);
		GetTileIndices(number_of_state_tilings, number_of_state_tiles, state_double_variables[i], number_of_state_double_variables, state_int_variables[i], number_of_state_int_variables, state_tile_indices);
		
		for (j = 0; j < max_number_of_actions; j++)
		{
			printf("\t%lf", ApproximateStateActionValueFunction(number_of_state_tilings, number_of_state_tiles, state_tile_indices, j, weights));
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/* Create variable for initial state */
	unsigned int initial_state_index = 0;
	
	/* Initialize episode to get initial state */
	initial_state_index = InitializeEpisode(number_of_states);
	
	/* Loop through episode and update the policy */
	LoopForeverThroughSteps(number_of_states, number_of_state_action_successor_states, state_action_successor_state_indices, state_action_successor_state_transition_probabilities_cumulative_sum, state_action_successor_state_rewards, max_number_of_actions, number_of_state_tilings, number_of_state_tiles, state_double_variables, number_of_state_double_variables, state_int_variables, number_of_state_int_variables, state_tile_indices, next_state_tile_indices, weights, approximate_state_action_value_function, policy, policy_cumulative_sum, alpha, beta, epsilon, average_reward_estimate, maximum_episode_length, initial_state_index);
	
	/*********************************************************************************************************/
	/**************************************** PRINT VALUES AND POLICIES ****************************************/
	/*********************************************************************************************************/
	
	printf("\nFinal weights:\n");
	for (i = 0; i < number_of_weights; i++)
	{
		printf("%lf\t", weights[i]);
	} // end of i loop
	printf("\n");
	
	printf("\nFinal state-action-value function:\n");
	for (i = 0; i < number_of_states; i++)
	{
		printf("%u", i);
		GetTileIndices(number_of_state_tilings, number_of_state_tiles, state_double_variables[i], number_of_state_double_variables, state_int_variables[i], number_of_state_int_variables, state_tile_indices);
		
		for (j = 0; j < max_number_of_actions; j++)
		{
			printf("\t%lf", ApproximateStateActionValueFunction(number_of_state_tilings, number_of_state_tiles, state_tile_indices, j, weights));
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/*********************************************************************************************************/
	/****************************************** FREE DYNAMIC MEMORY ******************************************/
	/*********************************************************************************************************/
	
	/* Free policy iteration arrays */
	free(policy_cumulative_sum);
	free(policy);
	free(approximate_state_action_value_function);
	
	/* Free weight arrays */
	free(weights);
	
	/* Free tiling arrays */
	free(next_state_tile_indices);
	free(state_tile_indices);
	
	for (i = 0; i < number_of_states; i++)
	{
		free(state_int_variables[i]);
		free(state_double_variables[i]);
	} // end of i loop
	
	free(state_int_variables);
	free(state_double_variables);

	/* Free environment arrays */
	for (i = 0; i < number_of_states; i++)
	{
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			free(state_action_successor_state_rewards[i][j]);
			free(state_action_successor_state_transition_probabilities_cumulative_sum[i][j]);
			free(state_action_successor_state_transition_probabilities[i][j]);
			free(state_action_successor_state_indices[i][j]);
		} // end of j loop
		free(state_action_successor_state_rewards[i]);
		free(state_action_successor_state_transition_probabilities_cumulative_sum[i]);
		free(state_action_successor_state_transition_probabilities[i]);
		free(state_action_successor_state_indices[i]);
		free(number_of_state_action_successor_states[i]);
	} // end of i loop
	free(state_action_successor_state_rewards);
	free(state_action_successor_state_transition_probabilities_cumulative_sum);
	free(state_action_successor_state_transition_probabilities);
	free(state_action_successor_state_indices);
	free(number_of_state_action_successor_states);
	free(number_of_actions_per_state);
	
	return 0;
} // end of main

/*********************************************************************************************************/
/*********************************************** FUNCTIONS ***********************************************/
/*********************************************************************************************************/

/* This function gets the tile indicies for each tiling */
void GetTileIndices(unsigned int num_tilings, unsigned int memory_size, double* doubles, unsigned int num_doubles, int* ints, unsigned int num_ints, unsigned int* tile_indices)
{
	unsigned int i, j;
	int qstate[MAX_NUM_VARS];
	int base[MAX_NUM_VARS];
	int coordinates[MAX_NUM_VARS * 2 + 1];   /* one interval number per relevant dimension */
	unsigned int num_coordinates = num_doubles + num_ints + 1;
	
	for (i = 0; i < num_ints; i++)
	{
		coordinates[num_doubles + 1 + i] = ints[i];
	} // end of i loop

	/* Quantize state to integers (henceforth, tile widths == num_tilings) */
	for (i = 0; i < num_doubles; i++)
	{
		qstate[i] = (int)floor(doubles[i] * num_tilings);
		base[i] = 0;
	}

	/* Compute the tile numbers */
	for (j = 0; j < num_tilings; j++)
	{
		/* Loop over each relevant dimension */
		for (i = 0; i < num_doubles; i++)
		{
			/* Find coordinates of activated tile in tiling space */
			coordinates[i] = qstate[i] - ModuloNegativeSafe(qstate[i] - base[i], num_tilings);

			/* Compute displacement of next tiling in quantized space */
			base[i] += 1 + (2 * i);
		} // end of i loop
		
		/* Add additional indices for tiling and hashing_set so they hash differently */
		coordinates[i] = j;

		tile_indices[j] = HashTiles(coordinates, num_coordinates, memory_size, 449);
	} // end of j loop
	
	return;
} // end of GetTileIndices function

/* This function takes the modulo of n by k even when n is negative */
int ModuloNegativeSafe(int n, int k)
{
	return (n >= 0) ? n % k : k - 1 - ((-n - 1) % k);
} // end of ModuloNegativeSafe function

/* This function takes an array of integers and returns the corresponding tile after hashing */
int HashTiles(int* ints, unsigned int num_ints, long m, int increment)
{
    static unsigned int rndseq[2048];
    static int first_call =  1;
    unsigned int i, k;
    long index;
    long sum = 0;

    /* If first call to hashing, initialize table of random numbers */
    if (first_call)
    {
        for (k = 0; k < 2048; k++)
        {
            rndseq[k] = 0;
            for (i = 0; i < (int)sizeof(int); ++i)
            {
                rndseq[k] = (rndseq[k] << 8) | (rand() & 0xff);
            } /// end of i loop
        } // end of k loop
        first_call = 0;
    }

    for (i = 0; i < num_ints; i++)
    {
        /* Add random table offset for this dimension and wrap around */
        index = ints[i];
        index += (increment * i);
        index %= 2048;
        while (index < 0)
        {
            index += 2048;
        }

        /* Add selected random number to sum */
        sum += (long)rndseq[(int)index];
    } // end of i loop
    
    index = (int)(sum % m);
    while (index < 0)
    {
        index += m;
    }

    return(index);
} // end of HashTiles function

/* This function initializes episodes */
unsigned int InitializeEpisode(unsigned int number_of_states)
{
	unsigned int initial_state_index;
	
	/* Initial state */
	initial_state_index = rand() % number_of_states; // randomly choose an initial state from all states
	
	return initial_state_index;
} // end of InitializeEpisode function

/* This function selects a policy with using epsilon-greedy from the approximate state-action-value function */
void EpsilonGreedyPolicyFromApproximateStateActionFunction(unsigned int max_number_of_actions, unsigned int number_of_state_tilings, unsigned int number_of_state_tiles, unsigned int* state_tile_indices, double* weights, double* approximate_state_action_value_function, double* policy, double* policy_cumulative_sum, double epsilon)
{
	unsigned int i, max_action_count = 1;
	double max_state_action_value = -DBL_MAX, max_policy_apportioned_probability_per_action = 1.0, remaining_apportioned_probability_per_action = 0.0;
	
	/* Update policy greedily from state-value function */
	for (i = 0; i < max_number_of_actions; i++)
	{
		/* Save max state action value and find the number of actions that have the same max state action value */
		approximate_state_action_value_function[i] = ApproximateStateActionValueFunction(number_of_state_tilings, number_of_state_tiles, state_tile_indices, i, weights);
		
		if (approximate_state_action_value_function[i] > max_state_action_value)
		{
			max_state_action_value = approximate_state_action_value_function[i];
			max_action_count = 1;
		}
		else if (approximate_state_action_value_function[i] == max_state_action_value)
		{
			max_action_count++;
		}
	} // end of i loop

	/* Apportion policy probability across ties equally for state-action pairs that have the same value and spread out epsilon otherwise */
	if (max_action_count == max_number_of_actions)
	{
		max_policy_apportioned_probability_per_action = 1.0 / max_action_count;
		remaining_apportioned_probability_per_action = 0.0;
	}
	else
	{
		max_policy_apportioned_probability_per_action = (1.0 - epsilon) / max_action_count;
		remaining_apportioned_probability_per_action = epsilon / (max_number_of_actions - max_action_count);
	}
	
	/* Update policy with our apportioned probabilities */
	for (i = 0; i < max_number_of_actions; i++)
	{
		if (approximate_state_action_value_function[i] == max_state_action_value)
		{
			policy[i] = max_policy_apportioned_probability_per_action;
		}
		else
		{
			policy[i] = remaining_apportioned_probability_per_action;
		}
	} // end of i loop
	
	/* Update policy cumulative sum */
	policy_cumulative_sum[0] = policy[0];
	for (i = 1; i < max_number_of_actions; i++)
	{
		policy_cumulative_sum[i] = policy_cumulative_sum[i - 1] + policy[i];
	} // end of i loop
	
	return;
} // end of EpsilonGreedyPolicyFromStateActionFunction function

/* This function loops forever through steps and updates the policy */
void LoopForeverThroughSteps(unsigned int number_of_states, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, unsigned int max_number_of_actions, unsigned int number_of_state_tilings, unsigned int number_of_state_tiles, double** state_double_variables, unsigned int number_of_state_double_variables, int** state_int_variables, unsigned int number_of_state_int_variables, unsigned int* state_tile_indices, unsigned int* next_state_tile_indices, double* weights, double* approximate_state_action_value_function, double* policy, double* policy_cumulative_sum, double alpha, double beta, double epsilon, double average_reward_estimate, unsigned int maximum_episode_length, unsigned int state_index)
{
	unsigned int i, j;
	unsigned int action_index, successor_state_transition_index, next_state_index;
	double probability, reward, delta, state_value_function_expected_value_on_policy;
		
	/* Loop forever (or at least until our maximum) */
	for (i = 0; i < maximum_episode_length; i++)
	{
		/* Get tiled feature indices of state */
		GetTileIndices(number_of_state_tilings, number_of_state_tiles, state_double_variables[state_index], number_of_state_double_variables, state_int_variables[state_index], number_of_state_int_variables, state_tile_indices);
		
		/* Get action */
		probability = UnifRand();
		
		/* Choose policy for chosen state by epsilon-greedy choosing from the state-action-value function */
		EpsilonGreedyPolicyFromApproximateStateActionFunction(max_number_of_actions, number_of_state_tilings, number_of_state_tiles, state_tile_indices, weights, approximate_state_action_value_function, policy, policy_cumulative_sum, epsilon);
		
		/* Find which action using probability */
		for (j = 0; j < max_number_of_actions; j++)
		{
			if (probability <= policy_cumulative_sum[j])
			{
				action_index = j;
				break; // break j loop since we found our index
			}
		} // end of i loop
		
		/* Get reward */
		probability = UnifRand();
		
		for (j = 0; j < number_of_state_action_successor_states[state_index][action_index]; j++)
		{
			if (probability <= state_action_successor_state_transition_probabilities_cumulative_sum[state_index][action_index][j])
			{
				successor_state_transition_index = j;
				break; // break j loop since we found our index
			}
		} // end of j loop
		
		/* Get reward from state and action */
		reward = state_action_successor_state_rewards[state_index][action_index][successor_state_transition_index];
		
		/* Get next state */
		next_state_index = state_action_successor_state_indices[state_index][action_index][successor_state_transition_index];
		
		/* Get tiled feature indices of next state */
		GetTileIndices(number_of_state_tilings, number_of_state_tiles, state_double_variables[next_state_index], number_of_state_double_variables, state_int_variables[next_state_index], number_of_state_int_variables, next_state_tile_indices);
		
		/* Choose policy for chosen state by epsilon-greedy choosing from the state-action-value function */
		EpsilonGreedyPolicyFromApproximateStateActionFunction(max_number_of_actions, number_of_state_tilings, number_of_state_tiles, next_state_tile_indices, weights, approximate_state_action_value_function, policy, policy_cumulative_sum, epsilon);

		/* Get next action, using expectation value */
		state_value_function_expected_value_on_policy = 0.0;
		
		for (j = 0; j < max_number_of_actions; j++)
		{
			state_value_function_expected_value_on_policy += policy[j] * ApproximateStateActionValueFunction(number_of_state_tilings, number_of_state_tiles, next_state_tile_indices, j, weights);
		} // end of j loop
		
		/* Calculate TD error delta */
		delta = reward - average_reward_estimate + state_value_function_expected_value_on_policy - ApproximateStateActionValueFunction(number_of_state_tilings, number_of_state_tiles, state_tile_indices, action_index, weights);
		
		/* Update average reward estimate */
		average_reward_estimate += beta * delta;
		
		/* Update weights */
		for (j = 0; j < number_of_state_tilings; j++)
		{
			weights[action_index * number_of_state_tiles + state_tile_indices[j]] += alpha * delta; // since this is linear grad(q(S, A, w)) = x(S, A, w) which is 1 for tiled index otherwise 0
		} // end of j loop
		
		/* Update state and action to next state and action */
		state_index = next_state_index;
	} // end of i loop
	
	return;
} // end of LoopForeverThroughSteps function

/* This function calculates the approximate state action value function w^T * x */
double ApproximateStateActionValueFunction(unsigned int number_of_state_tilings, unsigned int number_of_state_tiles, unsigned int* state_tile_indices, unsigned int action_index, double* weights)
{
	unsigned int i;
	double approximate_state_action_value_function = 0.0;
	
	for (i = 0; i < number_of_state_tilings; i++)
	{
		approximate_state_action_value_function += weights[action_index * number_of_state_tiles + state_tile_indices[i]];
	} // end of i loop
	
	return approximate_state_action_value_function;
} // end of ApproximateStateActionValueFunction function

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function