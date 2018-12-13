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

/* This function creates the feature vector */
void CreateFeatureVector(unsigned int number_of_state_tilings, unsigned int* state_tile_indices, unsigned int number_of_features, double* feature_vector);

/* This function resets the feature vector */
void ResetFeatureVector(unsigned int number_of_features, double* feature_vector);

/* This function initializes episodes */
unsigned int InitializeEpisode(unsigned int number_of_non_terminal_states, double* gamma_t, unsigned int number_of_policy_weights, double* policy_eligibility_trace, unsigned int number_of_features, double* state_value_eligibility_trace);

/* This function selects an action in current state */
unsigned int SelectAction(unsigned int max_number_of_actions, unsigned int number_of_features, double* feature_vector, double* policy_weights, double* policy, double* policy_cumulative_sum);

/* This function observes the reward from the environment by taking action in current state */
double ObserveReward(unsigned int state_index, unsigned int action_index, unsigned int* successor_state_transition_index, unsigned int** number_of_state_action_successor_states, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards);

/* This function loops through episodes and updates the policy */
void LoopThroughEpisode(unsigned int number_of_non_terminal_states, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, unsigned int max_number_of_actions, unsigned int number_of_state_tilings, unsigned int number_of_state_tiles, double** state_double_variables, unsigned int number_of_state_double_variables, int** state_int_variables, unsigned int number_of_state_int_variables, unsigned int* state_tile_indices, unsigned int number_of_features, double* feature_vector, double* next_feature_vector, unsigned int number_of_policy_weights, double* policy_weights, double* state_value_weights, double* policy, double* policy_cumulative_sum, double* policy_eligibility_trace, double* state_value_eligibility_trace, double policy_weight_alpha, double state_value_weight_alpha, double discounting_factor_gamma, double gamma_t, double policy_trace_decay_lambda, double state_value_trace_decay_lambda, unsigned int maximum_episode_length, unsigned int state_index);

/* This function calculates the approximate state-value function w^T * x */
double ApproximateStateValueFunction(unsigned int number_of_features, double* feature_vector, double* state_value_weights);

/* This function calculates the approximate policy pi(a | s, theta) = Softmax(theta^T * x(s, a)) */
void ApproximatePolicy(unsigned int max_number_of_actions, unsigned int number_of_features, double* feature_vector, double* policy_weights, double* policy, double* policy_cumulative_sum);

/* This function applies softmax activation function to given logits */
void ApplySoftmaxActivationFunction(unsigned int max_number_of_actions, double* policy);

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
	
	/* Get number of terminal states */
	unsigned int number_of_terminal_states = 0;
	
	FILE* infile_number_of_terminal_states = fopen("inputs/number_of_terminal_states.txt", "r");
	system_return = fscanf(infile_number_of_terminal_states, "%u", &number_of_terminal_states);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/number_of_terminal_states.txt\n");
	}
	fclose(infile_number_of_terminal_states);
	
	/* Get number of non-terminal states */
	unsigned int number_of_non_terminal_states = number_of_states - number_of_terminal_states;
	
	/* Get the number of actions per non-terminal state */
	unsigned int* number_of_actions_per_non_terminal_state;
	
	FILE* infile_number_of_actions_per_non_terminal_state = fopen("inputs/number_of_actions_per_non_terminal_state.txt", "r");
	number_of_actions_per_non_terminal_state = malloc(sizeof(int) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		system_return = fscanf(infile_number_of_actions_per_non_terminal_state, "%u", &number_of_actions_per_non_terminal_state[i]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/number_of_actions_per_non_terminal_state.txt\n");
		}
	} // end of i loop
	fclose(infile_number_of_actions_per_non_terminal_state);
	
	/* Get the number of actions per all states */
	unsigned int* number_of_actions_per_state;
	
	number_of_actions_per_state = malloc(sizeof(int) * number_of_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		number_of_actions_per_state[i] = number_of_actions_per_non_terminal_state[i];
	} // end of i loop
	
	for (i = 0; i < number_of_terminal_states; i++)
	{
		number_of_actions_per_state[i + number_of_non_terminal_states] = 0;
	} // end of i loop
	
	/* Get the number of state-action successor states */
	unsigned int** number_of_state_action_successor_states;
	
	FILE* infile_number_of_state_action_successor_states = fopen("inputs/number_of_state_action_successor_states.txt", "r");
	number_of_state_action_successor_states = malloc(sizeof(int*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		number_of_state_action_successor_states[i] = malloc(sizeof(int) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
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
	state_action_successor_state_indices = malloc(sizeof(unsigned int**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		state_action_successor_state_indices[i] = malloc(sizeof(unsigned int*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
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
	state_action_successor_state_transition_probabilities = malloc(sizeof(double**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		state_action_successor_state_transition_probabilities[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
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
	
	state_action_successor_state_transition_probabilities_cumulative_sum = malloc(sizeof(double**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		state_action_successor_state_transition_probabilities_cumulative_sum[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
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
	state_action_successor_state_rewards = malloc(sizeof(double**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		state_action_successor_state_rewards[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
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
	
	/*********************************************************************************************************/
	/**************************************** SETUP PARAMETERIZATION *****************************************/
	/*********************************************************************************************************/
	
	/* Get max number of actions */
	unsigned int max_number_of_actions = 0;
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		if (number_of_actions_per_non_terminal_state[i] > max_number_of_actions)
		{
			max_number_of_actions = number_of_actions_per_non_terminal_state[i];
		}
	} // end of i loop
	
	/* Set the number of features */
	unsigned int number_of_features = number_of_state_tiles;
	
	/* Create our feature vector */
	double* feature_vector;
	feature_vector = malloc(sizeof(double) * number_of_features);
	
	ResetFeatureVector(number_of_features, feature_vector); // initialize all features to zero
	
	/* Create our next feature vector */
	double* next_feature_vector;
	next_feature_vector = malloc(sizeof(double) * number_of_features);
	
	ResetFeatureVector(number_of_features, next_feature_vector); // initialize all features to zero
	
	/* Set the number of policy weights */
	unsigned int number_of_policy_weights = number_of_features * max_number_of_actions;
	
	/* Create our policy_weights */
	double* policy_weights;
	policy_weights = malloc(sizeof(double) * number_of_policy_weights);
	
	/* Initialize policy weights to zero */
	for (i = 0; i < number_of_policy_weights; i++)
	{
		policy_weights[i] = 0.0;
	} // end of i loop
	
	/* Create our policy_weights */
	double* state_value_weights;
	state_value_weights = malloc(sizeof(double) * (number_of_features + 1));
	
	/* Initialize state-value weights to zero */
	for (i = 0; i < (number_of_features + 1); i++)
	{
		state_value_weights[i] = 0.0;
	} // end of i loop
	
	/*********************************************************************************************************/
	/**************************************** SETUP POLICY ITERATION *****************************************/
	/*********************************************************************************************************/
	
	/* Set the number of episodes */
	unsigned int number_of_episodes = 100000;
	
	/* Set the maximum episode length */
	unsigned int maximum_episode_length = 200;
	
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
	
	/* Create eligibility trace for policy */
	double* policy_eligibility_trace;
	policy_eligibility_trace = malloc(sizeof(double) * number_of_policy_weights);
	for (i = 0; i < number_of_policy_weights; i++)
	{
		policy_eligibility_trace[i] = 0.0;
	} // end of i loop

	/* Create eligibility trace for state-values */
	double* state_value_eligibility_trace;
	state_value_eligibility_trace = malloc(sizeof(double) * (number_of_features + 1));
	for (i = 0; i < (number_of_features + 1); i++)
	{
		state_value_eligibility_trace[i] = 0.0;
	} // end of i loop
	
	/* Set policy weight learning rate alpha */
	double policy_weight_alpha = 0.001;
	
	/* Set state-value weight learning rate alpha */
	double state_value_weight_alpha = 0.001;
	
	/* Set discounting factor gamma */
	double discounting_factor_gamma = 1.0;
	
	/* Create discounting factor gamma series tracker */
	double gamma_t = 1.0;
	
	/* Set policy trace decay parameter lambda */
	double policy_trace_decay_lambda = 0.9;
	
	/* Set state-value trace decay parameter lambda */
	double state_value_trace_decay_lambda = 0.9;
	
	/* Set random seed */
	srand(0);
	
	/*********************************************************************************************************/
	/******************************************* RUN POLICY CONTROL ******************************************/
	/*********************************************************************************************************/
	
	printf("\nInitial policy weights:\n");
	for (i = 0; i < number_of_policy_weights; i++)
	{
		printf("%lf\t", policy_weights[i]);
	} // end of i loop
	printf("\n");
	
	printf("\nInitial state-value weights:\n");
	for (i = 0; i < (number_of_features + 1); i++)
	{
		printf("%lf\t", state_value_weights[i]);
	} // end of i loop
	printf("\n");
	
	printf("\nInitial state-value function:\n");
	for (i = 0; i < number_of_states; i++)
	{
		printf("%u", i);
		GetTileIndices(number_of_state_tilings, number_of_state_tiles, state_double_variables[i], number_of_state_double_variables, state_int_variables[i], number_of_state_int_variables, state_tile_indices);
		
		CreateFeatureVector(number_of_state_tilings, state_tile_indices, number_of_features, feature_vector);
		
		printf("\t%lf\n", ApproximateStateValueFunction(number_of_features, feature_vector, state_value_weights));
	} // end of i loop
	printf("\n");
	
	printf("\nInitial policy:\n");
	for (i = 0; i < number_of_states; i++)
	{
		printf("%u", i);
		GetTileIndices(number_of_state_tilings, number_of_state_tiles, state_double_variables[i], number_of_state_double_variables, state_int_variables[i], number_of_state_int_variables, state_tile_indices);
		
		CreateFeatureVector(number_of_state_tilings, state_tile_indices, number_of_features, feature_vector);
		
		ApproximatePolicy(max_number_of_actions, number_of_features, feature_vector, policy_weights, policy, policy_cumulative_sum);
		
		for (j = 0; j < max_number_of_actions; j++)
		{
			printf("\t%lf", policy[j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/* Create variable for episode's initial state */
	unsigned int initial_state_index = 0;
	
	/* Loop over episodes */
	for (i = 0; i < number_of_episodes; i++)
	{
		/* Initialize episode */
		initial_state_index = InitializeEpisode(number_of_non_terminal_states, &gamma_t, number_of_policy_weights, policy_eligibility_trace, number_of_features, state_value_eligibility_trace);
		
		/* Loop through episode and update the policy */
		LoopThroughEpisode( number_of_non_terminal_states, number_of_state_action_successor_states, state_action_successor_state_indices, state_action_successor_state_transition_probabilities_cumulative_sum, state_action_successor_state_rewards, max_number_of_actions, number_of_state_tilings, number_of_state_tiles, state_double_variables, number_of_state_double_variables, state_int_variables, number_of_state_int_variables, state_tile_indices, number_of_features, feature_vector, next_feature_vector, number_of_policy_weights, policy_weights, state_value_weights, policy, policy_cumulative_sum, policy_eligibility_trace, state_value_eligibility_trace, policy_weight_alpha, state_value_weight_alpha, discounting_factor_gamma, gamma_t, policy_trace_decay_lambda, state_value_trace_decay_lambda, maximum_episode_length, initial_state_index);
	} // end of i loop
	
	/*********************************************************************************************************/
	/*************************************** PRINT VALUES AND POLICIES ***************************************/
	/*********************************************************************************************************/
	
	printf("\nFinal policy weights:\n");
	for (i = 0; i < number_of_policy_weights; i++)
	{
		printf("%lf\t", policy_weights[i]);
	} // end of i loop
	printf("\n");
	
	printf("\nFinal state-value weights:\n");
	for (i = 0; i < (number_of_features + 1); i++)
	{
		printf("%lf\t", state_value_weights[i]);
	} // end of i loop
	printf("\n");
	
	printf("\nFinal state-value function:\n");
	for (i = 0; i < number_of_states; i++)
	{
		printf("%u", i);
		GetTileIndices(number_of_state_tilings, number_of_state_tiles, state_double_variables[i], number_of_state_double_variables, state_int_variables[i], number_of_state_int_variables, state_tile_indices);
		
		CreateFeatureVector(number_of_state_tilings, state_tile_indices, number_of_features, feature_vector);
		
		printf("\t%lf\n", ApproximateStateValueFunction(number_of_features, feature_vector, state_value_weights));
	} // end of i loop
	printf("\n");
	
	printf("\nFinal policy:\n");
	for (i = 0; i < number_of_states; i++)
	{
		printf("%u", i);
		GetTileIndices(number_of_state_tilings, number_of_state_tiles, state_double_variables[i], number_of_state_double_variables, state_int_variables[i], number_of_state_int_variables, state_tile_indices);
		
		CreateFeatureVector(number_of_state_tilings, state_tile_indices, number_of_features, feature_vector);
		
		ApproximatePolicy(max_number_of_actions, number_of_features, feature_vector, policy_weights, policy, policy_cumulative_sum);
		
		for (j = 0; j < max_number_of_actions; j++)
		{
			printf("\t%lf", policy[j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/*********************************************************************************************************/
	/****************************************** FREE DYNAMIC MEMORY ******************************************/
	/*********************************************************************************************************/
	
	/* Free policy iteration arrays */
	free(state_value_eligibility_trace);
	free(policy_eligibility_trace);
	free(policy_cumulative_sum);
	free(policy);
	
	/* Free parameterization arrays */
	free(state_value_weights);
	free(policy_weights);
	free(next_feature_vector);
	free(feature_vector);
	
	/* Free tiling arrays */
	free(state_tile_indices);
	
	for (i = 0; i < number_of_states; i++)
	{
		free(state_int_variables[i]);
		free(state_double_variables[i]);
	} // end of i loop
	
	free(state_int_variables);
	free(state_double_variables);

	/* Free environment arrays */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
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
	free(number_of_actions_per_non_terminal_state);
	
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

/* This function creates the feature vector */
void CreateFeatureVector(unsigned int number_of_state_tilings, unsigned int* state_tile_indices, unsigned int number_of_features, double* feature_vector)
{
	unsigned int i;
	
	/* First reset the feature vector to all zeros */
	ResetFeatureVector(number_of_features, feature_vector);
	
	for (i = 0; i < number_of_state_tilings; i++)
	{
		feature_vector[state_tile_indices[i]] = 1.0;
	} // end of i loop
	
	return;
} // end of CreateFeatureVector function

/* This function resets the feature vector */
void ResetFeatureVector(unsigned int number_of_features, double* feature_vector)
{
	unsigned int i;
	
	for (i = 0; i < number_of_features; i++)
	{
		feature_vector[i] = 0.0;
	} // end of i loop
	
	return;
} // end of ResetFeatureVector function

/* This function initializes episodes */
unsigned int InitializeEpisode(unsigned int number_of_non_terminal_states, double* gamma_t, unsigned int number_of_policy_weights, double* policy_eligibility_trace, unsigned int number_of_features, double* state_value_eligibility_trace)
{
	unsigned int i, state_index;
	
	/* Initial state */
	state_index = rand() % number_of_non_terminal_states; // randomly choose an initial state from all non-terminal states
	
	/* Reset discounting factor gamma series tracker */
	(*gamma_t) = 1.0;
	
	for (i = 0; i < number_of_policy_weights; i++)
	{
		policy_eligibility_trace[i] = 0.0;
	} // end of i loop
	
	for (i = 0; i < (number_of_features + 1); i++)
	{
		state_value_eligibility_trace[i] = 0.0;
	} // end of i loop
	
	return state_index;
} // end of InitializeEpisode function

/* This function selects an action in current state */
unsigned int SelectAction(unsigned int max_number_of_actions, unsigned int number_of_features, double* feature_vector, double* policy_weights, double* policy, double* policy_cumulative_sum)
{
	unsigned int i, action_index;
	double probability;
	
	/* Approximate policy for current state */
	ApproximatePolicy(max_number_of_actions, number_of_features, feature_vector, policy_weights, policy, policy_cumulative_sum);
	
	probability = UnifRand();
	
	/* Find which action using probability */
	for (i = 0; i < max_number_of_actions; i++)
	{
		if (probability <= policy_cumulative_sum[i])
		{
			action_index = i;
			
			break; // break i loop since we found our index
		}
	} // end of i loop
		
	return action_index;
} // end of SelectAction function

/* This function observes the reward from the environment by taking action in current state */
double ObserveReward(unsigned int state_index, unsigned int action_index, unsigned int* successor_state_transition_index, unsigned int** number_of_state_action_successor_states, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards)
{
	unsigned int i;
	double probability, reward;
	
	probability = UnifRand();
	
	/* Find which successor state using probability */
	for (i = 0; i < number_of_state_action_successor_states[state_index][action_index]; i++)
	{
		if (probability <= state_action_successor_state_transition_probabilities_cumulative_sum[state_index][action_index][i])
		{
			(*successor_state_transition_index) = i;
			
			break; // break i loop since we found our index
		}
	} // end of i loop
	
	/* Get reward from state and action */
	reward = state_action_successor_state_rewards[state_index][action_index][(*successor_state_transition_index)];
	
	return reward;
} // end of ObserveReward function

/* This function loops through episodes and updates the policy */
void LoopThroughEpisode(unsigned int number_of_non_terminal_states, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, unsigned int max_number_of_actions, unsigned int number_of_state_tilings, unsigned int number_of_state_tiles, double** state_double_variables, unsigned int number_of_state_double_variables, int** state_int_variables, unsigned int number_of_state_int_variables, unsigned int* state_tile_indices, unsigned int number_of_features, double* feature_vector, double* next_feature_vector, unsigned int number_of_policy_weights, double* policy_weights, double* state_value_weights, double* policy, double* policy_cumulative_sum, double* policy_eligibility_trace, double* state_value_eligibility_trace, double policy_weight_alpha, double state_value_weight_alpha, double discounting_factor_gamma, double gamma_t, double policy_trace_decay_lambda, double state_value_trace_decay_lambda, unsigned int maximum_episode_length, unsigned int state_index)
{
	unsigned int t, i, j, action_index, successor_state_transition_index, next_state_index;
	double probability, reward, delta;
		
	/* Loop through episode steps until termination */
	for (t = 0; t < maximum_episode_length; t++)
	{
		/* Get tiled feature indices of state */
		GetTileIndices(number_of_state_tilings, number_of_state_tiles, state_double_variables[state_index], number_of_state_double_variables, state_int_variables[state_index], number_of_state_int_variables, state_tile_indices);
		
		/* Create feature vector using state feature tilings */
		CreateFeatureVector(number_of_state_tilings, state_tile_indices, number_of_features, feature_vector);

		/* Get action */
		action_index = SelectAction(max_number_of_actions, number_of_features, feature_vector, policy_weights, policy, policy_cumulative_sum);
		
		/* Get reward */
		reward = ObserveReward(state_index, action_index, &successor_state_transition_index, number_of_state_action_successor_states, state_action_successor_state_transition_probabilities_cumulative_sum, state_action_successor_state_rewards);
		
		/* Get next state */
		next_state_index = state_action_successor_state_indices[state_index][action_index][successor_state_transition_index];
		
		/* Calculate TD error delta */
		delta = reward - ApproximateStateValueFunction(number_of_features, feature_vector, state_value_weights);
		
		/* Check to see if we actioned into a terminal state */
		if (next_state_index >= number_of_non_terminal_states)
		{
			/* v_hat(S', w) = 0 */
		}
		else
		{
			/* Get tiled feature indices of next state */
			GetTileIndices(number_of_state_tilings, number_of_state_tiles, state_double_variables[next_state_index], number_of_state_double_variables, state_int_variables[next_state_index], number_of_state_int_variables, state_tile_indices);
			
			/* Create next feature vector using next state feature tilings */
			CreateFeatureVector(number_of_state_tilings, state_tile_indices, number_of_features, next_feature_vector);
			
			/* Update TD error delta */
			delta += discounting_factor_gamma * ApproximateStateValueFunction(number_of_features, next_feature_vector, state_value_weights);
		}
		
		/* Update state-value eligibility traces */
		for (i = 0; i < number_of_features; i++)
		{
			state_value_eligibility_trace[i] = discounting_factor_gamma * state_value_trace_decay_lambda * state_value_eligibility_trace[i] + feature_vector[i];
		} // end of i loop
		state_value_eligibility_trace[number_of_features] = discounting_factor_gamma * state_value_trace_decay_lambda * state_value_eligibility_trace[number_of_features] + 1.0; // bias feature term
		
		/* Update state-value weights */
		for (i = 0; i < (number_of_features + 1); i++)
		{
			state_value_weights[i] += state_value_weight_alpha * delta * state_value_eligibility_trace[i];
		} // end of i loop
		
		/* Approximate policy for current state */
		ApproximatePolicy(max_number_of_actions, number_of_features, feature_vector, policy_weights, policy, policy_cumulative_sum);
		
		/* Update policy weights */
		for (i = 0; i < max_number_of_actions; i++)
		{
			if (i == action_index)
			{
				for (j = 0; j < number_of_features; j++)
				{
					policy_eligibility_trace[i * number_of_features + j] = discounting_factor_gamma * policy_trace_decay_lambda * policy_eligibility_trace[i * number_of_features + j] + gamma_t * ((1.0 - policy[i]) * feature_vector[j]);
				} // end of j loop
			}
			else
			{
				for (j = 0; j < number_of_features; j++)
				{
					policy_eligibility_trace[i * number_of_features + j] = discounting_factor_gamma * policy_trace_decay_lambda * policy_eligibility_trace[i * number_of_features + j] + gamma_t * (-policy[i] * feature_vector[j]);
				} // end of j loop				
			}
		} // end of i loop
		
		/* Update policy weights */
		for (i = 0; i < number_of_policy_weights; i++)
		{
			policy_weights[i] += policy_weight_alpha * delta * policy_eligibility_trace[i];
		} // end of i loop
		
		/* Check to see if we actioned into a terminal state */
		if (next_state_index >= number_of_non_terminal_states)
		{
			break;
		}
		
		gamma_t *= discounting_factor_gamma;
		state_index = next_state_index;
	} // end of t loop
	
	return;
} // end of LoopThroughEpisode function

/* This function calculates the approximate state-value function w^T * x */
double ApproximateStateValueFunction(unsigned int number_of_features, double* feature_vector, double* state_value_weights)
{
	unsigned int i;
	double approximate_state_value_function = 0.0;
	
	for (i = 0; i < number_of_features; i++)
	{
		approximate_state_value_function += state_value_weights[i] * feature_vector[i];
	} // end of i loop
	
	approximate_state_value_function += state_value_weights[number_of_features];
	
	return approximate_state_value_function;
} // end of ApproximateStateActionValueFunction function

/* This function calculates the approximate policy pi(a | s, theta) = Softmax(theta^T * x(s, a)) */
void ApproximatePolicy(unsigned int max_number_of_actions, unsigned int number_of_features, double* feature_vector, double* policy_weights, double* policy, double* policy_cumulative_sum)
{
	unsigned int i, j;
	
	for (i = 0; i < max_number_of_actions; i++)
	{
		policy[i] = 0.0;
		
		for (j = 0; j < number_of_features; j++)
		{
			policy[i] += policy_weights[i * number_of_features + j] * feature_vector[j];
		} // end of j loop
	} // end of i loop
	
	ApplySoftmaxActivationFunction(max_number_of_actions, policy);
	
	/* Calculate policy cumulative sum */
	policy_cumulative_sum[0] = policy[0];
	for (i = 1; i < max_number_of_actions; i++)
	{
		policy_cumulative_sum[i] = policy_cumulative_sum[i - 1] + policy[i];
	} // end of i loop
	
	return;
} // end of ApproximateStateActionValueFunction function

/* This function applies softmax activation function to given logits */
void ApplySoftmaxActivationFunction(unsigned int max_number_of_actions, double* policy)
{
	/* f(xi) = e^(xi - max(x)) / sum(e^(xj - max(x)), j, 0, n - 1) */

	unsigned int i;
	double max_logit, denominator_sum;

	max_logit = -DBL_MAX;
	for (i = 0; i < max_number_of_actions; i++)
	{
		if (policy[i] > max_logit)
		{
			max_logit = policy[i];
		}
	} // end of i loop

	denominator_sum = 0.0;
	for (i = 0; i < max_number_of_actions; i++)
	{
		/* Shift logits by the max logit to make numerically stable */
		policy[i] = exp(policy[i] - max_logit);
		denominator_sum += policy[i];
	} // end of i loop

	for (i = 0; i < max_number_of_actions; i++)
	{
		policy[i] /= denominator_sum;
	} // end of i loop
	
	return;
} // end of ApplySoftmaxActivationFunction function

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function