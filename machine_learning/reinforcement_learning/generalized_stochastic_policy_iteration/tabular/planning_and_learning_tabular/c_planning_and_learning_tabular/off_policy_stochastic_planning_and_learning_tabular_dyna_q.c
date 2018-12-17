#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function initializes episodes */
unsigned int InitializeEpisode(unsigned int number_of_non_terminal_states);

/* This function selects a policy with using epsilon-greedy from the state-action-value function */
void EpsilonGreedyPolicyFromStateActionFunction(unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function, double epsilon, unsigned int state_index, double** policy, double** policy_cumulative_sum);

/* This function loops through episodes and updates the policy */
void LoopThroughEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** environment_number_of_state_action_successor_states, unsigned int*** environment_state_action_successor_state_indices, double*** environment_state_action_successor_state_transition_probabilities_cumulative_sum, double*** environment_state_action_successor_state_rewards, unsigned int* model_number_of_seen_non_terminal_states, unsigned int* model_seen_non_terminal_states_stack, unsigned int* model_seen_non_terminal_states_stack_reverse_lookup, unsigned int* model_number_of_seen_non_terminal_states_actions, unsigned int** model_seen_non_terminal_states_actions_stack, unsigned int** model_seen_non_terminal_states_actions_stack_reverse_lookup, unsigned int** model_number_of_state_action_successor_states, unsigned int**** model_state_action_successor_state_indices, double**** model_state_action_successor_state_transition_probabilities, double**** model_state_action_successor_state_transition_probabilities_cumulative_sum, double**** model_state_action_successor_state_rewards, unsigned int**** model_state_action_successor_state_number_of_visits, double** state_action_value_function, unsigned int** state_action_value_function_max_tie_stack, double** policy, double** policy_cumulative_sum, double alpha, double epsilon, double discounting_factor_gamma, unsigned int maximum_episode_length, unsigned int number_of_planning_steps, unsigned int state_index);

/* This function updates what state and actions the model has seen */
void UpdateModelSeenStateActions(unsigned int state_index, unsigned int action_index, unsigned int* model_number_of_seen_non_terminal_states, unsigned int* model_seen_non_terminal_states_stack, unsigned int* model_seen_non_terminal_states_stack_reverse_lookup, unsigned int* model_number_of_seen_non_terminal_states_actions, unsigned int** model_seen_non_terminal_states_actions_stack, unsigned int** model_seen_non_terminal_states_actions_stack_reverse_lookup);

/* This function updates the model from environment experience */
void UpdateModelOfEnvironmentFromExperience(unsigned int state_index, unsigned int action_index, double reward, unsigned int next_state_index, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** model_number_of_state_action_successor_states, unsigned int**** model_state_action_successor_state_indices, double**** model_state_action_successor_state_transition_probabilities, double**** model_state_action_successor_state_transition_probabilities_cumulative_sum, double**** model_state_action_successor_state_rewards, unsigned int**** model_state_action_successor_state_number_of_visits);

/* This function reallocs model state action successor state arrays */
void ReallocModelStateActionSuccessorStateArrays(unsigned int state_index, unsigned int action_index, double reward, unsigned int next_state_index, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** model_number_of_state_action_successor_states, unsigned int**** model_state_action_successor_state_indices, double**** model_state_action_successor_state_transition_probabilities, double**** model_state_action_successor_state_transition_probabilities_cumulative_sum, double**** model_state_action_successor_state_rewards, unsigned int**** model_state_action_successor_state_number_of_visits);

/* This function uses model to plan via simulate experience */
void ModelSimualatePlanning(unsigned int number_of_planning_steps, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int model_number_of_seen_non_terminal_states, unsigned int* model_seen_non_terminal_states_stack, unsigned int* model_seen_non_terminal_states_stack_reverse_lookup, unsigned int* model_number_of_seen_non_terminal_states_actions, unsigned int** model_seen_non_terminal_states_actions_stack, unsigned int** model_seen_non_terminal_states_actions_stack_reverse_lookup, unsigned int** model_number_of_state_action_successor_states, unsigned int*** model_state_action_successor_state_indices, double*** model_state_action_successor_state_transition_probabilities, double*** model_state_action_successor_state_transition_probabilities_cumulative_sum, double*** model_state_action_successor_state_rewards, unsigned int*** model_state_action_successor_state_number_of_visits, unsigned int** state_action_value_function_max_tie_stack, double alpha, double discounting_factor_gamma, double** state_action_value_function);

/* This function reallocates more memory to the passed 3d unsigned int array */
void Realloc3dUnsignedInt(unsigned int**** array3d, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** model_number_of_state_action_successor_states, unsigned int state_index, unsigned int action_index);

/* This function reallocates more memory to the passed 3d double array */
void Realloc3dDouble(double**** array3d, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** model_number_of_state_action_successor_states, unsigned int state_index, unsigned int action_index);

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
	number_of_actions_per_non_terminal_state = malloc(sizeof(unsigned int) * number_of_non_terminal_states);
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
	
	number_of_actions_per_state = malloc(sizeof(unsigned int) * number_of_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		number_of_actions_per_state[i] = number_of_actions_per_non_terminal_state[i];
	} // end of i loop
	
	for (i = 0; i < number_of_terminal_states; i++)
	{
		number_of_actions_per_state[i + number_of_non_terminal_states] = 0;
	} // end of i loop
	
	/* Get the number of state-action successor states */
	unsigned int** environment_number_of_state_action_successor_states;
	
	FILE* infile_number_of_state_action_successor_states = fopen("inputs/number_of_state_action_successor_states.txt", "r");
	environment_number_of_state_action_successor_states = malloc(sizeof(unsigned int*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		environment_number_of_state_action_successor_states[i] = malloc(sizeof(unsigned int) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			system_return = fscanf(infile_number_of_state_action_successor_states, "%u\t", &environment_number_of_state_action_successor_states[i][j]);
			if (system_return == -1)
			{
				printf("Failed reading file inputs/number_of_state_action_successor_states.txt\n");
			}
		} // end of j loop
	} // end of i loop
	fclose(infile_number_of_state_action_successor_states);
	
	/* Get the state-action-successor state indices */
	unsigned int*** environment_state_action_successor_state_indices;
	
	FILE* infile_state_action_successor_state_indices = fopen("inputs/state_action_successor_state_indices.txt", "r");
	environment_state_action_successor_state_indices = malloc(sizeof(unsigned int**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		environment_state_action_successor_state_indices[i] = malloc(sizeof(unsigned int*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			environment_state_action_successor_state_indices[i][j] = malloc(sizeof(unsigned int*) * environment_number_of_state_action_successor_states[i][j]);
			for (k = 0; k < environment_number_of_state_action_successor_states[i][j]; k++)
			{
				system_return = fscanf(infile_state_action_successor_state_indices, "%u\t", &environment_state_action_successor_state_indices[i][j][k]);
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
	double*** environment_state_action_successor_state_transition_probabilities;
	
	FILE* infile_state_action_successor_state_transition_probabilities = fopen("inputs/state_action_successor_state_transition_probabilities.txt", "r");
	environment_state_action_successor_state_transition_probabilities = malloc(sizeof(double**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		environment_state_action_successor_state_transition_probabilities[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			environment_state_action_successor_state_transition_probabilities[i][j] = malloc(sizeof(double*) * environment_number_of_state_action_successor_states[i][j]);
			for (k = 0; k < environment_number_of_state_action_successor_states[i][j]; k++)
			{
				system_return = fscanf(infile_state_action_successor_state_transition_probabilities, "%lf\t", &environment_state_action_successor_state_transition_probabilities[i][j][k]);
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
	double*** environment_state_action_successor_state_transition_probabilities_cumulative_sum;
	
	environment_state_action_successor_state_transition_probabilities_cumulative_sum = malloc(sizeof(double**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		environment_state_action_successor_state_transition_probabilities_cumulative_sum[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			environment_state_action_successor_state_transition_probabilities_cumulative_sum[i][j] = malloc(sizeof(double*) * environment_number_of_state_action_successor_states[i][j]);
			
			if (environment_number_of_state_action_successor_states[i][j] > 0)
			{
				environment_state_action_successor_state_transition_probabilities_cumulative_sum[i][j][0] = environment_state_action_successor_state_transition_probabilities[i][j][0];
				
				for (k = 1; k < environment_number_of_state_action_successor_states[i][j]; k++)
				{
					environment_state_action_successor_state_transition_probabilities_cumulative_sum[i][j][k] = environment_state_action_successor_state_transition_probabilities_cumulative_sum[i][j][k - 1] + environment_state_action_successor_state_transition_probabilities[i][j][k];
				} // end of k loop
			}
		} // end of j loop
	} // end of i loop
	
	/* Get the state-action-successor state rewards */
	double*** environment_state_action_successor_state_rewards;
	
	FILE* infile_state_action_successor_state_rewards = fopen("inputs/state_action_successor_state_rewards.txt", "r");
	environment_state_action_successor_state_rewards = malloc(sizeof(double**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		environment_state_action_successor_state_rewards[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			environment_state_action_successor_state_rewards[i][j] = malloc(sizeof(double) * environment_number_of_state_action_successor_states[i][j]);
			for (k = 0; k < environment_number_of_state_action_successor_states[i][j]; k++)
			{
				system_return = fscanf(infile_state_action_successor_state_rewards, "%lf\t", &environment_state_action_successor_state_rewards[i][j][k]);
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
	/************************************** CREATE MODEL OF ENVIRONMENT **************************************/
	/*********************************************************************************************************/
	
	/* Create model state visit counters */
	unsigned int model_number_of_seen_non_terminal_states = 0;
	
	unsigned int* model_seen_non_terminal_states_stack;
	
	model_seen_non_terminal_states_stack = malloc(sizeof(unsigned int) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_seen_non_terminal_states_stack[i] = 0;
	} // end of i loop
	
	unsigned int* model_seen_non_terminal_states_stack_reverse_lookup;
	
	model_seen_non_terminal_states_stack_reverse_lookup = malloc(sizeof(unsigned int) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_seen_non_terminal_states_stack_reverse_lookup[i] = 0;
	} // end of i loop
	
	/* Create model state-action visit counters */
	unsigned int* model_number_of_seen_non_terminal_states_actions;
	
	model_number_of_seen_non_terminal_states_actions = malloc(sizeof(unsigned int) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_number_of_seen_non_terminal_states_actions[i] = 0;
	} // end of i loop
	
	unsigned int** model_seen_non_terminal_states_actions_stack;
	
	model_seen_non_terminal_states_actions_stack = malloc(sizeof(unsigned int*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_seen_non_terminal_states_actions_stack[i] = malloc(sizeof(unsigned int) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			model_seen_non_terminal_states_actions_stack[i][j] = 0;
		} // end of j loop
	} // end of i loop
	
	unsigned int** model_seen_non_terminal_states_actions_stack_reverse_lookup;
	
	model_seen_non_terminal_states_actions_stack_reverse_lookup = malloc(sizeof(unsigned int*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_seen_non_terminal_states_actions_stack_reverse_lookup[i] = malloc(sizeof(unsigned int) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			model_seen_non_terminal_states_actions_stack_reverse_lookup[i][j] = 0;
		} // end of j loop
	} // end of i loop
	
	/* Get the number of state-action successor states */
	unsigned int** model_number_of_state_action_successor_states;
	
	model_number_of_state_action_successor_states = malloc(sizeof(unsigned int*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_number_of_state_action_successor_states[i] = malloc(sizeof(unsigned int) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			model_number_of_state_action_successor_states[i][j] = 0;
		} // end of j loop
	} // end of i loop
	
	/* Get the state-action-successor state indices */
	unsigned int*** model_state_action_successor_state_indices;
	
	model_state_action_successor_state_indices = malloc(sizeof(unsigned int**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_state_action_successor_state_indices[i] = malloc(sizeof(unsigned int*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			model_state_action_successor_state_indices[i][j] = malloc(sizeof(unsigned int*) * model_number_of_state_action_successor_states[i][j]);
			for (k = 0; k < model_number_of_state_action_successor_states[i][j]; k++)
			{
				model_state_action_successor_state_indices[i][j][k] = 0;
			} // end of k loop
		} // end of j loop
	} // end of i loop
	
	/* Get the state-action-successor state transition probabilities */
	double*** model_state_action_successor_state_transition_probabilities;
	
	model_state_action_successor_state_transition_probabilities = malloc(sizeof(double**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_state_action_successor_state_transition_probabilities[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			model_state_action_successor_state_transition_probabilities[i][j] = malloc(sizeof(double*) * model_number_of_state_action_successor_states[i][j]);
			for (k = 0; k < model_number_of_state_action_successor_states[i][j]; k++)
			{
				model_state_action_successor_state_transition_probabilities[i][j][k] = 1.0;
			} // end of k loop
		} // end of j loop
	} // end of i loop
	
	/* Create the state-action-successor state transition probability cumulative sum array */
	double*** model_state_action_successor_state_transition_probabilities_cumulative_sum;
	
	model_state_action_successor_state_transition_probabilities_cumulative_sum = malloc(sizeof(double**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_state_action_successor_state_transition_probabilities_cumulative_sum[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			model_state_action_successor_state_transition_probabilities_cumulative_sum[i][j] = malloc(sizeof(double*) * model_number_of_state_action_successor_states[i][j]);
			
			if (model_number_of_state_action_successor_states[i][j] > 0)
			{
				model_state_action_successor_state_transition_probabilities_cumulative_sum[i][j][0] = model_state_action_successor_state_transition_probabilities[i][j][0];
				
				for (k = 1; k < model_number_of_state_action_successor_states[i][j]; k++)
				{
					model_state_action_successor_state_transition_probabilities_cumulative_sum[i][j][k] = model_state_action_successor_state_transition_probabilities_cumulative_sum[i][j][k - 1] + model_state_action_successor_state_transition_probabilities[i][j][k];
				} // end of k loop
			}
		} // end of j loop
	} // end of i loop
	
	/* Get the state-action-successor state rewards */
	double*** model_state_action_successor_state_rewards;
	
	model_state_action_successor_state_rewards = malloc(sizeof(double**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_state_action_successor_state_rewards[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			model_state_action_successor_state_rewards[i][j] = malloc(sizeof(double) * model_number_of_state_action_successor_states[i][j]);
			for (k = 0; k < model_number_of_state_action_successor_states[i][j]; k++)
			{
				model_state_action_successor_state_rewards[i][j][k] = 0.0;
			} // end of k loop
		} // end of j loop
	} // end of i loop
	
	/* Track the number of times each state-action-successor state triple has been visited */
	unsigned int*** model_state_action_successor_state_number_of_visits;
	
	model_state_action_successor_state_number_of_visits = malloc(sizeof(unsigned int**) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		model_state_action_successor_state_number_of_visits[i] = malloc(sizeof(unsigned int*) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			model_state_action_successor_state_number_of_visits[i][j] = malloc(sizeof(unsigned int*) * model_number_of_state_action_successor_states[i][j]);
			for (k = 0; k < model_number_of_state_action_successor_states[i][j]; k++)
			{
				model_state_action_successor_state_number_of_visits[i][j][k] = 0;
			} // end of k loop
		} // end of j loop
	} // end of i loop
	
	/*********************************************************************************************************/
	/**************************************** SETUP POLICY ITERATION *****************************************/
	/*********************************************************************************************************/
	
	/* Set the number of episodes */
	unsigned int number_of_episodes = 10000;
	
	/* Set the maximum episode length */
	unsigned int maximum_episode_length = 2000;
	
	/* Set the number of steps for the planning stage */
	unsigned int number_of_planning_steps = 50;
	
	/* Create state-action-value function array */
	double** state_action_value_function;
	state_action_value_function = malloc(sizeof(double*) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_action_value_function[i] = malloc(sizeof(double) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			state_action_value_function[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	unsigned int** state_action_value_function_max_tie_stack;
	state_action_value_function_max_tie_stack = malloc(sizeof(unsigned int*) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_action_value_function_max_tie_stack[i] = malloc(sizeof(unsigned int) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			state_action_value_function_max_tie_stack[i][j] = 0;
		} // end of j loop
	} // end of i loop
	
	/* Create policy array */
	double** policy;
	policy = malloc(sizeof(double*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		policy[i] = malloc(sizeof(double) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			policy[i][j] = 1.0 / number_of_actions_per_non_terminal_state[i];
		} // end of j loop
	} // end of i loop
	
	/* Create policy cumulative sum array */
	double** policy_cumulative_sum;
	policy_cumulative_sum = malloc(sizeof(double*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		policy_cumulative_sum[i] = malloc(sizeof(double) * number_of_actions_per_non_terminal_state[i]);
		policy_cumulative_sum[i][0] = policy[i][0];
		for (j = 1; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			policy_cumulative_sum[i][j] = policy_cumulative_sum[i][j - 1] + policy[i][j];
		} // end of j loop
	} // end of i loop
	
	/* Set learning rate alpha */
	double alpha = 0.1;
	
	/* Set epsilon for our epsilon level of exploration */
	double epsilon = 0.1;
	
	/* Set discounting factor gamma */
	double discounting_factor_gamma = 1.0;
	
	/* Set random seed */
	srand(0);
	
	/*********************************************************************************************************/
	/****************************************** RUN POLICY CONTROL *****************************************/
	/*********************************************************************************************************/
	
	printf("\nInitial state-action value function:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", state_action_value_function[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	printf("\nInitial policy:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", policy[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop

	unsigned int initial_state_index = 0;
	
	/* Loop over episodes */
	for (i = 0; i < number_of_episodes; i++)
	{
		/* Initialize episode to get initial state and action */
		initial_state_index = InitializeEpisode(number_of_non_terminal_states);
		
		/* Loop through episode and update the policy */
		LoopThroughEpisode(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, environment_number_of_state_action_successor_states, environment_state_action_successor_state_indices, environment_state_action_successor_state_transition_probabilities_cumulative_sum, environment_state_action_successor_state_rewards, &model_number_of_seen_non_terminal_states, model_seen_non_terminal_states_stack, model_seen_non_terminal_states_stack_reverse_lookup, model_number_of_seen_non_terminal_states_actions, model_seen_non_terminal_states_actions_stack, model_seen_non_terminal_states_actions_stack_reverse_lookup, model_number_of_state_action_successor_states, &model_state_action_successor_state_indices, &model_state_action_successor_state_transition_probabilities, &model_state_action_successor_state_transition_probabilities_cumulative_sum, &model_state_action_successor_state_rewards, &model_state_action_successor_state_number_of_visits, state_action_value_function, state_action_value_function_max_tie_stack, policy, policy_cumulative_sum, alpha, epsilon, discounting_factor_gamma, maximum_episode_length, number_of_planning_steps, initial_state_index);
	} // end of i loop
	
	/*********************************************************************************************************/
	/**************************************** PRINT VALUES AND POLICIES ****************************************/
	/*********************************************************************************************************/
	
	printf("\nFinal state-action value function:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", state_action_value_function[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	printf("\nFinal policy:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", policy[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/*********************************************************************************************************/
	/****************************************** FREE DYNAMIC MEMORY ******************************************/
	/*********************************************************************************************************/
	
	/* Policy iteration arrays */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		free(policy_cumulative_sum[i]);
		free(policy[i]);
	} // end of i loop
	free(policy_cumulative_sum);
	free(policy);
	
	for (i = 0; i < number_of_states; i++)
	{
		free(state_action_value_function_max_tie_stack[i]);
		free(state_action_value_function[i]);
	} // end of i loop
	free(state_action_value_function_max_tie_stack);
	free(state_action_value_function);
	
	/* Model arrays */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			free(model_state_action_successor_state_number_of_visits[i][j]);
			free(model_state_action_successor_state_rewards[i][j]);
			free(model_state_action_successor_state_transition_probabilities_cumulative_sum[i][j]);
			free(model_state_action_successor_state_transition_probabilities[i][j]);
			free(model_state_action_successor_state_indices[i][j]);
		} // end of j loop
		free(model_state_action_successor_state_number_of_visits[i]);
		free(model_state_action_successor_state_rewards[i]);
		free(model_state_action_successor_state_transition_probabilities_cumulative_sum[i]);
		free(model_state_action_successor_state_transition_probabilities[i]);
		free(model_state_action_successor_state_indices[i]);
		free(model_number_of_state_action_successor_states[i]);
		
		free(model_seen_non_terminal_states_actions_stack_reverse_lookup[i]);
		free(model_seen_non_terminal_states_actions_stack[i]);
	} // end of i loop
	free(model_state_action_successor_state_number_of_visits);
	free(model_state_action_successor_state_rewards);
	free(model_state_action_successor_state_transition_probabilities_cumulative_sum);
	free(model_state_action_successor_state_transition_probabilities);
	free(model_state_action_successor_state_indices);
	free(model_number_of_state_action_successor_states);
	
	free(model_seen_non_terminal_states_actions_stack_reverse_lookup);
	free(model_seen_non_terminal_states_actions_stack);
	free(model_number_of_seen_non_terminal_states_actions);
	free(model_seen_non_terminal_states_stack_reverse_lookup);
	free(model_seen_non_terminal_states_stack);
	
	/* Environment arrays */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			free(environment_state_action_successor_state_rewards[i][j]);
			free(environment_state_action_successor_state_transition_probabilities_cumulative_sum[i][j]);
			free(environment_state_action_successor_state_transition_probabilities[i][j]);
			free(environment_state_action_successor_state_indices[i][j]);
		} // end of j loop
		free(environment_state_action_successor_state_rewards[i]);
		free(environment_state_action_successor_state_transition_probabilities_cumulative_sum[i]);
		free(environment_state_action_successor_state_transition_probabilities[i]);
		free(environment_state_action_successor_state_indices[i]);
		free(environment_number_of_state_action_successor_states[i]);
	} // end of i loop
	free(environment_state_action_successor_state_rewards);
	free(environment_state_action_successor_state_transition_probabilities_cumulative_sum);
	free(environment_state_action_successor_state_transition_probabilities);
	free(environment_state_action_successor_state_indices);
	free(environment_number_of_state_action_successor_states);
	free(number_of_actions_per_state);
	free(number_of_actions_per_non_terminal_state);
	
	return 0;
} // end of main

/*********************************************************************************************************/
/*********************************************** FUNCTIONS ***********************************************/
/*********************************************************************************************************/

/* This function initializes episodes */
unsigned int InitializeEpisode(unsigned int number_of_non_terminal_states)
{
	unsigned int initial_state_index;
	
	/* Initial state */
	initial_state_index = rand() % number_of_non_terminal_states; // randomly choose an initial state from all non-terminal states
	
	return initial_state_index;
} // end of InitializeEpisode function

/* This function selects a policy with using epsilon-greedy from the state-action-value function */
void EpsilonGreedyPolicyFromStateActionFunction(unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function, double epsilon, unsigned int state_index, double** policy, double** policy_cumulative_sum)
{
	unsigned int i, max_action_count = 1;
	double max_state_action_value = -DBL_MAX, max_policy_apportioned_probability_per_action = 1.0, remaining_apportioned_probability_per_action = 0.0;
	
	/* Update policy greedily from state-value function */
	for (i = 0; i < number_of_actions_per_non_terminal_state[state_index]; i++)
	{
		/* Save max state action value and find the number of actions that have the same max state action value */
		if (state_action_value_function[state_index][i] > max_state_action_value)
		{
			max_state_action_value = state_action_value_function[state_index][i];
			max_action_count = 1;
		}
		else if (state_action_value_function[state_index][i] == max_state_action_value)
		{
			max_action_count++;
		}
	} // end of i loop

	/* Apportion policy probability across ties equally for state-action pairs that have the same value and spread out epsilon otherwise */
	if (max_action_count == number_of_actions_per_non_terminal_state[state_index])
	{
		max_policy_apportioned_probability_per_action = 1.0 / max_action_count;
		remaining_apportioned_probability_per_action = 0.0;
	}
	else
	{
		max_policy_apportioned_probability_per_action = (1.0 - epsilon) / max_action_count;
		remaining_apportioned_probability_per_action = epsilon / (number_of_actions_per_non_terminal_state[state_index] - max_action_count);
	}
	
	/* Update policy with our apportioned probabilities */
	for (i = 0; i < number_of_actions_per_non_terminal_state[state_index]; i++)
	{
		if (state_action_value_function[state_index][i] == max_state_action_value)
		{
			policy[state_index][i] = max_policy_apportioned_probability_per_action;
		}
		else
		{
			policy[state_index][i] = remaining_apportioned_probability_per_action;
		}
	} // end of i loop
	
	/* Update policy cumulative sum */
	policy_cumulative_sum[state_index][0] = policy[state_index][0];
	for (i = 1; i < number_of_actions_per_non_terminal_state[state_index]; i++)
	{
		policy_cumulative_sum[state_index][i] = policy_cumulative_sum[state_index][i - 1] + policy[state_index][i];
	} // end of i loop
	
	return;
} // end of EpsilonGreedyPolicyFromStateActionFunction function

/* This function loops through episodes and updates the policy */
void LoopThroughEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** environment_number_of_state_action_successor_states, unsigned int*** environment_state_action_successor_state_indices, double*** environment_state_action_successor_state_transition_probabilities_cumulative_sum, double*** environment_state_action_successor_state_rewards, unsigned int* model_number_of_seen_non_terminal_states, unsigned int* model_seen_non_terminal_states_stack, unsigned int* model_seen_non_terminal_states_stack_reverse_lookup, unsigned int* model_number_of_seen_non_terminal_states_actions, unsigned int** model_seen_non_terminal_states_actions_stack, unsigned int** model_seen_non_terminal_states_actions_stack_reverse_lookup, unsigned int** model_number_of_state_action_successor_states, unsigned int**** model_state_action_successor_state_indices, double**** model_state_action_successor_state_transition_probabilities, double**** model_state_action_successor_state_transition_probabilities_cumulative_sum, double**** model_state_action_successor_state_rewards, unsigned int**** model_state_action_successor_state_number_of_visits, double** state_action_value_function, unsigned int** state_action_value_function_max_tie_stack, double** policy, double** policy_cumulative_sum, double alpha, double epsilon, double discounting_factor_gamma, unsigned int maximum_episode_length, unsigned int number_of_planning_steps, unsigned int state_index)
{
	unsigned int i, j;
	unsigned int action_index, successor_state_transition_index, next_state_index, next_action_index, max_action_count;
	double probability, reward, max_action_value;
		
	/* Loop through episode steps until termination */
	for (i = 0; i < maximum_episode_length; i++)
	{
		/* Get epsilon-greedy action */
		probability = UnifRand();
		
		/* Choose policy for chosen state by epsilon-greedy choosing from the state-action-value function */
		EpsilonGreedyPolicyFromStateActionFunction(number_of_actions_per_non_terminal_state, state_action_value_function, epsilon, state_index, policy, policy_cumulative_sum);
		
		/* Find which action using probability */
		for (j = 0; j < number_of_actions_per_non_terminal_state[state_index]; j++)
		{
			if (probability <= policy_cumulative_sum[state_index][j])
			{
				action_index = j;
				break; // break j loop since we found our index
			}
		} // end of j loop
		
		/* Update what state and actions the model has seen */
		UpdateModelSeenStateActions(state_index, action_index, model_number_of_seen_non_terminal_states, model_seen_non_terminal_states_stack, model_seen_non_terminal_states_stack_reverse_lookup, model_number_of_seen_non_terminal_states_actions, model_seen_non_terminal_states_actions_stack, model_seen_non_terminal_states_actions_stack_reverse_lookup);
		
		/* Get reward */
		probability = UnifRand();
		for (j = 0; j < environment_number_of_state_action_successor_states[state_index][action_index]; j++)
		{
			if (probability <= environment_state_action_successor_state_transition_probabilities_cumulative_sum[state_index][action_index][j])
			{
				successor_state_transition_index = j;
				break; // break j loop since we found our index
			}
		} // end of j loop
		
		/* Get reward from state and action */
		reward = environment_state_action_successor_state_rewards[state_index][action_index][successor_state_transition_index];
		
		/* Get next state */
		next_state_index = environment_state_action_successor_state_indices[state_index][action_index][successor_state_transition_index];
		
		/* Check to see if we actioned into a terminal state */
		if (next_state_index >= number_of_non_terminal_states)
		{
			state_action_value_function[state_index][action_index] += alpha * (reward - state_action_value_function[state_index][action_index]);
			
			/* Update model of environment from experience */
			UpdateModelOfEnvironmentFromExperience(state_index, action_index, reward, next_state_index, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, model_number_of_state_action_successor_states, model_state_action_successor_state_indices, model_state_action_successor_state_transition_probabilities, model_state_action_successor_state_transition_probabilities_cumulative_sum, model_state_action_successor_state_rewards, model_state_action_successor_state_number_of_visits);
			
			/* Use updated model to simulate experience in planning phase */
			ModelSimualatePlanning(number_of_planning_steps, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, (*model_number_of_seen_non_terminal_states), model_seen_non_terminal_states_stack, model_seen_non_terminal_states_stack_reverse_lookup, model_number_of_seen_non_terminal_states_actions, model_seen_non_terminal_states_actions_stack, model_seen_non_terminal_states_actions_stack_reverse_lookup, model_number_of_state_action_successor_states, (*model_state_action_successor_state_indices), (*model_state_action_successor_state_transition_probabilities), (*model_state_action_successor_state_transition_probabilities_cumulative_sum), (*model_state_action_successor_state_rewards), (*model_state_action_successor_state_number_of_visits), state_action_value_function_max_tie_stack, alpha, discounting_factor_gamma, state_action_value_function);
			
			break; // break i loop, episode terminated since we ended up in a terminal state
		}
		else
		{
			/* Get next action, max action of next state */
			max_action_value = -DBL_MAX;
			max_action_count = 0;
			
			for (j = 0; j < number_of_actions_per_non_terminal_state[next_state_index]; j++)
			{
				if (max_action_value < state_action_value_function[next_state_index][j])
				{
					max_action_value = state_action_value_function[next_state_index][j];
					state_action_value_function_max_tie_stack[next_state_index][0] = j;
					max_action_count = 1;
				}
				else if (max_action_value == state_action_value_function[next_state_index][j])
				{
					state_action_value_function_max_tie_stack[next_state_index][max_action_count];
					max_action_count++;
				}
			} // end of j loop
			
			next_action_index = state_action_value_function_max_tie_stack[next_state_index][rand() % max_action_count];
			
			/* Calculate state-action-function using quintuple SARSargmax(a,Q) */
			state_action_value_function[state_index][action_index] += alpha * (reward + discounting_factor_gamma * state_action_value_function[next_state_index][next_action_index] - state_action_value_function[state_index][action_index]);
			
			/* Update model of environment from experience */
			UpdateModelOfEnvironmentFromExperience(state_index, action_index, reward, next_state_index, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, model_number_of_state_action_successor_states, model_state_action_successor_state_indices, model_state_action_successor_state_transition_probabilities, model_state_action_successor_state_transition_probabilities_cumulative_sum, model_state_action_successor_state_rewards, model_state_action_successor_state_number_of_visits);
			
			/* Use updated model to simulate experience in planning phase */
			ModelSimualatePlanning(number_of_planning_steps, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, (*model_number_of_seen_non_terminal_states), model_seen_non_terminal_states_stack, model_seen_non_terminal_states_stack_reverse_lookup, model_number_of_seen_non_terminal_states_actions, model_seen_non_terminal_states_actions_stack, model_seen_non_terminal_states_actions_stack_reverse_lookup, model_number_of_state_action_successor_states, (*model_state_action_successor_state_indices), (*model_state_action_successor_state_transition_probabilities), (*model_state_action_successor_state_transition_probabilities_cumulative_sum), (*model_state_action_successor_state_rewards), (*model_state_action_successor_state_number_of_visits), state_action_value_function_max_tie_stack, alpha, discounting_factor_gamma, state_action_value_function);
			
			/* Update state to next state */
			state_index = next_state_index;
		}
	} // end of i loop
	
	return;
} // end of LoopThroughEpisode function

/* This function updates what state and actions the model has seen */
void UpdateModelSeenStateActions(unsigned int state_index, unsigned int action_index, unsigned int* model_number_of_seen_non_terminal_states, unsigned int* model_seen_non_terminal_states_stack, unsigned int* model_seen_non_terminal_states_stack_reverse_lookup, unsigned int* model_number_of_seen_non_terminal_states_actions, unsigned int** model_seen_non_terminal_states_actions_stack, unsigned int** model_seen_non_terminal_states_actions_stack_reverse_lookup)
{	
	/* Check to see if state has already been visited */
	if ((*model_number_of_seen_non_terminal_states) == 0 || (model_seen_non_terminal_states_stack_reverse_lookup[state_index] == 0 && model_seen_non_terminal_states_stack[0] != state_index)) // if new state
	{
		/* Add to state stack */
		model_seen_non_terminal_states_stack[(*model_number_of_seen_non_terminal_states)] = state_index; // 1, 3, 2, 0, 4
		model_seen_non_terminal_states_stack_reverse_lookup[state_index] = (*model_number_of_seen_non_terminal_states); // 3, 0, 2, 1, 4
		
		/* Add to action stack */
		model_seen_non_terminal_states_actions_stack[state_index][model_number_of_seen_non_terminal_states_actions[state_index]] = action_index; // 2, 0, 3, 1
		model_seen_non_terminal_states_actions_stack_reverse_lookup[state_index][action_index] = model_number_of_seen_non_terminal_states_actions[state_index]; // 1, 3, 0, 2
		
		/* Increment counters */
		model_number_of_seen_non_terminal_states_actions[state_index]++;
		(*model_number_of_seen_non_terminal_states)++;
	}
	else // if already visited state
	{
		/* Check to see if action has already been visited */
		if (model_seen_non_terminal_states_actions_stack_reverse_lookup[state_index][action_index] == 0 && model_seen_non_terminal_states_actions_stack[state_index][0] != action_index)
		{
			/* Add to action stack */
			model_seen_non_terminal_states_actions_stack[state_index][model_number_of_seen_non_terminal_states_actions[state_index]] = action_index; // 2, 0, 3, 1
			model_seen_non_terminal_states_actions_stack_reverse_lookup[state_index][action_index] = model_number_of_seen_non_terminal_states_actions[state_index]; // 1, 3, 0, 2
			
			/* Increment counters */
			model_number_of_seen_non_terminal_states_actions[state_index]++;
		}
	}
	
	return;
} // end of UpdateModelSeenStateActions function

/* This function updates the model from environment experience */
void UpdateModelOfEnvironmentFromExperience(unsigned int state_index, unsigned int action_index, double reward, unsigned int next_state_index, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** model_number_of_state_action_successor_states, unsigned int**** model_state_action_successor_state_indices, double**** model_state_action_successor_state_transition_probabilities, double**** model_state_action_successor_state_transition_probabilities_cumulative_sum, double**** model_state_action_successor_state_rewards, unsigned int**** model_state_action_successor_state_number_of_visits)
{
	if (model_number_of_state_action_successor_states[state_index][action_index] == 0)
	{
		/* Realloc model state action successor state arrays */
		ReallocModelStateActionSuccessorStateArrays(state_index, action_index, reward, next_state_index, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, model_number_of_state_action_successor_states, model_state_action_successor_state_indices, model_state_action_successor_state_transition_probabilities, model_state_action_successor_state_transition_probabilities_cumulative_sum, model_state_action_successor_state_rewards, model_state_action_successor_state_number_of_visits);

		/* Update model state action successor state arrays */
		(*model_state_action_successor_state_indices)[state_index][action_index][model_number_of_state_action_successor_states[state_index][action_index]] = next_state_index;
		(*model_state_action_successor_state_rewards)[state_index][action_index][model_number_of_state_action_successor_states[state_index][action_index]] = reward;
		(*model_state_action_successor_state_number_of_visits)[state_index][action_index][model_number_of_state_action_successor_states[state_index][action_index]] = 1;
		
		(*model_state_action_successor_state_transition_probabilities)[state_index][action_index][0] = 1.0;
		(*model_state_action_successor_state_transition_probabilities_cumulative_sum)[state_index][action_index][0] = 1.0;
		
		model_number_of_state_action_successor_states[state_index][action_index]++;
	}
	else
	{
		unsigned int i;
		
		/* Check if already have next state accounted for */
		int found_next_state_index = -1;
	
		for (i = 0; i < model_number_of_state_action_successor_states[state_index][action_index]; i++)
		{
			if ((*model_state_action_successor_state_indices)[state_index][action_index][i] == next_state_index)
			{
				found_next_state_index = i;
				break; // break i loop since we found next state
			}
		} // end of i loop
		
		/* Update visit count of S' from S and A */
		if (found_next_state_index == -1) // if we did NOT find the next state S' having already been observed
		{
			/* Realloc model state action successor state arrays */
			ReallocModelStateActionSuccessorStateArrays(state_index, action_index, reward, next_state_index, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, model_number_of_state_action_successor_states, model_state_action_successor_state_indices, model_state_action_successor_state_transition_probabilities, model_state_action_successor_state_transition_probabilities_cumulative_sum, model_state_action_successor_state_rewards, model_state_action_successor_state_number_of_visits);

			/* Update model state action successor state arrays */
			(*model_state_action_successor_state_indices)[state_index][action_index][model_number_of_state_action_successor_states[state_index][action_index]] = next_state_index;
			(*model_state_action_successor_state_rewards)[state_index][action_index][model_number_of_state_action_successor_states[state_index][action_index]] = reward;
			(*model_state_action_successor_state_number_of_visits)[state_index][action_index][model_number_of_state_action_successor_states[state_index][action_index]] = 1;
			model_number_of_state_action_successor_states[state_index][action_index]++;
		}
		else
		{
			(*model_state_action_successor_state_number_of_visits)[state_index][action_index][found_next_state_index]++;
		}
		
		/* Update model transition probabilities */
		unsigned int total_visits_across_all_successor_states = 0;
		for (i = 0; i < model_number_of_state_action_successor_states[state_index][action_index]; i++)
		{
			total_visits_across_all_successor_states += (*model_state_action_successor_state_number_of_visits)[state_index][action_index][i];
		} // end of i loop
		
		for (i = 0; i < model_number_of_state_action_successor_states[state_index][action_index]; i++)
		{
			(*model_state_action_successor_state_transition_probabilities)[state_index][action_index][i] = (double)(*model_state_action_successor_state_number_of_visits)[state_index][action_index][i] / total_visits_across_all_successor_states;
		} // end of i loop
		
		/* Update model transition probability cumulative sums */
		(*model_state_action_successor_state_transition_probabilities_cumulative_sum)[state_index][action_index][0] = (*model_state_action_successor_state_transition_probabilities)[state_index][action_index][0];
		for (i = 1; i < model_number_of_state_action_successor_states[state_index][action_index]; i++)
		{
			(*model_state_action_successor_state_transition_probabilities_cumulative_sum)[state_index][action_index][i] = (*model_state_action_successor_state_transition_probabilities_cumulative_sum)[state_index][action_index][i - 1] + (*model_state_action_successor_state_transition_probabilities)[state_index][action_index][i];
		} // end of i loop
	}
	
	return;
} // end of UpdateModelOfEnvironmentFromExperience function

/* This function reallocs model state action successor state arrays */
void ReallocModelStateActionSuccessorStateArrays(unsigned int state_index, unsigned int action_index, double reward, unsigned int next_state_index, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** model_number_of_state_action_successor_states, unsigned int**** model_state_action_successor_state_indices, double**** model_state_action_successor_state_transition_probabilities, double**** model_state_action_successor_state_transition_probabilities_cumulative_sum, double**** model_state_action_successor_state_rewards, unsigned int**** model_state_action_successor_state_number_of_visits)
{
	Realloc3dUnsignedInt(model_state_action_successor_state_indices, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, model_number_of_state_action_successor_states, state_index, action_index);
	Realloc3dDouble(model_state_action_successor_state_transition_probabilities, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, model_number_of_state_action_successor_states, state_index, action_index);
	Realloc3dDouble(model_state_action_successor_state_transition_probabilities_cumulative_sum, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, model_number_of_state_action_successor_states, state_index, action_index);
	Realloc3dDouble(model_state_action_successor_state_rewards, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, model_number_of_state_action_successor_states, state_index, action_index);
	Realloc3dUnsignedInt(model_state_action_successor_state_number_of_visits, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, model_number_of_state_action_successor_states, state_index, action_index);
	
	return;
} // end of ReallocModelStateActionSuccessorStateArrays function

/* This function uses model to plan via simulate experience */
void ModelSimualatePlanning(unsigned int number_of_planning_steps, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int model_number_of_seen_non_terminal_states, unsigned int* model_seen_non_terminal_states_stack, unsigned int* model_seen_non_terminal_states_stack_reverse_lookup, unsigned int* model_number_of_seen_non_terminal_states_actions, unsigned int** model_seen_non_terminal_states_actions_stack, unsigned int** model_seen_non_terminal_states_actions_stack_reverse_lookup, unsigned int** model_number_of_state_action_successor_states, unsigned int*** model_state_action_successor_state_indices, double*** model_state_action_successor_state_transition_probabilities, double*** model_state_action_successor_state_transition_probabilities_cumulative_sum, double*** model_state_action_successor_state_rewards, unsigned int*** model_state_action_successor_state_number_of_visits, unsigned int** state_action_value_function_max_tie_stack, double alpha, double discounting_factor_gamma, double** state_action_value_function)
{
	unsigned int i, j;
	unsigned int state_index, action_index, next_state_index, next_action_index, successor_state_transition_index, max_action_count;
	double probability, reward, max_action_value;
	
	for (i = 0; i < number_of_planning_steps; i++)
	{
		/* Randomly choose state indices from previously seen states */
		state_index = model_seen_non_terminal_states_stack[rand() % model_number_of_seen_non_terminal_states];
		
		/* Randomly choose action indices from previously seen actions in previously seen states */
		action_index = model_seen_non_terminal_states_actions_stack[state_index][rand() % model_number_of_seen_non_terminal_states_actions[state_index]];
		
		/* Get reward */
		probability = UnifRand();
		
		for (j = 0; j < model_number_of_state_action_successor_states[state_index][action_index]; j++)
		{
			if (probability <= model_state_action_successor_state_transition_probabilities_cumulative_sum[state_index][action_index][j])
			{
				successor_state_transition_index = j;
				break; // break j loop since we found our index
			}
		} // end of j loop
		
		/* Get reward from state and action */
		reward = model_state_action_successor_state_rewards[state_index][action_index][successor_state_transition_index];
		
		/* Get next state */
		next_state_index = model_state_action_successor_state_indices[state_index][action_index][successor_state_transition_index];
		
		/* Check to see if we actioned into a terminal state */
		if (next_state_index >= number_of_non_terminal_states)
		{
			state_action_value_function[state_index][action_index] += alpha * (reward - state_action_value_function[state_index][action_index]);
		}
		else
		{
			/* Get next action, max action of next state */
			max_action_value = -DBL_MAX;
			max_action_count = 0;
			
			for (j = 0; j < number_of_actions_per_non_terminal_state[next_state_index]; j++)
			{
				if (max_action_value < state_action_value_function[next_state_index][j])
				{
					max_action_value = state_action_value_function[next_state_index][j];
					state_action_value_function_max_tie_stack[next_state_index][0] = j;
					max_action_count = 1;
				}
				else if (max_action_value == state_action_value_function[next_state_index][j])
				{
					state_action_value_function_max_tie_stack[next_state_index][max_action_count];
					max_action_count++;
				}
			} // end of j loop
			
			next_action_index = state_action_value_function_max_tie_stack[next_state_index][rand() % max_action_count];
			
			/* Calculate state-action-function using quintuple SARSargmax(a,Q) */
			state_action_value_function[state_index][action_index] += alpha * (reward + discounting_factor_gamma * state_action_value_function[next_state_index][next_action_index] - state_action_value_function[state_index][action_index]);
		}
	} // end of i loop
		
	return;
} // end of ModelSimualatePlanning function

/* This function reallocates more memory to the passed 3d unsigned int array */
void Realloc3dUnsignedInt(unsigned int**** array3d, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** model_number_of_state_action_successor_states, unsigned int state_index, unsigned int action_index)
{
	unsigned int i, j, k;
	unsigned int*** temp = NULL;

	/* Malloc first dimension of temp array (row) */
	temp = malloc(sizeof(unsigned int**) * number_of_non_terminal_states);

	/* Malloc second dimension of temp array (column) */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		temp[i] = malloc(sizeof(unsigned int*) * number_of_actions_per_non_terminal_state[i]);

		/* Malloc third dimension of temp array (depth) */
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			temp[i][j] = malloc(sizeof(unsigned int) * model_number_of_state_action_successor_states[i][j]);
		} // end of j loop
	} // end of i loop

	/* Transfer 3d array's elements to old rectanglular prism of temp array */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			for (k = 0; k < model_number_of_state_action_successor_states[i][j]; k++)
			{
				temp[i][j][k] = (*array3d)[i][j][k];
			} // end of k loop
		} // end of j loop
	} // end of i loop

	/* Free 3d array */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			free((*array3d)[i][j]);
		}
		free((*array3d)[i]);
	}
	free(*array3d);

	/* Malloc 3d array's first dimension (rows) */
	(*array3d) = malloc(sizeof(unsigned int**) * number_of_non_terminal_states);

	/* Malloc 3d array's second dimension (columns) */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		(*array3d)[i] = malloc(sizeof(unsigned int*) * number_of_actions_per_non_terminal_state[i]);

		/* Malloc 3d array's third dimension (depth) */
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			if (i == state_index && j == action_index)
			{
				(*array3d)[i][j] = malloc(sizeof(unsigned int) * (model_number_of_state_action_successor_states[i][j] + 1));
			}
			else
			{
				(*array3d)[i][j] = malloc(sizeof(unsigned int) * model_number_of_state_action_successor_states[i][j]);
			}
		} // end of j loop
	} // end of i loop

	/* Transfer temp array's elements from the old rectangular prism back to 3d array */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			for (k = 0; k < model_number_of_state_action_successor_states[i][j]; k++)
			{
				(*array3d)[i][j][k] = temp[i][j][k];
			} // end of k loop
		} // end of j loop
	} // end of i loop

	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			free(temp[i][j]);
		} // end of j loop
		free(temp[i]);
	} // end of i loop
	free(temp);

	return;
} // end of Realloc3dUnsignedInt function

/* This function reallocates more memory to the passed 3d double array */
void Realloc3dDouble(double**** array3d, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** model_number_of_state_action_successor_states, unsigned int state_index, unsigned int action_index)
{
	unsigned int i, j, k;
	double*** temp = NULL;

	/* Malloc first dimension of temp array (row) */
	temp = malloc(sizeof(double**) * number_of_non_terminal_states);

	/* Malloc second dimension of temp array (column) */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		temp[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);

		/* Malloc third dimension of temp array (depth) */
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			temp[i][j] = malloc(sizeof(double) * model_number_of_state_action_successor_states[i][j]);
		} // end of j loop
	} // end of i loop

	/* Transfer 3d array's elements to old rectanglular prism of temp array */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			for (k = 0; k < model_number_of_state_action_successor_states[i][j]; k++)
			{
				temp[i][j][k] = (*array3d)[i][j][k];
			} // end of k loop
		} // end of j loop
	} // end of i loop

	/* Free 3d array */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			free((*array3d)[i][j]);
		}
		free((*array3d)[i]);
	}
	free(*array3d);

	/* Malloc 3d array's first dimension (rows) */
	(*array3d) = malloc(sizeof(double**) * number_of_non_terminal_states);

	/* Malloc 3d array's second dimension (columns) */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		(*array3d)[i] = malloc(sizeof(double*) * number_of_actions_per_non_terminal_state[i]);

		/* Malloc 3d array's third dimension (depth) */
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			if (i == state_index && j == action_index)
			{
				(*array3d)[i][j] = malloc(sizeof(double) * (model_number_of_state_action_successor_states[i][j] + 1));
			}
			else
			{
				(*array3d)[i][j] = malloc(sizeof(double) * model_number_of_state_action_successor_states[i][j]);
			}
		} // end of j loop
	} // end of i loop

	/* Transfer temp array's elements from the old rectangular prism back to 3d array */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			for (k = 0; k < model_number_of_state_action_successor_states[i][j]; k++)
			{
				(*array3d)[i][j][k] = temp[i][j][k];
			} // end of k loop
		} // end of j loop
	} // end of i loop

	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			free(temp[i][j]);
		} // end of j loop
		free(temp[i]);
	} // end of i loop
	free(temp);

	return;
} // end of Realloc3dDouble function

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function