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
void EpsilonGreedyPolicyFromStateActionFunction(unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function1, double** state_action_value_function2, double epsilon, unsigned int state_index, double** policy, double** policy_cumulative_sum);

/* This function loops through episodes and updates the policy */
void LoopThroughEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, double** state_action_value_function1, double** state_action_value_function2, unsigned int** state_action_value_function_max_tie_stack, double** policy, double** policy_cumulative_sum, double alpha, double epsilon, double discounting_factor_gamma, unsigned int maximum_episode_length, unsigned int state_index);

/* This function updates the state-action-value function */
unsigned int UpdateStateActionValueFunction(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, double** not_updating_state_action_value_function, unsigned int** state_action_value_function_max_tie_stack, double alpha, double discounting_factor_gamma, unsigned int state_index, unsigned int action_index, double reward, unsigned int next_state_index, double** updating_state_action_value_function);

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
	/**************************************** SETUP POLICY ITERATION *****************************************/
	/*********************************************************************************************************/
	
	/* Set the number of episodes */
	unsigned int number_of_episodes = 10000;
	
	/* Set the maximum episode length */
	unsigned int maximum_episode_length = 200;
	
	/* Create state-action-value function array */
	double** state_action_value_function1;
	state_action_value_function1 = malloc(sizeof(double*) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_action_value_function1[i] = malloc(sizeof(double) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			state_action_value_function1[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	double** state_action_value_function2;
	state_action_value_function2 = malloc(sizeof(double*) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_action_value_function2[i] = malloc(sizeof(double) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			state_action_value_function2[i][j] = 0.0;
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
	/******************************************* RUN POLICY CONTROL ******************************************/
	/*********************************************************************************************************/
	
	printf("\nInitial state-action value function1:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", state_action_value_function1[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	printf("\nInitial state-action value function2:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", state_action_value_function2[i][j]);
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
		LoopThroughEpisode(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, number_of_state_action_successor_states, state_action_successor_state_indices, state_action_successor_state_transition_probabilities_cumulative_sum, state_action_successor_state_rewards, state_action_value_function1, state_action_value_function2, state_action_value_function_max_tie_stack, policy, policy_cumulative_sum, alpha, epsilon, discounting_factor_gamma, maximum_episode_length, initial_state_index);
	} // end of i loop
	
	/*********************************************************************************************************/
	/*************************************** PRINT VALUES AND POLICIES ***************************************/
	/*********************************************************************************************************/
	
	printf("\nFinal state-action value function1:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", state_action_value_function1[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	printf("\nFinal state-action value function2:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", state_action_value_function2[i][j]);
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
		free(state_action_value_function2[i]);
		free(state_action_value_function1[i]);
	} // end of i loop
	free(state_action_value_function_max_tie_stack);
	free(state_action_value_function2);
	free(state_action_value_function1);
	
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

/* This function initializes episodes */
unsigned int InitializeEpisode(unsigned int number_of_non_terminal_states)
{
	unsigned int initial_state_index;
	
	/* Initial state */
	initial_state_index = rand() % number_of_non_terminal_states; // randomly choose an initial state from all non-terminal states
	
	return initial_state_index;
} // end of InitializeEpisode function

/* This function selects a policy with using epsilon-greedy from the state-action-value function */
void EpsilonGreedyPolicyFromStateActionFunction(unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function1, double** state_action_value_function2, double epsilon, unsigned int state_index, double** policy, double** policy_cumulative_sum)
{
	unsigned int i, max_action_count = 1;
	double max_state_action_value = -DBL_MAX, max_policy_apportioned_probability_per_action = 1.0, remaining_apportioned_probability_per_action = 0.0;
	
	/* Update policy greedily from state-value function */
	for (i = 0; i < number_of_actions_per_non_terminal_state[state_index]; i++)
	{
		/* Save max state action value and find the number of actions that have the same max state action value */
		if (state_action_value_function1[state_index][i] + state_action_value_function2[state_index][i] > max_state_action_value)
		{
			max_state_action_value = state_action_value_function1[state_index][i] + state_action_value_function2[state_index][i];
			max_action_count = 1;
		}
		else if (state_action_value_function1[state_index][i] + state_action_value_function2[state_index][i] == max_state_action_value)
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
		if (state_action_value_function1[state_index][i] + state_action_value_function2[state_index][i] == max_state_action_value)
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
void LoopThroughEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, double** state_action_value_function1, double** state_action_value_function2, unsigned int** state_action_value_function_max_tie_stack, double** policy, double** policy_cumulative_sum, double alpha, double epsilon, double discounting_factor_gamma, unsigned int maximum_episode_length, unsigned int state_index)
{
	unsigned int t, i;
	unsigned int action_index, successor_state_transition_index, next_state_index;
	double probability, reward;
		
	/* Loop through episode steps until termination */
	for (t = 0; t < maximum_episode_length; t++)
	{
		/* Get epsilon-greedy action */
		probability = UnifRand();
		
		/* Choose policy for chosen state by epsilon-greedy choosing from the state-action-value function */
		EpsilonGreedyPolicyFromStateActionFunction(number_of_actions_per_non_terminal_state, state_action_value_function1, state_action_value_function2, epsilon, state_index, policy, policy_cumulative_sum);
		
		/* Find which action using probability */
		for (i = 0; i < number_of_actions_per_non_terminal_state[state_index]; i++)
		{
			if (probability <= policy_cumulative_sum[state_index][i])
			{
				action_index = i;
				break; // break i loop since we found our index
			}
		} // end of i loop
		
		/* Get reward */
		probability = UnifRand();
		
		for (i = 0; i < number_of_state_action_successor_states[state_index][action_index]; i++)
		{
			if (probability <= state_action_successor_state_transition_probabilities_cumulative_sum[state_index][action_index][i])
			{
				successor_state_transition_index = i;
				break; // break i loop since we found our index
			}
		} // end of i loop
		
		/* Get reward from state and action */
		reward = state_action_successor_state_rewards[state_index][action_index][successor_state_transition_index];
		
		/* Get next state */
		next_state_index = state_action_successor_state_indices[state_index][action_index][successor_state_transition_index];
		
		/* Update state action value equally randomly selecting from the two state-action-value functions */
		probability = UnifRand();
		
		if (probability <= 0.5)
		{
			state_index = UpdateStateActionValueFunction(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, state_action_value_function2, state_action_value_function_max_tie_stack, alpha, discounting_factor_gamma, state_index, action_index, reward, next_state_index, state_action_value_function1);
		}
		else
		{
			state_index = UpdateStateActionValueFunction(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, state_action_value_function1, state_action_value_function_max_tie_stack, alpha, discounting_factor_gamma, state_index, action_index, reward, next_state_index, state_action_value_function2);
		}
		
		/* Check to see if we actioned into a terminal state */
		if (state_index >= number_of_non_terminal_states)
		{
			break; // episode terminated since we ended up in a terminal state
		}
	} // end of t loop
	
	return;
} // end of LoopThroughEpisode function

/* This function updates the state-action-value function */
unsigned int UpdateStateActionValueFunction(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, double** not_updating_state_action_value_function, unsigned int** state_action_value_function_max_tie_stack, double alpha, double discounting_factor_gamma, unsigned int state_index, unsigned int action_index, double reward, unsigned int next_state_index, double** updating_state_action_value_function)
{
	unsigned int i;
	unsigned int next_action_index, max_action_count;
	double max_action_value;
	
	/* Check to see if we actioned into a terminal state */
	if (next_state_index >= number_of_non_terminal_states)
	{
		updating_state_action_value_function[state_index][action_index] += alpha * (reward - updating_state_action_value_function[state_index][action_index]);
	}
	else
	{
		/* Get next action, max action of next state */
		max_action_value = -DBL_MAX;
		max_action_count = 0;
		
		for (i = 0; i < number_of_actions_per_non_terminal_state[next_state_index]; i++)
		{
			if (max_action_value < updating_state_action_value_function[next_state_index][i])
			{
				max_action_value = updating_state_action_value_function[next_state_index][i];
				state_action_value_function_max_tie_stack[next_state_index][0] = i;
				max_action_count = 1;
			}
			else if (max_action_value == updating_state_action_value_function[next_state_index][i])
			{
				state_action_value_function_max_tie_stack[next_state_index][max_action_count];
				max_action_count++;					
			}
		} // end of i loop
		
		next_action_index = state_action_value_function_max_tie_stack[next_state_index][rand() % max_action_count];
		
		/* Calculate state-action-function using quintuple SARSA */
		updating_state_action_value_function[state_index][action_index] += alpha * (reward + discounting_factor_gamma * not_updating_state_action_value_function[next_state_index][next_action_index] - updating_state_action_value_function[state_index][action_index]);
		
		/* Update state and action to next state and action */
		state_index = next_state_index;
	}
	
	return state_index;
} // end of UpdateStateActionValueFunction function

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function