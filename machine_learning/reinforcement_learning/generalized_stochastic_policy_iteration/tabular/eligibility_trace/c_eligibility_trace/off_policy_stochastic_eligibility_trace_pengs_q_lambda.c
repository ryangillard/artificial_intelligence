#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function initializes episodes */
unsigned int InitializeEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, double** eligibility_trace);

/* This function resets the eligibility traces */
void ResetEligbilityTraces(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, double** eligibility_trace);

/* This function selects a policy with using epsilon-greedy from the state-action-value function */
void EpsilonGreedyPolicyFromStateActionFunction(unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function, double epsilon, unsigned int state_index, double** policy, double** policy_cumulative_sum);

/* This function loops through episodes and updates the policy */
void LoopThroughEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, double** state_action_value_function, unsigned int** state_action_value_function_max_tie_stack, double** policy, double** policy_cumulative_sum, double** eligibility_trace, double alpha, double epsilon, double discounting_factor_gamma, double trace_decay_lambda, unsigned int trace_update_type, unsigned int maximum_episode_length, unsigned int state_index);

/* This function updates the eligibility trace and state-action-value function */
void UpdateEligibilityTraceAndStateActionValueFunction(unsigned int state_index, unsigned int action_index, double delta, double delta_prime, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, double alpha, double discounting_factor_gamma, double trace_decay_lambda, unsigned int trace_update_type, double** state_action_value_function, double** eligibility_trace);

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
	unsigned int number_of_episodes = 20000;
	
	/* Set the maximum episode length */
	unsigned int maximum_episode_length = 200;
	
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
	
	/* Create eligibility trace array */
	double** eligibility_trace;
	eligibility_trace = malloc(sizeof(double*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		eligibility_trace[i] = malloc(sizeof(double) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			eligibility_trace[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Set learning rate alpha */
	double alpha = 0.1;
	
	/* Set epsilon for our epsilon level of exploration */
	double epsilon = 0.1;
	
	/* Set discounting factor gamma */
	double discounting_factor_gamma = 1.0;
	
	/* Set trace decay parameter lambda */
	double trace_decay_lambda = 0.9;
	
	/* Set the trace update type, 0 = accumulating, 1 = replacing */
	unsigned int trace_update_type = 0;
	
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
		/* Initialize episode to get initial state */
		InitializeEpisode(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, eligibility_trace);
		
		/* Loop through episode and update the policy */
		LoopThroughEpisode(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, number_of_state_action_successor_states, state_action_successor_state_indices, state_action_successor_state_transition_probabilities_cumulative_sum, state_action_successor_state_rewards, state_action_value_function, state_action_value_function_max_tie_stack, policy, policy_cumulative_sum, eligibility_trace, alpha, epsilon, discounting_factor_gamma, trace_decay_lambda, trace_update_type, maximum_episode_length, initial_state_index);
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
	
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		free(eligibility_trace[i]);
		free(policy_cumulative_sum[i]);
		free(policy[i]);
	} // end of i loop
	free(eligibility_trace);
	free(policy_cumulative_sum);
	free(policy);
	
	for (i = 0; i < number_of_states; i++)
	{
		free(state_action_value_function_max_tie_stack[i]);
		free(state_action_value_function[i]);
	} // end of i loop
	free(state_action_value_function_max_tie_stack);
	free(state_action_value_function);
	
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
unsigned int InitializeEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, double** eligibility_trace)
{
	unsigned int initial_state_index;
	
	/* Reset eligibility traces for new episode */
	ResetEligbilityTraces(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, eligibility_trace);

	/* Initial state */
	initial_state_index = rand() % number_of_non_terminal_states; // randomly choose an initial state from all non-terminal states
	
	return initial_state_index;
} // end of InitializeEpisode function

/* This function resets the eligibility traces */
void ResetEligbilityTraces(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, double** eligibility_trace)
{
	unsigned int i, j;
	
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			eligibility_trace[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	return;
} // end of ResetEligbilityTraces function

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
void LoopThroughEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, double** state_action_value_function, unsigned int** state_action_value_function_max_tie_stack, double** policy, double** policy_cumulative_sum, double** eligibility_trace, double alpha, double epsilon, double discounting_factor_gamma, double trace_decay_lambda, unsigned int trace_update_type, unsigned int maximum_episode_length, unsigned int state_index)
{
	unsigned int i, j;
	unsigned int action_index, successor_state_transition_index, next_state_index;
	double probability, reward, delta, delta_prime, state_value, next_state_value;
		
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
		
		/* Calculate the state value of the current state */
		state_value = 0.0;
		
		for (j = 0; j < number_of_actions_per_non_terminal_state[state_index]; j++)
		{
			state_value += policy[state_index][j] * state_action_value_function[state_index][j];
		} // end of j loop
		
		/* Check to see if we actioned into a terminal state */
		if (next_state_index >= number_of_non_terminal_states)
		{
			/* Calculate TD error deltas */
			delta_prime = reward - state_action_value_function[state_index][action_index]; 
			delta = reward - state_value;
						
			/* Update eligibility traces and state action value function with TD error */
			UpdateEligibilityTraceAndStateActionValueFunction(state_index, action_index, delta, delta_prime, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, alpha, discounting_factor_gamma, trace_decay_lambda, trace_update_type, state_action_value_function, eligibility_trace);
			
			break; // episode terminated since we ended up in a terminal state
		}
		else
		{
			/* Calculate the state value of the next state */
			next_state_value = 0.0;
			
			for (j = 0; j < number_of_actions_per_non_terminal_state[state_index]; j++)
			{
				next_state_value += policy[next_state_index][j] * state_action_value_function[next_state_index][j];
			} // end of j loop
			
			/* Calculate TD error deltas */
			delta_prime = reward + discounting_factor_gamma * next_state_value - state_action_value_function[state_index][action_index]; 
			delta = reward + discounting_factor_gamma * next_state_value - state_value;
						
			/* Update eligibility traces and state action value function with TD error */
			UpdateEligibilityTraceAndStateActionValueFunction(state_index, action_index, delta, delta_prime, number_of_non_terminal_states, number_of_actions_per_non_terminal_state, alpha, discounting_factor_gamma, trace_decay_lambda, trace_update_type, state_action_value_function, eligibility_trace);
			
			/* Update state and action to next state and action */
			state_index = next_state_index;
		}
	} // end of i loop
	
	return;
} // end of LoopThroughEpisode function

/* This function updates the eligibility trace and state-action-value function */
void UpdateEligibilityTraceAndStateActionValueFunction(unsigned int state_index, unsigned int action_index, double delta, double delta_prime, unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, double alpha, double discounting_factor_gamma, double trace_decay_lambda, unsigned int trace_update_type, double** state_action_value_function, double** eligibility_trace)
{
	unsigned int i, j;
	
	/* Update state-action-value function */
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			eligibility_trace[i][j] *= discounting_factor_gamma * trace_decay_lambda;
			
			state_action_value_function[i][j] += alpha * delta * eligibility_trace[i][j];	
		} // end of j loop
	} // end of i loop
	
	/* Apply delta prime now to state-action-value function */
	state_action_value_function[state_index][action_index] += alpha * delta_prime;
	
	/* Update eligibility trace */
	if (trace_update_type == 1) // replacing
	{
		eligibility_trace[state_index][action_index] = 1.0;
	}
	else // accumulating or unknown
	{
		eligibility_trace[state_index][action_index] += 1.0;
	}
	
	return;
} // end of UpdateEligibilityTraceAndStateActionValueFunction function

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function