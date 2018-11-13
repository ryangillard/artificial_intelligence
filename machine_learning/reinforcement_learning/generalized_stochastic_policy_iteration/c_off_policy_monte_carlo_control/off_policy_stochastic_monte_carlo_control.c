#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

double EPS = DBL_EPSILON * 10;

/*********************************************************************************************************/
/********************************************** STRUCTURES ***********************************************/
/*********************************************************************************************************/

struct Episode
{
	unsigned int state_index;
	unsigned int action_index;
	double reward;
};

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function generates episodes */
unsigned int GenerateEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, unsigned int maximum_episode_length, double** behavior_policy, double** behavior_policy_cumulative_sum, struct Episode* episode_log);

/* This function selects a policy greedily from the state-action-value function */
double GreedyPolicyFromStateActionFunction(unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function, unsigned int state_index, double** policy);

/* This function loops through episodes in reverse order and updates the target policy */
void LoopThroughEpisodeInReverse(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function, double** weights_cumulative_sum, double** target_policy, double** behavior_policy, double discounting_factor_gamma, struct Episode* episode_log, unsigned int episode_length);

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
	double minimum_reward = DBL_MAX;
	
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
				
				if (state_action_successor_state_rewards[i][j][k] < minimum_reward)
				{
					minimum_reward = state_action_successor_state_rewards[i][j][k];
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
	
	/* Create episode log */
	struct Episode* episode_log;
	episode_log = malloc(sizeof(struct Episode) * maximum_episode_length);
	for (i = 0; i < maximum_episode_length; i++)
	{
		episode_log[i].state_index = 0;
		episode_log[i].action_index = 0;
		episode_log[i].reward = 0.0;
	} // end of i loop
	
	/* Determine initializer so that we don't break prematurely from episode step loop */
	double state_action_value_function_initializer = 0.0;
	
	if (minimum_reward < 0)
	{
		state_action_value_function_initializer = 2.0 * minimum_reward;
	}
	else
	{
		state_action_value_function_initializer = 0.0;
	}
	
	/* Create state-action-value function array */
	double** state_action_value_function;
	state_action_value_function = malloc(sizeof(double*) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_action_value_function[i] = malloc(sizeof(double) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			state_action_value_function[i][j] = state_action_value_function_initializer;
		} // end of j loop
	} // end of i loop

	/* Create state-action-value function array */
	double** weights_cumulative_sum;
	weights_cumulative_sum = malloc(sizeof(double*) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		weights_cumulative_sum[i] = malloc(sizeof(double) * number_of_actions_per_state[i]);
		for (j = 0; j < number_of_actions_per_state[i]; j++)
		{
			weights_cumulative_sum[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Create target policy array */
	double** target_policy;
	target_policy = malloc(sizeof(double*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		target_policy[i] = malloc(sizeof(double) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			target_policy[i][j] = 1.0 / number_of_actions_per_non_terminal_state[i]; // greedy target policy
		} // end of j loop
	} // end of i loop
	
	/* Create behavior policy array */
	double** behavior_policy;
	behavior_policy = malloc(sizeof(double*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		behavior_policy[i] = malloc(sizeof(double) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			behavior_policy[i][j] = 1.0 / number_of_actions_per_non_terminal_state[i]; // random behavior policy
		} // end of j loop
	} // end of i loop
	
	/* Create behavior policy cumulative sum array */
	double** behavior_policy_cumulative_sum;
	behavior_policy_cumulative_sum = malloc(sizeof(double*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		behavior_policy_cumulative_sum[i] = malloc(sizeof(double) * number_of_actions_per_non_terminal_state[i]);
		behavior_policy_cumulative_sum[i][0] = behavior_policy[i][0];
		for (j = 1; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			behavior_policy_cumulative_sum[i][j] = behavior_policy_cumulative_sum[i][j - 1] + behavior_policy[i][j];
		} // end of j loop
	} // end of i loop
	
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
	
	printf("\nInitial target policy:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", target_policy[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	printf("\nInitial behavior policy:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", behavior_policy[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/* This function generates episodes */
	unsigned int episode_length = 0;
	
	/* Loop over episodes */
	for (i = 0; i < number_of_episodes; i++)
	{
		/* Generate episode and get the length */
		episode_length = GenerateEpisode(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, number_of_state_action_successor_states, state_action_successor_state_indices, state_action_successor_state_transition_probabilities, state_action_successor_state_transition_probabilities_cumulative_sum, state_action_successor_state_rewards, maximum_episode_length, behavior_policy, behavior_policy_cumulative_sum, episode_log);
		
		/* Loop through episode in reverse order and update the target policy */
		LoopThroughEpisodeInReverse(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, state_action_value_function, weights_cumulative_sum, target_policy, behavior_policy, discounting_factor_gamma, episode_log, episode_length);
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
	
	printf("\nFinal target policy:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%lf", target_policy[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/*********************************************************************************************************/
	/****************************************** FREE DYNAMIC MEMORY ******************************************/
	/*********************************************************************************************************/
	
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		free(behavior_policy_cumulative_sum[i]);
		free(behavior_policy[i]);
		free(target_policy[i]);
	} // end of i loop
	free(behavior_policy_cumulative_sum);
	free(behavior_policy);
	free(target_policy);
	
	for (i = 0; i < number_of_states; i++)
	{
		free(weights_cumulative_sum[i]);
		free(state_action_value_function[i]);
	} // end of i loop
	free(weights_cumulative_sum);
	free(state_action_value_function);
	
	free(episode_log);
	
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
} // end of main

/*********************************************************************************************************/
/*********************************************** FUNCTIONS ***********************************************/
/*********************************************************************************************************/

/* This function generates episodes */
unsigned int GenerateEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, unsigned int maximum_episode_length, double** behavior_policy, double** behavior_policy_cumulative_sum, struct Episode* episode_log)
{
	unsigned int i;
	unsigned int step_count = 0, current_state = 0, policy_action_index = 0, successor_state_transition_index = 0;
	double probability = 0.0;
	
	/* Initial state */
	current_state = rand() % number_of_non_terminal_states; // randomly choose an initial state from all non-terminal states
	
	/* Now repeat */
	while (step_count < maximum_episode_length)
	{
		/* Get state */
		episode_log[step_count].state_index = current_state;

		/* Get action */
		probability = UnifRand();
		for (i = 0; i < number_of_actions_per_non_terminal_state[current_state]; i++)
		{
			if (probability <= behavior_policy_cumulative_sum[current_state][i])
			{
				policy_action_index = i;
				break; // break i loop since we found our index
			}
		} // end of i loop
		episode_log[step_count].action_index = policy_action_index;
		
		/* Get reward */
		probability = UnifRand();
		for (i = 0; i < number_of_state_action_successor_states[current_state][policy_action_index]; i++)
		{
			if (probability <= state_action_successor_state_transition_probabilities_cumulative_sum[current_state][policy_action_index][i])
			{
				successor_state_transition_index = i;
				break; // break i loop since we found our index
			}
		} // end of i loop
		
		episode_log[step_count].reward = state_action_successor_state_rewards[current_state][policy_action_index][successor_state_transition_index];
	
		/* Increment step count */
		step_count++;
		
		/* Get next state */
		current_state = state_action_successor_state_indices[current_state][policy_action_index][successor_state_transition_index];

		/* Check to see if we actioned into a terminal state */
		if (current_state >= number_of_non_terminal_states)
		{
			break; // episode terminated since we ended up in a terminal state
		}
	}
	
	return step_count;
} // end of GenerateEpisode function

/* This function selects a policy greedily from the state-action-value function */
double GreedyPolicyFromStateActionFunction(unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function, unsigned int state_index, double** policy)
{
	unsigned int i;
	unsigned int max_action_count = 1;
	double max_state_action_value = -DBL_MAX, max_policy_apportioned_probability_per_action = 1.0;
	
	/* Update policy greedily from state-value function */
	for (i = 0; i < number_of_actions_per_non_terminal_state[state_index]; i++)
	{
		/* Save max state action value and find the number of actions that have the same max state action value */
		if (fabs(state_action_value_function[state_index][i] - max_state_action_value) <= EPS)
		{
			max_action_count++;
		}
		else if (state_action_value_function[state_index][i] > max_state_action_value)
		{
			max_state_action_value = state_action_value_function[state_index][i];
			max_action_count = 1;
		}
	} // end of i loop
	
	/* Apportion policy probability across ties equally for state-action pairs that have the same value and zero otherwise */
	max_policy_apportioned_probability_per_action = 1.0 / max_action_count;
	for (i = 0; i < number_of_actions_per_non_terminal_state[state_index]; i++)
	{
		if (fabs(state_action_value_function[state_index][i] - max_state_action_value) <= EPS)
		{
			policy[state_index][i] = max_policy_apportioned_probability_per_action;
		}
		else
		{
			policy[state_index][i] = 0.0;
		}
	} // end of i loop
	
	return max_policy_apportioned_probability_per_action;
} // end of GreedyPolicyFromStateActionFunction function

/* This function loops through episodes in reverse order and updates the target policy */
void LoopThroughEpisodeInReverse(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function, double** weights_cumulative_sum, double** target_policy, double** behavior_policy, double discounting_factor_gamma, struct Episode* episode_log, unsigned int episode_length)
{
	int i, j;
	unsigned int state_index, action_index;
	double expected_return = 0.0, weight = 1.0, max_policy_apportioned_probability_per_action;
		
	/* Loop through episode steps in reverse order */
	for (i = episode_length - 1; i >= 0; i--)
	{
		state_index = episode_log[i].state_index;
		action_index = episode_log[i].action_index;
		
		/* Calculate expected return */
		expected_return = discounting_factor_gamma * expected_return + episode_log[i].reward;
		
		/* Keep track of weight so that we can incrementally calculate average */
		weights_cumulative_sum[state_index][action_index] += weight;

		if (weights_cumulative_sum[state_index][action_index] != 0.0)
		{
			state_action_value_function[state_index][action_index] += weight / weights_cumulative_sum[state_index][action_index] * (expected_return - state_action_value_function[state_index][action_index]);
		}
		
		/* Choose policy for chosen state by greedily choosing from the state-action-value function */
		max_policy_apportioned_probability_per_action = GreedyPolicyFromStateActionFunction(number_of_actions_per_non_terminal_state, state_action_value_function, state_index, target_policy);
			
		/* Check to see if behavior action from episode is the same as target action */
		if (target_policy[state_index][action_index] != max_policy_apportioned_probability_per_action)
		{
			break; // break episode step loop, move on to next episode
		}
		
		/* Update weight based on behavior policy */
		if (behavior_policy[state_index][action_index] == 0.0)
		{
			weight = 0.0;
		}
		else
		{
			weight /= behavior_policy[state_index][action_index];
		}
	} // end of i loop
	
	return;
} // end of LoopThroughEpisodeInReverse function

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function