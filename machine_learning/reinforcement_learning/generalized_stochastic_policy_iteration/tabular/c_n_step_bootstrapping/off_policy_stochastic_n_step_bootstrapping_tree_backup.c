#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

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

/* This function initializes episodes */
unsigned int InitializeEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int maximum_episode_length, double** policy_cumulative_sum, struct Episode* episode_log);

/* This function selects a policy greedily from the state-action-value function */
void GreedyPolicyFromStateActionFunction(unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function, unsigned int state_index, double** policy, double** policy_cumulative_sum);

/* This function loops through episodes and updates the policy */
void LoopThroughEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, double** state_action_value_function, double** policy, double** policy_cumulative_sum, double alpha, double discounting_factor_gamma, unsigned int maximum_episode_length, unsigned int max_timestep, struct Episode* episode_log, unsigned int n_steps);

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
	
	/* Set the n steps */
	unsigned int n_steps = 4;
	
	/* Set the number of episodes */
	unsigned int number_of_episodes = 10000;
	
	/* Set the maximum episode length */
	unsigned int maximum_episode_length = 2000;
	
	/* Create episode log */
	struct Episode* episode_log;
	episode_log = malloc(sizeof(struct Episode) * (n_steps + 1));
	for (i = 0; i < (n_steps + 1); i++)
	{
		episode_log[i].state_index = 0;
		episode_log[i].action_index = 0;
		episode_log[i].reward = 0.0;
	} // end of i loop
	
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

	unsigned int max_timestep = maximum_episode_length;
	
	/* Loop over episodes */
	for (i = 0; i < number_of_episodes; i++)
	{
		/* Initialize episode to get initial state and action */
		max_timestep = InitializeEpisode(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, maximum_episode_length, policy_cumulative_sum, episode_log);

		/* Loop through episode and update the policy */
		LoopThroughEpisode(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, number_of_state_action_successor_states, state_action_successor_state_indices, state_action_successor_state_transition_probabilities_cumulative_sum, state_action_successor_state_rewards, state_action_value_function, policy, policy_cumulative_sum, alpha, discounting_factor_gamma, maximum_episode_length, max_timestep, episode_log, n_steps);
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
			printf("\t%.16f", state_action_value_function[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	printf("\nFinal policy:\n");
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		printf("%u", i);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			printf("\t%.16f", policy[i][j]);
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
		free(state_action_value_function[i]);
	} // end of i loop
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
	
	return 0;
} // end of main

/*********************************************************************************************************/
/*********************************************** FUNCTIONS ***********************************************/
/*********************************************************************************************************/

/* This function initializes episodes */
unsigned int InitializeEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int maximum_episode_length, double** policy_cumulative_sum, struct Episode* episode_log)
{
	unsigned int i;
	double probability = 0.0;
	
	/* Initial state */
	episode_log[0].state_index = rand() % number_of_non_terminal_states; // randomly choose an initial state from all non-terminal states
	
	/* Get initial action */
	episode_log[0].action_index = rand() % number_of_actions_per_non_terminal_state[episode_log[0].state_index]; // randomly choose an initial action as a function of the initial state
	
	return maximum_episode_length;
} // end of InitializeEpisode function

/* This function selects a policy greedily from the state-action-value function */
void GreedyPolicyFromStateActionFunction(unsigned int* number_of_actions_per_non_terminal_state, double** state_action_value_function, unsigned int state_index, double** policy, double** policy_cumulative_sum)
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
	
	/* Update policy cumulative sum */
	policy_cumulative_sum[state_index][0] = policy[state_index][0];
	for (i = 1; i < number_of_actions_per_non_terminal_state[state_index]; i++)
	{
		policy_cumulative_sum[state_index][i] = policy_cumulative_sum[state_index][i - 1] + policy[state_index][i];
	} // end of i loop
	
	return;
} // end of GreedyPolicyFromStateActionFunction function

/* This function loops through episodes and updates the policy */
void LoopThroughEpisode(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities_cumulative_sum, double*** state_action_successor_state_rewards, double** state_action_value_function, double** policy, double** policy_cumulative_sum, double alpha, double discounting_factor_gamma, unsigned int maximum_episode_length, unsigned int max_timestep, struct Episode* episode_log, unsigned int n_steps)
{
	int t, i, k, tau = 0;
	unsigned int t_mod_n_plus_1, t_plus_1_mod_n_plus_1, k_mod_n_plus_1, tau_mod_n_plus_1, successor_state_transition_index;
	double probability, state_value_function_expected_value_on_policy = 0.0, expected_return = 0.0;
		
	/* Loop through episode steps until termination */
	for (t = 0; t < maximum_episode_length; t++)
	{
		/* Spend a little memory to save computation time */
		t_mod_n_plus_1 = t % (n_steps + 1);
		t_plus_1_mod_n_plus_1 = (t + 1) % (n_steps + 1);
		
		if (t < max_timestep)
		{
			/* Get reward */
			probability = UnifRand();
			
			for (i = 0; i < number_of_state_action_successor_states[episode_log[t_mod_n_plus_1].state_index][episode_log[t_mod_n_plus_1].action_index]; i++)
			{
				if (probability <= state_action_successor_state_transition_probabilities_cumulative_sum[episode_log[t_mod_n_plus_1].state_index][episode_log[t_mod_n_plus_1].action_index][i])
				{
					successor_state_transition_index = i;
					break; // break i loop since we found our index
				}
			} // end of i loop
			
			/* Get reward from state and action */
			episode_log[t_plus_1_mod_n_plus_1].reward = state_action_successor_state_rewards[episode_log[t  % (n_steps + 1)].state_index][episode_log[t_mod_n_plus_1].action_index][successor_state_transition_index];
			
			/* Get next state */
			episode_log[t_plus_1_mod_n_plus_1].state_index = state_action_successor_state_indices[episode_log[t_mod_n_plus_1].state_index][episode_log[t_mod_n_plus_1].action_index][successor_state_transition_index];
			
			/* Check to see if we actioned into a terminal state */
			if (episode_log[t_plus_1_mod_n_plus_1].state_index >= number_of_non_terminal_states)
			{
				max_timestep = t + 1;
			}
			else
			{
				/* Get next action */
				episode_log[t_plus_1_mod_n_plus_1].action_index = rand() % number_of_actions_per_non_terminal_state[episode_log[t_plus_1_mod_n_plus_1].state_index]; // randomly choose a next action as a function of the next state
			}
		}
		
		tau = t - n_steps + 1; // tau is the time whose estimate is being updated
		
		if (tau >= 0)
		{
			/* Calculate expected return */
			if (t + 1 >= max_timestep)
			{
				/* Calculate expected return */
				expected_return = episode_log[max_timestep % (n_steps + 1)].reward;
			}
			else
			{
				/* Calculate expected state value function from policy */
				state_value_function_expected_value_on_policy = 0.0;
							
				for (i = 0; i < number_of_actions_per_non_terminal_state[episode_log[t_plus_1_mod_n_plus_1].state_index]; i++)
				{
					state_value_function_expected_value_on_policy += policy[episode_log[t_plus_1_mod_n_plus_1].state_index][i] * state_action_value_function[episode_log[t_plus_1_mod_n_plus_1].state_index][i];
				} // end of i loop
				
				/* Calculate expected return */
				expected_return = episode_log[t_plus_1_mod_n_plus_1].reward + discounting_factor_gamma * state_value_function_expected_value_on_policy;
			}
			
			for (k = min(t, max_timestep - 1); k >= tau + 1; k--)
			{
				/* Spend a little memory to save computation time */
				k_mod_n_plus_1 = k % (n_steps + 1);
				
				/* Calculate expected state value function from policy, however without including kth chosen action */
				state_value_function_expected_value_on_policy = 0.0;
							
				for (i = 0; i < number_of_actions_per_non_terminal_state[episode_log[k_mod_n_plus_1].state_index]; i++)
				{
					if (i != episode_log[k_mod_n_plus_1].action_index)
					{
						state_value_function_expected_value_on_policy += policy[episode_log[k_mod_n_plus_1].state_index][i] * state_action_value_function[episode_log[k_mod_n_plus_1].state_index][i];
					}
				} // end of i loop
				
				expected_return = episode_log[k_mod_n_plus_1].reward + discounting_factor_gamma * (state_value_function_expected_value_on_policy + policy[episode_log[k_mod_n_plus_1].state_index][episode_log[k_mod_n_plus_1].action_index] * expected_return);
			} // end of k loop
			
			/* Spend a little memory to save computation time */
			tau_mod_n_plus_1 = tau % (n_steps + 1);
			
			/* Calculate state-action-function at tau timestep */
			state_action_value_function[episode_log[tau_mod_n_plus_1].state_index][episode_log[tau_mod_n_plus_1].action_index] += alpha * (expected_return - state_action_value_function[episode_log[tau_mod_n_plus_1].state_index][episode_log[tau_mod_n_plus_1].action_index]);
			
			/* Choose policy for chosen state by greedily choosing from the state-action-value function */
			GreedyPolicyFromStateActionFunction(number_of_actions_per_non_terminal_state, state_action_value_function, episode_log[tau_mod_n_plus_1].state_index, policy, policy_cumulative_sum);
		}
		
		if (tau == max_timestep - 1)
		{
			break; // break episode step loop, move on to next episode
		}
	} // end of t loop
	
	return;
} // end of LoopThroughEpisode function

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function
