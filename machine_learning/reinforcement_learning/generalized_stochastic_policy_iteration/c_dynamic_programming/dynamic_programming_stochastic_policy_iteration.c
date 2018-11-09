#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function evaluates the value functions given the current policy */
void PolicyEvaluation(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities, double*** state_action_successor_state_rewards, double** policy, double convergence_threshold, double discounting_factor_gamma, unsigned int maximum_number_of_policy_evaluations, double* state_value_function, double** state_action_value_function);

/* This function greedily updates the policy based on the current value function */
int PolicyImprovement(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities, double*** state_action_successor_state_rewards, double** policy, double** old_policy, double discounting_factor_gamma, double* state_value_function);

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
	
	/* Create state-value function array */
	double* state_value_function;
	state_value_function = malloc(sizeof(double) * number_of_states);
	for (i = 0; i < number_of_states; i++)
	{
		state_value_function[i] = 0.0;
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
			policy[i][j] = 1.0 / number_of_actions_per_non_terminal_state[i]; // random policy
		} // end of j loop
	} // end of i loop
	
	/* Create policy array */
	double** old_policy;
	old_policy = malloc(sizeof(double*) * number_of_non_terminal_states);
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		old_policy[i] = malloc(sizeof(double) * number_of_actions_per_non_terminal_state[i]);
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			old_policy[i][j] = 1.0 / number_of_actions_per_non_terminal_state[i]; // random policy
		} // end of j loop
	} // end of i loop
	
	/* Set discounting factor gamma */
	double discounting_factor_gamma = 1.0;
	
	/* Set convergence threshold */
	double convergence_threshold = 0.001;
	
	/* Set maximum number of sweeps and policy evaluations */
	unsigned int maximum_number_of_sweeps = 30, maximum_number_of_policy_evaluations = 20;
	
	/*********************************************************************************************************/
	/****************************************** RUN POLICY ITERATION *****************************************/
	/*********************************************************************************************************/

	int policy_stable = 1;
	unsigned int number_of_sweeps = 0;

	do
	{
		printf("\nStarting sweep %u\n\n", number_of_sweeps);
		
		printf("State value function before sweep %u:\n", number_of_sweeps);
		for (i = 0; i < number_of_states; i++)
		{
			printf("%u\t%lf\n", i, state_value_function[i]);
		} // end of i loop
		
		printf("\nState-action value function before sweep %u:\n", number_of_sweeps);
		for (i = 0; i < number_of_non_terminal_states; i++)
		{
			printf("%u", i);
			for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
			{
				printf("\t%lf", state_action_value_function[i][j]);
			} // end of j loop
			printf("\n");
		} // end of i loop
	
		printf("\nPolicy before sweep %u:\n", number_of_sweeps);
		for (i = 0; i < number_of_non_terminal_states; i++)
		{
			printf("%u", i);
			for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
			{
				printf("\t%lf", policy[i][j]);
			} // end of j loop
			printf("\n");
		} // end of i loop
	
		/* Policy evaluation */
		PolicyEvaluation(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, number_of_state_action_successor_states, state_action_successor_state_indices, state_action_successor_state_transition_probabilities, state_action_successor_state_rewards, policy, convergence_threshold, discounting_factor_gamma, maximum_number_of_policy_evaluations, state_value_function, state_action_value_function);
	
		/* Policy improvement */
		policy_stable = PolicyImprovement(number_of_non_terminal_states, number_of_actions_per_non_terminal_state, number_of_state_action_successor_states, state_action_successor_state_indices, state_action_successor_state_transition_probabilities, state_action_successor_state_rewards, policy, old_policy, discounting_factor_gamma, state_value_function);
		
		number_of_sweeps++;
	} while (policy_stable == 0 && number_of_sweeps < maximum_number_of_sweeps);
	
	/*********************************************************************************************************/
	/**************************************** PRINT VALUES AND POLICY ****************************************/
	/*********************************************************************************************************/
	
	printf("Final state value function:\n");
	for (i = 0; i < number_of_states; i++)
	{
		printf("%u\t%lf\n", i, state_value_function[i]);
	} // end of i loop
	
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
		free(old_policy[i]);
		free(policy[i]);
	} // end of i loop
	free(old_policy);
	free(policy);
	
	for (i = 0; i < number_of_states; i++)
	{
		free(state_action_value_function[i]);
	} // end of i loop
	free(state_action_value_function);
	free(state_value_function);
	
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			free(state_action_successor_state_rewards[i][j]);
			free(state_action_successor_state_transition_probabilities[i][j]);
			free(state_action_successor_state_indices[i][j]);
		} // end of j loop
		free(state_action_successor_state_rewards[i]);
		free(state_action_successor_state_transition_probabilities[i]);
		free(state_action_successor_state_indices[i]);
		free(number_of_state_action_successor_states[i]);
	} // end of i loop
	free(state_action_successor_state_rewards);
	free(state_action_successor_state_transition_probabilities);
	free(state_action_successor_state_indices);
	free(number_of_state_action_successor_states);
	free(number_of_actions_per_state);
	free(number_of_actions_per_non_terminal_state);
} // end of main

/*********************************************************************************************************/
/*********************************************** FUNCTIONS ***********************************************/
/*********************************************************************************************************/

/* This function evaluates the value functions given the current policy */
void PolicyEvaluation(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities, double*** state_action_successor_state_rewards, double** policy, double convergence_threshold, double discounting_factor_gamma, unsigned int maximum_number_of_policy_evaluations, double* state_value_function, double** state_action_value_function)
{
	unsigned int i, j, k;
	double delta = 0.0, temp_state_value_function = 0.0, temp_state_action_value_function = 0.0;
	unsigned int number_of_policy_evaluations = 0;
	
	do
	{
		delta = 0.0;
		for (i = 0; i < number_of_non_terminal_states; i++)
		{
			/* Cache state-value function for state i */
			temp_state_value_function = state_value_function[i];
		
			/* Update state-value function based on current policy */
			state_value_function[i] = 0.0;
			for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
			{
				state_action_value_function[i][j] = 0.0;
				for (k = 0; k < number_of_state_action_successor_states[i][j]; k++)
				{
					if (i == state_action_successor_state_indices[i][j][k])
					{
						state_action_value_function[i][j] += state_action_successor_state_transition_probabilities[i][j][k] * (state_action_successor_state_rewards[i][j][k] + discounting_factor_gamma * temp_state_value_function);
					}
					else
					{
						state_action_value_function[i][j] += state_action_successor_state_transition_probabilities[i][j][k] * (state_action_successor_state_rewards[i][j][k] + discounting_factor_gamma * state_value_function[state_action_successor_state_indices[i][j][k]]);
					}
				} // end of k loop
				
				state_value_function[i] += policy[i][j] * state_action_value_function[i][j];
			} // end of j loop
		
			/* Update delta for convergence criteria to break while loop and update policy */
			delta = fmax(delta, fabs(temp_state_value_function - state_value_function[i]));
		} // end of i loop
		
		number_of_policy_evaluations++;
	} while (delta >= convergence_threshold && number_of_policy_evaluations < maximum_number_of_policy_evaluations);
	
	return;
} // end of PolicyEvaluation function

/* This function greedily updates the policy based on the current value function */
int PolicyImprovement(unsigned int number_of_non_terminal_states, unsigned int* number_of_actions_per_non_terminal_state, unsigned int** number_of_state_action_successor_states, unsigned int*** state_action_successor_state_indices, double*** state_action_successor_state_transition_probabilities, double*** state_action_successor_state_rewards, double** policy, double** old_policy, double discounting_factor_gamma, double* state_value_function)
{
	unsigned int i, j, k;
	int policy_stable = 1;
	double max_policy_value = -DBL_MAX;
	unsigned int max_policy_count = 0;
	
	for (i = 0; i < number_of_non_terminal_states; i++)
	{
		/* Cache policy for comparison later */
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			old_policy[i][j] = policy[i][j];
		} // end of j loop
		
		max_policy_value = -DBL_MAX;
		max_policy_count = 0;
	
		/* Update policy greedily from state-value function */
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			policy[i][j] = 0.0;
		
			for (k = 0; k < number_of_state_action_successor_states[i][j]; k++)
			{
				policy[i][j] += state_action_successor_state_transition_probabilities[i][j][k] * (state_action_successor_state_rewards[i][j][k] + discounting_factor_gamma * state_value_function[state_action_successor_state_indices[i][j][k]]);
			} // end of k loop
		
			/* Save max policy value and find the number of actions that have the same max policy value */
			if (policy[i][j] > max_policy_value)
			{
				max_policy_value = policy[i][j];
				max_policy_count = 1;
			}
			else if (policy[i][j] == max_policy_value)
			{
				max_policy_count++;
			}
		} // end of j loop

		/* Apportion policy probability across ties equally for state-action pairs that have the same value and zero otherwise */
		for (j = 0; j < number_of_actions_per_non_terminal_state[i]; j++)
		{
			if (policy[i][j] == max_policy_value)
			{
				policy[i][j] = 1.0 / max_policy_count;
			}
			else
			{
				policy[i][j] = 0.0;
			}

			/* If policy has changed from old policy */
			if (policy[i][j] != old_policy[i][j])
			{
				policy_stable = 0;
			}
		} // end of j loop
	} // end of i loop
	
	return policy_stable;
} // end of PolicyImprovement function
