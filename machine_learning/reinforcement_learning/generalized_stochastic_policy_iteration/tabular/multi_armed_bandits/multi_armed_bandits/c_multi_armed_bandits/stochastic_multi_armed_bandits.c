#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function loops through iterations and updates the policy */
void LoopThroughIterations(unsigned int number_of_iterations, unsigned int number_of_bandits, double* bandit_mean, double* bandit_variance, unsigned int* bandit_stochastic_change_frequencies, unsigned int* bandit_stochastic_change_counter, double global_bandit_mean_mean, double global_bandit_mean_variance, double global_bandit_variance_mean, double global_bandit_variance_variance, double* action_value_function, unsigned int* action_count, double* action_trace, double* policy, double* policy_cumulative_sum, double alpha, double epsilon, int action_selection_type, int action_value_update_type);

/* This function updates policy as some function of action-value function */
void UpdatePolicyFromActionValueFunction(unsigned int number_of_bandits, double* action_value_function, unsigned int* action_count, unsigned int iteration_count, double epsilon, int action_selection_type, double* policy, double* policy_cumulative_sum);

/* This function returns a random normal number with given mean and standard deviation */
double RNorm(double mu, double sigma);

/* This function returns a random normal number with zero mean and unit standard deviation */
double NormRand(void);

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void);

/*********************************************************************************************************/
/************************************************* MAIN **************************************************/
/*********************************************************************************************************/

int main(int argc, char* argv[])
{
	unsigned int i;
	int system_return;
	
	/*********************************************************************************************************/
	/**************************************** READ IN THE ENVIRONMENT ****************************************/
	/*********************************************************************************************************/
	
	/* Get the number of bandits */
	unsigned int number_of_bandits = 0;
	
	FILE* infile_number_of_bandits = fopen("inputs/number_of_bandits.txt", "r");
	system_return = fscanf(infile_number_of_bandits, "%u", &number_of_bandits);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/number_of_bandits.txt\n");
	}
	fclose(infile_number_of_bandits);

	/* Get the global bandit's mean for the means of each bandit */
	double global_bandit_mean_mean = 0.0;
	
	FILE* infile_global_bandit_mean_mean = fopen("inputs/global_bandit_mean_mean.txt", "r");
	system_return = fscanf(infile_global_bandit_mean_mean, "%lf", &global_bandit_mean_mean);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/global_bandit_mean_mean.txt\n");
	}
	fclose(infile_global_bandit_mean_mean);
	
	/* Get the global bandit's variance for the means of each bandit */
	double global_bandit_mean_variance = 0.0;
	
	FILE* infile_global_bandit_mean_variance = fopen("inputs/global_bandit_mean_variance.txt", "r");
	system_return = fscanf(infile_global_bandit_mean_variance, "%lf", &global_bandit_mean_variance);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/global_bandit_mean_variance.txt\n");
	}
	fclose(infile_global_bandit_mean_variance);
	
	double* bandit_mean;
	bandit_mean = malloc(sizeof(double) * number_of_bandits);
	for (i = 0; i < number_of_bandits; i++)
	{
		bandit_mean[i] = RNorm(global_bandit_mean_mean, sqrt(global_bandit_mean_variance));
	} // end of i loop
	
	/* Get the global bandit's mean for the variances of each bandit */
	double global_bandit_variance_mean = 0.0;
	
	FILE* infile_global_bandit_variance_mean = fopen("inputs/global_bandit_variance_mean.txt", "r");
	system_return = fscanf(infile_global_bandit_variance_mean, "%lf", &global_bandit_variance_mean);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/global_bandit_variance_mean.txt\n");
	}
	fclose(infile_global_bandit_variance_mean);
	
	/* Get the global bandit's variance for the variances of each bandit */
	double global_bandit_variance_variance = 0.0;
	
	FILE* infile_global_bandit_variance_variance = fopen("inputs/global_bandit_variance_variance.txt", "r");
	system_return = fscanf(infile_global_bandit_variance_variance, "%lf", &global_bandit_variance_variance);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/global_bandit_variance_variance.txt\n");
	}
	fclose(infile_global_bandit_variance_variance);
	
	double* bandit_variance;
	bandit_variance = malloc(sizeof(double) * number_of_bandits);
	for (i = 0; i < number_of_bandits; i++)
	{
		bandit_variance[i] = RNorm(global_bandit_variance_mean, sqrt(global_bandit_variance_variance));
	} // end of i loop

	unsigned int* bandit_stochastic_change_frequencies;
	bandit_stochastic_change_frequencies = malloc(sizeof(unsigned int) * number_of_bandits);
	
	FILE* infile_bandit_stochastic_change_frequencies = fopen("inputs/bandit_stochastic_change_frequencies.txt", "r");
	for (i = 0; i < number_of_bandits; i++)
	{
		system_return = fscanf(infile_bandit_stochastic_change_frequencies, "%u\n", &bandit_stochastic_change_frequencies[i]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/bandit_stochastic_change_frequencies.txt\n");
		}
	} // end of i loop
	fclose(infile_bandit_stochastic_change_frequencies);
	
	unsigned int* bandit_stochastic_change_counter;
	bandit_stochastic_change_counter = malloc(sizeof(unsigned int) * number_of_bandits);
	for (i = 0; i < number_of_bandits; i++)
	{
		bandit_stochastic_change_counter[i] = 0;
	} // end of i loop
	
	/*********************************************************************************************************/
	/**************************************** SETUP BANDIT ITERATION *****************************************/
	/*********************************************************************************************************/
	
	/* Set the number of iterations */
	unsigned int number_of_iterations = 2000;
	
	/* Create action-value function array */
	double* action_value_function;
	action_value_function = malloc(sizeof(double) * number_of_bandits);
	for (i = 0; i < number_of_bandits; i++)
	{
		action_value_function[i] = 0.0;
	} // end of i loop
	
	/* Create action count array */
	unsigned int* action_count;
	action_count = malloc(sizeof(unsigned int) * number_of_bandits);
	for (i = 0; i < number_of_bandits; i++)
	{
		action_count[i] = 0;
	} // end of i loop
	
	/* Create action trace array */
	double* action_trace;
	action_trace = malloc(sizeof(double) * number_of_bandits);
	for (i = 0; i < number_of_bandits; i++)
	{
		action_trace[i] = 0.0;
	} // end of i loop
	
	/* Create policy array */
	double* policy;
	policy = malloc(sizeof(double) * number_of_bandits);
	for (i = 0; i < number_of_bandits; i++)
	{
		policy[i] = 1.0 / number_of_bandits;
	} // end of i loop
	
	/* Create policy cumulative sum array */
	double* policy_cumulative_sum;
	policy_cumulative_sum = malloc(sizeof(double) * number_of_bandits);
  	policy_cumulative_sum[0] = policy[0];
	for (i = 1; i < number_of_bandits; i++)
	{
		policy_cumulative_sum[i] = policy_cumulative_sum[i - 1] + policy[i];
	} // end of i loop
	
	/* Set learning rate alpha */
	double alpha = 0.1;
	
	/* Set epsilon for our epsilon level of exploration */
	double epsilon = 0.1;
	
	/* Get action selection type (greedy, epsilon-greedy, upper-confidence-bound) */
	int action_selection_type = 0;
	
	FILE* infile_action_selection_type = fopen("inputs/action_selection_type.txt", "r");
	system_return = fscanf(infile_action_selection_type, "%d", &action_selection_type);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/action_selection_type.txt\n");
	}
	fclose(infile_action_selection_type);
	
	/* Get action value update type (sample-average, biased constant step-size, unbiased constant step-size) */
	int action_value_update_type = 0;
	
	FILE* infile_action_value_update_type = fopen("inputs/action_value_update_type.txt", "r");
	system_return = fscanf(infile_action_value_update_type, "%d", &action_value_update_type);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/action_value_update_type.txt\n");
	}
	fclose(infile_action_value_update_type);
	
	/* Set random seed */
	srand(0);
	
	/*********************************************************************************************************/
	/****************************************** RUN POLICY CONTROL *******************************************/
	/*********************************************************************************************************/

	printf("\nInitial bandit properties:\n");
	printf("i\tmean\tvar\tchg_frq\n");
	for (i = 0; i < number_of_bandits; i++)
	{
		printf("%u\t%lf\t%lf\t%u\n", i, bandit_mean[i], bandit_variance[i], bandit_stochastic_change_frequencies[i]);
	} // end of i loop
	
	printf("\nInitial action value function:\n");
	printf("i\tQ\n");
	for (i = 0; i < number_of_bandits; i++)
	{
		printf("%u\t%lf\n", i, action_value_function[i]);
	} // end of i loop
	
	printf("\nInitial policy:\n");
	printf("i\tpi\n");
	for (i = 0; i < number_of_bandits; i++)
	{
		printf("%u\t%lf\n", i, policy[i]);
	} // end of i loop
	
	/* This function loops through iterations and updates the policy */
	LoopThroughIterations(number_of_iterations, number_of_bandits, bandit_mean, bandit_variance, bandit_stochastic_change_frequencies, bandit_stochastic_change_counter, global_bandit_mean_mean, global_bandit_mean_variance, global_bandit_variance_mean, global_bandit_variance_variance, action_value_function, action_count, action_trace, policy, policy_cumulative_sum, alpha, epsilon, action_selection_type, action_value_update_type);
	
	/*********************************************************************************************************/
	/**************************************** PRINT VALUES AND POLICIES **************************************/
	/*********************************************************************************************************/

	printf("\nFinal bandit properties:\n");
	printf("i\tmean\tvar\tchg_frq\n");
	for (i = 0; i < number_of_bandits; i++)
	{
		printf("%u\t%lf\t%lf\t%u\n", i, bandit_mean[i], bandit_variance[i], bandit_stochastic_change_frequencies[i]);
	} // end of i loop

	printf("\nFinal action value function:\n");
	printf("i\tQ\n");
	for (i = 0; i < number_of_bandits; i++)
	{
		printf("%u\t%lf\n", i, action_value_function[i]);
	} // end of i loop
	
	printf("\nFinal policy:\n");
	printf("i\tpi\n");
	for (i = 0; i < number_of_bandits; i++)
	{
		printf("%u\t%lf\n", i, policy[i]);
	} // end of i loop
	
	/*********************************************************************************************************/
	/****************************************** FREE DYNAMIC MEMORY ******************************************/
	/*********************************************************************************************************/

	free(policy_cumulative_sum);
	free(policy);
	free(action_trace);
	free(action_count);
	free(action_value_function);
	free(bandit_stochastic_change_counter);
	free(bandit_stochastic_change_frequencies);
	free(bandit_variance);
	free(bandit_mean);
	
	return 0;
} // end of main

/*********************************************************************************************************/
/*********************************************** FUNCTIONS ***********************************************/
/*********************************************************************************************************/

/* This function loops through iterations and updates the policy */
void LoopThroughIterations(unsigned int number_of_iterations, unsigned int number_of_bandits, double* bandit_mean, double* bandit_variance, unsigned int* bandit_stochastic_change_frequencies, unsigned int* bandit_stochastic_change_counter, double global_bandit_mean_mean, double global_bandit_mean_variance, double global_bandit_variance_mean, double global_bandit_variance_variance, double* action_value_function, unsigned int* action_count, double* action_trace, double* policy, double* policy_cumulative_sum, double alpha, double epsilon, int action_selection_type, int action_value_update_type)
{
	unsigned int t, i;
	unsigned int action_index;
	double probability, reward;
		
	/* Loop through iterations until termination */
	for (t = 0; t < number_of_iterations; t++)
	{
		/* Choose policy by epsilon-greedy choosing from the action-value function */
		UpdatePolicyFromActionValueFunction(number_of_bandits, action_value_function, action_count, t + 1, epsilon, action_selection_type, policy, policy_cumulative_sum);
		
		/* Get action */
		probability = UnifRand();
		for (i = 0; i < number_of_bandits; i++)
		{
			if (probability <= policy_cumulative_sum[i])
			{
				action_index = i;
				break; // break i loop since we found our index
			}
		} // end of i loop
		
		/* Get reward from action */
		reward = RNorm(bandit_mean[action_index], bandit_variance[action_index]);
		
		/* Update action count */
		action_count[action_index]++;
		
		/* Update action-value function */
		if (action_value_update_type == 0) // sample-average method
		{
			action_value_function[action_index] += (1.0 / action_count[action_index]) * (reward - action_value_function[action_index]);
		}
		else if (action_value_update_type == 1) // biased constant step-size
		{
			action_value_function[action_index] += alpha * (reward - action_value_function[action_index]);
		}
		else if (action_value_update_type == 2) // unbiased constant step-size
		{
			/* Update action trace */
			action_trace[action_index] += alpha * (1.0 - action_trace[action_index]);

			action_value_function[action_index] += (alpha / action_trace[action_index]) * (reward - action_value_function[action_index]);
		}
		
		/* Mutate bandit statistics */
		for (i = 0; i < number_of_bandits; i++)
		{
			if (bandit_stochastic_change_frequencies[i] > 0)
			{
				bandit_stochastic_change_counter[i]++;
				
				if (bandit_stochastic_change_counter[i] == bandit_stochastic_change_frequencies[i])
				{
					bandit_mean[i] = RNorm(global_bandit_mean_mean, sqrt(global_bandit_mean_variance));
					bandit_variance[i] = RNorm(global_bandit_variance_mean, sqrt(global_bandit_variance_variance));
					
					bandit_stochastic_change_counter[i] = 0;
				}
			}
		} // end of i loop
	} // end of t loop
	
	return;
} // end of LoopThroughIterations 

/* This function updates policy as some function of action-value function */
void UpdatePolicyFromActionValueFunction(unsigned int number_of_bandits, double* action_value_function, unsigned int* action_count, unsigned int iteration_count, double epsilon, int action_selection_type, double* policy, double* policy_cumulative_sum)
{
	unsigned int i, max_action_count = 1;
	double action_value = 0.0, max_action_value = -DBL_MAX, max_policy_apportioned_probability_per_action = 1.0, remaining_apportioned_probability_per_action = 0.0;
	
	/* Update policy from value function */
	for (i = 0; i < number_of_bandits; i++)
	{
		/* Calculate action value depending on action selection type */
		if (action_selection_type == 0 || action_selection_type == 1) // greedy or epsilon-greedy
		{
			action_value = action_value_function[i];
		}
		else if (action_selection_type == 2) // upper-confidence-bound
		{
			if (action_count[i] == 0)
			{
				action_value = DBL_MAX;
				max_action_value = action_value;
				break;
			}
			else
			{
				action_value = action_value_function[i] + epsilon * sqrt(log((double)iteration_count) / action_count[i]);
			}
		}
		
		/* Save max action value and find the number of actions that have the same max action value */
		if (action_value > max_action_value)
		{
			max_action_value = action_value;
			max_action_count = 1;
		}
		else if (action_value == max_action_value)
		{
			max_action_count++;
		}
	} // end of i loop
	
	/* Apportion policy probability across ties equally for action pairs that have the same value and spread out epsilon otherwise */
	if (action_selection_type == 1) // epsilon-greedy
	{
		if (max_action_count == number_of_bandits)
		{
			max_policy_apportioned_probability_per_action = 1.0 / max_action_count;
			remaining_apportioned_probability_per_action = 0.0;
		}
		else
		{
			max_policy_apportioned_probability_per_action = (1.0 - epsilon) / max_action_count;
			remaining_apportioned_probability_per_action = epsilon / (number_of_bandits - max_action_count);
		}
	}
	else if (action_selection_type == 0 || action_selection_type == 2) // greedy or upper-confidence-bound
	{
		max_policy_apportioned_probability_per_action = 1.0 / max_action_count;
		remaining_apportioned_probability_per_action = 0.0;		
	}
	
	/* Update policy with our apportioned probabilities */
	for (i = 0; i < number_of_bandits; i++)
	{
		/* Calculate action value depending on action selection type */
		if (action_selection_type == 0 || action_selection_type == 1) // greedy or epsilon-greedy
		{
			action_value = action_value_function[i];
		}
		else if (action_selection_type == 2) // upper-confidence-bound
		{
			if (action_count[i] == 0)
			{
				action_value = DBL_MAX;
			}
			else
			{
				action_value = action_value_function[i] + epsilon * sqrt(log((double)iteration_count) / action_count[i]);
			}
		}
		
		if (action_value == max_action_value)
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
	for (i = 1; i < number_of_bandits; i++)
	{
		policy_cumulative_sum[i] = policy_cumulative_sum[i - 1] + policy[i];
	} // end of i loop
	
	return;
} // end of UpdatePolicyFromActionValueFunction function

/* This function returns a random normal number with given mean and standard deviation */
double RNorm(double mu, double sigma)
{
	if (sigma == 0.0)
	{
		return mu;
	}
	else
	{
		return mu + sigma * NormRand();
	}
} // end of RNorm function

/* This function returns a random normal number with zero mean and unit standard deviation */
double NormRand(void)
{
	const static double a[32] =
	{
		0.0000000, 0.03917609, 0.07841241, 0.1177699,
		0.1573107, 0.19709910, 0.23720210, 0.2776904,
		0.3186394, 0.36012990, 0.40225010, 0.4450965,
		0.4887764, 0.53340970, 0.57913220, 0.6260990,
		0.6744898, 0.72451440, 0.77642180, 0.8305109,
		0.8871466, 0.94678180, 1.00999000, 1.0775160,
		1.1503490, 1.22985900, 1.31801100, 1.4177970,
		1.5341210, 1.67594000, 1.86273200, 2.1538750
	};

	const static double d[31] =
	{
		0.0000000, 0.0000000, 0.0000000, 0.0000000,
		0.0000000, 0.2636843, 0.2425085, 0.2255674,
		0.2116342, 0.1999243, 0.1899108, 0.1812252,
		0.1736014, 0.1668419, 0.1607967, 0.1553497,
		0.1504094, 0.1459026, 0.1417700, 0.1379632,
		0.1344418, 0.1311722, 0.1281260, 0.1252791,
		0.1226109, 0.1201036, 0.1177417, 0.1155119,
		0.1134023, 0.1114027, 0.1095039
	};

	const static double t[31] =
	{
		7.673828e-4, 0.002306870, 0.003860618, 0.005438454,
		0.007050699, 0.008708396, 0.010423570, 0.012209530,
		0.014081250, 0.016055790, 0.018152900, 0.020395730,
		0.022811770, 0.025434070, 0.028302960, 0.031468220,
		0.034992330, 0.038954830, 0.043458780, 0.048640350,
		0.054683340, 0.061842220, 0.070479830, 0.081131950,
		0.094624440, 0.112300100, 0.136498000, 0.171688600,
		0.227624100, 0.330498000, 0.584703100
	};

	const static double h[31] =
	{
		0.03920617, 0.03932705, 0.03950999, 0.03975703,
		0.04007093, 0.04045533, 0.04091481, 0.04145507,
		0.04208311, 0.04280748, 0.04363863, 0.04458932,
		0.04567523, 0.04691571, 0.04833487, 0.04996298,
		0.05183859, 0.05401138, 0.05654656, 0.05953130,
		0.06308489, 0.06737503, 0.07264544, 0.07926471,
		0.08781922, 0.09930398, 0.11555990, 0.14043440,
		0.18361420, 0.27900160, 0.70104740
	};

	double s, u1, w, y, u2, aa, tt;
	int i;

	u1 = UnifRand();
	s = 0.0;
	if (u1 > 0.5)
	{
		s = 1.0;
	}
	u1 = u1 + u1 - s;
	u1 *= 32.0;
	i = (int) u1;
	if (i == 32)
	{
		i = 31;
	}

	if (i != 0)
	{
		u2 = u1 - i;
		aa = a[i - 1];
		while (u2 <= t[i - 1])
		{
			u1 = UnifRand();
			w = u1 * (a[i] - aa);
			tt = (w * 0.5 + aa) * w;
			for(;;)
			{
				if (u2 > tt)
				{
					goto deliver;
				}
				u1 = UnifRand();
				if (u2 < u1)
				{
					break;
				}
				tt = u1;
				u2 = UnifRand();
			}
			u2 = UnifRand();
		}
		w = (u2 - t[i - 1]) * h[i - 1];
	}
	else
	{
		i = 6;
		aa = a[31];
		for(;;)
		{
			u1 = u1 + u1;
			if (u1 >= 1.0)
			{
				break;
			}
			aa = aa + d[i - 1];
			i = i + 1;
		}
		u1 = u1 - 1.0;
		for(;;)
		{
			w = u1 * d[i - 1];
			tt = (w * 0.5 + aa) * w;
			for(;;)
			{
				u2 = UnifRand();
				if (u2 > tt)
				{
					goto jump;
				}
				u1 = UnifRand();
				if (u2 < u1)
				{
					break;
				}
				tt = u1;
			}
			u1 = UnifRand();
		}
		jump:;
	}

	deliver:
	y = aa + w;
	return (s == 1.0) ? -y : y;
}	// end of NormRand function

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function