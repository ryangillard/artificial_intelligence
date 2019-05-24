#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function loops through iterations and updates the policy */
void LoopThroughIterations(unsigned int number_of_iterations, unsigned int number_of_bandits, double* bandit_mean, double* bandit_variance, unsigned int* bandit_stochastic_change_frequencies, unsigned int* bandit_stochastic_change_counter, double global_bandit_mean_mean, double global_bandit_mean_variance, double global_bandit_variance_mean, double global_bandit_variance_variance, double* action_preference, double* policy, double* policy_cumulative_sum, double alpha, int average_reward_update_type);

/* This function updates policy based on action preference */
void UpdatePolicyFromActionPreference(unsigned int number_of_bandits, double* action_preference, double* policy, double* policy_cumulative_sum);

/* This function applies the softmax function */
void ApplySoftmaxFunction(unsigned int number_of_bandits, double* action_preference, double* policy);

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
	
	/* Create action preference function array */
	double* action_preference;
	action_preference = malloc(sizeof(double) * number_of_bandits);
	for (i = 0; i < number_of_bandits; i++)
	{
		action_preference[i] = 0.0;
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
	
	/* Get average reward update type (sample-average, constant step-size) */
	int average_reward_update_type = 0;
	
	FILE* infile_average_reward_update_type = fopen("inputs/average_reward_update_type.txt", "r");
	system_return = fscanf(infile_average_reward_update_type, "%d", &average_reward_update_type);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/average_reward_update_type.txt\n");
	}
	fclose(infile_average_reward_update_type);
	
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
	
	printf("\nInitial action preference function:\n");
	printf("i\tH\n");
	for (i = 0; i < number_of_bandits; i++)
	{
		printf("%u\t%lf\n", i, action_preference[i]);
	} // end of i loop
	
	printf("\nInitial policy:\n");
	printf("i\tpi\n");
	for (i = 0; i < number_of_bandits; i++)
	{
		printf("%u\t%lf\n", i, policy[i]);
	} // end of i loop
	
	/* This function loops through iterations and updates the policy */
	LoopThroughIterations(number_of_iterations, number_of_bandits, bandit_mean, bandit_variance, bandit_stochastic_change_frequencies, bandit_stochastic_change_counter, global_bandit_mean_mean, global_bandit_mean_variance, global_bandit_variance_mean, global_bandit_variance_variance, action_preference, policy, policy_cumulative_sum, alpha, average_reward_update_type);
	
	/*********************************************************************************************************/
	/************************************* PRINT PREFERENCES AND POLICIES ************************************/
	/*********************************************************************************************************/

	printf("\nFinal bandit properties:\n");
	printf("i\tmean\tvar\tchg_frq\n");
	for (i = 0; i < number_of_bandits; i++)
	{
		printf("%u\t%lf\t%lf\t%u\n", i, bandit_mean[i], bandit_variance[i], bandit_stochastic_change_frequencies[i]);
	} // end of i loop

	printf("\nFinal action preference function:\n");
	printf("i\tH\n");
	for (i = 0; i < number_of_bandits; i++)
	{
		printf("%u\t%lf\n", i, action_preference[i]);
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
	free(action_preference);
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
void LoopThroughIterations(unsigned int number_of_iterations, unsigned int number_of_bandits, double* bandit_mean, double* bandit_variance, unsigned int* bandit_stochastic_change_frequencies, unsigned int* bandit_stochastic_change_counter, double global_bandit_mean_mean, double global_bandit_mean_variance, double global_bandit_variance_mean, double global_bandit_variance_variance, double* action_preference, double* policy, double* policy_cumulative_sum, double alpha, int average_reward_update_type)
{
	unsigned int t, i;
	unsigned int action_index;
	double probability, reward, average_reward = 0.0;
		
	/* Loop through iterations until termination */
	for (t = 0; t < number_of_iterations; t++)
	{
		/* Choose policy by taking softmax of action preferences */
		UpdatePolicyFromActionPreference(number_of_bandits, action_preference, policy, policy_cumulative_sum);
		
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
		
		/* Update average reward */
		if (average_reward_update_type == 0) // sample-average method
		{
			average_reward += 1.0 / (t + 1) * (reward - average_reward);
		}
		else if (average_reward_update_type == 1) // constant step-size
		{
			average_reward += alpha * (reward - average_reward);
		}
		
		/* Update action preference */
		for (i = 0; i < number_of_bandits; i++)
		{
			if (i == action_index)
			{
				action_preference[i] += alpha * (reward - average_reward) * (1.0 - policy[i]);
			}
			else
			{
				action_preference[i] -= alpha * (reward - average_reward) * policy[i];
			}
		} // end of i loop
		
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

/* This function updates policy based on action preference */
void UpdatePolicyFromActionPreference(unsigned int number_of_bandits, double* action_preference, double* policy, double* policy_cumulative_sum)
{
	unsigned int i;
	
	/* Calculate probabilities by taking softmax of action preferences */
	ApplySoftmaxFunction(number_of_bandits, action_preference, policy);
	
	/* Update policy cumulative sum */
	policy_cumulative_sum[0] = policy[0];
	for (i = 1; i < number_of_bandits; i++)
	{
		policy_cumulative_sum[i] = policy_cumulative_sum[i - 1] + policy[i];
	} // end of i loop
	
	return;
} // end of UpdatePolicyFromActionPreference function

/* This function applies the softmax function */
void ApplySoftmaxFunction(unsigned int number_of_bandits, double* action_preference, double* policy)
{
	/* f(xi) = e^(xi - max(x)) / sum(e^(xj - max(x)), j, 0, n - 1) */

	unsigned int i;
	double max_logit, denominator_sum;

	max_logit = -DBL_MAX;
	for (i = 0; i < number_of_bandits; i++)
	{
		if (action_preference[i] > max_logit)
		{
			max_logit = action_preference[i];
		}
	} // end of i loop

	denominator_sum = 0.0;
	for (i = 0; i < number_of_bandits; i++)
	{
		/* Shift logits by the max logit to make numerically stable */
		policy[i] = exp(action_preference[i] - max_logit);
		denominator_sum += policy[i];
	} // end of i loop

	for (i = 0; i < number_of_bandits; i++)
	{
		policy[i] /= denominator_sum;
	} // end of i loop
	
	return;
} // end of ApplySoftmaxFunction function

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