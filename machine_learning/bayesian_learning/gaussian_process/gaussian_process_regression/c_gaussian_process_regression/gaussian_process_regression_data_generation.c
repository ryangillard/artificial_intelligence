#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function generates observation points of the true unknown function */
void TrueUnknownFunction(unsigned int num_training_points, unsigned int num_dimensions, double noise_variance, double** X_train, double* y);

/* This function returns a random uniform number within given range */
double RUnif(double range_min, double range_max);

/* This function returns a random normal number with given mean and standard deviation */
double RNorm(double mu, double sigma);

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void);

/* This function returns a random normal number with zero mean and unit standard deviation */
double NormRand(void);

/*********************************************************************************************************/
/************************************************* MAIN **************************************************/
/*********************************************************************************************************/

int main(int argc, char* argv[])
{
	int i, j, system_return = 0;
	
	/* Read inputs */
	/* Get the number of training points */
	unsigned int num_training_points = 0;
	
	FILE* infile_num_training_points = fopen("inputs/num_training_points.txt", "r");
	system_return = fscanf(infile_num_training_points, "%u", &num_training_points);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/num_training_points.txt\n");
	}
	fclose(infile_num_training_points);
	printf("num_training_points = %u\n", num_training_points);
	
	/* Get the number of test points */
	unsigned int num_test_points = 0;
	
	FILE* infile_num_test_points = fopen("inputs/num_test_points.txt", "r");
	system_return = fscanf(infile_num_test_points, "%u", &num_test_points);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/num_test_points.txt\n");
	}
	fclose(infile_num_test_points);
	printf("num_test_points = %u\n", num_test_points);
	
	/* Get the number of test points */
	unsigned int num_dimensions = 0;
	
	FILE* infile_num_dimensions = fopen("inputs/num_dimensions.txt", "r");
	system_return = fscanf(infile_num_dimensions, "%u", &num_dimensions);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/num_dimensions.txt\n");
	}
	fclose(infile_num_dimensions);
	printf("num_dimensions = %u\n", num_dimensions);
	
	/* Get noise variance */
	double noise_variance = 0.0;
	
	FILE* infile_noise_variance = fopen("inputs/noise_variance.txt", "r");
	system_return = fscanf(infile_noise_variance, "%lf", &noise_variance);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/noise_variance.txt\n");
	}
	fclose(infile_noise_variance);
	printf("noise_variance = %lf\n", noise_variance);
	
	/* Sample some input points and noisy versions of the function evaluated at these points */
	
	/* Now create X_train matrix */
	FILE* outfile_X_train = fopen("inputs/X_train.txt", "w");
	
	double** X_train;

	X_train = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		X_train[i] = malloc(sizeof(double) * num_dimensions);
		for (j = 0; j < num_dimensions; j++)
		{
			X_train[i][j] = RUnif(-5.0, 5.0);
			fprintf(outfile_X_train, "%lf\t", X_train[i][j]);
		} // end of j loop
		fprintf(outfile_X_train, "\n");
	} // end of i loop
	fclose(outfile_X_train);
	
	/* Now create noisy values vector y */
	double* y;
	y = malloc(sizeof(double) * num_training_points);
	
	/* Calculate noisy true function values */
	TrueUnknownFunction(num_training_points, num_dimensions, noise_variance, X_train, y);
	
	FILE* outfile_y = fopen("inputs/y.txt", "w");
	for (i = 0; i < num_training_points; i++)
	{
		fprintf(outfile_y, "%lf\n", y[i]);
	} // end of i loop
	fclose(outfile_y);
	
	/* Sample some test points that we are going to make predictions at */
	
	/* Now create X_test matrix */
	FILE* outfile_X_test = fopen("inputs/X_test.txt", "w");
	
	double** X_test;

	X_test = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		X_test[i] = malloc(sizeof(double) * num_dimensions);
		for (j = 0; j < num_dimensions; j++)
		{
			X_test[i][j] = RUnif(-5.0, 5.0);
			fprintf(outfile_X_test, "%lf\t", X_test[i][j]);
		} // end of j loop
		fprintf(outfile_X_test, "\n");
	} // end of i loop
	fclose(outfile_X_test);
	
	/* Free dynamically allocated memory */
	for (i = 0; i < num_test_points; i++)
	{
		free(X_test[i]);
	} // end of i loop
	free(X_test);
	
	free(y);
	
	for (i = 0; i < num_training_points; i++)
	{
		free(X_train[i]);
	} // end of i loop
	free(X_train);
	
	return 0;
} // end of main

/*********************************************************************************************************/
/*********************************************** FUNCTIONS ***********************************************/
/*********************************************************************************************************/

/* This function generates observation points of the true unknown function */
void TrueUnknownFunction(unsigned int num_training_points, unsigned int num_dimensions, double noise_variance, double** X_train, double* y)
{
	unsigned int i, j;

	for (i = 0; i < num_training_points; i++)
	{
		y[i] = noise_variance * RNorm(0.0, 1.0);
		for (j = 0; j < num_dimensions; j++)
		{
			y[i] += sin(0.9 * X_train[i][j]);
		} // end of j loop
	} // end of i loop
	
	return;
} // end of TrueUnknownFunction function

/* This function returns a random uniform number within given range */
double RUnif(double range_min, double range_max)
{
	return range_min + (range_max - range_min) * UnifRand();
} // end of RUnif function

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

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function

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