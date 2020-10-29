#include <stdio.h>      /* printf, scanf, puts */
#include <stdlib.h>     /* realloc, free, exit, NULL */
#include <math.h>
#include <float.h>

/* The type of metric we are using: 0 = mean, 1 = percentile */
int metric_type = 0;

/* If we are using a percentile as our metric, this is the value */
double metric_percentile = 0.5;

/*********************************************************************************/
/********************************** PROTOTYPES ***********************************/
/*********************************************************************************/

/* This function reads the input data */
void ReadInputData(unsigned int number_of_samples, unsigned int number_of_dimensions, double *max_values_per_dimension, double **X);

/* This function randomly initializes centroids */
void RandomlyInitializeCentroids(unsigned int number_of_dimensions, unsigned int number_of_centroids, double **centroids, double *max_values_per_dimension);

/* This function returns a random uniform number within given range */
double RUnif(double range_min, double range_max);

/* This function returns a random uniform number within range [0, 1] */
double UnifRand(void);

/* This function finds the best centroid random initialization */
void FindBestCentroidRandomInitialization(unsigned int number_of_samples, unsigned int number_of_dimensions, unsigned int number_of_centroids, unsigned int number_of_random_initializations, unsigned int max_iterations, double *max_values_per_dimension, double **X, double **centroids, double **initial_centroids, double **temp_centroids, double **best_centroids, unsigned int *closest_centroid, unsigned int *samples_per_centroid, double ***centroid_sample_distance_matrix);

/* This function iterates centroid positions to reduce the cost with the data*/
double IterateCentroids(unsigned int number_of_samples, unsigned int number_of_dimensions, unsigned int number_of_centroids, unsigned int max_iterations, double **X, double **centroids, double **temp_centroids, unsigned int *closest_centroid, unsigned int *samples_per_centroid, double ***centroid_sample_distance_matrix, int post_random_initialization);

/* This function finds the closest centroids */
double FindClosestCentroids(unsigned int number_of_samples, unsigned int number_of_dimensions, unsigned int number_of_centroids, double **X, double **centroids, double **temp_centroids, unsigned int *closest_centroid, unsigned int *samples_per_centroid, double ***centroid_sample_distance_matrix);

/* This function computes centroids */
void ComputeCentroids(unsigned int number_of_centroids, unsigned int number_of_dimensions, double **centroids, double **temp_centroids, unsigned int *samples_per_centroid, double ***centroid_sample_distance_matrix);

/* This function mallocs 1D arrays with unsigned ints */
void Malloc1DArrayUnsignedInt(unsigned int size1, unsigned long memory_unsigned_int, unsigned int initializer, unsigned int **array);

/* This function mallocs 1D arrays with doubles */
void Malloc1DArrayDouble(unsigned int size1, unsigned long memory_double, double initializer, double **array);

/* This function mallocs 2D arrays with doubles */
void Malloc2DArrayDouble(unsigned int size1, unsigned int size2, unsigned long memory_double, unsigned long memory_ptr, double initializer, double ***array);

/* This function mallocs 3D arrays with doubles */
void Malloc3DArrayDouble(unsigned int size1, unsigned int size2, unsigned int size3, unsigned long memory_double, unsigned long memory_ptr, double initializer, double ****array);

/* This function is used for qsort to sort items in increasing order for doubles */
static int IncOrderDouble(const void * a, const void * b);

/*********************************************************************************/
/************************************* MAIN **************************************/
/*********************************************************************************/

int main (int argc, char *argv[])
{
	unsigned int i, j, number_of_centroids = 3, number_of_random_initializations = 10, max_iterations = 500;
	double cost;
	int systemreturn;

	/*********************************************************************************/
	/************************************* SIZES *************************************/
	/*********************************************************************************/

	unsigned int number_of_dimensions;
	FILE *infile_number_of_dimensions = fopen("inputs/number_of_dimensions.txt", "r"); // read only
	systemreturn = fscanf(infile_number_of_dimensions, "%u", &number_of_dimensions);
	if (systemreturn == -1)
	{
		printf("reading inputs/number_of_dimensions.txt failed\n");
	}
	fclose(infile_number_of_dimensions);

	unsigned int number_of_samples;
	FILE *infile_number_of_samples = fopen("inputs/number_of_samples.txt", "r"); // read only
	systemreturn = fscanf(infile_number_of_samples, "%u", &number_of_samples);
	if (systemreturn == -1)
	{
		printf("reading inputs/number_of_samples.txt failed\n");
	}
	fclose(infile_number_of_samples);

	printf("number_of_dimensions = %u, number_of_samples = %u\n", number_of_dimensions, number_of_samples);

	/*********************************************************************************/
	/************************************ INPUTS *************************************/
	/*********************************************************************************/

	double *max_values_per_dimension;
	Malloc1DArrayDouble(number_of_dimensions, sizeof(double), 0, &max_values_per_dimension); // start with none

	double **X;
	Malloc2DArrayDouble(number_of_dimensions, number_of_samples, sizeof(double), sizeof(double*), -9, &X); // start with NULLs

	ReadInputData(number_of_samples, number_of_dimensions, max_values_per_dimension, X);

	/*********************************************************************************/
	/********************************** CENTROIDS ************************************/
	/*********************************************************************************/

	double **centroids;
	Malloc2DArrayDouble(number_of_dimensions, number_of_centroids, sizeof(double), sizeof(double*), -9, &centroids); // start with NULLs

	double **initial_centroids;
	Malloc2DArrayDouble(number_of_dimensions, number_of_centroids, sizeof(double), sizeof(double*), -9, &initial_centroids); // start with NULLs

	double **temp_centroids;
	Malloc2DArrayDouble(number_of_dimensions, number_of_centroids, sizeof(double), sizeof(double*), -9, &temp_centroids); // start with NULLs

	double **best_centroids;
	Malloc2DArrayDouble(number_of_dimensions, number_of_centroids, sizeof(double), sizeof(double*), -9, &best_centroids); // start with NULLs

	unsigned int *closest_centroid;
	Malloc1DArrayUnsignedInt(number_of_samples, sizeof(unsigned int), 0, &closest_centroid); // start with zeroth

	unsigned int *samples_per_centroid;
	Malloc1DArrayUnsignedInt(number_of_centroids, sizeof(unsigned int), 0, &samples_per_centroid); // start with none

	double ***centroid_sample_distance_matrix;
	Malloc3DArrayDouble(number_of_centroids, number_of_dimensions, number_of_samples, sizeof(double), sizeof(double*), DBL_MAX, &centroid_sample_distance_matrix); // start with DBL_MAX

	/*********************************************************************************/
	/*********************************** CLUSTER *************************************/
	/*********************************************************************************/

	if (number_of_random_initializations > 1) // if we are trying more than one random initialization
	{
		FindBestCentroidRandomInitialization(number_of_samples, number_of_dimensions, number_of_centroids, number_of_random_initializations, max_iterations, max_values_per_dimension, X, centroids, initial_centroids, temp_centroids, best_centroids, closest_centroid, samples_per_centroid, centroid_sample_distance_matrix);
	} // end of if we are trying more than one random initialization
	else // if we are NOT trying more than one random initialization
	{
		RandomlyInitializeCentroids(number_of_dimensions, number_of_centroids, centroids, max_values_per_dimension);
	} // end of if we are NOT trying more than one random initialization

	cost = IterateCentroids(number_of_samples, number_of_dimensions, number_of_centroids, max_iterations, X, centroids, temp_centroids, closest_centroid, samples_per_centroid, centroid_sample_distance_matrix, 1);

	/*********************************************************************************/
	/**************************** FREE DYNAMIC MEMEORY *******************************/
	/*********************************************************************************/

	/* Free arrays */
	for (i = 0; i < number_of_centroids; i++)
	{
		for (j = 0; j < number_of_dimensions; j++)
		{
			free(centroid_sample_distance_matrix[i][j]);
		} // end of j loop
		free(centroid_sample_distance_matrix[i]);
	} // end of i loop
	free(centroid_sample_distance_matrix);

	for (i = 0; i < number_of_dimensions; i++)
	{
		free(best_centroids[i]);
		free(temp_centroids[i]);
		free(initial_centroids[i]);
		free(centroids[i]);
		free(X[i]);
	} // end of i loop
	free(samples_per_centroid);
	free(closest_centroid);
	free(best_centroids);
	free(temp_centroids);
	free(initial_centroids);
	free(centroids);
	free(X);
	free(max_values_per_dimension);
} // end of main

/*********************************************************************************/
/*********************************** FUNCTIONS ***********************************/
/*********************************************************************************/

/* This function reads the input data */
void ReadInputData(unsigned int number_of_samples, unsigned int number_of_dimensions, double *max_values_per_dimension, double **X)
{
	unsigned int i, j;
	int systemreturn;

	FILE *infile_X = fopen("inputs/X.txt", "r"); // read only
	for (i = 0; i < number_of_samples; i++)
	{
		for (j = 0; j < number_of_dimensions; j++)
		{
			systemreturn = fscanf(infile_X, "%lf\t", &X[j][i]);
			if (systemreturn == -1)
			{
				printf("reading inputs/X.txt failed\n");
			}

			if (fabs(X[j][i]) > max_values_per_dimension[j])
			{
				max_values_per_dimension[j] = fabs(X[j][i]);
			}
		} // end of j loop
	} // end of i loop
	fclose(infile_X);

	for (j = 0; j < number_of_dimensions; j++)
	{
		printf(" j = %u, max_values_per_dimension[j] = %lf\n", j, max_values_per_dimension[j]);
	}
} // end of ReadInputData function

/* This function randomly initializes centroids */
void RandomlyInitializeCentroids(unsigned int number_of_dimensions, unsigned int number_of_centroids, double **centroids, double *max_values_per_dimension)
{
	int i, j;

	for (i = 0; i < number_of_dimensions; i++)
	{
		for (j = 0; j < number_of_centroids; j++)
		{
			centroids[i][j] = RUnif(-max_values_per_dimension[i], max_values_per_dimension[i]);
		} // end of j loop
	} // end of i loop
} // end of RandomlyInitializeCentroids function

/* This function returns a random uniform number within given range */
double RUnif(double range_min, double range_max)
{
	return range_min + (range_max - range_min) * UnifRand();
} // end of RUnif function

/* This function returns a random uniform number within range [0, 1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
} // end of UnifRand function

/* This function finds the best centroid random initialization */
void FindBestCentroidRandomInitialization(unsigned int number_of_samples, unsigned int number_of_dimensions, unsigned int number_of_centroids, unsigned int number_of_random_initializations, unsigned int max_iterations, double *max_values_per_dimension, double **X, double **centroids, double **initial_centroids, double **temp_centroids, double **best_centroids, unsigned int *closest_centroid, unsigned int *samples_per_centroid, double ***centroid_sample_distance_matrix)
{
	unsigned int i, j, r, best_random = 0;
	double cost, old_cost = DBL_MAX, best_random_cost = DBL_MAX;

	for (r = 0; r < number_of_random_initializations; r++)
	{
		RandomlyInitializeCentroids(number_of_dimensions, number_of_centroids, initial_centroids, max_values_per_dimension);

		for (i = 0; i < number_of_dimensions; i++)
		{
			for (j = 0; j < number_of_centroids; j++)
			{
				centroids[i][j] = initial_centroids[i][j];
			} // end of j loop
		} // end of i loop

		cost = IterateCentroids(number_of_samples, number_of_dimensions, number_of_centroids, max_iterations, X, centroids, temp_centroids, closest_centroid, samples_per_centroid, centroid_sample_distance_matrix, 0);

		printf("Random %u, cost = %.16f\n", r, cost);

		if (cost < best_random_cost)
		{
			best_random_cost = cost;
			best_random = r;

			for (i = 0; i < number_of_dimensions; i++)
			{
				for (j = 0; j < number_of_centroids; j++)
				{
					best_centroids[i][j] = initial_centroids[i][j];
				} // end of j loop
			} // end of i loop
		}
	} // end of r loop

	printf("\nBest random is %u with cost %.16f\n", best_random, best_random_cost);
	for (i = 0; i < number_of_dimensions; i++)
	{
		for (j = 0; j < number_of_centroids; j++)
		{
			centroids[i][j] = best_centroids[i][j];
		} // end of j loop
	} // end of i loop
} // end of FindBestCentroidRandomInitialization function

/* This function iterates centroid positions to reduce the cost with the data*/
double IterateCentroids(unsigned int number_of_samples, unsigned int number_of_dimensions, unsigned int number_of_centroids, unsigned int max_iterations, double **X, double **centroids, double **temp_centroids, unsigned int *closest_centroid, unsigned int *samples_per_centroid, double ***centroid_sample_distance_matrix, int post_random_initialization)
{
	unsigned int i, j, t;
	double cost, old_cost = DBL_MAX;
	int systemreturn;

	for (t = 0; t < max_iterations; t++)
	{
		if (post_random_initialization == 1)
		{
			FILE *outfile_iteration = fopen("outputs/iteration.txt", "w"); // write only
			fprintf(outfile_iteration,"%d\n", t);
			fclose(outfile_iteration);
		}

		cost = FindClosestCentroids(number_of_samples, number_of_dimensions, number_of_centroids, X, centroids, temp_centroids, closest_centroid, samples_per_centroid, centroid_sample_distance_matrix);

		if (post_random_initialization == 1)
		{
			printf("Iteration=%d, Cost=%.12f\n", t, cost);
		}

		if (cost >= old_cost)
		{
			break; // break iteration loop since there is no reduction in cost
		}
		else
		{
			if (post_random_initialization == 1)
			{
				FILE *outfile_kmeans_cost = fopen("outputs/kmeans_cost.txt", "w"); // write only
				fprintf(outfile_kmeans_cost, "%.12f\n", cost);
				fclose(outfile_kmeans_cost);

				FILE *outfile_kmeans_clusters = fopen("outputs/kmeans_clusters.txt", "w"); // write only
				for (i = 0; i < number_of_samples; i++)
				{
					for (j = 0; j < number_of_dimensions; j++)
					{
						fprintf(outfile_kmeans_clusters, "%lf\t", X[j][i]);
					} // end of j loop
					fprintf(outfile_kmeans_clusters, "%d\n", closest_centroid[i]);
				} // end of i loop
				fclose(outfile_kmeans_clusters);

				FILE *outfile_kmeans_centroids = fopen("outputs/kmeans_centroids.txt", "w"); // write only
				for (i = 0; i < number_of_centroids; i++)
				{
					for (j = 0; j < number_of_dimensions; j++)
					{
						fprintf(outfile_kmeans_centroids, "%lf\t", centroids[j][i]);
					} // end of j loop
					fprintf(outfile_kmeans_centroids, "%d\n", i);
				} // end of i loop
				fclose(outfile_kmeans_centroids);

				if (number_of_dimensions == 2)
				{
					systemreturn = system("gnuplot plotscripts/kmeans_clusters.gplot");
					if (systemreturn == -1)
					{
						printf("system gnuplot failed!\n");
					}
				}
				else if (number_of_dimensions == 3)
				{
					systemreturn = system("gnuplot plotscripts/kmeans_clusters_3d.gplot");
					if (systemreturn == -1)
					{
						printf("system gnuplot failed!\n");
					}
				}
			}

			ComputeCentroids(number_of_centroids, number_of_dimensions, centroids, temp_centroids, samples_per_centroid, centroid_sample_distance_matrix);
			old_cost = cost;
		}
	} // end of t loop

	return cost;
} // end of IterateCentroids function

/* This function finds the closest centroids */
double FindClosestCentroids(unsigned int number_of_samples, unsigned int number_of_dimensions, unsigned int number_of_centroids, double **X, double **centroids, double **temp_centroids, unsigned int *closest_centroid, unsigned int *samples_per_centroid, double ***centroid_sample_distance_matrix)
{
	unsigned int i, j, k;
	double dist, min_dist, cost = 0;

	/* Zero out centroid sample counts */
	for (j = 0; j < number_of_centroids; j++)
	{
		samples_per_centroid[j] = 0;
	} // end of j loop

	if (metric_type == 0) // if we are using mean as metric type
	{
		for (i = 0; i < number_of_samples; i++)
		{
			min_dist = DBL_MAX;
			for (j = 0; j < number_of_centroids; j++)
			{
				dist = 0;
				for (k = 0; k < number_of_dimensions; k++)
				{
					dist += ((X[k][i] - centroids[k][j]) * (X[k][i] - centroids[k][j])); // calculate the squared euclidean distance along each dimension
				} // end of k loop

				if (dist < min_dist) // if this distance is less than the current minimum distance
				{
					min_dist = dist;
					closest_centroid[i] = j;
				} // end of if this distance is less than the current minimum distance
			} // end of j loop

			cost += min_dist; // increment the cost with the distance from the ith sample to its closest centroid

			for (j = 0; j < number_of_dimensions; j++)
			{
				temp_centroids[j][closest_centroid[i]] += X[j][i];
			} // end of j loop

			samples_per_centroid[closest_centroid[i]]++; // increment the number of samples assigned to the closest centroid for the ith sample
		} // end of i loop
	}
	else // if we are using percentile as metric type
	{
		unsigned int index0, index1, index2;
		for (i = 0; i < number_of_samples; i++)
		{
			min_dist = DBL_MAX;
			for (j = 0; j < number_of_centroids; j++)
			{
				dist = 0;
				for (k = 0; k < number_of_dimensions; k++)
				{
					dist += fabs(X[k][i] - centroids[k][j]); // calculate the manhattan distance along each dimension
				} // end of k loop

				if (dist < min_dist) // if this distance is less than the current minimum distance
				{
					min_dist = dist;
					closest_centroid[i] = j;
				} // end of if this distance is less than the current minimum distance
			} // end of j loop

			cost += min_dist; // increment the cost with the distance from the ith sample to its closest centroid

			for (j = 0; j < number_of_dimensions; j++)
			{
				index0 = closest_centroid[i];
				index1 = j;
				index2 = samples_per_centroid[closest_centroid[i]];

				centroid_sample_distance_matrix[index0][index1][index2] = X[j][i];
			} // end of j loop

			samples_per_centroid[closest_centroid[i]]++; // increment the number of samples assigned to the closest centroid for the ith sample
		} // end of i loop
	} // end of if we are using percentile as metric type

	cost /= number_of_samples;

	return cost;
} // end of FindClosestCentroids function

/* This function computes centroids */
void ComputeCentroids(unsigned int number_of_centroids, unsigned int number_of_dimensions, double **centroids, double **temp_centroids, unsigned int *samples_per_centroid, double ***centroid_sample_distance_matrix)
{
	unsigned int i, j;
	double npercentilecalc, dpercentilecalc;
	int kpercentilecalc;

	if (metric_type == 0) // if we are using mean as metric type
	{
		for (i = 0; i < number_of_centroids; i++)
		{
			if (samples_per_centroid[i] > 0) // if the ith centroid has at least one sample
			{
				for (j = 0; j < number_of_dimensions; j++)
				{
					centroids[j][i] = temp_centroids[j][i] / samples_per_centroid[i]; // move the centroids to the mean of the samples in its cluster along each dimension
					temp_centroids[j][i] = 0;
				} // end of j loop
			} // end of if the ith centroid has at least one sample
			samples_per_centroid[i] = 0;
		} // end of i loop
	} // end of if we are using mean as metric type
	else // if we are using percentile as metric type
	{
		for (i = 0; i < number_of_centroids; i++)
		{
			if (samples_per_centroid[i] > 0) // if the ith centroid has at least one sample
			{
				for (j = 0; j < number_of_dimensions; j++)
				{
					qsort(centroid_sample_distance_matrix[i][j], samples_per_centroid[i], sizeof(double), IncOrderDouble);

					npercentilecalc = metric_percentile * (samples_per_centroid[i] - 1) + 1;
					kpercentilecalc = npercentilecalc;
					dpercentilecalc = npercentilecalc - kpercentilecalc;

					if (kpercentilecalc == 0)
					{
						centroids[j][i] = centroid_sample_distance_matrix[i][j][0];
					}
					else if (kpercentilecalc == samples_per_centroid[i])
					{
						centroids[j][i] = centroid_sample_distance_matrix[i][j][samples_per_centroid[i] - 1];
					}
					else
					{
						centroids[j][i] = centroid_sample_distance_matrix[i][j][kpercentilecalc - 1] + dpercentilecalc * (centroid_sample_distance_matrix[i][j][kpercentilecalc] - centroid_sample_distance_matrix[i][j][kpercentilecalc - 1]);
					}
				} // end of j loop
			} // end of if the ith centroid has at least one sample
			samples_per_centroid[i] = 0;
		} // end of i loop
	} // end of if we are using percentile as metric type
} // end of ComputeCentroids function

/* This function mallocs 1D arrays with unsigned ints */
void Malloc1DArrayUnsignedInt(unsigned int size1, unsigned long memory_unsigned_int, unsigned int initializer, unsigned int **array)
{
	unsigned int i;

	*array = malloc(memory_unsigned_int * size1);
	for (i = 0; i < size1; i++)
	{
		(*array)[i] = initializer;
	} // end of i loop
} // end of Malloc1DArrayInt function

/* This function mallocs 1D arrays with doubles */
void Malloc1DArrayDouble(unsigned int size1, unsigned long memory_double, double initializer, double **array)
{
	unsigned int i;

	*array = malloc(memory_double * size1);
	for (i = 0; i < size1; i++)
	{
		(*array)[i] = initializer;
	} // end of i loop
} // end of Malloc1DArrayDouble function

/* This function mallocs 2D arrays with doubles */
void Malloc2DArrayDouble(unsigned int size1, unsigned int size2, unsigned long memory_double, unsigned long memory_ptr, double initializer, double ***array)
{
	unsigned int i, j;

	*array = malloc(memory_ptr * size1);
	for (i = 0; i < size1; i++)
	{
		(*array)[i] = malloc(memory_double * size2);
		for (j = 0; j < size2; j++)
		{
			(*array)[i][j] = initializer;
		} // end of j loop
	} // end of i loop
} // end of Malloc2DArrayDouble function

/* This function mallocs 3D arrays with doubles */
void Malloc3DArrayDouble(unsigned int size1, unsigned int size2, unsigned int size3, unsigned long memory_double, unsigned long memory_ptr, double initializer, double ****array)
{
	unsigned int i, j, k;

	*array = malloc(memory_ptr * size1);
	for (i = 0; i < size1; i++)
	{
		(*array)[i] = malloc(memory_ptr * size2);
		for (j = 0; j < size2; j++)
		{
			(*array)[i][j] = malloc(memory_double * size3);
			for (k = 0; k < size3; k++)
			{
				(*array)[i][j][k] = initializer;
			} // end of k loop
		} // end of j loop
	} // end of i loop
} // end of Malloc3DArrayDouble function

/* This function is used for qsort to sort items in increasing order for doubles */
static int IncOrderDouble(const void * a, const void * b)
{
	if (*(double*)a > *(double*)b)
	{
		return 1;
	}
	else if (*(double*)a < *(double*)b)
	{
		return -1;
	}
	else
	{
		return 0;
	}
} // end of IncOrderDouble function
