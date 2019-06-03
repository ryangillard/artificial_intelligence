#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function calculates various similarity kernels between two matrices */
void CalculateKernel(int kernel_type, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double constant, double length_scale, double signal_variance, double noise_variance, double** kernel);

/* This function applies the linear kernel between two matrices */
void LinearKernel(unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double constant, double** kernel);

/* This function applies the squared exponential kernel between two matrices */
void SquaredExponentialKernel(unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double length_scale, double signal_variance, double noise_variance, double** kernel);

/* This function performs the Cholesky decomposition A = L * L**T */
int CholeskyDecomposition(int n, double** L);

/* This function computes the Cholesky factorization of a real symmetric positive definite matrix A using the recursive algorithm. */
int RecursiveCholeskyFactorization(int n, double** a, int a_row_offset, int a_col_offset);

/* This function solves the matrix equation X * A**T = B for triangular matrix A */
int SolveTriangularMatrixEquation(int left_side, int transa, int m, int n, double** a, double** b, int read_row_offset, int read_col_offset, int write_row_offset, int write_col_offset);

/* This function performs the symmetric rank k operation C := -A * A**T + C */
int SymmetricRankKOperation(int n, int k, double** a, int read_row_offset, int read_col_offset, int write_row_offset, int write_col_offset);

/* This solves a system of linear equations A * X = B with A = L*L**T */
int SolveLowerCholeskyFactorizationMatrixEquation(int n, int nrhs, double** a, double** b);

/* This function performs matrix multiplication between two given matrices */
void MatrixMultiplication(unsigned int m, unsigned int n, unsigned int p, double** A, double** B, int transpose_A, int transpose_B, double** C);

/* This function performs the dot product between two given vectors that are in 2D form */
double VectorDotProductRank2(unsigned int n, double** A, double** B, int a_col_vec, int b_col_vec);

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void);

/* This function returns a random normal number with given mean and standard deviation */
double RNorm(double mu, double sigma);

/* This function returns a random normal number with zero mean and unit standard deviation */
double NormRand(void);

/*********************************************************************************************************/
/************************************************* MAIN **************************************************/
/*********************************************************************************************************/

int main(int argc, char* argv[])
{
	int i, j, system_return = 0, error = 0;
	srand(12345);
	
	/*********************************************************************************************************/
	/********************************************* READ INPUTS ***********************************************/
	/*********************************************************************************************************/
	
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
	
	/* Get the number of dimensions */
	unsigned int num_dimensions = 0;
	
	FILE* infile_num_dimensions = fopen("inputs/num_dimensions.txt", "r");
	system_return = fscanf(infile_num_dimensions, "%u", &num_dimensions);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/num_dimensions.txt\n");
	}
	fclose(infile_num_dimensions);
	printf("num_dimensions = %u\n", num_dimensions);
	
	/* Get the number of samples */
	unsigned int num_samples = 0;
	
	FILE* infile_num_samples = fopen("inputs/num_samples.txt", "r");
	system_return = fscanf(infile_num_samples, "%u", &num_samples);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/num_samples.txt\n");
	}
	fclose(infile_num_samples);
	printf("num_samples = %u\n", num_samples);
	
	/* Get the kernel type */
	int kernel_type = 0;
	
	FILE* infile_kernel_type = fopen("inputs/kernel_type.txt", "r");
	system_return = fscanf(infile_kernel_type, "%d", &kernel_type);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/kernel_type.txt\n");
	}
	fclose(infile_kernel_type);
	printf("kernel_type = %d\n", kernel_type);
	
	/* Get kernel hyperparameters */
	
	/* Linear kernel hyperparameters */
	double constant = 0.0;
	
	FILE* infile_constant = fopen("inputs/constant.txt", "r");
	system_return = fscanf(infile_constant, "%lf", &constant);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/constant.txt\n");
	}
	fclose(infile_constant);
	printf("constant = %lf\n", constant);
	
	/* Squared exponential kernel hyperparameters */
	double length_scale = 0.0;
	
	FILE* infile_length_scale = fopen("inputs/length_scale.txt", "r");
	system_return = fscanf(infile_length_scale, "%lf", &length_scale);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/length_scale.txt\n");
	}
	fclose(infile_length_scale);
	printf("length_scale = %lf\n", length_scale);
	
	double signal_variance = 0.0;
	
	FILE* infile_signal_variance = fopen("inputs/signal_variance.txt", "r");
	system_return = fscanf(infile_signal_variance, "%lf", &signal_variance);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/signal_variance.txt\n");
	}
	fclose(infile_signal_variance);
	printf("signal_variance = %lf\n", signal_variance);
	
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
	
	/* Now get the X_train values */
	double** X_train;
	
	printf("\nX_train = \n");
	FILE* infile_X_train = fopen("inputs/X_train.txt", "r");
	X_train = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		X_train[i] = malloc(sizeof(double) * num_dimensions);
		for (j = 0; j < num_dimensions; j++)
		{
			system_return = fscanf(infile_X_train, "%lf\t", &X_train[i][j]);
			if (system_return == -1)
			{
				printf("Failed reading file inputs/X_train.txt\n");
			}
			else
			{
				printf("%lf\t", X_train[i][j]);
			}
		} // end of j loop
		printf("\n");
	} // end of i loop
	fclose(infile_X_train);
	
	/* Now get y noisy function values */
	double** y;
	
	printf("\ny = \n");
	FILE* infile_y = fopen("inputs/y.txt", "r");
	y = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		y[i] = malloc(sizeof(double) * 1);
		system_return = fscanf(infile_y, "%lf\n", &y[i][0]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/y.txt\n");
		}
		else
		{
			printf("%lf\n", y[i][0]);
		}
	} // end of i loop
	fclose(infile_y);
	
	/* Now get the X_test values */
	double** X_test;
	
	printf("\nX_test = \n");
	FILE* infile_X_test = fopen("inputs/X_test.txt", "r");
	X_test = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		X_test[i] = malloc(sizeof(double) * num_dimensions);
		for (j = 0; j < num_dimensions; j++)
		{
			system_return = fscanf(infile_X_test, "%lf\t", &X_test[i][j]);
			if (system_return == -1)
			{
				printf("Failed reading file inputs/X_test.txt\n");
			}
			else
			{
				printf("%lf\t", X_test[i][j]);
			}
		} // end of j loop
		printf("\n");
	} // end of i loop
	fclose(infile_X_test);
	
	/*********************************************************************************************************/
	/************************************* BUILD PREDICTIVE DISTRIBUTION *************************************/
	/*********************************************************************************************************/
	
	/* Create array to hold kernel_x_x */
	double** kernel_x_x;
	
	kernel_x_x = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		kernel_x_x[i] = malloc(sizeof(double) * num_training_points);
		for (j = 0; j < num_training_points; j++)
		{
			kernel_x_x[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Calculate kernel K(X, X) */
	CalculateKernel(kernel_type, num_training_points, num_training_points, num_dimensions, X_train, X_train, constant, length_scale, signal_variance, noise_variance, kernel_x_x);
	
	printf("\nkernel_x_x = \n");
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_training_points; j++)
		{
			printf("%e\t", kernel_x_x[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop

	/* Rather than find the actual inverse of K(X, X), Cholesky decompose since it is faster and more numerically stable */
	double** L;
	
	L = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		L[i] = malloc(sizeof(double) * num_training_points);
		for (j = 0; j < num_training_points; j++)
		{
			L[i][j] = kernel_x_x[i][j];
		} // end of j loop
	} // end of i loop
	
	/* Call function to perform Cholesky decomposition on our real symmetric positive definite matrix K(X, X) = L * L**T */
	error = CholeskyDecomposition(num_training_points, L);
	
	/* Now find kernel between our training and test points K(X, X_*) */
	double** kernel_x_x_star;
	
	kernel_x_x_star = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		kernel_x_x_star[i] = malloc(sizeof(double) * num_test_points);
		for (j = 0; j < num_test_points; j++)
		{
			kernel_x_x_star[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Calculate kernel K(X, X_*) */
	CalculateKernel(kernel_type, num_training_points, num_test_points, num_dimensions, X_train, X_test, constant, length_scale, signal_variance, noise_variance, kernel_x_x_star);
	
	printf("\nkernel_x_x_star = \n");
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_test_points; j++)
		{
			printf("%e\t", kernel_x_x_star[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/* Solve L * Z = K(X, X_*) for Z */
	double** Lk;
	
	Lk = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		Lk[i] = malloc(sizeof(double) * num_test_points);
		for (j = 0; j < num_test_points; j++)
		{
			Lk[i][j] = kernel_x_x_star[i][j];
		} // end of j loop
	} // end of i loop
	
	error = SolveLowerCholeskyFactorizationMatrixEquation(num_training_points, num_test_points, L, Lk);
	printf("ERROR: SolveLowerCholeskyFactorizationMatrixEquation, error = %d\n", error);
	
	printf("\nLk = \n");
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_test_points; j++)
		{
			printf("%e\t", Lk[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/* Now solve for L * Z = y for Z */
	double** Ly;
	
	Ly = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		Ly[i] = malloc(sizeof(double) * 1);
		Ly[i][0] = y[i][0];
	} // end of i loop
	
	error = SolveLowerCholeskyFactorizationMatrixEquation(num_training_points, 1, L, Ly);
	printf("ERROR: SolveLowerCholeskyFactorizationMatrixEquation, error = %d\n", error);
	
	printf("\nLy = \n");
	for (i = 0; i < num_training_points; i++)
	{
		printf("%e\n", Ly[i][0]);
	} // end of i loop
	
	/* Calculate mu of gaussian process at our test points */
	double** mu;
	
	mu = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		mu[i] = malloc(sizeof(double) * 1);
		mu[i][0] = 0.0;
	} // end of i loop
	
	MatrixMultiplication(num_test_points, 1, num_training_points, Lk, Ly, 1, 0, mu);
	
	printf("\nmu = \n");
	for (i = 0; i < num_test_points; i++)
	{
		printf("%e\n", mu[i][0]);
	} // end of i loop
	
	/* Now find kernel of the test points */
	double** kernel_x_star_x_star;
	
	kernel_x_star_x_star = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		kernel_x_star_x_star[i] = malloc(sizeof(double) * num_test_points);
		for (j = 0; j < num_test_points; j++)
		{
			kernel_x_star_x_star[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Calculate kernel K(X_*, X_*) */
	CalculateKernel(kernel_type, num_test_points, num_test_points, num_dimensions, X_test, X_test, constant, length_scale, signal_variance, noise_variance, kernel_x_star_x_star);
	
	printf("\nkernel_x_star_x_star = \n");
	for (i = 0; i < num_test_points; i++)
	{
		for (j = 0; j < num_test_points; j++)
		{
			printf("%e\t", kernel_x_star_x_star[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/* Calculate covariance of gaussian process at our test points */
	double** covariance;
	
	covariance = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		covariance[i] = malloc(sizeof(double) * num_test_points);
		for (j = 0; j < num_test_points; j++)
		{
			covariance[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	MatrixMultiplication(num_test_points, num_test_points, num_training_points, Lk, Lk, 1, 0, covariance);
	
	printf("\ncovariance = \n");
	for (i = 0; i < num_test_points; i++)
	{
		for (j = 0; j < num_test_points; j++)
		{
			covariance[i][j] = kernel_x_star_x_star[i][j] - covariance[i][j];
			printf("%e\t", covariance[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/* Calculate log marginal likelihood of gaussian process of training points */
	double log_marginal_likelihood = 0;
	
	/* Find first term, -0.5 * y**T * (K(X, X) + sigma_n^2 * I)^-1 * y */
	log_marginal_likelihood = -0.5 * VectorDotProductRank2(num_training_points, Ly, Ly, 1, 1);
	
	/* Next add second term, -0.5 * log(det(K(X, X) + sigma_n^2 * I)) */
	for (i = 0; i < num_training_points; i++)
	{
		log_marginal_likelihood -= log(L[i][i]);
	} // end of i loop
	
	/* Lastly add third term, the normalizing factor: -0.5 * n * log(2 * Pi) */
	log_marginal_likelihood -= 0.5 * num_training_points * log(2.0 * M_PI);
	
	printf("\nlog_marginal_likelihood = %lf\n", log_marginal_likelihood);
	printf("\nmarginal_likelihood = %lf\n", exp(log_marginal_likelihood));
	
	/*********************************************************************************************************/
	/************************************************* SAMPLE ************************************************/
	/*********************************************************************************************************/
	
	/* Sample from prior */
	double** L_prior;
	
	L_prior = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		L_prior[i] = malloc(sizeof(double) * num_test_points);
		for (j = 0; j < num_test_points; j++)
		{
			L_prior[i][j] = kernel_x_star_x_star[i][j];
		} // end of j loop
	} // end of i loop
	
	/* Call function to perform Cholesky decomposition on our real symmetric positive definite matrix K(X_*, X_*) = L_prior * L_prior**T */
	error = CholeskyDecomposition(num_test_points, L_prior);
	
	double** f_prior;
	
	f_prior = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		f_prior[i] = malloc(sizeof(double) * num_samples);
		for (j = 0; j < num_samples; j++)
		{
			f_prior[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Create array of random normals */
	double** random_normal_samples;
	
	random_normal_samples = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		random_normal_samples[i] = malloc(sizeof(double) * num_samples);
		for (j = 0; j < num_samples; j++)
		{
			random_normal_samples[i][j] = RNorm(0.0, 1.0);
		} // end of j loop
	} // end of i loop
	
	MatrixMultiplication(num_test_points, num_samples, num_test_points, L_prior, random_normal_samples, 0, 0, f_prior);
	
	/* Sample from posterior */
	double** L_posterior;
	
	L_posterior = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		L_posterior[i] = malloc(sizeof(double) * num_test_points);
		for (j = 0; j < num_test_points; j++)
		{
			L_posterior[i][j] = covariance[i][j];
		} // end of j loop
	} // end of i loop
	
	/* Call function to perform Cholesky decomposition on our real symmetric positive definite matrix covariance = L_posterior * L_posterior**T */
	error = CholeskyDecomposition(num_test_points, L_posterior);
	
	double** f_posterior;
	
	f_posterior = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		f_posterior[i] = malloc(sizeof(double) * num_samples);
		for (j = 0; j < num_samples; j++)
		{
			f_posterior[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Resample random normals */
	for (i = 0; i < num_test_points; i++)
	{
		for (j = 0; j < num_samples; j++)
		{
			random_normal_samples[i][j] = RNorm(0.0, 1.0);
		} // end of j loop
	} // end of i loop
	
	MatrixMultiplication(num_test_points, num_samples, num_test_points, L_posterior, random_normal_samples, 0, 0, f_posterior);
	
	/*********************************************************************************************************/
	/********************************************** WRITE OUTPUTS ********************************************/
	/*********************************************************************************************************/
	
	FILE* outfile_mu = fopen("outputs/mu.txt", "w");
	for (i = 0; i < num_test_points; i++)
	{
		fprintf(outfile_mu, "%lf\n", mu[i][0]);
	} // end of i loop
	fclose(outfile_mu);
	
	FILE* outfile_covariance = fopen("outputs/covariance.txt", "w");
	for (i = 0; i < num_test_points; i++)
	{
		for (j = 0; j < num_test_points; j++)
		{
			fprintf(outfile_covariance, "%lf\t", covariance[i][j]);
		} // end of j loop
		fprintf(outfile_covariance, "\n");
	} // end of i loop
	fclose(outfile_covariance);
	
	FILE* outfile_f_prior = fopen("outputs/f_prior.txt", "w");
	for (i = 0; i < num_test_points; i++)
	{
		for (j = 0; j < num_samples; j++)
		{
			fprintf(outfile_f_prior, "%lf\t", f_prior[i][j]);
		} // end of j loop
		fprintf(outfile_f_prior, "\n");
	} // end of i loop
	fclose(outfile_f_prior);
	
	FILE* outfile_f_posterior = fopen("outputs/f_posterior.txt", "w");
	for (i = 0; i < num_test_points; i++)
	{
		for (j = 0; j < num_samples; j++)
		{
			fprintf(outfile_f_posterior, "%lf\t", f_posterior[i][j]);
		} // end of j loop
		fprintf(outfile_f_posterior, "\n");
	} // end of i loop
	fclose(outfile_f_posterior);
	
	/*********************************************************************************************************/
	/*********************************************** FREE MEMORY *********************************************/
	/*********************************************************************************************************/
	
	/* Free dynamically allocated memory */
	for (i = 0; i < num_test_points; i++)
	{
		free(f_posterior[i]);
		free(L_posterior[i]);
		free(random_normal_samples[i]);
		free(f_prior[i]);
		free(L_prior[i]);
		free(covariance[i]);
		free(kernel_x_star_x_star[i]);
		free(mu[i]);
		free(X_test[i]);
	} // end of i loop
	free(f_posterior);
	free(L_posterior);
	free(random_normal_samples);
	free(f_prior);
	free(L_prior);
	free(covariance);
	free(kernel_x_star_x_star);
	free(mu);
	free(X_test);
	
	for (i = 0; i < num_training_points; i++)
	{
		free(Ly[i]);
		free(Lk[i]);
		free(kernel_x_x_star[i]);
		free(L[i]);
		free(kernel_x_x[i]);
		free(y[i]);
		free(X_train[i]);
	} // end of i loop
	free(Ly);
	free(Lk);
	free(kernel_x_x_star);
	free(L);
	free(kernel_x_x);
	free(y);
	free(X_train);
	
	return 0;
} // end of main

/*********************************************************************************************************/
/*********************************************** FUNCTIONS ***********************************************/
/*********************************************************************************************************/

/* This function calculates various similarity kernels between two matrices */
void CalculateKernel(int kernel_type, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double constant, double length_scale, double signal_variance, double noise_variance, double** kernel)
{
	if (kernel_type == 0) // linear
	{
		LinearKernel(a_rows, b_rows, a_cols, A, B, constant, kernel);
	}
	else // squared exponential
	{
		SquaredExponentialKernel(a_rows, b_rows, a_cols, A, B, length_scale, signal_variance, noise_variance, kernel);
	}
	
	return;
} // end of CalculateKernel function

/* This function applies the linear kernel between two matrices */
void LinearKernel(unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double constant, double** kernel)
{
	unsigned int i, j;
	
	MatrixMultiplication(a_rows, b_rows, a_cols, A, B, 0, 1, kernel);
	
	/* Add constant */
	for (i = 0; i < a_rows; i++)
	{
		for (j = 0; j < b_rows; j++)
		{
			kernel[i][j] += constant;
		} // end of j loop
	} // end of i loop
	
	return;
} // end of LinearKernel function

/* This function applies the squared exponential kernel between two matrices */
void SquaredExponentialKernel(unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double length_scale, double signal_variance, double noise_variance, double** kernel)
{
	unsigned int i, j, k;
	double a_squared_sum, b_squared_sum;
	
	MatrixMultiplication(a_rows, b_rows, a_cols, A, B, 0, 1, kernel);
	
	for (i = 0; i < a_rows; i++)
	{
		for (j = 0; j < b_rows; j++)
		{
			a_squared_sum = 0.0;
			b_squared_sum = 0.0;
			for (k = 0; k < a_cols; k++)
			{
				a_squared_sum += A[i][k] * A[i][k];
				b_squared_sum += B[j][k] * B[j][k];
			} // end of k loop
			
			kernel[i][j] = a_squared_sum + b_squared_sum - 2.0 * kernel[i][j];
			
			/* Take exponential with length-scale scaling it */
			kernel[i][j] = exp(-0.5 / (length_scale * length_scale) * kernel[i][j]);
			
			/* Scale by the signal variance and shift by the noise variance */
			kernel[i][j] *= signal_variance;
			
			/* Shift by the noise variance */
			if (i == j)
			{
				kernel[i][j] += noise_variance;
			}
		} // end of j loop
	} // end of i loop
	
	return;
} // end of SquaredExponentialKernel function

/* This function performs the Cholesky decomposition A = L * L**T */
int CholeskyDecomposition(int n, double** L)
{
	int i, j, error = 0;
	
	error = RecursiveCholeskyFactorization(n, L, 0, 0);
	
	/* Print L or error code message */
	if (error == 0)
	{
		/* Zero upper triangular matrix without diagonal since it wasn't updated */
		for (i = 0; i < n - 1; i++)
		{
			for (j = i + 1; j < n; j++)
			{
				L[i][j] = 0.0;
			} // end of j loop
		} // end of i loop
		
		printf("\nL = \n");
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				printf("%e\t", L[i][j]);
			} // end of j loop
			printf("\n");
		} // end of i loop
	}
	else if (error == 1)
	{
		printf("ERROR: RecursiveCholeskyFactorization somehow managed to start with n == 0\n");
	}
	else if (error == 2)
	{
		printf("ERROR: SolveTriangularMatrixEquation somehow managed to start with m or n == 0\n");
	}
	else if (error == 3)
	{
		printf("ERROR: SymmetricRankKOperation somehow managed to start with n or k == 0\n");
	}
	else
	{
		printf("ERROR: Matrix is not positive-definite!\n");
	}
	
	return error;
} // end of CholeskyFactorization function

/*
This function computes the Cholesky factorization of a real symmetric
positive definite matrix A using the recursive algorithm.

The factorization has the form
   A = L  * L**T, 
where L is lower triangular.

This is the recursive version of the algorithm. It divides
the matrix into four submatrices:

       [  A11 | A12  ]  where A11 is n1 by n1 and A22 is n2 by n2
   A = [ -----|----- ]  with n1 = n / 2 and n2 = n - n1
       [  A21 | A22  ]  

The function calls itself to factor A11. Update and scale A21, 
update A22, then calls itself to factor A22.
Modified from dpotrf2.
*/
int RecursiveCholeskyFactorization(int n, double** a, int a_row_offset, int a_col_offset)
{
	/*
	Args:
		n: The order of the matrix a. n >= 0.
		a: Matrix that we are modifying in place.
		a_row_offset: Number of rows we need to offset a.
		a_col_offset: Number of cols we need to offset a.
		
	Returns:
		Integer error code.
	*/
	
	int i, j, error = 0;
	
	/* Quick return if possible */
	if (n <= 0)
	{
		return 1;
	}

	if (n == 1) // n == 1 case
	{
		/* Test for non-positive-definiteness */
		if (a[a_row_offset][a_col_offset] <= 0.0 || isnan(a[a_row_offset][a_col_offset]))
		{
			return 99;
		}

		/* Factor */
		a[a_row_offset][a_col_offset] = sqrt(a[a_row_offset][a_col_offset]);
	}
	else // use recursive code
	{
		int n1 = n / 2;
		int n2 = n - n1;

		/* Factor A11 */
		error = RecursiveCholeskyFactorization(n1, a, a_row_offset, a_col_offset);
		if (error != 0)
		{
			return error;
		}

		/* Update and scale A21 */
		error = SolveTriangularMatrixEquation(0, 1, n2, n1, a, a, a_row_offset, a_col_offset, n1 + a_row_offset, a_col_offset);
		if (error != 0)
		{
			return error;
		}

		/* Update A22 */
		error = SymmetricRankKOperation(n2, n1, a, n1 + a_row_offset, a_col_offset, n1 + a_row_offset, n1 + a_col_offset);
		if (error != 0)
		{
			return error;
		}
		
		/* Factor A22 */
		error = RecursiveCholeskyFactorization(n2, a, n1 + a_row_offset, n1 + a_col_offset);
	}
	
	return error;
} // end of RecursiveCholeskyFactorization function

/*
This function solves the matrix equation X * A**T = B,
where X and B are m by n matrices, A is a non-unit, 
lower triangular matrix.
Modified from dtrsm.
*/
int SolveTriangularMatrixEquation(int left_side, int transa, int m, int n, double** a, double** b, int read_row_offset, int read_col_offset, int write_row_offset, int write_col_offset)
{
	/*
	Args:
		m: Number of rows of current submatrix of a.
		n: Number of cols of current submatrix of a.
		a: Matrix that we are modifying in place.
		read_row_offset: Number of rows we need to offset for reading from a.
		read_col_offset: Number of cols we need to offset for reading from a.
		write_row_offset: Number of rows we need to offset for writing to a.
		write_col_offset: Number of cols we need to offset for writing to a.
		
	Returns:
		Integer error code.
	*/
	
	int i, j, k;
	double temp;
	
	/* Quick return if possible. */
	if (m <= 0 || n <= 0)
	{
		return 2;
	}
	
	if (left_side == 1)
	{
		if (transa == 0)
		{
			/* Form  B := inv(A) * B. */
			for (j = 0; j < n; j++)
			{
				for (k = 0; k < m; k++)
				{
					if (b[k + write_row_offset][j + write_col_offset] != 0.0)
					{
						for (i = k + 1; i < m; i++)
						{
							b[i + write_row_offset][j + write_col_offset] -= b[k + write_row_offset][j + write_col_offset] * a[i + read_row_offset][k + read_col_offset];
						} // end of i loop
					}
				} // end of k loop
			} // end of j loop
		}
		else // transa == 1
		{
			/* Form  B := inv(A**T) * B. */
			for (j = 0; j < n; j++)
			{
				for (i = m - 1; i >= 0; i--)
				{
					temp = b[i + write_row_offset][j + write_col_offset];
					for (k = i + 1; k < m; k++)
					{
						temp -= a[k + read_row_offset][i + read_col_offset] * b[k + write_row_offset][j + write_col_offset];
					} // end of k loop
					temp /= a[i + read_row_offset][i + read_col_offset];
					b[i + write_row_offset][j + write_col_offset] = temp;
				} // end of i loop
			} // end of j loop
		}
	}
	else // left_side == 0
	{
		/* Form  B := B * inv( A**T ). */
		for (k = 0; k < n; k++)
		{
			temp = 1.0 / a[k + read_row_offset][k + read_col_offset];
			for (i = 0; i < m; i++)
			{
				b[i + write_row_offset][k + write_col_offset] *= temp;
			} // end of i loop

			for (j = k + 1; j < n; j++)
			{
				if (a[j + read_row_offset][k + read_col_offset] != 0.0)
				{
					temp = a[j + read_row_offset][k + read_col_offset];
					for (i = 0; i < m; i++)
					{
						b[i + write_row_offset][j + write_col_offset] -= temp * b[i + write_row_offset][k + write_col_offset];
					} // end of i loop
				}
			} // end of j loop
		} // end of k loop
	}
	
	return 0;
} // end of SolveTriangularMatrixEquation function

/*
This function performs the symmetric rank k operation C := -A * A**T + C,
where C is an n by n symmetric matrix and A is an n by k matrix.
Modified from dsyrk.
*/
int SymmetricRankKOperation(int n, int k, double** a, int read_row_offset, int read_col_offset, int write_row_offset, int write_col_offset)
{
	/*
	Args:
		n: Number of rows of current submatrix of a.
		k: Number of cols of current submatrix of a.
		a: Matrix that we are modifying in place.
		read_row_offset: Number of rows we need to offset for reading from a.
		read_col_offset: Number of cols we need to offset for reading from a.
		write_row_offset: Number of rows we need to offset for writing to a.
		write_col_offset: Number of cols we need to offset for writing to a.
		
	Returns:
		Integer error code.
	*/
	
	int i, j, l;
	double temp;
	
	/* Quick return if possible. */
	if (n <= 0 || k <= 0)
	{
		return 3;
	}
	
	/* Form  C := -A*A**T + C. */
	for (j = 0; j < n; j++)
	{
		for (l = 0; l < k; l++)
		{
			if (a[j + read_row_offset][l + read_col_offset] != 0.0)
			{
				temp = -a[j + read_row_offset][l + read_col_offset];
				for (i = j - 1; i < n; i++)
				{
					a[i + write_row_offset][j + write_col_offset] += temp * a[i + read_row_offset][l + read_col_offset];
				} // end of i loop
			}
		} // end of l loop
	} // end of j loop

	return 0;
} // end of SymmetricRankKOperation function

/*
This solves a system of linear equations A * X = B with a symmetric
positive definite matrix A using the Cholesky factorization
A = L*L**T computed by DPOTRF.
Modified from dtrsm.
*/
int SolveLowerCholeskyFactorizationMatrixEquation(int n, int nrhs, double** a, double** b)
{
	/*
	Args:
		n: The order of the matrix a. n >= 0.
		nrhs: Number of columns of the matrix B.  NRHS >= 0.
		a: The triangular factor L of size (n, n) from the 
			Cholesky factorization A = L*L**T, as computed by DPOTRF.
		b: On entry, the right hand side matrix B of size (n, nrhs). 
			On exit, the solution matrix X.
		
	Returns:
		Integer error code.
	*/
	
	int error = 0;
	
	/* Quick return if possible. */
	if (n <= 0 || nrhs <= 0)
	{
		return 4;
	}
	
	/* Solve A * X = B where A = L * L**T. */
	
	/* Solve L * X = B, overwriting B with X. */
	error = SolveTriangularMatrixEquation(1, 0, n, nrhs, a, b, 0, 0, 0, 0);
	if (error != 0)
	{
		return error;
	}
	
	/* Solve L**T * X = B, overwriting B with X. */
	error = SolveTriangularMatrixEquation(1, 1, n, nrhs, a, b, 0, 0, 0, 0);
	if (error != 0)
	{
		return error;
	}
	
	return 0;
} // end of SolveLowerCholeskyFactorizationMatrixEquation function

/* This function performs matrix multiplication between two given matrices */
void MatrixMultiplication(unsigned int m, unsigned int n, unsigned int p, double** A, double** B, int transpose_A, int transpose_B, double** C)
{
	/* C = [m, n] */
	/* A = [m, p] */
	/* B = [p, n] */

	unsigned int i, j, k;

	if (transpose_B == 0) // if B is NOT transposed
	{
		if (transpose_A == 0) // if A is NOT transposed
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < n; j++)
				{
					C[i][j] = 0.0;
					for (k = 0; k < p; k++)
					{
						C[i][j] += A[i][k] * B[k][j];
					} // end of k loop
				} // end of j loop
			} // end of i loop
		} // end of if a is NOT transposed
		else // if a is transposed
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < n; j++)
				{
					C[i][j] = 0.0;
					for (k = 0; k < p; k++)
					{
						C[i][j] += A[k][i] * B[k][j];
					} // end of k loop
				} // end of j loop
			} // end of i loop
		} // end of if a is  transposed
	} // end of if B is NOT transposed
	else // if B is transposed
	{
		if (transpose_A == 0)
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < n; j++)
				{
					C[i][j] = 0.0;
					for (k = 0; k < p; k++)
					{
						C[i][j] += A[i][k] * B[j][k];
					} // end of k loop
				} // end of j loop
			} // end of i loop
		} // end of if a is NOT transposed
		else
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < n; j++)
				{
					C[i][j] = 0.0;
					for (k = 0; k < p; k++)
					{
						C[i][j] += A[k][i] * B[j][k];
					} // end of k loop
				} // end of j loop
			} // end of i loop
		} // end of if a is transposed
	} // end of if b is transposed
	
	return;
} // end of MatrixMultiplication function

/* This function performs the dot product between two given vectors that are in 2D form */
double VectorDotProductRank2(unsigned int n, double** A, double** B, int a_col_vec, int b_col_vec)
{
	/* A = [m, p] */
	/* B = [p, n] */

	unsigned int i;
	double dot_product = 0.0;
	
	if (a_col_vec == 1)
	{
		if (b_col_vec == 1)
		{
			for (i = 0; i < n; i++)
			{
				dot_product += A[i][0] * B[i][0];
			} // end of i loop
		}
		else
		{
			for (i = 0; i < n; i++)
			{
				dot_product += A[i][0] * B[0][i];
			} // end of i loop
		}
	}
	else
	{
		if (b_col_vec == 1)
		{
			for (i = 0; i < n; i++)
			{
				dot_product += A[0][i] * B[i][0];
			} // end of i loop
		}
		else
		{
			for (i = 0; i < n; i++)
			{
				dot_product += A[0][i] * B[0][i];
			} // end of i loop
		}
	}

	return dot_product;
} // end of VectorDotProductRank2 function

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function

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