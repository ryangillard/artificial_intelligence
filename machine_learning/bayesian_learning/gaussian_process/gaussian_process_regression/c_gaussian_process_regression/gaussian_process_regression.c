#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "l_bfgs_b.h"

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function reads the model hyperparameters */
void ReadModelHyperparameters(unsigned int* num_training_points, unsigned int* num_test_points, unsigned int* num_dimensions, unsigned int* num_samples, int* kernel_type);

/* This function reads the intial kernel hyperparameters */
void ReadInitiaK_inv_k_starernelHyperparameters(int kernel_type, unsigned int* num_kernel_hyperparameters, double** kernel_hyperparameters);

/* This function reads the training features and targets */
void ReadTrainingData(unsigned int num_training_points, unsigned int num_dimensions, double*** X_train, double*** y);

/* This function reads in the test features */
void ReadTestData(unsigned int num_test_points, unsigned int num_dimensions, double*** X_test);

/* This function calculates various similarity kernels between two matrices */
void CalculateKernel(int kernel_type, double* kernel_hyperparameters, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double** kernel);

/* This function applies the linear kernel between two matrices */
void LinearKernel(double* kernel_hyperparameters, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double** kernel);

/* This function applies the squared exponential kernel between two matrices */
void SquaredExponentiaK_inv_k_starernel(double* kernel_hyperparameters, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double** kernel);

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

/* This function computes the inverse of a real symmetric positive definite matrix A using the Cholesky factorization A = L * L**T. */
int LowerCholeskyInverse(unsigned int n, double** a);

/* This function computes the inverse of a real lower triangular matrix A */
int LowerTriangularInverseSingularityCheck(unsigned int n, double** a);

/* This function computes the inverse of a real lower triangular matrix */
int LowerTriangularInverse(unsigned int n, double** a);

/* This function performs the matrix-vector operation x := A * x, where A is an n by n non-unit, lower triangular matrix. */
int MatrixVectorMultiplication(unsigned int n, double** a, unsigned int col_offset);

/* This function scales a vector by a constant. */
void ScaleVectorByConstant(unsigned int n, double da, double** a, unsigned int row_offset, unsigned int col_offset);

/* This function efficiently recombines a lower cholesky decomposition inverse A^-1 = L^-1 * L^-T */
void RecombineLowerCholeskyDecompositionInverse(unsigned int n, double** L, double** A);

/* This function optimizes kernel hyperparameters */
void OptimizeKernelHyperparameters(int kernel_type, unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** y, double** kernel_x_x, double** L, double** K_inv_y, unsigned int num_kernel_hyperparameters, double* kernel_hyperparameters);

/* This function reads the kernel hyperparameter optimization parameters */
void ReadKernelHyperparameterOptimizationParameters(unsigned int num_kernel_hyperparameters, int** kernel_hyperparameter_bounds_type, double*** kernel_hyperparameter_bounds_values);

/* This function performs the kernel hyperparameter optimzation loop */
void KernelHyperparameterOptimizerLoop(int kernel_type, unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** y, double** kernel_x_x, double** L, double** K_inv_y, double** alpha_alpha_t, double** kernel_x_x_inv, double** d_kernel_d_kernel_hyperparameter, double** d_kernel_d_kernel_hyperparameter_temp, unsigned int num_kernel_hyperparameters, double* d_negative_log_marginal_likelihood_d_kernel_hyperparameter, int* kernel_hyperparameter_bounds_type, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters);

/* This function calculates the gradient of the negative log marginal likelihood with respect to the linear kernel's hyperparameters */
void CalculateNegativeLogMarginalLikelihoodLinearKernelHyperparameterGradients(unsigned int num_training_points, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double* kernel_hyperparameters, double* d_negative_log_marginal_likelihood_d_kernel_hyperparameter);

/* This function calculates the gradient of the linear kernel with respect to the constant hyperparameter */
void CalculateLinearKernelConstantHyperparameterGradient(unsigned int num_training_points, double** d_kernel_d_kernel_hyperparameter);

/* This function calculates the gradient of the negative log marginal likelihood with respect to the squared exponential kernel's hyperparameters */
void CalculateNegativeLogMarginalLikelihoodSquaredExponentialKernelHyperparameterGradients(unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** kernel_x_x, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double** d_kernel_d_kernel_hyperparameter_temp, double* kernel_hyperparameters, double* d_negative_log_marginal_likelihood_d_kernel_hyperparameter);

/* This function calculates the gradient of the squared exponential kernel with respect to the length-scale hyperparameter */
void CalculateSquaredExponentialKernelLengthScaleHyperparameterGradient(unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** kernel_x_x, double* kernel_hyperparameters, double** d_kernel_d_kernel_hyperparameter_temp, double** d_kernel_d_kernel_hyperparameter);

/* This function calculates the gradient of the squared exponential kernel with respect to the signal variance hyperparameter */
void CalculateSquaredExponentialKernelSignalVarianceHyperparameterGradient(unsigned int num_training_points, double** kernel_x_x, double* kernel_hyperparameters, double** d_kernel_d_kernel_hyperparameter);

/* This function calculates the gradient of the squared exponential kernel with respect to the noise variance hyperparameter */
void CalculateSquaredExponentialKernelNoiseVarianceHyperparameterGradient(unsigned int num_training_points, double* kernel_hyperparameters, double** d_kernel_d_kernel_hyperparameter);

/* This function calculates the negative log marginal likelihood gradient with respect to a kernel's hyperparameter */
void CalculateNegativeLogMarginalLikelihoodGradient(unsigned int num_training_points, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, unsigned int hyperparameter_index, double* kernel_hyperparameters, double* d_negative_log_marginal_likelihood_d_kernel_hyperparameter);

/* This function performs matrix multiplication between two given matrices */
void MatrixMatrixMultiplication(unsigned int m, unsigned int n, unsigned int p, double** A, double** B, int transpose_A, int transpose_B, double** C);

/* This function calculates the trace of the matrix-matrix multiplication of matrices A and B */
int MatrixMatrixMultiplcationTrace(unsigned int m, unsigned int n, unsigned int p, double** A, double** B, int transpose_A, int transpose_B, double* trace);

/* This function performs the dot product between two given vectors that are in 2D form */
double VectorDotProductRank2(unsigned int n, double** A, double** B, int a_col_vec, int b_col_vec);

/* This function calculate log marginal likelihood of gaussian process of training points */
double CalculateLogMarginalLikelihood(unsigned int num_training_points, double** L, double** K_inv_y);

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
	int i, j, error = 0;
	srand(12345);
	
	/*********************************************************************************************************/
	/********************************************* READ INPUTS ***********************************************/
	/*********************************************************************************************************/
	
	/* Get model hyperparameters */
	unsigned int num_training_points = 0;
	unsigned int num_test_points = 0;
	unsigned int num_dimensions = 0;
	unsigned int num_samples = 0;
	int kernel_type = 0;
	
	ReadModelHyperparameters(&num_training_points, &num_test_points, &num_dimensions, &num_samples, &kernel_type);
	
	/* Get kernel hyperparameters */
	unsigned int num_kernel_hyperparameters = 0;
	double* kernel_hyperparameters;
	
	ReadInitiaK_inv_k_starernelHyperparameters(kernel_type, &num_kernel_hyperparameters, &kernel_hyperparameters);
	
	/* Get training data */
	double** X_train;
	double** y;
	
	ReadTrainingData(num_training_points, num_dimensions, &X_train, &y);
	
	/* Get test data */
	double** X_test;
	
	ReadTestData(num_test_points, num_dimensions, &X_test);

	/*********************************************************************************************************/
	/************************************* BUILD PREDICTIVE DISTRIBUTION *************************************/
	/*********************************************************************************************************/
	
	/* Create array to hold kernel_x_x */
	double** kernel_x_x;
	kernel_x_x = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		kernel_x_x[i] = malloc(sizeof(double) * num_training_points);
	} // end of i loop
	
	/* Calculate kernel K(X, X) */
	CalculateKernel(kernel_type, kernel_hyperparameters, num_training_points, num_training_points, num_dimensions, X_train, X_train, kernel_x_x);
	if (kernel_type == 1) // squared exponential
	{
		for (i = 0; i < num_training_points; i++)
		{
			for (j = 0; j < num_training_points; j++)
			{
				/* Shift by the noise variance */
				if (i == j)
				{
					kernel_x_x[i][j] += kernel_hyperparameters[2] * kernel_hyperparameters[2];
				}
			} // end of j loop
		} // end of i loop
	}
	
	printf("\nkernel_x_x = \n");
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_training_points; j++)
		{
			printf("%e\t", kernel_x_x[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop

	/* Rather than find the actual inverse of K(X, X), Cholesky decompose K(X, X) int L * L**T since it is faster and more numericaK_inv_k_star_inv_y stable */
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
	
	printf("\nL = \n");
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_training_points; j++)
		{
			printf("%e\t", L[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/* Now solve for K(X, X) * Z = y for Z */
	double** K_inv_y;
	K_inv_y = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		K_inv_y[i] = malloc(sizeof(double) * 1);
		K_inv_y[i][0] = y[i][0];
	} // end of i loop
	
	error = SolveLowerCholeskyFactorizationMatrixEquation(num_training_points, 1, L, K_inv_y);
	if (error != 0)
	{
		printf("ERROR: SolveLowerCholeskyFactorizationMatrixEquation, error = %d\n", error);
	}
	
	printf("\nK_inv_y = \n");
	for (i = 0; i < num_training_points; i++)
	{
		printf("%e\n", K_inv_y[i][0]);
	} // end of i loop
	
	/* Now find kernel between our training and test points K(X, X_*) */
	double** kernel_x_x_star;
	kernel_x_x_star = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		kernel_x_x_star[i] = malloc(sizeof(double) * num_test_points);
	} // end of i loop
	
	/* Calculate kernel K(X, X_*) */
	CalculateKernel(kernel_type, kernel_hyperparameters, num_training_points, num_test_points, num_dimensions, X_train, X_test, kernel_x_x_star);
	
	printf("\nkernel_x_x_star = \n");
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_test_points; j++)
		{
			printf("%e\t", kernel_x_x_star[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/* Calculate mu of gaussian process at our test points */
	double** mu;
	
	mu = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		mu[i] = malloc(sizeof(double) * 1);
		mu[i][0] = 0.0;
	} // end of i loop
	
	MatrixMatrixMultiplication(num_test_points, 1, num_training_points, kernel_x_x_star, K_inv_y, 1, 0, mu);
	
	printf("\nmu = \n");
	for (i = 0; i < num_test_points; i++)
	{
		printf("%e\n", mu[i][0]);
	} // end of i loop
	
	/* Solve K(X, X) * Z = K(X, X_*) for Z */
	double** K_inv_k_star;
	K_inv_k_star = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		K_inv_k_star[i] = malloc(sizeof(double) * num_test_points);
		for (j = 0; j < num_test_points; j++)
		{
			K_inv_k_star[i][j] = kernel_x_x_star[i][j];
		} // end of j loop
	} // end of i loop
	
	error = SolveLowerCholeskyFactorizationMatrixEquation(num_training_points, num_test_points, L, K_inv_k_star);
	if (error != 0)
	{
		printf("ERROR: SolveLowerCholeskyFactorizationMatrixEquation, error = %d\n", error);
	}
	
	printf("\nK_inv_k_star = \n");
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_test_points; j++)
		{
			printf("%e\t", K_inv_k_star[i][j]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	
	/* Now find kernel of the test points */
	double** kernel_x_star_x_star;
	
	kernel_x_star_x_star = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		kernel_x_star_x_star[i] = malloc(sizeof(double) * num_test_points);
	} // end of i loop
	
	/* Calculate kernel K(X_*, X_*) */
	CalculateKernel(kernel_type, kernel_hyperparameters, num_test_points, num_test_points, num_dimensions, X_test, X_test, kernel_x_star_x_star);
	
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
	
	MatrixMatrixMultiplication(num_test_points, num_test_points, num_training_points, kernel_x_x_star, K_inv_k_star, 1, 0, covariance);
	
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
	double log_marginal_likelihood = 0.0;
	log_marginal_likelihood = CalculateLogMarginalLikelihood(num_training_points, L, K_inv_y);	
	
	printf("\nlog_marginal_likelihood = %.16f\n", log_marginal_likelihood);
	printf("\nmarginal_likelihood = %.16f\n", exp(log_marginal_likelihood));
	
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
	
	MatrixMatrixMultiplication(num_test_points, num_samples, num_test_points, L_prior, random_normal_samples, 0, 0, f_prior);
	
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
	
	MatrixMatrixMultiplication(num_test_points, num_samples, num_test_points, L_posterior, random_normal_samples, 0, 0, f_posterior);

	/*********************************************************************************************************/
	/************************************* OPTIMIZE KERNEL HYPERPARAMETERS ***********************************/
	/*********************************************************************************************************/
	
	OptimizeKernelHyperparameters(kernel_type, num_training_points, num_dimensions, X_train, y, kernel_x_x, L, K_inv_y, num_kernel_hyperparameters, kernel_hyperparameters);
	
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
	
	/* Free dynamicaK_inv_k_star_inv_y allocated memory */
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
		free(K_inv_k_star[i]);
		free(K_inv_y[i]);
		free(kernel_x_x_star[i]);
		free(L[i]);
		free(kernel_x_x[i]);
		free(y[i]);
		free(X_train[i]);
	} // end of i loop
	free(K_inv_k_star);
	free(K_inv_y);
	free(kernel_x_x_star);
	free(L);
	free(kernel_x_x);
	free(y);
	free(X_train);
	
	free(kernel_hyperparameters);
	
	return 0;
} // end of main

/*********************************************************************************************************/
/*********************************************** FUNCTIONS ***********************************************/
/*********************************************************************************************************/

/* This function reads the model hyperparameters */
void ReadModelHyperparameters(unsigned int* num_training_points, unsigned int* num_test_points, unsigned int* num_dimensions, unsigned int* num_samples, int* kernel_type)
{
	int system_return = 0;
	
	/* Get the number of training points */
	FILE* infile_num_training_points = fopen("inputs/num_training_points.txt", "r");
	system_return = fscanf(infile_num_training_points, "%u", num_training_points);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/num_training_points.txt\n");
	}
	fclose(infile_num_training_points);
	printf("num_training_points = %u\n", (*num_training_points));
	
	/* Get the number of test points */
	FILE* infile_num_test_points = fopen("inputs/num_test_points.txt", "r");
	system_return = fscanf(infile_num_test_points, "%u", num_test_points);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/num_test_points.txt\n");
	}
	fclose(infile_num_test_points);
	printf("num_test_points = %u\n", (*num_test_points));
	
	/* Get the number of dimensions */
	FILE* infile_num_dimensions = fopen("inputs/num_dimensions.txt", "r");
	system_return = fscanf(infile_num_dimensions, "%u", num_dimensions);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/num_dimensions.txt\n");
	}
	fclose(infile_num_dimensions);
	printf("num_dimensions = %u\n", (*num_dimensions));
	
	/* Get the number of samples */
	FILE* infile_num_samples = fopen("inputs/num_samples.txt", "r");
	system_return = fscanf(infile_num_samples, "%u", num_samples);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/num_samples.txt\n");
	}
	fclose(infile_num_samples);
	printf("num_samples = %u\n", (*num_samples));
	
	/* Get the kernel type */
	FILE* infile_kernel_type = fopen("inputs/kernel_type.txt", "r");
	system_return = fscanf(infile_kernel_type, "%d", kernel_type);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/kernel_type.txt\n");
	}
	fclose(infile_kernel_type);
	printf("kernel_type = %d\n", (*kernel_type));
	
	return;
} // end of ReadModelHyperparameters function

/* This function reads the intial kernel hyperparameters */
void ReadInitiaK_inv_k_starernelHyperparameters(int kernel_type, unsigned int* num_kernel_hyperparameters, double** kernel_hyperparameters)
{
	int system_return = 0;
	
	if (kernel_type == 0) // linear
	{
		/* Linear kernel hyperparameters */
		(*num_kernel_hyperparameters) = 1;
		
		(*kernel_hyperparameters) = malloc(sizeof(double) * (*num_kernel_hyperparameters));
		
		/* Get constant */
		FILE* infile_constant = fopen("inputs/constant.txt", "r");
		system_return = fscanf(infile_constant, "%lf", &(*kernel_hyperparameters)[0]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/constant.txt\n");
		}
		fclose(infile_constant);
		printf("constant = %lf\n", (*kernel_hyperparameters)[0]);
	}
	else // squared exponential
	{
		/* Squared exponential kernel hyperparameters */
		(*num_kernel_hyperparameters) = 3;
		
		(*kernel_hyperparameters) = malloc(sizeof(double) * (*num_kernel_hyperparameters));

		/* Get length-scale */
		FILE* infile_length_scale = fopen("inputs/length_scale.txt", "r");
		system_return = fscanf(infile_length_scale, "%lf", &(*kernel_hyperparameters)[0]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/length_scale.txt\n");
		}
		fclose(infile_length_scale);
		printf("length_scale = %lf\n", (*kernel_hyperparameters)[0]);

		/* Get signal variance */
		FILE* infile_signal_variance = fopen("inputs/signal_variance.txt", "r");
		system_return = fscanf(infile_signal_variance, "%lf", &(*kernel_hyperparameters)[1]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/signal_variance.txt\n");
		}
		fclose(infile_signal_variance);
		printf("signal_variance = %lf\n", (*kernel_hyperparameters)[1]);

		/* Get noise variance */
		FILE* infile_noise_variance = fopen("inputs/noise_variance.txt", "r");
		system_return = fscanf(infile_noise_variance, "%lf", &(*kernel_hyperparameters)[2]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/noise_variance.txt\n");
		}
		fclose(infile_noise_variance);
		printf("noise_variance = %lf\n", (*kernel_hyperparameters)[2]);	
	}
	
	return;
} // end of ReadInitiaK_inv_k_starernelHyperparameters function

/* This function reads the training features and targets */
void ReadTrainingData(unsigned int num_training_points, unsigned int num_dimensions, double*** X_train, double*** y)
{
	unsigned int i, j, system_return = 0;
	
	/* Get the X_train values */
	printf("\nX_train = \n");
	FILE* infile_X_train = fopen("inputs/X_train.txt", "r");
	(*X_train) = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		(*X_train)[i] = malloc(sizeof(double) * num_dimensions);
		for (j = 0; j < num_dimensions; j++)
		{
			system_return = fscanf(infile_X_train, "%lf\t", &(*X_train)[i][j]);
			if (system_return == -1)
			{
				printf("Failed reading file inputs/X_train.txt\n");
			}
			else
			{
				printf("%lf\t", (*X_train)[i][j]);
			}
		} // end of j loop
		printf("\n");
	} // end of i loop
	fclose(infile_X_train);
	
	/* Now get y noisy function values */
	printf("\ny = \n");
	FILE* infile_y = fopen("inputs/y.txt", "r");
	(*y) = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		(*y)[i] = malloc(sizeof(double) * 1);
		system_return = fscanf(infile_y, "%lf\n", &(*y)[i][0]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/y.txt\n");
		}
		else
		{
			printf("%lf\n", (*y)[i][0]);
		}
	} // end of i loop
	fclose(infile_y);
	
	return;
} // end of ReadTrainingData function

/* This function reads in the test features */
void ReadTestData(unsigned int num_test_points, unsigned int num_dimensions, double*** X_test)
{
	unsigned int i, j, system_return = 0;
	
	/* Get the X_test values */
	printf("\nX_test = \n");
	FILE* infile_X_test = fopen("inputs/X_test.txt", "r");
	(*X_test) = malloc(sizeof(double*) * num_test_points);
	for (i = 0; i < num_test_points; i++)
	{
		(*X_test)[i] = malloc(sizeof(double) * num_dimensions);
		for (j = 0; j < num_dimensions; j++)
		{
			system_return = fscanf(infile_X_test, "%lf\t", &(*X_test)[i][j]);
			if (system_return == -1)
			{
				printf("Failed reading file inputs/X_test.txt\n");
			}
			else
			{
				printf("%lf\t", (*X_test)[i][j]);
			}
		} // end of j loop
		printf("\n");
	} // end of i loop
	fclose(infile_X_test);
	
	return;
} // end of ReadTestData function

/* This function calculates various similarity kernels between two matrices */
void CalculateKernel(int kernel_type, double* kernel_hyperparameters, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double** kernel)
{
	unsigned int i, j;
	
	/* Reset kernel */
	for (i = 0; i < a_rows; i++)
	{
		for (j = 0; j < b_rows; j++)
		{
			kernel[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Build kernel based on type with corresponding kernel hyperparameters */
	if (kernel_type == 0) // linear
	{
		LinearKernel(kernel_hyperparameters, a_rows, b_rows, a_cols, A, B, kernel);
	}
	else // squared exponential
	{
		SquaredExponentiaK_inv_k_starernel(kernel_hyperparameters, a_rows, b_rows, a_cols, A, B, kernel);
	}
	
	return;
} // end of CalculateKernel function

/* This function applies the linear kernel between two matrices */
void LinearKernel(double* kernel_hyperparameters, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double** kernel)
{
	unsigned int i, j;
	
	MatrixMatrixMultiplication(a_rows, b_rows, a_cols, A, B, 0, 1, kernel);
	
	/* Add constant */
	for (i = 0; i < a_rows; i++)
	{
		for (j = 0; j < b_rows; j++)
		{
			kernel[i][j] += kernel_hyperparameters[0];
		} // end of j loop
	} // end of i loop
	
	return;
} // end of LinearKernel function

/* This function applies the squared exponential kernel between two matrices */
void SquaredExponentiaK_inv_k_starernel(double* kernel_hyperparameters, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double** kernel)
{
	unsigned int i, j, k;
	double a_squared_sum, b_squared_sum;
	
	MatrixMatrixMultiplication(a_rows, b_rows, a_cols, A, B, 0, 1, kernel);
	
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
			kernel[i][j] = exp(-0.5 / (kernel_hyperparameters[0] * kernel_hyperparameters[0]) * kernel[i][j]);
			
			/* Scale by the signal variance and shift by the noise variance */
			kernel[i][j] *= kernel_hyperparameters[1] * kernel_hyperparameters[1];
		} // end of j loop
	} // end of i loop
	
	return;
} // end of SquaredExponentiaK_inv_k_starernel function

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
		printf("ERROR: L matrix is not positive-definite!\n");
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
						b[k + write_row_offset][j + write_col_offset] /= a[k + read_row_offset][k + read_col_offset];
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

/* This function computes the inverse of a real symmetric positive definite
matrix A using the Cholesky factorization A = L * L**T computed by DPOTRF.
Modified from dpotri.
*/
int LowerCholeskyInverse(unsigned int n, double** a)
{
	/* Quick return if possible */
	if (n == 0)
	{
		return -1;
	}
	
	/* Invert the triangular Cholesky factor L */
	return LowerTriangularInverseSingularityCheck(n, a);
} // end of LowerCholeskyInverse function

/* This function computes the inverse of a real lower triangular matrix A.
Modified from dtrtri.
*/
int LowerTriangularInverseSingularityCheck(unsigned int n, double** a)
{
	unsigned int i;
	
	/* Check for singularity */
	for (i = 0; i < n; i++)
	{
		if (a[i][i] == 0.0)
		{
			return i;
		}
	} // end of i loop
	
	return LowerTriangularInverse(n, a);
} // end of LowerTriangularInverseSingularityCheck function

/* 
This function computes the inverse of a real lower triangular matrix.
Modified from dtrti2.
*/
int LowerTriangularInverse(unsigned int n, double** a)
{
	int j;
	double ajj;
	
	int x, y;
	
	/* Compute inverse of lower triangular matrix. */
	for (j = n - 1; j >= 0; j--) // moving left through columns, starting from the end
	{
		a[j][j] = 1.0 / a[j][j];
		
		ajj = -a[j][j];
		
		if (j < n - 1)
		{
			/* Compute elements j + 1:n of j-th column. */
			MatrixVectorMultiplication(n - 1 - j, a, j);
			
			ScaleVectorByConstant(n - 1 - j, ajj, a, j + 1, j);
		}
	} // end of j loop
	
	return 0;
} // end of LowerTriangularInverse function

/*
This function performs the matrix-vector operation x := A * x,
where x is an n element vector and A is an n by n non-unit,
lower triangular matrix.
Modified from dtrmv.
*/
int MatrixVectorMultiplication(unsigned int n, double** a, unsigned int col_offset)
{
	/* n specifies the order of the matrix A */
	
	/* Quick return if possible. */
	if (n == 0)
	{
		return -2;
	}
	
	/* Start the operations. In this version the elements of A are
	accessed sequentiaK_inv_k_star_inv_y with one pass through A. */
	
	/* Form  x := A * x. */
	int i, j;
	double temp;
	
	for (j = n; j > 0; j--) // moving up rows
	{
		if (a[j + col_offset][col_offset] != 0.0)
		{
			temp = a[j + col_offset][col_offset];
			for (i = n; i >= j + 1; i--) // moving up rows, starting at the bottom
			{
				a[i + col_offset][col_offset] += temp * a[i + col_offset][j + col_offset];
			} // end of i loop
			a[j + col_offset][col_offset] *= a[j + col_offset][j + col_offset];
		}
	} // end of j loop
	
} // end of MatrixVectorMultiplication function

/*
This function scales a vector by a constant.
Modified from dscal.
*/
void ScaleVectorByConstant(unsigned int n, double da, double** a, unsigned int row_offset, unsigned int col_offset)
{
	unsigned int i;
	for (i = 0; i < n; i++) // moving down rows, starting at the offset
	{
		a[i + row_offset][col_offset] *= da;
	} // end of i loop
	
	return;
} // end of ScaleVectorByConstant function

/* This function efficiently recombines a lower cholesky decomposition inverse A^-1 = L^-1 * L^-T */
void RecombineLowerCholeskyDecompositionInverse(unsigned int n, double** L, double** A)
{
	unsigned int i, j, k;
	
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < i + 1; j++)
		{
			A[i][j] = 0.0;
			for (k = 0; k < j + 1; k++)
			{
				A[i][j] += L[i][k] * L[j][k];
			} // end of k loop
		} // end of j loop
	} // end of i loop
	
	return;
} // end of RecombineCholeskyDecompositionInverse function

/* This function optimizes kernel hyperparameters */
void OptimizeKernelHyperparameters(int kernel_type, unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** y, double** kernel_x_x, double** L, double** K_inv_y, unsigned int num_kernel_hyperparameters, double* kernel_hyperparameters)
{
	unsigned int i, j;
	
	double* d_negative_log_marginal_likelihood_d_kernel_hyperparameter;
	d_negative_log_marginal_likelihood_d_kernel_hyperparameter = malloc(sizeof(double) * num_kernel_hyperparameters);
	for (i = 0; i < num_kernel_hyperparameters; i++)
	{
		d_negative_log_marginal_likelihood_d_kernel_hyperparameter[i] = 0.0;
	} // end of i loop
	
	/* Read kernel hyperparameter optimization parameters */
	int* kernel_hyperparameter_bounds_type;
	kernel_hyperparameter_bounds_type = malloc(sizeof(int) * num_kernel_hyperparameters);
	for (i = 0; i < num_kernel_hyperparameters; i++)
	{
		kernel_hyperparameter_bounds_type[i] = 0;
	} // end of i loop
	
	double** kernel_hyperparameter_bounds_values;
	kernel_hyperparameter_bounds_values = malloc(sizeof(double*) * num_kernel_hyperparameters);
	for (i = 0; i < num_kernel_hyperparameters; i++)
	{
		kernel_hyperparameter_bounds_values[i] = malloc(sizeof(double) * 2);
		kernel_hyperparameter_bounds_values[i][0] = 0.0;
		kernel_hyperparameter_bounds_values[i][1] = 0.0;
	} // end of i loop
	
	ReadKernelHyperparameterOptimizationParameters(num_kernel_hyperparameters, &kernel_hyperparameter_bounds_type, &kernel_hyperparameter_bounds_values);
	
	/* Create array to hold alpha * alpha**T */
	double** alpha_alpha_t;
	alpha_alpha_t = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		alpha_alpha_t[i] = malloc(sizeof(double) * num_training_points);
		for (j = 0; j < num_training_points; j++)
		{
			alpha_alpha_t[i][j] = 0.0;
		} // end of j loop
	} // end of i loop

	/* Create array to hold K(X, X) inverse */
	double** kernel_x_x_inv;
	kernel_x_x_inv = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		kernel_x_x_inv[i] = malloc(sizeof(double) * num_training_points);
		for (j = 0; j < num_training_points; j++)
		{
			kernel_x_x_inv[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Create array to hold kernel hyperparameter gradient */
	double** d_kernel_d_kernel_hyperparameter;
	d_kernel_d_kernel_hyperparameter = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		d_kernel_d_kernel_hyperparameter[i] = malloc(sizeof(double) * num_training_points);
		for (j = 0; j < num_training_points; j++)
		{
			d_kernel_d_kernel_hyperparameter[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Create extra array that MAY be needed to hold intermediate kernel hyperparameter gradient results depending on kernel type */
	double** d_kernel_d_kernel_hyperparameter_temp;
	d_kernel_d_kernel_hyperparameter_temp = malloc(sizeof(double*) * num_training_points);
	for (i = 0; i < num_training_points; i++)
	{
		d_kernel_d_kernel_hyperparameter_temp[i] = malloc(sizeof(double) * num_training_points);
		for (j = 0; j < num_training_points; j++)
		{
			d_kernel_d_kernel_hyperparameter_temp[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	/* Iteratively optimize hyperparameters */
	KernelHyperparameterOptimizerLoop(kernel_type, num_training_points, num_dimensions, X_train, y, kernel_x_x, L, K_inv_y, alpha_alpha_t, kernel_x_x_inv, d_kernel_d_kernel_hyperparameter, d_kernel_d_kernel_hyperparameter_temp, num_kernel_hyperparameters, d_negative_log_marginal_likelihood_d_kernel_hyperparameter, kernel_hyperparameter_bounds_type, kernel_hyperparameter_bounds_values, kernel_hyperparameters);
	
	/* Free dynamic memory */
	for (i = 0; i < num_training_points; i++)
	{
		free(d_kernel_d_kernel_hyperparameter_temp[i]);
		free(d_kernel_d_kernel_hyperparameter[i]);
		free(kernel_x_x_inv[i]);
		free(alpha_alpha_t[i]);
	} // end of i loop
	free(d_kernel_d_kernel_hyperparameter_temp);
	free(d_kernel_d_kernel_hyperparameter);
	free(kernel_x_x_inv);
	free(alpha_alpha_t);
	
	for (i = 0; i < num_kernel_hyperparameters; i++)
	{
		free(kernel_hyperparameter_bounds_values[i]);
	} // end of i loop
	free(kernel_hyperparameter_bounds_values);
	free(kernel_hyperparameter_bounds_type);
	free(d_negative_log_marginal_likelihood_d_kernel_hyperparameter);
	
	return;
} // end of OptimizeKernelHyperparameters function

/* This function reads the kernel hyperparameter optimization parameters */
void ReadKernelHyperparameterOptimizationParameters(unsigned int num_kernel_hyperparameters, int** kernel_hyperparameter_bounds_type, double*** kernel_hyperparameter_bounds_values)
{
	unsigned int i;
	int system_return = 0;
	
	/* Get hyperparameter bounds existence */
	FILE* infile_kernel_hyperparameter_bounds_type = fopen("inputs/kernel_hyperparameter_bounds_type.txt", "r");
	for (i = 0; i < num_kernel_hyperparameters; i++)
	{
		system_return = fscanf(infile_kernel_hyperparameter_bounds_type, "%d\n", &(*kernel_hyperparameter_bounds_type)[i]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/kernel_hyperparameter_bounds_type.txt\n");
		}
		printf("i = %u, kernel_hyperparameter_bounds_type[i] = %d\n", i, (*kernel_hyperparameter_bounds_type)[i]);
	} // end of i loop
	fclose(infile_kernel_hyperparameter_bounds_type);
	
	/* Get hyperparameter bounds values */
	FILE* infile_kernel_hyperparameter_bounds_values = fopen("inputs/kernel_hyperparameter_bounds_values.txt", "r");
	for (i = 0; i < num_kernel_hyperparameters; i++)
	{
		system_return = fscanf(infile_kernel_hyperparameter_bounds_values, "%lf\t%lf\n", &(*kernel_hyperparameter_bounds_values)[i][0], &(*kernel_hyperparameter_bounds_values)[i][1]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/kernel_hyperparameter_bounds_values.txt\n");
		}
		printf("i = %u, kernel_hyperparameter_bounds_values[i][0] = %.16f, kernel_hyperparameter_bounds_values[i][1] = %.16f\n", i, (*kernel_hyperparameter_bounds_values)[i][0], (*kernel_hyperparameter_bounds_values)[i][1]);
	} // end of i loop
	fclose(infile_kernel_hyperparameter_bounds_values);
	
	return;
} // end of ReadKernelHyperparameterOptimizationParameters function

/* This function performs the kernel hyperparameter optimzation loop */
void KernelHyperparameterOptimizerLoop(int kernel_type, unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** y, double** kernel_x_x, double** L, double** K_inv_y, double** alpha_alpha_t, double** kernel_x_x_inv, double** d_kernel_d_kernel_hyperparameter, double** d_kernel_d_kernel_hyperparameter_temp, unsigned int num_kernel_hyperparameters, double* d_negative_log_marginal_likelihood_d_kernel_hyperparameter, int* kernel_hyperparameter_bounds_type, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters)
{
	unsigned int i, j, iterations = 0;
	int error = 0;
	double log_marginal_likelihood = 0.0;
	
	// long int nmax = 1024, mmax = 17;
	/* nmax is the dimension of the largest problem to be solved. */
	/* mmax is the maximum number of limited memory corrections. */
	
	/* Declare the variables needed by the code. 
	   A description of all these variables is given at the end of 
	   the driver. */
	static long int taskValue, csaveValue;
	static long int* task = &taskValue;
	static long int* csave = &csaveValue;
	static long int lsave[4];
	static long int n, m, iprint, nbd[1024], iwa[3072], isave[44];
	static double f, factr, pgtol, x[1024], l[1024], u[1024], g[1024], dsave[29];
	static double wa[43251];  // 2 * mmax * nmax + 5 * nmax + 11 * mmax * mmax + 8 * mmax
	
	/* We wish to have output at every iteration. */
	iprint = 101;
	
	/* We specify the tolerances in the stopping criteria. */
	factr = 1e1;
	pgtol = 1e-5;
	
	/* We specify the dimension n of the sample problem and the number */
	/* m of limited memory corrections stored.  (n and m should not */
	/* exceed the limits nmax and mmax respectively.) */
	n = num_kernel_hyperparameters;
	m = 5;
	
	/* We now provide nbd which defines the bounds on the variables: */
	/* l specifies the lower bounds, */
	/* u specifies the upper bounds. */
	for (i = 0; i < n; i ++)
	{
		nbd[i] = kernel_hyperparameter_bounds_type[i];
		l[i] = kernel_hyperparameter_bounds_values[i][0];
		u[i] = kernel_hyperparameter_bounds_values[i][1];
		
		printf("i = %u, nbd = %ld, l = %.16f, u = %.16f\n", i, nbd[i], l[i], u[i]);
	}
	
	/* We now define the starting point. */
	for (i = 0; i < n; i++)
	{
		x[i] = 1.;
	}
	
	/* We start the iteration by initializing task. */
	*task = (long int)START;
	
	/*		------- the beginning of the loop ---------- */
L111:
	/* This is the call to the L-BFGS-B code. */
	setulb(n, m, x, l, u, nbd, &f, g, factr, pgtol, wa, iwa, task, iprint, csave, lsave, isave, dsave);
	
	if (IS_FG(*task))
	{
		/* The minimization routine has returned to request the */
		/* function f and gradient g values at the current x. */
		
		/************************************************/
		/********************FUNCTION********************/
		/************************************************/
		
		/* Compute function value f for the sample problem. */
		
		/* Update kernel hyperparameters */
		for (i = 0; i < n; i++)
		{
			kernel_hyperparameters[i] = x[i];
		} // end of i loop
		
		/* Calculate kernel K(X, X) */
		CalculateKernel(kernel_type, kernel_hyperparameters, num_training_points, num_training_points, num_dimensions, X_train, X_train, kernel_x_x);
		if (kernel_type == 1) // squared exponential
		{
			for (i = 0; i < num_training_points; i++)
			{
				for (j = 0; j < num_training_points; j++)
				{
					/* Shift by the noise variance */
					if (i == j)
					{
						kernel_x_x[i][j] += kernel_hyperparameters[2] * kernel_hyperparameters[2];
					}
				} // end of j loop
			} // end of i loop
		}
		
		/* Perform Cholesky decomposition on our real symmetric positive definite matrix K(X, X) = L * L**T */
		for (i = 0; i < num_training_points; i++)
		{
			for (j = 0; j < num_training_points; j++)
			{
				L[i][j] = kernel_x_x[i][j];
			} // end of j loop
		} // end of i loop

		error = CholeskyDecomposition(num_training_points, L);
		
		/* Now solve for L * Z = y for Z */
		for (i = 0; i < num_training_points; i++)
		{
			K_inv_y[i][0] = y[i][0];
		} // end of i loop

		error = SolveLowerCholeskyFactorizationMatrixEquation(num_training_points, 1, L, K_inv_y);
		if (error != 0)
		{
			printf("ERROR: SolveLowerCholeskyFactorizationMatrixEquation, error = %d\n", error);
		}
		
		/* Evaluate log marginal likelihood with current hyperparameters */
		iterations++;
		log_marginal_likelihood = CalculateLogMarginalLikelihood(num_training_points, L, K_inv_y);
		printf("\niterations = %u, log_marginal_likelihood = %.16f\n", iterations, log_marginal_likelihood);
		
		/* Maximize log marginal likelihood so minimize the negative */
		f = -log_marginal_likelihood;
		
		/************************************************/
		/********************GRADIENT********************/
		/************************************************/
		
		/* Compute gradient g for the sample problem. */
		
		/* Calculate alpha * alpha**T */
		for (i = 0; i < num_training_points; i++)
		{
			for (j = 0; j < num_training_points; j++)
			{
				alpha_alpha_t[i][j] = 0.0;
			} // end of j loop
		} // end of i loop
		
		MatrixMatrixMultiplication(num_training_points, num_training_points, 1, K_inv_y, K_inv_y, 0, 1, alpha_alpha_t);
		
		/* Calculate K(X, X) inverse */
		for (i = 0; i < num_training_points; i++)
		{
			for (j = 0; j < num_training_points; j++)
			{
				kernel_x_x_inv[i][j] = L[i][j];
			} // end of j loop
		} // end of i loop

		error = LowerCholeskyInverse(num_training_points, kernel_x_x_inv);
		if (error != 0)
		{
			printf("ERROR: LowerCholeskyInverse returned error code %d\n", error);
		}

		for (i = 0; i < num_training_points; i++)
		{
			for (j = 0; j < num_training_points; j++)
			{
				alpha_alpha_t[i][j] -= kernel_x_x_inv[i][j];
			} // end of j loop
		} // end of i loop
		
		if (kernel_type == 0) // linear
		{
			CalculateNegativeLogMarginalLikelihoodLinearKernelHyperparameterGradients(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, kernel_hyperparameters, d_negative_log_marginal_likelihood_d_kernel_hyperparameter);
		} // end of linear
		else // squared exponential
		{
			CalculateNegativeLogMarginalLikelihoodSquaredExponentialKernelHyperparameterGradients(num_training_points, num_dimensions, X_train, kernel_x_x, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, d_kernel_d_kernel_hyperparameter_temp, kernel_hyperparameters, d_negative_log_marginal_likelihood_d_kernel_hyperparameter);
		} // end of squared exponential
		
		/* Update gradients */
		for (i = 0; i < n; i++)
		{
			g[i] = d_negative_log_marginal_likelihood_d_kernel_hyperparameter[i];
		} // end of i loop
		
		/* Go back to the minimization routine. */
		goto L111;
	}

	if ((*task) == NEW_X)
	{
		/* The minimization routine has returned with a new iterate, */
		/* and we have opted to continue the iteration. */
		goto L111;
	}
	
	/*		   ---------- the end of the loop ------------- */
	
	/* If task is neither FG nor NEW_X we terminate execution. */

	/* Final update of kernel hyperparameters */
	for (i = 0; i < n; i++)
	{
		kernel_hyperparameters[i] = x[i];
	} // end of i loop
	
	return;
} // end of KernelHyperparameterOptimizerLoop function

/* This function calculates the gradient of the negative log marginal likelihood with respect to the linear kernel's hyperparameters */
void CalculateNegativeLogMarginalLikelihoodLinearKernelHyperparameterGradients(unsigned int num_training_points, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double* kernel_hyperparameters, double* d_negative_log_marginal_likelihood_d_kernel_hyperparameter)
{
	/* Constant */
	CalculateLinearKernelConstantHyperparameterGradient(num_training_points, d_kernel_d_kernel_hyperparameter);
	
	CalculateNegativeLogMarginalLikelihoodGradient(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, 0, kernel_hyperparameters, d_negative_log_marginal_likelihood_d_kernel_hyperparameter);
	
	return;
} // end of CalculateNegativeLogMarginalLikelihoodLinearKernelHyperparameterGradients function

/* This function calculates the gradient of the linear kernel with respect to the constant hyperparameter */
void CalculateLinearKernelConstantHyperparameterGradient(unsigned int num_training_points, double** d_kernel_d_kernel_hyperparameter)
{
	unsigned int i, j;
	
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_training_points; j++)
		{
			d_kernel_d_kernel_hyperparameter[i][j] = 0.0;
		} // end of j loop
	} // end of i loop
	
	return;
} // end of CalculateLinearKernelConstantHyperparameterGradient function

/* This function calculates the gradient of the negative log marginal likelihood with respect to the squared exponential kernel's hyperparameters */
void CalculateNegativeLogMarginalLikelihoodSquaredExponentialKernelHyperparameterGradients(unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** kernel_x_x, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double** d_kernel_d_kernel_hyperparameter_temp, double* kernel_hyperparameters, double* d_negative_log_marginal_likelihood_d_kernel_hyperparameter)
{
	/* Length-scale */
	CalculateSquaredExponentialKernelLengthScaleHyperparameterGradient(num_training_points, num_dimensions, X_train, kernel_x_x, kernel_hyperparameters, d_kernel_d_kernel_hyperparameter_temp, d_kernel_d_kernel_hyperparameter);
	CalculateNegativeLogMarginalLikelihoodGradient(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, 0, kernel_hyperparameters, d_negative_log_marginal_likelihood_d_kernel_hyperparameter);
	
	/* Signal variance */
	CalculateSquaredExponentialKernelSignalVarianceHyperparameterGradient(num_training_points, kernel_x_x, kernel_hyperparameters, d_kernel_d_kernel_hyperparameter);
	CalculateNegativeLogMarginalLikelihoodGradient(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, 1, kernel_hyperparameters, d_negative_log_marginal_likelihood_d_kernel_hyperparameter);
	
	/* Noise variance */
	CalculateSquaredExponentialKernelNoiseVarianceHyperparameterGradient(num_training_points, kernel_hyperparameters, d_kernel_d_kernel_hyperparameter);
	CalculateNegativeLogMarginalLikelihoodGradient(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, 2, kernel_hyperparameters, d_negative_log_marginal_likelihood_d_kernel_hyperparameter);
	
	return;
} // end of CalculateNegativeLogMarginalLikelihoodSquaredExponentialKernelHyperparameterGradients function

/* This function calculates the gradient of the squared exponential kernel with respect to the length-scale hyperparameter */
void CalculateSquaredExponentialKernelLengthScaleHyperparameterGradient(unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** kernel_x_x, double* kernel_hyperparameters, double** d_kernel_d_kernel_hyperparameter_temp, double** d_kernel_d_kernel_hyperparameter)
{
	unsigned int i, j, k;
	double gradient = 0.0, a_squared_sum = 0.0, b_squared_sum = 0.0;
	
	MatrixMatrixMultiplication(num_training_points, num_training_points, num_dimensions, X_train, X_train, 0, 1, d_kernel_d_kernel_hyperparameter_temp);

	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_training_points; j++)
		{
			a_squared_sum = 0.0;
			b_squared_sum = 0.0;
			for (k = 0; k < num_dimensions; k++)
			{
				a_squared_sum += X_train[i][k] * X_train[i][k];
				b_squared_sum += X_train[j][k] * X_train[j][k];
			} // end of k loop
			
			d_kernel_d_kernel_hyperparameter_temp[i][j] = a_squared_sum + b_squared_sum - 2.0 * d_kernel_d_kernel_hyperparameter_temp[i][j];
		} // end of j loop
	} // end of i loop

	MatrixMatrixMultiplication(num_training_points, num_training_points, num_training_points, kernel_x_x, d_kernel_d_kernel_hyperparameter_temp, 0, 0, d_kernel_d_kernel_hyperparameter);

	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_training_points; j++)
		{
			d_kernel_d_kernel_hyperparameter[i][j] /= pow(kernel_hyperparameters[0], 3);
		} // end of j loop
	} // end of i loop
	
	return;
} // end of CalculateSquaredExponentialKernelLengthScaleHyperparameterGradient function

/* This function calculates the gradient of the squared exponential kernel with respect to the signal variance hyperparameter */
void CalculateSquaredExponentialKernelSignalVarianceHyperparameterGradient(unsigned int num_training_points, double** kernel_x_x, double* kernel_hyperparameters, double** d_kernel_d_kernel_hyperparameter)
{
	unsigned int i, j;
	
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_training_points; j++)
		{
			if (i == j)
			{
				d_kernel_d_kernel_hyperparameter[i][j] = 2.0 * (kernel_x_x[i][j] - kernel_hyperparameters[2] * kernel_hyperparameters[2]) / kernel_hyperparameters[1];
			}
			else
			{
				d_kernel_d_kernel_hyperparameter[i][j] = 2.0 * kernel_x_x[i][j] / kernel_hyperparameters[1];
			}
		} // end of j loop
	} // end of i loop
	
	return;
} // end of CalculateSquaredExponentialKernelSignalVarianceHyperparameterGradient function

/* This function calculates the gradient of the squared exponential kernel with respect to the noise variance hyperparameter */
void CalculateSquaredExponentialKernelNoiseVarianceHyperparameterGradient(unsigned int num_training_points, double* kernel_hyperparameters, double** d_kernel_d_kernel_hyperparameter)
{
	unsigned int i, j;
	
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_training_points; j++)
		{
			if (i == j)
			{
				d_kernel_d_kernel_hyperparameter[i][j] = 2.0 * kernel_hyperparameters[2];
			}
			else
			{
				d_kernel_d_kernel_hyperparameter[i][j] = 0.0;
			}
		} // end of j loop
	} // end of i loop
	
	return;
} // end of CalculateSquaredExponentialKernelNoiseVarianceHyperparameterGradient function

/* This function calculates the negative log marginal likelihood gradient with respect to a kernel's hyperparameter */
void CalculateNegativeLogMarginalLikelihoodGradient(unsigned int num_training_points, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, unsigned int hyperparameter_index, double* kernel_hyperparameters, double* d_negative_log_marginal_likelihood_d_kernel_hyperparameter)
{
	int error = 0;
	double d_log_marginal_likelihood_d_hyperparameter = 0.0;
	
	/* Calculate d_log_marginal_likelihood_d_hyperparameter */
	error = MatrixMatrixMultiplcationTrace(num_training_points, num_training_points, num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, 0, 0, &d_log_marginal_likelihood_d_hyperparameter);
	if (error != 0)
	{
		printf("ERROR: MatrixMatrixMultiplcationTrace returned error code %d\n", error);
	}
	else
	{
		d_negative_log_marginal_likelihood_d_kernel_hyperparameter[hyperparameter_index] = -d_log_marginal_likelihood_d_hyperparameter;
	}
	
	return;
} // end of CalculateNegativeLogMarginalLikelihoodGradient function

/* This function performs matrix multiplication between two given matrices */
void MatrixMatrixMultiplication(unsigned int m, unsigned int n, unsigned int p, double** A, double** B, int transpose_A, int transpose_B, double** C)
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
} // end of MatrixMatrixMultiplication function

/* This function calculates the trace of the matrix-matrix multiplication of matrices A and B */
int MatrixMatrixMultiplcationTrace(unsigned int m, unsigned int n, unsigned int p, double** A, double** B, int transpose_A, int transpose_B, double* trace)
{
	unsigned int i, j;
	
	/* Quick return if possible */
	if (m <= 0 || n <= 0 || p <=0) // axis needs to have at least some length
	{
		return 1;
	}
	else if (m != n) // to ensure square matrix result for trace
	{
		return 2;
	}
	
	(*trace) = 0.0;
	if (transpose_A == 0)
	{
		if (transpose_B == 0)
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < p; j++)
				{
					(*trace) += A[i][j] * B[j][i];
				} // end of j loop
			} // end of i loop
		}
		else
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < p; j++)
				{
					(*trace) += A[i][j] * B[i][j];
				} // end of j loop
			} // end of i loop
		}
	}
	else
	{
		if (transpose_B == 0)
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < p; j++)
				{
					(*trace) += A[j][i] * B[j][i];
				} // end of j loop
			} // end of i loop
		}
		else
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < p; j++)
				{
					(*trace) += A[j][i] * B[i][j];
				} // end of j loop
			} // end of i loop
		}
	}
	
	return 0;
} // end of MatrixMatrixMultiplcationTrace function

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

/* This function calculate log marginal likelihood of gaussian process of training points */
double CalculateLogMarginalLikelihood(unsigned int num_training_points, double** L, double** K_inv_y)
{
	unsigned int i;
	double log_marginal_likelihood = 0.0;

	/* Find first term, -0.5 * y**T * (K(X, X) + sigma_n^2 * I)^-1 * y */
	log_marginal_likelihood = -0.5 * VectorDotProductRank2(num_training_points, K_inv_y, K_inv_y, 1, 1);

	/* Next add second term, -0.5 * log(det(K(X, X) + sigma_n^2 * I)) */
	for (i = 0; i < num_training_points; i++)
	{
		log_marginal_likelihood -= log(L[i][i]);
	} // end of i loop

	/* Lastly add third term, the normalizing factor: -0.5 * n * log(2 * Pi) */
	log_marginal_likelihood -= 0.5 * num_training_points * log(2.0 * M_PI);
	
	return log_marginal_likelihood;
} // end of CalculateLogMarginalLikelihood function

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

void setulb(long int n, long int m, double* x, double* l, double* u, long int* nbd, double* f, double* g, double factr, double pgtol, double* wa, long int* iwa, long int* task, long int iprint, long int* csave, long int* lsave, long int* isave, double* dsave)
{
    /*
	************ 

	Subroutine setulb 

	This subroutine partitions the working arrays wa and iwa, and 
	  then uses the limited memory BFGS method to solve the bound 
	  constrained optimization problem by calling mainlb. 
	  (The direct method will be used in the subspace minimization.) 

	n is an integer variable. 
	  On entry n is the dimension of the problem. 
	  On exit n is unchanged. 

	m is an integer variable. 
	  On entry m is the maximum number of variable metric corrections 
	    used to define the limited memory matrix. 
	  On exit m is unchanged. 

	x is a double precision array of dimension n. 
	  On entry x is an approximation to the solution. 
	  On exit x is the current approximation. 

	l is a double precision array of dimension n. 
	  On entry l is the lower bound on x. 
	  On exit l is unchanged. 

	u is a double precision array of dimension n. 
	  On entry u is the upper bound on x. 
	  On exit u is unchanged. 

	nbd is an integer array of dimension n. 
	  On entry nbd represents the type of bounds imposed on the 
	    variables, and must be specified as follows: 
	    nbd(i)=0 if x(i) is unbounded, 
	           1 if x(i) has only a lower bound, 
	           2 if x(i) has both lower and upper bounds, and 
	           3 if x(i) has only an upper bound. 
	  On exit nbd is unchanged. 

	f is a double precision variable. 
	  On first entry f is unspecified. 
	  On final exit f is the value of the function at x. 

	g is a double precision array of dimension n. 
	  On first entry g is unspecified. 
	  On final exit g is the value of the gradient at x. 

	factr is a double precision variable. 
	  On entry factr >= 0 is specified by the user.  The iteration 
	    will stop when 

	    (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch 

	    where epsmch is the machine precision, which is automatically 
	    generated by the code. Typical values for factr: 1.d+12 for 
	    low accuracy; 1.d+7 for moderate accuracy; 1.d+1 for extremely 
	    high accuracy. 
	  On exit factr is unchanged. 

	pgtol is a double precision variable. 
	  On entry pgtol >= 0 is specified by the user.  The iteration 
	    will stop when 

	            max{|proj g_i | i = 1, ..., n} <= pgtol 

	    where pg_i is the ith component of the projected gradient. 
	  On exit pgtol is unchanged. 

	wa is a double precision working array of length 
	  (2mmax + 5)nmax + 12mmax^2 + 12mmax. 

	iwa is an integer working array of length 3nmax. 

	task is a working string of characters of length 60 indicating 
	  the current job when entering and quitting this subroutine. 

	iprint is an integer variable that must be set by the user. 
	  It controls the frequency and type of output generated: 
	   iprint<0    no output is generated; 
	   iprint=0    print only one line at the last iteration; 
	   0<iprint<99 print also f and |proj g| every iprint iterations; 
	   iprint=99   print details of every iteration except n-vectors; 
	   iprint=100  print also the changes of active set and final x; 
	   iprint>100  print details of every iteration including x and g; 
	  When iprint > 0, the file iterate.dat will be created to 
	                   summarize the iteration. 

	csave is a working string of characters of length 60. 

	lsave is a logical working array of dimension 4. 
	  On exit with 'task' = NEW_X, the following information is 
	                                                        available: 
	    If lsave(1) = .true.  then  the initial X has been replaced by 
	                                its projection in the feasible set; 
	    If lsave(2) = .true.  then  the problem is constrained; 
	    If lsave(3) = .true.  then  each variable has upper and lower 
	                                bounds; 

	isave is an integer working array of dimension 44. 
	  On exit with 'task' = NEW_X, the following information is 
	                                                        available: 
	    isave(22) = the total number of intervals explored in the 
	                    search of Cauchy points; 
	    isave(26) = the total number of skipped BFGS updates before 
	                    the current iteration; 
	    isave(30) = the number of current iteration; 
	    isave(31) = the total number of BFGS updates prior the current 
	                    iteration; 
	    isave(33) = the number of intervals explored in the search of 
	                    Cauchy point in the current iteration; 
	    isave(34) = the total number of function and gradient 
	                    evaluations; 
	    isave(36) = the number of function value or gradient 
	                             evaluations in the current iteration; 
	    if isave(37) = 0  then the subspace argmin is within the box; 
	    if isave(37) = 1  then the subspace argmin is beyond the box; 
	    isave(38) = the number of free variables in the current 
	                    iteration; 
	    isave(39) = the number of active constraints in the current 
	                    iteration; 
	    n + 1 - isave(40) = the number of variables leaving the set of 
	                      active constraints in the current iteration; 
	    isave(41) = the number of variables entering the set of active 
	                    constraints in the current iteration. 

	dsave is a double precision working array of dimension 29. 
	  On exit with 'task' = NEW_X, the following information is 
	                                                        available: 
	    dsave(1) = current 'theta' in the BFGS matrix; 
	    dsave(2) = f(x) in the previous iteration; 
	    dsave(3) = factr*epsmch; 
	    dsave(4) = 2-norm of the line search direction vector; 
	    dsave(5) = the machine precision epsmch generated by the code; 
	    dsave(7) = the accumulated time spent on searching for 
	                                                    Cauchy points; 
	    dsave(8) = the accumulated time spent on 
	                                            subspace minimization; 
	    dsave(9) = the accumulated time spent on line search; 
	    dsave(11) = the slope of the line search function at 
	                             the current point of line search; 
	    dsave(12) = the maximum relative step length imposed in 
	                                                      line search; 
	    dsave(13) = the infinity norm of the projected gradient; 
	    dsave(14) = the relative step length in the line search; 
	    dsave(15) = the slope of the line search function at 
	                            the starting point of the line search; 
	    dsave(16) = the square of the 2-norm of the line search 
	                                                 direction vector. 

	Subprograms called: 

	  L-BFGS-B Library ... mainlb. 


	References: 

	  [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited 
	  memory algorithm for bound constrained optimization'', 
	  SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208. 

	  [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: a 
	  limited memory FORTRAN code for solving bound constrained 
	  optimization problems'', Tech. Report, NAM-11, EECS Department, 
	  Northwestern University, 1994. 

	  (Postscript files of these papers are available via anonymous 
	   ftp to eecs.nwu.edu in the directory pub/lbfgs/lbfgs_bcm.) 

	                      *  *  * 

	NEOS, November 1994. (Latest revision June 1996.) 
	Optimization Technology Center. 
	Argonne National Laboratory and Northwestern University. 
	Written by 
	                   Ciyou Zhu 
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal. 


	************ 
	*/
	
	/* Local variables */
	static long int ld, lr, lt, lz, lwa, lwn, lss, lxp, lws, lwt, lsy, lwy, lsnd;

	/* Parameter adjustments */
	--lsave;
	--isave;
	--dsave;

	/* Function Body */
	if ( *task == START )
	{
		isave[1] = m * n;
		isave[2] = m * m;
		isave[3] = 4 * isave[2];
		isave[4] = 1;
		/* ws	  m * n */
		isave[5] = isave[4] + isave[1];
		/* wy	  m * n */
		isave[6] = isave[5] + isave[1];
		/* wsy	 m**2 */
		isave[7] = isave[6] + isave[2];
		/* wss	 m**2 */
		isave[8] = isave[7] + isave[2];
		/* wt	  m**2 */
		isave[9] = isave[8] + isave[2];
		/* wn	  4 * m**2 */
		isave[10] = isave[9] + isave[3];
		/* wsnd	4 * m**2 */
		isave[11] = isave[10] + isave[3];
		/* wz	  n */
		isave[12] = isave[11] + n;
		/* wr	  n */
		isave[13] = isave[12] + n;
		/* wd	  n */
		isave[14] = isave[13] + n;
		/* wt	  n */
		isave[15] = isave[14] + n;
		/* wxp	 n */
		isave[16] = isave[15] + n;
		/* wa	  8 * m */
	}
	lws = isave[4];
	lwy = isave[5];
	lsy = isave[6];
	lss = isave[7];
	lwt = isave[8];
	lwn = isave[9];
	lsnd = isave[10];
	lz = isave[11];
	lr = isave[12];
	ld = isave[13];
	lt = isave[14];
	lxp = isave[15];
	lwa = isave[16];
	
	mainlb(n, m, x, l, u, nbd, f, g, factr, pgtol, &wa[lws], &wa[lwy], &wa[lsy], &wa[lss], &wa[lwt], &wa[lwn], &wa[lsnd], &wa[lz], &wa[lr], &wa[ld], &wa[lt], &wa[lxp], &wa[lwa], &iwa[1], &iwa[n + 1], &iwa[(n << 1) + 1], task, iprint, csave, &lsave[1], &isave[22], &dsave[1]);
	
	return;
} // end of setulb function

void mainlb(long int n, long int m, double* x, double* l, double* u, long int* nbd, double* f, double* g, double factr, double pgtol, double* ws, double* wy, double* sy, double* ss, double* wt, double* wn, double* snd, double* z__, double* r__, double* d__, double* t, double* xp, double* wa, long int* index, long int* iwhere, long int* indx2, long int* task, long int iprint, long int* csave, long int* lsave, long int* isave, double* dsave)
{
    /*
	************ 
	
	Subroutine mainlb 

	This subroutine solves bound constrained optimization problems by 
	  using the compact formula of the limited memory BFGS updates. 

	n is an integer variable. 
	  On entry n is the number of variables. 
	  On exit n is unchanged. 

	m is an integer variable. 
	  On entry m is the maximum number of variable metric 
	     corrections allowed in the limited memory matrix. 
	  On exit m is unchanged. 

	x is a double precision array of dimension n. 
	  On entry x is an approximation to the solution. 
	  On exit x is the current approximation. 

	l is a double precision array of dimension n. 
	  On entry l is the lower bound of x. 
	  On exit l is unchanged. 

	u is a double precision array of dimension n. 
	  On entry u is the upper bound of x. 
	  On exit u is unchanged. 

	nbd is an integer array of dimension n. 
	  On entry nbd represents the type of bounds imposed on the 
	    variables, and must be specified as follows: 
	    nbd(i)=0 if x(i) is unbounded, 
	           1 if x(i) has only a lower bound, 
	           2 if x(i) has both lower and upper bounds, 
	           3 if x(i) has only an upper bound. 
	  On exit nbd is unchanged. 

	f is a double precision variable. 
	  On first entry f is unspecified. 
	  On final exit f is the value of the function at x. 

	g is a double precision array of dimension n. 
	  On first entry g is unspecified. 
	  On final exit g is the value of the gradient at x. 

	factr is a double precision variable. 
	  On entry factr >= 0 is specified by the user.  The iteration 
	    will stop when 

	    (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch 

	    where epsmch is the machine precision, which is automatically 
	    generated by the code. 
	  On exit factr is unchanged. 

	pgtol is a double precision variable. 
	  On entry pgtol >= 0 is specified by the user.  The iteration 
	    will stop when 

	            max{|proj g_i | i = 1, ..., n} <= pgtol 

	    where pg_i is the ith component of the projected gradient. 
	  On exit pgtol is unchanged. 

	ws, wy, sy, and wt are double precision working arrays used to 
	  store the following information defining the limited memory 
	     BFGS matrix: 
	     ws, of dimension n x m, stores S, the matrix of s-vectors; 
	     wy, of dimension n x m, stores Y, the matrix of y-vectors; 
	     sy, of dimension m x m, stores S'Y; 
	     ss, of dimension m x m, stores S'S; 
	     yy, of dimension m x m, stores Y'Y; 
	     wt, of dimension m x m, stores the Cholesky factorization 
	                             of (theta*S'S+LD^(-1)L'); see eq. 
	                             (2.26) in [3]. 

	wn is a double precision working array of dimension 2m x 2m 
	  used to store the LEL^T factorization of the indefinite matrix 
	            K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ] 
	                [L_a -R_z           theta*S'AA'S ] 

	  where     E = [-I  0] 
	                [ 0  I] 

	snd is a double precision working array of dimension 2m x 2m 
	  used to store the lower triangular part of 
	            N = [Y' ZZ'Y   L_a'+R_z'] 
	                [L_a +R_z  S'AA'S   ] 

	zn,rn,dn,tn, xpn,wa(8*m) are double precision working arrays. 
	  z  is used at different times to store the Cauchy point and 
	     the Newton point. 
	  xp is used to safeguard the projected Newton direction 

	sg(m),sgo(m),yg(m),ygo(m) are double precision working arrays. 

	index is an integer working array of dimension n. 
	  In subroutine freev, index is used to store the free and fixed 
	     variables at the Generalized Cauchy Point (GCP). 

	iwhere is an integer working array of dimension n used to record 
	  the status of the vector x for GCP computation. 
	  iwhere(i)=0 or -3 if x(i) is free and has bounds, 
	            1       if x(i) is fixed at l(i), and l(i) .ne. u(i) 
	            2       if x(i) is fixed at u(i), and u(i) .ne. l(i) 
	            3       if x(i) is always fixed, i.e.,  u(i)=x(i)=l(i) 
	           -1       if x(i) is always free, i.e., no bounds on it. 

	indx2 is an integer working array of dimension n. 
	  Within subroutine cauchy, indx2 corresponds to the array iorder. 
	  In subroutine freev, a list of variables entering and leaving 
	  the free set is stored in indx2, and it is passed on to 
	  subroutine formk with this information. 

	task is a working string of characters of length 60 indicating 
	  the current job when entering and leaving this subroutine. 

	iprint is an INTEGER variable that must be set by the user. 
	  It controls the frequency and type of output generated: 
	   iprint<0    no output is generated; 
	   iprint=0    print only one line at the last iteration; 
	   0<iprint<99 print also f and |proj g| every iprint iterations; 
	   iprint=99   print details of every iteration except n-vectors; 
	   iprint=100  print also the changes of active set and final x; 
	   iprint>100  print details of every iteration including x and g; 
	  When iprint > 0, the file iterate.dat will be created to 
	                   summarize the iteration. 

	csave is a working string of characters of length 60. 

	lsave is a logical working array of dimension 4. 

	isave is an integer working array of dimension 23. 

	dsave is a double precision working array of dimension 29. 


	Subprograms called 

	  L-BFGS-B Library ... cauchy, subsm, lnsrlb, formk, 

	   errclb, prn1lb, prn2lb, prn3lb, active, projgr, 

	   freev, cmprlb, matupd, formt. 

	  Minpack2 Library ... timer 

	  Linpack Library ... dcopy, ddot. 


	References: 

	  [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited 
	  memory algorithm for bound constrained optimization'', 
	  SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208. 

	  [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN 
	  Subroutines for Large Scale Bound Constrained Optimization'' 
	  Tech. Report, NAM-11, EECS Department, Northwestern University, 
	  1994. 

	  [3] R. Byrd, J. Nocedal and R. Schnabel "Representations of 
	  Quasi-Newton Matrices and their use in Limited Memory Methods'', 
	  Mathematical Programming 63 (1994), no. 4, pp. 129-156. 

	  (Postscript files of these papers are available via anonymous 
	   ftp to eecs.nwu.edu in the directory pub/lbfgs/lbfgs_bcm.) 

	                      *  *  * 

	NEOS, November 1994. (Latest revision June 1996.) 
	Optimization Technology Center. 
	Argonne National Laboratory and Northwestern University. 
	Written by 
	                   Ciyou Zhu 
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
	
	************ 
	*/
	
	/* System generated locals */
	long int ws_dim1, ws_offset, wy_dim1, wy_offset, sy_dim1, sy_offset, ss_dim1, ss_offset, wt_dim1, wt_offset, wn_dim1, wn_offset, snd_dim1, snd_offset, i__1 = 0;

	/* Local variables */
	static long int i, k;
	static double gd, dr, rr, dtd;
	static long int col;
	static double tol;
	static long int wrk;
	static double stp, cpu1, cpu2;
	static long int head;
	static double fold;
	static long int nact;
	static double ddum;
	static long int info, nseg;
	static double time;
	static long int nfgv, ifun, iter;
	static long int wordTemp;
	static long int *word=&wordTemp;
	static double time1, time2;
	static long int iback;
	static double gdold;
	static long int nfree;
	static long int boxed;
	static long int itail;
	static double theta;
	static double dnorm;
	static long int nskip, iword;
	static double xstep, stpmx;
	static long int ileave;
	static double cachyt;
	static long int itfile;
	static double epsmch;
	static long int updatd;
	static double sbtime;
	static long int prjctd;
	static long int iupdat;
	static double sbgnrm;
	static long int cnstnd;
	static long int nenter;
	static double lnscht;
	static long int nintol;

	/* Parameter adjustments */
	--indx2;
	--iwhere;
	--index;
	--xp;
	--t;
	--d__;
	--r__;
	--z__;
	--g;
	--nbd;
	--u;
	--l;
	--wa;
	snd_dim1 = 2 * m;
	snd_offset = 1 + snd_dim1;
	snd -= snd_offset;
	wn_dim1 = 2 * m;
	wn_offset = 1 + wn_dim1;
	wn -= wn_offset;
	wt_dim1 = m;
	wt_offset = 1 + wt_dim1;
	wt -= wt_offset;
	ss_dim1 = m;
	ss_offset = 1 + ss_dim1;
	ss -= ss_offset;
	sy_dim1 = m;
	sy_offset = 1 + sy_dim1;
	sy -= sy_offset;
	wy_dim1 = n;
	wy_offset = 1 + wy_dim1;
	wy -= wy_offset;
	ws_dim1 = n;
	ws_offset = 1 + ws_dim1;
	ws -= ws_offset;
	--lsave;
	--isave;
	--dsave;

	if (*task == START)
	{
		epsmch = DBL_EPSILON;
		timer(&time1);
		/* Initialize counters and scalars when task = 'START'. */
		/* for the limited memory BFGS matrices: */
		col = 0;
		head = 1;
		theta = 1.;
		iupdat = 0;
		updatd = 0;
		iback = 0;
		itail = 0;
		iword = 0;
		nact = 0;
		ileave = 0;
		nenter = 0;
		fold = 0.;
		dnorm = 0.;
		cpu1 = 0.;
		gd = 0.;
		stpmx = 0.;
		sbgnrm = 0.;
		stp = 0.;
		gdold = 0.;
		dtd = 0.;
		
		/* For operation counts: */
		iter = 0;
		nfgv = 0;
		nseg = 0;
		nintol = 0;
		nskip = 0;
		nfree = n;
		ifun = 0;
		
		/* For stopping tolerance: */
		tol = factr * epsmch;
		
		/* For measuring running time: */
		cachyt = 0.;
		sbtime = 0.;
		lnscht = 0.;
		
		/* 'word' records the status of subspace solutions. */
		*word = WORD_DEFAULT;
		
		/* 'info' records the termination information. */
		info = 0;
		
		itfile = 8;
		
		/* Check the input arguments for errors. */
		errclb(n, m, factr, &l[1], &u[1], &nbd[1], task, &info, &k);
		if (IS_ERROR(*task))
		{
			prn3lb(n, x, (*f), (*task), iprint, info, iter, nfgv, nintol, nskip, nact, sbgnrm, 0., stp, xstep, k, cachyt, sbtime, lnscht);
			return;
		}
		prn1lb(n, m, &l[1], &u[1], x, iprint, epsmch);
		
		/* Initialize iwhere & project x onto the feasible set. */
		active(n, &l[1], &u[1], &nbd[1], x, &iwhere[1], iprint, &prjctd, &cnstnd, &boxed);
		/* The end of the initialization. */
	}
	else
	{
		/* Restore local variables. */
		prjctd = lsave[1];
		cnstnd = lsave[2];
		boxed = lsave[3];
		updatd = lsave[4];
		nintol = isave[1];
		itfile = isave[3];
		iback = isave[4];
		nskip = isave[5];
		head = isave[6];
		col = isave[7];
		itail = isave[8];
		iter = isave[9];
		iupdat = isave[10];
		nseg = isave[12];
		nfgv = isave[13];
		info = isave[14];
		ifun = isave[15];
		iword = isave[16];
		nfree = isave[17];
		nact = isave[18];
		ileave = isave[19];
		nenter = isave[20];
		theta = dsave[1];
		fold = dsave[2];
		tol = dsave[3];
		dnorm = dsave[4];
		epsmch = dsave[5];
		cpu1 = dsave[6];
		cachyt = dsave[7];
		sbtime = dsave[8];
		lnscht = dsave[9];
		time1 = dsave[10];
		gd = dsave[11];
		stpmx = dsave[12];
		sbgnrm = dsave[13];
		stp = dsave[14];
		gdold = dsave[15];
		dtd = dsave[16];
		
		/* After returning from the driver go to the point where execution is to resume. */
		if (*task == FG_LN)
		{
			goto L666;
		}
		if (*task == NEW_X)
		{
			goto L777;
		}
		if (*task == FG_ST)
		{
			goto L111;
		}
		if (IS_STOP(*task))
		{
			if (*task == STOP_CPU)
			{
				/* Restore the previous iterate. */
				dcopy(n, &t[1], 1, x, 1);
				dcopy(n, &r__[1], 1, &g[1], 1);
				*f = fold;
			}
			goto L999;
		}
	}
	
	/* Compute f0 and g0. */
	*task = FG_START;
	
	/* Return to the driver to calculate f and g; reenter at 111. */
	goto L1000;
L111:
	nfgv = 1;
	
	/* Compute the infinity norm of the (-) projected gradient. */
	projgr(n, &l[1], &u[1], &nbd[1], x, &g[1], &sbgnrm);
	
	if (iprint >= 1)
	{
		printf("At iterate %5ld, f(x)= %5.2e, ||proj grad||_infty = %.2e\n", iter, *f, sbgnrm);
	}
	
	if (sbgnrm <= pgtol)
	{
		/* Terminate the algorithm. */
		*task = CONV_GRAD;
		goto L999;
	}
	/* ----------------- the beginning of the loop -------------------------- */
L222:
	if (iprint >= 99)
	{
		printf("ITERATION %5ld\n", i__1);
	}
	iword = -1;

	if (!cnstnd && col > 0)
	{
		/* Skip the search for GCP. */
		dcopy(n, x, 1, &z__[1], 1);
		wrk = updatd;
		nseg = 0;
		goto L333;
	}
	
	/* ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */

	/* Compute the Generalized Cauchy Point (GCP). */

	/* ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */
	
	timer(&cpu1);
	info = 0;
	cauchy(n, x, &l[1], &u[1], &nbd[1], &g[1], &indx2[1], &iwhere[1], &t[1], &d__[1], &z__[1], m, &wy[wy_offset], &ws[ws_offset], &sy[sy_offset], &wt[wt_offset], theta, col, head, &wa[1], &wa[(m << 1) + 1], &wa[(m << 2) + 1], &wa[m * 6 + 1], &nseg, iprint, sbgnrm, &info, epsmch);
	
	if (info != 0)
	{
		/* Singular triangular system detected; refresh the lbfgs memory. */
		info = 0;
		col = 0;
		head = 1;
		theta = 1.;
		iupdat = 0;
		updatd = 0;
		timer(&cpu2);
		cachyt = cachyt + cpu2 - cpu1;
		goto L222;
	}
	timer(&cpu2);
	cachyt = cachyt + cpu2 - cpu1;
	nintol += nseg;
	
	/* Count the entering and leaving variables for iter > 0; */
	/* Find the index set of free and active variables at the GCP. */
	freev(n, &nfree, &index[1], &nenter, &ileave, &indx2[1], &iwhere[1], &wrk, updatd, cnstnd, iprint, iter);
	nact = n - nfree;
L333:
	/* If there are no free variables or B = theta * I, then skip the subspace minimization. */
	if (nfree == 0 || col == 0)
	{
		goto L555;
	}
	
	/* ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */

	/* Subspace minimization. */

	/* ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */
	
	timer(&cpu1);
	
	/* Form  the LEL^T factorization of the indefinite */
	/* matrix	K = [-D -Y'ZZ'Y/theta	 L_a'-R_z'  ] */
	/*	 		    [L_a -R_z		   theta*S'AA'S ] */
	/* where E = [-I  0] */
	/*			 [ 0  I] */
	
	if (wrk)
	{
		formk(n, nfree, &index[1], nenter, ileave, &indx2[1], iupdat, updatd, &wn[wn_offset], &snd[snd_offset], m, &ws[ws_offset], &wy[wy_offset], &sy[sy_offset], theta, col, head, &info);
	}
	
	if (info != 0)
	{
		/* Nonpositive definiteness in Cholesky factorization; */
		/* Refresh the lbfgs memory and restart the iteration. */
		if (iprint >= 1)
		{
			printf(" Nonpositive definiteness in Cholesky factorization in formk;\n");
			printf("   refresh the lbfgs memory and restart the iteration.\n");
		}
		info = 0;
		col = 0;
		head = 1;
		theta = 1.;
		iupdat = 0;
		updatd = 0;
		timer(&cpu2);
		sbtime = sbtime + cpu2 - cpu1;
		goto L222;
	}
	
	/* Compute r = -Z'B(xcp - xk) - Z'g (using wa(2m + 1) = W'(xcp - x) from 'cauchy'). */
	cmprlb(n, m, x, &g[1], &ws[ws_offset], &wy[wy_offset], &sy[sy_offset], &wt[wt_offset], &z__[1], &r__[1], &wa[1], &index[1], theta, col, &head, nfree, cnstnd, &info);
	
	if (info != 0)
	{
		goto L444;
	}
	
	/* Call the direct method. */
	subsm(n, m, nfree, &index[1], &l[1], &u[1], &nbd[1], &z__[1], &r__[1], &xp[1], &ws[ws_offset], &wy[wy_offset], theta, x, &g[1], col, head, &iword, &wa[1], &wn[wn_offset], iprint, &info);
L444:
	if (info != 0)
	{
		/* Singular triangular system detected; */
		/* Refresh the lbfgs memory and restart the iteration. */
		if (iprint >= 1)
		{
			printf(" Singular triangular system detected;\n");
			printf("   refresh the lbfgs memory and restart the iteration.\n");
		}
		info = 0;
		col = 0;
		head = 1;
		theta = 1.;
		iupdat = 0;
		updatd = 0;
		timer(&cpu2);
		sbtime = sbtime + cpu2 - cpu1;
		goto L222;
	}
	
	timer(&cpu2);
	sbtime = sbtime + cpu2 - cpu1;
L555:
	/* ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */

	/* Line search and optimality tests. */

	/* ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */
	
	/* Generate the search direction d := z - x. */
	for (i = 0; i < n; ++i)
	{
		d__[i + 1] = z__[i + 1] - x[i];
	}
	timer(&cpu1);
L666:
	lnsrlb(n, &l[1], &u[1], &nbd[1], x, f, &fold, &gd, &gdold, &g[1], &d__[1], &r__[1], &t[1], &z__[1], &stp, &dnorm, &dtd, &xstep, &stpmx, iter, &ifun, &iback, &nfgv, &info, task, boxed, cnstnd, csave, &isave[22], &dsave[17]);
	
	if (info != 0 || iback >= 20)
	{
		/* Restore the previous iterate. */
		dcopy(n, &t[1], 1, x, 1);
		dcopy(n, &r__[1], 1, &g[1], 1);
		*f = fold;
		if (col == 0)
		{
			/* Abnormal termination. */
			if (info == 0)
			{
				info = -9;
				/* Restore the actual number of f and g evaluations etc. */
				--nfgv;
				--ifun;
				--iback;
			}
			*task = ABNORMAL;
			++iter;
			goto L999;
		}
		else
		{
			/* Refresh the lbfgs memory and restart the iteration. */
			if (iprint >= 1)
			{
				printf(" Bad direction in the line search;\n");
				printf("   refresh the lbfgs memory and restart the iteration.\n");
			}
			if (info == 0)
			{
				--nfgv;
			}
			info = 0;
			col = 0;
			head = 1;
			theta = 1.;
			iupdat = 0;
			updatd = 0;
			*task = RESTART;
			timer(&cpu2);
			lnscht = lnscht + cpu2 - cpu1;
			goto L222;
		}
	}
	else if (*task == FG_LN)
	{
		/* Return to the driver for calculating f and g; reenter at 666. */
		goto L1000;
	}
	else
	{
		/* Calculate and print out the quantities related to the new X. */
		timer(&cpu2);
		lnscht = lnscht + cpu2 - cpu1;
		++iter;
		
		/* Compute the infinity norm of the projected (-)gradient. */
		projgr(n, &l[1], &u[1], &nbd[1], x, &g[1], &sbgnrm);
		
		/* Print iteration information. */
		prn2lb(n, x, (*f), &g[1], iprint, iter, sbgnrm, word, iword, iback, xstep);
		goto L1000;
	}
L777:
	/* Test for termination. */
	if (sbgnrm <= pgtol)
	{
		/* Terminate the algorithm. */
		*task = CONV_GRAD;
		goto L999;
	}
	
	ddum = fmax(fmax(fabs(fold), fabs(*f)), 1.);
	
	if (fold - *f <= tol * ddum)
	{
		/* Terminate the algorithm. */
		*task = CONV_F;
		if (iback >= 10)
		{
			info = -5;
		}
		/* i.e., to issue a warning if iback > 10 in the line search. */
		goto L999;
	}
	
	/* Compute d = newx - oldx, r = newg - oldg, rr = y'y and dr = y's. */
	for (i = 0; i < n; ++i)
	{
		r__[i + 1] = g[i + 1] - r__[i + 1];
	}
	
	rr = ddot(n, &r__[1], 1, &r__[1], 1);
	
	if (stp == 1.)
	{
		dr = gd - gdold;
		ddum = -gdold;
	}
	else
	{
		dr = (gd - gdold) * stp;
		dscal(n, stp, &d__[1], 1);
		ddum = -gdold * stp;
	}
	
	if (dr <= epsmch * ddum)
	{
		/* Skip the L-BFGS update. */
		++nskip;
		updatd = 0;
		
		if (iprint >= 1)
		{
			printf("  ys=%10.3e  -gs=%10.3e BFGS update SKIPPED\n", dr, ddum);
		}
		goto L888;
	}
	
	/* ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */

	/* Update the L-BFGS matrix. */

	/* ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */
	
	updatd = 1;
	++iupdat;
	
	/* Update matrices WS and WY and form the middle matrix in B. */
	matupd(n, m, &ws[ws_offset], &wy[wy_offset], &sy[sy_offset], &ss[ss_offset], &d__[1], &r__[1], &itail, iupdat, &col, &head, &theta, rr, dr, stp, dtd);
	
	/* Form the upper half of the pds T = theta*SS + L*D^(-1)*L'; */
	/* Store T in the upper triangular of the array wt; */
	/* Cholesky factorize T to J*J' with J' stored in the upper triangular of wt. */
	formt(m, &wt[wt_offset], &sy[sy_offset], &ss[ss_offset], col, theta, &info);
	
	if (info != 0)
	{
		/* Nonpositive definiteness in Cholesky factorization; */
		/* Refresh the lbfgs memory and restart the iteration. */
		if (iprint >= 1)
		{
			printf(" Nonpositive definiteness in Cholesky factorization in formt;\n");
			printf("   refresh the lbfgs memory and restart the iteration.\n");
		}
		
		info = 0;
		col = 0;
		head = 1;
		theta = 1.;
		iupdat = 0;
		updatd = 0;
		goto L222;
	}
	
	/* Now the inverse of the middle matrix in B is */
	/* [D^(1/2)	        O] [-D^(1/2)  D^(-1/2)*L'] */
	/* [-L * D^(-1/2)   J] [0		           J'] */
L888:
	/* -------------------- the end of the loop ----------------------------- */
	goto L222;
L999:
	timer(&time2);
	time = time2 - time1;
	prn3lb(n, x, (*f), (*task), iprint, info, iter, nfgv, nintol, nskip, nact, sbgnrm, time, stp, xstep, k, cachyt, sbtime, lnscht);
L1000:
	/* Save local variables. */
	lsave[1] = prjctd;
	lsave[2] = cnstnd;
	lsave[3] = boxed;
	lsave[4] = updatd;
	isave[1] = nintol;
	isave[3] = itfile;
	isave[4] = iback;
	isave[5] = nskip;
	isave[6] = head;
	isave[7] = col;
	isave[8] = itail;
	isave[9] = iter;
	isave[10] = iupdat;
	isave[12] = nseg;
	isave[13] = nfgv;
	isave[14] = info;
	isave[15] = ifun;
	isave[16] = iword;
	isave[17] = nfree;
	isave[18] = nact;
	isave[19] = ileave;
	isave[20] = nenter;
	dsave[1] = theta;
	dsave[2] = fold;
	dsave[3] = tol;
	dsave[4] = dnorm;
	dsave[5] = epsmch;
	dsave[6] = cpu1;
	dsave[7] = cachyt;
	dsave[8] = sbtime;
	dsave[9] = lnscht;
	dsave[10] = time1;
	dsave[11] = gd;
	dsave[12] = stpmx;
	dsave[13] = sbgnrm;
	dsave[14] = stp;
	dsave[15] = gdold;
	dsave[16] = dtd;
	
	return;
} // end of mainlb function

void active(long int n, double* l, double* u, long int* nbd, double* x, long int* iwhere, long int iprint, long int* prjctd, long int* cnstnd, long int* boxed)
{
    /*
	************ 

	Subroutine active

	This subroutine initializes iwhere and projects the initial x to
	  the feasible set if necessary.

	iwhere is an integer array of dimension n.
	  On entry iwhere is unspecified.
	  On exit iwhere(i)=-1  if x(i) has no bounds
	                    3   if l(i)=u(i)
	                    0   otherwise.
	  In cauchy, iwhere is given finer gradations.


	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	************ 
	*/
	
	/* Local variables */
	static long int i, nbdd;

	/* Initialize nbdd, prjctd, cnstnd and boxed. */
	nbdd = 0;
	*prjctd = 0;
	*cnstnd = 0;
	*boxed = 1;
	
	/* Project the initial x to the feasible set if necessary. */
	for (i = 0; i < n; ++i)
	{
		if (nbd[i] > 0)
		{
			if (nbd[i] <= 2 && x[i] <= l[i])
			{
				if (x[i] < l[i])
				{
					*prjctd = 1;
					x[i] = l[i];
				}
				++nbdd;
			}
			else if (nbd[i] >= 2 && x[i] >= u[i])
			{
				if (x[i] > u[i])
				{
					*prjctd = 1;
					x[i] = u[i];
				}
				++nbdd;
			}
		}
	}
	
	/* Initialize iwhere and assign values to cnstnd and boxed. */
	for (i = 0; i < n; ++i)
	{
		if (nbd[i] != 2)
		{
			*boxed = 0;
		}
		
		if (nbd[i] == 0)
		{
			/* This variable is always free */
			iwhere[i] = -1;
			/* Otherwise set x(i) = mid(x(i), u(i), l(i)). */
		}
		else
		{
			*cnstnd = 1;
			if (nbd[i] == 2 && u[i] - l[i] <= 0.)
			{
				/* This variable is always fixed */
				iwhere[i] = 3;
			}
			else
			{
				iwhere[i] = 0;
			}
		}
	}
	
	if (iprint >= 0)
	{
		if (*prjctd)
		{
			printf("The initial X is infeasible. Restart with its projection\n");
		}
		
		if (!(*cnstnd))
		{
			printf("This problem is unconstrained\n");
		}
	}
	
	if (iprint > 0)
	{
		printf("At X0, %ld variables are exactly at the bounds\n", nbdd);
	}
	
	return;
} // end of active function
/* ======================= The end of active ============================= */

void bmv(long int m, double* sy, double* wt, long int col, double* v, double* p, long int* info)
{
	/*
	************

	Subroutine bmv

	This subroutine computes the product of the 2m x 2m middle matrix 
	  in the compact L-BFGS formula of B and a 2m vector v;  
	  it returns the product in p.
	  
	m is an integer variable.
	  On entry m is the maximum number of variable metric corrections
	    used to define the limited memory matrix.
	  On exit m is unchanged.

	sy is a double precision array of dimension m x m.
	  On entry sy specifies the matrix S'Y.
	  On exit sy is unchanged.

	wt is a double precision array of dimension m x m.
	  On entry wt specifies the upper triangular matrix J' which is 
	    the Cholesky factor of (thetaS'S+LD^(-1)L').
	  On exit wt is unchanged.

	col is an integer variable.
	  On entry col specifies the number of s-vectors (or y-vectors)
	    stored in the compact L-BFGS formula.
	  On exit col is unchanged.

	v is a double precision array of dimension 2col.
	  On entry v specifies vector v.
	  On exit v is unchanged.

	p is a double precision array of dimension 2col.
	  On entry p is unspecified.
	  On exit p is the product Mv.

	info is an integer variable.
	  On entry info is unspecified.
	  On exit info = 0       for normal return,
	               = nonzero for abnormal return when the system
	                           to be solved by dtrsl is singular.

	Subprograms called:

	  Linpack ... dtrsl.


	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	************
	*/
	
	/* System generated locals */
	long int sy_dim1, sy_offset, wt_dim1, wt_offset;

	/* Local variables */
	static long int i, k, i2;
	static double sum;

	/* Parameter adjustments */
	wt_dim1 = m;
	wt_offset = 1 + wt_dim1;
	wt -= wt_offset;
	sy_dim1 = m;
	sy_offset = 1 + sy_dim1;
	sy -= sy_offset;
	--p;
	--v;

	if (col == 0)
	{
		return;
	}
	
	/* PART I: solve [  D^(1/2)	       O ] [ p1 ] = [ v1 ] */
	/*			     [ -L * D^(-1/2)   J ] [ p2 ]   [ v2 ]. */
	/* solve Jp2 = v2 + LD^(-1)v1. */
	p[col + 1] = v[col + 1];
	for (i = 1; i < col; ++i)
	{
		i2 = col + i + 1;
		sum = 0.;
		for (k = 0; k < i; ++k)
		{
			sum += sy[i + 1 + (k + 1) * sy_dim1] * v[k + 1] / sy[k + 1 + (k + 1) * sy_dim1];
		}
		p[i2] = v[i2] + sum;
	}
	
	/* Solve the triangular system */
	dtrsl(&wt[wt_offset], m, col, &p[col + 1], 11, info);
	if (*info != 0)
	{
		return;
	}
	
	/* Solve D^(1/2)p1 = v1. */
	for (i = 0; i < col; ++i)
	{
		p[i + 1] = v[i + 1] / sqrt(sy[i + 1 + (i + 1) * sy_dim1]);
	}
	
	/* PART II: solve [ -D^(1/2)   D^(-1/2)*L'  ] [ p1 ] = [ p1 ] */
	/*                [  0		   J'		    ] [ p2 ]   [ p2 ]. */
	/* Solve J^Tp2 = p2. */
	dtrsl(&wt[wt_offset], m, col, &p[col + 1], 1, info);
	if (*info != 0)
	{
		return;
	}
	
	/* Compute p1 = -D^(-1/2)(p1 - D^(-1/2)L'p2) */
	/*            = -D^(-1/2)p1 + D^(-1)L'p2. */
	for (i = 0; i < col; ++i)
	{
		p[i + 1] = -p[i + 1] / sqrt(sy[i + 1 + (i + 1) * sy_dim1]);
	}
	
	for (i = 0; i < col; ++i)
	{
		sum = 0.;
		for (k = i + 1; k < col; ++k)
		{
			sum += sy[k + 1 + (i + 1) * sy_dim1] * p[col + k + 1] / sy[i + 1 + (i + 1) * sy_dim1];
		}
		p[i + 1] += sum;
	}
	
	return;
} // end of bmv function
/* ======================== The end of bmv =============================== */

void cauchy(long int n, double* x, double* l, double* u, long int* nbd, double* g, long int* iorder, long int* iwhere, double* t, double* d__, double* xcp, long int m, double* wy, double* ws, double* sy, double* wt, double theta, long int col, long int head, double* p, double* c__, double* wbp, double* v, long int* nseg, long int iprint, double sbgnrm, long int* info, double epsmch)
{
	/*
	************

	Subroutine cauchy

	For given x, l, u, g (with sbgnrm > 0), and a limited memory
	  BFGS matrix B defined in terms of matrices WY, WS, WT, and
	  scalars head, col, and theta, this subroutine computes the
	  generalized Cauchy point (GCP), defined as the first local
	  minimizer of the quadratic

	             Q(x + s) = g's + 1/2 s'Bs

	  along the projected gradient direction P(x-tg,l,u).
	  The routine returns the GCP in xcp. 
	  
	n is an integer variable.
	  On entry n is the dimension of the problem.
	  On exit n is unchanged.

	x is a double precision array of dimension n.
	  On entry x is the starting point for the GCP computation.
	  On exit x is unchanged.

	l is a double precision array of dimension n.
	  On entry l is the lower bound of x.
	  On exit l is unchanged.

	u is a double precision array of dimension n.
	  On entry u is the upper bound of x.
	  On exit u is unchanged.

	nbd is an integer array of dimension n.
	  On entry nbd represents the type of bounds imposed on the
	    variables, and must be specified as follows:
	    nbd(i)=0 if x(i) is unbounded,
	           1 if x(i) has only a lower bound,
	           2 if x(i) has both lower and upper bounds, and
	           3 if x(i) has only an upper bound. 
	  On exit nbd is unchanged.

	g is a double precision array of dimension n.
	  On entry g is the gradient of f(x).  g must be a nonzero vector.
	  On exit g is unchanged.

	iorder is an integer working array of dimension n.
	  iorder will be used to store the breakpoints in the piecewise
	  linear path and free variables encountered. On exit,
	    iorder(1),...,iorder(nleft) are indices of breakpoints
	                           which have not been encountered; 
	    iorder(nleft+1),...,iorder(nbreak) are indices of
	                                encountered breakpoints; and
	    iorder(nfree),...,iordern are indices of variables which
	            have no bound constraits along the search direction.

	iwhere is an integer array of dimension n.
	  On entry iwhere indicates only the permanently fixed (iwhere=3)
	  or free (iwhere= -1) components of x.
	  On exit iwhere records the status of the current x variables.
	  iwhere(i)=-3  if x(i) is free and has bounds, but is not moved
	            0   if x(i) is free and has bounds, and is moved
	            1   if x(i) is fixed at l(i), and l(i) .ne. u(i)
	            2   if x(i) is fixed at u(i), and u(i) .ne. l(i)
	            3   if x(i) is always fixed, i.e.,  u(i)=x(i)=l(i)
	            -1  if x(i) is always free, i.e., it has no bounds.

	t is a double precision working array of dimension n. 
	  t will be used to store the break points.

	d is a double precision array of dimension n used to store
	  the Cauchy direction P(x-tg)-x.

	xcp is a double precision array of dimension n used to return the
	  GCP on exit.

	m is an integer variable.
	  On entry m is the maximum number of variable metric corrections 
	    used to define the limited memory matrix.
	  On exit m is unchanged.

	ws, wy, sy, and wt are double precision arrays.
	  On entry they store information that defines the
	                        limited memory BFGS matrix:
	    ws(n,m) stores S, a set of s-vectors;
	    wy(n,m) stores Y, a set of y-vectors;
	    sy(m,m) stores S'Y;
	    wt(m,m) stores the
	            Cholesky factorization of (theta*S'S+LD^(-1)L').
	  On exit these arrays are unchanged.

	theta is a double precision variable.
	  On entry theta is the scaling factor specifying B_0 = theta I.
	  On exit theta is unchanged.

	col is an integer variable.
	  On entry col is the actual number of variable metric
	    corrections stored so far.
	  On exit col is unchanged.

	head is an integer variable.
	  On entry head is the location of the first s-vector (or y-vector)
	    in S (or Y).
	  On exit col is unchanged.

	p is a double precision working array of dimension 2m.
	  p will be used to store the vector p = W^(T)d.

	c is a double precision working array of dimension 2m.
	  c will be used to store the vector c = W^(T)(xcp-x).

	wbp is a double precision working array of dimension 2m.
	  wbp will be used to store the row of W corresponding
	    to a breakpoint.

	v is a double precision working array of dimension 2m.

	nseg is an integer variable.
	  On exit nseg records the number of quadratic segments explored
	    in searching for the GCP.

	sg and yg are double precision arrays of dimension m.
	  On entry sg  and yg store S'g and Y'g correspondingly.
	  On exit they are unchanged. 
 
	iprint is an INTEGER variable that must be set by the user.
	  It controls the frequency and type of output generated:
	   iprint<0    no output is generated;
	   iprint=0    print only one line at the last iteration;
	   0<iprint<99 print also f and |proj g| every iprint iterations;
	   iprint=99   print details of every iteration except n-vectors;
	   iprint=100  print also the changes of active set and final x;
	   iprint>100  print details of every iteration including x and g;
	  When iprint > 0, the file iterate.dat will be created to
	                   summarize the iteration.

	sbgnrm is a double precision variable.
	  On entry sbgnrm is the norm of the projected gradient at x.
	  On exit sbgnrm is unchanged.

	info is an integer variable.
	  On entry info is 0.
	  On exit info = 0       for normal return,
	               = nonzero for abnormal return when the the system
	                         used in routine bmv is singular.

	Subprograms called:
 
	  L-BFGS-B Library ... hpsolb, bmv.

	  Linpack ... dscal dcopy, daxpy.


	References:

	  [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
	  memory algorithm for bound constrained optimization'',
	  SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

	  [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
	  Subroutines for Large Scale Bound Constrained Optimization''
	  Tech. Report, NAM-11, EECS Department, Northwestern University,
	  1994.

	  (Postscript files of these papers are available via anonymous
	   ftp to eecs.nwu.edu in the directory pub/lbfgs/lbfgs_bcm.)

	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	************
	*/
	
	/* System generated locals */
	long int wy_dim1, wy_offset, ws_dim1, ws_offset, sy_dim1, sy_offset, wt_dim1, wt_offset;

	/* Local variables */
	static long int i, j;
	static double f1, f2, dt, tj, tl, tu, tj0;
	static long int ibp;
	static double dtm;
	static double wmc, wmp, wmw;
	static long int col2;
	static double dibp;
	static long int iter;
	static double zibp, tsum, dibp2;
	static long int bnded;
	static double neggi;
	static long int nfree;
	static double bkmin;
	static long int nleft;
	static double f2_org__;
	static long int nbreak, ibkmin;
	static long int pointr;
	static long int xlower, xupper;

	/* Parameter adjustments */
	wt_dim1 = m;
	wt_offset = 1 + wt_dim1;
	wt -= wt_offset;
	sy_dim1 = m;
	sy_offset = 1 + sy_dim1;
	sy -= sy_offset;
	ws_dim1 = n;
	ws_offset = 1 + ws_dim1;
	ws -= ws_offset;
	wy_dim1 = n;
	wy_offset = 1 + wy_dim1;
	wy -= wy_offset;

	/* Check the status of the variables, reset iwhere[i] if necessary;
	   Compute the Cauchy direction d and the breakpoints t; initialize
	     the derivative f1 and the vector p = W'd (for theta = 1). */
	if (sbgnrm <= 0.)
	{
		if (iprint >= 0)
		{
			printf("Subnorm = 0. GCP = X.\n");
		}
		
		dcopy(n, x, 1, xcp, 1);
		
		return;
	}
	bnded = 1;
	nfree = n + 1;
	nbreak = 0;
	ibkmin = 0;
	bkmin = 0.;
	col2 = col << 1;
	f1 = 0.;
	
	if (iprint >= 99)
	{
		printf("CAUCHY entered\n");
	}

	/* We set p to zero and build it up as we determine d. */
	for (i = 0; i < col2; ++i)
	{
		p[i] = 0.;
	}
	
	/* In the following loop we determine for each variable its bound
	     status and its breakpoint, and update p accordingly.
	     Smallest breakpoint is identified. */
	for (i = 0; i < n; ++i)
	{
		neggi = -g[i];
		if (iwhere[i] != 3 && iwhere[i] != -1)
		{
			/* If x[i] is not a constant and has bounds, compute the difference between x[i] and its bounds. */
			if (nbd[i] <= 2)
			{
				tl = x[i] - l[i];
			}
			
			if (nbd[i] >= 2)
			{
				tu = u[i] - x[i];
			}
			
			/* If a variable is close enough to a bound we treat it as at bound. */
			xlower = nbd[i] <= 2 && tl <= 0.;
			xupper = nbd[i] >= 2 && tu <= 0.;
			
			/* Reset iwhere[i]. */
			iwhere[i] = 0;
			if (xlower)
			{
				if (neggi <= 0.)
				{
					iwhere[i] = 1;
				}
			}
			else if (xupper)
			{
				if (neggi >= 0.)
				{
					iwhere[i] = 2;
				}
			}
			else
			{
				if (fabs(neggi) <= 0.)
				{
					iwhere[i] = -3;
				}
			}
		}
		
		pointr = head;
		if (iwhere[i] != 0 && iwhere[i] != -1)
		{
			d__[i] = 0.;
		}
		else
		{
			d__[i] = neggi;
			f1 -= neggi * neggi;
			/* Calculate p := p - W'e_i * (g_i). */
			for (j = 0; j < col; ++j)
			{
				p[j] += wy[i + 1 + pointr * wy_dim1] * neggi;
				p[col + j] += ws[i + 1 + pointr * ws_dim1] * neggi;
				pointr = pointr % m + 1;
			}
			
			if (nbd[i] <= 2 && nbd[i] != 0 && neggi < 0.)
			{
				/* x[i] + d[i] is bounded; compute t[i]. */
				++nbreak;
				iorder[nbreak - 1] = i;
				t[nbreak - 1] = tl / (-neggi);
				if (nbreak == 1 || t[nbreak - 1] < bkmin)
				{
					bkmin = t[nbreak - 1];
					ibkmin = nbreak;
				}
			}
			else if (nbd[i] >= 2 && neggi > 0.)
			{
				/* x[i] + d[i] is bounded; compute t[i]. */
				++nbreak;
				iorder[nbreak - 1] = i;
				t[nbreak - 1] = tu / neggi;
				if (nbreak == 1 || t[nbreak - 1] < bkmin)
				{
					bkmin = t[nbreak - 1];
					ibkmin = nbreak;
				}
			}
			else
			{
				/* x[i] + d[i] is not bounded. */
				--nfree;
				iorder[nfree - 1] = i;
				
				if (fabs(neggi) > 0.)
				{
					bnded = 0;
				}
			}
		}
	}
	
	/* The indices of the nonzero components of d are now stored
	     in iorder(1),...,iorder(nbreak) and iorder(nfree),...,iordern.
	     The smallest of the nbreak breakpoints is in t(ibkmin)=bkmin. */
	if (theta != 1.)
	{
		/* Complete the initialization of p for theta != 1. */
		dscal(col, theta, &p[col], 1);
	}
	
	/* Initialize GCP xcp = x. */
	dcopy(n, x, 1, xcp, 1);
	
	if (nbreak == 0 && nfree == n + 1)
	{
		/* Is a zero vector, return with the initial xcp as GCP. */
		if (iprint > 100)
		{
			printf("Cauchy X = ");
			for (i = 0; i < n; ++i)
			{
				printf("%5.2e ", xcp[i] );
			}
			printf("\n");
		}
		return;
	}
	
	/* Initialize c = W'(xcp - x) = 0. */
	for (j = 0; j < col2; ++j)
	{
		c__[j] = 0.;
	}
	
	/* Initialize derivative f2. */
	f2 = -theta * f1;
	f2_org__ = f2;
	if (col > 0)
	{
		bmv(m, &sy[sy_offset], &wt[wt_offset], col, p, v, info);
		if (*info != 0)
		{
			return;
		}
		f2 -= ddot(col2, v, 1, p, 1);
	}
	dtm = -f1 / f2;
	tsum = 0.;
	*nseg = 1;
	
	if (iprint >= 99)
	{
		printf("There are %ld breakpoints\n", nbreak);
	}
	
	/* If there are no breakpoints, locate the GCP and return. */
	if (nbreak == 0)
	{
		goto L888;
	}
	
	nleft = nbreak;
	iter = 1;
	tj = 0.;
	
	/* ------------------- the beginning of the loop ------------------------- */
L777:
	/* Find the next smallest breakpoint; */
	/* Compute dt = t(nleft) - t(nleft + 1). */
	tj0 = tj;
	
	if (iter == 1)
	{
		/* Since we already have the smallest breakpoint we need not do */
		/* heapsort yet. Often only one breakpoint is used and the */
		/* cost of heapsort is avoided. */
		tj = bkmin;
		ibp = iorder[ibkmin - 1];
	}
	else
	{
		if (iter == 2)
		{
			/* Replace the already used smallest breakpoint with the */
			/* breakpoint numbered nbreak > nlast, before heapsort call. */
			if (ibkmin != nbreak)
			{
				t[ibkmin - 1] = t[nbreak - 1];
				iorder[ibkmin - 1] = iorder[nbreak - 1];
			}
			/* Update heap structure of breakpoints */
			/* (if iter = 2, initialize heap). */
		}
		hpsolb(nleft, t, iorder, iter - 2);
		tj = t[nleft - 1];
		ibp = iorder[nleft - 1];
	}
	
	dt = tj - tj0;
	
	if (dt != 0. && iprint >= 100)
	{
		printf("Piece %ld --f1, f2 at start point %.2e %.2e\n", *nseg, f1, f2);
		printf("Distance to the next break point = %.2e\n", dt);
		printf("Distance to the stationary point = %.2e\n", dtm);
	}
	
	/* If a minimizer is within this interval, locate the GCP and return. */
	if (dtm < dt)
	{
		goto L888;
	}
	
	/* Otherwise fix one variable and reset the corresponding component of d to zero. */
	tsum += dt;
	--nleft;
	++iter;
	dibp = d__[ibp];
	d__[ibp] = 0.;
	
	if (dibp > 0.)
	{
		zibp = u[ibp] - x[ibp];
		xcp[ibp] = u[ibp];
		iwhere[ibp] = 2;
	}
	else
	{
		zibp = l[ibp] - x[ibp];
		xcp[ibp] = l[ibp];
		iwhere[ibp] = 1;
	}
	
	if (iprint >= 100)
	{
		printf("Variable %ld is fixed\n", ibp);
	}
	
	if (nleft == 0 && nbreak == n)
	{
		/* All n variables are fixed, return with xcp as GCP. */
		dtm = dt;
		goto L999;
	}
	
	/* Update the derivative information. */
	++(*nseg);
	dibp2 = dibp * dibp;
	
	/* Update f1 and f2. */
	
	/* Temporarily set f1 and f2 for col = 0. */
	f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp;
	f2 -= theta * dibp2;
	
	if (col > 0)
	{
		/* Update c = c + dt*p. */
		daxpy(col2, dt, p, 1, c__, 1);
		
		/* Choose wbp, the row of W corresponding to the breakpoint encountered. */
		pointr = head;
		for (j = 0; j < col; ++j)
		{
			wbp[j] = wy[ibp + 1 + pointr * wy_dim1];
			wbp[col + j] = theta * ws[ibp + 1 + pointr * ws_dim1];
			pointr = pointr % m + 1;
		}
		
		/* Compute (wbp)Mc, (wbp)Mp, and (wbp)M(wbp)'. */
		bmv(m, &sy[sy_offset], &wt[wt_offset], col, wbp, v, info);
		
		if (*info != 0)
		{
			return;
		}
		
		wmc = ddot(col2, c__, 1, v, 1);
		wmp = ddot(col2, p, 1, v, 1);
		wmw = ddot(col2, wbp, 1, v, 1);
		
		/* Update p = p - dibp * wbp. */
		daxpy(col2, -dibp, wbp, 1, p, 1);
		
		/* Complete updating f1 and f2 while col > 0. */
		f1 += dibp * wmc;
		f2 = f2 + dibp * 2. * wmp - dibp2 * wmw;
	}
	
	/* Computing MAX */
	f2 = fmax(epsmch * f2_org__, f2);
	if (nleft > 0)
	{
		dtm = -f1 / f2;
		goto L777;
		/* To repeat the loop for unsearched intervals. */
	}
	else if (bnded)
	{
		f1 = 0.;
		f2 = 0.;
		dtm = 0.;
	}
	else
	{
		dtm = -f1 / f2;
	}
	/* ------------------- the end of the loop ------------------------------- */
L888:
	if (iprint >= 99)
	{
		printf("\nGCP found in this segment. Piece %ld --f1, f2 at start point %.2e %.2e\n", *nseg, f1, f2);
		printf("Distance to the stationary point = %.2e\n", dtm);
	}
	
	if (dtm <= 0.)
	{
		dtm = 0.;
	}
	tsum += dtm;
	
	/* Move free variables (i.e., the ones w/o breakpoints) and */
	/* the variables whose breakpoints haven't been reached. */
	daxpy(n, tsum, d__, 1, xcp, 1);
L999:
	/* Update c = c + dtm*p = W'(x^c - x) */
	/* which will be used in computing r = Z'(B(x^c - x) + g). */
	if (col > 0)
	{
		daxpy(col2, dtm, p, 1, c__, 1);
	}
	
	if (iprint > 100)
	{

		printf("Cauchy X = ");
		for (i = 0; i < n; ++i)
		{
			printf("%5.2e ", xcp[i]);
		}
	}
	
	if (iprint >= 99)
	{
		printf("-------------- exit CAUCHY -----------\n");
	}
	
	return;
} // end of cauchy function
/* ====================== The end of cauchy ============================== */

void cmprlb(long int n, long int m, double* x, double* g, double* ws, double* wy, double* sy, double* wt, double* z__, double* r__, double* wa, long int* index, double theta, long int col, long int* head, long int nfree, long int cnstnd, long int* info)
{
	/*
	************

	Subroutine cmprlb 

	  This subroutine computes r=-Z'B(xcp-xk)-Z'g by using 
	    wa(2m+1)=W'(xcp-x) from subroutine cauchy.

	Subprograms called:

	  L-BFGS-B Library ... bmv.


	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	************
	*/
	
	/* System generated locals */
	long int ws_dim1, ws_offset, wy_dim1, wy_offset, sy_dim1, sy_offset, wt_dim1, wt_offset;

	/* Local variables */
	static long int i, j, k;
	static double a1, a2;
	static long int pointr;

	/* Parameter adjustments */
	wt_dim1 = m;
	wt_offset = 1 + wt_dim1;
	wt -= wt_offset;
	sy_dim1 = m;
	sy_offset = 1 + sy_dim1;
	sy -= sy_offset;
	wy_dim1 = n;
	wy_offset = 1 + wy_dim1;
	wy -= wy_offset;
	ws_dim1 = n;
	ws_offset = 1 + ws_dim1;
	ws -= ws_offset;

	if (!cnstnd && col > 0)
	{
		for (i = 0; i < n; ++i)
		{
			r__[i] = -g[i];
		}
	}
	else
	{
		for (i = 0; i < nfree; ++i)
		{
			k = index[i];
			r__[i] = -theta * (z__[k - 1] - x[k - 1]) - g[k - 1];
		}

		bmv(m, &sy[sy_offset], &wt[wt_offset], col, &wa[(m << 1)], wa, info);
		
		if (*info != 0)
		{
			*info = -8;
			return;
		}
		
		pointr = *head;
		for (j = 0; j < col; ++j)
		{
			a1 = wa[j];
			a2 = theta * wa[col + j];
			for (i = 0; i < nfree; ++i)
			{
				k = index[i];
				r__[i] += wy[k + pointr * wy_dim1] * a1 + ws[k + pointr * ws_dim1] * a2;
			}
			pointr = pointr % m + 1;
		}
	}
	
	return;
} // end of cmprlb function
/* ======================= The end of cmprlb ============================= */

void errclb(long int n, long int m, double factr, double *l, double *u, long int* nbd, long int* task, long int* info, long int* k)
{
	/*
	************ 

	Subroutine errclb 

	This subroutine checks the validity of the input data. 


						  *  *  * 

	NEOS, November 1994. (Latest revision June 1996.) 
	Optimization Technology Center. 
	Argonne National Laboratory and Northwestern University. 
	Written by 
					   Ciyou Zhu 
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal. 


	************ 
	*/
	
    /* Local variables */
    static long int i;

	/* Check the input arguments for errors. */
    if (n <= 0)
	{
		*task = ERROR_N0;
	}
	
    if (m <= 0)
	{
		*task = ERROR_M0;
	}
	
    if (factr < 0.)
	{
		*task = ERROR_FACTR;
	}
	
    /* Check the validity of the arrays nbd(i), u(i), and l(i). */
    for (i = 0; i < n; ++i)
	{
        if (nbd[i] < 0 || nbd[i] > 3)
		{
            /* Return */
            *task = ERROR_NBD;
            *info = -6;
            *k = i;
        }
		
        if (nbd[i] == 2)
		{
            if (l[i] > u[i])
			{
                /* Return */
                *task = ERROR_FEAS;
                *info = -7;
                *k = i;
            }
        }
    }
	
    return;
} // end of errclb function */
/* ======================= The end of errclb ============================= */

void formk(long int n, long int nsub, long int* ind, long int nenter, long int ileave, long int* indx2, long int iupdat, long int updatd, double* wn, double* wn1, long int m, double* ws, double* wy, double* sy, double theta, long int col, long int head, long int* info)
{
	/*
	************

	Subroutine formk 

	This subroutine forms  the LEL^T factorization of the indefinite

	  matrix    K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
	                [L_a -R_z           theta*S'AA'S ]
	                                               where E = [-I  0]
	                                                         [ 0  I]
	The matrix K can be shown to be equal to the matrix M^[-1]N
	  occurring in section 5.1 of [1], as well as to the matrix
	  Mbar^[-1] Nbar in section 5.3.

	n is an integer variable.
	  On entry n is the dimension of the problem.
	  On exit n is unchanged.

	nsub is an integer variable
	  On entry nsub is the number of subspace variables in free set.
	  On exit nsub is not changed.

	ind is an integer array of dimension nsub.
	  On entry ind specifies the indices of subspace variables.
	  On exit ind is unchanged. 

	nenter is an integer variable.
	  On entry nenter is the number of variables entering the 
	    free set.
	  On exit nenter is unchanged. 

	ileave is an integer variable.
	  On entry indx2(ileave),...,indx2n are the variables leaving
	    the free set.
	  On exit ileave is unchanged. 

	indx2 is an integer array of dimension n.
	  On entry indx2(1),...,indx2(nenter) are the variables entering
	    the free set, while indx2(ileave),...,indx2n are the
	    variables leaving the free set.
	  On exit indx2 is unchanged. 

	iupdat is an integer variable.
	  On entry iupdat is the total number of BFGS updates made so far.
	  On exit iupdat is unchanged. 

	updatd is a logical variable.
	  On entry 'updatd' is true if the L-BFGS matrix is updatd.
	  On exit 'updatd' is unchanged. 

	wn is a double precision array of dimension 2m x 2m.
	  On entry wn is unspecified.
	  On exit the upper triangle of wn stores the LEL^T factorization
	    of the 2*col x 2*col indefinite matrix
	                [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
	                [L_a -R_z           theta*S'AA'S ]

	wn1 is a double precision array of dimension 2m x 2m.
	  On entry wn1 stores the lower triangular part of 
	                [Y' ZZ'Y   L_a'+R_z']
	                [L_a+R_z   S'AA'S   ]
	    in the previous iteration.
	  On exit wn1 stores the corresponding updated matrices.
	  The purpose of wn1 is just to store these inner products
	  so they can be easily updated and inserted into wn.

	m is an integer variable.
	  On entry m is the maximum number of variable metric corrections
	    used to define the limited memory matrix.
	  On exit m is unchanged.

	ws, wy, sy, and wtyy are double precision arrays;
	theta is a double precision variable;
	col is an integer variable;
	head is an integer variable.
	  On entry they store the information defining the
	                                     limited memory BFGS matrix:
	    ws(n,m) stores S, a set of s-vectors;
	    wy(n,m) stores Y, a set of y-vectors;
	    sy(m,m) stores S'Y;
	    wtyy(m,m) stores the Cholesky factorization
	                              of (theta*S'S+LD^(-1)L')
	    theta is the scaling factor specifying B_0 = theta I;
	    col is the number of variable metric corrections stored;
	    head is the location of the 1st s- (or y-) vector in S (or Y).
	  On exit they are unchanged.

	info is an integer variable.
	  On entry info is unspecified.
	  On exit info =  0 for normal return;
	               = -1 when the 1st Cholesky factorization failed;
	               = -2 when the 2st Cholesky factorization failed.

	Subprograms called:

	  Linpack ... dcopy, dpofa, dtrsl.


	References:
	  [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
	  memory algorithm for bound constrained optimization'',
	  SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

	  [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: a
	  limited memory FORTRAN code for solving bound constrained
	  optimization problems'', Tech. Report, NAM-11, EECS Department,
	  Northwestern University, 1994.

	  (Postscript files of these papers are available via anonymous
	   ftp to eecs.nwu.edu in the directory pub/lbfgs/lbfgs_bcm.)

	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	************
	*/
	
	/* System generated locals */
	long int wn_dim1, wn_offset, wn1_dim1, wn1_offset, ws_dim1, ws_offset, wy_dim1, wy_offset, sy_dim1, sy_offset;

	/* Local variables */
	static long int i, k, k1, m2, is, js, iy, jy, is1, js1, col2, dend, pend;
	static long int upcl;
	static double temp1, temp2, temp3, temp4;
	static long int ipntr, jpntr, dbegin, pbegin;

	/* Parameter adjustments */
	--indx2;
	--ind;
	sy_dim1 = m;
	sy_offset = 1 + sy_dim1;
	sy -= sy_offset;
	wy_dim1 = n;
	wy_offset = 1 + wy_dim1;
	wy -= wy_offset;
	ws_dim1 = n;
	ws_offset = 1 + ws_dim1;
	ws -= ws_offset;
	wn1_dim1 = 2 * m;
	wn1_offset = 1 + wn1_dim1;
	wn1 -= wn1_offset;
	wn_dim1 = 2 * m;
	wn_offset = 1 + wn_dim1;
	wn -= wn_offset;

	/* Form the lower triangular part of
	          WN1 = [Y' ZZ'Y   L_a'+R_z'] 
	                [L_a+R_z   S'AA'S   ]
	   where L_a is the strictly lower triangular part of S'AA'Y
	         R_z is the upper triangular part of S'ZZ'Y. */
	
	if (updatd)
	{
		if (iupdat > m)
		{
			/* Shift old part of WN1. */
			for (jy = 1; jy <= m - 1; ++jy)
			{
				js = m + jy;
				dcopy(m - jy, &wn1[jy + 1 + (jy + 1) * wn1_dim1], 1, &wn1[jy + jy * wn1_dim1], 1);
				dcopy(m - jy, &wn1[js + 1 + (js + 1) * wn1_dim1], 1, &wn1[js + js * wn1_dim1], 1);
				dcopy(m - 1, &wn1[m + 2 + (jy + 1) * wn1_dim1], 1, &wn1[m + 1 + jy * wn1_dim1], 1);
			}
		}

		/* Put new rows in blocks (1,1), (2,1) and (2,2). */
		pbegin = 1;
		pend = nsub;
		dbegin = nsub + 1;
		dend = n;
		iy = col;
		is = m + col;
		ipntr = head + col - 1;
		if (ipntr > m)
		{
			ipntr -= m;
		}
		jpntr = head;
		for (jy = 1; jy <= col; ++jy)
		{
			js = m + jy;
			temp1 = 0.;
			temp2 = 0.;
			temp3 = 0.;
			
			/* Compute element jy of row 'col' of Y'ZZ'Y */
			for (k = pbegin; k <= pend; ++k)
			{
				k1 = ind[k];
				temp1 += wy[k1 + ipntr * wy_dim1] * wy[k1 + jpntr * wy_dim1];
			}
			
			/* Compute elements jy of row 'col' of L_a and S'AA'S */
			for (k = dbegin; k <= dend; ++k)
			{
				k1 = ind[k];
				temp2 += ws[k1 + ipntr * ws_dim1] * ws[k1 + jpntr * ws_dim1];
				temp3 += ws[k1 + ipntr * ws_dim1] * wy[k1 + jpntr * wy_dim1];
			}
			wn1[iy + jy * wn1_dim1] = temp1;
			wn1[is + js * wn1_dim1] = temp2;
			wn1[is + jy * wn1_dim1] = temp3;
			jpntr = jpntr % m + 1;
		}
		
		/* Put new column in block (2,1). */
		jy = col;
		jpntr = head + col - 1;
		if (jpntr > m)
		{
			jpntr -= m;
		}
		
		ipntr = head;
		
		for (i = 0; i < col; ++i)
		{
			is = m + i + 1;
			temp3 = 0.;
			
			/* Compute element i of column 'col' of R_z */
			for (k = pbegin; k <= pend; ++k)
			{
				k1 = ind[k];
				temp3 += ws[k1 + ipntr * ws_dim1] * wy[k1 + jpntr * wy_dim1];
			}
			ipntr = ipntr % m + 1;
			wn1[is + jy * wn1_dim1] = temp3;
		}
		upcl = col - 1;
	}
	else
	{
		upcl = col;
	}
	
	/* Modify the old parts in blocks (1,1) and (2,2) due to changes in the set of free variables. */
	ipntr = head;
	for (iy = 1; iy <= upcl; ++iy)
	{
		is = m + iy;
		jpntr = head;
		for (jy = 1; jy <= iy; ++jy)
		{
			js = m + jy;
			temp1 = 0.;
			temp2 = 0.;
			temp3 = 0.;
			temp4 = 0.;
			for (k = 1; k <= nenter; ++k)
			{
				k1 = indx2[k];
				temp1 += wy[k1 + ipntr * wy_dim1] * wy[k1 + jpntr * wy_dim1];
				temp2 += ws[k1 + ipntr * ws_dim1] * ws[k1 + jpntr * ws_dim1];
			}
			
			for (k = ileave; k <= n; ++k)
			{
				k1 = indx2[k];
				temp3 += wy[k1 + ipntr * wy_dim1] * wy[k1 + jpntr * wy_dim1];
				temp4 += ws[k1 + ipntr * ws_dim1] * ws[k1 + jpntr * ws_dim1];
			}
			wn1[iy + jy * wn1_dim1] = wn1[iy + jy * wn1_dim1] + temp1 - temp3;
			wn1[is + js * wn1_dim1] = wn1[is + js * wn1_dim1] - temp2 + temp4;
			jpntr = jpntr % m + 1;
		}
		ipntr = ipntr % m + 1;
	}
	
	/* Modify the old parts in block (2,1). */
	ipntr = head;
	for (is = m + 1; is <= m + upcl; ++is)
	{
		jpntr = head;
		for (jy = 1; jy <= upcl; ++jy)
		{
			temp1 = 0.;
			temp3 = 0.;
			for (k = 1; k <= nenter; ++k)
			{
				k1 = indx2[k];
				temp1 += ws[k1 + ipntr * ws_dim1] * wy[k1 + jpntr * wy_dim1];
			}
			
			for (k = ileave; k <= n; ++k)
			{
				k1 = indx2[k];
				temp3 += ws[k1 + ipntr * ws_dim1] * wy[k1 + jpntr * wy_dim1];
			}
			
			if (is <= jy + m)
			{
				wn1[is + jy * wn1_dim1] = wn1[is + jy * wn1_dim1] + temp1 - temp3;
			}
			else
			{
				wn1[is + jy * wn1_dim1] = wn1[is + jy * wn1_dim1] - temp1 + temp3;
			}
			jpntr = jpntr % m + 1;
		}
		ipntr = ipntr % m + 1;
	}
	
	/* Form the upper triangle of WN = [D+Y' ZZ'Y/theta    -L_a'+R_z'] */
	/*                                 [-L_a +R_z		 S'AA'S*theta] */
	m2 = m << 1;
	for (iy = 1; iy <= col; ++iy)
	{
		is = col + iy;
		is1 = m + iy;
		for (jy = 1; jy <= iy; ++jy)
		{
			js = col + jy;
			js1 = m + jy;
			wn[jy + iy * wn_dim1] = wn1[iy + jy * wn1_dim1] / theta;
			wn[js + is * wn_dim1] = wn1[is1 + js1 * wn1_dim1] * theta;
		}
		
		for (jy = 1; jy <= iy - 1; ++jy)
		{
			wn[jy + is * wn_dim1] = -wn1[is1 + jy * wn1_dim1];
		}
		
		for (jy = iy; jy <= col; ++jy)
		{
			wn[jy + is * wn_dim1] = wn1[is1 + jy * wn1_dim1];
		}
		wn[iy + iy * wn_dim1] += sy[iy + iy * sy_dim1];
	}
	
	/* Form the upper triangle of WN= [  LL'			L^-1(-L_a' + R_z')] */
	/*								  [(-L_a + R_z)L'^-1    S'AA'S * theta] */
	/* First Cholesky factor (1,1) block of wn to get LL' */
	/* with L' stored in the upper triangle of wn. */
	dpofa(&wn[wn_offset], m2, col, info);
	
	if (*info != 0)
	{
		*info = -1;
		return;
	}
	
	/* Then form L^-1(-L_a' + R_z') in the (1,2) block. */
	col2 = col << 1;
	for (js = col + 1; js <= col2; ++js)
	{
		dtrsl(&wn[wn_offset], m2, col, &wn[js * wn_dim1 + 1], 11, info);
	}
	
	/* Form S'AA'S * theta + (L^-1(-L_a' + R_z'))'L^-1(-L_a' + R_z') in the upper triangle of (2,2) block of wn. */
	for (is = col + 1; is <= col2; ++is)
	{
		for (js = is; js <= col2; ++js)
		{
			wn[is + js * wn_dim1] += ddot(col, &wn[is * wn_dim1 + 1], 1, &wn[js * wn_dim1 + 1], 1);
		}
	}
	
	/* Cholesky factorization of (2,2) block of wn. */
	dpofa(&wn[col + 1 + (col + 1) * wn_dim1], m2, col, info);
	
	if (*info != 0)
	{
		*info = -2;
		return;
	}
	
	return;
} // end of formk function */
/* ======================= The end of formk ============================== */

void formt(long int m, double* wt, double* sy, double* ss, long int col, double theta, long int* info)
{
	/*
	************

	Subroutine formt

	  This subroutine forms the upper half of the pos. def. and symm.
	    T = theta*SS + L*D^(-1)*L', stores T in the upper triangle
	    of the array wt, and performs the Cholesky factorization of T
	    to produce J*J', with J' stored in the upper triangle of wt.

	Subprograms called:

	  Linpack ... dpofa.


	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	************
	*/
	
	/* System generated locals */
	long int wt_dim1, wt_offset, sy_dim1, sy_offset, ss_dim1, ss_offset;

	/* Local variables */
	static long int i, j, k, k1;
	static double ddum;

	/* Parameter adjustments */
	ss_dim1 = m;
	ss_offset = 1 + ss_dim1;
	ss -= ss_offset;
	sy_dim1 = m;
	sy_offset = 1 + sy_dim1;
	sy -= sy_offset;
	wt_dim1 = m;
	wt_offset = 1 + wt_dim1;
	wt -= wt_offset;

	/* Form the upper half of  T = theta*SS + L*D^(-1)*L',
	     store T in the upper triangle of the array wt. */
	for (j = 0; j < col; ++j)
	{
		wt[(j + 1) * wt_dim1 + 1] = theta * ss[(j + 1) * ss_dim1 + 1];
	}
	
	for (i = 1; i < col; ++i)
	{
		for (j = i; j < col; ++j)
		{
			k1 = fmin(i + 1, j + 1) - 1;
			ddum = 0.;
			for (k = 0; k < k1; ++k)
			{
				ddum += sy[i + 1 + (k + 1) * sy_dim1] * sy[j + 1 + (k + 1) * sy_dim1] / sy[k + 1 + (k + 1) * sy_dim1];
			}
			wt[i + 1 + (j + 1) * wt_dim1] = ddum + theta * ss[i + 1 + (j + 1) * ss_dim1];
		}
	}
	
	/* Cholesky factorize T to J * J' with J' stored in the upper triangle of wt. */
	dpofa(&wt[wt_offset], m, col, info);
	
	if (*info != 0)
	{
		*info = -3;
	}
	
	return;
} // end of formt function */
/* ======================= The end of formt ============================== */

void freev(long int n, long int* nfree, long int* index, long int* nenter, long int* ileave, long int* indx2, long int* iwhere, long int* wrk, long int updatd, long int cnstnd, long int iprint, long int iter)
{
	/*
	************

	Subroutine freev 

	This subroutine counts the entering and leaving variables when
	  iter > 0, and finds the index set of free and active variables
	  at the GCP.

	cnstnd is a logical variable indicating whether bounds are present

	index is an integer array of dimension n
	  for i=1,...,nfree, index(i) are the indices of free variables
	  for i=nfree+1,...,n, index(i) are the indices of bound variables
	  On entry after the first iteration, index gives 
	    the free variables at the previous iteration.
	  On exit it gives the free variables based on the determination
	    in cauchy using the array iwhere.

	indx2 is an integer array of dimension n
	  On entry indx2 is unspecified.
	  On exit with iter>0, indx2 indicates which variables
	     have changed status since the previous iteration.
	  For i= 1,...,nenter, indx2(i) have changed from bound to free.
	  For i= ileave+1,...,n, indx2(i) have changed from free to bound.
 

	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	************
	*/
	
	/* Local variables */
	static long int i, k, iact;

	*nenter = 0;
	*ileave = n + 1;
	if (iter > 0 && cnstnd)
	{
		/* Count the entering and leaving variables. */
		for (i = 0; i < *nfree; ++i)
		{
			k = index[i];
			if (iwhere[k - 1] > 0)
			{
				--(*ileave);
				indx2[*ileave - 1] = k;
				if (iprint >= 100)
				{
					printf("Variable %ld leaves the set of free variables\n", k);
				}
			}
		}
		
		for (i = *nfree; i < n; ++i)
		{
			k = index[i];
			if (iwhere[k - 1] <= 0)
			{
				++(*nenter);
				indx2[*nenter - 1] = k;
				if (iprint >= 100)
				{
					printf("Variable %ld leaves the set of free variables\n", k);
				}
			}
		}
		
		if (iprint >= 99)
		{
			printf("%ld variables leave; %ld variables enter\n", n + 1 - *ileave, *nenter);
		}
	}
	
	*wrk = *ileave < n + 1 || *nenter > 0 || updatd;
	
	/* Find the index set of free and active variables at the GCP. */
	*nfree = 0;
	iact = n;
	
	for (i = 0; i < n; ++i)
	{
		if (iwhere[i] <= 0)
		{
			index[*nfree] = i + 1;
			++(*nfree);
		}
		else
		{
			--iact;
			index[iact] = i + 1;
		}
	}
	
	if (iprint >= 99)
	{
		printf("%ld variables are free at GCP iter %ld\n", *nfree, iter + 1);
	}
	
	return;
} // end of freev function */
/* ======================= The end of freev ============================== */

void hpsolb(long int n, double* t, long int* iorder, long int iheap)
{
	/*
	************

	Subroutine hpsolb 

	This subroutine sorts out the least element of t, and puts the
	  remaining elements of t in a heap.
 
	n is an integer variable.
	  On entry n is the dimension of the arrays t and iorder.
	  On exit n is unchanged.

	t is a double precision array of dimension n.
	  On entry t stores the elements to be sorted,
	  On exit tn stores the least elements of t, and t(1) to t(n-1)
	    stores the remaining elements in the form of a heap.

	iorder is an integer array of dimension n.
	  On entry iorder(i) is the index of t(i).
	  On exit iorder(i) is still the index of t(i), but iorder may be
	    permuted in accordance with t.

	iheap is an integer variable specifying the task.
	  On entry iheap should be set as follows:
	    iheap .eq. 0 if t(1) to tn is not in the form of a heap,
	    iheap .ne. 0 if otherwise.
	  On exit iheap is unchanged.


	References:
	  Algorithm 232 of CACM (J. W. J. Williams): HEAPSORT.

	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.

	************
	*/
	
	/* Local variables */
	static long int i, j, k;
	static double out, ddum;
	static long int indxin, indxou;

	/* Parameter adjustments */
	--iorder;
	--t;

	/* Function Body */
	if (iheap == 0)
	{
		/* Rearrange the elements t[0] to t[n - 1] to form a heap. */
		for (k = 1; k < n; ++k)
		{
			ddum = t[k + 1];
			indxin = iorder[k + 1];
			
			/* Add ddum to the heap. */
			i = k + 1;
L10:
			if (i > 1)
			{
				j = i / 2;
				if (ddum < t[j])
				{
					t[i] = t[j];
					iorder[i] = iorder[j];
					i = j;
					goto L10;
				}
			}
			t[i] = ddum;
			iorder[i] = indxin;
		}
	}
	
	/* Assign to 'out' the value of t[1], the least member of the heap, */
	/* and rearrange the remaining members to form a heap as */
	/* elements 1 to n - 1 of t. */
	if (n > 1)
	{
		i = 1;
		out = t[1];
		indxou = iorder[1];
		ddum = t[n];
		indxin = iorder[n];
		/* Restore the heap */
L30:
		j = i + i;
		if (j <= n - 1)
		{
			if (t[j + 1] < t[j])
			{
				++j;
			}
			
			if (t[j] < ddum)
			{
				t[i] = t[j];
				iorder[i] = iorder[j];
				i = j;
				goto L30;
			}
		}
		t[i] = ddum;
		iorder[i] = indxin;
		
		/* Put the least member in t[n]. */
		t[n] = out;
		iorder[n] = indxou;
	}
	
	return;
} // end of hpsolb function */
/* ====================== The end of hpsolb ============================== */

void lnsrlb(long int n, double* l, double* u, long int* nbd, double* x, double* f, double* fold, double* gd, double* gdold, double* g, double* d__, double* r__, double* t, double* z__, double* stp, double* dnorm, double* dtd, double* xstep, double* stpmx, long int iter, long int* ifun, long int* iback, long int* nfgv, long int* info, long int* task, long int boxed, long int cnstnd, long int* csave, long int* isave, double* dsave)
{
	/*
	**********

	Subroutine lnsrlb

	This subroutine calls subroutine dcsrch from the Minpack2 library
	  to perform the line search.  Subroutine dscrch is safeguarded so
	  that all trial points lie within the feasible region.

	Subprograms called:

	  Minpack2 Library ... dcsrch.

	  Linpack ... dtrsl, ddot.


	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	**********
	*/
	
	/* Local variables */
	static long int i;
	static double a1, a2;

	if (*task == FG_LN)
	{ 
		goto L556;
	}
	*dtd = ddot(n, d__, 1, d__, 1);
	*dnorm = sqrt(*dtd);
	
	/* Determine the maximum step length. */
	*stpmx = 1e10;
	if (cnstnd)
	{
		if (iter == 0)
		{
			*stpmx = 1.;
		}
		else
		{
			for (i = 0; i < n; ++i)
			{
				a1 = d__[i];
				if (nbd[i] != 0)
				{
					if (a1 < 0. && nbd[i] <= 2)
					{
						a2 = l[i] - x[i];
						if (a2 >= 0.)
						{
							*stpmx = 0.;
						}
						else if (a1 * *stpmx < a2)
						{
							*stpmx = a2 / a1;
						}
					}
					else if (a1 > 0. && nbd[i] >= 2)
					{
						a2 = u[i] - x[i];
						if (a2 <= 0.)
						{
							*stpmx = 0.;
						}
						else if (a1 * *stpmx > a2)
						{
							*stpmx = a2 / a1;
						}
					}
				}
			}
		}
	}
	
	if (iter == 0 && !boxed)
	{
		*stp = fmin(1. / *dnorm, *stpmx);
	}
	else
	{
		*stp = 1.;
	}
	
	dcopy(n, x, 1, t, 1);
	dcopy(n, g, 1, r__, 1);
	
	*fold = *f;
	*ifun = 0;
	*iback = 0;
	*csave = START;
L556:
	*gd = ddot(n, g, 1, d__, 1);
	if (*ifun == 0)
	{
		*gdold = *gd;
		if (*gd >= 0.)
		{
			/* The directional derivative >= 0. */
			/*  Line search is impossible. */
			printf("ascend direction in projection gd = %.2e\n", *gd); 
			*info = -4;
			return;
		}
	}

	dcsrch(f, gd, stp, FTOL, GTOL, XTOL, STEPMIN, (*stpmx), csave, isave, dsave);
	
	*xstep = *stp * *dnorm;
	
	if (!(IS_WARNING(*csave)) && !(IS_CONVERGED(*csave)))
	{
		*task = FG_LNSRCH;
		++(*ifun);
		++(*nfgv);
		*iback = *ifun - 1;
		
		if (*stp == 1.)
		{
			dcopy(n, z__, 1, x, 1);
		}
		else
		{
			for (i = 0; i < n; ++i)
			{
				x[i] = *stp * d__[i] + t[i];
			}
		}
	}
	else
	{
		*task = NEW_X;
	}
	
	return;
} // end of lnsrlb function */
/* ======================= The end of lnsrlb ============================= */

void matupd(long int n, long int m, double* ws, double* wy, double* sy, double* ss, double* d__, double* r__, long int* itail, long int iupdat, long int* col, long int* head, double* theta, double rr, double dr, double stp, double dtd)
{
	/*
	************

	Subroutine matupd

	  This subroutine updates matrices WS and WY, and forms the
	    middle matrix in B.

	Subprograms called:

	  Linpack ... dcopy, ddot.


	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	************
	*/
	
	/* System generated locals */
	long int ws_dim1, ws_offset, wy_dim1, wy_offset, sy_dim1, sy_offset, ss_dim1, ss_offset;

	/* Local variables */
	static long int j;
	static long int pointr;

	/* Parameter adjustments */
	ss_dim1 = m;
	ss_offset = 1 + ss_dim1;
	ss -= ss_offset;
	sy_dim1 = m;
	sy_offset = 1 + sy_dim1;
	sy -= sy_offset;
	wy_dim1 = n;
	wy_offset = 1 + wy_dim1;
	wy -= wy_offset;
	ws_dim1 = n;
	ws_offset = 1 + ws_dim1;
	ws -= ws_offset;

	/* Set pointers for matrices WS and WY. */
	if (iupdat <= m)
	{
		*col = iupdat;
		*itail = (*head + iupdat - 2) % m + 1;
	}
	else
	{
		*itail = *itail % m + 1;
		*head = *head % m + 1;
	}
	
	/* Update matrices WS and WY. */
	dcopy(n, d__, 1, &ws[*itail * ws_dim1 + 1], 1);
	dcopy(n, r__, 1, &wy[*itail * wy_dim1 + 1], 1);
	
	/* Set theta = yy / ys. */
	*theta = rr / dr;
	
	/* Form the middle matrix in B. */
	/* Update the upper triangle of SS, and the lower triangle of SY: */
	if (iupdat > m)
	{
		/* Move old information */
		for (j = 0; j < *col - 1; ++j)
		{
			dcopy(j + 1, &ss[(j + 1 + 1) * ss_dim1 + 2], 1, &ss[(j + 1) * ss_dim1 + 1], 1);
			dcopy(*col - j + 1, &sy[j + 1 + 1 + (j + 1 + 1) * sy_dim1], 1, &sy[j + 1 + (j + 1) * sy_dim1], 1);
		}
	}
	
	/* Add new information: the last row of SY and the last column of SS: */
	pointr = *head;
	for (j = 0; j < *col - 1; ++j)
	{
		sy[*col + (j + 1) * sy_dim1] = ddot(n, d__, 1, &wy[pointr * wy_dim1 + 1], 1);
		ss[j + 1 + *col * ss_dim1] = ddot(n, &ws[pointr * ws_dim1 + 1], 1, d__, 1);
		pointr = pointr % m + 1;
	}
	
	if (stp == 1.)
	{
		ss[*col + *col * ss_dim1] = dtd;
	}
	else
	{
		ss[*col + *col * ss_dim1] = stp * stp * dtd;
	}
	sy[*col + *col * sy_dim1] = dr;
	
	return;
} // end of matupd function */
/* ======================= The end of matupd ============================= */

void prn1lb(long int n, long int m, double* l, double* u, double* x, long int iprint, double epsmch)
{
	/*
	************ 

	Subroutine prn1lb 

	This subroutine prints the input data, initial point, upper and 
	  lower bounds of each variable, machine precision, as well as 
	  the headings of the output. 


						  *  *  * 

	NEOS, November 1994. (Latest revision June 1996.) 
	Optimization Technology Center. 
	Argonne National Laboratory and Northwestern University. 
	Written by 
					   Ciyou Zhu 
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal. 


	************ 
	*/

	/* Local variables */
	static long int i;

	if (iprint >= 0)
	{
		printf("		   * * *\n");
		printf("		RUNNING THE L-BFGS-B CODE\n");
		printf("		   * * *\n");
		printf("Machine precision = %.2e\n", epsmch);
		printf(" N = %10ld\n M = %10ld\n", n, m);
		if (iprint >= 1)
		{
			if (iprint > 100)
			{
				printf("L  =");
				for (i = 0; i < n; ++i)
				{
					printf("%.2e ", l[i]);
				}
				
				printf("\n");
				printf("X0 =");
				for (i = 0; i < n; ++i)
				{
					printf("%.2e ", x[i]);
				}
				
				printf("\n");
				printf("U  =");
				for (i = 0; i < n; ++i)
				{
					printf("%.2e ", u[i]);
				}
				printf("\n");
			}
		}
	}
	
	return;
} // end of prn1lb function */

void prn2lb(long int n, double* x, double f, double* g, long int iprint, long int iter, double sbgnrm, long int* word, long int iword, long int iback, double xstep)
{
	/*
	************ 

	Subroutine prn2lb 

	This subroutine prints out new information after a successful 
	  line search. 


						  *  *  * 

	NEOS, November 1994. (Latest revision June 1996.) 
	Optimization Technology Center. 
	Argonne National Laboratory and Northwestern University. 
	Written by 
					   Ciyou Zhu 
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal. 


	************ 
	*/
	
	/* 'word' records the status of subspace solutions. */

	/* Local variables */
	static long int i, imod;

	if (iword == 0)
	{
		*word = WORD_CON;
		/* The subspace minimization converged. */
	}
	else if (iword == 1)
	{
		*word = WORD_BND;
		/* The subspace minimization stopped at a bound. */
	}
	else if (iword == 5)
	{
		*word = WORD_TNT;
		/* The truncated Newton step has been used. */
	}
	else
	{
		*word = WORD_DEFAULT;
	}
	
	if (iprint >= 99)
	{
		printf("LINE SEARCH %ld times; norm of step = %.2e\n", iback, xstep);
		printf("At iterate %5ld, f(x)= %5.2e, ||proj grad||_infty = %.2e\n", iter, f, sbgnrm);
		if (iprint > 100)
		{
			printf("X =");
			for (i = 0; i < n; ++i)
			{
				printf("%.2e ", x[i]);
			}
			
			printf("\nG =");
			for (i = 0; i < n; ++i)
			{
				printf("%.2e ", g[i]);
			}
			printf("\n");
		}
	}
	else if (iprint > 0)
	{
		imod = iter % iprint;
		if (imod == 0)
		{
			printf("At iterate %5ld, f(x)= %5.2e, ||proj grad||_infty = %.2e\n", iter, f, sbgnrm);
		}
	}
	
	return;
} // end of prn2lb function */

void prn3lb(long int n, double* x, double f, long int task, long int iprint, long int info, long int iter, long int nfgv, long int nintol, long int nskip, long int nact, double sbgnrm, double time, double stp, double xstep, long int k, double cachyt, double sbtime, double lnscht)
{
	/*
	************ 

	Subroutine prn3lb 

	This subroutine prints out information when either a built-in 
	  convergence test is satisfied or when an error message is 
	  generated. 


						  *  *  * 

	NEOS, November 1994. (Latest revision June 1996.) 
	Optimization Technology Center. 
	Argonne National Laboratory and Northwestern University. 
	Written by 
					   Ciyou Zhu 
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal. 


	************ 
	*/
	
	/* Local variables */
	static long int i;
	
	if (IS_ERROR(task))
	{
		goto L999;
	}
	
	if (iprint >= 0)
	{
		printf("           * * * \n");
		printf("Tit   = total number of iterations\n");
		printf("Tnf   = total number of function evaluations\n");
		printf("Tnint = total number of segments explored during Cauchy searches\n");
		printf("Skip  = number of BFGS updates skipped\n");
		printf("Nact  = number of active bounds at final generalized Cauchy point\n");
		printf("Projg = norm of the final projected gradient\n");
		printf("F     = final function value\n");
		printf("           * * * \n");

		printf("   N    Tit   Tnf  Tnint  Skip  Nact      Projg        F\n");
		printf("%5ld %5ld %5ld %5ld %5ld %5ld\t%6.2e %9.5e\n", n, iter, nfgv, nintol, nskip, nact, sbgnrm, f);
		
		if (iprint >= 100)
		{
			printf("X = ");
			for (i = 0; i < n; ++i)
			{
				printf(" %.2e", x[i]);
			}
			printf("\n");
		}
		
		if (iprint >= 1)
		{
			printf("F(x) = %.9e\n", f);
		}
	}
L999:
	if (iprint >= 0)
	{
		printf("%ld\n", task);
		if (info != 0)
		{
			if (info == -1)
			{
				printf(" Matrix in 1st Cholesky factorization in formk is not Pos. Def.\n");
			}
			
			if (info == -2)
			{
				printf(" Matrix in 2nd Cholesky factorization in formk is not Pos. Def.\n");
			}
			
			if (info == -3)
			{
				printf(" Matrix in the Cholesky factorization in formt is not Pos. Def.\n");
			}
			
			if (info == -4)
			{
				printf(" Derivative >= 0, backtracking line search impossible.\n");
				printf("  Previous x, f and g restored.\n");
				printf(" Possible causes: 1 error in function or gradient evaluation;\n");
				printf("                  2 rounding errors dominate computation.\n");
			}
			
			if (info == -5)
			{
				printf(" Warning:  more than 10 function and gradient\n");
				printf("   evaluations in the last line search.  Termination\n");
				printf("   may possibly be caused by a bad search direction.\n");
			}
			
			if (info == -6)
			{
				printf(" Input nbd(%ld) is invalid\n", k);
			}
			
			if (info == -7)
			{
				printf(" l(%ld) > u(%ld). No feasible solution.\n", k, k);
			}
			
			if (info == -8)
			{
				printf(" The triangular system is singular.\n");
			}
			
			if (info == -9)
			{
				printf(" Line search cannot locate an adequate point after 20 function\n");
				printf("  and gradient evaluations.  Previous x, f and g restored.\n");
				printf(" Possible causes: 1 error in function or gradient evaluation;\n");
				printf("                  2 rounding error dominate computation.\n");
			}
		}

		if (iprint >= 1)
		{
			printf("Cauchy                time %.3e seconds.\n", cachyt);
			printf("Subspace minimization time %.3e seconds.\n", sbtime);
			printf("Line search           time %.3e seconds.\n", lnscht);
		}
		printf(" Total User time %.3e seconds.\n", time);
	}
	
	return;
} // end of prn3lb function */

void projgr(long int n, double* l, double* u, long int* nbd, double* x, double* g, double* sbgnrm)
{
	/*
	************

	Subroutine projgr

	This subroutine computes the infinity norm of the projected
	  gradient.


	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	************
	*/
	
	/* Local variables */
	static long int i;
	static double gi;

	*sbgnrm = 0.;
	for (i = 0; i < n; ++i)
	{
		gi = g[i];
		if (nbd[i] != 0)
		{
			if (gi < 0.)
			{
				if (nbd[i] >= 2)
				{
					gi = fmax(x[i] - u[i], gi);
				}
			}
			else
			{
				if (nbd[i] <= 2)
				{
					gi = fmin(x[i] - l[i], gi);
				}
			}
		}

		*sbgnrm = fmax(*sbgnrm, fabs(gi));
	}
	
	return;
} // end of projgr function */
/* ======================= The end of projgr ============================= */

void subsm(long int n, long int m, long int nsub, long int* ind, double* l, double* u, long int* nbd, double* x, double* d__, double* xp, double* ws, double* wy, double theta, double* xx, double* gg, long int col, long int head, long int* iword, double* wv, double* wn, long int iprint, long int* info)
{
	/*
	************

	Subroutine subsm

	Given xcp, l, u, r, an index set that specifies
	  the active set at xcp, and an l-BFGS matrix B 
	  (in terms of WY, WS, SY, WT, head, col, and theta), 
	  this subroutine computes an approximate solution
	  of the subspace problem

	  (P)   min Q(x) = r'(x-xcp) + 1/2 (x-xcp)' B (x-xcp)

	        subject to l<=x<=u
	                  x_i=xcp_i for all i in A(xcp)
	                
	  along the subspace unconstrained Newton direction 
	  
	     d = -(Z'BZ)^(-1) r.

	  The formula for the Newton direction, given the L-BFGS matrix
	  and the Sherman-Morrison formula, is

	     d = (1/theta)r + (1/theta*2) Z'WK^(-1)W'Z r.
 
	  where
	            K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
	                [L_a -R_z           theta*S'AA'S ]

	Note that this procedure for computing d differs 
	from that described in [1]. One can show that the matrix K is
	equal to the matrix M^[-1]N in that paper.

	n is an integer variable.
	  On entry n is the dimension of the problem.
	  On exit n is unchanged.

	m is an integer variable.
	  On entry m is the maximum number of variable metric corrections
	    used to define the limited memory matrix.
	  On exit m is unchanged.

	nsub is an integer variable.
	  On entry nsub is the number of free variables.
	  On exit nsub is unchanged.

	ind is an integer array of dimension nsub.
	  On entry ind specifies the coordinate indices of free variables.
	  On exit ind is unchanged.

	l is a double precision array of dimension n.
	  On entry l is the lower bound of x.
	  On exit l is unchanged.

	u is a double precision array of dimension n.
	  On entry u is the upper bound of x.
	  On exit u is unchanged.

	nbd is a integer array of dimension n.
	  On entry nbd represents the type of bounds imposed on the
	    variables, and must be specified as follows:
	    nbd(i)=0 if x(i) is unbounded,
	           1 if x(i) has only a lower bound,
	           2 if x(i) has both lower and upper bounds, and
	           3 if x(i) has only an upper bound.
	  On exit nbd is unchanged.

	x is a double precision array of dimension n.
	  On entry x specifies the Cauchy point xcp. 
	  On exit x(i) is the minimizer of Q over the subspace of
	                                                   free variables. 

	d is a double precision array of dimension n.
	  On entry d is the reduced gradient of Q at xcp.
	  On exit d is the Newton direction of Q. 

    xp is a double precision array of dimension n.
	  used to safeguard the projected Newton direction 

    xx is a double precision array of dimension n
	  On entry it holds the current iterate
	  On output it is unchanged

    gg is a double precision array of dimension n
	  On entry it holds the gradient at the current iterate
	  On output it is unchanged

	ws and wy are double precision arrays;
	theta is a double precision variable;
	col is an integer variable;
	head is an integer variable.
	  On entry they store the information defining the
	                                     limited memory BFGS matrix:
	    ws(n,m) stores S, a set of s-vectors;
	    wy(n,m) stores Y, a set of y-vectors;
	    theta is the scaling factor specifying B_0 = theta I;
	    col is the number of variable metric corrections stored;
	    head is the location of the 1st s- (or y-) vector in S (or Y).
	  On exit they are unchanged.

	iword is an integer variable.
	  On entry iword is unspecified.
	  On exit iword specifies the status of the subspace solution.
	    iword = 0 if the solution is in the box,
	            1 if some bound is encountered.

	wv is a double precision working array of dimension 2m.

	wn is a double precision array of dimension 2m x 2m.
	  On entry the upper triangle of wn stores the LEL^T factorization
	    of the indefinite matrix

	         K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
	             [L_a -R_z           theta*S'AA'S ]
	                                               where E = [-I  0]
	                                                         [ 0  I]
	  On exit wn is unchanged.

	iprint is an INTEGER variable that must be set by the user.
	  It controls the frequency and type of output generated:
	   iprint<0    no output is generated;
	   iprint=0    print only one line at the last iteration;
	   0<iprint<99 print also f and |proj g| every iprint iterations;
	   iprint=99   print details of every iteration except n-vectors;
	   iprint=100  print also the changes of active set and final x;
	   iprint>100  print details of every iteration including x and g;
	  When iprint > 0, the file iterate.dat will be created to
	                   summarize the iteration.

	info is an integer variable.
	  On entry info is unspecified.
	  On exit info = 0       for normal return,
	               = nonzero for abnormal return 
	                             when the matrix K is ill-conditioned.

	Subprograms called:

	  Linpack dtrsl.


	References:

	  [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
	  memory algorithm for bound constrained optimization'',
	  SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.



	                      *  *  *

	NEOS, November 1994. (Latest revision June 1996.)
	Optimization Technology Center.
	Argonne National Laboratory and Northwestern University.
	Written by
	                   Ciyou Zhu
	in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.


	************
	*/
	
	/* System generated locals */
	long int ws_dim1, ws_offset, wy_dim1, wy_offset, wn_dim1, wn_offset;

	/* Local variables */
	static long int i, j, k, m2;
	static double dk;
	static long int js, jy;
	static double xk;
	static long int ibd, col2;
	static double dd_p__, temp1, temp2, alpha;
	static long int pointr;

	/* Parameter adjustments */
	wn_dim1 = 2 * m;
	wn_offset = 1 + wn_dim1;
	wn -= wn_offset;
	wy_dim1 = n;
	wy_offset = 1 + wy_dim1;
	wy -= wy_offset;
	ws_dim1 = n;
	ws_offset = 1 + ws_dim1;
	ws -= ws_offset;

	if (nsub <= 0)
	{
		return;
	}
	
	if (iprint >= 99)
	{
		printf("---------------SUBSM entered---------\n");
	}
	
	/* Compute wv = W'Zd. */
	pointr = head;
	for (i = 0; i < col; ++i)
	{
		temp1 = 0.;
		temp2 = 0.;
		for (j = 0; j < nsub; ++j)
		{
			k = ind[j];
			temp1 += wy[k + pointr * wy_dim1] * d__[j];
			temp2 += ws[k + pointr * ws_dim1] * d__[j];
		}
		wv[i] = temp1;
		wv[col + i] = theta * temp2;
		pointr = pointr % m + 1;
	}
	
	/* Compute wv := K^(-1)wv. */
	m2 = m << 1;
	col2 = col << 1;
	
	dtrsl(&wn[wn_offset], m2, col2, wv, 11, info);
	
	if (*info != 0)
	{
		return;
	}
	
	for (i = 0; i < col; ++i)
	{
		wv[i] = -wv[i];
	}
	
	dtrsl(&wn[wn_offset], m2, col2, wv, 1, info);
	
	if (*info != 0)
	{
		return;
	}
	
	/* Compute d = (1 / theta)d + (1 / theta**2)Z'W wv. */
	pointr = head;
	for (jy = 0; jy < col; ++jy)
	{
		js = col + jy;
		for (i = 0; i < nsub; ++i)
		{
			k = ind[i];
			d__[i] = d__[i] + wy[k + pointr * wy_dim1] * wv[jy] / theta + ws[k + pointr * ws_dim1] * wv[js];
		}
		pointr = pointr % m + 1;
	}
	
	dscal(nsub, 1. / theta, d__, 1);

	/* ----------------------------------------------------------------- */
	/*	 Let us try the projection, d is the Newton direction */
	
	*iword = 0;
	dcopy(n, x, 1, xp, 1);

	for (i = 0; i < nsub; ++i)
	{
		k = ind[i] - 1;
		dk = d__[i];
		xk = x[k];
		if (nbd[k] != 0)
		{
			if (nbd[k] == 1)
			{
				/* Lower bounds only */
				x[k] = fmax(l[k], xk + dk);
				
				if (x[k] == l[k])
				{
					*iword = 1;
				}
			}
			else
			{
				if (nbd[k] == 2)
				{
					/* Upper and lower bounds */
					xk = fmax(l[k], xk + dk);
					x[k] = fmin(u[k], xk);
					
					if (x[k] == l[k] || x[k] == u[k])
					{
						*iword = 1;
					}
				}
				else
				{

					if (nbd[k] == 3)
					{
						/* Upper bounds only */
						x[k] = fmin(u[k], xk + dk);
						
						if (x[k] == u[k])
						{
							*iword = 1;
						}
					}
				}
			}

		}
		else
		{
			/* Free variables */
			x[k] = xk + dk;
		}
	}

	if (*iword == 0)
	{
		goto L911;
	}

	/* Check sign of the directional derivative */
	dd_p__ = 0.;
	for (i = 0; i < n; ++i)
	{
		dd_p__ += (x[i] - xx[i]) * gg[i];
	}
	
	if (dd_p__ > 0.)
	{
		dcopy(n, xp, 1, x, 1);
		printf("Positive dir derivative in projection \n");
		printf("Using the backtracking step\n");
	}
	else
	{
		goto L911;
	}

	/* ----------------------------------------------------------------- */

	alpha = 1.;
	temp1 = alpha;
	ibd = 0;
	for (i = 0; i < nsub; ++i)
	{
		k = ind[i] - 1;
		dk = d__[i];
		if (nbd[k] != 0)
		{
			if (dk < 0. && nbd[k] <= 2)
			{
				temp2 = l[k] - x[k];
				if (temp2 >= 0.)
				{
					temp1 = 0.;
				}
				else if (dk * alpha < temp2)
				{
					temp1 = temp2 / dk;
				}
			}
			else if (dk > 0. && nbd[k] >= 2)
			{
				temp2 = u[k] - x[k];
				if (temp2 <= 0.)
				{
					temp1 = 0.;
				}
				else if (dk * alpha > temp2)
				{
					temp1 = temp2 / dk;
				}
			}
			
			if (temp1 < alpha)
			{
				alpha = temp1;
				ibd = i;
			}
		}
	}
	
	if (alpha < 1.)
	{
		dk = d__[ibd];
		k = ind[ibd] - 1;
		if (dk > 0.)
		{
			x[k] = u[k];
			d__[ibd] = 0.;
		}
		else if (dk < 0.)
		{
			x[k] = l[k];
			d__[ibd] = 0.;
		}
	}
	
	for (i = 0; i < nsub; ++i)
	{
		k = ind[i] - 1;
		x[k] += alpha * d__[i];
	}
L911:
	if (iprint >= 99)
	{
		printf("----------------- exit SUBSM --------------\n");
	}
	
	return;
} // end of subsm function */

void dcsrch(double* f, double* g, double* stp, double ftol, double gtol, double xtol, double stpmin, double stpmax, long int* task, long int* isave, double* dsave)
{
	/*
	**********

	Subroutine dcsrch

	This subroutine finds a step that satisfies a sufficient
	decrease condition and a curvature condition.

	Each call of the subroutine updates an interval with 
	endpoints stx and sty. The interval is initially chosen 
	so that it contains a minimizer of the modified function

	      psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).

	If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
	interval is chosen so that it contains a minimizer of f. 

	The algorithm is designed to find a step that satisfies 
	the sufficient decrease condition 

	      f(stp) <= f(0) + ftol*stp*f'(0),

	and the curvature condition

	      abs(f'(stp)) <= gtol*abs(f'(0)).

	If ftol is less than gtol and if, for example, the function
	is bounded below, then there is always a step which satisfies
	both conditions. 

	If no step can be found that satisfies both conditions, then 
	the algorithm stops with a warning. In this case stp only 
	satisfies the sufficient decrease condition.

	A typical invocation of dcsrch has the following outline:

	task = 'START'
	10 continue
	   call dcsrch( ... )
	   if (task .eq. 'FG') then
	      Evaluate the function and the gradient at stp 
	      goto 10
	      end if

	NOTE: The user must no alter work arrays between calls.

	The subroutine statement is

	   subroutine dcsrch(f,g,stp,ftol,gtol,xtol,stpmin,stpmax,
	                     task,isave,dsave)
	where

	  f is a double precision variable.
	    On initial entry f is the value of the function at 0.
	       On subsequent entries f is the value of the 
	       function at stp.
	    On exit f is the value of the function at stp.

	  g is a double precision variable.
	    On initial entry g is the derivative of the function at 0.
	       On subsequent entries g is the derivative of the 
	       function at stp.
	    On exit g is the derivative of the function at stp.

	  stp is a double precision variable. 
	    On entry stp is the current estimate of a satisfactory 
	       step. On initial entry, a positive initial estimate 
	       must be provided. 
	    On exit stp is the current estimate of a satisfactory step
	       if task = 'FG'. If task = 'CONV' then stp satisfies
	       the sufficient decrease and curvature condition.

	  ftol is a double precision variable.
	    On entry ftol specifies a nonnegative tolerance for the 
	       sufficient decrease condition.
	    On exit ftol is unchanged.

	  gtol is a double precision variable.
	    On entry gtol specifies a nonnegative tolerance for the 
	       curvature condition. 
	    On exit gtol is unchanged.

	  xtol is a double precision variable.
	    On entry xtol specifies a nonnegative relative tolerance
	       for an acceptable step. The subroutine exits with a
	       warning if the relative difference between sty and stx
	       is less than xtol.
	    On exit xtol is unchanged.

	  stpmin is a double precision variable.
	    On entry stpmin is a nonnegative lower bound for the step.
	    On exit stpmin is unchanged.

	  stpmax is a double precision variable.
	    On entry stpmax is a nonnegative upper bound for the step.
	    On exit stpmax is unchanged.

	  task is a character variable of length at least 60.
	    On initial entry task must be set to 'START'.
	    On exit task indicates the required action:

	       If task(1:2) = 'FG' then evaluate the function and 
	       derivative at stp and call dcsrch again.

	       If task(1:4) = 'CONV' then the search is successful.

	       If task(1:4) = 'WARN' then the subroutine is not able
	       to satisfy the convergence conditions. The exit value of
	       stp contains the best point found during the search.

	       If task(1:5) = 'ERROR' then there is an error in the
	       input arguments.

	    On exit with convergence, a warning or an error, the
	       variable task contains additional information.

	  isave is an integer work array of dimension 2.
	    
	  dsave is a double precision work array of dimension 13.

	Subprograms called

	  MINPACK-2 ... dcstep

	MINPACK-1 Project. June 1983.
	Argonne National Laboratory. 
	Jorge J. More' and David J. Thuente.

	MINPACK-2 Project. October 1993.
	Argonne National Laboratory and University of Minnesota. 
	Brett M. Averick, Richard G. Carter, and Jorge J. More'. 

	**********
	*/
	
	/* Local variables */
	static double fm, gm, fx, fy, gx, gy, fxm, fym, gxm, gym, stx, sty;
	static long int stage;
	static double finit, ginit, width, ftest, gtest, stmin, stmax, width1;
	static long int brackt;

	/* Parameter adjustments */
	--dsave;
	--isave;

	/* Initialization block. */
	if (*task == START)
	{
		/* Check the input arguments for errors.  See lbfgsb.h for messages */
		if (*stp < stpmin)  *task=ERROR_SMALLSTP;
		if (*stp > stpmax)  *task=ERROR_LARGESTP;
		if (*g >= 0.)		*task=ERROR_INITIAL;
		if (ftol < 0.)	  *task=ERROR_FTOL;
		if (gtol < 0.)	  *task=ERROR_GTOL;
		if (xtol < 0.)	  *task=ERROR_XTOL;
		if (stpmin < 0.)	*task=ERROR_STP0;
		if (stpmax < stpmin) *task=ERROR_STP1;
		
		/* Exit if there are errors on input. */
		if (IS_ERROR(*task))
		{
			return;
		}
		
		/* Initialize local variables. */
		brackt = 0;
		stage = 1;
		finit = *f;
		ginit = *g;
		gtest = ftol * ginit;
		width = stpmax - stpmin;
		width1 = width * 2.;
		
		/* The variables stx, fx, gx contain the values of the step, function, and derivative at the best step. */
		stx = 0.;
		fx = finit;
		gx = ginit;
		
		/* The variables sty, fy, gy contain the value of the step, function, and derivative at sty. */
		sty = 0.;
		fy = finit;
		gy = ginit;
		
		/* The variables stp, f, g contain the values of the step, function, and derivative at stp. */
		stmin = 0.;
		stmax = *stp + *stp * 4.;
		*task = FG;
		goto L1000;
	}
	else
	{
		/* Restore local variables. */
		if (isave[1] == 1)
		{
			brackt = 1;
		}
		else
		{
			brackt = 0;
		}
		
		stage = isave[2];
		ginit = dsave[1];
		gtest = dsave[2];
		gx = dsave[3];
		gy = dsave[4];
		finit = dsave[5];
		fx = dsave[6];
		fy = dsave[7];
		stx = dsave[8];
		sty = dsave[9];
		stmin = dsave[10];
		stmax = dsave[11];
		width = dsave[12];
		width1 = dsave[13];
	}
	
	/* If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the algorithm enters the second stage. */
	ftest = finit + *stp * gtest;
	if (stage == 1 && *f <= ftest && *g >= 0.)
	{
		stage = 2;
	}
	/*	 Test for warnings. */
	if (brackt && (*stp <= stmin || *stp >= stmax))
	{
		*task = WARNING_ROUND;
	}
	
	if (brackt && stmax - stmin <= xtol * stmax)
	{
		*task = WARNING_XTOL;
	}
	
	if (*stp == stpmax && *f <= ftest && *g <= gtest)
	{
		*task = WARNING_STPMAX;
	}
	
	if (*stp == stpmin && (*f > ftest || *g >= gtest))
	{
		*task = WARNING_STPMIN;
	}
	
	/*	 Test for convergence. */
	if (*f <= ftest && fabs(*g) <= gtol * (-ginit))
	{
		*task = CONVERGENCE;
	}
	
	/*	 Test for termination. */
	if ((IS_WARNING(*task)) || (IS_CONVERGED(*task)))
	{
		goto L1000;
	}
	
	/* A modified function is used to predict the step during the */
	/* first stage if a lower function value has been obtained but */
	/* the decrease is not sufficient. */
	if (stage == 1 && *f <= fx && *f > ftest)
	{
		/* Define the modified function and derivative values. */
		fm = *f - *stp * gtest;
		fxm = fx - stx * gtest;
		fym = fy - sty * gtest;
		gm = *g - gtest;
		gxm = gx - gtest;
		gym = gy - gtest;
		
		/* Call dcstep to update stx, sty, and to compute the new step. */
		dcstep(&stx, &fxm, &gxm, &sty, &fym, &gym, stp, &fm, &gm, &brackt, &stmin, &stmax);
		
		/* Reset the function and derivative values for f. */
		fx = fxm + stx * gtest;
		fy = fym + sty * gtest;
		gx = gxm + gtest;
		gy = gym + gtest;
	}
	else
	{
		/* Call dcstep to update stx, sty, and to compute the new step. */
		dcstep(&stx, &fx, &gx, &sty, &fy, &gy, stp, f, g, &brackt, &stmin, &stmax);
	}
	
	/* Decide if a bisection step is needed. */
	if (brackt)
	{
		if (fabs(sty - stx) >= width1 * .66)
		{
			*stp = stx + (sty - stx) * .5;
		}
		width1 = width;
		width = fabs(sty - stx);
	}
	
	/* Set the minimum and maximum steps allowed for stp. */
	if (brackt)
	{
		stmin = fmin(stx,sty);
		stmax = fmax(stx,sty);
	}
	else
	{
		stmin = *stp + (*stp - stx) * 1.1;
		stmax = *stp + (*stp - stx) * 4.;
	}
	
	/* Force the step to be within the bounds stpmax and stpmin. */
	*stp = fmax(*stp, stpmin);
	*stp = fmin(*stp, stpmax);
	
	/* If further progress is not possible, let stp be the best */
	/* point obtained during the search. */
	if ((brackt && (*stp <= stmin || *stp >= stmax)) || (brackt && stmax - stmin <= xtol * stmax))
	{
		*stp = stx;
	}
	
	/* Obtain another function and derivative. */
	*task = FG;
L1000:
	/* Save local variables. */
	if (brackt)
	{
		isave[1] = 1;
	}
	else
	{
		isave[1] = 0;
	}
	isave[2] = stage;
	dsave[1] = ginit;
	dsave[2] = gtest;
	dsave[3] = gx;
	dsave[4] = gy;
	dsave[5] = finit;
	dsave[6] = fx;
	dsave[7] = fy;
	dsave[8] = stx;
	dsave[9] = sty;
	dsave[10] = stmin;
	dsave[11] = stmax;
	dsave[12] = width;
	dsave[13] = width1;
	
	return;
} // end of dcsrch function */
/* ====================== The end of dcsrch ============================== */

void dcstep(double* stx, double* fx, double* dx, double* sty, double* fy, double* dy, double* stp, double* fp, double* dp, long int* brackt, double* stpmin, double* stpmax)
{
	/*
	**********

	Subroutine dcstep

	This subroutine computes a safeguarded step for a search
	procedure and updates an interval that contains a step that
	satisfies a sufficient decrease and a curvature condition.

	The parameter stx contains the step with the least function
	value. If brackt is set to .true. then a minimizer has
	been bracketed in an interval with endpoints stx and sty.
	The parameter stp contains the current step. 
	The subroutine assumes that if brackt is set to .true. then

	      min(stx,sty) < stp < max(stx,sty),

	and that the derivative at stx is negative in the direction 
	of the step.

	The subroutine statement is

	  subroutine dcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,
	                    stpmin,stpmax)

	where

	  stx is a double precision variable.
	    On entry stx is the best step obtained so far and is an
	       endpoint of the interval that contains the minimizer. 
	    On exit stx is the updated best step.

	  fx is a double precision variable.
	    On entry fx is the function at stx.
	    On exit fx is the function at stx.

	  dx is a double precision variable.
	    On entry dx is the derivative of the function at 
	       stx. The derivative must be negative in the direction of 
	       the step, that is, dx and stp - stx must have opposite 
	       signs.
	    On exit dx is the derivative of the function at stx.

	  sty is a double precision variable.
	    On entry sty is the second endpoint of the interval that 
	       contains the minimizer.
	    On exit sty is the updated endpoint of the interval that 
	       contains the minimizer.

	  fy is a double precision variable.
	    On entry fy is the function at sty.
	    On exit fy is the function at sty.

	  dy is a double precision variable.
	    On entry dy is the derivative of the function at sty.
	    On exit dy is the derivative of the function at the exit sty.

	  stp is a double precision variable.
	    On entry stp is the current step. If brackt is set to .true.
	       then on input stp must be between stx and sty. 
	    On exit stp is a new trial step.

	  fp is a double precision variable.
	    On entry fp is the function at stp
	    On exit fp is unchanged.

	  dp is a double precision variable.
	    On entry dp is the the derivative of the function at stp.
	    On exit dp is unchanged.

	  brackt is an logical variable.
	    On entry brackt specifies if a minimizer has been bracketed.
	       Initially brackt must be set to .false.
	    On exit brackt specifies if a minimizer has been bracketed.
	       When a minimizer is bracketed brackt is set to .true.

	  stpmin is a double precision variable.
	    On entry stpmin is a lower bound for the step.
	    On exit stpmin is unchanged.

	  stpmax is a double precision variable.
	    On entry stpmax is an upper bound for the step.
	    On exit stpmax is unchanged.

	MINPACK-1 Project. June 1983
	Argonne National Laboratory. 
	Jorge J. More' and David J. Thuente.

	MINPACK-2 Project. October 1993.
	Argonne National Laboratory and University of Minnesota. 
	Brett M. Averick and Jorge J. More'.

	**********
	*/
	
	/* Local variables */
	static double p, q, r__, s, sgnd, stpc, stpf, stpq, gamma, theta;

	sgnd = *dp * (*dx / fabs(*dx));
	
	if (*fp > *fx)
	{
		/* First case: A higher function value. The minimum is bracketed. */
		/* If the cubic step is closer to stx than the quadratic step, the */
		/* cubic step is taken, otherwise the average of the cubic and */
		/* quadratic steps is taken. */
		
		theta = (*fx - *fp) * 3. / (*stp - *stx) + *dx + *dp;
		s = fmax(fmax(fabs(theta), fabs(*dx)), fabs(*dp));
		gamma = s * sqrt((theta / s) * (theta / s) - *dx / s * (*dp / s));
		
		if (*stp < *stx)
		{
			gamma = -gamma;
		}
		
		p = gamma - *dx + theta;
		q = gamma - *dx + gamma + *dp;
		r__ = p / q;
		stpc = *stx + r__ * (*stp - *stx);
		stpq = *stx + *dx / ((*fx - *fp) / (*stp - *stx) + *dx) / 2. * (*stp - *stx);
		
		if (fabs(stpc - *stx) < fabs(stpq - *stx))
		{
			stpf = stpc;
		}
		else
		{
			stpf = stpc + (stpq - stpc) / 2.;
		}
		*brackt = 1;
	}
	else if (sgnd < 0.)
	{
		/* Second case: A lower function value and derivatives of opposite */
		/* sign. The minimum is bracketed. If the cubic step is farther from */
		/* stp than the secant step, the cubic step is taken, otherwise the */
		/* secant step is taken. */
		
		theta = (*fx - *fp) * 3. / (*stp - *stx) + *dx + *dp;
		s = fmax(fmax(fabs(theta), fabs(*dx)), fabs(*dp));
		gamma = s * sqrt((theta / s) * (theta / s) - *dx / s * (*dp / s));
		if (*stp > *stx)
		{
			gamma = -gamma;
		}
		p = gamma - *dp + theta;
		q = gamma - *dp + gamma + *dx;
		r__ = p / q;
		stpc = *stp + r__ * (*stx - *stp);
		stpq = *stp + *dp / (*dp - *dx) * (*stx - *stp);
		
		if (fabs(stpc - *stp) > fabs(stpq - *stp))
		{
			stpf = stpc;
		}
		else
		{
			stpf = stpq;
		}
		*brackt = 1;
	}
	else if (fabs(*dp) < fabs(*dx))
	{
		/* Third case: A lower function value, derivatives of the same sign, */
		/* and the magnitude of the derivative decreases. */
		
		/* The cubic step is computed only if the cubic tends to infinity */
		/* in the direction of the step or if the minimum of the cubic */
		/* is beyond stp. Otherwise the cubic step is defined to be the */
		/* secant step. */
		
		theta = (*fx - *fp) * 3. / (*stp - *stx) + *dx + *dp;
		s = fmax(fmax(fabs(theta), fabs(*dx)), fabs(*dp));
		
		/* The case gamma = 0 only arises if the cubic does not tend */
		/* to infinity in the direction of the step. */

		gamma = s * sqrt((fmax(0., (theta / s) * (theta / s) - *dx / s * (*dp / s))));
		
		if (*stp > *stx)
		{
			gamma = -gamma;
		}
		p = gamma - *dp + theta;
		q = gamma + (*dx - *dp) + gamma;
		r__ = p / q;
		
		if (r__ < 0. && gamma != 0.)
		{
			stpc = *stp + r__ * (*stx - *stp);
		}
		else if (*stp > *stx)
		{
			stpc = *stpmax;
		}
		else
		{
			stpc = *stpmin;
		}
		
		stpq = *stp + *dp / (*dp - *dx) * (*stx - *stp);
		if (*brackt)
		{
			/* A minimizer has been bracketed. If the cubic step is */
			/* closer to stp than the secant step, the cubic step is */
			/* taken, otherwise the secant step is taken. */
			
			if (fabs(stpc - *stp) < fabs(stpq - *stp))
			{
				stpf = stpc;
			}
			else
			{
				stpf = stpq;
			}
			
			if (*stp > *stx)
			{
				stpf = fmin(*stp + (*sty - *stp) * .66, stpf);
			}
			else
			{
				stpf = fmax(*stp + (*sty - *stp) * .66, stpf);
			}
		}
		else
		{
			/* A minimizer has not been bracketed. If the cubic step is */
			/* farther from stp than the secant step, the cubic step is */
			/* taken, otherwise the secant step is taken. */
			
			if (fabs(stpc - *stp) > fabs(stpq - *stp))
			{
				stpf = stpc;
			}
			else
			{
				stpf = stpq;
			}
			stpf = fmin(*stpmax,stpf);
			stpf = fmax(*stpmin,stpf);
		}
	}
	else
	{
		/* Fourth case: A lower function value, derivatives of the same sign, */
		/* and the magnitude of the derivative does not decrease. If the */
		/* minimum is not bracketed, the step is either stpmin or stpmax, */
		/* otherwise the cubic step is taken. */
		
		if (*brackt)
		{
			theta = (*fp - *fy) * 3. / (*sty - *stp) + *dy + *dp;
			s = fmax(fmax(fabs(theta), fabs(*dy)), fabs(*dp));
			gamma = s * sqrt((theta / s) * (theta / s) - *dy / s * (*dp / s));
			
			if (*stp > *sty)
			{
				gamma = -gamma;
			}
			p = gamma - *dp + theta;
			q = gamma - *dp + gamma + *dy;
			r__ = p / q;
			stpc = *stp + r__ * (*sty - *stp);
			stpf = stpc;
		}
		else if (*stp > *stx)
		{
			stpf = *stpmax;
		}
		else
		{
			stpf = *stpmin;
		}
	}
	
	/* Update the interval which contains a minimizer. */
	if (*fp > *fx)
	{
		*sty = *stp;
		*fy = *fp;
		*dy = *dp;
	}
	else
	{
		if (sgnd < 0.)
		{
			*sty = *stx;
			*fy = *fx;
			*dy = *dx;
		}
		*stx = *stp;
		*fx = *fp;
		*dx = *dp;
	}
	
	/* Compute the new step. */
	*stp = stpf;
	
	return;
} // end of dcstep function */

void daxpy(long int n, double da, double* dx, long int incx, double* dy, long int incy)
{
	/* Local variables */
	static long int i, m, ix, iy, mp1;

	/* Constant times a vector plus a vector.
	   Uses unrolled loops for increments equal to one. */
	if (n <= 0)
	{
		return;
	}
	
	if (da == 0.)
	{
		return;
	}
	
	if (incx == 1 && incy == 1)
	{
		goto L20;
	}

	/* Code for unequal increments or equal increments not equal to 1 */
	ix = 1;
	iy = 1;
	
	if (incx < 0)
	{
		ix = (-n + 1) * incx + 1;
	}
	
	if (incy < 0)
	{
		iy = (-n + 1) * incy + 1;
	}
	
	for (i = 0; i < n; ++i)
	{
		dy[iy] += da * dx[ix];
		ix += incx;
		iy += incy;
	}
	
	return;

	/* Code for both increments equal to 1 clean-up loop */
L20:
	m = n % 4;
	if (m == 0)
	{
		goto L40;
	}
	
	for (i = 0; i < m; ++i)
	{
		dy[i] += da * dx[i];
	}
	
	if (n < 4)
	{
		return;
	}
L40:
	mp1 = m;
	for (i = mp1; i < n; i += 4)
	{
		dy[i] += da * dx[i];
		dy[i + 1] += da * dx[i + 1];
		dy[i + 2] += da * dx[i + 2];
		dy[i + 3] += da * dx[i + 3];
	}
	
	return;
} // end of daxpy function */

void dcopy(long int n, double* dx, long int incx, double* dy, long int incy)
{
	/* Local variables */
	static long int i, m, ix, iy, mp1;

	/* Copies a vector, x, to a vector, y. 
	   Uses unrolled loops for increments equal to one. */
	if (n <= 0)
	{
		return;
	}
	
	if (incx == 1 && incy == 1)
	{
		goto L20;
	}

	/* Code for unequal increments or equal increments not equal to 1 */
	ix = 1;
	iy = 1;
	
	if (incx < 0)
	{
		ix = (-n + 1) * incx + 1;
	}
	
	if (incy < 0)
	{
		iy = (-n + 1) * incy + 1;
	}
	
	for (i = 0; i < n; ++i)
	{
		dy[iy] = dx[ix];
		ix += incx;
		iy += incy;
	}
	
	return;

	/* Code for both increments equal to 1 clean-up loop */

L20:
	m = n % 7;
	if (m == 0)
	{
		goto L40;
	}
	
	for (i = 0; i < m; ++i)
	{
		dy[i] = dx[i];
	}
	
	if (n < 7)
	{
		return;
	}
L40:
	mp1 = m;
	for (i = mp1; i < n; i += 7)
	{
		dy[i] = dx[i];
		dy[i + 1] = dx[i + 1];
		dy[i + 2] = dx[i + 2];
		dy[i + 3] = dx[i + 3];
		dy[i + 4] = dx[i + 4];
		dy[i + 5] = dx[i + 5];
		dy[i + 6] = dx[i + 6];
	}
	
	return;
} // end of dcopy function */

double ddot(long int n, double* dx, long int incx, double* dy, long int incy)
{
	/* System generated locals */
	double ret_val;

	/* Local variables */
	static long int i, m;
	static double dtemp;
	static long int ix, iy, mp1;

	/* Forms the dot product of two vectors. 
	   Uses unrolled loops for increments equal to one. */
	ret_val = 0.;
	dtemp = 0.;
	if (n <= 0)
	{
		return ret_val;
	}
	
	if (incx == 1 && incy == 1)
	{
		goto L20;
	}

	/* Code for unequal increments or equal increments not equal to 1 */
	ix = 1;
	iy = 1;
	
	if (incx < 0)
	{
		ix = (-n + 1) * incx + 1;
	}
	
	if (incy < 0)
	{
		iy = (-n + 1) * incy + 1;
	}
	
	for (i = 0; i < n; ++i)
	{
		dtemp += dx[ix] * dy[iy];
		ix += incx;
		iy += incy;
	}
	ret_val = dtemp;
	
	return ret_val;

	/* Code for both increments equal to 1 clean-up loop */

L20:
	m = n % 5;
	if (m == 0)
	{
		goto L40;
	}
	
	for (i = 0; i < m; ++i)
	{
		dtemp += dx[i] * dy[i];
	}
	
	if (n < 5)
	{
		goto L60;
	}
L40:
	mp1 = m;
	for (i = mp1; i < n; i += 5)
	{
		dtemp = dtemp + dx[i] * dy[i] + dx[i + 1] * dy[i + 1] + dx[i + 2] * dy[i + 2] + dx[i + 3] * dy[i + 3] + dx[i + 4] * dy[i + 4];
	}
L60:
	ret_val = dtemp;
	
	return ret_val;
} // end of ddot function */

void dscal(long int n, double da, double* dx, long int incx)
{
	/* Local variables */
	static long int i, m, nincx, mp1;

	/* Scales a vector by a constant. 
	   Uses unrolled loops for increment equal to one. */
	if (n <= 0 || incx <= 0)
	{
		return;
	}
	
	if (incx == 1)
	{
		goto L20;
	}

	/* Code for increment not equal to 1 */
	nincx = n * incx;
	for (i = 0; incx < 0 ? i > nincx : i < nincx; i += incx)
	{
		dx[i] = da * dx[i];
	}
	
	return;

	/* Code for increment equal to 1 clean-up loop */
L20:
	m = n % 5;
	if (m == 0)
	{
		goto L40;
	}
	
	for (i = 0; i < m; ++i)
	{
		dx[i] = da * dx[i];
	}
	
	if (n < 5)
	{
		return;
	}
L40:
	mp1 = m;
	for (i = mp1; i < n; i += 5)
	{
		dx[i] = da * dx[i];
		dx[i + 1] = da * dx[i + 1];
		dx[i + 2] = da * dx[i + 2];
		dx[i + 3] = da * dx[i + 3];
		dx[i + 4] = da * dx[i + 4];
	}
	
	return;
} // end of dscal function */

void dpofa(double* a, long int lda, long int n, long int* info)
{
	/*
	**********

	Subroutine dpofa

	dpofa factors a double precision symmetric positive definite
	matrix.

	dpofa is usually called by dpoco, but it can be called
	directly with a saving in time if  rcond  is not needed.
	(time for dpoco) = (1 + 18/n)*(time for dpofa) .

	on entry

	   a       double precision(lda, n)
	           the symmetric matrix to be factored.  only the
	           diagonal and upper triangle are used.

	   lda     integer
	           the leading dimension of the array  a .

	   n       integer
	           the order of the matrix  a .

	on return

	   a       an upper triangular matrix  r  so that  a = trans(r)*r
	           where  trans(r)  is the transpose.
	           the strict lower triangle is unaltered.
	           if  info .ne. 0 , the factorization is not complete.

	   info    integer
	           = 0  for normal return.
	           = k  signals an error condition.  the leading minor
	                of order  k  is not positive definite.

	**********
	*/
	
	/* System generated locals */
	long int a_dim1, a_offset;

	/* Local variables */
	static long int j, k;
	static double s, t;
	static long int jm1;

	/* Parameter adjustments */
	a_dim1 = lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;

	/* Function Body */
	for (j = 1; j <= n; ++j)
	{
		*info = j;
		s = 0.;
		jm1 = j - 1;
		if (jm1 < 1)
		{
			goto L20;
		}

		for (k = 1; k <= jm1; ++k)
		{
			t = a[k + j * a_dim1] - ddot(k - 1, &a[k * a_dim1 + 1], 1, &a[j * a_dim1 + 1], 1);
			t /= a[k + k * a_dim1];
			a[k + j * a_dim1] = t;
			s += t * t;
		}
L20:
		s = a[j + j * a_dim1] - s;
		if (s <= 0.)
		{
			goto L40;
		}
		a[j + j * a_dim1] = sqrt(s);
	}
	*info = 0;
L40:
	return;
} // end of dpofa function */
/* ====================== The end of dpofa =============================== */

void dtrsl(double* t, long int ldt, long int n, double* b, long int job, long int* info)
{
	/*
	**********

	Subroutine dtrsl

	dtrsl solves systems of the form

	              t * x = b
	or
	              trans(t) * x = b

	where t is a triangular matrix of order n. here trans(t)
	denotes the transpose of the matrix t.

	on entry

	    t         double precision(ldt,n)
	              t contains the matrix of the system. the zero
	              elements of the matrix are not referenced, and
	              the corresponding elements of the array can be
	              used to store other information.

	    ldt       integer
	              ldt is the leading dimension of the array t.

	    n         integer
	              n is the order of the system.

	    b         double precision(n).
	              b contains the right hand side of the system.

	    job       integer
	              job specifies what kind of system is to be solved.
	              if job is

	                   00   solve t * x = b, t lower triangular,
	                   01   solve t * x = b, t upper triangular,
	                   10   solve trans(t) * x = b, t lower triangular,
	                   11   solve trans(t) * x = b, t upper triangular.

	on return

	    b         b contains the solution, if info .eq. 0.
	              otherwise b is unaltered.

	    info      integer
	              info contains zero if the system is nonsingular.
	              otherwise info contains the index of
	              the first zero diagonal element of t.

	**********
	*/
	
	/* System generated locals */
	long int t_dim1, t_offset;

	/* Local variables */
	static long int j, jj, case__;
	static double temp;

	/* Check for zero diagonal elements. */

	/* Parameter adjustments */
	t_dim1 = ldt;
	t_offset = 1 + t_dim1;
	t -= t_offset;
	--b;

	for (*info = 1; *info <= n; ++(*info))
	{
		if (t[*info + *info * t_dim1] == 0.)
		{
			goto L150;
		}
	}
	*info = 0;

	/* Determine the task and go to it. */
	case__ = 1;
	if (job % 10 != 0)
	{
		case__ = 2;
	}
	
	if (job % 100 / 10 != 0)
	{
		case__ += 2;
	}
	
	switch (case__)
	{
		case 1:  goto L20;
		case 2:  goto L50;
		case 3:  goto L80;
		case 4:  goto L110;
	}

	/* Solve t * x = b for t lower triangular */
L20:
	b[1] /= t[t_dim1 + 1];
	if (n < 2)
	{
		goto L40;
	}
	
	for (j = 2; j <= n; ++j)
	{
		temp = -b[j - 1];
		daxpy(n - j + 1, temp, &t[j + (j - 1) * t_dim1], 1, &b[j], 1);
		b[j] /= t[j + j * t_dim1];
	}
L40:
	goto L140;

	/* Solve t * x = b for t upper triangular. */
L50:
	b[n] /= t[n + n * t_dim1];
	if (n < 2)
	{
		goto L70;
	}

	for (jj = 2; jj <= n; ++jj)
	{
		j = n - jj + 1;
		temp = -b[j + 1];
		daxpy(j, temp, &t[(j + 1) * t_dim1 + 1], 1, &b[1], 1);
		b[j] /= t[j + j * t_dim1];
	}
L70:
	goto L140;

	/* Solve trans(t) * x = b for t lower triangular. */
L80:
	b[n] /= t[n + n * t_dim1];
	if (n < 2)
	{
		goto L100;
	}

	for (jj = 2; jj <= n; ++jj)
	{
		j = n - jj + 1;
		b[j] -= ddot(jj - 1, &t[j + 1 + j * t_dim1], 1, &b[j + 1], 1);
		b[j] /= t[j + j * t_dim1];
	}
L100:
	goto L140;

	/* Solve trans(t) * x = b for t upper triangular. */
L110:
	b[1] /= t[t_dim1 + 1];
	if (n < 2)
	{
		goto L130;
	}

	for (j = 2; j <= n; ++j)
	{
		b[j] -= ddot(j - 1, &t[j * t_dim1 + 1], 1, &b[1], 1);
		b[j] /= t[j + j * t_dim1];
	}
L130:
L140:
L150:
	return;
} // end of dtrsl function */

void timer(double *ttime)
{
	/* This routine computes cpu time */
	
    clock_t temp;

    temp    = clock();
    *ttime  = ((double) temp) / CLOCKS_PER_SEC;
	
    return;
} // end of timer function */