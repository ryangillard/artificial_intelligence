#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

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

/* This function efficientK_inv_y recombines a lower cholesky decomposition inverse A^-1 = L^-1 * L^-T */
void RecombineLowerCholeskyDecompositionInverse(unsigned int n, double** L, double** A);

/* This function optimizes kernel hyperparameters */
void OptimizeKernelHyperparameters(int kernel_type, unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** y, double** kernel_x_x, double** L, double** K_inv_y, unsigned int num_kernel_hyperparameters, double* kernel_hyperparameters);

/* This function reads the kernel hyperparameter optimization parameters */
void ReadKernelHyperparameterOptimizationParameters(double* gradient_tolerance, unsigned int* max_iterations, double* learning_rate, unsigned int num_kernel_hyperparameters, int*** kernel_hyperparameter_bounds_exists, double*** kernel_hyperparameter_bounds_values);

/* This function performs the kernel hyperparameter optimzation loop */
void KernelHyperparameterOptimizerLoop(int kernel_type, double gradient_tolerance, unsigned int max_iterations, double learning_rate, unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** y, double** kernel_x_x, double** L, double** K_inv_y, double** alpha_alpha_t, double** kernel_x_x_inv, double** d_kernel_d_kernel_hyperparameter, double** d_kernel_d_kernel_hyperparameter_temp, unsigned int num_kernel_hyperparameters, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters);

/* This function updates the constant hyperparameter of the linear kernel */
double UpdateLinearKernelConstantHyperparameter(unsigned int num_training_points, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double learning_rate, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters, double gnorm);

/* This function updates the length-scale hyperparameter of the squared exponential kernel */
double UpdateSquaredExponentiaK_inv_k_starernelLengthScaleHyperparameter(unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** kernel_x_x, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double** d_kernel_d_kernel_hyperparameter_temp, double learning_rate, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters, double gnorm);

/* This function updates the signal variance hyperparameter of the squared exponential kernel */
double UpdateSquaredExponentiaK_inv_k_starernelSignalVarianceHyperparameter(unsigned int num_training_points, double** kernel_x_x, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double learning_rate, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters, double gnorm);

/* This function updates the noise variance hyperparameter of the squared exponential kernel */
double UpdateSquaredExponentiaK_inv_k_starernelNoiseVarianceHyperparameter(unsigned int num_training_points, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double learning_rate, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters, double gnorm);

/* This function updates a kernel hyperparameter with gradient ascent */
double GradientAscentKernelHyperparameter(unsigned int num_training_points, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double learning_rate, unsigned int hyperparameter_index, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters, double gnorm);

/* This function uses the iterative conjugate gradient method that solves A * x = b */
void ConjugateGradientMethod(unsigned int n, double** A, double** b, double** x);

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

/* This function efficientK_inv_y recombines a lower cholesky decomposition inverse A^-1 = L^-1 * L^-T */
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
	
	/* Read kernel hyperparameter optimization parameters */
	double gradient_tolerance = 0.0, learning_rate = 0.0;
	unsigned int max_iterations = 0;
	
	int** kernel_hyperparameter_bounds_exists;
	kernel_hyperparameter_bounds_exists = malloc(sizeof(int*) * num_kernel_hyperparameters);
	for (i = 0; i < num_kernel_hyperparameters; i++)
	{
		kernel_hyperparameter_bounds_exists[i] = malloc(sizeof(int) * 2);
		kernel_hyperparameter_bounds_exists[i][0] = 0;
		kernel_hyperparameter_bounds_exists[i][1] = 0;
	} // end of i loop
	
	double** kernel_hyperparameter_bounds_values;
	kernel_hyperparameter_bounds_values = malloc(sizeof(double*) * num_kernel_hyperparameters);
	for (i = 0; i < num_kernel_hyperparameters; i++)
	{
		kernel_hyperparameter_bounds_values[i] = malloc(sizeof(double) * 2);
		kernel_hyperparameter_bounds_values[i][0] = 0.0;
		kernel_hyperparameter_bounds_values[i][1] = 0.0;
	} // end of i loop
	
	ReadKernelHyperparameterOptimizationParameters(&gradient_tolerance, &max_iterations, &learning_rate, num_kernel_hyperparameters, &kernel_hyperparameter_bounds_exists, &kernel_hyperparameter_bounds_values);
	
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

	/* Calculate K(X, X) inverse */
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
	
	/* Create extra array that MAY be needed to hold intermediate results depending on kernel type */
	double** d_kernel_d_kernel_hyperparameter_temp;
	
	if (kernel_type == 0) // linear
	{
		KernelHyperparameterOptimizerLoop(kernel_type, gradient_tolerance, max_iterations, learning_rate, num_training_points, num_dimensions, X_train, y, kernel_x_x, L, K_inv_y, alpha_alpha_t, kernel_x_x_inv, d_kernel_d_kernel_hyperparameter, d_kernel_d_kernel_hyperparameter_temp, num_kernel_hyperparameters, kernel_hyperparameter_bounds_exists, kernel_hyperparameter_bounds_values, kernel_hyperparameters);
	} // end of linear
	else // squared exponential
	{
		/* Allocate extra array that will be needed to hold intermediate results */
		d_kernel_d_kernel_hyperparameter_temp = malloc(sizeof(double*) * num_training_points);
		for (i = 0; i < num_training_points; i++)
		{
			d_kernel_d_kernel_hyperparameter_temp[i] = malloc(sizeof(double) * num_training_points);
			for (j = 0; j < num_training_points; j++)
			{
				d_kernel_d_kernel_hyperparameter_temp[i][j] = 0.0;
			} // end of j loop
		} // end of i loop
		
		KernelHyperparameterOptimizerLoop(kernel_type, gradient_tolerance, max_iterations, learning_rate, num_training_points, num_dimensions, X_train, y, kernel_x_x, L, K_inv_y, alpha_alpha_t, kernel_x_x_inv, d_kernel_d_kernel_hyperparameter, d_kernel_d_kernel_hyperparameter_temp, num_kernel_hyperparameters, kernel_hyperparameter_bounds_exists, kernel_hyperparameter_bounds_values, kernel_hyperparameters);
		
		/* Free dynamic memory */
		for (i = 0; i < num_training_points; i++)
		{
			free(d_kernel_d_kernel_hyperparameter_temp[i]);
		} // end of i loop
		free(d_kernel_d_kernel_hyperparameter_temp);
	} // end of squared exponential
	
	/* Free dynamic memory */
	for (i = 0; i < num_training_points; i++)
	{
		free(d_kernel_d_kernel_hyperparameter[i]);
		free(kernel_x_x_inv[i]);
		free(alpha_alpha_t[i]);
	} // end of i loop
	free(d_kernel_d_kernel_hyperparameter);
	free(kernel_x_x_inv);
	free(alpha_alpha_t);
	
	for (i = 0; i < num_kernel_hyperparameters; i++)
	{
		free(kernel_hyperparameter_bounds_exists[i]);
		free(kernel_hyperparameter_bounds_values[i]);
	} // end of i loop
	free(kernel_hyperparameter_bounds_exists);
	free(kernel_hyperparameter_bounds_values);
	
	return;
} // end of OptimizeKernelHyperparameters function

/* This function reads the kernel hyperparameter optimization parameters */
void ReadKernelHyperparameterOptimizationParameters(double* gradient_tolerance, unsigned int* max_iterations, double* learning_rate, unsigned int num_kernel_hyperparameters, int*** kernel_hyperparameter_bounds_exists, double*** kernel_hyperparameter_bounds_values)
{
	unsigned int i;
	int system_return = 0;
	
	/* Get the gradient tolerance */
	FILE* infile_gradient_tolerance = fopen("inputs/gradient_tolerance.txt", "r");
	system_return = fscanf(infile_gradient_tolerance, "%lf", gradient_tolerance);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/gradient_tolerance.txt\n");
	}
	fclose(infile_gradient_tolerance);
	printf("gradient_tolerance = %lf\n", (*gradient_tolerance));
	
	/* Get the maxmium number of iterations */
	FILE* infile_max_iterations = fopen("inputs/max_iterations.txt", "r");
	system_return = fscanf(infile_max_iterations, "%u", max_iterations);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/max_iterations.txt\n");
	}
	fclose(infile_max_iterations);
	printf("max_iterations = %u\n", (*max_iterations));
	
	/* Get the learning rate */
	FILE* infile_learning_rate = fopen("inputs/learning_rate.txt", "r");
	system_return = fscanf(infile_learning_rate, "%lf", learning_rate);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/learning_rate.txt\n");
	}
	fclose(infile_learning_rate);
	printf("learning_rate = %lf\n", (*learning_rate));
	
	/* Get hyperparameter bounds existence */
	FILE* infile_kernel_hyperparameter_bounds_exists = fopen("inputs/kernel_hyperparameter_bounds_exists.txt", "r");
	for (i = 0; i < num_kernel_hyperparameters; i++)
	{
		system_return = fscanf(infile_kernel_hyperparameter_bounds_exists, "%d\t%d\n", &(*kernel_hyperparameter_bounds_exists)[i][0], &(*kernel_hyperparameter_bounds_exists)[i][1]);
		if (system_return == -1)
		{
			printf("Failed reading file inputs/kernel_hyperparameter_bounds_exists.txt\n");
		}
		printf("i = %u, kernel_hyperparameter_bounds_exists[i][0] = %d, kernel_hyperparameter_bounds_exists[i][1] = %d\n", i, (*kernel_hyperparameter_bounds_exists)[i][0], (*kernel_hyperparameter_bounds_exists)[i][1]);
	} // end of i loop
	fclose(infile_kernel_hyperparameter_bounds_exists);
	
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
void KernelHyperparameterOptimizerLoop(int kernel_type, double gradient_tolerance, unsigned int max_iterations, double learning_rate, unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** y, double** kernel_x_x, double** L, double** K_inv_y, double** alpha_alpha_t, double** kernel_x_x_inv, double** d_kernel_d_kernel_hyperparameter, double** d_kernel_d_kernel_hyperparameter_temp, unsigned int num_kernel_hyperparameters, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters)
{
	unsigned int i, j, k = 0;
	int error = 0;
	double gnorm = DBL_MAX, log_marginal_likelihood = 0.0;
	
	while (gnorm > gradient_tolerance && k < max_iterations)
	{
		/* Reset gradient infinity norm */
		gnorm = 0.0;
		
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
		log_marginal_likelihood = CalculateLogMarginalLikelihood(num_training_points, L, K_inv_y);
		printf("\nk = %u, log_marginal_likelihood = %lf\n", k, log_marginal_likelihood);

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
			UpdateLinearKernelConstantHyperparameter(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, learning_rate, kernel_hyperparameter_bounds_exists, kernel_hyperparameter_bounds_values, kernel_hyperparameters, gnorm);
		} // end of linear
		else // squared exponential
		{
			/* Length-scale */
			gnorm = UpdateSquaredExponentiaK_inv_k_starernelLengthScaleHyperparameter(num_training_points, num_dimensions, X_train, kernel_x_x, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, d_kernel_d_kernel_hyperparameter_temp, learning_rate, kernel_hyperparameter_bounds_exists, kernel_hyperparameter_bounds_values, kernel_hyperparameters, gnorm);
			
			/* Signal variance */
			gnorm = UpdateSquaredExponentiaK_inv_k_starernelSignalVarianceHyperparameter(num_training_points, kernel_x_x, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, learning_rate, kernel_hyperparameter_bounds_exists, kernel_hyperparameter_bounds_values, kernel_hyperparameters, gnorm);
			
			/* Noise variance */
			gnorm = UpdateSquaredExponentiaK_inv_k_starernelNoiseVarianceHyperparameter(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, learning_rate, kernel_hyperparameter_bounds_exists, kernel_hyperparameter_bounds_values, kernel_hyperparameters, gnorm);
		} // end of squared exponential
		
		/* Increment iteration count */
		printf("gnorm = %lf at iteration %u\n", gnorm, k);
		printf("kernel_hyperparameters = \n");
		for (i = 0; i < num_kernel_hyperparameters; i++)
		{
				printf("%u\t%.16f\n", i, kernel_hyperparameters[i]);
		} // end of i loop
		
		k++;
	} // end of while loop
	
	return;
} // end of KernelHyperparameterOptimizerLoop function

/* This function updates the constant hyperparameter of the linear kernel */
double UpdateLinearKernelConstantHyperparameter(unsigned int num_training_points, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double learning_rate, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters, double gnorm)
{
	unsigned int i, j;
	
	for (i = 0; i < num_training_points; i++)
	{
		for (j = 0; j < num_training_points; j++)
		{
			d_kernel_d_kernel_hyperparameter[i][j] = kernel_hyperparameters[0];
		} // end of j loop
	} // end of i loop
	
	gnorm = GradientAscentKernelHyperparameter(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, learning_rate, 0, kernel_hyperparameter_bounds_exists, kernel_hyperparameter_bounds_values, kernel_hyperparameters, gnorm);
	
	return gnorm;
} // end of UpdateLinearKernelConstantHyperparameter function

/* This function updates the length-scale hyperparameter of the squared exponential kernel */
double UpdateSquaredExponentiaK_inv_k_starernelLengthScaleHyperparameter(unsigned int num_training_points, unsigned int num_dimensions, double** X_train, double** kernel_x_x, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double** d_kernel_d_kernel_hyperparameter_temp, double learning_rate, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters, double gnorm)
{
	unsigned int i, j, k;
	double a_squared_sum = 0.0, b_squared_sum = 0.0;
	
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
	
	gnorm = GradientAscentKernelHyperparameter(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, learning_rate, 0, kernel_hyperparameter_bounds_exists, kernel_hyperparameter_bounds_values, kernel_hyperparameters, gnorm);
	
	return gnorm;
} // end of UpdateSquaredExponentiaK_inv_k_starernelLengthScaleHyperparameter function

/* This function updates the signal variance hyperparameter of the squared exponential kernel */
double UpdateSquaredExponentiaK_inv_k_starernelSignalVarianceHyperparameter(unsigned int num_training_points, double** kernel_x_x, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double learning_rate, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters, double gnorm)
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
	
	gnorm = GradientAscentKernelHyperparameter(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, learning_rate, 1, kernel_hyperparameter_bounds_exists, kernel_hyperparameter_bounds_values, kernel_hyperparameters, gnorm);
	
	return gnorm;
} // end of UpdateSquaredExponentiaK_inv_k_starernelSignalVarianceHyperparameter function

/* This function updates the noise variance hyperparameter of the squared exponential kernel */
double UpdateSquaredExponentiaK_inv_k_starernelNoiseVarianceHyperparameter(unsigned int num_training_points, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double learning_rate, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters, double gnorm)
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
	
	gnorm = GradientAscentKernelHyperparameter(num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, learning_rate, 2, kernel_hyperparameter_bounds_exists, kernel_hyperparameter_bounds_values, kernel_hyperparameters, gnorm);

	return gnorm;
} // end of UpdateSquaredExponentiaK_inv_k_starernelNoiseVarianceHyperparameter function

/* This function updates a kernel hyperparameter with gradient ascent */
double GradientAscentKernelHyperparameter(unsigned int num_training_points, double** alpha_alpha_t, double** d_kernel_d_kernel_hyperparameter, double learning_rate, unsigned int hyperparameter_index, int** kernel_hyperparameter_bounds_exists, double** kernel_hyperparameter_bounds_values, double* kernel_hyperparameters, double gnorm)
{
	int error = 0;
	double d_log_marginal_likelihood_d_hyperparameter = 0.0, step_size = 0.0;
	
	/* Calculate d_log_marginal_likelihood_d_hyperparameter */
	error = MatrixMatrixMultiplcationTrace(num_training_points, num_training_points, num_training_points, alpha_alpha_t, d_kernel_d_kernel_hyperparameter, 0, 0, &d_log_marginal_likelihood_d_hyperparameter);
	if (error != 0)
	{
		printf("ERROR: MatrixMatrixMultiplcationTrace returned error code %d\n", error);
	}
	
	step_size = learning_rate * d_log_marginal_likelihood_d_hyperparameter;
	
	/* Check hyperparameter bounds */
	if (kernel_hyperparameters[hyperparameter_index] + step_size < kernel_hyperparameter_bounds_values[hyperparameter_index][0] && kernel_hyperparameter_bounds_exists[hyperparameter_index][0])
	{
		kernel_hyperparameters[hyperparameter_index] = kernel_hyperparameter_bounds_values[hyperparameter_index][0];
	}
	else if (kernel_hyperparameters[hyperparameter_index] + step_size > kernel_hyperparameter_bounds_values[hyperparameter_index][1] && kernel_hyperparameter_bounds_exists[hyperparameter_index][1])
	{
		kernel_hyperparameters[hyperparameter_index] = kernel_hyperparameter_bounds_values[hyperparameter_index][1];
	}
	else
	{
		kernel_hyperparameters[hyperparameter_index] += step_size;
	}
	
	if (fabs(d_log_marginal_likelihood_d_hyperparameter) > gnorm)
	{
		gnorm = fabs(d_log_marginal_likelihood_d_hyperparameter);
	}
	
	return gnorm;
} // end of GradientAscentKernelHyperparameter function

/* This function uses the iterative conjugate gradient method that solves A * x = b */
void ConjugateGradientMethod(unsigned int n, double** A, double** b, double** x)
{
	/* A = [n, n] */
	/* b = [n, 1] */
	/* x = [n, 1] */
	
	unsigned int i, j;
	
	double** r;
	r = malloc(sizeof(double*) * n);
	for (i = 0; i < n; i++)
	{
		r[i] = malloc(sizeof(double) * 1);
	} // end of i loop
	
	MatrixMatrixMultiplication(n, 1, n, A, x, 0, 0, r);
	
	for (i = 0; i < n; i++)
	{
		r[i][0] = b[i][0] - r[i][0];
	} // end of i loop
	
	double** p;
	p = malloc(sizeof(double*) * n);
	for (i = 0; i < n; i++)
	{
		p[i] = malloc(sizeof(double) * 1);
	} // end of i loop
	
	double rsold = 0.0, rsnew = 0;
	rsold = VectorDotProductRank2(n, r, r, 1, 1);
	
	double** Ap;
	Ap = malloc(sizeof(double*) * n);
	for (i = 0; i < n; i++)
	{
		Ap[i] = malloc(sizeof(double) * 1);
	} // end of i loop
	
	double alpha;
	for (i = 0; i < n; i++)
	{
		MatrixMatrixMultiplication(n, 1, n, A, p, 0, 0, Ap);
		
		alpha = rsold / VectorDotProductRank2(n, p, Ap, 1, 1);
		
		for (j = 0; j < n; j++)
		{
			x[j][0] += alpha * p[j][0];
			r[j][0] -= alpha * Ap[j][0];
		} // end of j loop
		
		rsnew = VectorDotProductRank2(n, r, r, 1, 1);
		if (sqrt(rsnew) < 1e-10)
		{
			break;
		}

		for (j = 0; j < n; j++)
		{
			p[j][0] = r[j][0] + (rsnew / rsold) * p[j][0];
		} // end of j loop
		
        rsold = rsnew;
	} // end of i loop
	
	/* Free dynamic memory */
	for (i = 0; i < n; i++)
	{
		free(Ap[i]);
		free(p[i]);
		free(r[i]);
	} // end of i loop
	free(Ap);
	free(p);
	free(r);

	return;
} // end of ConjugateGradientMethod function

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

	/* LastK_inv_y add third term, the normalizing factor: -0.5 * n * log(2 * Pi) */
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