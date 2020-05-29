/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function reads the model hyperparameters */
void ReadModelHyperparameters(unsigned int* num_training_points, unsigned int* num_test_points, unsigned int* num_dimensions, unsigned int* num_samples, int* kernel_type);

/* This function reads the intial kernel hyperparameters */
void ReadInitialKernelHyperparameters(int kernel_type, unsigned int* num_kernel_hyperparameters, double** kernel_hyperparameters);

/* This function reads the training features and targets */
void ReadTrainingData(unsigned int num_training_points, unsigned int num_dimensions, double*** X_train, double*** y);

/* This function reads in the test features */
void ReadTestData(unsigned int num_test_points, unsigned int num_dimensions, double*** X_test);

/* This function calculates various similarity kernels between two matrices */
void CalculateKernel(int kernel_type, double* kernel_hyperparameters, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double** kernel);

/* This function applies the linear kernel between two matrices */
void LinearKernel(double* kernel_hyperparameters, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double** kernel);

/* This function applies the squared exponential kernel between two matrices */
void SquaredExponentialKernel(double* kernel_hyperparameters, unsigned int a_rows, unsigned int b_rows, unsigned int a_cols, double** A, double** B, double** kernel);

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

/* This function calculates the negative log likelihood from training data */
double CalculateNegativeLogLikelihoodFromTrainData(int kernel_type, unsigned int num_training_points, unsigned int num_dimensions, double* kernel_hyperparameters, double** X_train, double** y, double** kernel_x_x, double** L, double** K_inv_y, unsigned int* n_function_evals);

/* This function calculates an approximation of the gradient of the negative log likelihood */
void CalculateNegativeLogLikelihoodFPrime(int kernel_type, unsigned int num_training_points, unsigned int num_dimensions, unsigned int num_kernel_hyperparameters, double* kernel_hyperparameters, double** X_train, double** y, double** kernel_x_x, double** L, double** K_inv_y, double epsilon, double f0, unsigned int* n_function_evals, double* g);

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
double CalculateLogMarginalLikelihood(unsigned int num_training_points, double** y, double** L, double** K_inv_y);

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void);

/* This function returns a random normal number with given mean and standard deviation */
double RNorm(double mu, double sigma);

/* This function returns a random normal number with zero mean and unit standard deviation */
double NormRand(void);