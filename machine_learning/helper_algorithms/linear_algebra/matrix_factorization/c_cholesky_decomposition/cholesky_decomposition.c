#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*********************************************************************************************************/
/********************************************** PROTOTYPES ***********************************************/
/*********************************************************************************************************/

/* This function computes the Cholesky factorization of a real symmetric positive definite matrix A using the recursive algorithm. */
int RecursiveCholeskyFactorization(int n, double** a, int a_row_offset, int a_col_offset);

/* This function solves the matrix equation X * A**T = B for triangular matrix A */
int SolveTriangularMatrixEquation(int m, int n, double** a, int a_row_offset, int a_col_offset, int b_row_offset, int b_col_offset);

/* This function performs the symmetric rank k operation C := -A * A**T + C */
int SymmetricRankKOperation(int n, int k, double** a, int a_row_offset, int a_col_offset, int c_row_offset, int c_col_offset);

/*********************************************************************************************************/
/************************************************* MAIN **************************************************/
/*********************************************************************************************************/

int main(int argc, char* argv[])
{
	int i, j, system_return = 0, error = 0;
	
	/* Read inputs */
	/* Get the order of symmetric matrix */
	int order = 0;
	
	FILE* infile_order = fopen("inputs/order.txt", "r");
	system_return = fscanf(infile_order, "%u", &order);
	if (system_return == -1)
	{
		printf("Failed reading file inputs/order.txt\n");
	}
	fclose(infile_order);
	printf("order = %d\n", order);
	
	/* Now get the matrix values */
	double** matrix;
	
	printf("matrix = \n");
	FILE* infile_matrix = fopen("inputs/matrix.txt", "r");
	matrix = malloc(sizeof(double*) * order);
	for (i = 0; i < order; i++)
	{
		matrix[i] = malloc(sizeof(double) * order);
		for (j = 0; j < order; j++)
		{
			system_return = fscanf(infile_matrix, "%lf\t", &matrix[i][j]);
			if (system_return == -1)
			{
				printf("Failed reading file inputs/matrix.txt\n");
			}
			else
			{
				printf("%lf\t", matrix[i][j]);
			}
		} // end of j loop
		printf("\n");
	} // end of i loop
	fclose(infile_matrix);
	
	/* Call function to perform Cholesky decomposition on our real symmetric positive definite matrix */
	error = RecursiveCholeskyFactorization(order, matrix, 0, 0);
	
	/* Print L or error code message */
	if (error == 0)
	{
		/* Zero upper triangular matrix without diagonal since it wasn't updated */
		for (i = 0; i < order - 1; i++)
		{
			for (j = i + 1; j < order; j++)
			{
				matrix[i][j] = 0.0;
			} // end of j loop
		} // end of i loop

		printf("L = \n");
		for (i = 0; i < order; i++)
		{
			for (j = 0; j < order; j++)
			{
				printf("%lf\t", matrix[i][j]);
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
	
	/* Free dynamically allocated memory */
	for (i = 0; i < order; i++)
	{
		free(matrix[i]);
	} // end of i loop
	free(matrix);
	
	return 0;
} // end of main

/*********************************************************************************************************/
/*********************************************** FUNCTIONS ***********************************************/
/*********************************************************************************************************/

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
	if (n == 0)
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
		error = SolveTriangularMatrixEquation(n2, n1, a, a_row_offset, a_col_offset, n1 + a_row_offset, a_col_offset);
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
int SolveTriangularMatrixEquation(int m, int n, double** a, int read_row_offset, int read_col_offset, int write_row_offset, int write_col_offset)
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
	if (m == 0 || n == 0)
	{
		return 2;
	}
	/* Form  B := B * inv( A**T ). */
	for (k = 0; k < n; k++)
	{
		temp = 1.0 / a[k + read_row_offset][k + read_col_offset];
		for (i = 0; i < m; i++)
		{
			a[i + write_row_offset][k + write_col_offset] *= temp;
		} // end of i loop
		
		for (j = k + 1; j < n; j++)
		{
			if (a[j + read_row_offset][k + read_col_offset] != 0.0)
			{
				temp = a[j + read_row_offset][k + read_col_offset];
				for (i = 0; i < m; i++)
				{
					a[i + write_row_offset][j + write_col_offset] -= temp * a[i + write_row_offset][k + write_col_offset];
				} // end of i loop
			}
		} // end of j loop
	} // end of k loop
	
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
	if (n == 0 || k == 0)
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