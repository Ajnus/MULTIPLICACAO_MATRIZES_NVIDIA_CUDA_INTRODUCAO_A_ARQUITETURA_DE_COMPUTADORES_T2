#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include "timer.h"
#include "matrix_lib.h"

float scalar_value = 0.0f;

struct matrix matrixA, matrixB, matrixC;

int store_matrix(struct matrix *matrix, char *filename);
int check_linear_errors(struct matrix *source, struct matrix *destination, float scalar_value);
int check_mult_errors(struct matrix *matA, struct matrix *matB, struct matrix *matC);
int print_matrix(struct matrix *matrix);
int initialize_matrix(struct matrix *matrix, float value, float inc);
int initialize_matrix(struct matrix *matrix, float value, float inc);
int load_matrix(struct matrix *matrix, char *filename);

int threadsPerBlock;
int blocksPerGrid;

int main(int argc, char *argv[]) {
  unsigned long int dimA_M, dimA_N, dimB_M, dimB_N;
  char *matrixA_filename, *matrixB_filename, *result1_filename, *result2_filename;
  char *eptr = NULL;
  clock_t start, stop, overall_t1, overall_t2;

  // Mark overall start time
  overall_t1 = clock();

  // Check arguments
  if(argc != 10) {
    printf("Usage: %s <scalar_value> <DimA_M> <DimA_N> <DimB_M> <DimB_N> <matrixA_filename> <matrixB_filename> <result1_filename> <result2_filename> <threads por bloco> <número de blocos>\n", argv[0]);
    return 1;
  }

  // Convert arguments
  scalar_value = strtof(argv[1], NULL);
  dimA_M = strtol(argv[2], &eptr, 10);
  dimA_N = strtol(argv[3], &eptr, 10);
  dimB_M = strtol(argv[4], &eptr, 10);
  dimB_N = strtol(argv[5], &eptr, 10);
  matrixA_filename = argv[6];
  matrixB_filename = argv[7];
  result1_filename = argv[8];
  result2_filename = argv[9];
  threadsPerBlock = atoi(argv[10]);
  blocksPerGrid = atoi(argv[11]);

  if((scalar_value == 0.0f) || (dimA_M == 0) || (dimA_N == 0) || (dimB_M == 0) || (dimB_N == 0)) {
    fprintf(stderr, "[%d] %s: erro na conversao do argumento: errno = %d\n", __LINE__, argv[0], errno);

    /* If a conversion error occurred, display a message and exit */
    if(errno == EINVAL) {
      fprintf(stderr, "Conversion error occurred: %d\n", errno);
      return 1;
    }

    /* If the value provided was out of range, display a warning message */
    if(errno == ERANGE) {
      fprintf(stderr, "The value provided was out of range: %d\n", errno);
      return 1;
  	}
  }

  /* Allocate the arrays of the three matrixes */
  float *a = (float*)aligned_alloc(32, dimA_M*dimA_N*sizeof(float));
  float *b = (float*)aligned_alloc(32, dimB_M*dimB_N*sizeof(float));
  float *c = (float*)aligned_alloc(32, dimA_M*dimB_N*sizeof(float));

  if((a==NULL) || (b==NULL) || (c==NULL)) {
  	fprintf(stderr, "%s: array allocation problem.", argv[0]);
	  return 1;
  }

  /* Initialize the three matrixes */
  matrixA.height = dimA_M;
  matrixA.width = dimA_N;
  matrixA.rows = a;
  if(!load_matrix(&matrixA, matrixA_filename)) {
  	printf("%s: matrixA initialization problem.", argv[0]);
	  return 1;
  }

  /* Print matrix */
  printf("---------- Matrix A ----------\n");
  print_matrix(&matrixA);

  matrixB.height = dimB_M;
  matrixB.width = dimB_N;
  matrixB.rows = b;
  if(!load_matrix(&matrixB, matrixB_filename)) {
  	fprintf(stderr, "%s: matrixB initialization problem.", argv[0]);
	  return 1;
  }

  /* Print matrix */
  printf("---------- Matrix B ----------\n");
  print_matrix(&matrixB);

  matrixC.height = dimA_M;
  matrixC.width = dimB_N;
  matrixC.rows = c;

  /* Print matrix */
  printf("---------- Matrix C ----------\n");
  print_matrix(&matrixC);

  /* Scalar product of matrix A */
  printf("[%d] Executing scalar_matrix_mult(%5.1f, matrixA)...\n", __LINE__, scalar_value);
  start = clock();
  if(!scalar_matrix_mult(scalar_value, &matrixA)) {
  	fprintf(stderr, "%s: scalar_matrix_mult problem.", argv[0]);
	  return 1;
  }
  stop = clock();
  printf("%lf ms\n", timedifference_msec(start, stop));

  /* Print matrix */
  printf("---------- Matrix A ----------\n");
  print_matrix(&matrixA);

  /* Write first result */
  printf("Writing first result: %s...\n", result1_filename);
  if(!store_matrix(&matrixA, result1_filename)) {
	  fprintf(stderr, "%s: failed to write first result to file.", argv[0]);
	  return 1;
  }

  /* Check for errors */
  check_linear_errors(&matrixA, &matrixC, scalar_value);

  /* Calculate the product between matrix A and matrix B */
  printf("[%d] Executing matrix_matrix_mult(matrixA, mattrixB, matrixC)...\n", __LINE__);
  start = clock();
  if(!matrix_matrix_mult(&matrixA, &matrixB, &matrixC)) {
  	fprintf(stderr, "%s: matrix_matrix_mult problem.", argv[0]);
	  return 1;
  }
  stop = clock();
  printf("%lf ms\n", timedifference_msec(start, stop));

  /* Print matrix */
  printf("---------- Matrix C ----------\n");
  print_matrix(&matrixC);

  /* Write second result */
  printf("Writing second result: %s...\n", result2_filename);
  if(!store_matrix(&matrixC, result2_filename)) {
  	fprintf(stderr, "%s: failed to write second result to file.", argv[0]);
	  return 1;
  }

  /* Check foor errors */
  printf("Checking matrixC for errors...\n");
  start = clock();
  check_mult_errors(&matrixA, &matrixB, &matrixC);
  stop = clock();
  printf("[%d] %lf ms\n", __LINE__, timedifference_msec(start, stop));

  // Mark overall stop time
  overall_t2 = clock();

  // Show elapsed overall time
  printf("[%d] Overall time: %lf ms\n", __LINE__, timedifference_msec(overall_t1, overall_t2));

  return 0;
}

int store_matrix(struct matrix *matrix, char *filename) {
  unsigned long int i = 0;
  unsigned long int n = 0;
  FILE *fd = NULL;

  /* Check the numbers of the elements of the matrix */
  n = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if(n == 0 || matrix->rows == NULL) return 0;

  /* Try to open file of floats */
  if((fd = fopen(filename, "wb")) == NULL) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  float *nxt_a = matrix->rows; 

  for(i=0; i<n; i+=8, nxt_a+=8) {
  	if(fwrite(nxt_a, sizeof(float), 8, fd) != 8) {
      fprintf(stderr, "Error writing to file %s: short write (less than 8 floats)\n", filename);
      return 1;
  	}
  }

  if(fd != NULL) fclose(fd);

  return 1;
}

int load_matrix(struct matrix *matrix, char *filename) {
  unsigned long int i = 0;
  unsigned long int n = 0;
  FILE *fd = NULL;

  /* Check the numbers of the elements of the matrix */
  n = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if(n == 0 || matrix->rows == NULL) return 0;

  /* Try to open file of floats */
  if((fd = fopen(filename, "rb")) == NULL) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  float *nxt_a = matrix->rows; 

  for(i=0; i<n; i+=8, nxt_a+=8) {
  	if(fread(nxt_a, sizeof(float), 8, fd) != 8) {
      fprintf(stderr, "[%d] Error reading from file %s: short read (less than 8 floats)\n", __LINE__, filename);
      return 0;
	  }
  }

  if(fd != NULL) fclose(fd);

  return 1;
}

int initialize_matrix(struct matrix *matrix, float value, float inc) {
  unsigned long int n;

  /* Check the numbers of the elements of the matrix */
  n = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if(n==0 || matrix->rows==NULL) return 0;

  return 1;
}

int print_matrix(struct matrix *matrix) {
  unsigned long int i;
  unsigned long int n;
  unsigned long int nxt_newLine;

  /* Check the numbers of the elements of the matrix */
  n = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if(n==0 || matrix->rows==NULL) return 0;

  /* Initialize new line controol */
  nxt_newLine = matrix->width - 1;

  /* Print matrix elements */
  for(i=0; i<n; i++) {
    printf("%5.1f ", matrix->rows[i]);
    if(i == nxt_newLine) {
	    printf("\n");
	    nxt_newLine += matrix->width;
    }
    if(i == 255) {
      printf("Ooops...256 printing limit found...skipping printing...\n");
      break;
    }
  }

  return 1;
}

int check_linear_errors(struct matrix *source, struct matrix *destination, float scalar_value) {
  for(int line=0; line<source->height; line++) {
    for(int row=0; row<source->width; row++) {
      int pos = line*source->width+row;
      if(fabs((source->rows[pos]-destination->rows[pos])/destination->rows[pos]) > 0.0001) {
        fprintf(stderr, "Linear error at [%d, %d] - %f x %f\n", line, row, source->rows[pos], destination->rows[pos]);
        return 0;
      }
    }
  }

  return 0;
}

int check_mult_errors(struct matrix *matA, struct matrix *matB, struct matrix *matC) {
  // Loop para calcular cada elemento da matriz resultante
  for (int i = 0; i < matC->height; i++) {
    for (int j = 0; j < matC->width; j++) {
      float sum = 0.0;
      for (int k = 0; k < matA->width; k++) {
        sum += matA->rows[i * matA->width + k] * matB->rows[k * matB->width + j];
      }
      if(fabs((matC->rows[i * matC->width + j]-sum)/sum) > 0.0001) {
        fprintf(stderr, "Multiplication error at [%d, %d] - %f x %f\n", i, j, matC->rows[i * matC->width + j], sum);
        return 0;
      }
    }
  }
  return 1;
}