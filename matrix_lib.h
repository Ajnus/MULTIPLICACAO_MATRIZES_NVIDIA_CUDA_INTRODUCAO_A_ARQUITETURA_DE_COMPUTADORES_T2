#ifndef MATRIX_LIB
#define MATRIX_LIB

struct matrix {
	unsigned long int height;
	unsigned long int width;
	float *rows;
};

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);

int matrix_matrix_mult(struct matrix *a, struct matrix *b, struct matrix *c);

#endif