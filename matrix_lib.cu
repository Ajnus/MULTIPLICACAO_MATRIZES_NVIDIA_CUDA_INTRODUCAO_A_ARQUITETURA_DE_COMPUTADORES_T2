#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h> // Biblioteca Intel Intrinsics
#include "matrix_lib.h"

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    // Verificar se a matriz passada é válida
    if (matrix == NULL || matrix->rows == NULL) {
        printf("Erro: Matriz inválida.\n");
        return 0;
    }

    // Determinar o número de elementos na matriz
    unsigned long int num_elements = matrix->height * matrix->width;

    // Vetor com o valor escalar replicado para ser usado na operação vetorial
    __m256 scalar = _mm256_set1_ps(scalar_value);

    // Iterar sobre os elementos da matriz em blocos de 8 floats (tamanho de um registrador AVX)
    for (unsigned long int i = 0; i < num_elements; i += 8) {
        // Carregar 8 elementos da matriz em um registrador AVX
        __m256 matrix_elements = _mm256_loadu_ps(&(matrix->rows[i]));

        // Multiplicar os elementos da matriz pelo valor escalar
        __m256 result = _mm256_mul_ps(matrix_elements, scalar);

        // Armazenar o resultado de volta na matriz
        _mm256_storeu_ps(&(matrix->rows[i]), result);
    }

    // Retornar 1 indicando que a operação foi bem-sucedida
    return 1;
}

int matrix_matrix_mult(struct matrix *A, struct matrix *B, struct matrix *C) {
    // Verificar se as dimensões são válidas para a multiplicação de matrizes
    if (A->width != B->height) {
        printf("Erro: As dimensões das matrizes não são compatíveis para multiplicação.\n");
        return 0;
    }

    unsigned long int linhaC = A->height;    // Número de linhas da matriz resultante
    unsigned long int colunaC = B->width;    // Número de colunas da matriz resultante

    // Zera matriz C
    for (unsigned long int i = 0; i < linhaC; i++) {
        for (unsigned long int j = 0; j < colunaC; j++) {
            C->rows[i * colunaC + j] = 0;
        }
    }

    // Iterar cada elemento das linhas da matriz A com todas as linhas da matriz B
    for (unsigned long int i = 0; i < A->height; i++) {
        for (unsigned long int j = 0; j < A->width; j++) {
            // Carregar o elemento de A em um registrador AVX e replicá-lo
            __m256 a_element = _mm256_set1_ps(A->rows[i * A->width + j]);

            // Iterar sobre as colunas da matriz B em blocos de 8 floats (tamanho de um registrador AVX)
            for (unsigned long int k = 0; k < B->width; k += 8) {
                // Carregar 8 elementos de B em um registrador AVX
                __m256 b_elements = _mm256_loadu_ps(&(B->rows[j * B->width + k]));

                // Multiplicar os elementos de A pelo bloco de elementos de B
                __m256 result = _mm256_mul_ps(a_element, b_elements);

                // Somar o resultado ao bloco correspondente da matriz C
                __m256 c_elements = _mm256_loadu_ps(&(C->rows[i * colunaC + k]));
                c_elements = _mm256_add_ps(c_elements, result);
                _mm256_storeu_ps(&(C->rows[i * colunaC + k]), c_elements);
            }
        }
    }
    return 1;
}
