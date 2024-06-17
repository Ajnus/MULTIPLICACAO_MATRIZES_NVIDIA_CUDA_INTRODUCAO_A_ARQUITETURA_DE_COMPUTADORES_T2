#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

extern int threadsPerBlock;
extern int blocksPerGrid;

__global__ void scalar_matrix_mult_kernel(float scalar_value, float *matrix_rows, unsigned long int num_elements)
{
    unsigned long int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements)
    {
        matrix_rows[idx] *= scalar_value;
    }
}

__global__ void matrix_matrix_mult_kernel(float *matrixA, float *matrixB, float *matrixC, unsigned long int heightA, unsigned long int widthA, unsigned long int widthB)
{
    unsigned long int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < heightA && col < widthB)
    {
        float value = 0;
        for (unsigned long int k = 0; k < widthA; k++)
        {
            value += matrixA[row * widthA + k] * matrixB[k * widthB + col];
        }
        matrixC[row * widthB + col] = value;
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix)
{
    cudaError_t cudaError;

    // Verificar se a matriz passada é válida
    if (matrix == NULL || matrix->rows == NULL)
    {
        printf("Erro: Matriz inválida.\n");
        return 0;
    }

    // Determinar o número de elementos na matriz
    unsigned long int num_elements = matrix->height * matrix->width;

    // Alocar memória na GPU para a matriz
    float *d_matrix_rows;
    cudaError = cudaMalloc(&d_matrix_rows, num_elements * sizeof(float));
    if (cudaError != cudaSuccess)
    {
        printf("cudaMalloc d_x returned error %s (code %d)\n",
               cudaGetErrorString(cudaError), cudaError);
        return 0;
    }

    // Copiar os dados da matriz da CPU para a GPU
    cudaError = cudaMemcpy(d_matrix_rows, matrix->rows, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess)
    {
        printf("cudaMemcpy (h_x -> d_x) returned error %s (code %d), line(% d)\n ", cudaGetErrorString(cudaError), cudaError,
               __LINE__);
        return 0;
    }

    // Lançar o kernel
    scalar_matrix_mult_kernel<<<blocksPerGrid, threadsPerBlock>>>(scalar_value, d_matrix_rows, num_elements);

    // Esperar a execução do kernel terminar
    cudaDeviceSynchronize();

    cudaGetLastError();

    // Copiar os resultados da GPU de volta para a CPU
    cudaError = cudaMemcpy(matrix->rows, d_matrix_rows, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess)
    {
        printf("cudaMemcpy (d_x -> h_y) returned error %s (code %d), line(% d)\n ", cudaGetErrorString(cudaError),
               cudaError,
               __LINE__);
        return 0;
    }

    // Liberar a memória na GPU
    cudaFree(d_matrix_rows);

    // Retornar 1 indicando que a operação foi bem-sucedida
    return 1;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC)
{
    cudaError_t cudaError;

    // Verificar se as dimensões são válidas para a multiplicação de matrizes
    if (matrixA->width != matrixB->height)
    {
        printf("Erro: As dimensões das matrizes não são compatíveis para multiplicação.\n");
        return 0;
    }

    unsigned long int heightA = matrixA->height;
    unsigned long int widthA = matrixA->width;
    unsigned long int widthB = matrixB->width;

    // Alocar memória na GPU para as matrizes
    float *d_matrixA, *d_matrixB, *d_matrixC;
    cudaError = cudaMalloc(&d_matrixA, heightA * widthA * sizeof(float));
    if (cudaError != cudaSuccess)
    {
        printf("cudaMalloc d_x returned error %s (code %d)\n",
               cudaGetErrorString(cudaError), cudaError);
        return 0;
    }

    cudaError = cudaMalloc(&d_matrixB, widthA * widthB * sizeof(float));
    if (cudaError != cudaSuccess)
    {
        printf("cudaMalloc d_x returned error %s (code %d)\n",
               cudaGetErrorString(cudaError), cudaError);
        return 0;
    }

    cudaError = cudaMalloc(&d_matrixC, heightA * widthB * sizeof(float));
    if (cudaError != cudaSuccess)
    {
        printf("cudaMalloc d_x returned error %s (code %d)\n",
               cudaGetErrorString(cudaError), cudaError);
        return 0;
    }

    // Copiar os dados das matrizes da CPU para a GPU
    cudaError = cudaMemcpy(d_matrixA, matrixA->rows, heightA * widthA * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess)
    {
        printf("cudaMemcpy (h_x -> d_x) returned error %s (code %d), line(% d)\n ", cudaGetErrorString(cudaError), cudaError,
               __LINE__);
        return 0;
    }

    cudaError = cudaMemcpy(d_matrixB, matrixB->rows, widthA * widthB * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess)
    {
        printf("cudaMemcpy (h_x -> d_x) returned error %s (code %d), line(% d)\n ", cudaGetErrorString(cudaError), cudaError,
               __LINE__);
        return 0;
    }

    // Definir o número de threads por bloco e o número de blocos
    // dim3 threadsPerBlock(16, 16);
    // dim3 blocksPerGrid((widthB + threadsPerBlock.x - 1) / threadsPerBlock.x, (heightA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Lançar o kernel
    matrix_matrix_mult_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrixA, d_matrixB, d_matrixC, heightA, widthA, widthB);

    // Esperar a execução do kernel terminar
    cudaDeviceSynchronize();

    cudaGetLastError();

    // Copiar os resultados da GPU de volta para a CPU
    cudaError = cudaMemcpy(matrixC->rows, d_matrixC, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess)
    {
        printf("cudaMemcpy (h_x -> d_x) returned error %s (code %d), line(% d)\n ", cudaGetErrorString(cudaError), cudaError,
               __LINE__);
        return 0;
    }

    // Liberar a memória na GPU
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);

    return 1;
}