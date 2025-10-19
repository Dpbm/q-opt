#include <iostream>
#include <printf.h>
#include <cmath>
#include <map>
#include <vector>

#define NUM_ITEMS 5
#define MAX_WEIGHT 3.0

using namespace std;

__constant__ float itemsGPU[NUM_ITEMS][NUM_ITEMS];
const float itemsCPU[NUM_ITEMS][NUM_ITEMS] = {
    {2.0, 2.6, 2.5, 2.3, 2.1}, // laptop
    {0.0, 0.6, 1.1, 0.9, 0.7}, // notebook
    {0.0, 0.0, 0.5, 0.8, 0.6}, // book
    {0.0, 0.0, 0.0, 0.3, 0.4}, // umbrella
    {0.0, 0.0, 0.0, 0.0, 0.1}, // apple
};

__constant__ float* slacksGPU;

namespace Visualizer{
    template <typename T>
    __global__ void show_matrix(T** x, size_t rows, size_t cols){
        printf("====COMPLETE MATRIX====\n");
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                printf("%d ", x[i][j]);
            }
            printf("\n");
        }
    }

    __global__ void show_output(float* outputs, size_t len){
        printf("====OUTPUTS ARRAY====\n");
        for(size_t i = 0; i < len; i++){
            printf("%f ",outputs[i]);
        }
        printf("\n");

    }

};
    

__device__ void get_binary_value(u_int8_t* values, int value, u_int8_t size){
    for(size_t i = 0; i < size; i++){
        int bin_index = size-1-i;
        int bin_converted = pow(2,bin_index);

        int updated_value = value - bin_converted;

        if(updated_value >= 0){
            value -= bin_converted;
            values[i] = (u_int8_t)1;
        }else{
            values[i] = (u_int8_t)0;
        }
    }
}

__global__ void generate_x_vector(u_int8_t** x, size_t size){
    get_binary_value(x[threadIdx.x], threadIdx.x, size);
} 


__global__ void apply_ops(u_int8_t** x, float* outputs){
    float tempData[NUM_ITEMS];
    for(size_t i = 0; i < NUM_ITEMS; i++){
        for(size_t j = 0; j < NUM_ITEMS; j++){
            tempData[i] += itemsGPU[j][i] * (float)x[threadIdx.x][j];
        }
    }
    for(size_t i = 0; i < NUM_ITEMS; i++){
        outputs[threadIdx.x] += tempData[i] * (float)x[threadIdx.x][i];
    }
}


__host__ size_t calculate_slack_vars(){
    printf("-=-=-=-=-=-Calculating Slack variables-=-=-=-=-=-=-=-=-\n");

    size_t amount = 0;
    map<float,bool> mapped_diffs;
    vector<float> slacks_weights;

    for(size_t i = 0; i < NUM_ITEMS; i++){
        for(size_t j = 0; j < NUM_ITEMS; j++){
            if(i == j) continue;

            float diff = MAX_WEIGHT - (itemsCPU[i][i] + itemsCPU[j][j]);
            
            if(diff == 0 || mapped_diffs[diff]) continue;

            printf("Found slack total=%ld, diff=%f\n", amount, diff);    


            mapped_diffs[diff] = true;
            slacks_weights.push_back(diff);
            amount ++;
        }
    }

    size_t size_slacks_bytes = amount * sizeof(float);
    float* slacks = (float*)malloc(size_slacks_bytes);
    for(size_t i = 0; i < amount; i++)
        slacks[i] = slacks_weights.at(i);

    printf("Copied into GPU memory\n");
    cudaMemcpyToSymbol(slacksGPU, slacks, size_slacks_bytes);

    free(slacks);

    return amount;
}


int main(){ 
    printf("Qubo with CUDA for KNAPSACK of two items\n");
    
    size_t amount_of_slack = calculate_slack_vars();

    u_int8_t total_bits = 6;
    u_int8_t total_threads = pow(2,total_bits);

    size_t bytes_size_matrix = total_threads*(sizeof(u_int8_t*));
    size_t bytes_size_row = total_bits*(sizeof(u_int8_t));

    cudaMemcpyToSymbol(itemsGPU, itemsCPU, NUM_ITEMS*NUM_ITEMS*sizeof(float));

    u_int8_t** gpu_x;
    cudaError_t status =  cudaMalloc(&gpu_x, bytes_size_matrix);
    if(status != cudaSuccess){
        printf("Failed on allocate matrix on gpu!");
        return 1;
    }

    u_int8_t** host_x = (u_int8_t**)malloc(bytes_size_matrix);
    if(host_x == nullptr){
        printf("Failed on allocate host matrix!");
        return 1;
    }

    for(size_t i = 0; i < total_threads; i++){
        status = cudaMalloc(&host_x[i], bytes_size_row);
        if(status != cudaSuccess){
            printf("Failed on allocate matrix rows on gpu!");
            return 1;
        }
    }

    float* outputs;
    status = cudaMalloc(&outputs, total_threads*sizeof(float));
    if(status != cudaSuccess){
        printf("Failed on allocate outputs array on gpu!");
        return 1;
    }

    cudaMemcpy(gpu_x,host_x,bytes_size_matrix, cudaMemcpyHostToDevice);

    generate_x_vector<<<1, total_threads>>>(gpu_x, total_bits);
    cudaDeviceSynchronize();

    Visualizer::show_matrix<u_int8_t><<<1,1>>>(gpu_x, total_threads, total_bits);
    cudaDeviceSynchronize();

    // apply_ops<<<1, total_threads>>>(gpu_x,outputs);
    // cudaDeviceSynchronize();

    // Visualizer::show_output<<<1,1>>>(outputs, total_threads);
    // cudaDeviceSynchronize();


    for(size_t i = 0; i < total_threads; i++){
        cudaFree(host_x[i]);
    }
    cudaFree(host_x);
    cudaFree(gpu_x);

    return 0;
}