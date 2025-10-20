#include <iostream>
#include <printf.h>
#include <cmath>
#include <map>
#include <vector>

#define NUM_ITEMS 5
#define MAX_WEIGHT 3.0
#define MAX_ITEMS 2.0
#define ITEMS {2.0, 0.6, 0.5, 0.3, 0.1}

// arbitrary values for penalty
#define P1 2
#define P2 10

#define MAX_THREADS_PER_BLOCK 1024

using namespace std;



__constant__ float itemsGPU[NUM_ITEMS] = ITEMS;
const float itemsCPU[NUM_ITEMS] = ITEMS;


__device__ float* slacksGPU;
__host__ void calculate_slack_vars(vector<float> &slacks_weights, size_t* amount_of_slacks){
    printf("-=-=-=-=-=-Calculating Slack variables-=-=-=-=-=-=-=-=-\n");

    size_t amount = 0;
    map<float,bool> mapped_diffs;

    for(size_t i = 0; i < NUM_ITEMS; i++){
        for(size_t j = 0; j < NUM_ITEMS; j++){
            if(i == j) continue;

            float diff = MAX_WEIGHT - (itemsCPU[i] + itemsCPU[j]);
            
            if(diff == 0 || mapped_diffs[diff]) continue;

            printf("Found slack total=%ld, diff=%f\n", amount, diff);    


            mapped_diffs[diff] = true;
            slacks_weights.push_back(diff);
            amount ++;
        }
    }

    *amount_of_slacks = amount;

}


__device__ float* qubo;
__device__ float* penalty_weight;
__device__ float* penalty_amount;
__global__ void eval_qubo(size_t max, size_t total_bits, size_t total_slacks, int threads_y){
    int index = (blockIdx.x*threads_y)+threadIdx.y;
    if(index >= max){
        return;
    }

    int bit_index = total_bits - threadIdx.x - 1;
    int bin_to_dec = index;
    bool is_one = (bin_to_dec >> bit_index)%2 == 1;
    
    int value = 0;
    int penalty_1 = 0;
    int penalty_2 = 0;
    

    if(is_one){
        int max_var_index = total_bits-total_slacks-1;

        if(bit_index <= max_var_index){
            value += itemsGPU[bit_index];
            penalty_2 += itemsGPU[bit_index];
        }else{
            int slack_index = bit_index-max_var_index;
            penalty_2 += slacksGPU[slack_index];
        }


        penalty_1 = 1;
    }

    size_t mem_index = index;
    qubo[mem_index] += value;
    penalty_amount[mem_index] += penalty_1;
    penalty_weight[mem_index] += penalty_2;

}

__global__ void sum_up_values(size_t max){
    int index = (blockIdx.x*MAX_THREADS_PER_BLOCK)+threadIdx.x;
    if(index >= max) return;

    printf("%ld = %f %f %f\n", index, qubo[index], penalty_amount[index], penalty_weight[index]);
    
    qubo[index] += P1*pow((penalty_amount[index] - MAX_ITEMS),2) + P2*pow((penalty_weight[index] - MAX_WEIGHT), 2);
}

__global__ void show_values(size_t size){
    for(size_t i = 0; i < size; i++){
        printf("%ld = %f \n", i, qubo[i]);
    }
}

int main(){ 
    printf("Qubo with CUDA for KNAPSACK of two items\n");
    
    vector<float> slacks_weights;
    size_t amount_of_slacks;
    calculate_slack_vars(slacks_weights, &amount_of_slacks);

    size_t size_slacks_bytes = amount_of_slacks * sizeof(float);
    float* slacks = (float*)malloc(size_slacks_bytes);
    for(size_t i = 0; i < amount_of_slacks; i++)
        slacks[i] = slacks_weights.at(i);

    float* gpu_slacks_temp;
    auto status = cudaMalloc(&gpu_slacks_temp, size_slacks_bytes);
    if(status != cudaSuccess){
        printf("Failed on allocate memory on GPU\n");
        return 1;
    }

    status = cudaMemcpy(gpu_slacks_temp, slacks, size_slacks_bytes, cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
        printf("Failed on copy data to GPU\n");
        return 1;
    }
    
    status = cudaMemcpyToSymbol(slacksGPU, &gpu_slacks_temp, sizeof(float*));
    if(status != cudaSuccess){
        printf("Failed on copy symbol to gpu\n");
        return 1;
    }



    size_t total_bits = NUM_ITEMS + amount_of_slacks;
    size_t amount_of_computations = pow(2,total_bits);


    int threads_y = floor(MAX_THREADS_PER_BLOCK/total_bits);
    dim3 threads_dim_eval(total_bits, threads_y, 1);
    dim3 blocks_dim_eval(ceil(amount_of_computations/threads_y),1,1);


    dim3 threads_dim_sum(MAX_THREADS_PER_BLOCK,1,1);
    dim3 blocks_dim_sum(ceil(amount_of_computations/MAX_THREADS_PER_BLOCK),1,1);

    size_t values_array_size = amount_of_computations*sizeof(float);
        
    
    float* local_qubo;
    float* local_penalty1;
    float* local_penalty2;

    status = cudaMalloc(&local_qubo, values_array_size);
    if(status != cudaSuccess){
        printf("Failed on allocate memory for qubo\n");
        return 1;
    }
    
    status = cudaMalloc(&local_penalty1, values_array_size);
    if(status != cudaSuccess){
        printf("Failed on allocate memory for amount penalty\n");
        return 1;
    }
    
    status = cudaMalloc(&local_penalty2, values_array_size);
    if(status != cudaSuccess){
        printf("Failed on allocate memory for weight penalty\n");
        return 1;
    }


    status = cudaMemcpyToSymbol(qubo, &local_qubo, sizeof(float*));
    if(status != cudaSuccess){
        printf("Failed on copy qubo symbol to gpu\n");
        return 1;
    }
    
    status = cudaMemcpyToSymbol(penalty_amount, &local_penalty1, sizeof(float*));
    if(status != cudaSuccess){
        printf("Failed on copy penalty amount symbol to gpu\n");
        return 1;
    }
    
    status = cudaMemcpyToSymbol(penalty_weight, &local_penalty2, sizeof(float*));
    if(status != cudaSuccess){
        printf("Failed on copy penalty weight symbol to gpu\n");
        return 1;
    }


    
    printf("-=-=-=-=-=-Evaluating-=-=-=-=-=-=-\n");
    eval_qubo<<<blocks_dim_eval, threads_dim_eval>>>(amount_of_computations,total_bits,amount_of_slacks, threads_y);
    cudaDeviceSynchronize();

    sum_up_values<<<blocks_dim_sum, threads_dim_sum>>>(amount_of_computations);
    cudaDeviceSynchronize();

    show_values<<<1,1>>>(amount_of_computations);
    cudaDeviceSynchronize();

    free(slacks);
    cudaFree(gpu_slacks_temp);
    cudaFree(local_qubo);
    cudaFree(local_penalty1);
    cudaFree(local_penalty2);

    return 0;
}