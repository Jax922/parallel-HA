#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <numeric>

using std::vector;
using std::string;

// constexpr int N = 753411;
constexpr float EPSILON = 0.0001f;
// constexpr int BLOCK_SIZE = 256;
// constexpr int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
constexpr int KENREL_LOOP = 300;

__host__ __device__
float compute(const float x) { return powf(x / (x-2.3f), 3.0f); }

void map_cpu(const vector<float> &input, vector<float> &output)
{
    for (size_t i = 0; i < input.size(); i++)
    {
        output[i] = compute(input[i]);
    }
}

__global__
void map_gpu(const float *input, float *output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
       output[i] = compute(input[i]);
    }
}

void write_logger(const string &filename, const string &content)
{
    std::ofstream logger(filename, std::ios::app);
    logger << content;
    logger.close();
}


void compute_exec_time(int N = 753411, const bool log = true, const string &filename="logger.csv")
{
    vector<float> a_input(N), a_output_cpu(N), a_output_gpu(N);
    float *p_input, *p_output;
    
    for (size_t i = 0; i < N; i++)
    {
        a_input[i] = i;
    }

    // cpu exec time compute
    auto cpu_exec_start = std::chrono::system_clock::now();
    map_cpu(a_input, a_output_cpu);
    auto cpu_exec_end = std::chrono::system_clock::now();
    std::chrono::duration<double> cpu_exec_time = cpu_exec_end - cpu_exec_start;
   

    // gpu exec time compute
    cudaMalloc((void**)&p_input, N * sizeof(float));
    cudaMalloc((void**)&p_output, N * sizeof(float));

    cudaMemcpy(p_input, a_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t gpu_exec_start, gpu_exec_end;
    cudaEventCreate(&gpu_exec_start);
    cudaEventCreate(&gpu_exec_end);
    cudaEventRecord(gpu_exec_start);

    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (size_t i = 0; i < KENREL_LOOP; i++)
    {
        map_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(p_input, p_output, N);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(gpu_exec_end);
    cudaEventSynchronize(gpu_exec_end);
    float gpu_exec_time = 0;
    cudaEventElapsedTime(&gpu_exec_time, gpu_exec_start, gpu_exec_end);

    cudaMemcpy(a_output_gpu.data(), p_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // check the validation
    bool validation = true;
    for (size_t i = 0; i < N; i++)
    {
        if (fabs(a_output_cpu[i] - a_output_gpu[i]) > EPSILON)
        {
           validation = false;
        }
    }

    // print the result
    if(!log)
    {
        std::cout << "Validation: " << validation << "\n";
        std::cout << "CPU elapsed time: " << cpu_exec_time.count() * 1000 << "ms\n";
        std::cout << "GPU exec time: " << gpu_exec_time << "ms\n";
    }
   

    // write the result to the logger
    if(log)
    {
        string logger_content = std::to_string(N) + "," + std::to_string(cpu_exec_time.count() * 1000) + "," + std::to_string(gpu_exec_time/KENREL_LOOP) + "," + std::to_string(validation) + "\n";
        write_logger(filename, logger_content);
    }

    if (validation == false)
    {
        std::cout << "Invalidation" << N << "\n";
    }
    

    cudaFree(p_input);
    cudaFree(p_output);
}


int main()
{
    // N = 753411
    std::cout << "N=753411\n";
    const int N = 753411;
    compute_exec_time(N, false);

    std::cout << "N<=500\n";
    vector<int> N_list_1_500(500);
    std::iota(N_list_1_500.begin(), N_list_1_500.end(), 1);
    for (auto N : N_list_1_500)
    {
        compute_exec_time(N, true, string("logger_1_500.csv"));
    }

    std::cout << "N<=1000\n";
    vector<int> N_list_500_1000(500);
    std::iota(N_list_500_1000.begin(), N_list_500_1000.end(), 500);
    for (auto N : N_list_500_1000)
    {
        compute_exec_time(N, true, string("logger_500_1000.csv"));
    }

    std::cout << "N>1000\n";
    vector<int> N_list_1000_10000000 = {1000, 10000, 100000, 1000000, 10000000};
    for (auto N : N_list_1000_10000000)
    {
        compute_exec_time(N, true, string("logger_1000_10000000.csv"));
    }
}