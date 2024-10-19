#include <cuda_runtime.h>
#include <iostream>

using namespace std;

void printDeviceProperties() {
    cudaDeviceProp prop;
    int device;
    
    // Assign device to memory
    cudaGetDevice(&device);
    
    // Retrieve properties for the current device
    cudaGetDeviceProperties(&prop, device);
    
    // Print some properties
    cout << "Device Name: " << prop.name << std::endl;
    cout << "Device Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes" << std::endl;
    cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
    cout << "WarpSize: " << prop.warpSize << std::endl;
    cout << "SMEM / SM " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    cout << "Warps per SM (threads per sm/warpsize) " << prop.maxThreadsPerMultiProcessor / prop.warpSize << std::endl;

    // specific to our kernel
    cout << "------------------------------------------" << std::endl;
    cout << "Registers per thread: " << 32*32  << endl;
    cout << "SMEM per Block: " << 32*32*4 << " B" << endl;
    cout << "threads per Block: " << 32*32  << endl;

}

int main() {
    printDeviceProperties();
    return 0;
}