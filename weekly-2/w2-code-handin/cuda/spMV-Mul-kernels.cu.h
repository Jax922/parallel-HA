#ifndef SP_MV_MUL_KERS
#define SP_MV_MUL_KERS

__global__ void
replicate0(int tot_size, char* flags_d) {
    // ... fill in your implementation here ...
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tot_size)
    {
        if (flags_d[idx] == 1)
        {
            int offset = idx + 1;
            int repCount = flags_d[idx];
            while (repCount > 1) {
                flags_d[offset] = 1;
                offset++;
                repCount--;
            }
        }
    }
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    // ... fill in your implementation here ...
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mat_rows) {
        int startIndex = (idx > 0) ? mat_shp_sc_d[idx - 1] : 0;
        int endIndex = mat_shp_sc_d[idx];
        
        if (startIndex == endIndex) {
            // Empty row
            flags_d[startIndex] = 0;
        } else {
            flags_d[startIndex] = 1;
            for (int i = startIndex + 1; i < endIndex; i++) {
                flags_d[i] = 0;
            }
        }
    }

}

__global__ void 
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    // ... fill in your implementation here ...
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tot_size) {
        int colIndex = mat_inds[idx];
        tmp_pairs[idx] = mat_vals[idx] * vct[colIndex];
    }
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    // ... fill in your implementation here ...
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mat_rows) {
        // int lastIndex = (idx > 0) ? mat_shp_sc_d[idx - 1] : 0;
        int currentIndex = mat_shp_sc_d[idx] - 1;
        res_vct_d[idx] = tmp_scan[currentIndex];
    }

}

#endif
