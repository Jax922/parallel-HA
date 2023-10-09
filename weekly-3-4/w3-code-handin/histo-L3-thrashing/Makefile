CXX        = nvcc -O3

SRC_GPU = main-gpu.cu
SRC_CPU = main-omp.c
HELPERS = ../helper.h parallelPlan.h goldenSeq.h
KERNELS = kernels.cu.h
EXEC_GPU = histo-gpu
EXEC_CPU = histo-cpu

default: compile_cpu run_cpu compile_gpu run_gpu

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile_cpu: $(EXEC_CPU)
compile_gpu: $(EXEC_GPU)

$(EXEC_GPU): $(SRC_GPU) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(EXEC_GPU) $(SRC_GPU)

$(EXEC_CPU): $(SRC_CPU) $(HELPERS)
	g++ -O3 -fopenmp -o $(EXEC_CPU) $(SRC_CPU)


# N_LIST = [262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912]

run_gpu: $(EXEC_GPU)
	./$(EXEC_GPU) 536870912 41943040
	# for n in $(N_LIST); do \
	# 	./$(EXEC_GPU) $$n 41943040; \
	# done

run_cpu: $(EXEC_CPU)
	./$(EXEC_CPU) 536870912 134217728
	# for n in $(N_LIST); do \
	# 	./$(EXEC_CPU) $$n 134217728; \
	# done

clean:
	rm -f $(EXEC_GPU) $(EXEC_CPU)

