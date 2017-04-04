CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC = "/usr/local/cuda/bin/nvcc"

PTX_SUFFIX = nvvm


SM_TARGETS 	= -gencode=arch=compute_52,code=\"sm_52,compute_52\" 
SM_DEF 		= -DSM520
TEST_ARCH 	= 520

GENCODE_SM50    := -gencode arch=compute_52,code=sm_52
GENCODE_FLAGS   := $(GENCODE_SM50)

CDP_SUFFIX = nocdp

ABI_SUFFIX = abi

CPU_ARCH = -m64
CPU_ARCH_SUFFIX = x86_64
NPPI = -lnppi

NVCCFLAGS += $(SM_DEF) -Xptxas -v -Xcudafe -\# 

BIN_SUFFIX = sm$(SM_ARCH)_$(PTX_SUFFIX)_$(ABI_SUFFIX)_$(CDP_SUFFIX)_$(CPU_ARCH_SUFFIX)

CUB_DIR = ./cub/
INC += -I$(CUB_DIR) -I$(CUB_DIR)test 
BIN_DIR = ./bin/

select:
	$(NVCC) $(DEFINES) -lcurand $(SM_TARGETS) -o $(BIN_DIR)select_$(BIN_SUFFIX) select.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

join:
	$(NVCC) $(DEFINES) -lcurand $(SM_TARGETS) -o $(BIN_DIR)join_$(BIN_SUFFIX) join.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

groupby:
	$(NVCC) $(DEFINES) -lcurand $(SM_TARGETS) -o $(BIN_DIR)groupby_$(BIN_SUFFIX) groupby.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

sort:
	$(NVCC) $(DEFINES) -lcurand $(SM_TARGETS) -o $(BIN_DIR)sort_$(BIN_SUFFIX) sort.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

topk:
	$(NVCC) $(DEFINES) -lcurand $(SM_TARGETS) -o $(BIN_DIR)topk_$(BIN_SUFFIX) topk.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

setup:
	wget https://github.com/NVlabs/cub/archive/1.6.4.zip
	unzip 1.6.4.zip
	mv cub-1.6.4 cub
	rm 1.6.4.zip
	mkdir -p bin


