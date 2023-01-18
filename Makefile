# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
ROCM_PATH ?= /opt/rocm
HIPCC=$(ROCM_PATH)/bin/hipcc

EXE=TransferBench
CXXFLAGS = -O3 -I. -lnuma -L$(ROCM_PATH)/hsa/lib -lhsa-runtime64 -ferror-limit=5

all: $(EXE)

$(EXE): $(EXE).cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o $(EXE)
