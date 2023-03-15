# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
ROCM_PATH ?= /opt/rocm
HIPCC=$(ROCM_PATH)/bin/hipcc

EXE=TransferBench
CXXFLAGS = -O3 -I. -Iinclude -I$(ROCM_PATH)/include -lnuma -L$(ROCM_PATH)/lib -lhsa-runtime64
LDFLAGS    += -lpthread
all: $(EXE)

$(EXE): $(EXE).cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o ../$@ $(LDFLAGS)

clean:
	rm -f *.o ../$(EXE)
