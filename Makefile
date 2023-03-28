#
# Copyright (c) 2023      Advanced Micro Devices, Inc. All rights reserved.
#

all:
	cd src ; make

TransferBenchCuda:
	cd src ; make TransferBenchCuda

clean:
	cd src ; make clean
