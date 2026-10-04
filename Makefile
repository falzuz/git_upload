NVCC ?= nvcc
ARCH ?= sm_80
TARGET ?= bosegascl0
SEED ?= 1948175546068024
NVCCFLAGS ?= -std=c++17 -Wno-deprecated-gpu-targets

.PHONY: all FORCE
all: $(TARGET)

$(TARGET): main.cu Parameters.h Helpers.cu Lattice.cu Propagator.cu Observables.cu SaveResults.cu FORCE
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -DSIMULATION_SEED=$(SEED)ULL -o $@ main.cu -lcurand -lcufft -lstdc++fs

FORCE:
