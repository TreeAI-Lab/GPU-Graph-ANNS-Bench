# Builds sample_top32, edge_length_stats and cagra_build_csr into $(BIN_DIR).
# Run it through build.sh, which sources env.sh for W, BIN_DIR and ENV_CUVS.
#
# sample_top32 and cagra_build_csr link the libcuvs built from source by third_party/build/cuvs.sh.
# That build carries the CUVS_CAGRA_NO_REVERSE patch, which cagra_build_csr --ab needs.
# The RAPIDS headers (RAFT, RMM, CCCL, ...) and the logger sources come from the same cuVS build tree.
# edge_length_stats is plain C++ with OpenMP.

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMMON := $(HERE)../common

ENV_CUVS ?= $(W)/envs/cuvs-build
CUVS_SRC ?= $(W)/src/cuvs
# Layout of third_party/build/cuvs.sh: CMake build tree $(W)/build/cuvs/build with the dependency sources in its
# _deps/, install prefix $(W)/build/cuvs/install (the same variables as src/baselines/Makefile).
CUVS_INSTALL ?= $(W)/build/cuvs/install
CUVS_BUILD_DIR ?= $(W)/build/cuvs/build
CUVS_DEPS ?= $(CUVS_BUILD_DIR)/_deps
RAFT_INCLUDE ?= $(CUVS_DEPS)/raft-src/cpp/include
CUTLASS_INCLUDE ?= $(CUVS_DEPS)/nvidiacutlass-src/include

CXX_CUVS := $(ENV_CUVS)/bin/x86_64-conda-linux-gnu-c++
CXX_CPU ?= g++

CXXFLAGS_COMMON := -std=c++17 -O3 -fopenmp -DNDEBUG -Wno-deprecated-declarations

INCLUDES_CUVS := \
	-I$(CUVS_INSTALL)/include \
	-I$(ENV_CUVS)/include \
	-I$(ENV_CUVS)/targets/x86_64-linux/include \
	-I$(RAFT_INCLUDE) \
	-I$(CUVS_DEPS)/raft-build/include \
	-I$(CUVS_DEPS)/rmm-src/include \
	-I$(CUVS_DEPS)/rmm-build/include \
	-I$(CUVS_DEPS)/cccl-src/thrust \
	-I$(CUVS_DEPS)/cccl-src/libcudacxx/include \
	-I$(CUVS_DEPS)/cccl-src/cub \
	-I$(CUVS_DEPS)/fmt-src/include \
	-I$(CUVS_DEPS)/spdlog-src/include \
	-I$(CUVS_DEPS)/nvtx3-src/c/include \
	-I$(CUVS_DEPS)/cuco-src/include \
	-I$(CUTLASS_INCLUDE) \
	-I$(CUVS_DEPS)/nvidiacutlass-build/include \
	-I$(HERE) \
	-I$(COMMON)

DEFINES_CUVS := \
	-DFMT_HEADER_ONLY=1 \
	-DSPDLOG_FMT_EXTERNAL \
	-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
	-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA \
	-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP \
	-DTHRUST_DISABLE_ABI_NAMESPACE \
	-DTHRUST_IGNORE_ABI_NAMESPACE_ERROR

LDFLAGS_CUVS := -L$(CUVS_INSTALL)/lib \
	-L$(ENV_CUVS)/lib \
	-L$(ENV_CUVS)/targets/x86_64-linux/lib \
	-Wl,-rpath,$(CUVS_INSTALL)/lib \
	-Wl,-rpath,$(ENV_CUVS)/lib \
	-Wl,-rpath,$(ENV_CUVS)/targets/x86_64-linux/lib \
	-Wl,-rpath-link,$(CUVS_INSTALL)/lib -Wl,-rpath-link,$(ENV_CUVS)/lib

LIBS_CUVS := -lcuvs -lcudart -lcublas -lcusolver -lcusparse -lcurand -lnvJitLink \
	-lgomp -ldl -lpthread -lm

LOGGER_SRCS := \
	$(CUVS_DEPS)/rmm-build/include/rmm/logger_impl/logger.cpp \
	$(CUVS_DEPS)/raft-build/include/raft/core/logger_impl/logger.cpp

TOOLS := sample_top32 edge_length_stats cagra_build_csr

.PHONY: all clean $(TOOLS)

all: $(TOOLS)

$(TOOLS): %: $(BIN_DIR)/%

$(BIN_DIR)/sample_top32: $(HERE)sample_top32.cpp $(HERE)ibin_io.hpp $(COMMON)/dataset_io.hpp $(LOGGER_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX_CUVS) $(CXXFLAGS_COMMON) $(DEFINES_CUVS) $(INCLUDES_CUVS) \
		$(HERE)sample_top32.cpp $(LOGGER_SRCS) \
		-o $@ $(LDFLAGS_CUVS) $(LIBS_CUVS)

$(BIN_DIR)/edge_length_stats: $(HERE)edge_length_stats.cpp $(HERE)csr_io.hpp $(HERE)ibin_io.hpp $(COMMON)/dataset_io.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX_CPU) $(CXXFLAGS_COMMON) -I$(HERE) -I$(COMMON) \
		$(HERE)edge_length_stats.cpp -o $@ -lgomp -lpthread -lm

# Needs the patched libcuvs: optimize() reads CUVS_CAGRA_NO_REVERSE only in the patched build.
$(BIN_DIR)/cagra_build_csr: $(HERE)cagra_build_csr.cpp $(COMMON)/dataset_io.hpp $(LOGGER_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX_CUVS) $(CXXFLAGS_COMMON) $(DEFINES_CUVS) $(INCLUDES_CUVS) \
		-I$(CUVS_SRC)/cpp/include \
		$(HERE)cagra_build_csr.cpp $(LOGGER_SRCS) \
		-o $@ $(LDFLAGS_CUVS) $(LIBS_CUVS)

clean:
	rm -f $(addprefix $(BIN_DIR)/,$(TOOLS))
