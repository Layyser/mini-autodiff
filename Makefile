CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -I. -MMD -MP -O2

# Base name of the program
PROG_NAME = run

# Build output directory: keeps .o/.d files (and binaries) out of the
# source tree, mirroring the src/ subdirectory layout.
BUILD_DIR = build

# --- Sources, split the way real engines do: ---
#   core/  - the autodiff engine itself (tensor, compute graph, memory, ops)
#   nn/    - higher-level API built on top of core (optimizers, losses)
#   main.cc - example program
SRCS = main.cc core/tensor.cc core/ops.cc nn/optimizer.cc
OBJS = $(addprefix $(BUILD_DIR)/,$(SRCS:.cc=.o))
DEPS = $(OBJS:.o=.d)

# --- Test sources (everything except main.cc, plus the test driver) ---
LIB_SRCS = $(filter-out main.cc,$(SRCS))
LIB_OBJS = $(addprefix $(BUILD_DIR)/,$(LIB_SRCS:.cc=.o))
TEST_SRCS = tests/test_gradcheck.cc
TEST_OBJS = $(addprefix $(BUILD_DIR)/,$(TEST_SRCS:.cc=.o))
TEST_DEPS = $(TEST_OBJS:.o=.d)
TEST_NAME = test_gradcheck

# --- Handle Windows and Linux ---
ifeq ($(OS),Windows_NT)
	# Windows Settings
	TARGET = $(BUILD_DIR)/$(PROG_NAME).exe
	TEST_TARGET = $(BUILD_DIR)/$(TEST_NAME).exe
	# 'rmdir /S /Q' recursively force-removes a directory tree.
	CLEAN_CMD = rmdir /S /Q
else
	# Linux/Mac Settings
	TARGET = $(BUILD_DIR)/$(PROG_NAME)
	TEST_TARGET = $(BUILD_DIR)/$(TEST_NAME)
	CLEAN_CMD = rm -rf
endif

# --- Targets ---

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- Tests ---
test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(LIB_OBJS) $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	$(CLEAN_CMD) $(BUILD_DIR)

# Include the generated dependency files
-include $(DEPS)
-include $(TEST_DEPS)
