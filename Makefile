CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -I. -MMD -MP -O2

# Base name of the program
PROG_NAME = run

SRCS = main.cc value.cc
OBJS = $(SRCS:.cc=.o)
DEPS = $(SRCS:.cc=.d)

# --- Handle Windows and Linux ---
ifeq ($(OS),Windows_NT)
	# Windows Settings
	TARGET = $(PROG_NAME).exe
	# 'del' is the Windows command. 
	# /Q = Quiet (don't ask for confirmation), /F = Force
	CLEAN_CMD = del /Q /F
else
	# Linux/Mac Settings
	TARGET = $(PROG_NAME)
	CLEAN_CMD = rm -f
endif

# --- Targets ---

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $<

clean:
	$(CLEAN_CMD) $(OBJS) $(DEPS) $(TARGET)

softclean:
	$(CLEAN_CMD) $(OBJS) $(DEPS)

# Include the generated dependency files
-include $(DEPS)