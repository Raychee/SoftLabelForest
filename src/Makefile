include ../Makefile.inc

INCS  += $(INCLUDE_PYTHON) $(INCLUDE_NUMPY)
LIBS  += -lboost_python -lboost_system -lpython2.7 -lgomp
OBJ    = main_python_module.o SoftDecisionModel.o Q.o
TARGET = $(RUN_PATH)/SoftDecisionModel.so

# $(CC) [-c] {$<|$^} $({CFLAGS|CLFLAGS}) $({LFLAGS|LLFLAGS}) $(INCS) [$(LIBS)] -o $@

.PHONY: all test check

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(LLFLAGS) $(LIBS) -o $@

main_python_module.o: main_python_module.cpp SoftDecisionModel.hpp \
					  Data.hpp Data.inl Q.hpp
	$(CC) $< -c $(CLFLAGS) $(INCS) -o $@

SoftDecisionModel.o: SoftDecisionModel.cpp SoftDecisionModel.hpp \
					 GDSolver.hpp GDSolver.inl Q.hpp Data.hpp Data.inl \
					 Array.hpp
	$(CC) $< -c $(CLFLAGS) -fopenmp $(INCS) -o $@

Q.o: Q.cpp Q.hpp
	$(CC) $< -c $(CLFLAGS) $(INCS) -o $@

test:
	$(CC) test_python.cpp $(CLFLAGS) $(LLFLAGS) $(INCS) $(LIBS) -o $(RUN_PATH)/tp.so

check: SoftDecisionModel.cpp main_python_module.cpp test_python.cpp
	$(CC) $^ $(CFLAGS) -fsyntax-only $(INCS)
