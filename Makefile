CC := nvcc  # main compiler
# CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
TARGET := bin/half

SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name "*.$(SRCEXT)")
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -g -std=c++14 -arch=sm_60
CFLAGS += -gencode=arch=compute_60,code=sm60
CFLAGS += -gencode=arch=compute_61,code=sm_61
CFLAGS += -gencode=arch=compute_70,code=sm_70
CFLAGS += -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75
CFLAGS += -Xcompiler "-fopenmp -O3 -fomit-frame-pointer -fno-operator-names -msse3 -fexcess-precision=fast -funroll-loops -march=native -mfpmath=sse"
LIB := -lgomp -lcuda -lcudart -lcublas -lgsl
INC := -I include
MKDIR = mkdir -p bin

$(TARGET): $(OBJECTS)
	${MKDIR}
	@echo " Linking..."
	@echo " $(CC) $^ -o $(TARGET) $(LIB)"; $(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

.PHONY: clean
