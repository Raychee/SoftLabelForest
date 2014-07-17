OS = $(shell uname)

O ?=

DAT_PATH = ../dat
RUN_PATH = ../run
TMP_PATH = ../tmp

CC = g++

LD      = $(CC)
CFLAGS  = -std=c++11 -pipe
CLFLAGS = -fPIC
LFLAGS  =
LLFLAGS =
INCS 	=
LIBS    =

ifdef O
CFLAGS += -O$(O)
else
CFLAGS += -Wall -Wextra -pedantic -ggdb3
endif

ifeq ($(OS), Darwin)
ifeq ($(CC), clang++)
CFLAGS  += -stdlib=libc++
endif
CFLAGS     += -mmacosx-version-min=10.7
LFLAGS     += -Wl,-rpath,@executable_path/../lib
LLFLAGS    += -dynamiclib
INCS       += -I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/c++/v1
INCS       += -I/usr/local/lib/gcc/i686-apple-darwin11/4.2.1/include
LIBS       += -L/usr/local/Cellar/gcc/4.8.2/lib/gcc/x86_64-apple-darwin13.1.0/4.8.2
LIB_SUFFIX = dylib
else
LFLAGS     += -Wl,-rpath,'$$ORIGIN/../lib'
LLFLAGS    += -shared
LIB_SUFFIX = so
endif

CLFLAGS += $(CFLAGS)
LLFLAGS += $(LFLAGS)

ifeq ($(OS), Darwin)
INCLUDE_PYTHON = -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7
INCLUDE_NUMPY  = -I/Library/Python/2.7/site-packages/numpy/core/include
else
INCLUDE_PYTHON = -I/usr/include/python2.7
INCLUDE_NUMPY  = -I/usr/local/anaconda/lib/python2.7/site-packages/numpy/core/include
endif
