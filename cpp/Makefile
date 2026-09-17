# Build for the C++ port of the Qwen3.5 assistant.
#
#     make            build/ai (the assistant) and build/tests
#     make ai         just the assistant
#     make test       build and run the self-tests
#     make config     show which MLX this build resolved to
#     make bundle     copy MLX into build/, making the tree self-contained
#     make clean
#
# POINTING AT MLX
# ---------------
# MLX is used as an ordinary dynamic library: its headers plus libmlx.dylib.
# Three ways to say where it is, in order of precedence:
#
#   make MLX_DIR=/opt/mlx
#       A prefix laid out as <prefix>/include/mlx/mlx.h and
#       <prefix>/lib/libmlx.dylib. Both `cmake --install` and the Python wheel
#       produce this layout, so it is the usual answer.
#
#   make MLX_INCLUDE=~/src/mlx MLX_LIB=~/src/mlx-build/mlx
#       For when the headers and the library are not under a common prefix --
#       notably an UNINSTALLED cmake build tree, where the headers are the source
#       checkout itself. Either may be given alone; the other falls back to
#       MLX_DIR.
#
#   (nothing)
#       Autodetected: the mlx Python wheel first -- override the interpreter with
#       PYTHON=python3.12 if you have several -- then /opt/mlx, /usr/local,
#       /opt/homebrew.
#
# Whatever route is taken, the binary records MLX_LIB as its runtime search path,
# so it keeps working from any directory. `make bundle` instead copies MLX into
# build/lib and points the binary at @loader_path, which detaches it from the
# install entirely. `make config` prints what was resolved; every failure below
# names the knob that fixes it.

PYTHON ?= python3

# A literal comma, for use inside $(if ...) where commas separate arguments.
comma := ,

# Only go looking if the caller did not pin a prefix.
ifeq ($(origin MLX_DIR),undefined)
  MLX_WHEEL := $(shell $(PYTHON) -c 'import mlx; print(mlx.__path__[0])' 2>/dev/null)
  MLX_SEARCH := $(MLX_WHEEL) /opt/mlx /usr/local /opt/homebrew
  MLX_DIR := $(firstword \
      $(foreach d,$(MLX_SEARCH),$(if $(wildcard $(d)/include/mlx/mlx.h),$(d),)))
  # Found nothing, and no explicit header path either: report where we looked
  # rather than falling through and complaining about a bare '/include'.
  ifeq ($(strip $(MLX_DIR)),)
    ifeq ($(origin MLX_INCLUDE),undefined)
      $(error Could not find MLX. Looked for include/mlx/mlx.h under \
        $(if $(MLX_WHEEL),'$(MLX_WHEEL)' (the $(PYTHON) wheel)$(comma),\
        no mlx module for '$(PYTHON)'$(comma)) /opt/mlx, /usr/local and \
        /opt/homebrew. Fix with any one of: `pip3 install mlx`; \
        `make PYTHON=<interpreter that has mlx>`; `make MLX_DIR=<prefix>`; \
        `make MLX_INCLUDE=<headers> MLX_LIB=<dir with libmlx.dylib>`)
    endif
  endif
endif

MLX_INCLUDE ?= $(MLX_DIR)/include
MLX_LIB     ?= $(MLX_DIR)/lib

ifeq ($(wildcard $(MLX_INCLUDE)/mlx/mlx.h),)
  $(error Cannot find mlx/mlx.h under MLX_INCLUDE='$(MLX_INCLUDE)'. \
    Install the wheel (pip3 install mlx), or point the build at MLX: \
    `make MLX_DIR=/prefix` for an installed tree, or \
    `make MLX_INCLUDE=/path/to/mlx-source MLX_LIB=/path/to/build/mlx` for an \
    uninstalled cmake build. See the header of this Makefile, or run `make config`)
endif

ifeq ($(wildcard $(MLX_LIB)/libmlx.dylib),)
  $(error Cannot find libmlx.dylib under MLX_LIB='$(MLX_LIB)'. \
    Headers were found under '$(MLX_INCLUDE)', so this is likely a split layout: \
    pass MLX_LIB=<directory containing libmlx.dylib>. If you built MLX with \
    BUILD_SHARED_LIBS=OFF there is no dylib -- rebuild it shared, or use the wheel)
endif

MLX_VERSION := $(shell awk '\
  /define MLX_VERSION_MAJOR/{a=$$3} /define MLX_VERSION_MINOR/{b=$$3} \
  /define MLX_VERSION_PATCH/{c=$$3} END{if(a!="")print a"."b"."c}' \
  $(MLX_INCLUDE)/mlx/version.h 2>/dev/null)

# MLX loads mlx.metallib at runtime and looks for it next to libmlx.dylib. It is
# not needed to build, but without it every GPU kernel fails, so `make config`
# reports on it.
MLX_METALLIB := $(wildcard $(MLX_LIB)/mlx.metallib)

CXX ?= clang++
# MLX's headers go in with -isystem: bf16.h trips -Wdeprecated-copy once per
# translation unit, and it is not our code to fix.
CXXFLAGS += -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -MMD -MP \
            -I src -isystem $(MLX_INCLUDE)

# Where the binary looks for libmlx.dylib at runtime. `make bundle` overrides it.
RPATH ?= $(MLX_LIB)

# CoreFoundation is ours (Unicode NFC); libmlx pulls in Metal and Accelerate
# itself, so they do not need repeating here.
LDFLAGS += -L $(MLX_LIB) -lmlx -Wl,-rpath,$(RPATH) -framework CoreFoundation

# Everything except the two entry points is shared between the binaries.
COMMON := $(filter-out src/main.cpp src/tests.cpp,$(wildcard src/*.cpp))
COMMON_OBJ := $(COMMON:src/%.cpp=build/%.o)

.PHONY: all ai test config bundle clean clean-bin unicode
all: build/ai build/tests

ai: build/ai

build/ai: $(COMMON_OBJ) build/main.o
	$(CXX) $^ $(LDFLAGS) -o $@

build/tests: $(COMMON_OBJ) build/tests.o
	$(CXX) $^ $(LDFLAGS) -o $@

build/%.o: src/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: build/tests
	./build/tests

config:
	@echo "MLX version   $(if $(MLX_VERSION),$(MLX_VERSION),unknown)"
	@echo "MLX_INCLUDE   $(MLX_INCLUDE)"
	@echo "MLX_LIB       $(MLX_LIB)"
	@echo "RPATH         $(RPATH)"
	@echo "mlx.metallib  $(if $(MLX_METALLIB),$(MLX_METALLIB),\
NOT FOUND next to libmlx.dylib -- GPU kernels will fail to load)"
	@echo "CXX           $(CXX) ($(shell $(CXX) --version | head -1))"

# Self-contained tree: build/{ai,tests} + build/lib. MLX finds its metallib next
# to the dylib, so both have to travel together.
bundle:
	@mkdir -p build/lib
	cp $(MLX_LIB)/libmlx.dylib build/lib/
ifeq ($(MLX_METALLIB),)
	@echo "warning: no mlx.metallib next to libmlx.dylib; the bundle will not run on GPU"
else
	cp $(MLX_METALLIB) build/lib/
endif
	$(MAKE) clean-bin
	$(MAKE) RPATH='@loader_path/lib' build/ai build/tests
	@echo
	@echo "bundled: build/ is self-contained. Verify with:"
	@echo "    otool -L build/ai | grep mlx"

# Force a relink without recompiling every object.
clean-bin:
	rm -f build/ai build/tests

# Regenerate the Unicode category tables (needs python3; the output is committed
# so an ordinary build never runs this).
unicode:
	$(PYTHON) tools/gen_unicode.py > src/unicode_tables.cpp

clean:
	rm -rf build

-include $(wildcard build/*.d)
