# Makefile для GStreamer nvdsvirtualcam plugin - ФИНАЛЬНАЯ ВЕРСИЯ

# Компилятор и флаги
CXX = g++
NVCC = /usr/local/cuda-12.6/bin/nvcc

# Базовые флаги компиляции
CXXFLAGS = -fPIC -Wall -Wextra -O3 -std=c++14
NVCCFLAGS = -m64 -Xcompiler -fPIC -Xcompiler -fvisibility=hidden
NVCCFLAGS += -gencode arch=compute_87,code=sm_87
NVCCFLAGS += -O3 -use_fast_math --expt-relaxed-constexpr
# Register usage validation for occupancy control (CLAUDE.md §4.5)
NVCCFLAGS += --ptxas-options=-v

# Пути к заголовочным файлам
INCLUDES = -I. \
           -I/usr/local/cuda-12.6/include \
           -I/opt/nvidia/deepstream/deepstream-7.1/sources/includes

# GStreamer флаги
GST_CFLAGS = $(shell pkg-config --cflags gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
GST_LIBS = $(shell pkg-config --libs gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)

# DeepStream библиотеки
DS_PATH = /opt/nvidia/deepstream/deepstream-7.1
DS_LIBS = -L$(DS_PATH)/lib \
          -lnvdsgst_helper \
          -lnvdsgst_meta \
          -lnvds_meta \
          -lnvbufsurface \
          -lnvbufsurftransform \
          -Wl,-rpath,$(DS_PATH)/lib

# CUDA библиотеки - БЕЗ cudaEGL для aarch64!
CUDA_LIBS = -L/usr/local/cuda-12.6/lib64 -lcudart -lcuda

# Для Jetson добавляем EGL если нужно
ifeq ($(shell uname -m),aarch64)
    # На Jetson используем стандартный EGL
    CUDA_LIBS += -L/usr/lib/aarch64-linux-gnu -lEGL
endif

# Целевая библиотека
TARGET = libnvdsvirtualcam.so

# Исходные файлы - включаем allocator
CPP_SOURCES = gstnvdsvirtualcam.cpp gstnvdsvirtualcam_allocator.cpp
CUDA_SOURCES = cuda_virtual_cam_kernel.cu

# Объектные файлы
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
OBJECTS = $(CPP_OBJECTS) $(CUDA_OBJECTS)

# Цветовые коды
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m

# Основная цель
all: check-deps $(TARGET)
	@echo -e "$(GREEN)✓ Build complete!$(NC)"
	@echo -e "$(BLUE)Plugin location: $(PWD)/$(TARGET)$(NC)"

# Проверка зависимостей
check-deps:
	@echo -e "$(BLUE)Checking dependencies...$(NC)"
	@if [ ! -f "gstnvdsvirtualcam_allocator.cpp" ]; then \
		echo -e "$(RED)Error: gstnvdsvirtualcam_allocator.cpp not found!$(NC)"; \
		echo -e "$(YELLOW)Please copy it from DeepStream sources$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "gstnvdsvirtualcam_allocator.h" ]; then \
		echo -e "$(RED)Error: gstnvdsvirtualcam_allocator.h not found!$(NC)"; \
		echo -e "$(YELLOW)Please copy it from DeepStream sources$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(GREEN)✓ All required files present$(NC)"

# Линковка библиотеки
$(TARGET): $(OBJECTS)
	@echo -e "$(YELLOW)🔗 Linking $(TARGET)...$(NC)"
	@echo -e "$(BLUE)Objects: $(OBJECTS)$(NC)"
	$(CXX) -shared -o $@ $^ $(DS_LIBS) $(CUDA_LIBS) $(GST_LIBS)
	@echo -e "$(GREEN)✓ Library created$(NC)"
	@echo -e "$(BLUE)Checking library dependencies...$(NC)"
	@ldd $(TARGET) | grep -E "(nvds|cuda)" || true

# Компиляция C++ файлов
%.o: %.cpp
	@echo -e "$(BLUE)🔨 Compiling $<...$(NC)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GST_CFLAGS) -c $< -o $@

# Компиляция CUDA файлов с подавлением warnings
%.o: %.cu
	@echo -e "$(YELLOW)🚀 Compiling CUDA $<...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -std=c++14 -Xcudafe "--diag_suppress=1301" -c $< -o $@

# Очистка
clean:
	@echo -e "$(YELLOW)🧹 Cleaning...$(NC)"
	rm -f $(OBJECTS) $(TARGET)
	rm -rf ~/.cache/gstreamer-1.0/
	@echo -e "$(GREEN)✓ Clean complete$(NC)"

# Установка локально
install: $(TARGET)
	@echo -e "$(YELLOW)📦 Installing plugin...$(NC)"
	@mkdir -p ~/.local/share/gstreamer-1.0/plugins/
	@cp $(TARGET) ~/.local/share/gstreamer-1.0/plugins/
	@echo -e "$(GREEN)✓ Plugin installed to ~/.local/share/gstreamer-1.0/plugins/$(NC)"
	@rm -rf ~/.cache/gstreamer-1.0/

# Проверка загрузки плагина
test-load: $(TARGET) install
	@echo -e "$(BLUE)🔍 Testing plugin load...$(NC)"
	@export LD_LIBRARY_PATH=$(DS_PATH)/lib:$$LD_LIBRARY_PATH; \
	export GST_PLUGIN_PATH=~/.local/share/gstreamer-1.0/plugins:$$GST_PLUGIN_PATH; \
	gst-inspect-1.0 nvdsvirtualcam > /dev/null 2>&1 && \
		echo -e "$(GREEN)✓ Plugin loads successfully$(NC)" || \
		(echo -e "$(RED)✗ Plugin failed to load$(NC)" && \
		 echo -e "$(YELLOW)Run 'make debug' for details$(NC)")

# Отладка загрузки
debug: $(TARGET) install
	@echo -e "$(YELLOW)🔍 Debugging plugin load...$(NC)"
	export LD_LIBRARY_PATH=$(DS_PATH)/lib:$$LD_LIBRARY_PATH; \
	export GST_PLUGIN_PATH=~/.local/share/gstreamer-1.0/plugins:$$GST_PLUGIN_PATH; \
	GST_DEBUG=2 gst-inspect-1.0 nvdsvirtualcam 2>&1

# Детальная отладка
debug-verbose: $(TARGET) install
	@echo -e "$(YELLOW)🔍 Verbose debugging...$(NC)"
	export LD_LIBRARY_PATH=$(DS_PATH)/lib:$$LD_LIBRARY_PATH; \
	export GST_PLUGIN_PATH=~/.local/share/gstreamer-1.0/plugins:$$GST_PLUGIN_PATH; \
	GST_DEBUG=*:4 gst-inspect-1.0 nvdsvirtualcam 2>&1 | grep -E "(ERROR|WARN|nvds)"

# Проверка символов
check-symbols: $(TARGET)
	@echo -e "$(BLUE)📋 Checking exported symbols...$(NC)"
	@nm -D $(TARGET) | grep "plugin_init" || echo -e "$(RED)Warning: plugin_init not found!$(NC)"
	@echo -e "$(BLUE)📋 Checking undefined symbols...$(NC)"
	@ldd -r $(TARGET) 2>&1 | grep undefined || echo -e "$(GREEN)No undefined symbols$(NC)"

# Проверка зависимостей библиотеки
check-libs: $(TARGET)
	@echo -e "$(BLUE)📚 Library dependencies:$(NC)"
	@ldd $(TARGET)
	@echo -e "$(BLUE)📚 Missing libraries:$(NC)"
	@ldd $(TARGET) | grep "not found" || echo -e "$(GREEN)All libraries found$(NC)"

# Быстрый тест
test: $(TARGET) install
	@echo -e "$(BLUE)🧪 Running quick test...$(NC)"
	export LD_LIBRARY_PATH=$(DS_PATH)/lib:$$LD_LIBRARY_PATH; \
	export GST_PLUGIN_PATH=~/.local/share/gstreamer-1.0/plugins:$$GST_PLUGIN_PATH; \
	gst-launch-1.0 --gst-plugin-path=~/.local/share/gstreamer-1.0/plugins \
		videotestsrc num-buffers=1 ! \
		video/x-raw,width=4096,height=2048 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! \
		nvdsvirtualcam yaw=0 pitch=0 roll=0 fov=55 ! \
		fakesink 2>&1 | grep -E "(ERROR|WARNING)" || echo -e "$(GREEN)✓ Test passed$(NC)"

# Помощь
help:
	@echo -e "$(BLUE)nvdsvirtualcam Makefile$(NC)"
	@echo -e "$(YELLOW)Commands:$(NC)"
	@echo -e "  $(GREEN)make$(NC)              - Build the plugin"
	@echo -e "  $(GREEN)make clean$(NC)        - Clean build files"  
	@echo -e "  $(GREEN)make install$(NC)      - Install plugin locally"
	@echo -e "  $(GREEN)make test-load$(NC)    - Test if plugin loads"
	@echo -e "  $(GREEN)make debug$(NC)        - Debug plugin loading"
	@echo -e "  $(GREEN)make check-symbols$(NC) - Check exported symbols"
	@echo -e "  $(GREEN)make check-libs$(NC)   - Check library dependencies"
	@echo -e "  $(GREEN)make test$(NC)         - Run quick test"

.PHONY: all clean install test-load debug debug-verbose check-symbols check-libs test help check-deps