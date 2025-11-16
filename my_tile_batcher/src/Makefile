# Makefile для GStreamer nvtilebatcher plugin с CUDA ядром и Allocator
# Плагин для разбиения панорамного видео на тайлы

# Компилятор и флаги
CXX = g++
NVCC = /usr/local/cuda-12.6/bin/nvcc

# Базовые флаги компиляции
CXXFLAGS = -fPIC -D__aarch64__ -Wall -Wextra -O3 -std=c++14 -DPACKAGE=\"nvtilebatcher\"
NVCCFLAGS = -m64 -Xcompiler -fPIC -Xcompiler -fvisibility=hidden
NVCCFLAGS += -gencode arch=compute_87,code=sm_87
NVCCFLAGS += -O3 -use_fast_math --expt-relaxed-constexpr

# Пути к заголовочным файлам
INCLUDES = -I. \
           -I/usr/local/cuda-12.6/include \
           -I/opt/nvidia/deepstream/deepstream-7.1/sources/includes \
           -I/opt/nvidia/deepstream/deepstream-7.1/sources/gst-plugins/gst-nvmsgconv \
           -I/opt/nvidia/deepstream/deepstream-7.1/sources/gst-plugins/gst-nvmsgbroker

# GStreamer флаги
GST_CFLAGS = $(shell pkg-config --cflags gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
GST_LIBS = $(shell pkg-config --libs gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)

# DeepStream библиотеки
DS_LIBS = -L/opt/nvidia/deepstream/deepstream-7.1/lib \
          -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper \
          -lnvbufsurface -lnvbufsurftransform \
          -lnvds_batch_jpegenc

# CUDA библиотеки
CUDA_LIBS = -L/usr/local/cuda-12.6/lib64 -lcudart -lcuda

# Целевая библиотека
TARGET = libnvtilebatcher.so

# Исходные файлы
CPP_SOURCES = gstnvtilebatcher.cpp gstnvtilebatcher_allocator.cpp
CUDA_SOURCES = cuda_tile_extractor.cu

# Объектные файлы
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Все объектные файлы
OBJECTS = $(CPP_OBJECTS) $(CUDA_OBJECTS)

# Цветовые коды для красивого вывода
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
CYAN = \033[0;36m
MAGENTA = \033[0;35m
NC = \033[0m # No Color

# Основная цель
all: check-deps $(TARGET)
	@echo -e "$(GREEN)✓ Build complete!$(NC)"
	@echo -e "$(BLUE)Plugin location: $(PWD)/$(TARGET)$(NC)"
	@echo -e "$(CYAN)Components built:$(NC)"
	@echo -e "  - Main plugin (gstnvtilebatcher.o)"
	@echo -e "  - Custom allocator (gstnvtilebatcher_allocator.o)"
	@echo -e "  - CUDA kernel (cuda_tile_extractor.o)"

# Проверка зависимостей
check-deps:
	@echo -e "$(BLUE)Checking dependencies...$(NC)"
	@if [ ! -f "$(NVCC)" ]; then \
		echo -e "$(RED)Error: NVCC not found at $(NVCC)$(NC)"; \
		exit 1; \
	fi
	@if [ ! -d "/opt/nvidia/deepstream/deepstream-7.1" ]; then \
		echo -e "$(RED)Error: DeepStream SDK not found$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(GREEN)✓ All dependencies found$(NC)"

# Линковка библиотеки
$(TARGET): $(OBJECTS)
	@echo -e "$(YELLOW)🔗 Linking $(TARGET) with CUDA support and custom allocator...$(NC)"
	$(CXX) -shared -o $@ $^ $(GST_LIBS) $(DS_LIBS) $(CUDA_LIBS)
	@echo -e "$(GREEN)✓ Library created: $@$(NC)"
	@echo -e "$(CYAN)  Size: $$(du -h $@ | cut -f1)$(NC)"

# Компиляция C++ файлов
%.o: %.cpp
	@echo -e "$(BLUE)🔨 Compiling $<...$(NC)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GST_CFLAGS) -c $< -o $@

# Компиляция CUDA файлов
%.o: %.cu
	@echo -e "$(MAGENTA)🚀 Compiling CUDA kernel $<...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Очистка
clean:
	@echo -e "$(YELLOW)🧹 Cleaning...$(NC)"
	rm -f $(OBJECTS) $(TARGET)
	rm -f *.o
	@echo -e "$(GREEN)✓ Clean complete$(NC)"

# Установка плагина
install: $(TARGET)
	@echo -e "$(YELLOW)📦 Installing plugin...$(NC)"
	@mkdir -p ~/.local/share/gstreamer-1.0/plugins/
	@cp $(TARGET) ~/.local/share/gstreamer-1.0/plugins/
	@echo -e "$(GREEN)✓ Plugin installed to ~/.local/share/gstreamer-1.0/plugins/$(NC)"
	@echo -e "$(CYAN)✓ Components integrated:$(NC)"
	@echo -e "  - CUDA kernel for tile extraction"
	@echo -e "  - Custom allocator for memory management"
	@echo -e "$(BLUE)Run 'gst-inspect-1.0 nvtilebatcher' to verify$(NC)"

# Установка в системную директорию (требует sudo)
install-system: $(TARGET)
	@echo -e "$(YELLOW)📦 Installing plugin system-wide...$(NC)"
	@if [ -w "/usr/lib/aarch64-linux-gnu/gstreamer-1.0/" ]; then \
		cp $(TARGET) /usr/lib/aarch64-linux-gnu/gstreamer-1.0/; \
		echo -e "$(GREEN)✓ Plugin installed to system directory$(NC)"; \
	else \
		echo -e "$(RED)Error: Need sudo permissions. Run: sudo make install-system$(NC)"; \
		exit 1; \
	fi

# Удаление плагина
uninstall:
	@echo -e "$(YELLOW)🗑️  Uninstalling plugin...$(NC)"
	@rm -f ~/.local/share/gstreamer-1.0/plugins/$(TARGET)
	@echo -e "$(GREEN)✓ Plugin uninstalled$(NC)"

# Проверка плагина
check: $(TARGET) install
	@echo -e "$(BLUE)=== nvtilebatcher Plugin Check ===$(NC)"
	@echo -e "$(YELLOW)Plugin info:$(NC)"
	@gst-inspect-1.0 nvtilebatcher || echo -e "$(RED)Plugin not found in GStreamer registry$(NC)"

# Тестовый запуск с тестовым видео
test-simple: $(TARGET) install
	@echo -e "$(MAGENTA)🚀 Testing with test video...$(NC)"
	GST_DEBUG=nvtilebatcher:4 gst-launch-1.0 -v \
		videotestsrc num-buffers=30 pattern=ball ! \
		video/x-raw,width=6528,height=1632,framerate=30/1 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! \
		nvtilebatcher gpu-id=0 silent=false ! \
		fakesink sync=false
	@echo -e "$(GREEN)✓ Simple test complete$(NC)"

# Тест с реальным изображением
test-image: $(TARGET) install
	@echo -e "$(BLUE)🧪 Testing with panoramic image...$(NC)"
	@if [ ! -f "panorama.jpg" ]; then \
		echo -e "$(RED)Error: panorama.jpg not found$(NC)"; \
		echo -e "$(YELLOW)Please provide a 6528x1632 panorama image$(NC)"; \
		exit 1; \
	fi
	gst-launch-1.0 -v \
		filesrc location=panorama.jpg ! \
		jpegdec ! \
		videoconvert ! \
		video/x-raw,format=RGBA,width=6528,height=1632 ! \
		nvvideoconvert ! \
		'video/x-raw(memory:NVMM),format=RGBA' ! \
		nvtilebatcher gpu-id=0 ! \
		fakesink
	@echo -e "$(GREEN)✓ Image test complete$(NC)"

# Тест с Python
test-python: $(TARGET) install
	@echo -e "$(BLUE)🐍 Running Python test script...$(NC)"
	@if [ -f "test_tilebatcher.py" ]; then \
		python3 test_tilebatcher.py; \
	else \
		echo -e "$(RED)Error: test_tilebatcher.py not found$(NC)"; \
		exit 1; \
	fi

# Проверка утечек памяти с valgrind
test-memcheck: $(TARGET) install
	@echo -e "$(YELLOW)🔍 Checking for memory leaks...$(NC)"
	valgrind --leak-check=full --show-leak-kinds=all \
		gst-launch-1.0 \
		videotestsrc num-buffers=10 ! \
		video/x-raw,width=6528,height=1632,framerate=30/1 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! \
		nvtilebatcher ! \
		fakesink
	@echo -e "$(GREEN)✓ Memory check complete$(NC)"

# Бенчмарк производительности
benchmark: $(TARGET) install
	@echo -e "$(MAGENTA)📊 Benchmarking performance (300 frames)...$(NC)"
	@echo -e "$(CYAN)Processing 300 frames at 30fps...$(NC)"
	@time GST_DEBUG=nvtilebatcher:2 gst-launch-1.0 \
		videotestsrc num-buffers=300 ! \
		video/x-raw,width=6528,height=1632,framerate=30/1 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! \
		nvtilebatcher silent=false ! \
		fakesink sync=false
	@echo -e "$(GREEN)✓ Benchmark complete$(NC)"

# Профилирование с nvprof
profile: $(TARGET) install
	@echo -e "$(BLUE)📊 Profiling CUDA kernels...$(NC)"
	@which nvprof >/dev/null 2>&1 || (echo -e "$(RED)nvprof not found. Try nsys instead$(NC)" && exit 1)
	nvprof --print-gpu-summary \
		gst-launch-1.0 \
		videotestsrc num-buffers=30 ! \
		video/x-raw,width=6528,height=1632,framerate=30/1 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! \
		nvtilebatcher ! \
		fakesink sync=false
	@echo -e "$(GREEN)✓ Profiling complete$(NC)"

# Отладка с gdb
debug: $(TARGET) install
	@echo -e "$(YELLOW)🔍 Starting debugger...$(NC)"
	GST_DEBUG=nvtilebatcher:5 gdb --args gst-launch-1.0 \
		videotestsrc num-buffers=1 ! \
		video/x-raw,width=6528,height=1632,framerate=30/1 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! \
		nvtilebatcher ! \
		fakesink

# Помощь
help:
	@echo -e "$(BLUE)nvtilebatcher Makefile with Custom Allocator$(NC)"
	@echo -e "$(YELLOW)Build targets:$(NC)"
	@echo -e "  $(GREEN)make$(NC)              - Build plugin with allocator"
	@echo -e "  $(GREEN)make clean$(NC)        - Clean all build files"
	@echo -e "  $(GREEN)make install$(NC)      - Install to user directory"
	@echo -e "  $(GREEN)make uninstall$(NC)    - Remove installed plugin"
	@echo -e ""
	@echo -e "$(YELLOW)Testing targets:$(NC)"
	@echo -e "  $(GREEN)make test-simple$(NC)  - Test with test video"
	@echo -e "  $(GREEN)make test-image$(NC)   - Test with panorama.jpg"
	@echo -e "  $(GREEN)make test-python$(NC)  - Run Python test script"
	@echo -e "  $(GREEN)make test-memcheck$(NC) - Check for memory leaks"
	@echo -e "  $(GREEN)make benchmark$(NC)    - Benchmark performance"
	@echo -e "  $(GREEN)make profile$(NC)      - Profile CUDA kernels"
	@echo -e "  $(GREEN)make debug$(NC)        - Debug with gdb"
	@echo -e ""
	@echo -e "$(YELLOW)Other:$(NC)"
	@echo -e "  $(GREEN)make check$(NC)        - Verify plugin installation"
	@echo -e "  $(GREEN)make help$(NC)         - Show this help"

.PHONY: all clean install install-system uninstall check test-simple test-image test-python test-memcheck benchmark profile debug help check-deps

# Зависимости
gstnvtilebatcher.o: gstnvtilebatcher.h gstnvtilebatcher_allocator.h
gstnvtilebatcher_allocator.o: gstnvtilebatcher_allocator.h
cuda_tile_extractor.o: cuda_tile_extractor.cu