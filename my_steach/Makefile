# Makefile для GStreamer nvdsstitch plugin

# Компилятор и флаги
CXX = g++
NVCC = /usr/local/cuda-12.6/bin/nvcc

# Базовые флаги компиляции
CXXFLAGS = -fPIC -Wall -Wextra -O3 -std=c++14
NVCCFLAGS = -m64 -Xcompiler -fPIC -Xcompiler -fvisibility=hidden
NVCCFLAGS += -gencode arch=compute_87,code=sm_87
NVCCFLAGS += -O3 -use_fast_math --expt-relaxed-constexpr

# Пути к заголовочным файлам
INCLUDES = -I. \
           -Isrc \
           -I/usr/local/cuda-12.6/include \
           -I/opt/nvidia/deepstream/deepstream-7.1/sources/includes

# GStreamer флаги
GST_CFLAGS = $(shell pkg-config --cflags gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
GST_LIBS = $(shell pkg-config --libs gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)

# DeepStream библиотеки
DS_LIBS = -L/opt/nvidia/deepstream/deepstream-7.1/lib \
          -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvbufsurface -lnvbufsurftransform

# CUDA библиотеки
CUDA_LIBS = -L/usr/local/cuda-12.6/lib64 -lcudart -lcuda

# Целевая библиотека
TARGET = libnvdsstitch.so

# Исходные файлы
CPP_SOURCES = src/gstnvdsstitch.cpp \
              src/gstnvdsstitch_allocator.cpp

# CUDA исходные файлы - оригинальная версия
CUDA_SOURCES = src/cuda_stitch_kernel.cu \

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
NC = \033[0m # No Color

# Основная цель
all: check-cuda $(TARGET)
	@echo -e "$(GREEN)✓ Build complete!$(NC)"
	@echo -e "$(BLUE)Plugin location: $(PWD)/$(TARGET)$(NC)"

# Проверка CUDA
check-cuda:
	@echo -e "$(BLUE)Checking CUDA installation...$(NC)"
	@if [ ! -f "$(NVCC)" ]; then \
		echo -e "$(RED)Error: NVCC not found at $(NVCC)$(NC)"; \
		exit 1; \
	fi
	@$(NVCC) --version | head -n 1

# Линковка библиотеки
$(TARGET): $(OBJECTS)
	@echo -e "$(YELLOW)🔗 Linking $(TARGET)...$(NC)"
	$(CXX) -shared -o $@ $^ $(GST_LIBS) $(DS_LIBS) $(CUDA_LIBS)
	@echo -e "$(GREEN)✓ Library created: $@$(NC)"

# Компиляция C++ файлов
src/%.o: src/%.cpp
	@echo -e "$(BLUE)🔨 Compiling $<...$(NC)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GST_CFLAGS) -c $< -o $@

# Компиляция CUDA файлов
src/%.o: src/%.cu
	@echo -e "$(YELLOW)🚀 Compiling CUDA $<...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -std=c++14 -c $< -o $@



# Специальное правило для ОПТИМИЗИРОВАННОГО texture kernel
src/cuda_stitch_kernel_optimized.o: src/cuda_stitch_kernel_optimized.cu
	@echo -e "$(BLUE)🚀 Compiling OPTIMIZED TEXTURE kernel...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -std=c++14 -c $< -o $@
	@echo -e "$(GREEN)✓ Optimized texture kernel compiled$(NC)"

# Очистка
clean:
	@echo -e "$(YELLOW)🧹 Cleaning...$(NC)"
	rm -f $(OBJECTS) $(TARGET)
	rm -f src/*.o
	@echo -e "$(GREEN)✓ Clean complete$(NC)"

# Установка плагина
install: $(TARGET)
	@echo -e "$(YELLOW)📦 Installing plugin...$(NC)"
	@mkdir -p ~/.local/share/gstreamer-1.0/plugins/
	@cp $(TARGET) ~/.local/share/gstreamer-1.0/plugins/
	@echo -e "$(GREEN)✓ Plugin installed to ~/.local/share/gstreamer-1.0/plugins/$(NC)"
	@echo -e "$(BLUE)Run 'gst-inspect-1.0 nvdsstitch' to verify$(NC)"

# Удаление плагина
uninstall:
	@echo -e "$(YELLOW)🗑️  Uninstalling plugin...$(NC)"
	@rm -f ~/.local/share/gstreamer-1.0/plugins/$(TARGET)
	@echo -e "$(GREEN)✓ Plugin uninstalled$(NC)"

# Проверка версии
check-version: $(TARGET)
	@echo -e "$(BLUE)=== nvdsstitch Version Check ===$(NC)"
	@echo -e "$(YELLOW)Library info:$(NC)"
	@ls -lh $(TARGET)
	@echo -e "\n$(YELLOW)CUDA kernels in library:$(NC)"
	@echo -e "\n$(YELLOW)Available kernels:$(NC)"
	@echo -e "  $(BLUE)• Texture kernel (standard)$(NC)"
	@echo -e "  $(GREEN)• Sorted kernel (optimized)$(NC)"

# Тестовый запуск
test: $(TARGET)
	@echo -e "$(BLUE)🧪 Running test pipeline...$(NC)"
	@echo -e "$(YELLOW)Watch for kernel selection messages...$(NC)"
	GST_DEBUG=nvdsstitch:5 gst-launch-1.0 \
		videotestsrc pattern=ball num-buffers=100 ! \
		video/x-raw,width=3840,height=2160,framerate=30/1 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! mux.sink_0 \
		videotestsrc pattern=snow num-buffers=100 ! \
		video/x-raw,width=3840,height=2160,framerate=30/1 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! mux.sink_1 \
		nvstreammux name=mux batch-size=2 width=3840 height=2160 ! \
		nvdsstitch ! fakesink 2>&1 | grep -E "(SORTED|TEXTURE|Sequential|kernel)"

# Профилирование
profile: $(TARGET)
	@echo -e "$(BLUE)📊 Profiling with Nsight Systems...$(NC)"
	nsys profile --stats=true --output=stitch_profile \
		gst-launch-1.0 \
		videotestsrc num-buffers=300 ! \
		video/x-raw,width=3840,height=2160,framerate=30/1 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! mux.sink_0 \
		videotestsrc num-buffers=300 ! \
		video/x-raw,width=3840,height=2160,framerate=30/1 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! mux.sink_1 \
		nvstreammux name=mux batch-size=2 width=3840 height=2160 ! \
		nvdsstitch ! fakesink
	@echo -e "$(GREEN)✓ Profile saved to stitch_profile.nsys-rep$(NC)"

# Сравнение производительности

# Помощь
help:
	@echo -e "$(BLUE)nvdsstitch Makefile$(NC)"
	@echo -e "$(YELLOW)Available targets:$(NC)"
	@echo -e "  $(GREEN)make$(NC)              - Build the plugin"
	@echo -e "  $(GREEN)make clean$(NC)        - Clean build files"
	@echo -e "  $(GREEN)make install$(NC)      - Install plugin to user directory"
	@echo -e "  $(GREEN)make uninstall$(NC)    - Remove installed plugin"
	@echo -e "  $(GREEN)make check-version$(NC)- Check which kernels are included"
	@echo -e "  $(GREEN)make test$(NC)         - Run test pipeline"
	@echo -e "  $(GREEN)make profile$(NC)      - Profile with Nsight Systems"
	@echo -e "  $(GREEN)make benchmark$(NC)    - Compare kernel performance"
	@echo -e "  $(GREEN)make help$(NC)         - Show this help"

.PHONY: all clean install uninstall check-cuda check-version test profile benchmark help

# Зависимости заголовочных файлов
$(CPP_OBJECTS): src/gstnvdsstitch.h src/nvdsstitch_config.h src/cuda_stitch_kernel.h
src/cuda_stitch_kernel.o: src/cuda_stitch_kernel.h src/nvdsstitch_config.h