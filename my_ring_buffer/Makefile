# Makefile для GStreamer Ring Buffer plugin
# GPU COPY VERSION with CUDA

# Компилятор и флаги
CXX = g++

# Базовые флаги компиляции
CXXFLAGS = -fPIC -Wall -Wextra -O3 -std=c++14 -DPACKAGE=\"nvdsringbuf\"

# Пути к заголовочным файлам
INCLUDES = -I. \
           -Isrc \
           -I/opt/nvidia/deepstream/deepstream-7.1/sources/includes \
           -I/usr/local/cuda/include

# GStreamer флаги
GST_CFLAGS = $(shell pkg-config --cflags gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
GST_LIBS = $(shell pkg-config --libs gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)

# DeepStream библиотеки (для работы с NVMM) - как в nvdsstitch
DS_LIBS = -L/opt/nvidia/deepstream/deepstream-7.1/lib \
          -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvbufsurface -lnvbufsurftransform

# CUDA библиотеки (для GPU→GPU копирования и EGL interop)
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart -lcuda

# Целевая библиотека
TARGET = libgstnvdsringbuf.so

# Исходные файлы
CPP_SOURCES = src/gstnvdsringbuf.cpp

# Объектные файлы
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

# Цветовые коды для красивого вывода
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Основная цель
all: $(TARGET)
	@echo -e "$(GREEN)✓ Ring Buffer Plugin build complete!$(NC)"
	@echo -e "$(BLUE)Plugin location: $(PWD)/$(TARGET)$(NC)"
	@echo -e "$(YELLOW)Next steps:$(NC)"
	@echo -e "  1. Run 'make install' to install the plugin"
	@echo -e "  2. Run 'make test' to test basic functionality"
	@echo -e "  3. Use 'GST_PLUGIN_PATH=$$PWD gst-inspect-1.0 ringbuffer' to verify"

# Линковка библиотеки
$(TARGET): $(CPP_OBJECTS)
	@echo -e "$(YELLOW)🔗 Linking $(TARGET)...$(NC)"
	$(CXX) -shared -o $@ $^ $(GST_LIBS) $(DS_LIBS) $(CUDA_LIBS)
	@echo -e "$(GREEN)✓ Library created: $@$(NC)"

# Компиляция C++ файлов
src/%.o: src/%.cpp
	@echo -e "$(BLUE)🔨 Compiling $<...$(NC)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GST_CFLAGS) -c $< -o $@

# Очистка
clean:
	@echo -e "$(YELLOW)🧹 Cleaning...$(NC)"
	rm -f $(CPP_OBJECTS) $(TARGET)
	rm -f src/*.o
	@echo -e "$(GREEN)✓ Clean complete$(NC)"

# Установка плагина
install: $(TARGET)
	@echo -e "$(YELLOW)📦 Installing plugin...$(NC)"
	@mkdir -p ~/.local/share/gstreamer-1.0/plugins/
	@cp $(TARGET) ~/.local/share/gstreamer-1.0/plugins/
	@echo -e "$(GREEN)✓ Plugin installed to ~/.local/share/gstreamer-1.0/plugins/$(NC)"
	@echo -e "$(BLUE)Run 'gst-inspect-1.0 ringbuffer' to verify$(NC)"

# Удаление плагина
uninstall:
	@echo -e "$(YELLOW)🗑️  Uninstalling plugin...$(NC)"
	@rm -f ~/.local/share/gstreamer-1.0/plugins/$(TARGET)
	@echo -e "$(GREEN)✓ Plugin uninstalled$(NC)"

# Проверка плагина
inspect: $(TARGET)
	@echo -e "$(BLUE)🔍 Inspecting ring buffer plugin...$(NC)"
	GST_PLUGIN_PATH=$(PWD) gst-inspect-1.0 ringbuffer

# Тестовый запуск (простой passthrough)
test: $(TARGET)
	@echo -e "$(BLUE)🧪 Running test pipeline (passthrough mode)...$(NC)"
	@echo -e "$(YELLOW)This will test that plugin loads correctly$(NC)"
	GST_PLUGIN_PATH=$(PWD) GST_DEBUG=ringbuffer:5 gst-launch-1.0 \
		videotestsrc num-buffers=100 ! \
		video/x-raw,width=1920,height=1080,framerate=30/1 ! \
		nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! \
		ringbuffer max-buffers=10 delay-seconds=1.0 ! \
		fakesink 2>&1 | grep -E "(ringbuffer|Ring|received)"

# Помощь
help:
	@echo -e "$(BLUE)Ring Buffer Plugin Makefile$(NC)"
	@echo -e "$(YELLOW)Available targets:$(NC)"
	@echo -e "  $(GREEN)make$(NC)              - Build the plugin"
	@echo -e "  $(GREEN)make clean$(NC)        - Clean build files"
	@echo -e "  $(GREEN)make install$(NC)      - Install plugin to user directory"
	@echo -e "  $(GREEN)make uninstall$(NC)    - Remove installed plugin"
	@echo -e "  $(GREEN)make inspect$(NC)      - Inspect plugin with gst-inspect"
	@echo -e "  $(GREEN)make test$(NC)         - Run test pipeline"
	@echo -e "  $(GREEN)make help$(NC)         - Show this help"
	@echo ""
	@echo -e "$(BLUE)Configuration:$(NC)"
	@echo -e "  Default max-buffers: 210"
	@echo -e "  Default delay: 7.0 seconds"
	@echo -e "  Supported format: NVMM RGBA only"

.PHONY: all clean install uninstall inspect test help

# Зависимости заголовочных файлов
$(CPP_OBJECTS): src/gstnvdsringbuf.h
