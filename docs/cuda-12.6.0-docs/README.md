# CUDA 12.6.0 Documentation

This directory contains local copies of NVIDIA CUDA 12.6.0 documentation for offline reference.

## Required Documentation

### 1. CUDA C++ Programming Guide
- **Filename**: `cuda-c-programming-guide.html` or `cuda-c-programming-guide.pdf`
- **Online URL**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- **Purpose**: Core CUDA programming concepts, architecture, API reference

### 2. CUDA C++ Best Practices Guide
- **Filename**: `cuda-c-best-practices-guide.html` or `cuda-c-best-practices-guide.pdf`
- **Online URL**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- **Purpose**: Performance optimization, best practices, common pitfalls

### 3. CUDA for Tegra
- **Filename**: `cuda-for-tegra.html` or `cuda-for-tegra.pdf`
- **Online URL**: https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html
- **Purpose**: Jetson-specific optimizations and constraints

## How to Download

### Option 1: Download HTML Documentation
```bash
# Navigate to this directory
cd /home/user/ds_pipeline/docs/cuda-12.6.0-docs

# Download CUDA Programming Guide (if wget/curl available)
wget -O cuda-c-programming-guide.html https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

# Download Best Practices Guide
wget -O cuda-c-best-practices-guide.html https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

# Download CUDA for Tegra
wget -O cuda-for-tegra.html https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html
```

### Option 2: Download PDF Documentation
Visit NVIDIA's documentation site and download PDF versions:
1. Go to https://docs.nvidia.com/cuda/
2. Navigate to each guide
3. Look for "PDF Download" link
4. Save to this directory

### Option 3: Use NVIDIA Documentation Archive
If specific version 12.6.0 is needed:
1. Visit https://docs.nvidia.com/cuda/archive/12.6.0/
2. Download HTML or PDF documentation
3. Extract to this directory

## Directory Structure (Expected)

```
docs/cuda-12.6.0-docs/
├── README.md (this file)
├── cuda-c-programming-guide.html (or .pdf)
├── cuda-c-best-practices-guide.html (or .pdf)
├── cuda-for-tegra.html (or .pdf)
└── [optional] Other CUDA 12.6 documentation
```

## Usage in Project

These local documentation files are referenced in:
- `docs/DOCS_NOTES.md` - Best practices and examples
- `CLAUDE.md` - Project rules and guidelines (after enhancement)

## Why Local Documentation?

1. **Offline Access**: Work without internet connection
2. **Version Control**: Ensure consistent documentation across team
3. **Fast Reference**: No network latency
4. **Specific Version**: Lock to CUDA 12.6.0 for Jetson Orin NX

## Additional Resources

- **Jetson Orin Documentation**: `docs/hw_arch/nvidia_jetson_orin_nx_16GB_super_arch.pdf`
- **DeepStream Documentation**: `docs/ds_doc/7.1/`
- **Camera Documentation**: `docs/camera_doc/`

---

**Last Updated**: 2025-11-19
**CUDA Version**: 12.6.0
**Target Platform**: NVIDIA Jetson Orin NX 16GB
