#!/bin/bash
# Script to build GLIBC 2.32 and 2.38 for GPT-OSS vLLM compatibility
# This addresses the GLIBC version requirement issue on older systems

echo "Setting up GLIBC build environment..."

# Base paths - adjust to your system
export BASE_PATH=/scr/jshen3
export SRC="$BASE_PATH/src"
export GLIBC_32_PATH="$BASE_PATH/glibc"
export GLIBC_38_PATH="$BASE_PATH/glibc-2.38"

# Create directories
mkdir -p "$SRC"
echo "Created source directory: $SRC"

echo "=================================="
echo "Building GLIBC 2.32"
echo "=================================="

cd "$SRC"

# Download GLIBC 2.32 if not already present
if [ ! -f glibc-2.32.tar.gz ]; then
    echo "Downloading GLIBC 2.32..."
    wget -c https://ftp.gnu.org/gnu/glibc/glibc-2.32.tar.gz
fi

# Extract and build GLIBC 2.32
if [ ! -d glibc-2.32 ]; then
    echo "Extracting GLIBC 2.32..."
    tar -zxvf glibc-2.32.tar.gz
fi

cd glibc-2.32

# Create build directory
mkdir -p glibc-build
cd glibc-build

# Create installation directory
mkdir -p "$GLIBC_32_PATH"

echo "Configuring GLIBC 2.32..."
../configure --prefix="$GLIBC_32_PATH"

echo "Building GLIBC 2.32 (this will take 10-20 minutes)..."
make -j"$(nproc)"

echo "Installing GLIBC 2.32..."
make install

echo "✓ GLIBC 2.32 installed to: $GLIBC_32_PATH"

echo "=================================="
echo "Building GLIBC 2.38"
echo "=================================="

cd "$SRC"

# Download GLIBC 2.38 if not already present
if [ ! -f glibc-2.38.tar.xz ]; then
    echo "Downloading GLIBC 2.38..."
    wget -c https://ftp.gnu.org/gnu/glibc/glibc-2.38.tar.xz
fi

# Extract GLIBC 2.38
if [ ! -d glibc-2.38 ]; then
    echo "Extracting GLIBC 2.38..."
    tar -xf glibc-2.38.tar.xz
fi

# Create build directory
mkdir -p glibc-2.38-build
cd glibc-2.38-build

echo "Configuring GLIBC 2.38..."
../glibc-2.38/configure --prefix="$GLIBC_38_PATH" --disable-werror

echo "Building GLIBC 2.38 (this will take 15-25 minutes)..."
make -j"$(nproc)"

echo "Installing GLIBC 2.38..."
make install

echo "✓ GLIBC 2.38 installed to: $GLIBC_38_PATH"

echo "=================================="
echo "Build Summary"
echo "=================================="
echo "GLIBC 2.32 path: $GLIBC_32_PATH"
echo "GLIBC 2.38 path: $GLIBC_38_PATH"
echo ""
echo "You can now run: bash run_gpt_oss_20b.sh"
echo ""
echo "To test the custom loader manually:"
echo "export GLIBC_NEW=$GLIBC_38_PATH"
echo "export CONDA=/scr/jshen3/miniconda3/envs/gpt-oss"
echo 'export GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"'
echo 'export LD_LIBRARY_PATH="$GLIBC_NEW/lib:$CONDA/lib:$GCC_LIBDIR:${CUDA_HOME:+$CUDA_HOME/lib64}:$LD_LIBRARY_PATH"'
echo ""
echo '$GLIBC_NEW/lib/ld-linux-x86-64.so.2 \'
echo '  --library-path "$GLIBC_NEW/lib:$CONDA/lib:$GCC_LIBDIR:${CUDA_HOME:+$CUDA_HOME/lib64}:$LD_LIBRARY_PATH" \'
echo '  "$CONDA/bin/python" -c "import vllm; print(\"✓ vLLM imports successfully!\")"'

echo ""
echo "Build completed successfully!"