cmake_minimum_required(VERSION 3.27)
project(xgrn_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MLX_DIR is exported by mlx/extension.py to the package root, which
# contains share/cmake/MLX/MLXConfig.cmake.
find_package(MLX CONFIG REQUIRED)

# Find Python interpreter for nanobind.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind's cmake config installed by `uv add nanobind`.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN mlx
  cpp/bindings.cpp
  cpp/simdgroup_matmul.cpp
  cpp/simdgroup_matmul_primitive.cpp
)

# Drop the built .so into the importable Python package dir so
# `import xgrn_ext` works without a sys.path dance.
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(_core PRIVATE cpp)
target_link_libraries(_core PRIVATE mlx)
target_compile_options(_core PRIVATE -Wall -Wno-unused-parameter)
target_link_options(_core PRIVATE -Wl,-rpath,@loader_path)
# Allow the extension to find libmlx.dylib at runtime via the installed wheel.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mlx, os; print(os.path.join(mlx.__path__[0], 'lib'))"
  OUTPUT_VARIABLE MLX_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
set_target_properties(_core PROPERTIES BUILD_RPATH "${MLX_LIB_DIR}" INSTALL_RPATH "${MLX_LIB_DIR}")
