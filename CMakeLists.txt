cmake_minimum_required(VERSION 3.5)
project(pipeann)

set(CMAKE_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(USE_AIO "Use AIO instead of liburing" ON)
option(BUILD_PYTHON_INTERFACE "Build Python interface (requires pybind11)" OFF)
option(USE_RABITQ "Enable RaBitQ neighbor implementation (requires Eigen3)" OFF)
option(DGAI_STRICT_GRAPH_CHECKS
       "Hard-assert graph invariants on insert/search hot paths; post-merge full-graph verify/stats in overall_performance" OFF)
option(DGAI_DISABLE_GRAPH_SPLIT
       "Disable page split in decouple insert; allocate new graph nodes at file tail only" OFF)
option(DGAI_MERGE_COORD_EXCLUSIVE
       "A/B: hold exclusive merge_lock for entire decouple merge (big-lock baseline)" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_COMPILER_LAUNCHER ccache)
# set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

include_directories(
	${PROJECT_SOURCE_DIR}/include 
	${PROJECT_SOURCE_DIR}/include/tsl/include 
	$ENV{HOME}/Env/eigen3/include
)

add_compile_options(
	-fopenmp 
	-fopenmp-simd
	-Wall
	-Wextra
	-Wfatal-errors
	-Wno-class-memaccess
	-Wno-unused-variable
	-Wno-unused-parameter
	-Wno-unused-but-set-variable
	-Wno-uninitialized
	-Wno-maybe-uninitialized
)
add_compile_options(-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -Wl,--no-as-needed)

message("Additional definitions: " $ENV{ADDITIONAL_DEFINITIONS})
if(DGAI_STRICT_GRAPH_CHECKS)
	message(STATUS "Strict graph checks enabled (DGAI_STRICT_GRAPH_CHECKS)")
	add_definitions(-DDGAI_STRICT_GRAPH_CHECKS)
endif()
if(DGAI_DISABLE_GRAPH_SPLIT)
	message(STATUS "Graph page split disabled (DGAI_DISABLE_GRAPH_SPLIT); new nodes go to file tail")
	add_definitions(-DDGAI_DISABLE_GRAPH_SPLIT)
endif()
if(DGAI_MERGE_COORD_EXCLUSIVE)
	message(STATUS "Exclusive merge_lock for whole decouple merge (DGAI_MERGE_COORD_EXCLUSIVE)")
	add_definitions(-DDGAI_MERGE_COORD_EXCLUSIVE)
endif()

add_definitions(
	$ENV{ADDITIONAL_DEFINITIONS}
	-DNDEBUG
	# -DIN_PLACE_RECORD_UPDATE # disable design #1
	-DBG_IO_THREAD # enable opt #1.2
	# -DSKIP_BUILD_PQ
	# -DSKIP_BUILD_GRAPH
	-DENABLE_REORDER

	# -DNO_POLLUTE_ORIGINAL

	# optimizations.
	# -DREAD_ONLY_TESTS
	# -DNO_MAPPING # starling: must disable this!
	-DOVERLAP_INIT
	-DMEM_INDEX_PQ_SEARCH  # decouple topology-only nav graph uses PQ distance; required when _mem.index.data is absent
)

include_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src/include
)
link_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src
)

if(USE_AIO)
	message(STATUS "Using AIO for asynchronous I/O, compatible but slower.")
	link_libraries(aio)
	add_definitions(-DUSE_AIO)
else()
	message(STATUS "Using liburing for asynchronous I/O, faster but requires kernel support.")
	link_libraries(uring)
endif()

# Use tcmalloc, as jemalloc does not support python (static TLS block).
link_libraries(
	tcmalloc
)
# add_compile_options(-fsanitize=address,undefined -lasan -lubsan)
# add_link_options(-fsanitize=address,undefined)

# check for AVX512 and AVX2, add the corresponding definitions.
# First, for the compiler.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

# Then, for the runtime.
# test_cpu.cpp returns a bitmask: bit0=AVX2, bit1=AVX512F, bit2=AVX512VPOPCNTDQ
set(CPU_SUPPORTS_AVX512 FALSE)
set(CPU_SUPPORTS_AVX2 FALSE)
set(CPU_SUPPORTS_AVX512_VPOPCNTDQ FALSE)
try_run(
	RUN_RESULT
	COMPILE_RESULT
	"${CMAKE_BINARY_DIR}"
	SOURCES "${PROJECT_SOURCE_DIR}/tests/test_cpu.cpp"
	COMPILE_OUTPUT_VARIABLE COMPILE_OUT
	RUN_OUTPUT_VARIABLE RUN_OUT
)

# RUN_RESULT is a bitmask: bit0=AVX2, bit1=AVX512F, bit2=AVX512VPOPCNTDQ
if(COMPILE_RESULT)
	math(EXPR HAS_AVX2 "${RUN_RESULT} & 1")
	math(EXPR HAS_AVX512F "${RUN_RESULT} & 2")
	math(EXPR HAS_VPOPCNTDQ "${RUN_RESULT} & 4")
	if(HAS_AVX2)
		set(CPU_SUPPORTS_AVX2 TRUE)
		message(STATUS "CPU supports AVX2")
	endif()
	if(HAS_AVX512F)
		set(CPU_SUPPORTS_AVX512 TRUE)
		message(STATUS "CPU supports AVX512F")
	endif()
	if(HAS_VPOPCNTDQ)
		set(CPU_SUPPORTS_AVX512_VPOPCNTDQ TRUE)
		message(STATUS "CPU supports AVX512VPOPCNTDQ")
	endif()
else()
	message(STATUS "CPU feature detection failed")
endif()

if(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512_VPOPCNTDQ)
	message(STATUS "Enabling AVX512 + VPOPCNTDQ.")
	add_definitions(-DUSE_AVX512)
	add_definitions(-DUSE_AVX512_VPOPCNTDQ)
elseif(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512)
    message(STATUS "Compiler and CPU support -mavx512f. Enabling AVX512.")
	add_definitions(-DUSE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2 AND CPU_SUPPORTS_AVX2)
	message(STATUS "Compiler and CPU support -mavx2. Enabling AVX2.")
	add_definitions(-DUSE_AVX2)
else()
    message(STATUS "Compiler or CPU do not support -mavx512f and -mavx2.")
endif()

if(USE_RABITQ)
    message(STATUS "RaBitQ neighbor enabled (requires Eigen3).")
    add_definitions(-DUSE_RABITQ)
else()
    message(STATUS "RaBitQ neighbor disabled.")
endif()

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()

	
find_package(BLAS)
if(BLAS_FOUND)
	message(STATUS "BLAS found ${BLAS_LIBRARIES}")
    if("${BLAS_LIBRARIES}" MATCHES "mkl")
		message(STATUS "BLAS is MKL, adding mkl_rt")
		set(BLAS_LIBRARIES mkl_rt)
		if(DEFINED ENV{MKLROOT})
			include_directories($ENV{MKLROOT}/include)
		endif()
    endif()
	link_libraries(${BLAS_LIBRARIES})
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)
