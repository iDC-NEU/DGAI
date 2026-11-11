set(CMAKE_CXX_STANDARD 17)

file(GLOB PROJECT_CPP *.cpp search/*.cpp update/*.cpp utils/*.cpp)
add_library(${PROJECT_NAME} ${PROJECT_CPP})
find_path(METIS_INCLUDE_DIR metis.h)
find_library(METIS_LIBRARY NAMES metis)
if(NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
    message(FATAL_ERROR "METIS is required (install libmetis-dev or set CMAKE_PREFIX_PATH)")
endif()
target_include_directories(${PROJECT_NAME} PUBLIC ${METIS_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${METIS_LIBRARY})
# Some standalone static METIS builds keep GKlib as a separate archive.
find_library(GKLIB_LIBRARY NAMES GKlib)
if(GKLIB_LIBRARY)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${GKLIB_LIBRARY})
endif()

# python module.
# pip install "pybind11[global]"
if(BUILD_PYTHON_INTERFACE)
    find_package(pybind11 REQUIRED)
    file(GLOB PYBIND_CPP python/*.cpp)
    pybind11_add_module(C ${PYBIND_CPP})
    target_link_libraries(C PRIVATE pipeann)
endif()
