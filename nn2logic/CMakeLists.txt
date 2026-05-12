cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(nn2logic LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection" FORCE)  # requires C++20 in current sources
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")


set(GUROBI_DIR "/Library/gurobi1101/macos_universal2" "~/gurobi1101/linux64")  # FIXME


# configure how long a path is allowed to be
# set to 0 to use boost::dynamic_bitset instead (slower)
set(MAX_PATH_LENGTH 64)


# otherwise FMT will cry
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# CPM
set(CPM_DOWNLOAD_VERSION 0.38.2)  # Download CPM package manager
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()
include(${CPM_DOWNLOAD_LOCATION})

# add pybind11
CPMAddPackage("gh:pybind/pybind11@3.0.0")
pybind11_add_module(nn2logic)
target_compile_features(nn2logic PRIVATE cxx_std_20)
install(TARGETS nn2logic DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

# indicators
CPMAddPackage("gh:p-ranav/indicators#9c855c9")
target_link_libraries(nn2logic PRIVATE indicators::indicators)

# fmt
CPMAddPackage("gh:fmtlib/fmt#12.1.0")
target_link_libraries(nn2logic PRIVATE fmt::fmt)
#add_compile_definitions(FMT_HEADER_ONLY)

# Eigen
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
target_link_libraries(nn2logic PRIVATE Eigen3::Eigen)

# Json
CPMAddPackage("gh:nlohmann/json@3.11.3")
include_directories(${nlohmann_json_SOURCE_DIR}/include)
target_link_libraries(nn2logic PRIVATE nlohmann_json::nlohmann_json)

CPMAddPackage("gh:pybind/pybind11_json#0.2.14")
include_directories(${pybind11_json_SOURCE_DIR}/include)

find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
target_link_libraries(nn2logic PRIVATE Boost::program_options)
include_directories(${Boost_INCLUDE_DIRS})
if (DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
  include_directories("${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
endif()
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../tools/vcpkg/installed/x64-windows/include")

# Gurobi
find_package(GUROBI REQUIRED)
include_directories(${GUROBI_INCLUDE_DIRS})
target_link_libraries(nn2logic PRIVATE ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})

# cpu count
include(ProcessorCount)
ProcessorCount(N)
add_compile_definitions(NUM_CPU=${N})

# tbb
find_package(TBB CONFIG COMPONENTS tbb)
if (NOT APPLE)
  if (TARGET TBB::tbb)
    target_link_libraries(nn2logic PRIVATE TBB::tbb)
  endif()
ENDIF()

# template
file(COPY src/sim/template DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(nn2logic PUBLIC BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")

# pass option values
IF (MAX_PATH_LENGTH GREATER 0)
  target_compile_definitions(nn2logic PUBLIC MAX_PATH_LENGTH=${MAX_PATH_LENGTH})
ENDIF()

target_include_directories(nn2logic PRIVATE src)
add_subdirectory(src)