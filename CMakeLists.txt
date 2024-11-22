#
# module: CMakeLists.txt
# author: Bruce Palmer
# description: CMake build for GA. Only MPI-based runtimes are supported.
# 
# DISCLAIMER
#
# This material was prepared as an account of work sponsored by an
# agency of the United States Government.  Neither the United States
# Government nor the United States Department of Energy, nor Battelle,
# nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
# ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
# COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
# SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
# INFRINGE PRIVATELY OWNED RIGHTS.
#
#
# ACKNOWLEDGMENT
#
# This software and its documentation were produced with United States
# Government support under Contract Number DE-AC06-76RLO-1830 awarded by
# the United States Department of Energy.  The United States Government
# retains a paid-up non-exclusive, irrevocable worldwide license to
# reproduce, prepare derivative works, perform publicly and display
# publicly by or for the US Government, including the right to
# distribute to other US Government contractors.
#
# -*- mode: cmake -*-
# -------------------------------------------------------------
# file: CMakeLists.txt
# -------------------------------------------------------------

cmake_minimum_required (VERSION 3.18.0 FATAL_ERROR)
project (GlobalArrays VERSION 5.8.0 LANGUAGES C)

# Append local CMake directory
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/linalg-modules")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
include(ga-utils)

ga_option(ENABLE_CXX ON)
ga_option(ENABLE_FORTRAN OFF)
ga_option(CMAKE_CXX_EXTENSIONS OFF)

set(NOFORT 1)
set(ENABLE_F77 0)

if(ENABLE_CXX)
  enable_language(CXX)
endif()

ga_option(CMAKE_BUILD_TYPE Release)
ga_option(LINALG_VENDOR BLIS)
ga_option(ENABLE_TESTS ON)
ga_option(ENABLE_PROFILING OFF)
#Options for user provided LinAlg libraries
ga_option(ENABLE_DPCPP OFF)
ga_option(ENABLE_DEV_MODE OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_LINK_DEPENDS_NO_SHARED ON)

include(${PROJECT_SOURCE_DIR}/cmake/ga-compiler-options.cmake)

if(MSVC AND ENABLE_FORTRAN)
  message(FATAL_ERROR "MSVC build needs ENABLE_FORTRAN=OFF")
endif()

if(ENABLE_PROFILING)
  if(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    message(FATAL_ERROR "GA profiling is currently not supported with Clang compilers. Please set ENABLE_PROFILING=OFF")
  endif()
endif()

# GA Runtime options: MPI runtime used to build GA
ga_option(GA_RUNTIME MPI_2SIDED)

if (GA_RUNTIME STREQUAL MPI_PROGRESS_RANK)
  set (MPI_PR TRUE CACHE BOOL "use MPI progress ranks protocol for communication")
elseif (GA_RUNTIME STREQUAL MPI_RMA)
  set (MPI3 TRUE CACHE BOOL "use MPI RMA protocols for communication")
elseif (GA_RUNTIME STREQUAL MPI_MULTITHREADED)
  set (MPI_MT TRUE CACHE BOOL "use MPI multi-threading protocols for communication")
elseif (GA_RUNTIME STREQUAL MPI_PROGRESS_THREAD)
  set (MPI_PT TRUE CACHE BOOL "use MPI progress thread protocols for communication")
elseif (GA_RUNTIME STREQUAL MPI_2SIDED)
  set (MPI_TS TRUE CACHE BOOL "use MPI 2-sided protocol for communication")
else()
  message(STATUS "GA_RUNTIME provided \"${GA_RUNTIME}\" is not supported")
endif()

option (MSG_COMMS_MPI "Using MPI runtime for communication" ON)
option (ENABLE_ARMCI_MEM_OPTION "User option for managing memory" ON)
option (ENABLE_CUDA_MEM "User option for CUDA memory" OFF)
option (F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS "Fortran/C interface property" ON)

message(STATUS "Checking MPI ...")
find_package (MPI REQUIRED)
set(GA_EXTRA_LIBS ${GA_EXTRA_LIBS} MPI::MPI_C)

include(${PROJECT_SOURCE_DIR}/cmake/ga-checks.cmake)

CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/f2c_dummy.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/f2c_cmake.h )

# set some flags for MS Visual C Compiler
if (MSVC)
  add_definitions(/D NOMINMAX /D _CRT_SECURE_NO_WARNINGS /D _CRT_NONSTDC_NO_WARNINGS)
endif()
   
# hardwire various GA configuration parameters. Use convention that parameters
# are defined and set to 0 if not used
set (CYGWIN 0)
set (DECOSF 0)

if (CMAKE_SYSTEM_NAME STREQUAL Linux)
  set(LINUX 1)
  set(LINUX64 1)
endif()

set(USE_I8 OFF)
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(USE_I8 ON)
  set(BLAS_SIZE 8)
endif()

option (ENABLE_I8 "Use 8 byte Fortran integers" ${USE_I8})
message(STATUS "void size: ${CMAKE_SIZEOF_VOID_P}, USE_I8: ${USE_I8}, ENABLE_I8: ${ENABLE_I8}")

# hardwire memcpy and strchr since these have been standard for years
set (HAVE_MEMCPY 1)
set (HAVE_STRCHR 1)

# miscellaneous hardwired parameters (mostly not used)
set (ENABLE_CHECKPOINT 0)
set (GA_PROFILING 0)
set (ENABLE_TRACE 0)
set (STATS 1)
set (USE_MALLOC 0)
if(ENABLE_PROFILING)
  set (GA_PROFILING 1)
endif()
if(USE_CRAYSHASTA)
  set(__CRAYXE 1)
  list(APPEND GA_EXTRA_LIBS pmi2)
endif()

# hardwire ARMCI configuration options
set (HAVE_ARMCI_GROUP_COMM 1)
set (HAVE_ARMCI_GROUP_COMM_MEMBER 0)
set (HAVE_ARMCI_INITIALIZED 1)

# suppress any checks to see if test codes run. Only check for compilation.
# use for cross-compilation situations
option (CHECK_COMPILATION_ONLY "Check compilation only" OFF)

# need to set these variable even if only compiling C/C++
set (F77_GETARG GETARG)
set (F77_GETARG_ARGS "i,s")
set (F77_GETARG_DECLS "external GETARG")
set (F77_IARGC IARGC)
set (FORTRAN_I8_FLAG -i8)
set (FORTRAN_I4_FLAG -i4)

# check_fortran_compiler_flag("-fallow-argument-mismatch" _fam)
# ga_is_valid(_fam _fam_flag_exists)
# if(_fam_flag_exists)
#   set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")
# endif()
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" AND CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL "10.1")
  set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")
endif()

# Hardwire these settings. No way to check for Fortran integer size in CMake
# (that we can find)
set (CM_SIZEOF_F77_DOUBLE 8)
set (CM_SIZEOF_F77_REAL 4)
set (CM_SIZEOF_F77_INTEGER 4)
set (F2C_INTEGER_C_TYPE int)
set (F2C_REAL_C_TYPE float)
set (F2C_DOUBLE_PRECISION_C_TYPE double)
if (ENABLE_I8)
  set (CM_SIZEOF_F77_INTEGER 8)
  set (F2C_INTEGER_C_TYPE long)
  set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${FORTRAN_I8_FLAG}")
  message(STATUS "Fortran flags: ${CMAKE_Fortran_FLAGS}")
endif()
set (C_POINTER_AS_INTEGER int)
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set (C_POINTER_AS_INTEGER long)
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(NOUSE_MMAP 1)
  endif()
  set(GA_FORTRAN_ACCESS_INDEX integer*8)
else()
  set(GA_FORTRAN_ACCESS_INDEX integer*4)
endif()
# set (CM_BLAS_SIZE ${CM_SIZEOF_F77_INTEGER})

# -------------------------------------------------------------
# Create include files from templates
# -------------------------------------------------------------
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/config.h )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/farg.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/gaf2c/farg.h )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/typesf2c.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/gaf2c/typesf2c.h )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/matypes.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/ma/matypes.h )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mafdecls.fh.in
                ${CMAKE_CURRENT_BINARY_DIR}/ma/mafdecls.fh )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/global.fh.in
                ${CMAKE_CURRENT_BINARY_DIR}/global/src/global.fh )

include_directories(AFTER ${MPI_INCLUDE_PATH})
add_definitions (-DHAVE_CONFIG_H -DOMPI_SKIP_MPICXX)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
list(APPEND GA_EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT})

if(NOT ENABLE_FORTRAN OR BUILD_SHARED_LIBS)
  if(NOT MSVC)
    list(APPEND GA_EXTRA_LIBS m)
  endif()
endif()
if(NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
  if(NOT MSVC)
    list(APPEND GA_EXTRA_LIBS rt)
  endif()
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fno-aggressive-loop-optimizations)
endif()

if(ENABLE_PROFILING)
  set(ga_profiler $<TARGET_OBJECTS:ga_tools>)
endif()

add_library(ga
  $<TARGET_OBJECTS:ma>
  $<TARGET_OBJECTS:gaf2c>
  $<TARGET_OBJECTS:armci_comex>
  $<TARGET_OBJECTS:ga_src>
#  ${ga_ref_blas}
#  ${ga_profiler}
)
add_library(GlobalArrays::ga ALIAS ga)

set (CMAKE_REQUIRED_LIBRARIES lapack blas)
target_link_libraries(ga PUBLIC ${GA_EXTRA_LIBS})
target_include_directories(ga
  INTERFACE
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/ga>
)

if(DEFINED GA_GCC_TOOLCHAIN_FLAG)
  target_compile_options(ga PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:${GA_GCC_TOOLCHAIN_FLAG}>)
endif()

# -------------------------------------------------------------
# Subdirectories
# -------------------------------------------------------------
add_subdirectory(global/src)
add_subdirectory(comex)
add_subdirectory(ma)
add_subdirectory(gaf2c)

if(ENABLE_TESTS)
  if (NOT GA_TEST_NPROCS)
    set(GA_TEST_NPROCS ${MPIEXEC_MAX_NUMPROCS})
  endif()
  # turn testing on
  enable_testing()
  add_subdirectory(global/testing)
  add_subdirectory(global/c++)
endif()


# --------------------------------------
# Global Arrays installation
# --------------------------------------

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/GlobalArrays)

#dra eaf elio sf
install(TARGETS ma ga_src
    EXPORT globalarrays-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(TARGETS ga
    EXPORT globalarrays-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

#Export the targets to a script
install(EXPORT globalarrays-targets
    FILE
        globalarrays-targets.cmake
    NAMESPACE
        GlobalArrays::
    DESTINATION
        ${INSTALL_CONFIGDIR}
)

#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/globalarrays-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/globalarrays-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/globalarrays-config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

#Install the config, configversion files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/globalarrays-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/globalarrays-config-version.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

list(REMOVE_AT CMAKE_MODULE_PATH 0)
list(REMOVE_AT CMAKE_MODULE_PATH 0)

# export(EXPORT globalarrays-targets
#     FILE ${CMAKE_CURRENT_BINARY_DIR}/globalarrays-targets.cmake
#     NAMESPACE GlobalArrays::)

