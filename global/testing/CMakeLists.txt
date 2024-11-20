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

include_directories(BEFORE 
  ${PROJECT_SOURCE_DIR}/global/src
  ${PROJECT_BINARY_DIR}/global/src
  ${PROJECT_SOURCE_DIR}/ma 
  ${PROJECT_BINARY_DIR}/ma
  ${PROJECT_BINARY_DIR}/gaf2c
  ${PROJECT_SOURCE_DIR}/comex/src-armci
  ${PROJECT_SOURCE_DIR}/LinAlg/lapack+blas
  ${PROJECT_SOURCE_DIR}/global/testing
  ${PROJECT_BINARY_DIR})

# -------------------------------------------------------------
# Build test executables
# -------------------------------------------------------------

set(TEST_NPROCS_1 1)
set(TEST_NPROCS_2 2)
set(TEST_NPROCS_4 4)
if(MPI_PR) 
  set(TEST_NPROCS_1 2)
  set(TEST_NPROCS_2 3)
  set(TEST_NPROCS_4 5)
endif()

# This test uses random() and srandom() which are not available on
# Windoze
#FIXME:Works upto 10 ranks
add_executable (testsparse.x testsparse.c)
ga_add_parallel_test(testsparse testsparse.x)
target_link_libraries(testsparse.x ga)
add_executable (j_iter.x j_iter.c)
ga_add_parallel_test(j_iter j_iter.x)
target_link_libraries(j_iter.x ga)
add_executable (laplace.x laplace.c)
ga_add_parallel_test(laplace laplace.x)
target_link_libraries(laplace.x ga)
add_executable (sprs_test.x sprs_test.c)
ga_add_parallel_test(sprs_test sprs_test.x)
target_link_libraries(sprs_test.x ga)
add_executable (sddmm_test.x sddmm_test.c)
ga_add_parallel_test(sddmm_test sddmm_test.x)
target_link_libraries(sddmm_test.x ga)
add_executable (putget.x putget.c)
ga_add_parallel_test(putget putget.x)
target_link_libraries(putget.x ga)
add_executable (putget_GA.x putget_GA.c)
ga_add_parallel_test(putget_GA putget_GA.x)
target_link_libraries(putget_GA.x ga)
