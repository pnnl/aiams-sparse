#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <cassert>

#include "mpi.h"
#include <omp.h>
#include "ga.h"
#include "ga-mpi.h"
#include "utils.hpp"
#include "graph.hpp"

#define NMAT 1920

int main(int argc, char* argv[]) {
  int s_a;
  std::string filename(argv[1]);

  MPI_Init(&argc, &argv);
  GA_Initialize();
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int ppern = GA_Cluster_nprocs(GA_Cluster_proc_nodeid(me));
  MPI_Comm comm = GA_MPI_Comm();

  BinaryEdgeList list(comm);
  Graph *graph;

  if (me == 0) printf(" nprocs: %d ppern: %d file: (%s)\n",
      nprocs,ppern,filename.c_str());
  graph = list.read(me,nprocs,ppern,filename);
  if (me == 0) printf(" completed read file\n");
  s_a = graph->create_sparse_matrix();
  if (me == 0) printf(" completed create_sparse_matrix\n");

  NGA_Sprs_array_export(s_a,"graph.matrix");

  int iloop;
  int s_list[NMAT];
  s_list[0] = s_a;
  for (iloop=1; iloop<NMAT; iloop++) {
    s_list[iloop] = NGA_Sprs_array_duplicate(s_a);
    if (me == 0) printf(" duplicate %d\n",iloop+1);
  }
  if (me == 0) printf(" completed matrix list\n");
  double td = 0.0;
  for (iloop=0; iloop<NMAT; iloop++) {
    if (me == 0) printf(" Perform matrix-matrix multiply %d\n",iloop+1);
    double td0 = MPI_Wtime();
    int s_b = NGA_Sprs_array_matmat_multiply(s_list[iloop], s_list[iloop]);
    double td1 = MPI_Wtime();
    td += td1 - td0;
    if (me == 0) printf(" [%d] %f\n",iloop+1,td1-td0);
    NGA_Sprs_array_destroy(s_b);
  }

  double  tdt = 0.0;
  MPI_Reduce(&td, &tdt, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  tdt = tdt/static_cast<double>(nprocs*NMAT);
  if (me == 0)  
      std::cout << "Average time to perform GA SpGEMM (secs.): " << tdt << std::endl;
  for (iloop=0; iloop<NMAT; iloop++) {
    NGA_Sprs_array_destroy(s_list[iloop]);
  }
  GA_Terminate();
  MPI_Finalize();
  return 0;
}
