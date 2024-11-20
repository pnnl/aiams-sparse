#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>

#include "mpi.h"
#include <omp.h>
#include "vars.hpp"
#include "ga.h"


static char *dataset;
static INDPREC neuron;
static INDPREC layer;
static INDPREC batch;
static INDPREC input;
static VALPREC bias;
static std::string inputFileName;
static std::string outFilePath;

int blocksize;

int *weight_matrices;
int currfeat_s;
int nextfeat_s;


//FEATPREC *currfeat;
//FEATPREC *nextfeat; 

//INDPREC *active;   
int64_t crbatch;

double timeio;
double timetot;
double timeinfer;
double timebalance = 0.0;
double timekernel = 0.0;
double timecopy = 0.0;

//#define USE_SPMM

INDPREC *numbatch;
INDPREC *batchdispl;

int64_t *categories;

FEATPREC ReLU(FEATPREC x){
    return x<0.0?0.0:(x>32.0?32.0:x);
 };

#ifdef USE_SPMM
/* Apply ReLU function to a standard global array */
void Dense_ReLU(int g_a, void *bias)
{
  void *ptr;
  FEATPREC *data;
  int lo[2], hi[2], ld;
  int me = GA_Nodeid();
  FEATPREC lbias = *static_cast<FEATPREC*>(bias);
  FEATPREC value;
  NGA_Distribution(g_a,me,lo,hi);
  NGA_Access(g_a,lo,hi,&ptr,&ld);
  data = static_cast<FEATPREC*>(ptr);

  int i, j, idim, jdim;
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      value = data[i*ld+j];
      value += lbias;
      data[i*ld+j] = value<0.0?0.0:value>32.0?32.0:value;
    }
  }
  NGA_Release(g_a,lo,hi);
}
#endif

double kernel_spmm(INDPREC l) {

  int tmp_s;
//  if (l>0) NGA_Sprs_array_destroy(nextfeat_s);

  double t0 = GA_Wtime();
   // multiply weight matrix by feature matrix, apply constant bias and then
   // apply ReLU activation function
   if (GA_Nodeid() == 0) {
     std::cout << " Starting SPMM operation"<<std::endl;
   }
   double tbeg = GA_Wtime();
#ifdef USE_SPMM
   int tmps;
   nextfeat_s = NGA_Sprs_array_sprsdns_multiply(weight_matrices[l],currfeat_s);
#else
   nextfeat_s = NGA_Sprs_array_matmat_multiply(weight_matrices[l],currfeat_s);
#endif
   double t_mult = GA_Wtime()-tbeg;
   void *tbias = static_cast<void*>(&bias);
   if (GA_Nodeid() == 0) {
     std::cout << " Completed SPMM operation. Starting activate ReLU"<<std::endl;
   }
   tbeg = GA_Wtime();
#ifdef USE_SPMM
   Dense_ReLU(nextfeat_s, tbias);
#else
   NGA_Sprs_array_activate_ReLU64(nextfeat_s, tbias, &categories, &crbatch);
//   char fbuf[128];
//   sprintf(fbuf,"feature_level_%d.mm",l);
//   NGA_Sprs_array_export(nextfeat_s,fbuf);
#endif
   double t_activate = GA_Wtime()-tbeg;
   // Need to do something with categories
   if (GA_Nodeid() == 0) {
     std::cout << " Completed activate ReLU"<<std::endl;
     printf(" Multiply: %16.8f Activate: %16.8f\n",t_mult,t_activate);
   }

//    tmp_s = currfeat_s;
#ifdef USE_SPMM
    GA_Destroy(currfeat_s);
#else
    NGA_Sprs_array_destroy(currfeat_s);
#endif
    currfeat_s = nextfeat_s;
//    nextfeat_s = tmp_s;
    double t1 = GA_Wtime();
    return double(t1-t0);
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);
    GA_Initialize();
    int me = GA_Nodeid();
    parseCommandLine(argc, argv);
    if (inputFileName.empty() || !neuron)
    {
      std::cout << "Input arguments missing...exiting!!!" << std::endl;
      return 0;
    }

    std::ifstream f(inputFileName.c_str());
    if (!f.good())
    {
      std::cout << "Input file path not found...exiting!!!" << std::endl;
      return 0;
    }

    if (!outFilePath.empty())
    {
      std::ifstream f(outFilePath.c_str());
      if (!f.good())
      {
        std::cout << "Output file path not found...exiting!!!" << std::endl;
        return 0;
      }
    }

    dataset = (char*)inputFileName.c_str();
    if (me == 0) {
      std::cout << "Number of processors: "<< GA_Nnodes() << std::endl;
      std::cout << "Input data path: " << dataset << std::endl;
      std::cout << "#Neurons: " << neuron << std::endl;
      std::cout << "#Layers: " << layer << std::endl;
      std::cout << "#Batches: " << batch << std::endl;
      std::cout << "#Inputs: " << input << std::endl;
      std::cout << "Bias: " << bias << std::endl;
    }

    weight_matrices = new int[layer];

    /*
    dataset = "/lus/grand/projects/GRACE/spdnn/dataset";///qfs/projects/pacer/leeh736/dataset"; 
    char *chartemp;
    neuron = 65536;
    layer = 1920;
    batch = 60000;
    input = 392191985; // 392191985; // 98858913; // 25019051; //6374505;
    bias = 0;
    */
    crbatch = static_cast<int64_t>(batch);
//    currfeat = new FEATPREC[neuron*(long)crbatch];
//    nextfeat = new FEATPREC[neuron*(long)crbatch];
  
//    active = new int [crbatch];
    categories = new int64_t[crbatch];
    for (int i=0; i<crbatch; i++) categories[i] = i;

    if (me == 0) {
      printf("%d neurons, %d layers", neuron, layer) ;
      printf("\n");
      printf("READING WEIGHTS\n");
    }
    readweights();
    if (me == 0) printf("READING INPUT\n");
    readinput();

//    for(int k = 0; k < crbatch; k++){
//      active[k] = neuron;
//      categories[k] = k;
//    }
    
    if (me == 0) {
      printf("INFERENCE......\n");
      printf("for %d layers......\n", layer);
    }
    double spmm_times = 0; 
    clock_t total_start = clock();
    for(int i = 0; i < layer; ++i) {
      auto t = kernel_spmm(i);
      spmm_times += double(t);
      if (me == 0) {
        printf("[%d]:(%lf)\n", i,t);
        fflush(stdout);
      }
    }
    clock_t end_start = clock();
    auto gemm_time = double(spmm_times);
    auto all_time = double(end_start - total_start)  / CLOCKS_PER_SEC;
    if (me == 0) {
      printf("Inference time : %lfs, %lfs, %f TTEPS\n", gemm_time,
          all_time, long((long)batch * (long)neuron * 32 * layer)
          / gemm_time / 1e12);
    }
	 
    std::string slayer = std::to_string(layer);
    std::string sneuron = std::to_string(neuron);
    std::string sbatch = std::to_string(batch);
    if (me == 0) {
#if defined(USE_OMP_HOST)
      std::string outfilename = outFilePath + "/" + slayer + "-"
        + sneuron + "-" + sbatch + "cpu-results.txt";
#else
      std::string outfilename = outFilePath + "/" + slayer + "-"
        + sneuron + "-" + sbatch + "gpu-results.txt";
#endif

      std::cout << "Storing output results in: " << outfilename << std::endl;
      std::ofstream ofile(outfilename);
      if (ofile.is_open())
      {
        for (INDPREC i = 0; i < crbatch; i++)
          ofile << categories[i] + 1 << "\n";
      }
      ofile.close();
    }
    for (int l=0; l<layer; l++) {
      NGA_Sprs_array_destroy(weight_matrices[l]);
    }
#ifdef USE_SPMM
    GA_Destroy(currfeat_s);
#else
    NGA_Sprs_array_destroy(currfeat_s);
#endif
    delete [] weight_matrices;
    GA_Terminate();
    MPI_Finalize();
    return 0;
}

void readweights(){
  int me = GA_Nodeid();
  INDPREC **csrdispl;   
  INDPREC **csrindex;
  VALPREC **csrvalue;
  csrdispl = new INDPREC*[layer];
  csrindex = new INDPREC*[layer];
  csrvalue = new VALPREC*[layer];
  // Set up data structures for weight matrices
  long totnz = 0;
    for(INDPREC l = 0; l < layer; l++){
        INDPREC rownz[neuron];
        // number of non-zeros per row is 32
        for(INDPREC n = 0; n < neuron; n++)
            rownz[n] = 32;
        csrdispl[l] = new INDPREC[neuron+1];
        csrdispl[l][0] = 0;
        // calculate displacement for each row
        for(INDPREC n = 1; n < neuron+1; n++)
            csrdispl[l][n] = csrdispl[l][n-1]+rownz[n-1];
        totnz += csrdispl[l][neuron];
        csrindex[l] = new INDPREC[csrdispl[l][neuron]];
        csrvalue[l] = new VALPREC[csrdispl[l][neuron]];
    }

    if (me == 0) {
      printf("weights: %ld (%f GB)\n",totnz,totnz*(sizeof(INDPREC)
            +sizeof(VALPREC))/1.0e9);
    }
    
    char filename[500];
    sprintf(filename,"%s/neuron%d.bin",dataset,neuron);
    if (me == 0) {
      printf("open filename = %s\n", filename);
    }
    FILE *weightf = fopen(filename,"rb");
    for(INDPREC l = 0; l < layer; l++){
        INDPREC *row = new INDPREC[csrdispl[l][neuron]];
        INDPREC *col = new INDPREC[csrdispl[l][neuron]];
        VALPREC *val = new VALPREC[csrdispl[l][neuron]];
        fread(row, sizeof(INDPREC), csrdispl[l][neuron], weightf);
        fread(col, sizeof(INDPREC), csrdispl[l][neuron], weightf);
        fread(val, sizeof(VALPREC), csrdispl[l][neuron],weightf);
        INDPREC rownz[neuron];
        for(INDPREC n = 0; n < neuron; n++)
            rownz[n] = 0;
        for(INDPREC n = 0; n < csrdispl[l][neuron]; n++){
            csrindex[l][csrdispl[l][row[n]-1]+rownz[row[n]-1]] = col[n]-1;
            csrvalue[l][csrdispl[l][row[n]-1]+rownz[row[n]-1]] = val[n];
            rownz[row[n]-1]++;
        }
        delete[] row;
        delete[] col;
        delete[] val;
    }
    fclose(weightf);
    // Create global arrays containing weight matrices
    for(INDPREC l = 0; l < layer; l++){
      int64_t idim = (int64_t)neuron;
      int64_t jdim = (int64_t)neuron;
      if (me == 0) printf("Create array for level %d\n",l);
      if (sizeof(VALPREC) == sizeof(float)) {
        weight_matrices[l] = NGA_Sprs_array_create64(idim, jdim, C_FLOAT);
      } else {
        weight_matrices[l] = NGA_Sprs_array_create64(idim, jdim, C_DBL);
      }
      int s_a = weight_matrices[l];
      // Only add elements on process 0, then distribute to other processes
      if (me == 0) {
        for(INDPREC n = 0; n < neuron; n++) {
          int jmin = csrdispl[l][n];
          int jmax = csrdispl[l][n+1];
          int64_t ii = n;
          for (int j=jmin; j<jmax; j++) {
            int64_t jj = static_cast<int64_t>(csrindex[l][j]);
            NGA_Sprs_array_add_element64(s_a,ii,jj,&csrvalue[l][j]);
          }
        }
      }
      // Distribute using a collective operation
      NGA_Sprs_array_assemble(s_a);
#if 0
      {
        char weight_file[64];
        sprintf(weight_file,"weight_%d.m",l);
        NGA_Sprs_array_export(s_a,weight_file);
      }
#endif
    }
    for(INDPREC l = 0; l < layer; l++){
      delete [] csrdispl[l];
      delete [] csrindex[l];
      delete [] csrvalue[l];
    }
    delete [] csrdispl;
    delete [] csrindex;
    delete [] csrvalue;
}


void readinput(){
    char filename[500];
    int me = GA_Nodeid();
    if (me == 0) {
      printf("features: %ld (%f GB)\n",neuron*(long)batch*2,
          neuron*(long)batch*2*sizeof(FEATPREC)/1.0e9);
    }
    sprintf(filename, "%s/sparse-images-%d.bin", dataset, neuron);
    FILE *inputf = fopen(filename,"rb");
    INDPREC *row = new INDPREC[input];
    INDPREC *col = new INDPREC[input];
    VALPREC *val = new VALPREC[input];
    fread(row,sizeof(INDPREC),input,inputf);
    fread(col,sizeof(INDPREC),input,inputf);
    fread(val,sizeof(VALPREC),input,inputf);
    fclose(inputf);
#ifdef USE_SPMM
    {
      int dims[2];
      dims[0] = neuron;
      dims[1] = batch;
      int two = 2;
      currfeat_s = GA_Create_handle();
      if (sizeof(VALPREC) == sizeof(float)) {
        GA_Set_data(currfeat_s,two,dims,C_FLOAT);
      } else {
        GA_Set_data(currfeat_s,two,dims,C_DBL);
      }
      GA_Allocate(currfeat_s);
      GA_Zero(currfeat_s);
      if (me==0) {
        int *indices = new int[2*input];
        int **subsArray = new int*[input];
        int n;
        for (n=0; n<input; n++) {
          subsArray[n] = &indices[2*n];
          indices[2*n] = row[n]-1;
          indices[2*n+1] = col[n]-1;
        }
        NGA_Scatter(currfeat_s,val,subsArray,n);
        delete [] indices;
        delete [] subsArray;
      }
      GA_Sync();
    }
#else
    int64_t idim = (int64_t)neuron;
    int64_t jdim = (int64_t)batch;
    if (sizeof(VALPREC) == sizeof(float)) {
      currfeat_s = NGA_Sprs_array_create64(idim, jdim, C_FLOAT);
    } else {
      currfeat_s = NGA_Sprs_array_create64(idim, jdim, C_DBL);
    }
    int64_t ii, jj;
    VALPREC cval;
    // only add elements on process 0
    if (me == 0) {
      for (int n = 0; n<input; n++) {
        ii = (int64_t)(row[n]-1);
        jj = (int64_t)(col[n]-1);
        cval = (VALPREC)val[n];
        NGA_Sprs_array_add_element64(currfeat_s,ii,jj,&cval);
      }
    }
    NGA_Sprs_array_assemble(currfeat_s);
#endif
    delete[] row;
    delete[] col;
    delete[] val;
}

void parseCommandLine(int argc, char** const argv)
{
  int ret;
  optind = 1;

  while ((ret = getopt(argc, argv, "f:o:l:b:i:n:a:h")) != -1) {
    switch (ret) {
      case 'f':
        inputFileName.assign(optarg);
        break;
      case 'o':
        outFilePath.assign(optarg);
        break;
      case 'a':
        batch = atoi(optarg);
        break;
      case 'l':
        layer = atoi(optarg);
        break;
      case 'n':
        neuron = atoi(optarg);
        break;
      case 'i':
        input = atoi(optarg);
        break;
      case 'b':
        bias = atof(optarg);
        break;
      case 'h':
        std::cout << "./inference -f <file-path> -i <input>"
          " -o <output path> -a <#batches> -n <#neurons>"
          " -l <#layers> -b <bias>" << std::endl;
        break;  
     default:
        assert(0 && "Should not reach here!!");
        break;
    }
  }
} // parseCommandLine
