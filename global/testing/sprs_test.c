#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

#define WRITE_VTK
#define CG_SOLVE 1
#define NDIM 1024

/**
 *  Solve Laplace's equation on a cubic domain using the sparse matrix
 *  functionality in GA.
 */

#define MAX_FACTOR 1024
void grid_factor(int p, int xdim, int ydim, int zdim,
    int *idx, int *idy, int *idz) {
  int i, j, k; 
  int ip, ifac, pmax, prime[MAX_FACTOR];
  int fac[MAX_FACTOR];
  int ix, iy, iz, ichk;

  i = 1;
/**
 *   factor p completely
 *   first, find all prime numbers, besides 1, less than or equal to 
 *   the square root of p
 */
  ip = p;
  pmax = 0;
  for (i=2; i<=ip; i++) {
    ichk = 1;
    for (j=0; j<pmax; j++) {
      if (i%prime[j] == 0) {
        ichk = 0;
        break;
      }
    }
    if (ichk) {
      pmax = pmax + 1;
      if (pmax > MAX_FACTOR) printf("Overflow in grid_factor\n");
      prime[pmax-1] = i;
    }
  }
/**
 *   find all prime factors of p
 */
  ip = p;
  ifac = 0;
  for (i=0; i<pmax; i++) {
    while(ip%prime[i] == 0) {
      ifac = ifac + 1;
      fac[ifac-1] = prime[i];
      ip = ip/prime[i];
    }
  }
/**
 *  p is prime
 */
  if (ifac==0) {
    ifac++;
    fac[0] = p;
  }
/**
 *    find three factors of p of approximately the same size
 */
  *idx = 1;
  *idy = 1;
  *idz = 1;
  for (i = ifac-1; i >= 0; i--) {
    ix = xdim/(*idx);
    iy = ydim/(*idy);
    iz = zdim/(*idz);
    if (ix >= iy && ix >= iz && ix > 1) {
      *idx = fac[i]*(*idx);
    } else if (iy >= ix && iy >= iz && iy > 1) {
      *idy = fac[i]*(*idy);
    } else if (iz >= ix && iz >= iy && iz > 1) {
      *idz = fac[i]*(*idz);
    } else {
      printf("Too many processors in grid factoring routine\n");
    }
  }
}

/**
 * subroutine to set up a sparse matrix for testing purposes
 * @param s_a sparse matrix handle
 * @param a pointer to a regular matrix that is equivalent to s_a
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 * @param flag if 1, decrease number of off-diagonal elements by 1
 */
void setup_matrix(int *s_a, void **a, int64_t dim, int type, int flag)
{
  int64_t jlo, jhi, idx; 
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t i, j;
  int64_t skip_len, onum;
  void *d_val, *o_val;
  int size;
  int nskip = 5;

  if (me == 0) {
    printf("\n  Create sparse matrix of size %ld x %ld\n",dim,dim);
  }

  /* Create sparse matrix */
  *s_a = NGA_Sprs_array_create64(dim, dim, type);

  /* Determine column block set by me */
  jlo = dim*me/nprocs;
  jhi = dim*(me+1)/nprocs-1;
  if (me == nprocs-1) jhi = dim-1;

  /* set up data values. Diagonal values are 2, off-diagonal values are -1 */
  if (type == C_INT) {
    size = sizeof(int);
  } else if (type == C_LONG) {
    size = sizeof(long);
  } else if (type == C_LONGLONG) {
    size = sizeof(long long);
  } else if (type == C_FLOAT) {
    size = sizeof(float);
  } else if (type == C_DBL) {
    size = sizeof(double);
  } else if (type == C_SCPL) {
    size = 2*sizeof(float);
  } else if (type == C_DCPL) {
    size = 2*sizeof(double);
  }

  d_val = malloc(size);
  o_val = malloc(size);
  *a = malloc(dim*dim*size);
  memset(*a,0,dim*dim*size);

  if (type == C_INT) {
    *((int*)(d_val)) = 2;
    *((int*)(o_val)) = -1;
  } else if (type == C_LONG) {
    *((long*)(d_val)) = 2;
    *((long*)(o_val)) = -1;
  } else if (type == C_LONGLONG) {
    *((long long*)(d_val)) = 2;
    *((long long*)(o_val)) = -1;
  } else if (type == C_FLOAT) {
    *((float*)(d_val)) = 2.0;
    *((float*)(o_val)) = -1.0;
  } else if (type == C_DBL) {
    *((double*)(d_val)) = 2.0;
    *((double*)(o_val)) = -1.0;
  } else if (type == C_SCPL) {
    ((float*)d_val)[0]= 2.0;
    ((float*)d_val)[1]= 0.0;
    ((float*)o_val)[0]= -1.0;
    ((float*)o_val)[1]= 0.0;
  } else if (type == C_DCPL) {
    ((double*)d_val)[0]= 2.0;
    ((double*)d_val)[1]= 0.0;
    ((double*)o_val)[0]= -1.0;
    ((double*)o_val)[1]= 0.0;
  }

  /* loop over all columns in column block and add elements for each column.
   * Currently assume that each column has 5 elements, one on the diagonal 
   * and 4 others off the diagonal. Final matrix is partitioned into row blocks
   * so this guarantees that sorting routines for elements are tested */
  skip_len = dim/nskip;
  if (skip_len < 2)  {
    nskip = dim/2;
    skip_len = dim/nskip;
  }
  if (flag) {
    onum = nskip;
  } else {
    onum = nskip-1;
  }
  for (j=jlo; j<=jhi; j++) {
    NGA_Sprs_array_add_element64(*s_a,j,j,d_val);
    for (i=0; i<onum-1; i++) {
      int idx = (j+(i+1)*skip_len)%dim;
      NGA_Sprs_array_add_element64(*s_a,idx,j,o_val);
    }
  }
  /* create local array with same values */
  for (j=0; j<dim; j++) {
      memcpy(*a+size*(j+j*dim),d_val,size);
    for (i=0; i<onum-1; i++) {
      int idx = (j+(i+1)*skip_len)%dim;
      memcpy(*a+size*(j+idx*dim),o_val,size);
    }
  }

  if (NGA_Sprs_array_assemble(*s_a) && me == 0) {
    printf("\n  Sparse array assembly completed\n\n");
  }
  free(d_val);
  free(o_val);
}

/**
 * subroutine to set up a dense matrix for testing purposes
 * @param g_a dense matrix handle
 * @param a pointer to a regular matrix that is equivalent to g_a
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_dense_matrix(int *g_a, void **a, int64_t dim, int type)
{
  int64_t jlo, jhi, idx; 
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t i, j;
  int64_t skip_len;
  void *d_val, *o_val;
  int size;
  int nskip = 5;
  int two = 2;
  int dims[2],lo[2],hi[2],ld[2];

  if (me == 0) {
    printf("\n  Create dense matrix of size %ld x %ld\n",dim,dim);
  }

  /* Create dense matrix */
  *g_a = NGA_Create_handle();
  dims[0] = dim;
  dims[1] = dim;
  NGA_Set_data(*g_a,two,dims,type);
  NGA_Allocate(*g_a);

  /* set up data values. Diagonal values are 2, off-diagonal values are -1 */
  if (type == C_INT) {
    size = sizeof(int);
  } else if (type == C_LONG) {
    size = sizeof(long);
  } else if (type == C_LONGLONG) {
    size = sizeof(long long);
  } else if (type == C_FLOAT) {
    size = sizeof(float);
  } else if (type == C_DBL) {
    size = sizeof(double);
  } else if (type == C_SCPL) {
    size = 2*sizeof(float);
  } else if (type == C_DCPL) {
    size = 2*sizeof(double);
  }

  /* allocate local buffer */
  *a = malloc(dim*dim*size);
  memset(*a,0,dim*dim*size);

  d_val = malloc(size);
  o_val = malloc(size);

  if (type == C_INT) {
    *((int*)(d_val)) = 2;
    *((int*)(o_val)) = -1;
  } else if (type == C_LONG) {
    *((long*)(d_val)) = 2;
    *((long*)(o_val)) = -1;
  } else if (type == C_LONGLONG) {
    *((long long*)(d_val)) = 2;
    *((long long*)(o_val)) = -1;
  } else if (type == C_FLOAT) {
    *((float*)(d_val)) = 2.0;
    *((float*)(o_val)) = -1.0;
  } else if (type == C_DBL) {
    *((double*)(d_val)) = 2.0;
    *((double*)(o_val)) = -1.0;
  } else if (type == C_SCPL) {
    ((float*)d_val)[0]= 2.0;
    ((float*)d_val)[1]= 0.0;
    ((float*)o_val)[0]= -1.0;
    ((float*)o_val)[1]= 0.0;
  } else if (type == C_DCPL) {
    ((double*)d_val)[0]= 2.0;
    ((double*)d_val)[1]= 0.0;
    ((double*)o_val)[0]= -1.0;
    ((double*)o_val)[1]= 0.0;
  }

  skip_len = dim/nskip;
  if (skip_len < 2)  {
    nskip = dim/2;
    skip_len = dim/nskip;
  }
  /* set all values in local buffer */
  for (j=0; j<dim; j++) {
    memcpy(*a+size*(j+j*dim),d_val,size);
    for (i=0; i<nskip-1; i++) {
      int idx = (j+(i+1)*skip_len)%dim;
      memcpy(*a+size*(j+idx*dim),o_val,size);
    }
  }

  /* copy values in local buffer to global array */
  if (me == 0) {
    lo[0] = 0;
    hi[0] = dim-1;
    ld[0] = dim;
    lo[1] = 0;
    hi[1] = dim-1;
    ld[1] = dim;
    NGA_Put(*g_a,lo,hi,*a,ld);
  }
  GA_Sync();

  free(d_val);
  free(o_val);
}

/**
 * subroutine to set up a sparse matrix for testing activate relu
 * function
 * @param s_a sparse matrix handle
 * @param a pointer to a regular matrix that is equivalent to s_a
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_matrix_relu(int *s_a, void **a, int64_t dim, int type)
{
  int64_t jlo, jhi, idx; 
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t i, j;
  int64_t skip_len;
  void *d_val, *o_val;
  int size;
  int nskip = 5;

  if (me == 0) {
    printf("\n  Create sparse matrix of size %ld x %ld\n",dim,dim);
  }

  /* Create sparse matrix */
  *s_a = NGA_Sprs_array_create64(dim, dim, type);

  /* Determine column block set by me */
  jlo = dim*me/nprocs;
  jhi = dim*(me+1)/nprocs-1;
  if (me == nprocs-1) jhi = dim-1;

  /* set up data values. Diagonal values are 2, off-diagonal values are -1 or
   * 2 */
  if (type == C_INT) {
    size = sizeof(int);
  } else if (type == C_LONG) {
    size = sizeof(long);
  } else if (type == C_LONGLONG) {
    size = sizeof(long long);
  } else if (type == C_FLOAT) {
    size = sizeof(float);
  } else if (type == C_DBL) {
    size = sizeof(double);
  } else if (type == C_SCPL) {
    size = 2*sizeof(float);
  } else if (type == C_DCPL) {
    size = 2*sizeof(double);
  }

  d_val = malloc(size);
  o_val = malloc(size);
  *a = malloc(dim*dim*size);
  memset(*a,0,dim*dim*size);

  if (type == C_INT) {
    *((int*)(d_val)) = 2;
    *((int*)(o_val)) = -1;
  } else if (type == C_LONG) {
    *((long*)(d_val)) = 2;
    *((long*)(o_val)) = -1;
  } else if (type == C_LONGLONG) {
    *((long long*)(d_val)) = 2;
    *((long long*)(o_val)) = -1;
  } else if (type == C_FLOAT) {
    *((float*)(d_val)) = 2.0;
    *((float*)(o_val)) = -1.0;
  } else if (type == C_DBL) {
    *((double*)(d_val)) = 2.0;
    *((double*)(o_val)) = -1.0;
  } else if (type == C_SCPL) {
    ((float*)d_val)[0]= 2.0;
    ((float*)d_val)[1]= 0.0;
    ((float*)o_val)[0]= -1.0;
    ((float*)o_val)[1]= 0.0;
  } else if (type == C_DCPL) {
    ((double*)d_val)[0]= 2.0;
    ((double*)d_val)[1]= 0.0;
    ((double*)o_val)[0]= -1.0;
    ((double*)o_val)[1]= 0.0;
  }

  /* loop over all columns in column block and add elements for each column.
   * Currently assume that each column has 5 elements, one on the diagonal 
   * and 4 others off the diagonl. Final matrix is partitioned into row blocks
   * so this guarantees that sorting routines for elements are tested */
  skip_len = dim/nskip;
  if (skip_len < 2)  {
    nskip = dim/2;
    skip_len = dim/nskip;
  }
  for (j=jlo; j<=jhi; j++) {
    NGA_Sprs_array_add_element64(*s_a,j,j,d_val);
    for (i=0; i<nskip-1; i++) {
      int idx = (j+(i+1)*skip_len)%dim;
      if (idx%2 == 0) {
        NGA_Sprs_array_add_element64(*s_a,idx,j,o_val);
      } else {
        NGA_Sprs_array_add_element64(*s_a,idx,j,d_val);
      }
    }
  }
  /* create local array with same values */
  for (j=0; j<dim; j++) {
      memcpy(*a+size*(j+j*dim),d_val,size);
    for (i=0; i<nskip-1; i++) {
      int idx = (j+(i+1)*skip_len)%dim;
      if (idx%2 == 0) {
        memcpy(*a+size*(j+idx*dim),o_val,size);
      } else {
        memcpy(*a+size*(j+idx*dim),d_val,size);
      }
    }
  }

  if (NGA_Sprs_array_assemble(*s_a) && me == 0) {
    printf("\n  Sparse array assembly completed\n\n");
  }
  free(d_val);
  free(o_val);
}

/**
 * subroutine to set up a diagonal matrix for testing purposes
 * @param g_d handle to 1D array representing diagonal matrix
 * @param a pointer to a local array that is equivalent to g_d
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_diag_matrix(int *g_d, void **d, int64_t dim, int type)
{
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t ilo, ihi, ld;
  int64_t i, j;
  int size;
  void *ptr;

  if (me == 0) {
    printf("  Create diagonal matrix of size %ld x %ld\n",dim,dim);
  }

  /* Create a 1D global array */
  *g_d = NGA_Create_handle();
  NGA_Set_data64(*g_d,1,&dim,type);
  GA_Allocate(*g_d);

  /* Determine row block set by me */
  ilo = dim*me/nprocs;
  ihi = dim*(me+1)/nprocs-1;
  if (me == nprocs-1) ihi = dim-1;

  /* set up data values. Diagonal values are 2, off-diagonal values are -1 */
  if (type == C_INT) {
    size = sizeof(int);
  } else if (type == C_LONG) {
    size = sizeof(long);
  } else if (type == C_LONGLONG) {
    size = sizeof(long long);
  } else if (type == C_FLOAT) {
    size = sizeof(float);
  } else if (type == C_DBL) {
    size = sizeof(double);
  } else if (type == C_SCPL) {
    size = 2*sizeof(float);
  } else if (type == C_DCPL) {
    size = 2*sizeof(double);
  }


  /* get pointers to local data */
  NGA_Distribution64(*g_d,me,&ilo,&ihi);
  NGA_Access64(*g_d,&ilo,&ihi,&ptr,&ld);
  /* set diagonal values */
  for (i=ilo; i<=ihi; i++) {
    if (type == C_INT) {
      ((int*)ptr)[i-ilo] = (int)i;
    } else if (type == C_LONG) {
      ((long*)ptr)[i-ilo] = (long)i;
    } else if (type == C_LONGLONG) {
      ((long long*)ptr)[i-ilo] = (long long)i;
    } else if (type == C_FLOAT) {
      ((float*)ptr)[i-ilo] = (float)i;
    } else if (type == C_DBL) {
      ((double*)ptr)[i-ilo] = (double)i;
    } else if (type == C_SCPL) {
      ((float*)ptr)[2*(i-ilo)] = (float)i;
      ((float*)ptr)[2*(i-ilo)+1] = 0;
    } else if (type == C_DCPL) {
      ((double*)ptr)[2*(i-ilo)] = (double)i;
      ((double*)ptr)[2*(i-ilo)+1] = 0;
    }
  }
  NGA_Release64(*g_d,&ilo,&ihi);
  NGA_Sync();

  /* make copy of g_d in local array */
  *d = malloc(size*dim);
  ilo = 0;
  ihi = dim-1;
  NGA_Get64(*g_d,&ilo,&ihi,*d,&ld);

  if (me == 0) {
    printf("\n  Diagonal array completed\n\n");
  }
}

/**
 * subroutine to set up a dense matrix with unique values at all entries
 * for testing purposes
 * @param g_a dense matrix handle
 * @param a pointer to a regular matrix that is equivalent to g_a
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_cdense_matrix(int *g_a, void **a, int64_t dim, int type)
{
  int64_t jlo, jhi, idx; 
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t i, j;
  void *val;
  int size;
  int two = 2;
  int dims[2],lo[2],hi[2],ld[2];

  if (me == 0) {
    printf("\n  Create dense matrix of size %ld x %ld\n",dim,dim);
  }

  /* Create dense matrix */
  *g_a = NGA_Create_handle();
  dims[0] = dim;
  dims[1] = dim;
  NGA_Set_data(*g_a,two,dims,type);
  NGA_Allocate(*g_a);

  /* set up data values.  */
  if (type == C_INT) {
    size = sizeof(int);
  } else if (type == C_LONG) {
    size = sizeof(long);
  } else if (type == C_LONGLONG) {
    size = sizeof(long long);
  } else if (type == C_FLOAT) {
    size = sizeof(float);
  } else if (type == C_DBL) {
    size = sizeof(double);
  } else if (type == C_SCPL) {
    size = 2*sizeof(float);
  } else if (type == C_DCPL) {
    size = 2*sizeof(double);
  }

  /* allocate local buffer */
  *a = malloc(dim*dim*size);
  memset(*a,0,dim*dim*size);

  val = malloc(size);

  /* set all values in local buffer */
  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      if (type == C_INT) {
        *((int*)val) = (int)(j+i*dim);
      } else if (type == C_LONG) {
        *((long*)val) = (long)(j+i*dim);
      } else if (type == C_LONGLONG) {
        *((long long*)val) = (long long)(j+i*dim);
      } else if (type == C_FLOAT) {
        *((float*)val) = (float)(j+i*dim);
      } else if (type == C_DBL) {
        *((double*)val) = (double)(j+i*dim);
      } else if (type == C_SCPL) {
        ((float*)val)[0] = (float)(j+i*dim);
        ((float*)val)[1] = 0.0;
      } else if (type == C_DCPL) {
        ((double*)val)[0] = (double)(j+i*dim);
        ((double*)val)[1] = 0.0;
      }
      memcpy(*a+size*(j+i*dim),val,size);
    }
  }

  /* copy values in local buffer to global array */
  if (me == 0) {
    lo[0] = 0;
    hi[0] = dim-1;
    ld[0] = dim;
    lo[1] = 0;
    hi[1] = dim-1;
    ld[1] = dim;
    NGA_Put(*g_a,lo,hi,*a,ld);
  }
  GA_Sync();

  free(val);
}

/**
 * subroutine to set up matrix for quantization test
 * @param s_a sparse matrix handle
 * @param a pointer to a regular matrix that is equivalent to s_a
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_quant_matrix(int *s_a, void **a, int64_t dim, int type)
{
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t i, j, idx, n;
#define REPEAT_LEN 16
#define SKIP_LEN 7

  if (type == C_SCPL || type == C_DCPL) {
    GA_Error("(setup_quant_matrix) Illegal data type requested",type);
  }

  if (me == 0) {
    printf("\n  Create sparse matrix of size %ld x %ld\n",dim,dim);
  }

  if (type == C_INT) {
   *a = malloc(dim*dim*sizeof(int));
  } else if (type == C_LONG) {
   *a = malloc(dim*dim*sizeof(long));
  } else if (type == C_LONGLONG) {
   *a = malloc(dim*dim*sizeof(long long));
  } else if (type == C_FLOAT) {
   *a = malloc(dim*dim*sizeof(float));
  } else if (type == C_DBL) {
   *a = malloc(dim*dim*sizeof(double));
  }
  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      idx = i*dim+j;
      if (idx%SKIP_LEN == 0 || i == j) {
        n = idx%REPEAT_LEN;
        if (type == C_INT) {
          ((int*)(*a))[idx] = (int)n;
        } else if (type == C_LONG) {
          ((long*)(*a))[idx] = (long)n;
        } else if (type == C_LONGLONG) {
          ((long long*)(*a))[idx] = (long long)n;
        } else if (type == C_FLOAT) {
          ((float*)(*a))[idx] = (float)n;
        } else if (type == C_DBL) {
          ((double*)(*a))[idx] = (double)n;
        }
      } else {
        if (type == C_INT) {
          ((int*)(*a))[idx] = (int)0;
        } else if (type == C_LONG) {
          ((long*)(*a))[idx] = (long)0;
        } else if (type == C_LONGLONG) {
          ((long long*)(*a))[idx] = (long long)0;
        } else if (type == C_FLOAT) {
          ((float*)(*a))[idx] = (float)0;
        } else if (type == C_DBL) {
          ((double*)(*a))[idx] = (double)0;
        }
      }
    }
  }

  /* Create sparse matrix */
  *s_a = NGA_Sprs_array_create64(dim, dim, type);

  if (me == 0) {
    /* loop over all elements in a and add non-zero elements to sparse array */
    for (i=0; i<dim; i++) {
      for (j=0; j<dim; j++) {
        if (type == C_INT) {
          if (((int*)(*a))[i*dim+j] != (int)0 || i == j) {
            int *aa = (int*)(*a);
            NGA_Sprs_array_add_element(*s_a,i,j,&aa[i*dim+j]);
          }
        } else if (type == C_LONG) {
          if (((long*)(*a))[i*dim+j] != (long)0 || i == j) {
            long *aa = (long*)(*a);
            NGA_Sprs_array_add_element(*s_a,i,j,&aa[i*dim+j]);
          }
        } else if (type == C_LONGLONG) {
          if (((long long*)(*a))[i*dim+j] != (long long)0 || i == j) {
            long long *aa = (long long*)(*a);
            NGA_Sprs_array_add_element(*s_a,i,j,&aa[i*dim+j]);
          }
        } else if (type == C_FLOAT) {
          if (((float*)(*a))[i*dim+j] != (float)0 || i == j) {
            float *aa = (float*)(*a);
            NGA_Sprs_array_add_element(*s_a,i,j,&aa[i*dim+j]);
          }
        } else if (type == C_DBL) {
          if (((double*)(*a))[i*dim+j] != (double)0 || i == j) {
            double *aa = (double*)(*a);
            NGA_Sprs_array_add_element(*s_a,i,j,&aa[i*dim+j]);
          }
        }
      }
    }
  }

  if (NGA_Sprs_array_assemble(*s_a) && me == 0) {
    printf("\n  Sparse array assembly completed\n\n");
  }
}

int matrix_test(int type)
{
  int s_a, s_b, s_c, g_a, g_b, g_c, g_d;
  int64_t dim = NDIM;
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int one = 1;
  int64_t ilo, ihi, jlo, jhi;
  int64_t i, j, k, l, iproc;
  int64_t ld;
  void *ptr;
  int64_t *idx, *jdx;
  int *nz_map;
  int ok;
  char op[2],plus[2];
  void *shift_val;
  void *a, *b, *c, *cp, *d;
  int *ab;
  double tbeg, time;
  double diff;
  int64_t lo[2],hi[2],tld[2];
  int all_ok = 1;
  
  /* create sparse matrix */
  setup_matrix(&s_a, &a, dim, type, 0);

  /* extract diagonal of s_a to g_d */
  tbeg = GA_Wtime();
  NGA_Sprs_array_get_diag(s_a, &g_d);
  time = GA_Wtime()-tbeg;

  /* check values of g_d to see if they are all 2 */
  NGA_Distribution64(g_d,me,&ilo,&ihi);
  ld = ihi-ilo;
  NGA_Access64(g_d,&ilo,&ihi,&ptr,&ld);
  ok = 1;
  for (i=ilo; i<=ihi; i++) {
    if (type == C_INT) {
      if (((int*)ptr)[i-ilo] != 2) {
        printf("p[%d] diag[%ld]: %d ilo: %ld ihi: %ld\n",me,i,((int*)ptr)[i-ilo],ilo,ihi);
        ok = 0;
      }
    } else if (type == C_LONG) {
      if (((long*)ptr)[i-ilo] != 2) {
        ok = 0;
      }
    } else if (type == C_LONGLONG) {
      if (((long long*)ptr)[i-ilo] != 2) {
        ok = 0;
      }
    } else if (type == C_FLOAT) {
      if (((float*)ptr)[i-ilo] != 2.0) {
        ok = 0;
      }
    } else if (type == C_DBL) {
      if (((double*)ptr)[i-ilo] != 2.0) {
        ok = 0;
      }
    } else if (type == C_SCPL) {
      if (((float*)ptr)[2*(i-ilo)] != 2.0 ||
          ((float*)ptr)[2*(i-ilo)+1] != 0.0) {
        ok = 0;
      }
    } else if (type == C_DCPL) {
      if (((double*)ptr)[2*(i-ilo)] != 2.0 ||
          ((double*)ptr)[2*(i-ilo)+1] != 0.0) {
        ok = 0;
      }
    }
  }
  NGA_Release64(g_d,&ilo,&ihi);
  op[0] = '*';
  op[1] = '\0';
  plus[0] = '+';
  plus[1] = '\0';
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix get diagonal operation PASSES**\n");
      printf("    Time for matrix get diagonal operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix get diagonal operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;
  NGA_Destroy(g_d);

  /**
   * Test shift diagonal operation
   */
  if (type == C_INT) {
    shift_val = malloc(sizeof(int));
    *((int*)shift_val) = 1;
  } else if (type == C_LONG) {
    shift_val = malloc(sizeof(long));
    *((long*)shift_val) = 1;
  } else if (type == C_LONGLONG) {
    shift_val = malloc(sizeof(long long));
    *((long long*)shift_val) = 1;
  } else if (type == C_FLOAT) {
    shift_val = malloc(sizeof(float));
    *((float*)shift_val) = 1.0;
  } else if (type == C_DBL) {
    shift_val = malloc(sizeof(double));
    *((double*)shift_val) = 1.0;
  } else if (type == C_SCPL) {
    shift_val = malloc(sizeof(SingleComplex));
    ((float*)shift_val)[0] = 1.0;
    ((float*)shift_val)[1] = 0.0;
  } else if (type == C_DCPL) {
    shift_val = malloc(sizeof(DoubleComplex));
    ((double*)shift_val)[0] = 1.0;
    ((double*)shift_val)[1] = 0.0;
  }
  tbeg = GA_Wtime();
  NGA_Sprs_array_shift_diag(s_a, shift_val);
  time = GA_Wtime()-tbeg;

  /* extract diagonal of s_a to g_d */
  NGA_Sprs_array_get_diag(s_a, &g_d);

  /* check values of g_d to see if they are all 3 */
  NGA_Distribution64(g_d,me,&ilo,&ihi);
  ld = ihi-ilo;
  NGA_Access64(g_d,&ilo,&ihi,&ptr,&ld);
  ok = 1;
  for (i=ilo; i<=ihi; i++) {
    if (type == C_INT) {
      if (((int*)ptr)[i-ilo] != 3) {
        ok = 0;
      }
    } else if (type == C_LONG) {
      if (((long*)ptr)[i-ilo] != 3) {
        ok = 0;
      }
    } else if (type == C_LONGLONG) {
      if (((long long*)ptr)[i-ilo] != 3) {
        ok = 0;
      }
    } else if (type == C_FLOAT) {
      if (((float*)ptr)[i-ilo] != 3.0) {
        ok = 0;
      }
    } else if (type == C_DBL) {
      if (((double*)ptr)[i-ilo] != 3.0) {
        ok = 0;
      }
    } else if (type == C_SCPL) {
      if (((float*)ptr)[2*(i-ilo)] != 3.0 ||
          ((float*)ptr)[2*(i-ilo)+1] != 0.0) {
        ok = 0;
      }
    } else if (type == C_DCPL) {
      if (((double*)ptr)[2*(i-ilo)] != 3.0 ||
          ((double*)ptr)[2*(i-ilo)+1] != 0.0) {
        ok = 0;
      }
    }
  }
  NGA_Release64(g_d,&ilo,&ihi);
  op[0] = '*';
  op[1] = '\0';
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("\n    **Sparse matrix shift diagonal operation PASSES**\n");
      printf("    Time for matrix get diagonal operation: %16.8f\n",time);
    } else {
      printf("\n    **Sparse matrix shift diagonal operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;
  NGA_Destroy(g_d);

  NGA_Sprs_array_destroy(s_a);
  free(shift_val);
  free(a);

  /* Create a fresh copy of sparse matrix */
  setup_matrix(&s_a, &a, dim, type, 0);

  /* Create diagonal matrix */
  setup_diag_matrix(&g_d, &d, dim, type);

  /* Do a right hand multiply */
  tbeg = GA_Wtime();
  NGA_Sprs_array_diag_right_multiply(s_a, g_d);
  time = GA_Wtime()-tbeg;
  /* Do a right hand multiply in regular arrays */
#define MULTIPLY_REAL_AIJ_DJ_M(_a, _d, _i, _j, _type)  \
  {                                                    \
    _type _aij = ((_type*)_a)[_j+dim*_i];              \
    _type _dj  = ((_type*)_d)[_j];                     \
    ((_type*)_a)[_j+dim*_i] = _aij*_dj;                \
  }

#define MULTIPLY_COMPLEX_AIJ_DJ_M(_a, _d, _i, _j, _type)      \
  {                                                           \
    _type _aij_r = ((_type*)_a)[2*(_j+dim*_i)];               \
    _type _aij_i = ((_type*)_a)[2*(_j+dim*_i)+1];             \
    _type _dj_r  = ((_type*)_d)[2*_j];                        \
    _type _dj_i  = ((_type*)_d)[2*_j+1];                      \
    ((_type*)_a)[2*(_j+dim*_i)] = _aij_r*_dj_r-_aij_i*_dj_i;  \
    ((_type*)_a)[2*(_j+dim*_i)+1] = _aij_r*_dj_i+_aij_i*_dj_r;\
  }

  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      if (type == C_INT) {
        MULTIPLY_REAL_AIJ_DJ_M(a, d, i, j, int);
      } else if (type == C_LONG) {
        MULTIPLY_REAL_AIJ_DJ_M(a, d, i, j, long);
      } else if (type == C_LONGLONG) {
        MULTIPLY_REAL_AIJ_DJ_M(a, d, i, j, long long);
      } else if (type == C_FLOAT) {
        MULTIPLY_REAL_AIJ_DJ_M(a, d, i, j, float);
      } else if (type == C_DBL) {
        MULTIPLY_REAL_AIJ_DJ_M(a, d, i, j, double);
      } else if (type == C_SCPL) {
        MULTIPLY_COMPLEX_AIJ_DJ_M(a, d, i, j, float);
      } else if (type == C_DCPL) {
        MULTIPLY_COMPLEX_AIJ_DJ_M(a, d, i, j, double);
      }
    }
  }

#undef MULTIPLY_REAL_AIJ_DJ_M
#undef MULTIPLY_COMPLEX_AIJ_DJ_M
  /* Compare matrix from sparse array operations with
   * local left multiply. Start by getting row block owned by this
   * process */
  ok = 1;
  NGA_Sprs_array_row_distribution64(s_a, me, &ilo, &ihi);
  /* loop over column blocks */
  for (iproc = 0; iproc<nprocs; iproc++) {
    NGA_Sprs_array_column_distribution64(s_a, iproc, &jlo, &jhi);
    if (jhi >= jlo) {
      int64_t nrows = ihi-ilo+1;
      /* column block corresponding to iproc has data. Get pointers
       * to index and data arrays */
      NGA_Sprs_array_access_col_block64(s_a, iproc, &idx, &jdx, &ptr);
      if (idx != NULL) {
        for (i=0; i<nrows; i++) {
          int64_t nvals = idx[i+1]-idx[i];
          for (j=0; j<nvals; j++) {
            if (type == C_INT) {
              if (((int*)ptr)[idx[i]+j]
                  != ((int*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_LONG) {
              if (((long*)ptr)[idx[i]+j]
                  != ((long*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_LONGLONG) {
              if (((long long*)ptr)[idx[i]+j]
                  != ((long long*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_FLOAT) {
              if (((float*)ptr)[idx[i]+j]
                  != ((float*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_DBL) {
              if (((double*)ptr)[idx[i]+j]
                  != ((double*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_SCPL) {
              float rval = ((float*)ptr)[2*(idx[i]+j)];
              float ival = ((float*)ptr)[2*(idx[i]+j)+1];
              float ra = ((float*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])];
              float ia = ((float*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
              if (rval != ra || ival != ia) {
                ok = 0;
              }
            } else if (type == C_DCPL) {
              double rval = ((double*)ptr)[2*(idx[i]+j)];
              double ival = ((double*)ptr)[2*(idx[i]+j)+1];
              double ra = ((double*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])];
              double ia = ((double*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
              if (rval != ra || ival != ia) {
                ok = 0;
              }
            }
          }
        }
      }
    }
  }
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix right diagonal multiply operation PASSES**\n");
      printf("    Time for matrix right diagonal multiply operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix right diagonal multiply operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;

  NGA_Sprs_array_destroy(s_a);
  free(a);
  free(d);

  /* Create a fresh copy of sparse matrix */
  setup_matrix(&s_a, &a, dim, type, 0);

  /* Create diagonal matrix */
  setup_diag_matrix(&g_d, &d, dim, type);

  /* Do a left hand multiply */
  tbeg = GA_Wtime();
  NGA_Sprs_array_diag_left_multiply(s_a, g_d);
  time = GA_Wtime()-tbeg;
  /* Do a right hand multiply in regular arrays */
#define MULTIPLY_REAL_DI_AIJ_M(_a, _d, _i, _j, _type)  \
  {                                                    \
    _type _aij = ((_type*)_a)[_j+dim*_i];              \
    _type _di  = ((_type*)_d)[_i];                     \
    ((_type*)_a)[_j+dim*_i] = _di*_aij;                \
  }

#define MULTIPLY_COMPLEX_DI_AIJ_M(_a, _d, _i, _j, _type)      \
  {                                                           \
    _type _aij_r = ((_type*)_a)[2*(_j+dim*_i)];               \
    _type _aij_i = ((_type*)_a)[2*(_j+dim*_i)+1];             \
    _type _di_r  = ((_type*)_d)[2*_i];                        \
    _type _di_i  = ((_type*)_d)[2*_i+1];                      \
    ((_type*)_a)[2*(_j+dim*_i)] = _di_r*_aij_r-_di_i*_aij_i;  \
    ((_type*)_a)[2*(_j+dim*_i)+1] = _di_i*_aij_r+_di_r*_aij_i;\
  }

  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      if (type == C_INT) {
        MULTIPLY_REAL_DI_AIJ_M(a, d, i, j, int);
      } else if (type == C_LONG) {
        MULTIPLY_REAL_DI_AIJ_M(a, d, i, j, long);
      } else if (type == C_LONGLONG) {
        MULTIPLY_REAL_DI_AIJ_M(a, d, i, j, long long);
      } else if (type == C_FLOAT) {
        MULTIPLY_REAL_DI_AIJ_M(a, d, i, j, float);
      } else if (type == C_DBL) {
        MULTIPLY_REAL_DI_AIJ_M(a, d, i, j, double);
      } else if (type == C_SCPL) {
        MULTIPLY_COMPLEX_DI_AIJ_M(a, d, i, j, float);
      } else if (type == C_DCPL) {
        MULTIPLY_COMPLEX_DI_AIJ_M(a, d, i, j, double);
      }
    }
  }

#undef MULTIPLY_REAL_DI_AIJ_M
#undef MULTIPLY_COMPLEX_DI_AIJ_M
  /* Compare matrix from sparse array operations with
   * local left multiply. Start by getting row block owned by this
   * process */
  ok = 1;
  NGA_Sprs_array_row_distribution64(s_a, me, &ilo, &ihi);
  /* loop over column blocks */
  for (iproc = 0; iproc<nprocs; iproc++) {
    NGA_Sprs_array_column_distribution64(s_a, iproc, &jlo, &jhi);
    if (jhi >= jlo) {
      int64_t nrows = ihi-ilo+1;
      /* column block corresponding to iproc has data. Get pointers
       * to index and data arrays */
      NGA_Sprs_array_access_col_block64(s_a, iproc, &idx, &jdx, &ptr);
      if (idx != NULL) {
        for (i=0; i<nrows; i++) {
          int64_t nvals = idx[i+1]-idx[i];
          for (j=0; j<nvals; j++) {
            if (type == C_INT) {
              if (((int*)ptr)[idx[i]+j] != ((int*)a)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_LONG) {
              if (((long*)ptr)[idx[i]+j] != ((long*)a)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_LONGLONG) {
              if (((long long*)ptr)[idx[i]+j] 
                  != ((long long*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_FLOAT) {
              if (((float*)ptr)[idx[i]+j] != ((float*)a)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_DBL) {
              if (((double*)ptr)[idx[i]+j] != ((double*)a)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_SCPL) {
              float rval = ((float*)ptr)[2*(idx[i]+j)];
              float ival = ((float*)ptr)[2*(idx[i]+j)+1];
              float ra = ((float*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])];
              float ia = ((float*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
              if (rval != ra || ival != ia) {
                ok = 0;
              }
            } else if (type == C_DCPL) {
              double rval = ((double*)ptr)[2*(idx[i]+j)];
              double ival = ((double*)ptr)[2*(idx[i]+j)+1];
              double ra = ((double*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])];
              double ia = ((double*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
              if (rval != ra || ival != ia) {
                ok = 0;
              }
            }
          }
        }
      }
    }
  }
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix left diagonal multiply operation PASSES**\n");
      printf("    Time for matrix left diagonal multiply operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix left diagonal multiply operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;

  NGA_Sprs_array_destroy(s_a);
  free(a);
  free(d);

  /* create sparse matrix A */
  setup_matrix(&s_a, &a, dim, type, 0);

  nz_map = (int*)malloc(dim*dim*sizeof(int));
  for (i=0; i<dim*dim; i++) nz_map[i] = 0;
  time = 0.0;
  for (k=0; k<nprocs; k++) {
    for (l=0; l<nprocs; l++) {
      tbeg = GA_Wtime();
      if (!NGA_Sprs_array_get_block64(s_a, k, l, &idx, &jdx, &ptr,
          &ilo, &ihi, &jlo, &jhi)) {
        continue;
      }
      time += GA_Wtime()-tbeg;
      /*
      printf("p[%d] block [%d,%d] ilo: %d ihi: %d jlo: %d jhi: %d\n",
      me,k,l,ilo,ihi,jlo,jhi);
      */
      /* check for correctness */
      ok = 1;
      for (i=ilo; i<=ihi; i++) {
        int64_t jcols = idx[i+1-ilo]-idx[i-ilo];
        /*
        printf("p[%d]     row: %d jcols: %d\n",me,i,jcols);
        */

        for (j=0; j<jcols; j++) {
          nz_map[i*dim+jdx[idx[i-ilo]+j]] = 1;
          /*
        printf("p[%d]         row: %d jcols: %d col: %d\n",me,i,jcols,
            jdx[idx[i-ilo]+j]+jlo);
            */
          if (type == C_INT) {
            if (((int*)ptr)[idx[i-ilo]+j]
                != ((int*)a)[i*dim+jdx[idx[i-ilo]+j]]) {
              if (ok) printf("(int) block [%d,%d] element i: %d j: %d"
                  " expected: %d actual: %d\n",
                  me,k,i,jdx[idx[i-ilo]+j],((int*)a)[i*dim+jdx[idx[i-ilo]+j]],
                  ((int*)ptr)[idx[i-ilo]+j]);
              ok = 0;
            }
          } else if (type == C_LONG) {
            if (((long*)ptr)[idx[i-ilo]+j]
                != ((long*)a)[i*dim+jdx[idx[i-ilo]+j]]) {
              printf("(long) block [%d,%d] element i: %d j: %d"
                  " expected: %ld actual: %ld\n",
                  me,k,i,jdx[idx[i-ilo]+j],((long*)a)[i*dim
                  +jdx[idx[i-ilo]+j]], ((long*)ptr)[idx[i-ilo]+j]);
              ok = 0;
            }
          } else if (type == C_LONGLONG) {
            if (((long long*)ptr)[idx[i-ilo]+j] != ((long long*)a)[i*dim
                +jdx[idx[i-ilo]+j]]) {
              printf("(long long) block [%d,%d] element i: %d j: %d"
                  " expected: %ld actual: %ld\n",
                  me,k,i,jdx[idx[i-ilo]+j],
                  (long)(((long long*)a)[i*dim+jdx[idx[i-ilo]+j]]),
                  (long)(((long long*)ptr)[idx[i-ilo]+j]));
              ok = 0;
            }
          } else if (type == C_FLOAT) {
            if (((float*)ptr)[idx[i-ilo]+j] 
                != ((float*)a)[i*dim+jdx[idx[i-ilo]+j]]) {
              printf("(float) block [%d,%d] element i: %d j: %d"
                  " expected: %f actual: %f\n",
                  me,k,i,jdx[idx[i-ilo]+j],((float*)a)[i*dim
                  +jdx[idx[i-ilo]+j]], ((float*)ptr)[idx[i-ilo]+j]);
              ok = 0;
            }
          } else if (type == C_DBL) {
            if (((double*)ptr)[idx[i-ilo]+j]
                != ((double*)a)[i*dim+jdx[idx[i-ilo]+j]]) {
              printf("(double) block [%d,%d] element i: %d j: %d"
                  " expected: %f actual: %f\n",
                  me,k,i,jdx[idx[i-ilo]+j],((double*)a)[i*dim
                  +jdx[idx[i-ilo]+j]], ((double*)ptr)[idx[i-ilo]+j]);
              ok = 0;
            }
          } else if (type == C_SCPL) {
            if (((float*)ptr)[2*(idx[i-ilo]+j)] 
                != ((float*)a)[2*(i*dim+jdx[idx[i-ilo]+j])] ||
                ((float*)ptr)[2*(idx[i-ilo]+j)+1] 
                != ((float*)a)[2*(i*dim+jdx[idx[i-ilo]+j])+1]) {
              printf("(single complex) block [%d,%d] element i: %d j: %d"
                  " expected: (%f,%f) actual: (%f,%f)\n",
                  me,k,i,jdx[idx[i-ilo]+j],
                  ((float*)a)[2*(i*dim+jdx[idx[i-ilo]+j])],
                  ((float*)a)[2*(i*dim+jdx[idx[i-ilo]+j])+1],
                  ((float*)ptr)[2*(idx[i-ilo]+j)],
                  ((float*)ptr)[2*(idx[i-ilo]+j)+1]);
              ok = 0;
            }
          } else if (type == C_DCPL) {
            if (((double*)ptr)[2*(idx[i-ilo]+j)] 
                != ((double*)a)[2*(i*dim+jdx[idx[i-ilo]+j])] ||
                ((double*)ptr)[2*(idx[i-ilo]+j)+1] 
                != ((double*)a)[2*(i*dim+jdx[idx[i-ilo]+j])+1]) {
              printf("(double complex) block [%d,%d] element i: %d j: %d"
                  " expected: (%f,%f) actual: (%f,%f)\n",
                  me,k,i,jdx[idx[i-ilo]+j],
                  ((double*)a)[2*(i*dim+jdx[idx[i-ilo]+j])],
                  ((double*)a)[2*(i*dim+jdx[idx[i-ilo]+j])+1],
                  ((double*)ptr)[2*(idx[i-ilo]+j)],
                  ((double*)ptr)[2*(idx[i-ilo]+j)+1]);
              ok = 0;
            }
          }
        }
      }
      if (idx != NULL) free(idx);
      if (idx != NULL) free(jdx);
      if (idx != NULL) free(ptr);
    }
  }
  for (i=0; i<dim*dim; i++) {
    if (type == C_INT) {
      if (((int*)a)[i] != 0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_LONG) {
      if (((long*)a)[i] != 0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_LONGLONG) {
      if (((long long*)a)[i] != 0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_FLOAT) {
      if (((float*)a)[i] != 0.0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_DBL) {
      if (((double*)a)[i] != 0.0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_SCPL) {
      if ((((float*)a)[2*i] != 0.0 || ((float*)a)[2*i+1] != 0.0)
          && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_SCPL) {
      if ((((double*)a)[2*i] != 0.0 || ((double*)a)[2*i+1] != 0.0)
          && nz_map[i] == 0) {
        ok = 0;
      }
    }
  }
  free(nz_map);

  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)(nprocs*nprocs*nprocs);
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix get block operation PASSES**\n");
      printf("    Time for matrix get block operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix get block operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;

  NGA_Sprs_array_destroy(s_a);
  free(a);

  /* create sparse matrix A */
  setup_matrix(&s_a, &a, dim, type, 0);
  /* create sparse matrix B */
  setup_matrix(&s_b, &b, dim, type, 0);

  /* multiply sparse matrix A times sparse matrix B */
  tbeg = GA_Wtime();
  s_c = NGA_Sprs_array_matmat_multiply(s_a, s_b);
  time = GA_Wtime()-tbeg;


  /* Do regular matrix-matrix multiply of A and B */
  if (type == C_INT) {
    c = malloc(dim*dim*sizeof(int));
  } else if (type == C_LONG) {
    c = malloc(dim*dim*sizeof(long));
  } else if (type == C_LONGLONG) {
    c = malloc(dim*dim*sizeof(long long));
  } else if (type == C_FLOAT) {
    c = malloc(dim*dim*sizeof(float));
  } else if (type == C_DBL) {
    c = malloc(dim*dim*sizeof(double));
  } else if (type == C_SCPL) {
    c = malloc(dim*dim*2*sizeof(float));
  } else if (type == C_DCPL) {
    c = malloc(dim*dim*2*sizeof(double));
  }

#define REAL_MATMAT_MULTIPLY_M(_type, _a, _b, _c, _dim)     \
{                                                           \
  int _i, _j, _k;                                           \
  _type *_aa = (_type*)_a;                                  \
  _type *_bb = (_type*)_b;                                  \
  _type *_cc = (_type*)_c;                                  \
  for (_i=0; _i<_dim; _i++) {                               \
    for (_j=0; _j<_dim; _j++) {                             \
      _cc[_j+_i*_dim] = (_type)0;                           \
      for(_k=0; _k<_dim; _k++) {                            \
        _cc[_j+_i*_dim] += _aa[_k+_i*_dim]*_bb[_j+_k*_dim]; \
      }                                                     \
    }                                                       \
  }                                                         \
}

#define COMPLEX_MATMAT_MULTIPLY_M(_type, _a, _b, _c, _dim)  \
{                                                           \
  int _i, _j, _k;                                           \
  _type *_aa = (_type*)_a;                                  \
  _type *_bb = (_type*)_b;                                  \
  _type *_cc = (_type*)_c;                                  \
  _type _ar, _ai, _br, _bi;                                 \
  for (_i=0; _i<_dim; _i++) {                               \
    for (_j=0; _j<_dim; _j++) {                             \
      _cc[2*(_j+_i*_dim)] = (_type)0;                       \
      _cc[2*(_j+_i*_dim)+1] = (_type)0;                     \
      for(_k=0; _k<_dim; _k++) {                            \
        _ar = _aa[2*(_k+_i*_dim)];                          \
        _ai = _aa[2*(_k+_i*_dim)+1];                        \
        _br = _bb[2*(_j+_k*_dim)];                          \
        _bi = _bb[2*(_j+_k*_dim)+1];                        \
        _cc[2*(_j+_i*_dim)] += _ar*_br-_ai*_bi;             \
        _cc[2*(_j+_i*_dim)+1] += _ar*_bi+_ai*_br;           \
      }                                                     \
    }                                                       \
  }                                                         \
}

  if (type == C_INT) {
    REAL_MATMAT_MULTIPLY_M(int, a, b, c, dim);
    REAL_MATMAT_MULTIPLY_M(int, c, a, b, dim);
  } else if (type == C_LONG) {
    REAL_MATMAT_MULTIPLY_M(long, a, b, c, dim);
    REAL_MATMAT_MULTIPLY_M(long, c, a, b, dim);
  } else if (type == C_LONGLONG) {
    REAL_MATMAT_MULTIPLY_M(long long, a, b, c, dim);
    REAL_MATMAT_MULTIPLY_M(long long, c, a, b, dim);
  } else if (type == C_FLOAT) {
    REAL_MATMAT_MULTIPLY_M(float, a, b, c, dim);
    REAL_MATMAT_MULTIPLY_M(float, c, a, b, dim);
  } else if (type == C_DBL) {
    REAL_MATMAT_MULTIPLY_M(double, a, b, c, dim);
    REAL_MATMAT_MULTIPLY_M(double, c, a, b, dim);
  } else if (type == C_SCPL) {
    COMPLEX_MATMAT_MULTIPLY_M(float, a, b, c, dim);
    COMPLEX_MATMAT_MULTIPLY_M(float, c, a, b, dim);
  } else if (type == C_DCPL) {
    COMPLEX_MATMAT_MULTIPLY_M(double, a, b, c, dim);
    COMPLEX_MATMAT_MULTIPLY_M(double, c, a, b, dim);
  }

#undef REAL_MATMAT_MULTIPLY_M
#undef COMPLEX_MATMAT_MULTIPLY_M

  NGA_Sprs_array_destroy(s_b);
  s_b = NGA_Sprs_array_matmat_multiply(s_c, s_a);

  ab = (int*)malloc(dim*dim*sizeof(int));
  for (i=0; i<dim*dim; i++) ab[i] = 0;
  nz_map = (int*)malloc(dim*dim*sizeof(int));
  for (i=0; i<dim*dim; i++) nz_map[i] = 0;
  /* Compare results from regular matrix-matrix multiply with
   * sparse matrix-matrix multiply */
  ok = 1;
  NGA_Sprs_array_row_distribution64(s_b, me, &ilo, &ihi);
  /* loop over column blocks */
  ld = 0;
  for (iproc = 0; iproc<nprocs; iproc++) {
    int64_t nrows = ihi-ilo+1;
    /* column block corresponding to iproc has data. Get pointers
     * to index and data arrays */
    NGA_Sprs_array_access_col_block64(s_b, iproc, &idx, &jdx, &ptr);
    /*
        printf("p[%d] iproc: %ld idx: %p jdx: %p ptr: %p\n",me,iproc,idx,jdx,ptr);
        */
    if (idx != NULL) {
      for (i=0; i<nrows; i++) {
        int64_t nvals = idx[i+1]-idx[i];
        /*
        printf("p[%d] iproc: %ld i: %ld nrows: %ld nvals: %ld\n",me,
            iproc,i+ilo,nrows,nvals);
            */
        for (j=0; j<nvals; j++) {
          void *tptr = b;
          /*
        printf("p[%d]     iproc: %ld i: %ld j: %ld nvals: %ld\n",me,
            iproc,i+ilo,jdx[idx[i]+j],nvals);
            */
          ld++;
          nz_map[(i+ilo)*dim+jdx[idx[i]+j]] = 1;
          if (type == C_INT) {
            /*
            printf("p[%d]        proc: %ld i: %ld j: %ld c: %d\n",me,
                iproc,i+ilo,jdx[idx[i]+j],((int*)ptr)[idx[i]+j]);
                */
            ab[(i+ilo)*dim+jdx[idx[i]+j]] = ((int*)ptr)[idx[i]+j];
            if (((int*)ptr)[idx[i]+j] != ((int*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %d actual: %d\n",me,
                    i+ilo,jdx[idx[i]+j],((int*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((int*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_LONG) {
            if (((long*)ptr)[idx[i]+j] != ((long*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                    i+ilo,jdx[idx[i]+j],((long*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((long*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_LONGLONG) {
            if (((long long*)ptr)[idx[i]+j] 
                != ((long long*)tptr)[(i+ilo)*dim + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                    i+ilo,jdx[idx[i]+j],(long)((long long*)tptr)[(i+ilo)*dim
                    +jdx[idx[i]+j]],
                    (long)((long long*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_FLOAT) {
            if (((float*)ptr)[idx[i]+j] != ((float*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                    i+ilo,jdx[idx[i]+j],((float*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((float*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_DBL) {
            if (((double*)ptr)[idx[i]+j] != ((double*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                    i+ilo,jdx[idx[i]+j],((double*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((double*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_SCPL) {
            float rval = ((float*)ptr)[2*(idx[i]+j)];
            float ival = ((float*)ptr)[2*(idx[i]+j)+1];
            float ra = ((float*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])];
            float ia = ((float*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
            if (rval != ra || ival != ia) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: (%f,%f) actual: (%f,%f)\n",me,
                    i+ilo,jdx[idx[i]+j],ra,ia,rval,ival);
              }
              ok = 0;
            }
          } else if (type == C_DCPL) {
            double rval = ((double*)ptr)[2*(idx[i]+j)];
            double ival = ((double*)ptr)[2*(idx[i]+j)+1];
            double ra = ((double*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])];
            double ia = ((double*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
            if (rval != ra || ival != ia) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: (%f,%f) actual: (%f,%f)\n",me,
                    i+ilo,jdx[idx[i]+j],ra,ia,rval,ival);
              }
              ok = 0;
            }
          }
        }
      }
    }
  }
  /* Only do non-zero check for integers */
  if (type == C_INT) {
    for (i=0; i<dim*dim; i++) {
      if (((int*)ab)[i] != 0 && nz_map[i] == 0) {
        ok = 0;
      }
    }
  }
  free(ab);
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix-matrix multiply operation PASSES**\n");
      printf("    Time for matrix-matrix multiply operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix-matrix multiply operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;

  for (i=0; i<dim*dim; i++) nz_map[i] = 0;
  time = 0.0;
  for (k=0; k<nprocs; k++) {
    for (l=0; l<nprocs; l++) {
      void *tptr = b;
      tbeg = GA_Wtime();
      if (!NGA_Sprs_array_get_block64(s_b, k, l, &idx, &jdx, &ptr,
          &ilo, &ihi, &jlo, &jhi)) continue;
      time += GA_Wtime()-tbeg;
      /* check for correctness */
      ok = 1;
      for (i=ilo; i<=ihi; i++) {
        int64_t jcols = idx[i+1-ilo]-idx[i-ilo];
        for (j=0; j<jcols; j++) {
          nz_map[i*dim+jdx[idx[i-ilo]+j]] = 1;
          if (type == C_FLOAT) {
            if (((float*)ptr)[idx[i-ilo]+j] 
                != ((float*)tptr)[i*dim+jdx[idx[i-ilo]+j]]) {
              printf("(float) block [%d,%d] element i: %d j: %d"
                  " expected: %f actual: %f\n",
                  me,k,i,jdx[idx[i-ilo]+j],((float*)tptr)[i*dim
                  +jdx[idx[i-ilo]+j]], ((float*)ptr)[idx[i-ilo]+j]);
              ok = 0;
            }
          } else if (type == C_DBL) {
            if (((double*)ptr)[idx[i-ilo]+j]
                != ((double*)tptr)[i*dim+jdx[idx[i-ilo]+j]]) {
              printf("(double) block [%d,%d] element i: %d j: %d"
                  " expected: %f actual: %f\n",
                  me,k,i,jdx[idx[i-ilo]+j],((double*)tptr)[i*dim
                  +jdx[idx[i-ilo]+j]], ((double*)ptr)[idx[i-ilo]+j]);
              ok = 0;
            }
          }
        }
      }
      if (idx != NULL) free(idx);
      if (idx != NULL) free(jdx);
      if (idx != NULL) free(ptr);
    }
  }
  for (i=0; i<dim*dim; i++) {
    if (type == C_FLOAT) {
      if (((float*)c)[i] != 0.0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_DBL) {
      if (((double*)c)[i] != 0.0 && nz_map[i] == 0) {
        ok = 0;
      }
    }
  }

  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)(nprocs*nprocs*nprocs);
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix get block operation PASSES**\n");
      printf("    Time for matrix get block operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix get block operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;
  free(a);
  free(b);
  free(c);
  NGA_Sprs_array_destroy(s_a);
  NGA_Sprs_array_destroy(s_b);
  NGA_Sprs_array_destroy(s_c);

  if (type == C_FLOAT) {
    float bias = 0.0;
    int64_t batch = dim;
    int64_t *categories = (int64_t*)malloc(dim*sizeof(int64_t));;
    for (i=0; i<dim; i++) categories[i] = i;
    /* create sparse matrix A */
    setup_matrix_relu(&s_a, &a, dim, type);
    NGA_Sprs_array_activate_ReLU64(s_a, (void*)&bias, &categories, &batch);
    GA_Sync();

    for (i=0; i<dim; i++) {
      for (j=0; j<dim; j++) {
        float x = ((float*)a)[i*dim+j];
        x = x<0.0?0.0:x>32.0?32.0:x;
        ((float*)a)[i*dim+j] = x;
      }
    }
    for (i=0; i<dim*dim; i++) nz_map[i] = 0;
    time = 0.0;
    ok = 1;
    for (k=0; k<nprocs; k++) {
      for (l=0; l<nprocs; l++) {
        tbeg = GA_Wtime();
        if (!NGA_Sprs_array_get_block64(s_a, k, l, &idx, &jdx, &ptr,
            &ilo, &ihi, &jlo, &jhi)) continue;
        time += GA_Wtime()-tbeg;
        /* check for correctness */
        for (i=ilo; i<=ihi; i++) {
          int64_t jcols = idx[i+1-ilo]-idx[i-ilo];
          for (j=0; j<jcols; j++) {
            nz_map[i*dim+jdx[idx[i-ilo]+j]] = 1;
            /*
            printf("p[%d]         row: %d jcols: %d col: %d\n",me,i,jcols,
            jdx[idx[i-ilo]+j]+jlo);
            */
            if (type == C_FLOAT) {
              if (((float*)ptr)[idx[i-ilo]+j] 
                  != ((float*)a)[i*dim+jdx[idx[i-ilo]+j]]) {
                printf("(float) block [%d,%d] element i: %d j: %d"
                    " expected: %f actual: %f\n",
                    me,k,i,jdx[idx[i-ilo]+j],((float*)a)[i*dim
                    +jdx[idx[i-ilo]+j]], ((float*)ptr)[idx[i-ilo]+j]);
                ok = 0;
              }
            } else if (type == C_DBL) {
              if (((double*)ptr)[idx[i-ilo]+j]
                  != ((double*)a)[i*dim+jdx[idx[i-ilo]+j]]) {
                printf("(double) block [%d,%d] element i: %d j: %d"
                    " expected: %f actual: %f\n",
                    me,k,i,jdx[idx[i-ilo]+j],((double*)a)[i*dim
                    +jdx[idx[i-ilo]+j]], ((double*)ptr)[idx[i-ilo]+j]);
                ok = 0;
              }
            }
          }
        }
        if (idx != NULL) free(idx);
        if (idx != NULL) free(jdx);
        if (idx != NULL) free(ptr);
      }
    }
    for (i=0; i<dim*dim; i++) {
      if (type == C_FLOAT) {
        if (((float*)a)[i] != 0.0 && nz_map[i] == 0) {
          printf("p[%d] NZ: %d a: %f\n",NGA_Nodeid(),i,((float*)a)[i]);
          ok = 0;
        }
      } else if (type == C_DBL) {
        if (((double*)a)[i] != 0.0 && nz_map[i] == 0) {
          printf("p[%d] NZ: %d a: %f\n",NGA_Nodeid(),i,((double*)a)[i]);
          ok = 0;
        }
      }
    }

    GA_Igop(&ok,1,op);
    GA_Dgop(&time,1,plus);
    time /= (double)(nprocs*nprocs*nprocs);
    if (me == 0) {
      if (ok) {
        printf("    **Sparse matrix ReLU operation PASSES**\n");
        printf("    Time for matrix ReLU operation: %16.8f\n",time);
      } else {
        printf("    **Sparse matrix ReLU operation FAILS**\n");
      }
    }
    if (!ok) all_ok = 0;

    for (i=0; i<dim*dim; i++) nz_map[i] = 0;
    time = 0.0;
    for (k=0; k<nprocs; k++) {
      for (l=0; l<nprocs; l++) {
        void *tptr = a;
        tbeg = GA_Wtime();
        if (!NGA_Sprs_array_get_block64(s_a, k, l, &idx, &jdx, &ptr,
            &ilo, &ihi, &jlo, &jhi)) continue;
        time += GA_Wtime()-tbeg;
        /* check for correctness */
        ok = 1;
        for (i=ilo; i<=ihi; i++) {
          int64_t jcols = idx[i+1-ilo]-idx[i-ilo];
          for (j=0; j<jcols; j++) {
            nz_map[i*dim+jdx[idx[i-ilo]+j]] = 1;
            if (type == C_FLOAT) {
              if (((float*)ptr)[idx[i-ilo]+j] 
                  != ((float*)tptr)[i*dim+jdx[idx[i-ilo]+j]]) {
                printf("(float) block [%d,%d] element i: %d j: %d"
                    " expected: %f actual: %f\n",
                    me,k,i,jdx[idx[i-ilo]+j],((float*)tptr)[i*dim
                    +jdx[idx[i-ilo]+j]], ((float*)ptr)[idx[i-ilo]+j]);
                ok = 0;
              }
            } else if (type == C_DBL) {
              if (((double*)ptr)[idx[i-ilo]+j]
                  != ((double*)tptr)[i*dim+jdx[idx[i-ilo]+j]]) {
                printf("(double) block [%d,%d] element i: %d j: %d"
                    " expected: %f actual: %f\n",
                    me,k,i,jdx[idx[i-ilo]+j],((double*)tptr)[i*dim
                    +jdx[idx[i-ilo]+j]], ((double*)ptr)[idx[i-ilo]+j]);
                ok = 0;
              }
            }
          }
        }
        if (idx != NULL) free(idx);
        if (jdx != NULL) free(jdx);
        if (ptr != NULL) free(ptr);
      }
    }
    for (i=0; i<dim*dim; i++) {
      if (type == C_FLOAT) {
        if (((float*)a)[i] != 0.0 && nz_map[i] == 0) {
          ok = 0;
        }
      } else if (type == C_DBL) {
        if (((double*)a)[i] != 0.0 && nz_map[i] == 0) {
          ok = 0;
        }
      }
    }

    GA_Igop(&ok,1,op);
    GA_Dgop(&time,1,plus);
    time /= (double)(nprocs*nprocs*nprocs);
    if (me == 0) {
      if (ok) {
        printf("    **Sparse matrix get block operation PASSES**\n");
        printf("    Time for matrix get block operation: %16.8f\n",time);
      } else {
        printf("    **Sparse matrix get block operation FAILS**\n");
      }
    }
    if (!ok) all_ok = 0;
    NGA_Sprs_array_destroy(s_a);
    free(a);
    free(categories);
  }
  free(nz_map);

  /* create sparse matrix A */
  setup_matrix(&s_a, &a, dim, type, 0);
#if 0
  if (me==0 && type == C_SCPL) {
    printf("Dense matrix A\n\n");
    for (i=0; i<dim; i++) {
      printf("row[%2d]: ",i);
      for (j=0; j<dim; j++) {
        printf(" (%2.0f,%2.0f)",((float*)a)[2*(j+i*dim)],((float*)a)[2*(j+i*dim)+1]);
      }
      printf("\n");
    }
  }
  NGA_Sprs_array_export(s_a,"c_mat.txt");
#endif
  /* create dense matrix B */
  setup_dense_matrix(&g_b, &b, dim, type);
#if 0
  if (me==0 && type == C_SCPL) {
    printf("Dense matrix B\n\n");
    for (i=0; i<dim; i++) {
      printf("row[%2d]: ",i);
      for (j=0; j<dim; j++) {
        printf(" (%2.0f,%2.0f)",((float*)b)[2*(j+i*dim)],((float*)b)[2*(j+i*dim)+1]);
      }
      printf("\n");
    }
  }
//  GA_Print(g_b);
#endif

  /* multiply sparse matrix A times dense matrix B */
  tbeg = GA_Wtime();
  g_c = NGA_Sprs_array_sprsdns_multiply(s_a, g_b);
  time = GA_Wtime()-tbeg;

  /* Do regular matrix-matrix multiply of A and B */
  if (type == C_INT) {
    c = malloc(dim*dim*sizeof(int));
    memset(c,0,dim*dim*sizeof(int));
  } else if (type == C_LONG) {
    c = malloc(dim*dim*sizeof(long));
    memset(c,0,dim*dim*sizeof(long));
  } else if (type == C_LONGLONG) {
    c = malloc(dim*dim*sizeof(long long));
    memset(c,0,dim*dim*sizeof(long long));
  } else if (type == C_FLOAT) {
    c = malloc(dim*dim*sizeof(float));
    memset(c,0,dim*dim*sizeof(float));
  } else if (type == C_DBL) {
    c = malloc(dim*dim*sizeof(double));
    memset(c,0,dim*dim*sizeof(double));
  } else if (type == C_SCPL) {
    c = malloc(dim*dim*2*sizeof(float));
    memset(c,0,dim*dim*2*sizeof(float));
  } else if (type == C_DCPL) {
    c = malloc(dim*dim*2*sizeof(double));
    memset(c,0,dim*dim*2*sizeof(double));
  }

#define REAL_MATMAT_MULTIPLY_M(_type, _a, _b, _c, _dim)     \
{                                                           \
  int _i, _j, _k;                                           \
  _type *_aa = (_type*)_a;                                  \
  _type *_bb = (_type*)_b;                                  \
  _type *_cc = (_type*)_c;                                  \
  for (_i=0; _i<_dim; _i++) {                               \
    for (_j=0; _j<_dim; _j++) {                             \
      _cc[_j+_i*_dim] = (_type)0;                           \
      for(_k=0; _k<_dim; _k++) {                            \
        _cc[_j+_i*_dim] += _aa[_k+_i*_dim]*_bb[_j+_k*_dim]; \
      }                                                     \
    }                                                       \
  }                                                         \
}

#define COMPLEX_MATMAT_MULTIPLY_M(_type, _a, _b, _c, _dim)  \
{                                                           \
  int _i, _j, _k;                                           \
  _type *_aa = (_type*)_a;                                  \
  _type *_bb = (_type*)_b;                                  \
  _type *_cc = (_type*)_c;                                  \
  _type _ar, _ai, _br, _bi;                                 \
  for (_i=0; _i<_dim; _i++) {                               \
    for (_j=0; _j<_dim; _j++) {                             \
      _cc[2*(_j+_i*_dim)] = (_type)0;                       \
      _cc[2*(_j+_i*_dim)+1] = (_type)0;                     \
      for(_k=0; _k<_dim; _k++) {                            \
        _ar = _aa[2*(_k+_i*_dim)];                          \
        _ai = _aa[2*(_k+_i*_dim)+1];                        \
        _br = _bb[2*(_j+_k*_dim)];                          \
        _bi = _bb[2*(_j+_k*_dim)+1];                        \
        _cc[2*(_j+_i*_dim)] += _ar*_br-_ai*_bi;             \
        _cc[2*(_j+_i*_dim)+1] += _ar*_bi+_ai*_br;           \
      }                                                     \
    }                                                       \
  }                                                         \
}

  if (type == C_INT) {
    REAL_MATMAT_MULTIPLY_M(int, a, b, c, dim);
  } else if (type == C_LONG) {
    REAL_MATMAT_MULTIPLY_M(long, a, b, c, dim);
  } else if (type == C_LONGLONG) {
    REAL_MATMAT_MULTIPLY_M(long long, a, b, c, dim);
  } else if (type == C_FLOAT) {
    REAL_MATMAT_MULTIPLY_M(float, a, b, c, dim);
  } else if (type == C_DBL) {
    REAL_MATMAT_MULTIPLY_M(double, a, b, c, dim);
  } else if (type == C_SCPL) {
    COMPLEX_MATMAT_MULTIPLY_M(float, a, b, c, dim);
  } else if (type == C_DCPL) {
    COMPLEX_MATMAT_MULTIPLY_M(double, a, b, c, dim);
  }
#if 0
  if (me==0 && type == C_INT) {
    printf("Dense matrix C\n\n");
    for (i=0; i<dim; i++) {
      printf("row[%2d]: ",i);
      for (j=0; j<dim; j++) {
        printf(" %2d",((int*)c)[j+i*dim]);
      }
      printf("\n");
    }
  }
#endif

  NGA_Sprs_array_destroy(s_a);
  NGA_Destroy(g_b);

  if (type == C_INT) {
    cp = malloc(dim*dim*sizeof(int));
    for (i=0; i<dim*dim; i++) ((int*)cp)[i] = 0;
  } else if (type == C_LONG) {
    cp = malloc(dim*dim*sizeof(long));
    for (i=0; i<dim*dim; i++) ((long*)cp)[i] = 0;
  } else if (type == C_LONGLONG) {
    cp = malloc(dim*dim*sizeof(long long));
    for (i=0; i<dim*dim; i++) ((long long*)cp)[i] = 0;
  } else if (type == C_FLOAT) {
    cp = malloc(dim*dim*sizeof(float));
    for (i=0; i<dim*dim; i++) ((float*)cp)[i] = 0.0;
  } else if (type == C_DBL) {
    cp = malloc(dim*dim*sizeof(double));
    for (i=0; i<dim*dim; i++) ((double*)cp)[i] = 0.0;
  } else if (type == C_SCPL) {
    cp = malloc(dim*dim*2*sizeof(float));
    for (i=0; i<2*dim*dim; i++) ((float*)cp)[i] = 0.0;
  } else if (type == C_DCPL) {
    cp = malloc(dim*dim*2*sizeof(double));
    for (i=0; i<2*dim*dim; i++) ((double*)cp)[i] = 0.0;
  }
  /* Compare results from regular matrix-matrix multiply with
   * sparse-dense matrix-matrix multiply */
  ok = 1;
  lo[0] = 0;
  hi[0] = dim-1;
  lo[1] = 0;
  hi[1] = dim-1;
  tld[0] = dim;
  tld[1] = dim;
  NGA_Get64(g_c,lo,hi,cp,tld);
  GA_Sync();

  /* Compare contents of c and cp (sprsdns multiply vs serial multiply) */
  ok = 1;
  if (type == C_INT) {
    for (i=0; i<dim*dim; i++) {
      if (((int*)c)[i] != ((int*)cp)[i]) ok = 0;
    }
  } else if (type == C_LONG) {
    for (i=0; i<dim*dim; i++) {
      if (((long*)c)[i] != ((long*)cp)[i]) ok = 0;
    }
  } else if (type == C_LONGLONG) {
    for (i=0; i<dim*dim; i++) {
      if (((long long*)c)[i] != ((long long*)cp)[i]) ok = 0;
    }
  } else if (type == C_FLOAT) {
    for (i=0; i<dim*dim; i++) {
      if (((float*)c)[i] != ((float*)cp)[i]) ok = 0;
    }
  } else if (type == C_DBL) {
    for (i=0; i<dim*dim; i++) {
      if (((double*)c)[i] != ((double*)cp)[i]) ok = 0;
    }
  } else if (type == C_SCPL) {
    for (i=0; i<2*dim*dim; i++) {
      if (((float*)c)[i] != ((float*)cp)[i]) ok = 0;
    }
  } else if (type == C_DCPL) {
    for (i=0; i<2*dim*dim; i++) {
      if (((double*)c)[i] != ((double*)cp)[i]) ok = 0;
    }
  }

  free(a);
  free(b);
  free(c);
  free(cp);
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sparse-dense matrix-matrix multiply operation PASSES**\n");
      printf("    Time for sparse-dense matrix-matrix"
          " multiply operation: %16.8f\n",time);
    } else {
      printf("    **Sparse-dense matrix-matrix multiply operation FAILS**\n");
    }
  }

  /* create dense matrix A */
  setup_dense_matrix(&g_a, &a, dim, type);
#if 0
  if (me==0 && type == C_INT) {
    printf("Dense matrix A\n\n");
    for (i=0; i<dim; i++) {
      printf("row[%2d]: ",i);
      for (j=0; j<dim; j++) {
        printf(" %2d",((int*)a)[j+i*dim]);
      }
      printf("\n");
    }
  }
#endif
  /* create sparse matrix B */
  setup_matrix(&s_b, &b, dim, type, 0);
#if 0
  if (me==0 && type == C_INT) {
    printf("Dense matrix B\n\n");
    for (i=0; i<dim; i++) {
      printf("row[%2d]: ",i);
      for (j=0; j<dim; j++) {
        printf(" %2d",((int*)b)[j+i*dim]);
      }
      printf("\n");
    }
  }
//  GA_Print(g_b);
#endif

  /* multiply dense matrix A times sparse matrix B */
  tbeg = GA_Wtime();
  g_c = NGA_Sprs_array_dnssprs_multiply(g_a, s_b);
  time = GA_Wtime()-tbeg;

  /* Do regular matrix-matrix multiply of A and B */
  if (type == C_INT) {
    c = malloc(dim*dim*sizeof(int));
    memset(c,0,dim*dim*sizeof(int));
  } else if (type == C_LONG) {
    c = malloc(dim*dim*sizeof(long));
    memset(c,0,dim*dim*sizeof(long));
  } else if (type == C_LONGLONG) {
    c = malloc(dim*dim*sizeof(long long));
    memset(c,0,dim*dim*sizeof(long long));
  } else if (type == C_FLOAT) {
    c = malloc(dim*dim*sizeof(float));
    memset(c,0,dim*dim*sizeof(float));
  } else if (type == C_DBL) {
    c = malloc(dim*dim*sizeof(double));
    memset(c,0,dim*dim*sizeof(double));
  } else if (type == C_SCPL) {
    c = malloc(dim*dim*2*sizeof(float));
    memset(c,0,dim*dim*2*sizeof(float));
  } else if (type == C_DCPL) {
    c = malloc(dim*dim*2*sizeof(double));
    memset(c,0,dim*dim*2*sizeof(double));
  }

  if (type == C_INT) {
    REAL_MATMAT_MULTIPLY_M(int, a, b, c, dim);
  } else if (type == C_LONG) {
    REAL_MATMAT_MULTIPLY_M(long, a, b, c, dim);
  } else if (type == C_LONGLONG) {
    REAL_MATMAT_MULTIPLY_M(long long, a, b, c, dim);
  } else if (type == C_FLOAT) {
    REAL_MATMAT_MULTIPLY_M(float, a, b, c, dim);
  } else if (type == C_DBL) {
    REAL_MATMAT_MULTIPLY_M(double, a, b, c, dim);
  } else if (type == C_SCPL) {
    COMPLEX_MATMAT_MULTIPLY_M(float, a, b, c, dim);
  } else if (type == C_DCPL) {
    COMPLEX_MATMAT_MULTIPLY_M(double, a, b, c, dim);
  }
#if 0
  if (me==0 && type == C_INT) {
    printf("Dense matrix C\n\n");
    for (i=0; i<dim; i++) {
      printf("row[%2d]: ",i);
      for (j=0; j<dim; j++) {
        printf(" %2d",((int*)c)[j+i*dim]);
      }
      printf("\n");
    }
  }
#endif


#undef REAL_MATMAT_MULTIPLY_M
#undef COMPLEX_MATMAT_MULTIPLY_M

  NGA_Destroy(g_a);
  NGA_Sprs_array_destroy(s_b);

  if (type == C_INT) {
    cp = malloc(dim*dim*sizeof(int));
    for (i=0; i<dim*dim; i++) ((int*)cp)[i] = 0;
  } else if (type == C_LONG) {
    cp = malloc(dim*dim*sizeof(long));
    for (i=0; i<dim*dim; i++) ((long*)cp)[i] = 0;
  } else if (type == C_LONGLONG) {
    cp = malloc(dim*dim*sizeof(long long));
    for (i=0; i<dim*dim; i++) ((long long*)cp)[i] = 0;
  } else if (type == C_FLOAT) {
    cp = malloc(dim*dim*sizeof(float));
    for (i=0; i<dim*dim; i++) ((float*)cp)[i] = 0.0;
  } else if (type == C_DBL) {
    cp = malloc(dim*dim*sizeof(double));
    for (i=0; i<dim*dim; i++) ((double*)cp)[i] = 0.0;
  } else if (type == C_SCPL) {
    cp = malloc(dim*dim*2*sizeof(float));
    for (i=0; i<2*dim*dim; i++) ((float*)cp)[i] = 0.0;
  } else if (type == C_DCPL) {
    cp = malloc(dim*dim*2*sizeof(double));
    for (i=0; i<2*dim*dim; i++) ((double*)cp)[i] = 0.0;
  }
  /* Compare results from regular matrix-matrix multiply with
   * sparse-dense matrix-matrix multiply */
  ok = 1;
  lo[0] = 0;
  hi[0] = dim-1;
  lo[1] = 0;
  hi[1] = dim-1;
  tld[0] = dim;
  tld[1] = dim;
  NGA_Get64(g_c,lo,hi,cp,tld);
  GA_Sync();

  /* Compare contents of c and cp (sprsdns multiply vs serial multiply) */
  ok = 1;
  if (type == C_INT) {
    for (i=0; i<dim*dim; i++) {
      if (((int*)c)[i] != ((int*)cp)[i]) ok = 0;
    }
  } else if (type == C_LONG) {
    for (i=0; i<dim*dim; i++) {
      if (((long*)c)[i] != ((long*)cp)[i]) ok = 0;
    }
  } else if (type == C_LONGLONG) {
    for (i=0; i<dim*dim; i++) {
      if (((long long*)c)[i] != ((long long*)cp)[i]) ok = 0;
    }
  } else if (type == C_FLOAT) {
    for (i=0; i<dim*dim; i++) {
      if (((float*)c)[i] != ((float*)cp)[i]) ok = 0;
    }
  } else if (type == C_DBL) {
    for (i=0; i<dim*dim; i++) {
      if (((double*)c)[i] != ((double*)cp)[i]) ok = 0;
    }
  } else if (type == C_SCPL) {
    for (i=0; i<2*dim*dim; i++) {
      if (((float*)c)[i] != ((float*)cp)[i]) ok = 0;
    }
  } else if (type == C_DCPL) {
    for (i=0; i<2*dim*dim; i++) {
      if (((double*)c)[i] != ((double*)cp)[i]) ok = 0;
    }
  }

#if 0
  if (me==0 && type == C_INT) {
    printf("Dense matrix Cp\n\n");
    for (i=0; i<dim; i++) {
      printf("row[%2d]: ",i);
      for (j=0; j<dim; j++) {
        printf(" %2d",((int*)cp)[j+i*dim]);
      }
      printf("\n");
    }
  }
#endif
  free(a);
  free(b);
  free(c);
  free(cp);
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Dense-sparse matrix-matrix multiply operation PASSES**\n");
      printf("    Time for dense-sparse matrix-matrix"
          " multiply operation: %16.8f\n",time);
    } else {
      printf("    **Dense-sparse matrix-matrix multiply operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;

  /* create sparse matrix A */
  setup_matrix(&s_a, &a, dim, type, 0);
  /* create sparse matrix B */
  setup_matrix(&s_b, &b, dim, type, 1);

  /* multiply sparse matrix A times sparse matrix B */
  tbeg = GA_Wtime();
  s_c = NGA_Sprs_array_elementwise_multiply(s_a, s_b);
  time = GA_Wtime()-tbeg;


  /* Do regular matrix-matrix multiply of A and B */
  if (type == C_INT) {
    c = malloc(dim*dim*sizeof(int));
  } else if (type == C_LONG) {
    c = malloc(dim*dim*sizeof(long));
  } else if (type == C_LONGLONG) {
    c = malloc(dim*dim*sizeof(long long));
  } else if (type == C_FLOAT) {
    c = malloc(dim*dim*sizeof(float));
  } else if (type == C_DBL) {
    c = malloc(dim*dim*sizeof(double));
  } else if (type == C_SCPL) {
    c = malloc(dim*dim*2*sizeof(float));
  } else if (type == C_DCPL) {
    c = malloc(dim*dim*2*sizeof(double));
  }

#define REAL_ELEMENTWISE_MULTIPLY_M(_type, _a, _b, _c, _dim)    \
{                                                               \
  int _i, _j;                                                   \
  _type *_aa = (_type*)_a;                                      \
  _type *_bb = (_type*)_b;                                      \
  _type *_cc = (_type*)_c;                                      \
  for (_i=0; _i<_dim; _i++) {                                   \
    for (_j=0; _j<_dim; _j++) {                                 \
      _cc[_j+_i*_dim] = _aa[_j+_i*_dim]*_bb[_j+_i*_dim];        \
    }                                                           \
  }                                                             \
}

#define COMPLEX_ELEMENTWISE_MULTIPLY_M(_type, _a, _b, _c, _dim) \
{                                                               \
  int _i, _j;                                                   \
  _type *_aa = (_type*)_a;                                      \
  _type *_bb = (_type*)_b;                                      \
  _type *_cc = (_type*)_c;                                      \
  for (_i=0; _i<_dim; _i++) {                                   \
    for (_j=0; _j<_dim; _j++) {                                 \
      _type _ar = _aa[2*(_j+_i*_dim)];                          \
      _type _ai = _aa[2*(_j+_i*_dim)+1];                        \
      _type _br = _bb[2*(_j+_i*_dim)];                          \
      _type _bi = _bb[2*(_j+_i*_dim)+1];                        \
      _cc[2*(_j+_i*_dim)] = _ar*_br-_ai*_bi;                    \
      _cc[2*(_j+_i*_dim)+1] = _ar*_bi+_ai*_br;                  \
    }                                                           \
  }                                                             \
}

  if (type == C_INT) {
    REAL_ELEMENTWISE_MULTIPLY_M(int, a, b, c, dim);
    REAL_ELEMENTWISE_MULTIPLY_M(int, c, a, b, dim);
  } else if (type == C_LONG) {
    REAL_ELEMENTWISE_MULTIPLY_M(long, a, b, c, dim);
    REAL_ELEMENTWISE_MULTIPLY_M(long, c, a, b, dim);
  } else if (type == C_LONGLONG) {
    REAL_ELEMENTWISE_MULTIPLY_M(long long, a, b, c, dim);
    REAL_ELEMENTWISE_MULTIPLY_M(long long, c, a, b, dim);
  } else if (type == C_FLOAT) {
    REAL_ELEMENTWISE_MULTIPLY_M(float, a, b, c, dim);
    REAL_ELEMENTWISE_MULTIPLY_M(float, c, a, b, dim);
  } else if (type == C_DBL) {
    REAL_ELEMENTWISE_MULTIPLY_M(double, a, b, c, dim);
    REAL_ELEMENTWISE_MULTIPLY_M(double, c, a, b, dim);
  } else if (type == C_SCPL) {
    COMPLEX_ELEMENTWISE_MULTIPLY_M(float, a, b, c, dim);
    COMPLEX_ELEMENTWISE_MULTIPLY_M(float, c, a, b, dim);
  } else if (type == C_DCPL) {
    COMPLEX_ELEMENTWISE_MULTIPLY_M(double, a, b, c, dim);
    COMPLEX_ELEMENTWISE_MULTIPLY_M(double, c, a, b, dim);
  }

#undef REAL_ELEMENTWISE_MULTIPLY_M
#undef COMPLEX_ELEMENTWISE_MULTIPLY_M

  NGA_Sprs_array_destroy(s_b);
  s_b = NGA_Sprs_array_elementwise_multiply(s_c, s_a);

  ab = (int*)malloc(dim*dim*sizeof(int));
  for (i=0; i<dim*dim; i++) ab[i] = 0;
  nz_map = (int*)malloc(dim*dim*sizeof(int));
  for (i=0; i<dim*dim; i++) nz_map[i] = 0;
  /* Compare results from regular matrix-matrix multiply with
   * sparse matrix-matrix multiply */
  ok = 1;
  NGA_Sprs_array_row_distribution64(s_b, me, &ilo, &ihi);
  /* loop over column blocks */
  ld = 0;
  for (iproc = 0; iproc<nprocs; iproc++) {
    int64_t nrows = ihi-ilo+1;
    /* column block corresponding to iproc has data. Get pointers
     * to index and data arrays */
    NGA_Sprs_array_access_col_block64(s_b, iproc, &idx, &jdx, &ptr);
    /*
        printf("p[%d] iproc: %ld idx: %p jdx: %p ptr: %p\n",me,iproc,idx,jdx,ptr);
        */
    if (idx != NULL) {
      for (i=0; i<nrows; i++) {
        int64_t nvals = idx[i+1]-idx[i];
        /*
        printf("p[%d] iproc: %ld i: %ld nrows: %ld nvals: %ld\n",me,
            iproc,i+ilo,nrows,nvals);
            */
        for (j=0; j<nvals; j++) {
          void *tptr = b;
          /*
        printf("p[%d]     iproc: %ld i: %ld j: %ld nvals: %ld\n",me,
            iproc,i+ilo,jdx[idx[i]+j],nvals);
            */
          ld++;
          nz_map[(i+ilo)*dim+jdx[idx[i]+j]] = 1;
          if (type == C_INT) {
            /*
            printf("p[%d]        proc: %ld i: %ld j: %ld c: %d\n",me,
                iproc,i+ilo,jdx[idx[i]+j],((int*)ptr)[idx[i]+j]);
                */
            ab[(i+ilo)*dim+jdx[idx[i]+j]] = ((int*)ptr)[idx[i]+j];
            if (((int*)ptr)[idx[i]+j] != ((int*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %d actual: %d\n",me,
                    i+ilo,jdx[idx[i]+j],((int*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((int*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_LONG) {
            if (((long*)ptr)[idx[i]+j] != ((long*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                    i+ilo,jdx[idx[i]+j],((long*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((long*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_LONGLONG) {
            if (((long long*)ptr)[idx[i]+j] 
                != ((long long*)tptr)[(i+ilo)*dim + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                    i+ilo,jdx[idx[i]+j],(long)((long long*)tptr)[(i+ilo)*dim
                    +jdx[idx[i]+j]],
                    (long)((long long*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_FLOAT) {
            if (((float*)ptr)[idx[i]+j] != ((float*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                    i+ilo,jdx[idx[i]+j],((float*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((float*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_DBL) {
            if (((double*)ptr)[idx[i]+j] != ((double*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                    i+ilo,jdx[idx[i]+j],((double*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((double*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_SCPL) {
            float rval = ((float*)ptr)[2*(idx[i]+j)];
            float ival = ((float*)ptr)[2*(idx[i]+j)+1];
            float ra = ((float*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])];
            float ia = ((float*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
            if (rval != ra || ival != ia) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: (%f,%f) actual: (%f,%f)\n",me,
                    i+ilo,jdx[idx[i]+j],ra,ia,rval,ival);
              }
              ok = 0;
            }
          } else if (type == C_DCPL) {
            double rval = ((double*)ptr)[2*(idx[i]+j)];
            double ival = ((double*)ptr)[2*(idx[i]+j)+1];
            double ra = ((double*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])];
            double ia = ((double*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
            if (rval != ra || ival != ia) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: (%f,%f) actual: (%f,%f)\n",me,
                    i+ilo,jdx[idx[i]+j],ra,ia,rval,ival);
              }
              ok = 0;
            }
          }
        }
      }
    }
  }
  /* Only do non-zero check for integers */
  if (type == C_INT) {
    for (i=0; i<dim*dim; i++) {
      if (((int*)ab)[i] != 0 && nz_map[i] == 0) {
        ok = 0;
      }
    }
  }
  free(a);
  free(b);
  free(c);
  free(ab);
  free(nz_map);
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sparse elementwise multiply operation PASSES**\n");
      printf("    Time for elementwise multiply operation: %16.8f\n",time);
    } else {
      printf("    **Sparse elementwise multiply operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;

  /* Create two dense matrices with mostly non-zero values */
  setup_cdense_matrix(&g_a, &a, dim, type);
  setup_cdense_matrix(&g_b, &b, dim, type);
  setup_matrix(&s_c, &c, dim, type, 0);

  tbeg = GA_Wtime();
  NGA_Sprs_array_sampled_multiply(g_a, g_b, s_c, 0);
  time = GA_Wtime()-tbeg;

  /* compare results in sparse matrix with result from direct
   * sampled multiply */
#define REAL_SAMPLE_ELEMENT_M(_type)                  \
{                                                     \
  if (((_type*)c)[j+i*dim] != (_type)0) {             \
    ((_type*)c)[j+i*dim] = (_type)0;                  \
    for (k=0; k<dim; k++) {                           \
      ((_type*)c)[j+i*dim] += ((_type*)a)[k+i*dim]    \
                           * ((_type*)b)[j+k*dim];    \
    }                                                 \
  }                                                   \
}

#define COMPLEX_SAMPLE_ELEMENT_M(_type)               \
{                                                     \
  if (((_type*)c)[2*(j+i*dim)] != (_type)0 ||         \
      ((_type*)c)[2*(j+i*dim)+1] != (_type)0) {       \
    ((_type*)c)[2*(j+i*dim)] = (_type)0;              \
    ((_type*)c)[2*(j+i*dim)+1] = (_type)0;            \
    for (k=0; k<dim; k++) {                           \
      _type ra, ia, rb, ib;                           \
      ra = ((_type*)a)[2*(k+i*dim)];                  \
      ia = ((_type*)a)[2*(k+i*dim)+1];                \
      rb = ((_type*)b)[2*(j+k*dim)];                  \
      ib = ((_type*)b)[2*(j+k*dim)+1];                \
      ((_type*)c)[2*(j+i*dim)] += ra*rb-ia*ib;        \
      ((_type*)c)[2*(j+i*dim)+1] += ra*ib+ia*rb;      \
    }                                                 \
  }                                                   \
}

  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      if (type == C_INT) {
        REAL_SAMPLE_ELEMENT_M(int);
      } else if (type == C_LONG) {
        REAL_SAMPLE_ELEMENT_M(long);
      } else if (type == C_LONGLONG) {
        REAL_SAMPLE_ELEMENT_M(long long);
      } else if (type == C_FLOAT) {
        REAL_SAMPLE_ELEMENT_M(float);
      } else if (type == C_DBL) {
        REAL_SAMPLE_ELEMENT_M(double);
      } else if (type == C_SCPL) {
        COMPLEX_SAMPLE_ELEMENT_M(float);
      } else if (type == C_DCPL) {
        COMPLEX_SAMPLE_ELEMENT_M(double);
      }
    }
  }
#undef REAL_SAMPLE_ELEMENT_M
#undef COMPLEX_SAMPLE_ELEMENT_M

  /* Compare result in matrix c with sparse matrix s_c */
  ok = 1;
  NGA_Sprs_array_row_distribution64(s_c, me, &ilo, &ihi);
  /* loop over column blocks */
  ld = 0;
  for (iproc = 0; iproc<nprocs; iproc++) {
    int64_t nrows = ihi-ilo+1;
    NGA_Sprs_array_access_col_block64(s_c, iproc, &idx, &jdx, &ptr);
    /* column block corresponding to iproc has data. Get pointers
     * to index and data arrays */
    if (idx != NULL) {
      for (i=0; i<nrows; i++) {
        int64_t nvals = idx[i+1]-idx[i];
        for (j=0; j<nvals; j++) {
          void *tptr = c;
          ld++;
          if (type == C_INT) {
            if (((int*)ptr)[idx[i]+j] != ((int*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %d actual: %d\n",me,
                    i+ilo,jdx[idx[i]+j],((int*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((int*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_LONG) {
            if (((long*)ptr)[idx[i]+j] != ((long*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                    i+ilo,jdx[idx[i]+j],((long*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((long*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_LONGLONG) {
            if (((long long*)ptr)[idx[i]+j] 
                != ((long long*)tptr)[(i+ilo)*dim + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                    i+ilo,jdx[idx[i]+j],(long)((long long*)tptr)[(i+ilo)*dim
                    +jdx[idx[i]+j]],
                    (long)((long long*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_FLOAT) {
            if (((float*)ptr)[idx[i]+j] != ((float*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                    i+ilo,jdx[idx[i]+j],((float*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((float*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_DBL) {
            if (((double*)ptr)[idx[i]+j] != ((double*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                    i+ilo,jdx[idx[i]+j],((double*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((double*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_SCPL) {
            float rval = ((float*)ptr)[2*(idx[i]+j)];
            float ival = ((float*)ptr)[2*(idx[i]+j)+1];
            float ra = ((float*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])];
            float ia = ((float*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
            if (rval != ra || ival != ia) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: (%f,%f) actual: (%f,%f)\n",me,
                    i+ilo,jdx[idx[i]+j],ra,ia,rval,ival);
              }
              ok = 0;
            }
          } else if (type == C_DCPL) {
            double rval = ((double*)ptr)[2*(idx[i]+j)];
            double ival = ((double*)ptr)[2*(idx[i]+j)+1];
            double ra = ((double*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])];
            double ia = ((double*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
            if (rval != ra || ival != ia) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: (%f,%f) actual: (%f,%f)\n",me,
                    i+ilo,jdx[idx[i]+j],ra,ia,rval,ival);
              }
              ok = 0;
            }
          }
        }
      }
    }
  }

  free(a);
  free(b);
  free(c);
  GA_Destroy(g_a);
  GA_Destroy(g_b);
  NGA_Sprs_array_destroy(s_c);
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sampled matrix-matrix multiply operation PASSES**\n");
      printf("    Time for sampled multiply operation: %16.8f\n",time);
    } else {
      printf("    **Sparse sampled matrix-matrix multiply operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;

  /* create an ordinary global array with sparse non-zeros */
  setup_dense_matrix(&g_a, &a, dim, type);
  /* copy dense matrix g_a to sparse matrix s_a */
  tbeg = GA_Wtime();
  s_a = NGA_Sprs_array_create_from_dense64(g_a);
  time = GA_Wtime()-tbeg;
  /* check values in sparse array */
  nz_map = (int*)malloc(dim*dim*sizeof(int));
  for (i=0; i<dim*dim; i++) nz_map[i] = 0;
  ab = (int*)malloc(dim*dim*sizeof(int));
  for (i=0; i<dim*dim; i++) ab[i] = 0;
  ok = 1;
  NGA_Sprs_array_row_distribution64(s_a, me, &ilo, &ihi);
  /* loop over column blocks */
  ld = 0;
  for (iproc = 0; iproc<nprocs; iproc++) {
    int64_t nrows = ihi-ilo+1;
    /* column block corresponding to iproc has data. Get pointers
     * to index and data arrays */
    NGA_Sprs_array_access_col_block64(s_a, iproc, &idx, &jdx, &ptr);
    if (idx != NULL) {
      for (i=0; i<nrows; i++) {
        int64_t nvals = idx[i+1]-idx[i];
        for (j=0; j<nvals; j++) {
          void *tptr = a;
          ld++;
          nz_map[(i+ilo)*dim+jdx[idx[i]+j]] = 1;
          if (type == C_INT) {
            ab[(i+ilo)*dim+jdx[idx[i]+j]] = ((int*)ptr)[idx[i]+j];
            if (((int*)ptr)[idx[i]+j] != ((int*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %d actual: %d\n",me,
                    i+ilo,jdx[idx[i]+j],((int*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((int*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_LONG) {
            if (((long*)ptr)[idx[i]+j] != ((long*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                    i+ilo,jdx[idx[i]+j],((long*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((long*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_LONGLONG) {
            if (((long long*)ptr)[idx[i]+j] 
                != ((long long*)tptr)[(i+ilo)*dim + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                    i+ilo,jdx[idx[i]+j],(long)((long long*)tptr)[(i+ilo)*dim
                    +jdx[idx[i]+j]],
                    (long)((long long*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_FLOAT) {
            if (((float*)ptr)[idx[i]+j] != ((float*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                    i+ilo,jdx[idx[i]+j],((float*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((float*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_DBL) {
            if (((double*)ptr)[idx[i]+j] != ((double*)tptr)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                    i+ilo,jdx[idx[i]+j],((double*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((double*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_SCPL) {
            float rval = ((float*)ptr)[2*(idx[i]+j)];
            float ival = ((float*)ptr)[2*(idx[i]+j)+1];
            float ra = ((float*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])];
            float ia = ((float*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
            if (rval != ra || ival != ia) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: (%f,%f) actual: (%f,%f)\n",me,
                    i+ilo,jdx[idx[i]+j],ra,ia,rval,ival);
              }
              ok = 0;
            }
          } else if (type == C_DCPL) {
            double rval = ((double*)ptr)[2*(idx[i]+j)];
            double ival = ((double*)ptr)[2*(idx[i]+j)+1];
            double ra = ((double*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])];
            double ia = ((double*)tptr)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
            if (rval != ra || ival != ia) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: (%f,%f) actual: (%f,%f)\n",me,
                    i+ilo,jdx[idx[i]+j],ra,ia,rval,ival);
              }
              ok = 0;
            }
          }
        }
      }
    }
  }
  /* Only do non-zero check for integers */
  if (type == C_INT) {
    for (i=0; i<dim*dim; i++) {
      if (((int*)ab)[i] != 0 && nz_map[i] == 0) {
        ok = 0;
      }
    }
  }
  GA_Destroy(g_a);
  NGA_Sprs_array_destroy(s_a);
  free(ab);
  free(nz_map);
  free(a);
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Create from dense array operation PASSES**\n");
      printf("    Time for create from dense array operation: %16.8f\n",time);
    } else {
      printf("    **Sparse create from dense array operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;

  /* create an ordinary global array with sparse non-zeros */
  setup_dense_matrix(&g_a, &a, dim, type);
  /* copy dense matrix g_a to sparse matrix s_a */
  s_a = NGA_Sprs_array_create_from_dense64(g_a);
  /* now copy sparse matrix back to dense matrix g_b */
  tbeg = GA_Wtime();
  g_b = NGA_Sprs_array_create_from_sparse(s_a);
  time = GA_Wtime()-tbeg;
  g_c = GA_Duplicate(g_a,"new copy");
  NGA_Zero(g_c);
  /* check to see if answer is correct by subtracting g_a from g_b */
  ok = 1;
  {
    int lo[2], hi[2];
    double diff;
    lo[0] = 0;
    hi[0] = dim-1;
    lo[1] = 0;
    hi[1] = dim-1;
    if (type == C_INT) {
      int tone = 1;
      int tmone = -1;
      NGA_Add_patch(&tone,g_a,lo,hi,&tmone,g_b,lo,hi,g_c,lo,hi);
      diff = (double)GA_Idot(g_c,g_c);
    } else if (type == C_LONG) {
      long tone = 1;
      long tmone = -1;
      NGA_Add_patch(&tone,g_a,lo,hi,&tmone,g_b,lo,hi,g_c,lo,hi);
      diff = (double)GA_Ldot(g_c,g_c);
    } else if (type == C_LONGLONG) {
      long long tone = 1;
      long long tmone = -1;
      NGA_Add_patch(&tone,g_a,lo,hi,&tmone,g_b,lo,hi,g_c,lo,hi);
      diff = (double)GA_Lldot(g_c,g_c);
    } else if (type == C_FLOAT) {
      float tone = 1.0;
      float tmone = -1.0;
      NGA_Add_patch(&tone,g_a,lo,hi,&tmone,g_b,lo,hi,g_c,lo,hi);
      diff = (double)GA_Fdot(g_c,g_c);
    } else if (type == C_DBL) {
      double tone = 1.0;
      double tmone = -1.0;
      NGA_Add_patch(&tone,g_a,lo,hi,&tmone,g_b,lo,hi,g_c,lo,hi);
      diff = (double)GA_Ddot(g_c,g_c);
    } else if (type == C_SCPL) {
      float tone[2] = {1.0,0.0};
      float tmone[2] = {-1.0,0.0};
      float *cdiff;
      SingleComplex zdiff;
      NGA_Add_patch(&tone,g_a,lo,hi,&tmone,g_b,lo,hi,g_c,lo,hi);
      zdiff = GA_Cdot(g_c,g_c);
      cdiff = (float*)&zdiff;
      diff = (double)cdiff[0];
    } else if (type == C_DCPL) {
      double tone[2] = {1.0,0.0};
      double tmone[2] = {-1.0,0.0};
      double *cdiff;
      DoubleComplex zdiff;
      NGA_Add_patch(&tone,g_a,lo,hi,&tmone,g_b,lo,hi,g_c,lo,hi);
      zdiff = GA_Zdot(g_c,g_c);
      cdiff = (double*)&zdiff;
      diff = (double)cdiff[0];
    }
    if (diff > 1.0e-8) ok = 0;
  }
  GA_Destroy(g_a);
  GA_Destroy(g_b);
  GA_Destroy(g_c);
  NGA_Sprs_array_destroy(s_a);
  free(a);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Create from sparse array operation PASSES**\n");
      printf("    Time for create from sparse array operation: %16.8f\n",time);
    } else {
      printf("    **Sparse create from sparse array operation FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;


  /* set up sparse array with range of values */
  if (type != C_SCPL && type != C_DCPL) {
    setup_quant_matrix(&s_a, &a, dim, type);
#if 1
    /* apply quantization operation to sparse array */
    tbeg = GA_Wtime();
    if (type == C_INT) {
      int max = 128;
      s_b = NGA_Sprs_array_quantize256(s_a, &max);
    } else if (type == C_LONG) {
      long max = 128;
      s_b = NGA_Sprs_array_quantize256(s_a, &max);
    } else if (type == C_LONGLONG) {
      long long max = 128;
      s_b = NGA_Sprs_array_quantize256(s_a, &max);
    } else if (type == C_FLOAT) {
      float max = 128.0;
      s_b = NGA_Sprs_array_quantize256(s_a, &max);
    } else if (type == C_DBL) {
      double max = 128.0;
      s_b = NGA_Sprs_array_quantize256(s_a, &max);
    }
    time = GA_Wtime()-tbeg;
    /* apply quantization operation to local array */
    for (i=0; i<dim; i++) {
      for (j=0; j<dim; j++) {
        if (type == C_INT) {
          int *aa = (int*)a;
          aa[i*dim+j] = (int)((((double)aa[i*dim+j])/128.0)*255.0);
        } else if (type == C_LONG) {
          long *aa = (long*)a;
          aa[i*dim+j] = (long)((int)((((double)aa[i*dim+j])/128.0)*255.0));
        } else if (type == C_LONGLONG) {
          long long *aa = (long long*)a;
          aa[i*dim+j] = (long long)((int)((((double)aa[i*dim+j])/128.0)*255.0));
        } else if (type == C_FLOAT) {
          float *aa = (float*)a;
          aa[i*dim+j] = (float)((int)((((double)aa[i*dim+j])/128.0)*255.0));
        } else if (type == C_DBL) {
          double *aa = (double*)a;
          aa[i*dim+j] = (double)((int)((((double)aa[i*dim+j])/128.0)*255.0));
        }
      }
    }
    ok = 1;
    NGA_Sprs_array_row_distribution64(s_b, me, &ilo, &ihi);
    /* loop over column blocks */
    ld = 0;
    for (iproc = 0; iproc<nprocs; iproc++) {
      int64_t nrows = ihi-ilo+1;
      /* column block corresponding to iproc has data. Get pointers
       * to index and data arrays */
      NGA_Sprs_array_access_col_block64(s_b, iproc, &idx, &jdx, &ptr);
      if (idx != NULL) {
        for (i=0; i<nrows; i++) {
          int64_t nvals = idx[i+1]-idx[i];
          for (j=0; j<nvals; j++) {
            void *tptr = a;
            ld++;
            if (type == C_INT) {
              if (((int*)ptr)[idx[i]+j] != ((int*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]]) {
                if (ok) {
                  printf("p[%d] [%ld,%ld] expected: %d actual: %d\n",me,
                      i+ilo,jdx[idx[i]+j],((int*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                      ((int*)ptr)[idx[i]+j]);
                }
                ok = 0;
              }
            } else if (type == C_LONG) {
              if (((long*)ptr)[idx[i]+j] != ((long*)tptr)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                if (ok) {
                  printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                      i+ilo,jdx[idx[i]+j],((long*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                      ((long*)ptr)[idx[i]+j]);
                }
                ok = 0;
              }
            } else if (type == C_LONGLONG) {
              if (((long long*)ptr)[idx[i]+j] 
                  != ((long long*)tptr)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                if (ok) {
                  printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                      i+ilo,jdx[idx[i]+j],(long)((long long*)tptr)[(i+ilo)*dim
                      +jdx[idx[i]+j]],
                      (long)((long long*)ptr)[idx[i]+j]);
                }
                ok = 0;
              }
            } else if (type == C_FLOAT) {
              if (((float*)ptr)[idx[i]+j] != ((float*)tptr)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                if (ok) {
                  printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                      i+ilo,jdx[idx[i]+j],((float*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                      ((float*)ptr)[idx[i]+j]);
                }
                ok = 0;
              }
            } else if (type == C_DBL) {
              if (((double*)ptr)[idx[i]+j] != ((double*)tptr)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                if (ok) {
                  printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                      i+ilo,jdx[idx[i]+j],((double*)tptr)[(i+ilo)*dim+jdx[idx[i]+j]],
                      ((double*)ptr)[idx[i]+j]);
                }
                ok = 0;
              }
            }
          }
        }
      }
    }
#endif
    free(a);
    NGA_Sprs_array_destroy(s_a);
    NGA_Sprs_array_destroy(s_b);
    GA_Igop(&ok,1,op);
    GA_Dgop(&time,1,plus);
    time /= (double)nprocs;
    if (me == 0) {
      if (ok) {
        printf("    **Quantize 256 operation PASSES**\n");
        printf("    Time for quantize 256 operation: %16.8f\n",time);
      } else {
        printf("    **Sparse quantize 256 operation FAILS**\n");
      }
    }
  }
  if (!ok) all_ok = 0;

  return all_ok;
}

int main(int argc, char **argv) {
  int me,nproc;
  int ok = 1;

  /**
   * Initialize GA
   */
  MP_INIT(argc,argv);

  /* Initialize GA */
  NGA_Initialize();

  me = GA_Nodeid();
  nproc = GA_Nnodes();
  if (me == 0) {
    printf("\nTesting sparse matrices of size %d x %d on %d processors\n\n",
        NDIM,NDIM,nproc);
  }

  /**
   * Test different data types
   */
  if (me == 0) {
    printf("\nTesting matrices of type int\n");
  }
  ok = ok &&  matrix_test(C_INT);

#if 1
  if (me == 0) {
    printf("\nTesting matrices of type long\n");
  }
  ok = ok && matrix_test(C_LONG);

  if (me == 0) {
    printf("\nTesting matrices of type long long\n");
  }
  ok = ok && matrix_test(C_LONGLONG);

  if (me == 0) {
    printf("\nTesting matrices of type float\n");
  }
  ok = ok && matrix_test(C_FLOAT);

  if (me == 0) {
    printf("\nTesting matrices of type double\n");
  }
  ok = ok && matrix_test(C_DBL);

  if (me == 0) {
    printf("\nTesting matrices of type single complex\n");
  }
  ok = ok && matrix_test(C_SCPL);

  if (me == 0) {
    printf("\nTesting matrices of type double complex\n");
  }
  ok = ok && matrix_test(C_DCPL);
#endif

  if (me == 0) {
    printf("\nSparse matrix tests complete\n\n");
    if (ok) {
      printf("All tests PASSED\n");
    } else {
      printf("Some tests FAILED\n");
    }
  }

  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
