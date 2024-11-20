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

int sddmm_test(int type)
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
  int64_t lo[2],hi[2],tld[2];
  int all_ok = 1;

  op[0] = '*';
  op[1] = '\0';
  plus[0] = '+';
  plus[1] = '\0';

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

  tbeg = GA_Wtime();
  NGA_Sprs_array_sampled_multiply(g_a, g_b, s_c, 1);
  time = GA_Wtime()-tbeg;
#if 0
  if (type == C_INT && me == 0) {
    printf("\nMatrix A\n\n");
    for (i=0; i<dim; i++) {
      for (j=0; j<dim; j++) {
        printf(" %6d",((int*)a)[i*dim+j]);
      }
      printf("\n");
    }
  }
  if (type == C_INT && me == 0) {
    printf("\nTranspose Matrix B\n\n");
    for (i=0; i<dim; i++) {
      for (j=0; j<dim; j++) {
        printf(" %6d",((int*)b)[j*dim+i]);
      }
      printf("\n");
    }
  }
  if (type == C_INT && me == 0) {
    printf("\nMatrix C\n\n");
    for (i=0; i<dim; i++) {
      for (j=0; j<dim; j++) {
        printf(" %6d",((int*)c)[i*dim+j]);
      }
      printf("\n");
    }
  }
#endif
#undef COMPLEX_SAMPLE_ELEMENT_M

  /* compare results in sparse matrix with result from direct
   * sampled multiply */
#define REAL_SAMPLE_ELEMENT_M(_type)                  \
{                                                     \
  if (((_type*)c)[j+i*dim] != (_type)0) {             \
    ((_type*)c)[j+i*dim] = (_type)0;                  \
    for (k=0; k<dim; k++) {                           \
      ((_type*)c)[j+i*dim] += ((_type*)a)[k+i*dim]    \
                           * ((_type*)b)[k+j*dim];    \
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
      rb = ((_type*)b)[2*(k+j*dim)];                  \
      ib = ((_type*)b)[2*(k+j*dim)+1];                \
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
#if 0
  if (type == C_INT && me == 0) {
    printf("\nMatrix C'\n\n");
    for (i=0; i<dim; i++) {
      for (j=0; j<dim; j++) {
        printf(" %6d",((int*)c)[i*dim+j]);
      }
      printf("\n");
    }
    printf("\n");
  }
#endif

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

  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sampled matrix-matrix multiply operation with transpose PASSES**\n");
      printf("    Time for sampled multiply operation with transpose: %16.8f\n",time);
    } else {
      printf("    **Sparse sampled matrix-matrix multiply operation with transpose FAILS**\n");
    }
  }
  if (!ok) all_ok = 0;

  free(a);
  free(b);
  free(c);
  GA_Destroy(g_a);
  GA_Destroy(g_b);
  NGA_Sprs_array_destroy(s_c);

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
  ok = ok &&  sddmm_test(C_INT);

#if 1
  if (me == 0) {
    printf("\nTesting matrices of type long\n");
  }
  ok = ok && sddmm_test(C_LONG);

  if (me == 0) {
    printf("\nTesting matrices of type long long\n");
  }
  ok = ok && sddmm_test(C_LONGLONG);

  if (me == 0) {
    printf("\nTesting matrices of type float\n");
  }
  ok = ok && sddmm_test(C_FLOAT);

  if (me == 0) {
    printf("\nTesting matrices of type double\n");
  }
  ok = ok && sddmm_test(C_DBL);

  if (me == 0) {
    printf("\nTesting matrices of type single complex\n");
  }
  ok = ok && sddmm_test(C_SCPL);

  if (me == 0) {
    printf("\nTesting matrices of type double complex\n");
  }
  ok = ok && sddmm_test(C_DCPL);
#endif

  if (me == 0) {
    printf("\nSDDMM matrix tests complete\n\n");
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
