/* $Id: sparse.array.c,v 1.80.2.18 2007/12/18 22:22:27 d3g293 Exp $ */
/* 
 * module: sparse.array.c
 * author: Bruce Palmer
 * description: implements a sparse data layout for 2D arrays (matrices)
 * 
 * DISCLAIMER
 *
 * This material was prepared as an account of work sponsored by an
 * agency of the United States Government.  Neither the United States
 * Government nor the United States Department of Energy, nor Battelle,
 * nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
 * COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
 * SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 * INFRINGE PRIVATELY OWNED RIGHTS.
 *
 *
 * ACKNOWLEDGMENT
 *
 * This software and its documentation were produced with United States
 * Government support under Contract Number DE-AC06-76RLO-1830 awarded by
 * the United States Department of Energy.  The United States Government
 * retains a paid-up non-exclusive, irrevocable worldwide license to
 * reproduce, prepare derivative works, perform publicly and display
 * publicly by or for the US Government, including the right to
 * distribute to other US Government contractors.
 */
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#define MAX_INT_VALUE 2147483648

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif
#if HAVE_STDINT_H
#   include <stdint.h>
#endif
#if HAVE_MATH_H
#   include <math.h>
#endif
#if HAVE_ASSERT_H
#   include <assert.h>
#endif
#if HAVE_STDDEF_H
#include <stddef.h>
#endif

#include "global.h"
#include "globalp.h"
#include "base.h"
#include "ga_sparse.array.h"
#include "armci.h"
#include "macdecls.h"
#include "ga-papi.h"
#include "ga-wapi.h"
#include "thread-safe.h"

#define ENABLE_PREALLOC 1

extern int _comex_fence;
extern double _comex_time_send;
extern double _comex_time_recv;
extern double _comex_time_isend;
extern double _comex_time_irecv;
extern double _comex_time_wait;

_sparse_array *SPA;
/**
 * Initial number of values that can be stored on a processor before needing to
 * increase size of local buffers
 */
#define INIT_BUF_SIZE 1024

double time_get;
double time_create;
double time_setdata;
double time_setirreg;
double time_alloc;
double time_gop1;
double time_gop2;
double time_sync;
long   cnt_get;
long   size_get;
/**
 * Internal function to initialize sparse array data structures. This needs to
 * be called somewhere inside ga_initialize
 */
void sai_init_sparse_arrays()
{
  Integer i;
  SPA = (_sparse_array*)malloc(MAX_ARRAYS*sizeof(_sparse_array));
  for (i=0; i<MAX_ARRAYS; i++) {
    SPA[i].active = 0;
    SPA[i].ready = 0;
    SPA[i].nblocks = 0;
    SPA[i].blkidx = NULL;
    SPA[i].blksize = NULL;
    SPA[i].offset = NULL;
    SPA[i].idx = NULL;
    SPA[i].jdx = NULL;
    SPA[i].val = NULL;
    SPA[i].max_nnz = 0;
    SPA[i].g_data = GA_OFFSET-1;
    SPA[i].g_i = GA_OFFSET-1;
    SPA[i].g_j = GA_OFFSET-1;
  }
}

/**
 * Internal function to finalize sparse array data structures. This needs to
 * be called somewhere inside ga_terminate
 */
void sai_terminate_sparse_arrays()
{
  Integer i;
  for (i=0; i<MAX_ARRAYS; i++) {
    Integer ga = SPA[i].g_data + GA_OFFSET;
    if (GA[ga].actv==1) pnga_destroy(SPA[i].g_data);
    ga = SPA[i].g_i + GA_OFFSET;
    if (GA[ga].actv==1) pnga_destroy(SPA[i].g_i);
    ga = SPA[i].g_j + GA_OFFSET;
    if (GA[ga].actv==1) pnga_destroy(SPA[i].g_j);
    if (SPA[i].blkidx) free(SPA[i].blkidx);
    if (SPA[i].blksize) free(SPA[i].blksize);
    if (SPA[i].offset) free(SPA[i].offset);
    if (SPA[i].idx) free(SPA[i].idx);
    if (SPA[i].jdx) free(SPA[i].jdx);
    if (SPA[i].val) free(SPA[i].val);
  }
  free(SPA);
}

/**
 * Create a new sparse array
 * @param idim,jdim I (row) and J (column) dimensions of sparse array
 * @param type type of data stored in sparse array
 * @param size size of indices stored in sparse array
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_create =  pnga_sprs_array_create
#endif
Integer pnga_sprs_array_create(Integer idim, Integer jdim, Integer type, Integer size)
{
  Integer i, hdl, s_a;
  GAvalidtypeM(pnga_type_f2c((int)type));
  if (idim <= 0 || jdim <= 0)
    pnga_error("(ga_sprs_array_create) Invalid array dimenensions",0);
  for (i=0; i<MAX_ARRAYS; i++) {
    if (!SPA[i].active) {
      SPA[i].active = 1;
      SPA[i].idx = (Integer*)malloc(INIT_BUF_SIZE*sizeof(Integer));
      SPA[i].jdx = (Integer*)malloc(INIT_BUF_SIZE*sizeof(Integer));
      SPA[i].type = pnga_type_f2c((int)(type));
      SPA[i].idx_size = size;
      SPA[i].size = GAsizeofM(SPA[i].type);
      SPA[i].val = malloc(INIT_BUF_SIZE*SPA[i].size);
      SPA[i].nval = 0;
      SPA[i].maxval = INIT_BUF_SIZE;
      SPA[i].idim = idim;
      SPA[i].jdim = jdim;
      SPA[i].grp = pnga_pgroup_get_default();
      SPA[i].nprocs = pnga_pgroup_nnodes(SPA[i].grp);
      hdl = i;
      s_a = hdl - GA_OFFSET;
      break;
    }
  }
  return s_a;
}

/**
 * Add an element to a sparse array. Element indices are zero based.
 * @param s_a sparse array handle
 * @param idx, jdx I and J zero based indices of sparse array element
 * @param val sparse array element value
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_add_element =  pnga_sprs_array_add_element
#endif
void pnga_sprs_array_add_element(Integer s_a, Integer idx, Integer jdx, void *val)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer nval = SPA[hdl].nval;
  Integer size = SPA[hdl].size;
  Integer idx_size = sizeof(Integer);
  /* Check to see if array is active and not ready */
  if (!SPA[hdl].active) 
    pnga_error("(ga_sprs_array_add_element) Array not active",hdl);
  if (SPA[hdl].ready) 
    pnga_error("(ga_sprs_array_add_element) Array is already distributed",hdl);
  /* Check to see if existing buffers can hold another value. If not, then
   * create new buffers that are twice as large and copy old data to new
   * buffers*/
  if (nval >= SPA[hdl].maxval) {
    Integer i;
    Integer *tidx;
    Integer *tjdx;
    char *tval;
    char *oval = (char*)SPA[hdl].val;
    Integer me = pnga_pgroup_nodeid(SPA[hdl].grp);

    tidx = (Integer*)malloc(2*SPA[hdl].maxval*sizeof(Integer));
    tjdx = (Integer*)malloc(2*SPA[hdl].maxval*sizeof(Integer));
    tval = (char*)malloc(2*SPA[hdl].maxval*SPA[hdl].size);
    /* copy data in old arrays to new, larger array */
    for (i=0; i<nval; i++) {
      tidx[i] = SPA[hdl].idx[i];
      tjdx[i] = SPA[hdl].jdx[i];
      memcpy((tval+i*size),(oval+i*size),(size_t)size);
    }
    /* get rid of old arrays */
    free(SPA[hdl].idx);
    free(SPA[hdl].jdx);
    free(SPA[hdl].val);
    /* re-assign local buffers */
    SPA[hdl].idx = tidx;
    SPA[hdl].jdx = tjdx;
    SPA[hdl].val = tval;
    /* add new values */
    SPA[hdl].maxval = 2*SPA[hdl].maxval;
  }
  /* add new value to buffers */
  SPA[hdl].idx[nval] = idx;
  SPA[hdl].jdx[nval] = jdx;
  memcpy((char*)SPA[hdl].val+size*nval,(char*)val,(size_t)size);
  SPA[hdl].nval++;
}
/*
void find_lims(Integer dim, Integer proc, Integer nproc, Integer *lo, Integer *hi)
{
  Integer tlo, thi;
  tlo = (dim*proc)/nproc;
  while ((tlo*nproc)/dim < proc) {
    tlo++;
  }
  while ((tlo*nproc)/dim > proc) {
    tlo--;
  }
  if ((tlo*nproc)/dim != proc) {
    tlo++;
  }
  if (proc < nproc-1) {
    thi = (dim*(proc+1))/nproc;
    while ((thi*nproc)/dim < proc+1) {
      thi++;
    }
    while ((thi*nproc)/dim > proc+1) {
      thi--;
    }
    thi--;
  } else {
    thi = dim-1;
  }
  *lo = tlo;
  *hi = thi;
  printf("p[%d] dim: %ld proc: %ld nproc: %ld lo: %ld hi: %ld\n",
      pnga_nodeid(),dim,proc,nproc,*lo,*hi);
}
*/

/**
 * Prepare sparse array for use by distributing values into row blocks based on
 * processor and subdivide each row into column blocks, also based on processor
 * @param s_a sparse array handle
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_assemble =  pnga_sprs_array_assemble
#endif
logical pnga_sprs_array_assemble(Integer s_a)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer lo, hi, ld;
  Integer i,j,ilo,ihi,jlo,jhi;
  int64_t *offset;
  Integer *count;
  Integer *top;
  Integer *list;
  Integer *idx = SPA[hdl].idx;
  Integer *jdx = SPA[hdl].jdx;
  Integer nvals = SPA[hdl].nval;
  Integer nproc = pnga_pgroup_nnodes(SPA[hdl].grp);
  Integer me = pnga_pgroup_nodeid(SPA[hdl].grp);
  Integer elemsize = SPA[hdl].size;
  Integer iproc;
  Integer g_offset;
  Integer g_blk;
  Integer ret = 1;
  int64_t *size;
  Integer *map;
  Integer totalvals;
  Integer one = 1;
  Integer nrows, irow, idim, jdim;
  char *vals;
  Integer ncnt, icnt, jcnt;
  Integer longidx;
  int *isdx;
  int *jsdx;
  int64_t *ildx;
  int64_t *jldx;
  int64_t *row_info;
  int64_t max_nnz;
  Integer *row_nnz;
  Integer nnz;

  /* set variable that distinguishes between long and ints for indices */
  if (SPA[hdl].idx_size == sizeof(int64_t)) {
    longidx = 1;
  } else {
    longidx = 0;
  }
  idim = SPA[hdl].idim;
  jdim = SPA[hdl].jdim;
  /* Create a linked list for values on this process and bin values by which
   * processor owns them */
  count = (Integer*)malloc(nproc*sizeof(Integer));
  top = (Integer*)malloc(nproc*sizeof(Integer));
  list = (Integer*)malloc(nvals*sizeof(Integer));
  for (i=0; i<nproc; i++) {
    count[i] = 0;
    top[i] = -1;
  }
  for (i=0; i<nvals; i++) {
    iproc = (idx[i]*nproc)/idim;
    if (iproc >= nproc) iproc = nproc-1;
    count[iproc]++;
    list[i] = top[iproc];
    top[iproc] = i;
  }

  /* Create global array to store information on sparse blocks */
  {
    Integer dims[3], chunk[3];
    Integer three = 3;
    dims[0] = 7;
    dims[1] = nproc;
    dims[2] = nproc;
    chunk[0] = 7;
    chunk[1] = -1;
    chunk[2] = -1;
    /* g_blk contains information about how data for each sparse
     * block is laid out in g_j and g_data. The last two dimensions
     * describe the location of sparse block in nproc X nproc array
     * of sparse blocks corresponding to original sparse matrix.
     *
     * First dimension contains the following information on each
     * block
     *    ilo: lowest row index of block
     *    ihi: highest row index of block
     *    jlo: lowest column index of block
     *    jhi: highest column index of block
     *    offset: offset in g_j and g_data for column indices and data
     *            values for block
     *    blkend: last index  g_j and g_data for block
     *    iblk: column block index
     * Indices in bounding block are unit based.
     */
    g_blk = pnga_create_handle();
    pnga_set_pgroup(g_blk,SPA[hdl].grp);
    pnga_set_data(g_blk,three,dims,C_LONG);
    pnga_set_chunk(g_blk,chunk);
    if (!pnga_allocate(g_blk)) ret = 0;
    SPA[hdl].g_blk = g_blk;
    row_info = (int64_t*)malloc(7*nproc*sizeof(int64_t));
  }


  /* determine how many values of matrix are stored on this process and what the
   * offset on remote processes is for the data. Create a global array to
   * perform this calculation */
  g_offset = pnga_create_handle();
  pnga_set_data(g_offset,one,&nproc,C_LONG);
  pnga_set_pgroup(g_offset,SPA[hdl].grp);
  if (!pnga_allocate(g_offset)) ret = 0;
  pnga_zero(g_offset);
  offset = (int64_t*)malloc(nproc*sizeof(int64_t));
  for (i=0; i<nproc; i++) offset[i] = 0;
  for (i=0; i<nproc; i++) {
    /* internal indices are unit based, so iproc needs
     * to be incremented by 1 */
    iproc = (i+me)%nproc+1;
    /* C indices are still zero based so need to subtract 1 from iproc */
    if (count[iproc-1] > 0) {
      /* offset locations are not deterministic (ordering cannot be guaranteed),
       * but do guarantee that a space is available to hold data going to each
       * processor */
      offset[iproc-1] = (int64_t)pnga_read_inc(g_offset,&iproc,count[iproc-1]);
    }
  }
  pnga_pgroup_sync(SPA[hdl].grp);
  size = (int64_t*)malloc(nproc*sizeof(int64_t));
  /* internal indices are unit based */
  ilo = 1;
  ihi = nproc;
  pnga_get(g_offset,&ilo,&ihi,size,&nproc);

  /* we now know how much data is on all processors (size) and have an offset on
   * remote processors that we can use to store data from this processor. Start by
   * constructing global arrays to hold data */
  map = (Integer*)malloc(nproc*sizeof(Integer));
  /* internal indices are unit based so start map at 1 */
  map[0] = 1;
  totalvals = size[0];
  for (i=1; i<nproc; i++) {
    map[i] = map[i-1] + (Integer)size[i-1];
    totalvals += (Integer)size[i];
  }
  free(size);
  nnz = totalvals;
  SPA[hdl].g_data = pnga_create_handle();
  pnga_set_data(SPA[hdl].g_data,one,&totalvals,SPA[hdl].type);
  pnga_set_irreg_distr(SPA[hdl].g_data,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_data)) ret = 0;
  SPA[hdl].g_j = pnga_create_handle();
  if (longidx) {
    pnga_set_data(SPA[hdl].g_j,one,&totalvals,C_LONG);
  } else {
    pnga_set_data(SPA[hdl].g_j,one,&totalvals,C_INT);
  }
  pnga_set_irreg_distr(SPA[hdl].g_j,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_j)) ret = 0;
  /* create temporary array using g_i to hold *all* i indices. We will fix it up
   * later to only hold location of first j value
   */
  SPA[hdl].g_i = pnga_create_handle();
  if (longidx) {
    pnga_set_data(SPA[hdl].g_i,one,&totalvals,C_LONG);
  } else {
    pnga_set_data(SPA[hdl].g_i,one,&totalvals,C_INT);
  }
  pnga_set_irreg_distr(SPA[hdl].g_i,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_i)) ret = 0;

  /* fill up global arrays with data */
  for (i=0; i<nproc; i++) {
    iproc = (i+me)%nproc;
    if (count[iproc] > 0) {
      Integer j;
      ncnt = 0;
      char *vbuf = (char*)malloc(count[iproc]*elemsize);
      int64_t *ilbuf;
      int64_t *jlbuf;
      int *ibuf;
      int *jbuf;
      if (longidx) {
        ilbuf = (int64_t*)malloc(count[iproc]*sizeof(int64_t));
        jlbuf = (int64_t*)malloc(count[iproc]*sizeof(int64_t));
      } else {
        ibuf = (int*)malloc(count[iproc]*sizeof(int));
        jbuf = (int*)malloc(count[iproc]*sizeof(int));
      } 
      vals = SPA[hdl].val;
      /* fill up buffers with data going to process iproc */
      j = top[iproc]; 
      if (longidx) {
        while (j >= 0) {
          memcpy(((char*)vbuf+ncnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
          ilbuf[ncnt] = (int64_t)SPA[hdl].idx[j];
          jlbuf[ncnt] = (int64_t)SPA[hdl].jdx[j];
          ncnt++;
          j = list[j];
        }
      } else {
        while (j >= 0) {
          memcpy(((char*)vbuf+ncnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
          ibuf[ncnt] = (int)SPA[hdl].idx[j];
          jbuf[ncnt] = (int)SPA[hdl].jdx[j];
          ncnt++;
          j = list[j];
        }
      }
      /* send data to global arrays */
      lo = map[iproc];
      lo += (Integer)offset[iproc];
      hi = lo+count[iproc]-1;
      if (hi>=lo) {
        pnga_put(SPA[hdl].g_data,&lo,&hi,vbuf,&count[iproc]);
        if (longidx) {
          pnga_put(SPA[hdl].g_i,&lo,&hi,ilbuf,&count[iproc]);
          pnga_put(SPA[hdl].g_j,&lo,&hi,jlbuf,&count[iproc]);
        } else {
          pnga_put(SPA[hdl].g_i,&lo,&hi,ibuf,&count[iproc]);
          pnga_put(SPA[hdl].g_j,&lo,&hi,jbuf,&count[iproc]);
        }
      }
      free(vbuf);
      if (longidx) {
        free(ilbuf);
        free(jlbuf);
      } else {
        free(ibuf);
        free(jbuf);
      }
    }
  }
  pnga_pgroup_sync(SPA[hdl].grp);
  /* Local buffers are no longer needed */
  free(SPA[hdl].idx);
  free(SPA[hdl].jdx);
  free(SPA[hdl].val);
  
  /* All data has been moved so that each process has a row block of the sparse
   * matrix. Now need to organize data within each process into column blocks.
   * Start by binning data by column index */
  pnga_distribution(SPA[hdl].g_data,me,&lo,&hi);
  free(list);
  nvals = hi - lo + 1;
  list = (Integer*)malloc(nvals*sizeof(Integer));
  /* get pointer to list of j indices */
  for (i=0; i<nproc; i++) {
    count[i] = 0;
    top[i] = -1;
    offset[i] = -1;
  }
  if (lo <= hi) {
    if (longidx) {
      pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jldx,&ld);
      for (i=0; i<nvals; i++) {
        iproc = (((Integer)jldx[i])*nproc)/jdim;
        if (iproc >= nproc) iproc = nproc-1;
        count[iproc]++;
        list[i] = top[iproc];
        top[iproc] = i;
      }
    } else {
      pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jsdx,&ld);
      for (i=0; i<nvals; i++) {
        iproc = (((Integer)jsdx[i])*nproc)/jdim;
        if (iproc >= nproc) iproc = nproc-1;
        count[iproc]++;
        list[i] = top[iproc];
        top[iproc] = i;
      }
    }
    pnga_release(SPA[hdl].g_j,&lo,&hi);
  }
  /* find out how many column blocks have data */
  ncnt = 0;
  for (i=0; i<nproc; i++)
    if (count[i] > 0) ncnt++;
  
  SPA[hdl].nblocks = ncnt;
  if (ncnt > 0) SPA[hdl].blkidx = (Integer*)malloc(ncnt*sizeof(Integer));
  if (ncnt > 0) SPA[hdl].offset = (Integer*)malloc(ncnt*sizeof(Integer));
  if (ncnt > 0) SPA[hdl].blksize = (Integer*)malloc(ncnt*sizeof(Integer));
  /* allocate local buffers to sort everything into column blocks */
  SPA[hdl].val = malloc(nvals*SPA[hdl].size);
  SPA[hdl].idx = malloc(nvals*sizeof(Integer));
  SPA[hdl].jdx = malloc(nvals*sizeof(Integer));
  if (lo <= hi) {
    pnga_access_ptr(SPA[hdl].g_data,&lo,&hi,&vals,&ld);
    if (longidx) {
      pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&ildx,&ld);
      pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jldx,&ld);
    } else {
      pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&isdx,&ld);
      pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jsdx,&ld);
    }
  } else {
    ildx = NULL;
    jldx = NULL;
    isdx = NULL;
    jsdx = NULL;
  }
  ncnt = 0;
  icnt = 0;
  for (i=0; i<nproc; i++) {
    Integer j = top[i];
    if (j >= 0) {
      char* vbuf = SPA[hdl].val;
      Integer* ibuf = SPA[hdl].idx;
      Integer* jbuf = SPA[hdl].jdx;
      SPA[hdl].blkidx[ncnt] = i;
      /* copy values from global array to local buffers */
      SPA[hdl].blksize[ncnt] = 0;
      if (longidx) {
        while(j >= 0) {
          if (icnt >= nvals) {
            printf("p[%ld] ICNT: %ld exceeds NVALS: %ld\n",me,icnt,nvals);
          }
          memcpy(((char*)vbuf+icnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
          ibuf[icnt] = (Integer)ildx[j];
          jbuf[icnt] = (Integer)jldx[j];
          j = list[j];
          SPA[hdl].blksize[ncnt]++;
          icnt++;
        }
      } else {
        while(j >= 0) {
          memcpy(((char*)vbuf+icnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
          if (icnt >= nvals) {
            printf("p[%ld] ICNT: %ld exceeds NVALS: %ld\n",me,icnt,nvals);
          }
          ibuf[icnt] = (Integer)isdx[j];
          jbuf[icnt] = (Integer)jsdx[j];
          j = list[j];
          SPA[hdl].blksize[ncnt]++;
          icnt++;
        }
      }
      ncnt++;
    }
  }
  free(count);
  free(top);
  if (lo <= hi) {
    pnga_release(SPA[hdl].g_i,&lo,&hi);
  }
  /* Values have all been sorted into column blocks within the row block. Now
   * need to sort them by row. Start by evaluating lower and upper row indices */
  SPA[hdl].ilo = (SPA[hdl].idim*me)/nproc;
  while ((SPA[hdl].ilo*nproc)/idim < me) {
    SPA[hdl].ilo++;
  }
  while ((SPA[hdl].ilo*nproc)/idim > me) {
    SPA[hdl].ilo--;
  }
  if ((SPA[hdl].ilo*nproc)/idim != me) {
    SPA[hdl].ilo++;
  }
  if (me < nproc-1) {
    SPA[hdl].ihi = (SPA[hdl].idim*(me+1))/nproc;
    while ((SPA[hdl].ihi*nproc)/idim < me+1) {
      SPA[hdl].ihi++;
    }
    while ((SPA[hdl].ihi*nproc)/idim > me+1) {
      SPA[hdl].ihi--;
    }
    SPA[hdl].ihi--;
  } else {
    SPA[hdl].ihi = SPA[hdl].idim-1;
  }

  nrows = SPA[hdl].ihi - SPA[hdl].ilo + 1;
  /* Resize the i-index array to account for row blocks */
  pnga_destroy(SPA[hdl].g_i);
  /* Calculate number of row values that will need to be stored. Add an extra
   * row to account for the total number of rows in the column block */
  for (i=0; i<nproc; i++) {
    offset[i] = 0;
    map[i] = 0;
  }
  offset[me] = (nrows+1)*SPA[hdl].nblocks;
  pnga_pgroup_gop(SPA[hdl].grp,C_LONG,offset,nproc,"+");
  /* Construct new version of g_i */
  map[0] = 1;
  nvals = offset[0];
  for (i=1; i<nproc; i++) {
    map[i] = map[i-1] + offset[i-1];
    nvals += offset[i];
  }
  SPA[hdl].g_i = pnga_create_handle();
  if (longidx) {
    pnga_set_data(SPA[hdl].g_i,one,&nvals,C_LONG);
  } else {
    pnga_set_data(SPA[hdl].g_i,one,&nvals,C_INT);
  }
  pnga_set_irreg_distr(SPA[hdl].g_i,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_i)) ret = 0;
  pnga_distribution(SPA[hdl].g_i,me,&lo,&hi);
  /*
  printf("p[%ld] Distribution on g_i lo: %ld hi: %ld\n",me,lo,hi);
  */
  if (lo <= hi) {
    if (longidx) {
      pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&ildx,&ld);
    } else {
      pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&isdx,&ld);
    }
  } else {
    ildx = NULL;
    isdx = NULL;
  }

  /* Bin up elements by row index for each column block */
  count = (Integer*)malloc(nrows*sizeof(Integer));
  top = (Integer*)malloc(nrows*sizeof(Integer));
  ncnt = 0;
  icnt = 0;
  jcnt = 0;
  row_nnz = (Integer*)malloc(nrows*sizeof(Integer));
  for (i=0; i<nrows; i++) row_nnz[i] = 0;
  for (i=0; i<SPA[hdl].nblocks; i++) {
    char *vptr = vals+ncnt*SPA[hdl].size;
    Integer *ibuf = SPA[hdl].idx + ncnt;
    Integer *jbuf = SPA[hdl].jdx + ncnt;
    for (j=0; j<nrows; j++) {
      top[j] = -1;
      count[j] = 0;
    }
    icnt = 0;
    for (j=0; j<SPA[hdl].blksize[i]; j++) {
      irow = ibuf[j]-SPA[hdl].ilo;
      if (irow < 0 || irow >= nrows) {
        printf("p[%ld] (assemble) irow: %ld out of bounds: %ld\n",me,irow,nrows);
      }
      list[j] = top[irow];
      top[irow] = j;
      count[irow]++;
      icnt++;
    }
    /* copy elements back into  global arrays after sorting by row index.
     * Offset values in g_i are relative to the start of the column block.
     */
    icnt = 0;
    char *vbuf = (char*)SPA[hdl].val+ncnt*SPA[hdl].size;
    if (longidx) {
      for (j=0; j<nrows; j++) {
        irow = top[j];
        (ildx+i*(nrows+1))[j] = (int64_t)icnt;
        while (irow >= 0) {
          jldx[jcnt] = (int64_t)jbuf[irow];
          memcpy((vptr+icnt*elemsize),(vbuf+irow*elemsize),(size_t)elemsize);
          irow = list[irow];
          icnt++;
          jcnt++;
          row_nnz[j]++;
        }
        /*
        if ((ildx+i*(nrows+1))[j] == (int64_t)icnt) {
          printf("p[%ld] ilo: %ld ihi: %ld icol: %ld row: %ld has no elements\n",
              me,SPA[hdl].ilo,SPA[hdl].ihi,SPA[hdl].blkidx[i],j+SPA[hdl].ilo);
        }
        */
      }
      if (icnt > 0) (ildx+i*(nrows+1))[nrows] = (int64_t)icnt; 
      /*
      find_lims(SPA[hdl].jdim,SPA[hdl].blkidx[i],nproc,&jlo,&jhi);
      printf("p[%ld] column offsets for block [%ld,%ld] offset: %ld jlo: %ld jhi: %ld\n",
          me,me,SPA[hdl].blkidx[i],i*(nrows+1),jlo,jhi);
      for (j=0; j<nrows; j++) {
        printf("p[%ld]        row: %ld idx: %ld idx+1: %ld",me,
            SPA[hdl].ilo+j,(ildx+i*(nrows+1))[j],(ildx+i*(nrows+1))[j+1]);
        if ((ildx+i*(nrows+1))[j+1]-(ildx+i*(nrows+1))[j] > 0) {
          printf(" first j: %ld val: %d\n",
              jldx[(ildx+i*(nrows+1))[j]],
              ((int*)vptr)[(ildx+i*(nrows+1))[j]]);
        } else {
          printf("\n");
        }
      }
     */
    } else {
      for (j=0; j<nrows; j++) {
        irow = top[j];
        (isdx+i*(nrows+1))[j] = (int)icnt;
        while (irow >= 0) {
          jsdx[jcnt] = (int)jbuf[irow];
          memcpy((vptr+icnt*elemsize),(vbuf+irow*elemsize),(size_t)elemsize);
          irow = list[irow];
          icnt++;
          jcnt++;
          row_nnz[j]++;
        }
      }
      if (icnt > 0) (isdx+i*(nrows+1))[nrows] = (int)icnt; 
    }
    /* TODO: (maybe) sort each row so that individual elements are arranged in
     * order of increasing j */
    SPA[hdl].offset[i] = ncnt;
    ncnt += SPA[hdl].blksize[i];
  }
  /* Find maximum number of non-zeros per row on this processors */
  max_nnz = 0;
  for (i=0; i<nrows; i++) {
    if (max_nnz < (int64_t)row_nnz[i]) max_nnz = (int64_t)row_nnz[i];
  }
  free(row_nnz);
  pnga_pgroup_gop(SPA[hdl].grp,C_LONG,&max_nnz,1,"max");
  SPA[hdl].max_nnz = (Integer)max_nnz;
  free(SPA[hdl].val);
  free(SPA[hdl].idx);
  free(SPA[hdl].jdx);
  SPA[hdl].val = NULL;
  SPA[hdl].idx = NULL;
  SPA[hdl].jdx = NULL;
  /* set up g_blk */
  for (i=0; i<nproc; i++) {
    Integer jbot, jtop;
    int iblk;
    /* find offset for row block on this processor
     * in g_j (should be the same for g_data */
    pnga_distribution(SPA[hdl].g_j,me,&jbot,&jtop);

    /* calculate column limits for processor i */
    jlo = (SPA[hdl].jdim*i)/nproc;
    while ((jlo*nproc)/jdim < i) {
      jlo++;
    }
    while ((jlo*nproc)/jdim > i) {
      jlo--;
    }
    if (i < nproc-1) {
      jhi = (SPA[hdl].jdim*(i+1))/nproc;
      while ((jhi*nproc)/jdim < i+1) {
        jhi++;
      }
      while ((jhi*nproc)/jdim > i+1) {
        jhi--;
      }
      jhi--;
    } else {
      jhi = SPA[hdl].jdim-1;
    }
    /* set indices to unit based indexing */
    jlo++;
    jhi++;
    /* find index for block in blksize and offset arrays */
    iblk = -1;
    for (j=0; j<SPA[hdl].nblocks; j++) {
      if (SPA[hdl].blkidx[j] == i) {
        iblk = j;
        break;
      }
    }
    if (iblk != -1) {
      /* block has data */
      row_info[i*7  ] = SPA[hdl].ilo+1;
      row_info[i*7+1] = SPA[hdl].ihi+1;
      row_info[i*7+2] = jlo;
      row_info[i*7+3] = jhi;
      row_info[i*7+4] = jbot+SPA[hdl].offset[iblk];
      row_info[i*7+5] = row_info[i*7+4]+SPA[hdl].blksize[iblk]-1;
    } else {
      /* block contains no data */
      row_info[i*7  ] = SPA[hdl].ilo+1;
      row_info[i*7+1] = SPA[hdl].ihi+1;
      row_info[i*7+2] = jlo;
      row_info[i*7+3] = jhi;
      if (i == 0) {
        row_info[4] = 1;
        row_info[5] = 0;
      } else {
        row_info[i*7+4] = row_info[(i-1)*7+4]+1;
        row_info[i*7+5] = row_info[i*7+4]-1;
      }
    }
    row_info[i*7+6] = iblk;
  }
  /* copy data in row info to g_blk */
  {
    Integer tlo[3], thi[3], tld[2];
    tlo[0] = 1;
    tlo[1] = me+1;
    tlo[2] = 1;
    thi[0] = 7;
    thi[1] = me+1;
    thi[2] = nproc;
    tld[0] = 7;
    tld[1] = 1;
    pnga_put(g_blk,tlo,thi,row_info,tld);
    pnga_pgroup_sync(SPA[hdl].grp);
  }

  pnga_release(SPA[hdl].g_data,&lo,&hi);
  pnga_release(SPA[hdl].g_i,&lo,&hi);
  pnga_release(SPA[hdl].g_j,&lo,&hi);

  pnga_destroy(g_offset);

  free(row_info);
  free(count);
  free(top);
  free(list);
  free(offset);
  free(map);

  SPA[hdl].ready = 1;
  /*
  if (me == 0) {
    printf("  Creating sparse matrix with %ld non-zeros\n",(long)nnz);
  }
  */
  return ret;
}

/**
 * Return the range of rows held by processor iproc. Note that this will return
 * valid index ranges for the processor even if none of the rows contain
 * non-zero values. This function is zero-based
 * @param s_a sparse array handle
 * @param iproc process for which index ranges are requested
 * @param lo,hi low and high values of the row indices held by this processor */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_row_distribution =  pnga_sprs_array_row_distribution
#endif
void pnga_sprs_array_row_distribution(Integer s_a, Integer iproc, Integer *lo,
    Integer *hi)
{
  Integer hdl = GA_OFFSET + s_a;
  if (iproc == pnga_pgroup_nodeid(SPA[hdl].grp)) {
    *lo = SPA[hdl].ilo;
    *hi = SPA[hdl].ihi;
  } else {
    Integer nproc = SPA[hdl].nprocs;
    Integer idim = SPA[hdl].idim;
    *lo = (SPA[hdl].idim*iproc)/nproc;
    while (((*lo)*nproc)/idim < iproc) {
      (*lo)++;
    }
    while (((*lo)*nproc)/idim > iproc) {
      (*lo)--;
    }
    if (iproc < nproc-1) {
      (*hi) = (idim*(iproc+1))/nproc;
      while (((*hi)*nproc)/idim < iproc+1) {
        (*hi)++;
      }
      while (((*hi)*nproc)/idim > iproc+1) {
        (*hi)--;
      }
      (*hi)--;
    } else {
      *hi = idim-1;
    }
  }
}

/**
 * Return the range of columns in column block iproc. Note that this will return
 * valid index ranges for the processor even if the column block contains no
 * non-zero values. This function is zero-based
 * @param s_a sparse array handle
 * @param iproc process for which index ranges are requested
 * @param lo,hi low and high values of the row indices held by this processor */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_column_distribution =  pnga_sprs_array_column_distribution
#endif
void pnga_sprs_array_column_distribution(Integer s_a, Integer iproc, Integer *lo,
    Integer *hi)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer nproc = SPA[hdl].nprocs;
  Integer jdim = SPA[hdl].jdim;
  *lo = (SPA[hdl].jdim*iproc)/nproc;
  while (((*lo)*nproc)/jdim < iproc) {
    (*lo)++;
  }
  while (((*lo)*nproc)/jdim > iproc) {
    (*lo)--;
  }
  if (iproc < nproc-1) {
    (*hi) = (jdim*(iproc+1))/nproc;
    while (((*hi)*nproc)/jdim < iproc+1) {
      (*hi)++;
    }
    while (((*hi)*nproc)/jdim > iproc+1) {
      (*hi)--;
    }
    (*hi)--;
  } else {
    *hi = jdim-1;
  }
}

/**
 * Return list of column blocks containing data on this processor
 * @param void *idx indices of colum blocks containing data
 * @param Integer n number of column blocks with data
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_col_block_list =  pnga_sprs_array_col_block_list
#endif
void pnga_sprs_array_col_block_list(Integer s_a, Integer **idx, Integer *n)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer i, index;
  *n = SPA[hdl].nblocks;
  *idx = (Integer*)malloc(SPA[hdl].nblocks*sizeof(Integer));
  for (i=0; i<SPA[hdl].nblocks; i++) {
    (*idx)[i] = SPA[hdl].blkidx[i];
  }
}


/**
 * Return pointers to the compressed sparse row formatted data corresponding to
 * the column block icol. If the column block has no non-zero values, the
 * pointers are returned as null.
 * @param s_a sparse array handle
 * @param icol index indicating column block (corresponds to a processor
 *             location)
 * @param idx location of first index in jdx correspoding to local row index i
 *            idx[i+1]-idx[i]+1 corresponds to the number of non-zero values
 *            in local row i
 * @param jdx column indices of non-zero zero matrix values
 * @param val array of non-zero matrix values
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_access_col_block =  pnga_sprs_array_access_col_block
#endif
void pnga_sprs_array_access_col_block(Integer s_a, Integer icol,
    void *idx, void *jdx, void *val)
{
  Integer hdl = GA_OFFSET + s_a;
  char *lptr;
  Integer i,index;
  Integer longidx;
  if (SPA[hdl].idx_size == sizeof(int64_t)) {
    longidx = 1;
  } else {
    longidx = 0;
  }
  index = -1;
  for (i=0; i<SPA[hdl].nblocks; i++) {
    if (SPA[hdl].blkidx[i] == icol) {
      index = i;
      break;
    }
  }
  if (index == -1) {
    if (longidx) {
      *(int**)idx = NULL;
      *(int**)jdx = NULL;
    } else {
      *(int64_t**)idx = NULL;
      *(int64_t**)jdx = NULL;
    }
    *(char**)val = NULL;
  }  else {
    int *tidx;
    int *tjdx;
    int64_t *tlidx;
    int64_t *tljdx;
    char *lptr;
    Integer lo, hi, ld;
    Integer me = pnga_pgroup_nodeid(SPA[hdl].grp);
    Integer offset = SPA[hdl].offset[index];

    /* access local portions of GAs containing data */
    pnga_distribution(SPA[hdl].g_data,me,&lo,&hi);
    pnga_access_ptr(SPA[hdl].g_data,&lo,&hi,&lptr,&ld);
    pnga_distribution(SPA[hdl].g_i,me,&lo,&hi);
    if (longidx) {
      pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&tlidx,&ld);
    } else {
      pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&tidx,&ld);
    }
    pnga_distribution(SPA[hdl].g_j,me,&lo,&hi);
    if (longidx) {
      pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&tljdx,&ld);
    } else {
      pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&tjdx,&ld);
    }
    /* shift pointers to correct location */
    ld = SPA[hdl].ihi - SPA[hdl].ilo + 2;
    lptr = lptr + offset*SPA[hdl].size;
    if (longidx) {
      tljdx = tljdx + (int64_t)offset;
      tlidx = tlidx + (int64_t)(ld*index);
      *(int64_t**)idx = tlidx;
      *(int64_t**)jdx = tljdx;
    } else {
      tjdx = tjdx + (int)offset;
      tidx = tidx + (int)(ld*index);
      *(int**)idx = tidx;
      *(int**)jdx = tjdx;
    }
    *(char**)val = lptr;
    pnga_release(SPA[hdl].g_data,&lo,&hi);
    pnga_release(SPA[hdl].g_i,&lo,&hi);
    pnga_release(SPA[hdl].g_j,&lo,&hi);
  }
}

/**
 * Function to support fortran interface for access column block functionality.
 * Return indices to the compressed sparse row formatted data corresponding to
 * the column block icol. If the column block has no non-zero values, the
 * function returns zero.
 * @param s_a sparse array handle
 * @param icol index indicating column block (corresponds to a processor
 *             location)
 * @param idx index for starting location of offsets for column indices
 * @param jdx index for starting location of column indices of non-zero zero
 * @param vdx index for starting location of non-zero matrix values
 * @return 0 if no values for this column block
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_access_col_block_idx =  pnga_sprs_array_access_col_block_idx
#endif
int pnga_sprs_array_access_col_block_idx(Integer s_a, Integer icol,
    AccessIndex *idx, AccessIndex *jdx, AccessIndex *vdx)
{
  Integer s_hdl = GA_OFFSET + s_a;
  void *vptr;
  Integer *iptr, *jptr;
  unsigned long lref=0, lptr;
  pnga_sprs_array_access_col_block(s_a, icol, &iptr, &jptr, &vptr);
  /* iproc corresponds to a block with no data */
  if (iptr == NULL && jptr == NULL && vptr == NULL) {
    *idx = 0;
    *jdx = 0;
    *vdx = 0;
    return 0;
  }
  *idx = (AccessIndex) ((Integer*)iptr - INT_MB);
  *jdx = (AccessIndex) ((Integer*)jptr - INT_MB);
  lref = (unsigned long)INT_MB;
  /* if that array data is a fortran integer then it will be set to either the
   * C int or long data type */
  if (SPA[s_hdl].type == C_INT || SPA[s_hdl].type == C_LONG) {
    *vdx = (AccessIndex) ((Integer*)vptr - INT_MB);
  } else if (SPA[s_hdl].type == C_FLOAT) {
    *vdx = (AccessIndex) ((float*)vptr - FLT_MB);
  } else if (SPA[s_hdl].type == C_DBL) {
    *vdx = (AccessIndex) ((double*)vptr - DBL_MB);
  } else if (SPA[s_hdl].type == C_SCPL) {
    *vdx = (AccessIndex) ((SingleComplex*)vptr - SCPL_MB);
  } else if (SPA[s_hdl].type == C_DCPL) {
    *vdx = (AccessIndex) ((DoubleComplex*)vptr - DCPL_MB);
  }

#ifdef BYTE_ADDRESSABLE_MEMORY
    /* check the allignment */
    lptr = (unsigned long)vptr;
    if( lptr%elemsize != lref%elemsize ){
      printf("%d: lptr=%lu(%lu) lref=%lu(%lu)\n",(int)GAme,lptr,lptr%elemsize,
          lref,lref%elemsize);
      pnga_error("sprs_array_access_col_block: MA addressing problem: base address misallignment",
          handle);
    }
#endif

    /* adjust index for Fortran addressing */
    (*idx) ++ ;
    FLUSH_CACHE;
    return 1;
}

/**
 * Multiply a sparse matrix by a sparse vector
 * @param s_a handle for sparse matrix
 * @param g_a handle for vector
 * @param g_v handle for product vector
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_matvec_multiply =  pnga_sprs_array_matvec_multiply
#endif
void pnga_sprs_array_matvec_multiply(Integer s_a, Integer g_a, Integer g_v)
{
  Integer s_hdl = GA_OFFSET + s_a;
  int local_sync_begin,local_sync_end;
  Integer s_grp = SPA[s_hdl].grp;
  Integer me = pnga_pgroup_nodeid(s_grp);
  Integer nproc = pnga_pgroup_nnodes(s_grp);

  Integer ilo, ihi, jlo, jhi, klo, khi;
  void  *vsum, *vptr;
  int idx_size = SPA[s_hdl].idx_size;
  int64_t *ilptr = NULL, *jlptr = NULL;
  int *iptr = NULL, *jptr = NULL;
  Integer i, j, iproc, ncols;
  double one_r = 1.0;
  Integer one = 1;
  Integer adim, vdim, arank, vrank, dims[GA_MAX_DIM];
  Integer atype, vtype;
  Integer zflag;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  /* Check that g_hdl and v_hdl are both vectors and that sizes
   * match */
  if (local_sync_begin) pnga_sync();
  pnga_inquire(g_a, &atype, &arank, dims);
  adim = dims[0];
  pnga_inquire(g_v, &vtype, &vrank, dims);
  vdim = dims[0];
  if (arank != 1) pnga_error("rank of A must be 1 (vector)",arank);
  if (vrank != 1) pnga_error("rank of V must be 1 (vector)",vrank);
  if (adim != SPA[s_hdl].jdim) {
    pnga_error("length of A must equal second dimension of sparse matrix",adim);
  }
  if (vdim != SPA[s_hdl].jdim) {
    pnga_error("length of V must equal second dimension of sparse matrix",vdim);
  }
  if (atype != SPA[s_hdl].type || vtype != SPA[s_hdl].type) {
    pnga_error("Data type of sparse matrix and A and V vectors must match",
        SPA[s_hdl].type);
  }
  /* accumulate operation does not support type C_LONGLONG so fail if this
   * data type encountered */
  if (SPA[s_hdl].type == C_LONGLONG) {
    pnga_error("Data type of sparse matrix and A and V vectors"
        " cannot be of type long long",SPA[s_hdl].type);
  }


#define SPRS_REAL_MULTIPLY_M(_type,_iptr,_jptr)              \
  {                                                          \
    _type *_buf = (_type*)malloc((jhi-jlo+1)*sizeof(_type)); \
    _type *_sum = (_type*)vsum;                              \
    _type *_ptr = (_type*)vptr;                              \
    if (zflag) {                                             \
      for (i=ilo; i<=ihi; i++) {                             \
        _sum[i-ilo] = (_type)0;                              \
      }                                                      \
      zflag = 0;                                             \
    }                                                        \
    klo = jlo+1;                                             \
    khi = jhi+1;                                             \
    pnga_get(g_a,&klo,&khi,_buf,&one);                       \
    for (i=ilo; i<=ihi; i++) {                               \
      ncols = _iptr[i+1-ilo]-_iptr[i-ilo];                   \
      for (j=0; j<ncols; j++) {                              \
        _sum[i-ilo] += _ptr[_iptr[i-ilo]+j]                  \
                     * _buf[_jptr[_iptr[i-ilo]+j]-jlo];      \
      }                                                      \
    }                                                        \
    free(_buf);                                              \
  }
  
#define SPRS_COMPLEX_MULTIPLY_M(_type,_iptr,_jptr)             \
  {                                                            \
    _type *_buf = (_type*)malloc((jhi-jlo+1)*2*sizeof(_type)); \
    _type *_sum = (_type*)vsum;                                \
    _type *_ptr = (_type*)vptr;                                \
    _type rbuf,ibuf,rval,ival;                                 \
    if (zflag) {                                               \
      for (i=ilo; i<=ihi; i++) {                               \
        _sum[2*(i-ilo)] = 0.0;                                 \
        _sum[2*(i-ilo)+1] = 0.0;                               \
      }                                                        \
      zflag = 0;                                               \
    }                                                          \
    klo = jlo+1;                                               \
    khi = jhi+1;                                               \
    pnga_get(g_a,&klo,&khi,_buf,&one);                         \
    for (i=ilo; i<=ihi; i++) {                                 \
      ncols = _iptr[i+1-ilo]-_iptr[i-ilo];                     \
      for (j=0; j<ncols; j++) {                                \
            rbuf = _buf[2*(_jptr[_iptr[i-ilo]+j]-jlo)];        \
            ibuf = _buf[2*(_jptr[_iptr[i-ilo]+j]-jlo)+1];      \
            rval = _buf[2*(_iptr[i-ilo]+j)];                   \
            ival = _buf[2*(_iptr[i-ilo]+j)+1];                 \
            _sum[2*(i-ilo)] = rval*rbuf-ival*ibuf;             \
            _sum[2*(i-ilo)+1] = rval*ibuf+ival*rbuf;           \
      }                                                        \
    }                                                          \
    free(_buf);                                                \
  }
  
  /* Make sure product vector is zero */
  pnga_mask_sync(local_sync_begin,local_sync_end);
  pnga_zero(g_v);
  /* multiply sparse matrix by sparse vector */
  pnga_sprs_array_row_distribution(s_a,me,&ilo,&ihi);
  vsum = (void*)malloc((ihi-ilo+1)*SPA[s_hdl].size);
  zflag = 1;
  for (iproc=0; iproc<nproc; iproc++) {
    pnga_sprs_array_column_distribution(s_a,iproc,&jlo,&jhi);
    if (idx_size == 4) {
      pnga_sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
      if (vptr != NULL) {
        if (SPA[s_hdl].type == C_INT) {
          SPRS_REAL_MULTIPLY_M(int,iptr,jptr);
        } else if (SPA[s_hdl].type == C_LONG) {
          SPRS_REAL_MULTIPLY_M(long,iptr,jptr);
        } else if (SPA[s_hdl].type == C_LONGLONG) {
          SPRS_REAL_MULTIPLY_M(long long,iptr,jptr);
        } else if (SPA[s_hdl].type == C_FLOAT) {
          SPRS_REAL_MULTIPLY_M(float,iptr,jptr);
        } else if (SPA[s_hdl].type == C_DBL) {
          SPRS_REAL_MULTIPLY_M(double,iptr,jptr);
        } else if (SPA[s_hdl].type == C_SCPL) {
          SPRS_COMPLEX_MULTIPLY_M(float,iptr,jptr);
        } else if (SPA[s_hdl].type == C_DCPL) {
          SPRS_COMPLEX_MULTIPLY_M(double,iptr,jptr);
        }
      }
    } else {
      pnga_sprs_array_access_col_block(s_a,iproc,&ilptr,&jlptr,&vptr);
      if (vptr != NULL) {
        if (SPA[s_hdl].type == C_INT) {
          SPRS_REAL_MULTIPLY_M(int,ilptr,jlptr);
        } else if (SPA[s_hdl].type == C_LONG) {
          SPRS_REAL_MULTIPLY_M(long,ilptr,jlptr);
        } else if (SPA[s_hdl].type == C_LONGLONG) {
          SPRS_REAL_MULTIPLY_M(long long,ilptr,jlptr);
        } else if (SPA[s_hdl].type == C_FLOAT) {
          SPRS_REAL_MULTIPLY_M(float,ilptr,jlptr);
        } else if (SPA[s_hdl].type == C_DBL) {
          SPRS_REAL_MULTIPLY_M(double,ilptr,jlptr);
        } else if (SPA[s_hdl].type == C_SCPL) {
          SPRS_COMPLEX_MULTIPLY_M(float,ilptr,jlptr);
        } else if (SPA[s_hdl].type == C_DCPL) {
          SPRS_COMPLEX_MULTIPLY_M(double,ilptr,jlptr);
        }
      }
    }
  }
#undef SPRS_REAL_MULTIPLY_M
#undef SPRS_COMPLEX_MULTIPLY_M

  if (ihi>=ilo) {
    klo = ilo + 1;
    khi = ihi + 1;
    pnga_acc(g_v,&klo,&khi,vsum,&one,&one_r);
  }
  if (local_sync_end)  pnga_sync();
  free(vsum);
}

/**
 * Delete a sparse array and free any resources it may be using
 * @param s_a sparse array handle
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_destroy =  pnga_sprs_array_destroy
#endif
logical pnga_sprs_array_destroy(Integer s_a)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer ret = 1;
  if (SPA[hdl].ready) {
    if (!pnga_destroy(SPA[hdl].g_data)) ret = 0;
    if (!pnga_destroy(SPA[hdl].g_i)) ret = 0;
    if (!pnga_destroy(SPA[hdl].g_j)) ret = 0;
    if (!pnga_destroy(SPA[hdl].g_blk)) ret = 0;
    if (SPA[hdl].blkidx != NULL) {
      free(SPA[hdl].blkidx);
      SPA[hdl].blkidx = NULL;
    }
    if (SPA[hdl].blksize != NULL) {
      free(SPA[hdl].blksize);
      SPA[hdl].blksize = NULL;
    }
    if (SPA[hdl].offset != NULL) {
      free(SPA[hdl].offset);
      SPA[hdl].offset = NULL;
    }
  } else if (SPA[hdl].active) {
    free(SPA[hdl].val);
    SPA[hdl].val = NULL;
    free(SPA[hdl].idx);
    SPA[hdl].idx = NULL;
    free(SPA[hdl].jdx);
    SPA[hdl].jdx = NULL;
  }
  SPA[hdl].active = 0;
  SPA[hdl].ready = 0;
  return ret;
}

/**
 * Print out values of sparse matrix using i,j,val format. Only non-zero
 * value of matrix are printed. This routine gathers all data on one process so
 * it may cause a memory overflow for a large matrix
 * @param s_a sparse array handle
 * @param file name of file to store sparse array
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_export =  pnga_sprs_array_export
#endif
void pnga_sprs_array_export(Integer s_a, const char* file)
{
  Integer hdl = GA_OFFSET + s_a;
  int size  = SPA[hdl].size;
  int type = SPA[hdl].type;
  char frmt[32];
  char *cptr;
  int offset;
  Integer nlen_data;
  Integer nlen_j;
  Integer nlen_i;
  Integer irow;
  int idx_size = SPA[hdl].idx_size;
  void *vptr;
  void *iptr;
  void *jptr;
  Integer lo, hi, ld;
  char op[2];
  FILE *SPRS;
  int *ilo, *ihi, *nblock;
  Integer me = pnga_pgroup_nodeid(SPA[hdl].grp);
  Integer nprocs = pnga_pgroup_nnodes(SPA[hdl].grp);
  Integer iproc, iblock;
  Integer nblocks;
  int *istart;
  int icnt;
  Integer g_blks;
  Integer one = 1;
  Integer total;
  int *blkmap;
  int *blkoffset;
  int *blksize;
  int64_t nnz;

  /* find the total number of nonzero elements on each process */
  nnz = 0;
  for (iproc = 0; iproc<nprocs; iproc++) {
    pnga_distribution(SPA[hdl].g_data,iproc,&lo,&hi);
    if (hi > nnz) nnz = hi;
  }

  /* find low and hi row indices on each processor */
  ilo = (int*)malloc(nprocs*sizeof(int));
  ihi = (int*)malloc(nprocs*sizeof(int));
  nblock = (int*)malloc(nprocs*sizeof(int));
  for (iproc=0; iproc<nprocs; iproc++) {
    ilo[iproc] = 0;
    ihi[iproc] = 0;
    nblock[iproc] = 0;
  }
  ilo[me] = (int)SPA[hdl].ilo;
  ihi[me] = (int)SPA[hdl].ihi;
  nblock[me] = (int)SPA[hdl].nblocks;
  op[0] = '+';
  op[1] = '\0';
  pnga_pgroup_gop(SPA[hdl].grp,C_INT,ilo,nprocs,op);
  pnga_pgroup_gop(SPA[hdl].grp,C_INT,ihi,nprocs,op);
  pnga_pgroup_gop(SPA[hdl].grp,C_INT,nblock,nprocs,op);
  /* construct a map of all blocks */
  nblocks = 0;
  blkoffset = (int*)malloc(nprocs*sizeof(int));
  for (iproc=0; iproc<nprocs; iproc++) nblocks += nblock[iproc];
  blkmap = (int*)malloc(nblocks*sizeof(int));
  blksize = (int*)malloc(nblocks*sizeof(int));
  blkoffset[0] = 0;
  for (iproc=1; iproc<nprocs; iproc++) {
    blkoffset[iproc] = blkoffset[iproc-1]+nblock[iproc-1];
  }
  for (iproc=0; iproc<nblocks; iproc++) {
    blkmap[iproc] = 0;
    blksize[iproc] = 0;
  }
  for (iproc=0; iproc<nblock[me]; iproc++) {
    blkmap[iproc+blkoffset[me]] = SPA[hdl].blkidx[iproc];
    blksize[iproc+blkoffset[me]] = SPA[hdl].blksize[iproc];
  }
  pnga_pgroup_gop(SPA[hdl].grp,C_INT,blkmap,nblocks,op);
  pnga_pgroup_gop(SPA[hdl].grp,C_INT,blksize,nblocks,op);
  /* format print statement */
  if (idx_size == 4) {
    strncpy(frmt,"\%d \%d",5);
    offset = 5;
  } else {
    strncpy(frmt,"\%ld \%ld",7);
    offset = 7;
  }
  cptr = frmt+offset;
  if (type == C_FLOAT || type == C_DBL) {
    strncpy(cptr," \%14.6e",7);
    offset = 7;
  } else if (type == C_INT) {
    strncpy(cptr," \%d",3);
    offset = 3;
  } else if (type == C_LONG) {
    strncpy(cptr," \%ld",4);
    offset = 4;
  } else if (type == C_SCPL || type == C_DCPL) {
    strncpy(cptr," \%14.6e \%14.6e",14);
    offset = 14;
  } 
  cptr += offset;
  cptr[0] = '\n';
  cptr++;
  cptr[0] = '\0';

  if (me == 0) {
    /* open file */
    SPRS = fopen(file,"w");
    fprintf(SPRS,"\%\%MatrixMarket matrix coordinate real general\n");
    fprintf(SPRS,"%ld %ld %ld\n",SPA[hdl].idim, SPA[hdl].jdim, nnz);
    /* Loop over all row blocks */
    for (iproc = 0; iproc<nprocs; iproc++) {
      /* find out how much data is on each process, allocate
       * local buffers to hold it and copy data to local buffer */
      /* Copy data from global arrays to local buffers */
      Integer iilo, iihi, istride, ioffset, itop;
      int ibl;
      ld = 1;
      pnga_distribution(SPA[hdl].g_data, iproc, &lo, &hi);
      nlen_data = hi-lo+1;
      vptr = malloc(nlen_data*size);
      pnga_get(SPA[hdl].g_data,&lo,&hi,vptr,&ld);

      pnga_distribution(SPA[hdl].g_i, iproc, &lo, &hi);
      nlen_i = hi-lo+1;
      iptr = malloc(nlen_i*idx_size);
      pnga_get(SPA[hdl].g_i,&lo,&hi,iptr,&ld);

      pnga_distribution(SPA[hdl].g_j, iproc, &lo, &hi);
      nlen_j = hi-lo+1;
      jptr = malloc(nlen_j*idx_size);
      pnga_get(SPA[hdl].g_j,&lo,&hi,jptr,&ld);

      nblocks = nblock[iproc];
      /* loop over rows. Not the most efficient way of printing out information
       * but makes output slightly neater */
      istride = ihi[iproc]-ilo[iproc]+2;
      for (irow = 0; irow<(ihi[iproc]-ilo[iproc]+1); irow++) {
        int loffset = 0;
        iilo = irow+ilo[iproc];
        for (ibl = 0; ibl<nblocks; ibl++) {
          Integer j, k;
          Integer jlo, jhi, nblk;
          nblk = (Integer)blkmap[blkoffset[iproc]+ibl];
          pnga_sprs_array_column_distribution(s_a,nblk,&jlo,&jhi);
          ioffset = ibl*istride+irow;
          if (idx_size == 4) {
            int *idx = (int*)iptr;
            int *jdx = (int*)jptr+loffset;
            int tlo = idx[ioffset];
            int thi = idx[ioffset+1];
            for (k=tlo; k<thi; k++) {
              if (type == C_FLOAT) {
                float val = ((float*)vptr)[k+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,val);
              } else if (type == C_DBL) {
                double val = ((double*)vptr)[k+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,val);
              } else if (type == C_INT) {
                int val = ((int*)vptr)[k+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,val);
              } else if (type == C_LONG) {
                int64_t val = ((int64_t*)vptr)[k+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,val);
              } else if (type == C_SCPL) {
                float rval = ((float*)vptr)[2*k+loffset];
                float ival = ((float*)vptr)[2*k+1+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,rval,ival);
              } else if (type == C_DCPL) {
                double rval = ((double*)vptr)[2*k+loffset];
                double ival = ((double*)vptr)[2*k+1+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,rval,ival);
              }
            }
          } else {
            int64_t *idx = (int64_t*)iptr;
            int64_t *jdx = (int64_t*)jptr+loffset;
            int64_t tlo = idx[ioffset];
            int64_t thi = idx[ioffset+1];
            for (k=tlo; k<thi; k++) {
              if (type == C_FLOAT) {
                float val = ((float*)vptr)[k+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,val);
              } else if (type == C_DBL) {
                double val = ((double*)vptr)[k+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,val);
              } else if (type == C_INT) {
                int val = ((int*)vptr)[k+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,val);
              } else if (type == C_LONG) {
                int64_t val = ((int64_t*)vptr)[k+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,val);
              } else if (type == C_SCPL) {
                float rval = ((float*)vptr)[2*k+loffset];
                float ival = ((float*)vptr)[2*k+1+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,rval,ival);
              } else if (type == C_DCPL) {
                double rval = ((double*)vptr)[2*k+loffset];
                double ival = ((double*)vptr)[2*k+1+loffset];
                fprintf(SPRS,frmt,iilo+1,jdx[k]+1,rval,ival);
              }
            }
          }
          loffset += blksize[blkoffset[iproc]+ibl];
        }
      }
      /* free local buffers */
      free(vptr);
      free(jptr);
      free(iptr);
    }
    fclose(SPRS);
  }
  free(ilo);
  free(ihi);
  free(nblock);
  free(blkmap);
  free(blksize);
  pnga_pgroup_sync(SPA[hdl].grp);
}

/**
 * Extract diagonal portion of matrix and store it in a distributed vector.
 * Vector has same partition as row partition of matrix
 * @param s_a sparse array handle
 * @param g_d handle of 1D array containint diagonal
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_get_diag =  pnga_sprs_array_get_diag
#endif
void pnga_sprs_array_get_diag(Integer s_a, Integer *g_d)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer *map;
  Integer grp = SPA[hdl].grp;
  Integer me = pnga_pgroup_nodeid(grp);
  Integer nproc = pnga_pgroup_nnodes(grp);
  Integer iproc;
  Integer one = 1;
  Integer lo, hi;
  Integer ilo, ihi, jlo, jhi, ncols;
  int *iptr, *jptr;
  int64_t *ilptr, *jlptr;
  int idx_size = SPA[hdl].idx_size;
  Integer i, j;
  void *diag, *vptr;
  char op[2];
  /* Get row distribution of matrix */
  map = (Integer*)malloc((nproc+1)*sizeof(Integer));
  for (iproc=0; iproc<nproc+1; iproc++) map[iproc] = 0;
  map[me] = SPA[hdl].ilo+1;
  if (me == nproc-1) map[me+1] = SPA[hdl].ihi+1;
  op[0] = '+';
  op[1] = '\0';
  if (sizeof(Integer) == 4) {
    pnga_pgroup_gop(grp, C_INT, map, nproc+1, op);
  } else {
    pnga_pgroup_gop(grp, C_LONG, map, nproc+1, op);
  }
  /* Create array to hold diagonal */
  *g_d = pnga_create_handle();
  pnga_set_data(*g_d,one,&SPA[hdl].idim,SPA[hdl].type);
  pnga_set_irreg_distr(*g_d,map,&nproc);
  pnga_allocate(*g_d);
  /* zero all elements. If diagonal element is not found for a row, then it
   * must be zero */
  pnga_zero(*g_d);
  pnga_distribution(*g_d,me,&lo,&hi);
  pnga_access_ptr(*g_d, &lo, &hi, &diag, &one);
  /* extract diagonal elements */
  pnga_sprs_array_row_distribution(s_a,me,&ilo,&ihi);
  pnga_sprs_array_column_distribution(s_a,me,&jlo,&jhi);
  if (idx_size == 4) {
    pnga_sprs_array_access_col_block(s_a,me,&iptr,&jptr,&vptr);
    for (i=ilo; i<=ihi; i++) {
      ncols = iptr[i+1-ilo]-iptr[i-ilo];
      for (j=0; j<ncols; j++) {
        if (i == jptr[iptr[i-ilo]+j]) {
          if (SPA[hdl].type == C_INT) {
            ((int*)diag)[i-ilo] = ((int*)vptr)[iptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_LONG) {
            ((long*)diag)[i-ilo] = ((long*)vptr)[iptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_LONGLONG) {
            ((long long*)diag)[i-ilo] = ((long long*)vptr)[iptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_FLOAT) {
            ((float*)diag)[i-ilo] = ((float*)vptr)[iptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_DBL) {
            ((double*)diag)[i-ilo] = ((double*)vptr)[iptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_SCPL) {
            ((SingleComplex*)diag)[i-ilo] = ((SingleComplex*)vptr)[iptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_DCPL) {
            ((DoubleComplex*)diag)[i-ilo] = ((DoubleComplex*)vptr)[iptr[i-ilo]+j];
          }
        }
      }
    }
  } else {
    pnga_sprs_array_access_col_block(s_a,me,&ilptr,&jlptr,&vptr);
    for (i=ilo; i<=ihi; i++) {
      ncols = ilptr[i+1-ilo]-ilptr[i-ilo];
      for (j=0; j<ncols; j++) {
        if (i == jlptr[ilptr[i-ilo]+j]) {
          if (SPA[hdl].type == C_INT) {
            ((int*)diag)[i-ilo] = ((int*)vptr)[ilptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_LONG) {
            ((long*)diag)[i-ilo] = ((long*)vptr)[ilptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_LONGLONG) {
            ((long long*)diag)[i-ilo] = ((long long*)vptr)[ilptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_FLOAT) {
            ((float*)diag)[i-ilo] = ((float*)vptr)[ilptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_DBL) {
            ((double*)diag)[i-ilo] = ((double*)vptr)[ilptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_SCPL) {
            ((SingleComplex*)diag)[i-ilo] = ((SingleComplex*)vptr)[ilptr[i-ilo]+j];
          } else if (SPA[hdl].type == C_DCPL) {
            ((DoubleComplex*)diag)[i-ilo] = ((DoubleComplex*)vptr)[ilptr[i-ilo]+j];
          }
        }
      }
    }
  }
  free(map);
  pnga_release_update(*g_d, &lo, &hi);
  pnga_pgroup_sync(grp);
}

/**
 * Left multiply sparse matrix by vector representing a diagonal matrix
 * @param s_a handle of sparse matrix
 * @param g_d handle of vector representing diagonal matrix
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_diag_left_multiply =  pnga_sprs_array_diag_left_multiply
#endif
void pnga_sprs_array_diag_left_multiply(Integer s_a, Integer g_d)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer d_hdl = GA_OFFSET + g_d;
  Integer grp = SPA[hdl].grp;
  Integer me = pnga_pgroup_nodeid(grp);
  Integer nproc = pnga_pgroup_nnodes(grp);
  Integer ilo, ihi, jlo, jhi, klo, khi;
  Integer i, j, iproc, ncols;
  Integer type = SPA[hdl].type;
  int idx_size = SPA[hdl].idx_size;
  int *iptr = NULL, *jptr = NULL;
  int64_t *ilptr = NULL, *jlptr = NULL;
  Integer one = 1;
  void *vbuf;
  void *vptr;

  /* check for basic compatibility */
  if (SPA[hdl].idim != GA[d_hdl].dims[0]) {
    pnga_error("(pnga_sprs_array_diag_left_multiply) dimensions don't match",0);
  }
  if (type != GA[d_hdl].type) {
    pnga_error("(pnga_sprs_array_diag_left_multiply) data types don't match",0);
  }
  if (GA[d_hdl].ndim != 1) {
    pnga_error("(pnga_sprs_array_diag_left_multiply) vector not of dimension 1",0);
  }

#define SPRS_REAL_LEFT_MULTIPLY_M(_type,_iptr)                        \
  {                                                                   \
    _type *_buf = (_type*)vbuf;                                       \
    _type *_ptr = (_type*)vptr;                                       \
    for (i=ilo; i<=ihi; i++) {                                        \
      ncols = _iptr[i+1-ilo]-_iptr[i-ilo];                            \
      for (j=0; j<ncols; j++) {                                       \
        _ptr[_iptr[i-ilo]+j] = _buf[i-ilo]*_ptr[_iptr[i-ilo]+j];      \
      }                                                               \
    }                                                                 \
  }

#define SPRS_COMPLEX_LEFT_MULTIPLY_M(_type,_iptr)                     \
  {                                                                   \
    _type *_buf = (_type*)vbuf;                                       \
    _type *_ptr = (_type*)vptr;                                       \
    _type rbuf, ibuf, rval, ival;                                     \
    for (i=ilo; i<=ihi; i++) {                                        \
      ncols = _iptr[i+1-ilo]-_iptr[i-ilo];                            \
      for (j=0; j<ncols; j++) {                                       \
        rbuf = _buf[2*(i-ilo)];                                       \
        ibuf = _buf[2*(i-ilo)+1];                                     \
        rval = _ptr[2*(_iptr[i-ilo]+j)];                              \
        ival = _ptr[2*(_iptr[i-ilo]+j)+1];                            \
        _ptr[2*(_iptr[i-ilo]+j)] = rbuf*rval-ibuf*ival;               \
        _ptr[2*(_iptr[i-ilo]+j)+1] = rbuf*ival+ibuf*rval;             \
      }                                                               \
    }                                                                 \
  }

  /* get block from diagonal array corresponding to this row block (there is
   * only one) */
  pnga_sprs_array_row_distribution(s_a,me,&ilo,&ihi);
  vbuf = malloc((ihi-ilo+1)*SPA[hdl].size);
  klo = ilo+1;
  khi = ihi+1;
  pnga_get(g_d,&klo,&khi,vbuf,&one);
  /* loop over blocks in sparse array */
  for (iproc=0; iproc<nproc; iproc++) {
    pnga_sprs_array_column_distribution(s_a,iproc,&jlo,&jhi);
    if (idx_size == 4) {
      pnga_sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
      if (vptr != NULL) {
        if (type == C_INT) {
          SPRS_REAL_LEFT_MULTIPLY_M(int,iptr);
        } else if (type == C_LONG) {
          SPRS_REAL_LEFT_MULTIPLY_M(long,iptr);
        } else if (type == C_LONGLONG) {
          SPRS_REAL_LEFT_MULTIPLY_M(long long,iptr);
        } else if (type == C_FLOAT) {
          SPRS_REAL_LEFT_MULTIPLY_M(float,iptr);
        } else if (type == C_DBL) {
          SPRS_REAL_LEFT_MULTIPLY_M(double,iptr);
        } else if (type == C_SCPL) {
          SPRS_COMPLEX_LEFT_MULTIPLY_M(float,iptr);
        } else if (type == C_DCPL) {
          SPRS_COMPLEX_LEFT_MULTIPLY_M(double,iptr);
        }
      }
    } else {
      pnga_sprs_array_access_col_block(s_a,iproc,&ilptr,&jlptr,&vptr);
      if (vptr != NULL) {
        if (type == C_INT) {
          SPRS_REAL_LEFT_MULTIPLY_M(int,ilptr);
        } else if (type == C_LONG) {
          SPRS_REAL_LEFT_MULTIPLY_M(long,ilptr);
        } else if (type == C_LONGLONG) {
          SPRS_REAL_LEFT_MULTIPLY_M(long long,ilptr);
        } else if (type == C_FLOAT) {
          SPRS_REAL_LEFT_MULTIPLY_M(float,ilptr);
        } else if (type == C_DBL) {
          SPRS_REAL_LEFT_MULTIPLY_M(double,ilptr);
        } else if (type == C_SCPL) {
          SPRS_COMPLEX_LEFT_MULTIPLY_M(float,ilptr);
        } else if (type == C_DCPL) {
          SPRS_COMPLEX_LEFT_MULTIPLY_M(double,ilptr);
        }
      }
    }
  }
  free(vbuf);

#undef SPRS_REAL_LEFT_MULTIPLY
#undef SPRS_COMPLEX_LEFT_MULTIPLY

  pnga_pgroup_sync(grp);
}

/**
 * Right multiply sparse matrix by vector representing a diagonal matrix
 * @param s_a handle of sparse matrix
 * @param g_d handle of vector representing diagonal matrix
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_diag_right_multiply =  pnga_sprs_array_diag_right_multiply
#endif
void pnga_sprs_array_diag_right_multiply(Integer s_a, Integer g_d)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer d_hdl = GA_OFFSET + g_d;
  Integer grp = SPA[hdl].grp;
  Integer me = pnga_pgroup_nodeid(grp);
  Integer nproc = pnga_pgroup_nnodes(grp);
  Integer ilo, ihi, jlo, jhi, klo, khi;
  Integer i, j, iproc, ncols;
  Integer type = SPA[hdl].type;
  int idx_size = SPA[hdl].idx_size;
  int *iptr = NULL, *jptr = NULL;
  int64_t *ilptr = NULL, *jlptr = NULL;
  Integer one = 1;
  void *vbuf;
  void *vptr;

  /* check for basic compatibility */
  if (SPA[hdl].idim != GA[d_hdl].dims[0]) {
    pnga_error("(pnga_sprs_array_diag_right_multiply) dimensions don't match",0);
  }
  if (type != GA[d_hdl].type) {
    pnga_error("(pnga_sprs_array_diag_right_multiply) data types don't match",0);
  }
  if (GA[d_hdl].ndim != 1) {
    pnga_error("(pnga_sprs_array_diag_right_multiply) vector not of dimension 1",0);
  }

#define SPRS_REAL_RIGHT_MULTIPLY_M(_type,_iptr,_jptr)                  \
  {                                                                    \
    _type *_buf = (_type*)vbuf;                                        \
    _type *_ptr = (_type*)vptr;                                        \
    for (i=ilo; i<=ihi; i++) {                                         \
      ncols = _iptr[i+1-ilo]-_iptr[i-ilo];                             \
      for (j=0; j<ncols; j++) {                                        \
        _ptr[_iptr[i-ilo]+j] = _ptr[_iptr[i-ilo]+j]                    \
        * _buf[_jptr[_iptr[i-ilo]+j]-jlo];                             \
      }                                                                \
    }                                                                  \
  }

#define SPRS_COMPLEX_RIGHT_MULTIPLY_M(_type,_iptr,_jptr)               \
  {                                                                    \
    _type *_buf = (_type*)vbuf;                                        \
    _type *_ptr = (_type*)vptr;                                        \
    _type rval, ival, rbuf, ibuf;                                      \
    for (i=ilo; i<=ihi; i++) {                                         \
      ncols = _iptr[i+1-ilo]-_iptr[i-ilo];                             \
      for (j=0; j<ncols; j++) {                                        \
        rbuf = _buf[2*(_jptr[_iptr[i-ilo]+j]-jlo)];                    \
        ibuf = _buf[2*(_jptr[_iptr[i-ilo]+j]-jlo)+1];                  \
        rval = _ptr[2*(_iptr[i-ilo]+j)];                               \
        ival = _ptr[2*(_iptr[i-ilo]+j)+1];                             \
        _ptr[2*(_iptr[i-ilo]+j)] = rbuf*rval-ibuf*ival;                \
        _ptr[2*(_iptr[i-ilo]+j)+1] = rbuf*ival+ibuf*rval;              \
      }                                                                \
    }                                                                  \
  }
  /* get block from diagonal array corresponding to this row block (there is
   * only one) */
  pnga_sprs_array_row_distribution(s_a,me,&ilo,&ihi);
  /* loop over blocks in sparse array */
  for (iproc=0; iproc<nproc; iproc++) {
    pnga_sprs_array_column_distribution(s_a,iproc,&jlo,&jhi);
    if (idx_size == 4) {
      pnga_sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
      vbuf = malloc((jhi-jlo+1)*SPA[hdl].size);
      klo = jlo+1;
      khi = jhi+1;
      pnga_get(g_d,&klo,&khi,vbuf,&one);
      if (vptr != NULL) {
        if (type == C_INT) {
          SPRS_REAL_RIGHT_MULTIPLY_M(int,iptr,jptr);
        } else if (type == C_LONG) {
          SPRS_REAL_RIGHT_MULTIPLY_M(long,iptr,jptr);
        } else if (type == C_LONGLONG) {
          SPRS_REAL_RIGHT_MULTIPLY_M(long long,iptr,jptr);
        } else if (type == C_FLOAT) {
          SPRS_REAL_RIGHT_MULTIPLY_M(float,iptr,jptr);
        } else if (type == C_DBL) {
          SPRS_REAL_RIGHT_MULTIPLY_M(double,iptr,jptr);
        } else if (type == C_SCPL) {
          SPRS_COMPLEX_RIGHT_MULTIPLY_M(float,iptr,jptr);
        } else if (type == C_DCPL) {
          SPRS_COMPLEX_RIGHT_MULTIPLY_M(double,iptr,jptr);
        }
      }
    } else {
      pnga_sprs_array_access_col_block(s_a,iproc,&ilptr,&jlptr,&vptr);
      vbuf = malloc((jhi-jlo+1)*SPA[hdl].size);
      klo = jlo+1;
      khi = jhi+1;
      pnga_get(g_d,&klo,&khi,vbuf,&one);
      if (vptr != NULL) {
        if (type == C_INT) {
          SPRS_REAL_RIGHT_MULTIPLY_M(int,ilptr,jlptr);
        } else if (type == C_LONG) {
          SPRS_REAL_RIGHT_MULTIPLY_M(long,ilptr,jlptr);
        } else if (type == C_LONGLONG) {
          SPRS_REAL_RIGHT_MULTIPLY_M(long long,ilptr,jlptr);
        } else if (type == C_FLOAT) {
          SPRS_REAL_RIGHT_MULTIPLY_M(float,ilptr,jlptr);
        } else if (type == C_DBL) {
          SPRS_REAL_RIGHT_MULTIPLY_M(double,ilptr,jlptr);
        } else if (type == C_SCPL) {
          SPRS_COMPLEX_RIGHT_MULTIPLY_M(float,ilptr,jlptr);
        } else if (type == C_DCPL) {
          SPRS_COMPLEX_RIGHT_MULTIPLY_M(double,ilptr,jlptr);
        }
      }
    }
    free(vbuf);
  }

#undef SPRS_REAL_RIGHT_MULTIPLY_M
#undef SPRS_COMPLEX_RIGHT_MULTIPLY_M

  pnga_pgroup_sync(grp);
}

/**
 * Shift diagonal values of sparse matrix by adding a constant to all diagonal
 * values
 * @param s_a handle of sparse matrix
 * @param shift pointer to shift value
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_shift_diag =  pnga_sprs_array_shift_diag
#endif
void pnga_sprs_array_shift_diag(Integer s_a, void *shift)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer grp = SPA[hdl].grp;
  Integer me = pnga_pgroup_nodeid(grp);
  Integer nproc = pnga_pgroup_nnodes(grp);
  Integer ilo, ihi;
  Integer i, j, iproc, ncols;
  Integer type = SPA[hdl].type;
  int idx_size = SPA[hdl].idx_size;
  int *iptr, *jptr;
  int64_t *ilptr, *jlptr;
  void *vptr;

  /* get block from diagonal array corresponding to this row block (there is
   * only one) */
  pnga_sprs_array_row_distribution(s_a,me,&ilo,&ihi);

#define SPRS_REAL_SHIFT_DIAG_M(_type,_iptr,_jptr)             \
  {                                                           \
    _type _shift = *((_type*)shift);                          \
    _type *_vptr = (_type*)vptr;                              \
    for (i=ilo; i<=ihi; i++) {                                \
      ncols = _iptr[i+1-ilo]-_iptr[i-ilo];                    \
      for (j=0; j<ncols; j++) {                               \
        if (i == _jptr[_iptr[i-ilo]+j]) {                     \
          _vptr[_iptr[i-ilo]+j] += _shift;                    \
        }                                                     \
      }                                                       \
    }                                                         \
  }                

#define SPRS_COMPLEX_SHIFT_DIAG_M(_type,_iptr,_jptr)          \
  {                                                           \
    _type _rshift = ((_type*)shift)[0];                       \
    _type _ishift = ((_type*)shift)[1];                       \
    _type *_vptr = (_type*)vptr;                              \
    for (i=ilo; i<=ihi; i++) {                                \
      ncols = _iptr[i+1-ilo]-_iptr[i-ilo];                    \
      for (j=0; j<ncols; j++) {                               \
        if (i == _jptr[_iptr[i-ilo]+j]) {                     \
          _vptr[2*(_iptr[i-ilo]+j)] += _rshift;               \
          _vptr[2*(_iptr[i-ilo]+j)+1] += _ishift;             \
        }                                                     \
      }                                                       \
    }                                                         \
  }                

  /* loop over blocks in sparse array */
  for (iproc=0; iproc<nproc; iproc++) {
    if (idx_size == 4) {
      pnga_sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
      if (vptr != NULL) {
        if (type == C_INT) {
          SPRS_REAL_SHIFT_DIAG_M(int,iptr,jptr);
        } else if (type == C_LONG) {
          SPRS_REAL_SHIFT_DIAG_M(long,iptr,jptr);
        } else if (type == C_LONGLONG) {
          SPRS_REAL_SHIFT_DIAG_M(long long,iptr,jptr);
        } else if (type == C_FLOAT) {
          SPRS_REAL_SHIFT_DIAG_M(float,iptr,jptr);
        } else if (type == C_DBL) {
          SPRS_REAL_SHIFT_DIAG_M(double,iptr,jptr);
        } else if (type == C_SCPL) {
          SPRS_COMPLEX_SHIFT_DIAG_M(float,iptr,jptr);
        } else if (type == C_DCPL) {
          SPRS_COMPLEX_SHIFT_DIAG_M(double,iptr,jptr);
        }
      }
    } else {
      pnga_sprs_array_access_col_block(s_a,iproc,&ilptr,&jlptr,&vptr);
      if (vptr != NULL) {
        if (type == C_INT) {
          SPRS_REAL_SHIFT_DIAG_M(int,ilptr,jlptr);
        } else if (type == C_LONG) {
          SPRS_REAL_SHIFT_DIAG_M(long,ilptr,jlptr);
        } else if (type == C_LONGLONG) {
          SPRS_REAL_SHIFT_DIAG_M(long long,ilptr,jlptr);
        } else if (type == C_FLOAT) {
          SPRS_REAL_SHIFT_DIAG_M(float,ilptr,jlptr);
        } else if (type == C_DBL) {
          SPRS_REAL_SHIFT_DIAG_M(double,ilptr,jlptr);
        } else if (type == C_SCPL) {
          SPRS_COMPLEX_SHIFT_DIAG_M(float,ilptr,jlptr);
        } else if (type == C_DCPL) {
          SPRS_COMPLEX_SHIFT_DIAG_M(double,ilptr,jlptr);
        }
      }
    }
  }

#undef SPRS_REAL_SHIFT_DIAG_M
#undef SPRS_COMPLEX_SHIFT_DIAG_M

  pnga_pgroup_sync(grp);
}

/**
 * Duplicate an existing sparse array
 * @param s_a sparse array that is to be duplicated
 * @return handle of duplicate array
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_duplicate =  pnga_sprs_array_duplicate
#endif
Integer pnga_sprs_array_duplicate(Integer s_a)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer new_hdl;
  Integer grp = SPA[hdl].grp;
  Integer me = pnga_pgroup_nodeid(grp);
  Integer nproc = pnga_pgroup_nnodes(grp);
  Integer s_dup;
  Integer lo, hi;
  Integer i;
  char p_trans[2];

  /* create new array with same properties as old array */
  s_dup = pnga_sprs_array_create(SPA[hdl].idim,SPA[hdl].jdim,SPA[hdl].type,
      SPA[hdl].idx_size);
  /* find handle for new array and duplicate internal GAs */
  new_hdl = GA_OFFSET + s_dup;
  if (!pnga_duplicate(SPA[hdl].g_data,&SPA[new_hdl].g_data,"sparse_data_copy")) {
    pnga_error("(pnga_sprs_array_duplicate) Could not duplicate g_data",0);
  }
  if (!pnga_duplicate(SPA[hdl].g_i,&SPA[new_hdl].g_i,"sparse_i_index_copy")) {
    pnga_error("(pnga_sprs_array_duplicate) Could not duplicate g_i",0);
  }
  if (!pnga_duplicate(SPA[hdl].g_j,&SPA[new_hdl].g_j,"sparse_j_index_copy")) {
    pnga_error("(pnga_sprs_array_duplicate) Could not duplicate g_j",0);
  }
  if (!pnga_duplicate(SPA[hdl].g_blk,&SPA[new_hdl].g_blk,"sparse_g_block_copy")) {
    pnga_error("(pnga_sprs_array_duplicate) Could not duplicate g_blk",0);
  }
  /* Copy data from old array to new array */
  p_trans[0]='N';
  p_trans[1]='\0';
  pnga_distribution(SPA[new_hdl].g_i,me,&lo,&hi);
  pnga_copy_patch(p_trans,SPA[hdl].g_i,&lo,&hi,SPA[new_hdl].g_i,&lo,&hi);
  pnga_distribution(SPA[new_hdl].g_j,me,&lo,&hi);
  pnga_copy_patch(p_trans,SPA[hdl].g_j,&lo,&hi,SPA[new_hdl].g_j,&lo,&hi);
  pnga_distribution(SPA[new_hdl].g_data,me,&lo,&hi);
  pnga_copy_patch(p_trans,SPA[hdl].g_data,&lo,&hi,SPA[new_hdl].g_data,&lo,&hi);
  {
    Integer tlo[3],thi[3];
    pnga_distribution(SPA[new_hdl].g_blk,me,tlo,thi);
    pnga_copy_patch(p_trans,SPA[hdl].g_blk,tlo,thi,SPA[new_hdl].g_blk,tlo,thi);
  }
  /* copy remaining data structures */
  SPA[new_hdl].ilo = SPA[hdl].ilo;
  SPA[new_hdl].ihi = SPA[hdl].ihi;
  SPA[new_hdl].nblocks = SPA[hdl].nblocks;
  SPA[new_hdl].nval = SPA[hdl].nval;
  SPA[new_hdl].maxval = SPA[hdl].maxval;
  SPA[new_hdl].max_nnz = SPA[hdl].max_nnz;
  SPA[new_hdl].ready = SPA[hdl].ready;
  SPA[new_hdl].grp = SPA[hdl].grp;
  SPA[new_hdl].val = NULL;
  SPA[new_hdl].idx = NULL;
  SPA[new_hdl].jdx = NULL;
  if (SPA[new_hdl].nblocks > 0) {
    SPA[new_hdl].blkidx
      = (Integer*)malloc(SPA[new_hdl].nblocks*sizeof(Integer));
    SPA[new_hdl].blksize
      = (Integer*)malloc(SPA[new_hdl].nblocks*sizeof(Integer));
    SPA[new_hdl].offset
      = (Integer*)malloc(SPA[new_hdl].nblocks*sizeof(Integer));
  }
  for (i=0; i<SPA[new_hdl].nblocks; i++) {
    SPA[new_hdl].blkidx[i] = SPA[hdl].blkidx[i];
    SPA[new_hdl].blksize[i] = SPA[hdl].blksize[i];
    SPA[new_hdl].offset[i] = SPA[hdl].offset[i];
  }
  return s_dup;
}

/**
 * Retrieve a sparse block as a CSR data structure
 * @param s_a sparse array from which data is retrieved
 * @param irow, icol row and column indices of the sparse block.
 *                   These indices range in value from 0 to nproc-1,
 *                   where nproc is the number of processors in
 *                   the process group hosting the array.
 * @param idx array holding offsets into jdx and data arrays for each row
 * @param jdx array holding column indices of non-zero elements
 * @param data array holding values of non-zero elements
 * @param ilo, ihi, jlo, jhi bounding indices of block in matrix
 * @param max_nnz maximum number of non-zero per row in the block
 * @return false if block has no data
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_get_block =  pnga_sprs_array_get_block
#endif
logical pnga_sprs_array_get_block(Integer s_a, Integer irow, Integer icol,
   void **idx, void **jdx, void **data, Integer *ilo, Integer *ihi,
    Integer *jlo, Integer *jhi)
{
  Integer hdl = GA_OFFSET + s_a;
  int64_t params[7];
  Integer lo[3], hi[3];
  Integer ld[2];
  int64_t len;
  int longidx;
  int i, index;
  Integer offset, dummy;
  logical ret = 1;
  double tbeg;

  /* retrieve location of block in distributed arrays */
  lo[0] = 1;
  lo[1] = irow+1;
  lo[2] = icol+1;
  hi[0] = 7;
  hi[1] = irow+1;
  hi[2] = icol+1;
  ld[0] = 7;
  ld[1] = 1;
  tbeg = pnga_wtime();
  pnga_get(SPA[hdl].g_blk,lo,hi,params,ld);
  time_get += pnga_wtime()-tbeg;
  cnt_get++;
  size_get += 7*sizeof(long);

  *ilo = params[0];
  *ihi = params[1];
  *jlo = params[2];
  *jhi = params[3];
  /* allocate arrays to hold block */
  len = params[5]-params[4]+1;
  index = params[6];
  /*
  printf("p[%ld] Getting block [%d,%d] ilo: %ld ihi: %ld jlo: %ld jhi: %ld len: %ld clo: %ld chi: %ld nodata: %d\n",
      pnga_nodeid(),irow,icol,*ilo,*ihi,*jlo,*jhi,len,params[4],params[5],params[6]);
      */
  if (len > 0) {
    *idx = (Integer*)malloc((*ihi-*ilo+2)*sizeof(Integer));
    *jdx = (Integer*)malloc(len*sizeof(Integer));
    *data = malloc(len*SPA[hdl].size);
    lo[0] = params[4];
    hi[0] = params[5];
    ld[0] = 1;
    tbeg = pnga_wtime();
    pnga_get(SPA[hdl].g_j,lo,hi,*jdx,ld);
    pnga_get(SPA[hdl].g_data,lo,hi,*data,ld);
    time_get += pnga_wtime()-tbeg;
    cnt_get += 2;
    size_get += (hi[0]-lo[0]+1)*SPA[hdl].idx_size;
    size_get += (hi[0]-lo[0]+1)*SPA[hdl].size;

    if (index == -1) {
      printf("p[%d] irow: %d icol: %d iblk: %d ilo: %d ihi: %d"
          " jlo: %d jhi: %d len: %d\n",
          pnga_nodeid(),irow,icol,index,*ilo,*ihi,*jlo,*jhi,len);
      pnga_error("sprs_array_get_block no block found",index);
    }
    pnga_distribution(SPA[hdl].g_i,irow,&offset,&dummy);
    lo[0] = offset+index*(*ihi-*ilo+2);
    hi[0] = offset+(index+1)*(*ihi-*ilo+2)-1;
    tbeg = pnga_wtime();
    pnga_get(SPA[hdl].g_i,lo,hi,*idx,ld);
    time_get += pnga_wtime()-tbeg;
    cnt_get++;
    size_get += (hi[0]-lo[0]+1)*SPA[hdl].idx_size;
    /*
    printf("p[%ld] row offsets in get_block for block: %ld icol %ld\n",
        pnga_nodeid(),index,SPA[hdl].blkidx[index]);
    for (i=lo[0]; i<hi[0]; i++) {
      int64_t *ptr = (int64_t*)*idx;
      printf("p[%ld]     row: %ld idx: %ld idx+1: %ld",pnga_nodeid(),
          i,ptr[i-lo[0]],ptr[i+1-lo[0]]);
      if (ptr[i+1-lo[0]]-ptr[i-lo[0]] > 0) {
        printf(" first j: %ld val: %d\n",
            ((int64_t*)(*jdx))[ptr[i-lo[0]]],
            ((int*)(*data))[ptr[i-lo[0]]]);
      } else {
        printf("\n");
      }
    }
    */
  } else {
    /* block has no data */
    *idx == NULL;
    *jdx == NULL;
    *data == NULL;
    ret = 0;
  }
  return ret;
}
/**
 * Utility function to resize map data structure if number of elements
 * exceeds available space. If number of elements exceeds available space
 * double size of buffers and copy old buffers to new buffers
 * @param top array of starting elements for each bin
 * @param list linked list of elements
 * @param idx row index of each element
 * @param jdx column index of each element
 * @param data value of each element
 * @param idim total number of rows in sparse array
 * @param jdim total number of columns in sparse array
 * @param elemsize size of individual data elements
 * @param bufsize current size of map arrays
 * @param ncnt total number of elements stored in map arrays
 */
void update_map(Integer **top, Integer **list, Integer **idx, Integer **jdx,
    void **data, Integer idim, Integer jdim, Integer elemsize,
    Integer *bufsize, Integer *ncnt)
{
  Integer *ttop;
  Integer *tlist;
  Integer *tidx, *tjdx;
  Integer lcnt;
  Integer newsize;
  void *tdata;
  char *nptr;
  char *optr;
  Integer ii,jj;
  newsize = 2*(*bufsize);
  ttop = (Integer*)malloc(newsize*sizeof(Integer));
  tlist = (Integer*)malloc(newsize*sizeof(Integer));
  tdata = malloc(newsize*elemsize);
  lcnt = 0;
  for (ii=0; ii<newsize; ii++) tlist[ii] = -1;
  tidx = (Integer*)malloc(newsize*sizeof(Integer));
  tjdx = (Integer*)malloc(newsize*sizeof(Integer));
  for (ii=0; ii<*bufsize; ii++) {
    jj = (*top)[ii];
    while (jj >= 0) {
      Integer itmp = (*idx)[jj];
      Integer jtmp = (*jdx)[jj];
      Integer index = (jdim*itmp + jtmp)%newsize;
      tlist[lcnt] = ttop[index];
      ttop[index] = lcnt;
      tidx[lcnt] = itmp; 
      tjdx[lcnt] = jtmp; 
      nptr = (char*)tdata + lcnt*elemsize;
      optr = (char*)(*data) + jj*elemsize;
      memcpy(nptr,optr,elemsize);
      jj = (*list)[jj];
      lcnt++;
    }
  }
  free(*top);
  free(*list);
  free(*idx);
  free(*jdx);
  free(*data);
  *bufsize = newsize;
  *top = ttop;
  *list = tlist;
  *idx = tidx;
  *jdx = tjdx;
  *data = tdata;
  *ncnt = lcnt;
}

#if ENABLE_PREALLOC
/**
 * Macros for sparse block matrix-matrix multiply. Note that bounds
 * ilo_a, ihi_a, jlo_b, jhi_b are unit based, so any index that has
 * these values subtracted from it must also be unit based.
 */
#define SPRS_REAL_MATMAT_MULTIPLY_M(_type,_idxa,_jdxa,_idxb,_jdxb) \
{                                                                  \
  for (i=ilo_a; i<=ihi_a; i++) {                                   \
    Integer kcols = _idxa[i+1-ilo_a]-_idxa[i-ilo_a];               \
    for (k=0; k<kcols; k++) {                                      \
      Integer kdx = _jdxa[_idxa[i-ilo_a]+k]+1;                     \
      Integer jcols = _idxb[kdx+1-ilo_b]-_idxb[kdx-ilo_b];         \
      _type val_a = ((_type*)data_a)[_idxa[i-ilo_a]+k];            \
      for (j=0; j<jcols; j++) {                                    \
        Integer jj = _jdxb[_idxb[kdx-ilo_b]+j]+1;                  \
        _type val_b = ((_type*)data_b)[_idxb[kdx-ilo_b]+j];        \
        /* Check to see if c_ij already exists */                  \
        Integer ldx = ((i-1)*jdim+jj-1)%bufsize;                   \
        ldx = top[ldx];                                            \
        while(ldx >= 0) {                                          \
          if (i == idx[ldx] && jj == jdx[ldx]) break;              \
          ldx = list[ldx];                                         \
        }                                                          \
        if (ldx >= 0) {                                            \
          /* add product to existing value*/                       \
          ((_type*)data)[ldx] += val_a*val_b;                      \
        } else {                                                   \
          /* add new value to list */                              \
          ((_type*)data)[lcnt] = val_a*val_b;                      \
          idx[lcnt] = i;                                           \
          jdx[lcnt] = jj;                                          \
          ldx = ((i-1)*jdim+jj-1)%bufsize;                         \
          list[lcnt] = top[ldx];                                   \
          top[ldx] = lcnt;                                         \
          lcnt++;                                                  \
        }                                                          \
      }                                                            \
    }                                                              \
  }                                                                \
}

#define SPRS_COMPLEX_MATMAT_MULTIPLY_M(_type,_idxa,_jdxa,_idxb,_jdxb) \
{                                                                     \
  for (i=ilo_a; i<=ihi_a; i++) {                                      \
    Integer kcols = _idxa[i+1-ilo_a]-_idxa[i-ilo_a];                  \
    for (k=0; k<kcols; k++) {                                         \
      Integer kdx = _jdxa[_idxa[i-ilo_a]+k]+1;                        \
      Integer jcols = _idxb[kdx+1-ilo_b]-_idxb[kdx-ilo_b];            \
      /*_type rval_a = ((_type*)data_a)[2*(kdx-jlo_a)];    */         \
      /*_type ival_a = ((_type*)data_a)[2*(kdx-jlo_a)+1];  */         \
      _type rval_a = ((_type*)data_a)[2*(idx_a[i-ilo_a]+k)];          \
      _type ival_a = ((_type*)data_a)[2*(idx_a[i-ilo_a]+k)+1];        \
      for (j=0; j<jcols; j++) {                                       \
        Integer jj = _jdxb[_idxb[kdx-ilo_b]+j]+1;                     \
        /*_type rval_b = ((_type*)data_b)[2*(jj-jlo_b)];    */        \
        /*_type ival_b = ((_type*)data_b)[2*(jj-jlo_b)+1];  */        \
        _type rval_b = ((_type*)data_b)[2*(idx_b[kdx-ilo_b]+j)];      \
        _type ival_b = ((_type*)data_b)[2*(idx_b[kdx-ilo_b]+j)+1];    \
        /* Check to see if c_ij already exists */                     \
        Integer ldx = ((i-1)*jdim+jj-1)%bufsize;                      \
        ldx = top[ldx];                                               \
        while(ldx >= 0) {                                             \
          if (i == idx[ldx] && jj == jdx[ldx]) break;                 \
          ldx = list[ldx];                                            \
        }                                                             \
        if (ldx >= 0) {                                               \
          /* add product to existing value*/                          \
          ((_type*)data)[2*ldx] += rval_a*rval_b-ival_a*ival_b;       \
          ((_type*)data)[2*ldx+1] += rval_a*ival_b+ival_a*rval_b;     \
        } else {                                                      \
          /* add new value to list */                                 \
          ((_type*)data)[2*lcnt] = rval_a*rval_b-ival_a*ival_b;       \
          ((_type*)data)[2*lcnt+1] = rval_a*ival_b+ival_a*rval_b;     \
          idx[lcnt] = i;                                              \
          jdx[lcnt] = jj;                                             \
          ldx = ((i-1)*jdim+jj-1)%bufsize;                            \
          list[lcnt] = top[ldx];                                      \
          top[ldx] = lcnt;                                            \
          lcnt++;                                                     \
        }                                                             \
      }                                                               \
    }                                                                 \
  }                                                                   \
}
#else
/**
 * Macros for sparse block matrix-matrix multiply. Note that bounds
 * ilo_a, ihi_a, jlo_b, jhi_b are unit based, so any index that has
 * these values subtracted from it must also be unit based.
 */
#define SPRS_REAL_MATMAT_MULTIPLY_M(_type,_idxa,_jdxa,_idxb,_jdxb) \
{                                                                  \
  for (i=ilo_a; i<=ihi_a; i++) {                                   \
    Integer kcols = _idxa[i+1-ilo_a]-_idxa[i-ilo_a];               \
    for (k=0; k<kcols; k++) {                                      \
      Integer kdx = _jdxa[_idxa[i-ilo_a]+k]+1;                     \
      Integer jcols = _idxb[kdx+1-ilo_b]-_idxb[kdx-ilo_b];         \
      /*_type val_a = ((_type*)data_a)[kdx-jlo_a]; */              \
      _type val_a = ((_type*)data_a)[_idxa[i-ilo_a]+k];            \
      for (j=0; j<jcols; j++) {                                    \
        Integer jj = _jdxb[_idxb[kdx-ilo_b]+j]+1;                  \
        /* _type val_b = ((_type*)data_b)[jj-jlo_b]; */            \
        _type val_b = ((_type*)data_b)[_idxb[kdx-ilo_b]+j];        \
        /* Check to see if c_ij already exists */                  \
        Integer ldx = ((i-1)*jdim+jj-1)%bufsize;                   \
        ldx = top[ldx];                                            \
        while(ldx >= 0) {                                          \
          if (i == idx[ldx] && jj == jdx[ldx]) break;              \
          ldx = list[ldx];                                         \
        }                                                          \
        if (ldx >= 0) {                                            \
          /* add product to existing value*/                       \
          ((_type*)data)[ldx] += val_a*val_b;                      \
        } else {                                                   \
          /* add new value to list */                              \
          if (lcnt == bufsize)                                     \
            update_map(&top, &list, &idx, &jdx, &data, idim,       \
                jdim, elemsize, &bufsize, &lcnt);                  \
          ((_type*)data)[lcnt] = val_a*val_b;                      \
          idx[lcnt] = i;                                           \
          jdx[lcnt] = jj;                                          \
          ldx = ((i-1)*jdim+jj-1)%bufsize;                         \
          list[lcnt] = top[ldx];                                   \
          top[ldx] = lcnt;                                         \
          lcnt++;                                                  \
        }                                                          \
      }                                                            \
    }                                                              \
  }                                                                \
}

#define SPRS_COMPLEX_MATMAT_MULTIPLY_M(_type,_idxa,_jdxa,_idxb,_jdxb) \
{                                                                     \
  for (i=ilo_a; i<=ihi_a; i++) {                                      \
    Integer kcols = _idxa[i+1-ilo_a]-_idxa[i-ilo_a];                  \
    for (k=0; k<kcols; k++) {                                         \
      Integer kdx = _jdxa[_idxa[i-ilo_a]+k]+1;                        \
      Integer jcols = _idxb[kdx+1-ilo_b]-_idxb[kdx-ilo_b];            \
      /*_type rval_a = ((_type*)data_a)[2*(kdx-jlo_a)];    */         \
      /*_type ival_a = ((_type*)data_a)[2*(kdx-jlo_a)+1];  */         \
      _type rval_a = ((_type*)data_a)[2*(idx_a[i-ilo_a]+k)];          \
      _type ival_a = ((_type*)data_a)[2*(idx_a[i-ilo_a]+k)+1];        \
      for (j=0; j<jcols; j++) {                                       \
        Integer jj = _jdxb[_idxb[kdx-ilo_b]+j]+1;                     \
        /*_type rval_b = ((_type*)data_b)[2*(jj-jlo_b)];    */        \
        /*_type ival_b = ((_type*)data_b)[2*(jj-jlo_b)+1];  */        \
        _type rval_b = ((_type*)data_b)[2*(idx_b[kdx-ilo_b]+j)];      \
        _type ival_b = ((_type*)data_b)[2*(idx_b[kdx-ilo_b]+j)+1];    \
        /* Check to see if c_ij already exists */                     \
        Integer ldx = ((i-1)*jdim+jj-1)%bufsize;                      \
        ldx = top[ldx];                                               \
        while(ldx >= 0) {                                             \
          if (i == idx[ldx] && jj == jdx[ldx]) break;                 \
          ldx = list[ldx];                                            \
        }                                                             \
        if (ldx >= 0) {                                               \
          /* add product to existing value*/                          \
          ((_type*)data)[2*ldx] += rval_a*rval_b-ival_a*ival_b;       \
          ((_type*)data)[2*ldx+1] += rval_a*ival_b+ival_a*rval_b;     \
        } else {                                                      \
          /* add new value to list */                                 \
          if (lcnt == bufsize) update_map(&top, &list, &idx, &jdx,    \
              &data, idim, jdim, elemsize, &bufsize, &lcnt);          \
          ((_type*)data)[2*lcnt] = rval_a*rval_b-ival_a*ival_b;       \
          ((_type*)data)[2*lcnt+1] = rval_a*ival_b+ival_a*rval_b;     \
          idx[lcnt] = i;                                              \
          jdx[lcnt] = jj;                                             \
          ldx = ((i-1)*jdim+jj-1)%bufsize;                            \
          list[lcnt] = top[ldx];                                      \
          top[ldx] = lcnt;                                            \
          lcnt++;                                                     \
        }                                                             \
      }                                                               \
    }                                                                 \
  }                                                                   \
}
#endif

/**
 * Multiply sparse matrices A and B to get sparse matrix C
 * C = A.B
 * @param s_a sparse array A
 * @param s_b sparse array B
 * @return handle of sparse array C
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_matmat_multiply =  pnga_sprs_array_matmat_multiply
#endif
Integer pnga_sprs_array_matmat_multiply(Integer s_a, Integer s_b)
{
  Integer *top;
  Integer *list;
  Integer *idx;
  Integer *jdx;
  Integer lcnt;
  Integer icnt;
  Integer hdl_a = s_a+GA_OFFSET;
  Integer hdl_b = s_b+GA_OFFSET;
  Integer hdl_c;
  Integer bufsize;
  Integer elemsize;
  Integer idim, jdim;
  Integer i, j, k, l, m, n;
  void *data;
  Integer nprocs = pnga_pgroup_nnodes(SPA[hdl_a].grp);
  Integer me = pnga_pgroup_nodeid(SPA[hdl_a].grp);
  Integer longidx;
  Integer type;
  Integer ihi, ilo;
  Integer rowdim;
  Integer *count;
  Integer nblocks;
  Integer s_c;
  Integer ilen;
  Integer max_nnz_a, max_nnz_b;
  Integer *row_nnz;
  Integer nnz;
  int64_t max_nnz;
  double tbeg, time, tbeg2, time2;
  tbeg = pnga_wtime();
  /* Do some initial verification to see if matrix multiply is possible */
  if (SPA[hdl_a].type != SPA[hdl_b].type) {
    pnga_error("(ga_sprs_array_matmat_multiply) types of sparse matrices"
    " A and B must match",0);
  }
  type = SPA[hdl_a].type;
  if (SPA[hdl_a].grp != SPA[hdl_b].grp) {
    pnga_error("(ga_sprs_array_matmat_multiply) matrices A and B must"
    " be on the same group",0);
  }
  if (SPA[hdl_a].jdim != SPA[hdl_b].idim) {
    pnga_error("(ga_sprs_array_matmat_multiply) column dimension of"
      " A must match row dimension of B",0);
  }
  if (SPA[hdl_a].idx_size != SPA[hdl_b].idx_size) {
    pnga_error("(ga_sprs_array_matmat_multiply) size of integer"
      " indices of A and B must match",0);
  }
  if (SPA[hdl_a].idx_size == sizeof(int64_t)) {
    longidx = 1;
  } else {
    longidx = 0;
  }
  /* Allocate initial buffers to hold elements of product matrix. This
   * algorithm assumes that product matrix remains relatively sparse. */
#if ENABLE_PREALLOC
  {
    Integer nrows = SPA[hdl_a].ihi-SPA[hdl_a].ilo + 1;
    max_nnz_a = SPA[hdl_a].max_nnz;
    max_nnz_b = SPA[hdl_b].max_nnz;
    if (max_nnz_a*max_nnz_b > SPA[hdl_b].jdim) {
      bufsize = nrows*SPA[hdl_b].jdim;
    } else {
      bufsize = nrows*max_nnz_a*max_nnz_b;
    }
  }
#else
  bufsize = INIT_BUF_SIZE;
#endif
  elemsize = SPA[hdl_a].size;
  idim = SPA[hdl_a].idim;
  jdim = SPA[hdl_b].jdim;
  top = (Integer*)malloc(bufsize*sizeof(Integer));
  list = (Integer*)malloc(bufsize*sizeof(Integer));
  idx = (Integer*)malloc(bufsize*sizeof(Integer));
  jdx = (Integer*)malloc(bufsize*sizeof(Integer));
  data = malloc(bufsize*elemsize);
  for (i=0; i<bufsize; i++) top[i] = -1;
  for (i=0; i<bufsize; i++) list[i] = -1;
  lcnt = 0;
  pnga_pgroup_sync(SPA[hdl_a].grp);
  time = pnga_wtime()-tbeg;
  if (me == 0) {
    printf(" Mat-mat multiply initialization:            %16.8f\n",time);
  }
  /* loop over processors in row and then loop over processor in column.
   * Multiply block pairs */
  time2 = 0.0;
  time_get = 0.0;
  cnt_get = 0;
  size_get = 0;
  tbeg = pnga_wtime();
  for (l=0; l<nprocs; l++) {
    for (n=0; n<nprocs; n++) {
      Integer *idx_a, *jdx_a, *idx_b, *jdx_b;
      void *data_a, *data_b, *iptr, *jptr;
      Integer ilo_a, ihi_a, jlo_a, jhi_a;
      Integer ilo_b, ihi_b, jlo_b, jhi_b;
      Integer ii,jlen;
      if (!pnga_sprs_array_get_block(s_a, me, l, &iptr, &jptr,
            &data_a, &ilo_a, &ihi_a, &jlo_a, &jhi_a)) continue;
      idx_a = (Integer*)iptr;
      jdx_a = (Integer*)jptr;
      if (!pnga_sprs_array_get_block(s_b, l, n, &iptr, &jptr,
            &data_b, &ilo_b, &ihi_b, &jlo_b, &jhi_b)) continue;
      idx_b = (Integer*)iptr;
      jdx_b = (Integer*)jptr;
      tbeg2 = pnga_wtime();
      if (type == C_INT) {
        SPRS_REAL_MATMAT_MULTIPLY_M(int,idx_a,jdx_a,idx_b,jdx_b);
      } else if (type == C_LONG) {
        SPRS_REAL_MATMAT_MULTIPLY_M(long,idx_a,jdx_a,idx_b,jdx_b);
      } else if (type == C_LONGLONG) {
        SPRS_REAL_MATMAT_MULTIPLY_M(long long,idx_a,jdx_a,idx_b,jdx_b);
      } else if (type == C_FLOAT) {
        SPRS_REAL_MATMAT_MULTIPLY_M(float,idx_a,jdx_a,idx_b,jdx_b);
      } else if (type == C_DBL) {
        SPRS_REAL_MATMAT_MULTIPLY_M(double,idx_a,jdx_a,idx_b,jdx_b);
      } else if (type == C_SCPL) {
        SPRS_COMPLEX_MATMAT_MULTIPLY_M(float,idx_a,jdx_a,idx_b,jdx_b);
      } else if (type == C_DCPL) {
        SPRS_COMPLEX_MATMAT_MULTIPLY_M(double,idx_a,jdx_a,idx_b,jdx_b);
      }
      time2 += pnga_wtime()-tbeg2;
      if (idx_a != NULL) free(idx_a);
      if (jdx_a != NULL) free(jdx_a);
      if (data_a != NULL) free(data_a);
      if (idx_b != NULL) free(idx_b);
      if (jdx_b != NULL) free(jdx_b);
      if (data_b != NULL) free(data_b);
    }
  }
  tbeg2 = pnga_wtime();
  pnga_pgroup_sync(SPA[hdl_a].grp);
  time_sync = pnga_wtime()-tbeg2;
  time = pnga_wtime()-tbeg;
  if (me == 0) {
    printf(" Mat-mat multiply matrix kernel multiply:    %16.8f\n",time);
    printf("   Time spent in local multiplies:      %16.8f\n",time2);
    printf("   Time spent in get:                   %16.8f\n",time_get);
    printf("   Time spent in sync:                  %16.8f\n",time_sync);
    printf("   Number of times get called:          %16ld\n",cnt_get);
    printf("   Number of bytes moved in get:        %16ld\n",size_get);
  }

  /* At this point all blocks have been multiplied and the resulting
   * values are stored in the link list defined by top, list, idx,
   * jdx, data. No data needs to be moved, but all elements need to
   * be resorted into blocks and then into rows. Start by binning
   * data into column and row blocks. */
  tbeg = pnga_wtime();
  free(top);
  for (i=0; i<bufsize; i++) list[i] = -1;
  ilo = SPA[hdl_a].ilo;
  ihi = SPA[hdl_a].ihi;
  rowdim = ihi-ilo+1;
  top = (Integer*)malloc(nprocs*sizeof(Integer));
  count = (Integer*)malloc(nprocs*sizeof(Integer));
  for (i=0; i<nprocs; i++) top[i] = -1;
  for (i=0; i<nprocs; i++) count[i] = 0;
  /* bin up all data elements into column blocks */
  for (i=0; i<lcnt; i++) {
    /* jdx is unit based so need to subtract 1 */
    Integer np = ((jdx[i]-1)*nprocs/jdim);
    if (np >= nprocs) np = nprocs-1;
    list[i] = top[np];
    top[np] = i;
    count[np]++;
  }
  /* count up number of column blocks with data */
  nblocks = 0;
  for (i=0; i<nprocs; i++) {
    if (count[i] > 0) nblocks++;
  }

  /* create a new sparse array to hold product array */
  s_c = pnga_sprs_array_create(idim,jdim,type,SPA[hdl_a].idx_size);
  hdl_c = GA_OFFSET + s_c;
  free(SPA[hdl_c].idx);
  free(SPA[hdl_c].jdx);
  free(SPA[hdl_c].val);

  /* set up array of offsets */
  SPA[hdl_c].blkidx = (Integer*)malloc(nblocks*sizeof(Integer));
  SPA[hdl_c].blksize = (Integer*)malloc(nblocks*sizeof(Integer));
  SPA[hdl_c].offset = (Integer*)malloc(nblocks*sizeof(Integer));
  nblocks = 0;
  for (i=0; i<nprocs; i++) {
    if (count[i] > 0) {
      SPA[hdl_c].blkidx[nblocks] = i;
      SPA[hdl_c].blksize[nblocks] = count[i];
      if (nblocks>0) SPA[hdl_c].offset[nblocks] = SPA[hdl_c].offset[nblocks-1]
        + SPA[hdl_c].blksize[nblocks-1];
      nblocks++;
    }
  }
  SPA[hdl_c].offset[0] = 0;
  for (i=1; i<nblocks; i++) {
    SPA[hdl_c].offset[i] = SPA[hdl_c].offset[i-1]+SPA[hdl_c].blksize[i-1];
  }
  pnga_pgroup_sync(SPA[hdl_a].grp);
  time = pnga_wtime()-tbeg;
  if (me == 0) {
    printf(" Mat-mat multiply create product array:      %16.8f\n",time);
  }

  tbeg = pnga_wtime();
  SPA[hdl_c].nblocks = nblocks;
  SPA[hdl_c].ilo = ilo;
  SPA[hdl_c].ihi = ihi;
  SPA[hdl_c].type = type;
  SPA[hdl_c].nprocs = SPA[hdl_a].nprocs;
  SPA[hdl_c].idx = NULL;
  SPA[hdl_c].jdx = NULL;
  SPA[hdl_c].val = NULL;
  SPA[hdl_c].nval = lcnt;
  SPA[hdl_c].maxval = bufsize;
  SPA[hdl_c].size = SPA[hdl_a].size;
  /* sort data into column blocks */
  SPA[hdl_c].idx = (Integer*)malloc(lcnt*sizeof(Integer));
  SPA[hdl_c].jdx = (Integer*)malloc(lcnt*sizeof(Integer));
  SPA[hdl_c].val = (Integer*)malloc(lcnt*SPA[hdl_c].size);
  icnt = 0;
  for (n=0; n<nprocs; n++) {
    Integer elemsize = SPA[hdl_c].size;
    i = top[n];
    while (i >= 0) {
      SPA[hdl_c].idx[icnt] = idx[i];
      SPA[hdl_c].jdx[icnt] = jdx[i];
      memcpy(&((char*)SPA[hdl_c].val)[icnt*elemsize],
          &((char*)data)[i*elemsize],elemsize);
      i = list[i];
      icnt++;
    }
  }
  free(idx);
  free(jdx);
  free(data);
  pnga_pgroup_sync(SPA[hdl_a].grp);
  time = pnga_wtime()-tbeg;
  if (me == 0) {
    printf(" Mat-mat multiply sort into column blocks:   %16.8f\n",time);
  }

  /* Set up global arrays to hold distributed indices and non-zero values */
  tbeg = pnga_wtime();
  time_create = 0.0;
  time_setdata = 0.0;
  time_setirreg = 0.0;
  time_alloc = 0.0;
  time_gop1 = 0.0;
  time_gop2 = 0.0;
  {
    int64_t isize = (rowdim+1)*nblocks;
    int64_t totalsize = 0;
    Integer ndim = 1;
    Integer *offset = (Integer*)malloc(nprocs*sizeof(Integer));
    Integer *tmp = (Integer*)malloc(nprocs*sizeof(Integer));
    Integer *map = (Integer*)malloc(nprocs*sizeof(Integer));
    double ttbeg;
    /* set up array to hold row indices. Evaluate offsets for rows */
    for (i=0; i<nprocs; i++) {
      offset[i] = 0;
      tmp[i] = 0;
    }
    tmp[me] = isize;
    ttbeg = pnga_wtime();
    if (sizeof(Integer) == 8) {
      pnga_pgroup_gop(SPA[hdl_a].grp,C_LONG,tmp,nprocs,"+");
    } else {
      pnga_pgroup_gop(SPA[hdl_a].grp,C_INT,tmp,nprocs,"+");
    }
    time_gop1 += pnga_wtime()-ttbeg;
    offset[0] = 0;
    for (i=0; i<nprocs; i++) {
      totalsize += tmp[i];
      if (i>0) offset[i] = offset[i-1]+tmp[i-1];
      map[i] = offset[i]+1; /* 1-based indexing for map array */
    }

    ttbeg = pnga_wtime();
    SPA[hdl_c].g_i = pnga_create_handle();
    time_create += pnga_wtime()-ttbeg;
    ttbeg = pnga_wtime();
    if (longidx) {
      pnga_set_data(SPA[hdl_c].g_i,ndim,&totalsize,C_LONG);
    } else {
      pnga_set_data(SPA[hdl_c].g_i,ndim,&totalsize,C_INT);
    }
    time_setdata += pnga_wtime()-ttbeg;
    ttbeg = pnga_wtime();
    pnga_set_irreg_distr(SPA[hdl_c].g_i,map,&nprocs);
    time_setirreg += pnga_wtime()-ttbeg;
    ttbeg = pnga_wtime();
    pnga_allocate(SPA[hdl_c].g_i);
    time_alloc += pnga_wtime()-ttbeg;
    /* set up arrays to hold column indices and data. Evaluate offsets
     * for row blocks*/
    for (i=0; i<nprocs; i++) {
      offset[i] = 0;
      tmp[i] = 0;
    }
    tmp[me] = lcnt;
    ttbeg = pnga_wtime();
    if (sizeof(Integer) == 8) {
      pnga_pgroup_gop(SPA[hdl_a].grp,C_LONG,tmp,nprocs,"+");
    } else {
      pnga_pgroup_gop(SPA[hdl_a].grp,C_INT,tmp,nprocs,"+");
    }
    time_gop2 += pnga_wtime()-ttbeg;
    offset[0] = 0;
    totalsize = 0;
    for (i=0; i<nprocs; i++) {
      totalsize += tmp[i];
      if (i>0) offset[i] = offset[i-1]+tmp[i-1];
      map[i] = offset[i]+1; /* 1-based indexing for map array */
    }
    ttbeg = pnga_wtime();
    SPA[hdl_c].g_j = pnga_create_handle();
    SPA[hdl_c].g_data = pnga_create_handle();
    time_create += pnga_wtime()-ttbeg;
    nnz = totalsize;
    ttbeg = pnga_wtime();
    if (longidx) {
      pnga_set_data(SPA[hdl_c].g_j,ndim,&totalsize,C_LONG);
    } else {
      pnga_set_data(SPA[hdl_c].g_j,ndim,&totalsize,C_INT);
    }
    pnga_set_data(SPA[hdl_c].g_data,ndim,&totalsize,SPA[hdl_c].type);
    time_setdata += pnga_wtime()-ttbeg;
    ttbeg = pnga_wtime();
    pnga_set_irreg_distr(SPA[hdl_c].g_j,map,&nprocs);
    pnga_set_irreg_distr(SPA[hdl_c].g_data,map,&nprocs);
    time_setirreg += pnga_wtime()-ttbeg;
    ttbeg = pnga_wtime();
    pnga_allocate(SPA[hdl_c].g_j);
    pnga_allocate(SPA[hdl_c].g_data);
    time_alloc += pnga_wtime()-ttbeg;
    free(map);
    free(tmp);
    free(offset);
  }
  time = pnga_wtime()-tbeg;
  if (me == 0) {
    printf(" Mat-mat multiply set up final arrays:       %16.8f\n",time);
    printf("   Time spent creating handle:          %16.8f\n",time_create);
    printf("   Time spent in setting data:          %16.8f\n",time_setdata);
    printf("   Time spent in setting irregular;     %16.8f\n",time_setirreg);
    printf("   Time spent in allocation:            %16.8f\n",time_alloc);
    printf("   Time spent in first reduction:       %16.8f\n",time_gop1);
    printf("   Time spent in second reduction:      %16.8f\n",time_gop2);
  }
  /* organize row block into column blocks in CSR format */
  tbeg = pnga_wtime();
  {
    int *ti, *tj;
    int64_t *lti, *ltj;
    void *tdata;
    Integer tlo, thi, tld;
    char *cdata, *ctdata;
    /* Get pointers to global arrays */
    pnga_distribution(SPA[hdl_c].g_i,me,&tlo,&thi);
    if (longidx) {
      pnga_access_ptr(SPA[hdl_c].g_i,&tlo,&thi,&lti,&tld);
    } else {
      pnga_access_ptr(SPA[hdl_c].g_i,&tlo,&thi,&ti,&tld);
    }
    pnga_distribution(SPA[hdl_c].g_j,me,&tlo,&thi);
    if (longidx) {
      pnga_access_ptr(SPA[hdl_c].g_j,&tlo,&thi,&ltj,&tld);
    } else {
      pnga_access_ptr(SPA[hdl_c].g_j,&tlo,&thi,&tj,&tld);
    }
    pnga_distribution(SPA[hdl_c].g_data,me,&tlo,&thi);
    pnga_access_ptr(SPA[hdl_c].g_data,&tlo,&thi,&tdata,&tld);
    idx = SPA[hdl_c].idx;
    jdx = SPA[hdl_c].jdx;
    cdata = (char*)SPA[hdl_c].val;
    ctdata = (char*)tdata;
    ilen = ihi - ilo + 1;
    row_nnz = (Integer*)malloc(ilen*sizeof(Integer));
    for (n=0; n<ilen; n++) row_nnz[n] = 0;
    /* loop over column blocks */
    for (n=0; n<nblocks; n++) {
      Integer *rowtop;
      Integer *rowlist;
      Integer irow;
      Integer offset_i;
      Integer offset_j;
      j = SPA[hdl_c].blkidx[n];
      /* sort blocks into rows */
      rowtop = (Integer*)malloc(ilen*sizeof(Integer));
      rowlist = (Integer*)malloc(count[j]*sizeof(Integer));
      for (i=0; i<ilen; i++) rowtop[i] = -1;
      for (i=0; i<count[j]; i++) rowlist[i] = -1;
      offset_i = n*(ilen+1);
      offset_j = SPA[hdl_c].offset[n];
      /* sort block into rows */
      icnt = 0;
      while (icnt < SPA[hdl_c].blksize[n]) {
        /* idx is unit based and ilo is zero based. Use offset_j since
         * data in idx is partitioned the same way as for jdx and data */
        Integer id = idx[icnt+offset_j]-1-ilo;
        rowlist[icnt] = rowtop[id];
        rowtop[id] = icnt;
        icnt++;
      }

      /* now organize data in g_i, g_j, g_data */
      icnt = 0;
      if (longidx) {
        for (irow=0; irow<ilen; irow++) {
          Integer jd = rowtop[irow];
          lti[offset_i+irow] = icnt;
          while (jd >= 0) {
            ltj[offset_j+icnt] = (int64_t)jdx[offset_j+jd]-1;
            memcpy(&ctdata[elemsize*(offset_j+icnt)],
                &cdata[(offset_j+jd)*elemsize],elemsize);
            row_nnz[irow]++;
            icnt++;
            jd = rowlist[jd];
          }
        }
        lti[offset_i+ilen] = icnt;
      } else {
        for (irow=0; irow<ilen; irow++) {
          Integer jd = rowtop[irow];
          ti[offset_i+irow] = icnt;
          while (jd >= 0) {
            tj[offset_j+icnt] = (int)jdx[offset_j+jd]-1;
            memcpy(&ctdata[elemsize*(offset_j+icnt)],
                &cdata[(offset_j+jd)*elemsize],elemsize);
            icnt++;
            row_nnz[irow]++;
            jd = rowlist[jd];
          }
        }
        ti[offset_i+ilen] = icnt;
      }
      /* clean up arrays */
      free(rowtop);
      free(rowlist);
    }
    /* Find maximum number of non-zeros per row on this processors */
    max_nnz = 0;
    for (i=0; i<ilen; i++) {
      if (max_nnz < (int64_t)row_nnz[i]) max_nnz = (int64_t)row_nnz[i];
    }
    free(row_nnz);
    pnga_pgroup_gop(SPA[hdl_c].grp,C_LONG,&max_nnz,1,"max");
    SPA[hdl_c].max_nnz = (Integer)max_nnz;

    pnga_distribution(SPA[hdl_c].g_i,me,&tlo,&thi);
    pnga_release(SPA[hdl_c].g_i,&tlo,&thi);
    pnga_distribution(SPA[hdl_c].g_data,me,&tlo,&thi);
    pnga_release(SPA[hdl_c].g_data,&tlo,&thi);
    pnga_release(SPA[hdl_c].g_j,&tlo,&thi);
  }
  free(top);
  free(list);
  free(SPA[hdl_c].idx);
  free(SPA[hdl_c].jdx);
  free(SPA[hdl_c].val);
  SPA[hdl_c].idx = NULL;
  SPA[hdl_c].jdx = NULL;
  SPA[hdl_c].val= NULL;
  free(count);
  pnga_pgroup_sync(SPA[hdl_a].grp);
  time = pnga_wtime()-tbeg;
  if (me == 0) {
    printf(" Mat-mat multiply create final distribution: %16.8f\n",time);
  }

  tbeg = pnga_wtime();
  /*DEBUG*/
#if 0
  {
    int *ti, *tj;
    int64_t *lti, *ltj;
    void *tdata;
    Integer tlo, thi, tld;
    char *cdata, *ctdata;
    /* Get pointers to global arrays */
    pnga_distribution(SPA[hdl_c].g_i,me,&tlo,&thi);
    if (longidx) {
      pnga_access_ptr(SPA[hdl_c].g_i,&tlo,&thi,&lti,&tld);
    } else {
      pnga_access_ptr(SPA[hdl_c].g_i,&tlo,&thi,&ti,&tld);
    }
    pnga_distribution(SPA[hdl_c].g_j,me,&tlo,&thi);
    if (longidx) {
      pnga_access_ptr(SPA[hdl_c].g_j,&tlo,&thi,&ltj,&tld);
    } else {
      pnga_access_ptr(SPA[hdl_c].g_j,&tlo,&thi,&tj,&tld);
    }
    pnga_distribution(SPA[hdl_c].g_data,me,&tlo,&thi);
    pnga_access_ptr(SPA[hdl_c].g_data,&tlo,&thi,&tdata,&tld);
    cdata = (char*)data;
    ctdata = (char*)tdata;
    ilo = SPA[hdl_c].ilo;
    ihi = SPA[hdl_c].ihi;
    /* loop over column blocks */
    for (n=0; n<nblocks; n++) {
      Integer ilen = ihi - ilo + 1;
      Integer icnt = 0;
      Integer offset_i = n*(ilen+1);
      Integer offset_j = SPA[hdl_c].offset[n];
      Integer irow;
      Integer *iptr = lti + offset_i;
      Integer *jptr = ltj + offset_j;
      int *vptr = (int*)(ctdata+offset_j*elemsize);
      printf("p[%ld] block: %ld iptr: %p jptr: %p vptr: %p\n",me,n,iptr,jptr,vptr);
      for (irow=0; irow<ilen; irow++) {
        Integer jlo, jhi;
        Integer jlen = iptr[irow+1]-iptr[irow];
        printf("p[%d] irow: %ld i[%ld]: %ld i[%ld]: %ld\n",
            me,irow,irow,iptr[irow],irow+1,iptr[irow+1]);
        for (j=0; j<jlen; j++) {
          printf("p[%d] i: %ld iptr[%ld]+j: %ld j: %ld val: %d\n",
              me,irow+ilo,irow,iptr[irow]+j,jptr[iptr[irow]+j],vptr[iptr[irow]+j]);
        }
      }
    }
  } /*END DEBUG*/
#endif
  /* Create global array to store information on sparse blocks */
  {
    Integer dims[3], chunk[3];
    Integer three = 3;
    Integer g_blk;
    int64_t *row_info;
    Integer jlo, jhi;
    dims[0] = 7;
    dims[1] = nprocs;
    dims[2] = nprocs;
    chunk[0] = 7;
    chunk[1] = -1;
    chunk[2] = -1;
    /* g_blk contains information about how data for each sparse
     * block is laid out in g_j and g_data. The last two dimensions
     * describe location of sparse block in nproc X nproc array
     * of sparse blocks corresponding to original sparse matrix.
     *
     * First dimension contains the following information on each
     * block
     *    ilo: lowest row index of block
     *    ihi: highest row index of block
     *    jlo: lowest column index of block
     *    jhi: highest column index of block
     *    offset: offset in g_j and g_data for column indices and data
     *            values for block
     *    blkend: last index  g_j and g_data for block
     */
    g_blk = pnga_create_handle();
    pnga_set_pgroup(g_blk,SPA[hdl_c].grp);
    pnga_set_data(g_blk,three,dims,C_LONG);
    pnga_set_chunk(g_blk,chunk);
    if (!pnga_allocate(g_blk)) {
      pnga_error("(pnga_sprs_matmat_multiply) Failure allocating g_blk",0);
    }
    SPA[hdl_c].g_blk = g_blk;
    row_info = (int64_t*)malloc(7*nprocs*sizeof(int64_t));
    /* set up g_blk */
    for (i=0; i<nprocs; i++) {
      Integer jlo, jhi;
      Integer jbot, jtop;
      int iblk;
      /* find offset for row block on this processor
       * in g_j (should be the same for g_data */
      pnga_distribution(SPA[hdl_c].g_j,me,&jbot,&jtop);

      /* calculate column limits for processor i */
      jlo = (SPA[hdl_c].jdim*i)/nprocs;
      while ((jlo*nprocs)/jdim < i) {
        jlo++;
      }
      while ((jlo*nprocs)/jdim > i) {
        jlo--;
      }
      if (i < nprocs-1) {
        jhi = (SPA[hdl_c].jdim*(i+1))/nprocs;
        while ((jhi*nprocs)/jdim < i+1) {
          jhi++;
        }
        while ((jhi*nprocs)/jdim > i+1) {
          jhi--;
        }
        jhi--;
      } else {
        jhi = SPA[hdl_c].jdim-1;
      }
      /* set indices to fortran indexing */
      jlo++;
      jhi++;
      /* find index for block in blksize and offset arrays */
      iblk = -1;
      for (j=0; j<SPA[hdl_c].nblocks; j++) {
        if (SPA[hdl_c].blkidx[j] == i) {
          iblk = j;
          break;
        }
      }
      if (iblk != -1) {
        /* block has data */
        row_info[i*7  ] = SPA[hdl_c].ilo+1;
        row_info[i*7+1] = SPA[hdl_c].ihi+1;
        row_info[i*7+2] = jlo;
        row_info[i*7+3] = jhi;
        row_info[i*7+4] = jbot+SPA[hdl_c].offset[iblk];
        row_info[i*7+5] = row_info[i*7+4]+SPA[hdl_c].blksize[iblk]-1;
      } else {
        /* block contains no data */
        row_info[i*7  ] = SPA[hdl_c].ilo+1;
        row_info[i*7+1] = SPA[hdl_c].ihi+1;
        row_info[i*7+2] = jlo;
        row_info[i*7+3] = jhi;
        if (i == 0) {
          row_info[4] = 1;
          row_info[5] = 0;
        } else {
          row_info[i*7+4] = row_info[(i-1)*7+4]+1;
          row_info[i*7+5] = row_info[(i-1)*7+4]-1;
        }
      }
      row_info[i*7+6] = iblk;
    }
    /* copy data in row info to g_blk */
    {
      Integer tlo[3], thi[3], tld[2];
      tlo[0] = 1;
      tlo[1] = me+1;
      tlo[2] = 1;
      thi[0] = 7;
      thi[1] = me+1;
      thi[2] = nprocs;
      tld[0] = 7;
      tld[1] = 1;
      pnga_put(g_blk,tlo,thi,row_info,tld);
      pnga_pgroup_sync(SPA[hdl_c].grp);
    }
    free(row_info);
  }
  SPA[hdl_c].ready = 1;
  pnga_pgroup_sync(SPA[hdl_a].grp);
  time = pnga_wtime()-tbeg;
  if (me == 0) {
    printf(" Mat-mat multiply create array metadata:     %16.8f\n",time);
    printf(" Mat-mat multiply number of nonzeros:        %16d\n",nnz);
  }
  return s_c;
}
#undef SPRS_REAL_MATMAT_MULTIPLY_M
#undef SPRS_COMPLEX_MATMAT_MULTIPLY_M

#define SPRS_REAL_ACTIVATE_M(_type,_iptr,_jptr,_vptr,    \
    _idx,_jdx,_vals)                                     \
  {                                                      \
    _type *_ptr = (_type*)_vptr;                         \
    _type x, val;                                        \
    for (i=ilo; i<=ihi; i++) {                           \
      Integer ncols = _iptr[i+1-ilo]-_iptr[i-ilo];       \
      for (j=0; j<ncols; j++) {                          \
        _ptr[_iptr[i-ilo]+j] += *((_type*)bias);         \
        val = _ptr[_iptr[i-ilo]+j];                      \
        x = val<0.0?0.0:val>32.0?32.0:val;               \
        if (x>0.0) {                                     \
          _idx[total_vals] = i;                          \
          _jdx[total_vals] = _jptr[_iptr[i-ilo]+j];      \
          ((_type*)_vals)[total_vals] = x;               \
          total_vals++;                                  \
        }                                                \
      }                                                  \
    }                                                    \
  }

/**
 * Apply activation function to sparse array
 * @param s_a sparse array handle
 * @param bias constant value added to non-zero elements
 * @param categories array containing indices of features from input that are
 *                   still retained on output
 * @param features number of features remaining in output array
 * @return true if successful
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_activate_relu =  pnga_sprs_array_activate_relu
#endif
logical pnga_sprs_array_activate_relu(Integer s_a, void *bias, void **categories,
    void *features)
{
  Integer handle = s_a + GA_OFFSET;
  Integer i, j, n, ncols, iproc;
  Integer jdim = SPA[handle].jdim;
  Integer idx_size = SPA[handle].idx_size;
  void *vptr;
  int64_t *ilptr=NULL, *jlptr=NULL;
  int *iptr=NULL, *jptr=NULL;
  Integer ilo, ihi;
  Integer jlo, jhi;
  Integer *idx, *jdx;
  void *vals;
  Integer total_vals;
  Integer ret = 1;

  ilo = SPA[handle].ilo;
  ihi = SPA[handle].ihi;

  /* allocate arrays to hold values after applying activation
   * function */
  total_vals = 0;
  for (i=0; i<SPA[handle].nblocks; i++) total_vals += SPA[handle].blksize[i];
  idx = (Integer*)malloc(total_vals*sizeof(Integer));
  jdx = (Integer*)malloc(total_vals*sizeof(Integer));
  vals = malloc(total_vals*SPA[handle].size);
  /* loop over all non-zero blocks */
  total_vals = 0;
  for (n=0; n<SPA[handle].nblocks; n++) {
    iproc = SPA[handle].blkidx[n];
    pnga_sprs_array_column_distribution(s_a,iproc,&jlo,&jhi);
    /* Apply bias to all non-zero values  and then apply activation
     * function. If value is less than zero, then drop value from matrix */
    if (idx_size == 4) {
      pnga_sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
      if (vptr != NULL) {
        if (SPA[handle].type == C_FLOAT) {
          SPRS_REAL_ACTIVATE_M(float,iptr,jptr,vptr,idx,jdx,vals);
        } else if (SPA[handle].type == C_DBL) {
          SPRS_REAL_ACTIVATE_M(double,iptr,jptr,vptr,idx,jdx,vals);
        }
      }
    } else {
      pnga_sprs_array_access_col_block(s_a,iproc,&ilptr,&jlptr,&vptr);
      if (vptr != NULL) {
        if (SPA[handle].type == C_FLOAT) {
          SPRS_REAL_ACTIVATE_M(float,ilptr,jlptr,vptr,idx,jdx,vals);
        } else if (SPA[handle].type == C_DBL) {
          SPRS_REAL_ACTIVATE_M(double,ilptr,jlptr,vptr,idx,jdx,vals);
        }
      }
    }
  }
  /* Need to resort sparse array to handle new values. Start by reallocating
   * global arrays containing sparse data */
  pnga_destroy(SPA[handle].g_i);
  pnga_destroy(SPA[handle].g_j);
  pnga_destroy(SPA[handle].g_data);
  pnga_destroy(SPA[handle].g_blk);
  {
    Integer longidx, iproc;
    Integer idim, jdim;
    Integer icnt, jcnt, ncnt, nrows;
    Integer *row_nnz;
    Integer *active;
    Integer *compress;
    Integer *list, *top, *count;
    int64_t *offset, *row_info;
    Integer nproc = pnga_pgroup_nnodes(SPA[handle].grp);
    Integer me = pnga_pgroup_nodeid(SPA[handle].grp);
    Integer *totals;
    Integer total;
    Integer *map;
    char op[4];
    int *isdx, *jsdx;
    int64_t *ildx, *jldx;
    Integer max_nnz;
    Integer lo, hi, ld;
    Integer irow, elemsize;
    Integer one = 1;
    /* set variable that distinguishes between long and ints for indices */
    if (SPA[handle].idx_size == sizeof(int64_t)) {
      longidx = 1;
    } else {
      longidx = 0;
    }
    /* calculate number of active features */
    idim = SPA[handle].idim;
    jdim = SPA[handle].jdim;
    active = (Integer*)malloc(jdim*sizeof(Integer));
    compress = (Integer*)malloc(jdim*sizeof(Integer));
    for (i=0; i<jdim; i++) {
      active[i] = -1;
      compress[i] = -1;
    }
    for (i=0; i<total_vals; i++) {
      if (jdx[i] >= jdim) {
        printf("p[%d] idx out of bounds: %d jdim: %d\n",pnga_nodeid(),
            idx[i],jdim);
      }
      active[jdx[i]] = 1;
    }
    strcpy(op,"max");
    if (sizeof(Integer) == 4) {
      pnga_pgroup_gop(SPA[handle].grp,C_INT,active,jdim,op);
    } else {
      pnga_pgroup_gop(SPA[handle].grp,C_LONG,active,jdim,op);
    }
    /* calculate compression array */
    jcnt=0;
    for (i=0; i<jdim; i++) {
      if (active[i] > 0) {
        compress[i] = jcnt;
        jcnt++;
      }
    }
    /* remap all old column indices to new indices */
    for (i=0; i<total_vals; i++) {
      jdx[i] = compress[jdx[i]];
    }

    if (longidx) {
      *((int64_t*)features) = jcnt;
      j=0;
      for (i=0; i<jdim; i++) {
        if (active[i] > 0) {
          ((int64_t*)(*categories))[j] = ((int64_t*)(*categories))[i];
          j++;
        }
      }
    } else {
      *((int*)features) = jcnt;
      j=0;
      for (i=0; i<jdim; i++) {
        if (active[i] > 0) {
          ((int*)(*categories))[j] = ((int*)(*categories))[i];
          j++;
        }
      }
    }
    jdim = jcnt;
    SPA[handle].jdim = jdim;

    /* get total number of non-zeros on each processor */
    totals = (Integer*)malloc(nproc*sizeof(Integer));
    for (i=0; i<nproc; i++) totals[i] = 0;
    totals[me] = total_vals;
    strcpy(op,"+");
    if (sizeof(Integer) == 4) {
      pnga_pgroup_gop(SPA[handle].grp,C_INT,totals,nproc,op);
    } else {
      pnga_pgroup_gop(SPA[handle].grp,C_LONG,totals,nproc,op);
    }
    /* reassemble array. At this point there is a complete set
     * of new values after applying the activation function.
     * These need to be stored in appropriate column blocks*/
    elemsize = SPA[handle].size;
    map = (Integer*)malloc(nproc*sizeof(Integer));
    /* calculate map array */
    map[0] = 1;
    total = totals[0];
    for (i=1; i<nproc; i++) {
      map[i] = map[i-1]+totals[i-1];
      total += totals[i];
    }
    SPA[handle].g_data = pnga_create_handle();
    pnga_set_data(SPA[handle].g_data,one,&total,SPA[handle].type);
    pnga_set_irreg_distr(SPA[handle].g_data,map,&nproc);
    if (!pnga_allocate(SPA[handle].g_data)) return 0;
    SPA[handle].g_j = pnga_create_handle();
    if (idx_size == 4) {
      pnga_set_data(SPA[handle].g_j,one,&total,C_INT);
    } else {
      pnga_set_data(SPA[handle].g_j,one,&total,C_LONG);
    }
    pnga_set_irreg_distr(SPA[handle].g_j,map,&nproc);
    if (!pnga_allocate(SPA[handle].g_j)) return 0;

    list = (Integer*)malloc(total_vals*sizeof(Integer));

    /* Create global array to store information on sparse blocks */
    {
      Integer dims[3], chunk[3];
      Integer three = 3;
      Integer g_blk;
      dims[0] = 7;
      dims[1] = nproc;
      dims[2] = nproc;
      chunk[0] = 7;
      chunk[1] = -1;
      chunk[2] = -1;
      g_blk = pnga_create_handle();
      pnga_set_pgroup(g_blk,SPA[handle].grp);
      pnga_set_data(g_blk,three,dims,C_LONG);
      pnga_set_chunk(g_blk,chunk);
      if (!pnga_allocate(g_blk)) return 0;
      SPA[handle].g_blk = g_blk;
      row_info = (int64_t*)malloc(7*nproc*sizeof(int64_t));
    }

    /* start sorting everything into column blocks */
    count = (Integer*)malloc(nproc*sizeof(Integer));
    top = (Integer*)malloc(nproc*sizeof(Integer));
    offset = (int64_t*)malloc(nproc*sizeof(int64_t));
    for (i=0; i<nproc; i++) {
      count[i] = 0;
      top[i] = -1;
      offset[i] = -1;
    }
    for (i=0; i<total_vals; i++) {
      /* jdx is zero based */
      iproc = (((Integer)jdx[i])*nproc)/jdim;
      if (iproc >= nproc) iproc = nproc-1;
      count[iproc]++;
      list[i] = top[iproc];
      top[iproc] = i;
    }
    /* find out how many column blocks have data */
    ncnt = 0;
    for (i=0; i<nproc; i++)
      if (count[i] > 0) ncnt++;
    free(SPA[handle].blkidx);
    free(SPA[handle].offset);
    free(SPA[handle].blksize);
    SPA[handle].nblocks = ncnt;
    if (ncnt > 0) SPA[handle].blkidx = (Integer*)malloc(ncnt*sizeof(Integer));
    if (ncnt > 0) SPA[handle].offset = (Integer*)malloc(ncnt*sizeof(Integer));
    if (ncnt > 0) SPA[handle].blksize = (Integer*)malloc(ncnt*sizeof(Integer));
    /* allocate local buffers to sort everything into column blocks */
    SPA[handle].val = malloc(total_vals*SPA[handle].size);
    SPA[handle].idx = malloc(total_vals*sizeof(Integer));
    SPA[handle].jdx = malloc(total_vals*sizeof(Integer));

    ncnt = 0;
    icnt = 0;
    for (i=0; i<nproc; i++) {
      Integer j = top[i];
      if (j >= 0) {
        char* vbuf = SPA[handle].val;
        Integer* ibuf = SPA[handle].idx;
        Integer* jbuf = SPA[handle].jdx;
        SPA[handle].blkidx[ncnt] = i;
        /* copy values from global array to local buffers */
        SPA[handle].blksize[ncnt] = 0;
        while(j >= 0) {
          if (icnt >= total_vals) {
            printf("p[%ld] ICNT: %ld exceeds NVALS: %ld\n",me,icnt,total_vals);
          }
          memcpy(((char*)vbuf+icnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
          ibuf[icnt] = idx[j];
          jbuf[icnt] = jdx[j];
          j = list[j];
          SPA[handle].blksize[ncnt]++;
          icnt++;
        }
        ncnt++;
      }
    }
    free(count);
    free(top);
    /* Values have all been sorted into column blocks within the row block. Now
     * need to sort them by row. */
    nrows = SPA[handle].ihi - SPA[handle].ilo + 1;
    /* Calculate number of row values that will need to be stored. Add an
     * extra row to account for the total number of rows in the column block */
    for (i=0; i<nproc; i++) {
      offset[i] = 0;
      map[i] = 0;
    }
    offset[me] = (nrows+1)*SPA[handle].nblocks;
    pnga_pgroup_gop(SPA[handle].grp,C_LONG,offset,nproc,op);
    /* Construct g_i */
    map[0] = 1;
    total_vals = offset[0];
    for (i=1; i<nproc; i++) {
      map[i] = map[i-1] + offset[i-1];
      total_vals += offset[i];
    }
    SPA[handle].g_i = pnga_create_handle();
    if (longidx) {
      pnga_set_data(SPA[handle].g_i,one,&total_vals,C_LONG);
    } else {
      pnga_set_data(SPA[handle].g_i,one,&total_vals,C_INT);
    }
    pnga_set_irreg_distr(SPA[handle].g_i,map,&nproc);
    if (!pnga_allocate(SPA[handle].g_i)) return 0;
    pnga_distribution(SPA[handle].g_i,me,&lo,&hi);
    if (lo <= hi) {
      if (longidx) {
        pnga_access_ptr(SPA[handle].g_i,&lo,&hi,&ildx,&ld);
      } else {
        pnga_access_ptr(SPA[handle].g_i,&lo,&hi,&isdx,&ld);
      }
    } else {
      ildx = NULL;
      isdx = NULL;
    }
    pnga_distribution(SPA[handle].g_j,me,&lo,&hi);
    if (lo <= hi) {
      if (longidx) {
        jldx = NULL;
        pnga_access_ptr(SPA[handle].g_j,&lo,&hi,&jldx,&ld);
      } else {
        pnga_access_ptr(SPA[handle].g_j,&lo,&hi,&jsdx,&ld);
      }
    } else {
      jldx = NULL;
      jsdx = NULL;
    }
    free(vals);
    if (lo <= hi) {
      pnga_access_ptr(SPA[handle].g_data,&lo,&hi,&vals,&ld);
      /* Bin up elements by row index for each column block */
      count = (Integer*)malloc(nrows*sizeof(Integer));
      top = (Integer*)malloc(nrows*sizeof(Integer));
      ncnt = 0;
      icnt = 0;
      jcnt = 0;
      row_nnz = (Integer*)malloc(nrows*sizeof(Integer));
      for (i=0; i<nrows; i++) row_nnz[i] = 0;
      for (i=0; i<SPA[handle].nblocks; i++) {
        char *vptr = vals+ncnt*SPA[handle].size;
        Integer *ibuf = SPA[handle].idx + ncnt;
        Integer *jbuf = SPA[handle].jdx + ncnt;
        for (j=0; j<nrows; j++) {
          top[j] = -1;
          count[j] = 0;
        }
        for (j=0; j<SPA[handle].blksize[i]; j++) {
          /* ibuf and ilo are zero based */
          irow = ibuf[j]-SPA[handle].ilo;
          if (irow < 0 || irow >= nrows) {
            printf("p[%ld] (activate_relu) irow: %ld idx: %ld ilo: %ld"
                " nrows: %ld out of bounds\n",me,irow,ibuf[j],SPA[handle].ilo,nrows);
          }
          list[j] = top[irow];
          top[irow] = j;
          count[irow]++;
        }
        /* copy elements back into  global arrays after sorting by row index.
         * Offset values in g_i are relative to the start of the column block.
         */
        icnt = 0;
        char *vbuf = (char*)SPA[handle].val+ncnt*SPA[handle].size;
        if (longidx) {
          for (j=0; j<nrows; j++) {
            irow = top[j];
            (ildx+i*(nrows+1))[j] = (int64_t)icnt;
            while (irow >= 0) {
              jldx[jcnt] = (int64_t)jbuf[irow];
              memcpy((vptr+icnt*elemsize),(vbuf+irow*elemsize),(size_t)elemsize);
              //            printf("p[%d] SORT i: %d j: %d val: %f icnt: %d jcnt: %d\n",
              //                pnga_nodeid(),irow+ilo,jldx[jcnt],((float*)vptr)[icnt],
              //                (int)icnt,(int)jcnt);
              irow = list[irow];
              icnt++;
              jcnt++;
              row_nnz[j]++;
            }
          }
          if (icnt > 0) (ildx+i*(nrows+1))[nrows] = (int64_t)icnt;
        } else {
          for (j=0; j<nrows; j++) {
            irow = top[j];
            (isdx+i*(nrows+1))[j] = (int)icnt;
            while (irow >= 0) {
              jsdx[jcnt] = (int)jbuf[irow];
              memcpy((vptr+icnt*elemsize),(vbuf+irow*elemsize),(size_t)elemsize);
              irow = list[irow];
              icnt++;
              jcnt++;
              row_nnz[j]++;
            }
          }
          if (icnt > 0) (isdx+i*(nrows+1))[nrows] = (int)icnt;
        }
        SPA[handle].offset[i] = ncnt;
        ncnt += SPA[handle].blksize[i];
      }
      /* Find maximum number of non-zeros per row on this processors */
      max_nnz = 0;
      for (i=0; i<nrows; i++) {
        if (max_nnz < (int64_t)row_nnz[i]) max_nnz = (int64_t)row_nnz[i];
      }
      if (row_nnz != NULL) free(row_nnz);
    } else {
      count = NULL;
      top = NULL;
      row_nnz = NULL;
    }
    pnga_pgroup_gop(SPA[handle].grp,C_LONG,&max_nnz,1,"max");
    SPA[handle].max_nnz = (Integer)max_nnz;
    free(SPA[handle].val);
    free(SPA[handle].idx);
    free(SPA[handle].jdx);
    SPA[handle].val = NULL;
    SPA[handle].idx = NULL;
    SPA[handle].jdx = NULL;
    /* set up g_blk */
    for (i=0; i<nproc; i++) {
      Integer jbot, jtop;
      int iblk;
      /* find offset for row block on this processor
       * in g_j (should be the same for g_data */
      pnga_distribution(SPA[handle].g_j,me,&jbot,&jtop);

      /* calculate column limits for processor i */
      jlo = (SPA[handle].jdim*i)/nproc;
      while ((jlo*nproc)/jdim < i) {
        jlo++;
      }
      while ((jlo*nproc)/jdim > i) {
        jlo--;
      }
      if (i < nproc-1) {
        jhi = (SPA[handle].jdim*(i+1))/nproc;
        while ((jhi*nproc)/jdim < i+1) {
          jhi++;
        }
        while ((jhi*nproc)/jdim > i+1) {
          jhi--;
        }
        jhi--;
      } else {
        jhi = SPA[handle].jdim-1;
      }
      /* set indices to unit based indexing */
      jlo++;
      jhi++;
      /* find index for block in blksize and offset arrays */
      iblk = -1;
      for (j=0; j<SPA[handle].nblocks; j++) {
        if (SPA[handle].blkidx[j] == i) {
          iblk = j;
          break;
        }
      }
      if (iblk != -1) {
        /* block has data */
        row_info[i*7  ] = SPA[handle].ilo+1;
        row_info[i*7+1] = SPA[handle].ihi+1;
        row_info[i*7+2] = jlo;
        row_info[i*7+3] = jhi;
        row_info[i*7+4] = jbot+SPA[handle].offset[iblk];
        row_info[i*7+5] = row_info[i*7+4]+SPA[handle].blksize[iblk]-1;
      } else {
        /* block contains no data */
        row_info[i*7  ] = SPA[handle].ilo+1;
        row_info[i*7+1] = SPA[handle].ihi+1;
        row_info[i*7+2] = jlo;
        row_info[i*7+3] = jhi;
        if (i == 0) {
          row_info[4] = 1;
          row_info[5] = 0;
        } else {
          row_info[i*7+4] = row_info[(i-1)*7+4]+1;
          row_info[i*7+5] = row_info[i*7+4]-1;
        }
      }
      row_info[i*7+6] = iblk;
    }
    /* copy data in row info to g_blk */
    {
      Integer tlo[3], thi[3], tld[2];
      tlo[0] = 1;
      tlo[1] = me+1;
      tlo[2] = 1;
      thi[0] = 7;
      thi[1] = me+1;
      thi[2] = nproc;
      tld[0] = 7;
      tld[1] = 1;
      pnga_put(SPA[handle].g_blk,tlo,thi,row_info,tld);
      pnga_pgroup_sync(SPA[handle].grp);
      /* pnga_print(g_blk); */
    }
    free(active);
    free(compress);
    free(totals);
    free(map);
    free(list);
    free(row_info);
    free(offset);
    if (count != NULL) free(count);
    if (top != NULL) free(top);
  }
  /* Clean up utility arrays */
  free(idx);
  free(jdx);
  return ret;
}
#undef SPRS_REAL_ACTIVATE_M

#define SPRS_REAL_FILTER_M(_type,_idx_type,_vptr)           \
  {                                                         \
    _type *_ptr = (_type*)_vptr;                            \
    _type _val;                                             \
    _idx_type _ii, _jj, _ld;                                \
    Integer _idx, _jdx;                                     \
    _ld = (_idx_type)(hi[0]-lo[0]+1);                       \
    for (j=lo[1]; j<=hi[1]; j++) {                          \
      _jj = (_idx_type)(j-lo[1]);                           \
      for (i=lo[0]; i<=hi[0]; i++) {                        \
        _ii = (_idx_type)(i-lo[0]);                         \
        _val = _ptr[_ii+_jj*_ld];                           \
        if (_val != (_type)0 || i==j) {                     \
          if (trans) {                                      \
            _idx = (Integer)j;                              \
            _jdx =(Integer) i;                              \
          } else {                                          \
            _idx = (Integer)i;                              \
            _jdx = (Integer)j;                              \
          }                                                 \
          _idx--;                                           \
          _jdx--;                                           \
          pnga_sprs_array_add_element(s_a,_idx,_jdx,&_val); \
        }                                                   \
      }                                                     \
    }                                                       \
  }

#define SPRS_COMPLEX_FILTER_M(_type,_idx_type,_vptr)        \
  {                                                         \
    _type *_ptr = (_type*)_vptr;                            \
    _type _val[2];                                          \
    _idx_type _ii, _jj, _ld;                                \
    Integer _idx, _jdx;                                     \
    _ld = (_idx_type)(hi[0]-lo[0]+1);                       \
    for (j=lo[1]; j<=hi[1]; j++) {                          \
      _jj = (_idx_type)(j-lo[1]);                           \
      for (i=lo[0]; i<=hi[0]; i++) {                        \
        _ii = (_idx_type)(i-lo[0]);                         \
        _val[0] = _ptr[2*(_ii+_jj*_ld)];                    \
        _val[1] = _ptr[2*(_ii+_jj*_ld)+1];                  \
        if (_val[0] != (_type)0 || _val[1] != (_type)0      \
            || i==j) {                                      \
          if (trans) {                                      \
            _idx = (Integer)j;                              \
            _jdx =(Integer) i;                              \
          } else {                                          \
            _idx = (Integer)i;                              \
            _jdx = (Integer)j;                              \
          }                                                 \
          _idx--;                                           \
          _jdx--;                                           \
          pnga_sprs_array_add_element(s_a,_idx,_jdx,_val);  \
        }                                                   \
      }                                                     \
    }                                                       \
  }

/**
 * Create sparse array from dense array
 * @param g_a dense array handle
 * @param size of indices in sparse array
 * @param trans flag used for C-interface
 * @return handle of sparse array
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_create_from_dense = pnga_sprs_array_create_from_dense
#endif
Integer pnga_sprs_array_create_from_dense(Integer g_a, Integer idx_size,
    Integer trans)
{
  Integer handle = g_a + GA_OFFSET, s_a;
  Integer i, j, idx, jdx, lo[2], hi[2], ld;
  Integer idim, jdim;
  int grp = GA[handle].p_handle;
  int me = (int)pnga_pgroup_nodeid(grp);
  void *vptr;
  int type = GA[handle].type;

  /* Check dimension */
  if (GA[handle].ndim != 2) {
    pnga_error("(ga_sprs_array_create_from_dense) global array"
        " must be of dimension 2",0);
  }
  if (trans) {
    idim = (Integer)GA[handle].dims[1];
    jdim = (Integer)GA[handle].dims[0];
  } else {
    idim = (Integer)GA[handle].dims[0];
    jdim = (Integer)GA[handle].dims[1];
  }

  /* create sparse array */
  s_a = pnga_sprs_array_create(idim, jdim, type, idx_size);
  /* find bounding indices of locally owned block from global array */
  pnga_distribution(g_a,me,lo,hi);
  /* get pointer to locally owned block */
  pnga_access_ptr(g_a, lo, hi, &vptr, &ld);
  /* add all non-zero entries from g_a to s_a. Add diagonal elements
   * to s_a even if they are zero */
  if (idx_size == 4) {
    if (type == C_INT) {
      SPRS_REAL_FILTER_M(int,int,vptr);
    } else if (type == C_LONG) {
      SPRS_REAL_FILTER_M(long,int,vptr);
    } else if (type == C_LONGLONG) {
      SPRS_REAL_FILTER_M(long long,int,vptr);
    } else if (type == C_FLOAT) {
      SPRS_REAL_FILTER_M(float,int,vptr);
    } else if (type == C_DBL) {
      SPRS_REAL_FILTER_M(double,int,vptr);
    } else if (type == C_SCPL) {
      SPRS_COMPLEX_FILTER_M(float,int,vptr);
    } else if (type == C_DCPL) {
      SPRS_COMPLEX_FILTER_M(double,int,vptr);
    } else {
      pnga_error("(ga_sprs_array_create_from_dense) Unknown datatype"
          " encountered",type);
    }
  } else {
    if (type == C_INT) {
      SPRS_REAL_FILTER_M(int,int64_t,vptr);
    } else if (type == C_LONG) {
      SPRS_REAL_FILTER_M(long,int64_t,vptr);
    } else if (type == C_LONGLONG) {
      SPRS_REAL_FILTER_M(long long,int64_t,vptr);
    } else if (type == C_FLOAT) {
      SPRS_REAL_FILTER_M(float,int64_t,vptr);
    } else if (type == C_DBL) {
      SPRS_REAL_FILTER_M(double,int64_t,vptr);
    } else if (type == C_SCPL) {
      SPRS_COMPLEX_FILTER_M(float,int64_t,vptr);
    } else if (type == C_DCPL) {
      SPRS_COMPLEX_FILTER_M(double,int64_t,vptr);
    } else {
      pnga_error("(ga_sprs_array_create_from_dense) Unknown datatype"
          " encountered",type);
    }
  }
  if (!pnga_sprs_array_assemble(s_a)) {
    pnga_error("(ga_sprs_array_create_from_dense) failed to create"
        " sparse array from dense array",0);
  }
  return s_a;
}
#undef SPRS_REAL_FILTER_M
#undef SPRS_COMPLEX_FILTER_M

/**
 * Create dense array from sparse array
 * @param s_a sparse array handle
 * @param trans flag used for C-interface
 * @return handle of dense array
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_create_from_sparse = pnga_sprs_array_create_from_sparse
#endif
Integer pnga_sprs_array_create_from_sparse(Integer s_a,
    Integer trans)
{
  Integer *map;
  Integer handle = GA_OFFSET + s_a;
  Integer nprocs = pnga_pgroup_nnodes(SPA[handle].grp);
  Integer me = pnga_pgroup_nodeid(SPA[handle].grp);
  Integer g_a;
  Integer dims[2];
  Integer two = 2;
  Integer nblock[2];
  Integer idx_size = SPA[handle].idx_size;
  Integer ilo, ihi;
  Integer lo[2], hi[2], ld;
  Integer i, j, iblock;
  Integer elemsize = SPA[handle].size;
  void *g_data;
  void *s_data;
  char plus[2];

  /* allocate and fill map array to create new GA */
  map = (Integer*)malloc((nprocs+1)*sizeof(Integer));
  for (i=0; i<nprocs+1; i++) map[i] = 0;
  if (trans) {
    map[me+1] = SPA[handle].ilo+1;
  } else {
    map[me+1] = SPA[handle].ilo+1;
  }
  plus[0] = '+';
  plus[1] = '\0';
  if (sizeof(Integer) == 4) {
    pnga_pgroup_gop(SPA[handle].grp, C_INT, map, nprocs+1, plus);
  } else {
    pnga_pgroup_gop(SPA[handle].grp, C_LONG, map, nprocs+1, plus);
  }
  if (trans) {
    map[0] = 1;
  } else {
    map[nprocs] = 1;
  }
  g_a = pnga_create_handle();
  dims[0] = SPA[handle].idim;
  dims[1] = SPA[handle].jdim;
  pnga_set_data(g_a,two,dims,SPA[handle].type);
  if (trans) {
    nblock[0] = 1;
    nblock[1] = nprocs;
  } else {
    nblock[0] = nprocs;
    nblock[1] = 1;
  }
  pnga_set_irreg_distr(g_a, map, nblock);
  if (!pnga_allocate(g_a)) {
    pnga_error("(ga_sprs_array_create_from_sparse) failed to create"
        " dense array",0);
  }
  free(map);
  /* Initialize global array to zero */
  pnga_zero(g_a);
  /* get limits of locally held row block */
  ilo = SPA[handle].ilo;
  ihi = SPA[handle].ihi;
  if (trans) {
    lo[0] = 1;
    hi[0] = SPA[handle].idim;
    lo[1] = ilo+1;
    hi[1] = ihi+1;
  } else {
    lo[0] = ilo+1;
    hi[0] = ihi+1;
    lo[1] = 1;
    hi[1] = SPA[handle].jdim;
  }
  pnga_access_ptr(g_a,lo,hi,&g_data,&ld);
  /* loop over all column blocks in row block */
  for (iblock=0; iblock<SPA[handle].nblocks; iblock++) {
    void *iptr, *jptr;
    int64_t *lidx, *ljdx;
    int *sidx, *sjdx;
    Integer idx, jdx, n;
    Integer jlo, jhi;
    n = SPA[handle].blkidx[iblock];
    pnga_sprs_array_column_distribution(s_a,n,&jlo,&jhi);
    pnga_sprs_array_access_col_block(s_a,n,&iptr,&jptr,&s_data);
//  printf("p[%d] Got to 1 ilo: %d ihi: %d jlo: %d jhi: %d\n",me,ilo,ihi,jlo,jhi);
    for (i=ilo; i<=ihi; i++) {
      Integer ncols;
      if (idx_size == 4) {
        sidx = (int*)iptr;
        sjdx = (int*)jptr;
        ncols =  (Integer)(sidx[i+1-ilo]-sidx[i-ilo]);
//  printf("p[%d] Got to 2 ncols: %d\n",me,ncols);
        for (j=0; j<ncols; j++) {
          jdx = sjdx[sidx[i-ilo]+j];
//  printf("p[%d] Got to 3 jdx: %d elemsize: %d\n",me,jdx,elemsize);
          if (trans) {
            memcpy(((char*)g_data+elemsize*((i-lo[1]+1)*ld+jdx-lo[0]+1)),
                ((char*)s_data+elemsize*(sidx[i-ilo]+j)),
                (size_t)elemsize);
          }  else {
            memcpy(((char*)g_data+elemsize*(i-lo[0]+1+ld*(jdx-lo[1]+1))),
                ((char*)s_data+elemsize*(sidx[i-ilo]+j)),
                (size_t)elemsize);
          }
        }
//  printf("p[%d] Got to 4 jdx: %d elemsize: %d\n",me,jdx,elemsize);
      } else {
        lidx = (int64_t*)iptr;
        ljdx = (int64_t*)jptr;
        ncols =  (Integer)(lidx[i+1-ilo]-lidx[i-ilo]);
        for (j=0; j<ncols; j++) {
          jdx = ljdx[lidx[i-ilo]+j];
          if (trans) {
            memcpy(((char*)g_data+elemsize*((i-lo[1]+1)*ld+jdx-lo[0]+1)),
                ((char*)s_data+elemsize*(lidx[i-ilo]+j)),
                (size_t)elemsize);
          }  else {
            memcpy(((char*)g_data+elemsize*(i-lo[0]+1+ld*(jdx-lo[1]+1))),
                ((char*)s_data+elemsize*(lidx[i-ilo]+j)),
                (size_t)elemsize);
          }
        }
      }
    }
  }
  pnga_release(g_a,lo,hi); 
  pnga_pgroup_sync(SPA[handle].grp);
  return g_a;
}

#define REAL_SPRSDNS_MULTIPLY_M(_type,_ilo_a,_ihi_a,_jlo_a,_jhi_a,   \
    _lo_b,_hi_b,_iptr_a,_jptr_a,_ptr_a,_ptr_b,_ptr_c,_ld_c)          \
{                                                                    \
  Integer _i, _j, _k, _ii, _jj, _kk;                                 \
  Integer _ld_b = _hi_b-_lo_b+1;                                     \
  for (_i=_ilo_a; _i<=_ihi_a; _i++) {                                \
    _ii = _i - _ilo_a;                                               \
    Integer _ncols = _iptr_a[_ii+1]-_iptr_a[_ii];                    \
    for (_k=0; _k<_ncols; _k++) {                                    \
      _kk = _jptr_a[_iptr_a[_ii]+_k]-_jlo_a;                         \
      if (trans) {                                                   \
        for (_j=_lo_b; _j<=_hi_b; _j++) {                            \
          ((_type*)_ptr_c)[(_j-_lo_b) + (_i-_ilo_a)*_ld_c]           \
            += ((_type*)_ptr_a)[_iptr_a[_ii]+_k]                     \
            *((_type*)_ptr_b)[_kk*_ld_b+(_j-_lo_b)];                 \
        }                                                            \
      } else {                                                       \
        for (_j=_lo_b; _j<=_hi_b; _j++) {                            \
          ((_type*)_ptr_c)[_ld_c*(_j-_lo_b) + (_i-_ilo_a)]           \
            += ((_type*)_ptr_a)[_iptr_a[_ii]+_k]                     \
            *((_type*)_ptr_b)[_kk+(_j-_lo_b)*_ld_b];                 \
        }                                                            \
      }                                                              \
    }                                                                \
  }                                                                  \
}

#define COMPLEX_SPRSDNS_MULTIPLY_M(_type,_ilo_a,_ihi_a,_jlo_a,_jhi_a,\
    _lo_b,_hi_b,_iptr_a,_jptr_a,_ptr_a,_ptr_b,_ptr_c,_ld_c)          \
{                                                                    \
  Integer _i, _j, _k, _ii, _jj, _kk;                                 \
  Integer _ld_b = _hi_b-_lo_b+1;                                     \
  for (_i=_ilo_a; _i<=_ihi_a; _i++) {                                \
    _ii = _i - _ilo_a;                                               \
    Integer _ncols = _iptr_a[_ii+1]-_iptr_a[_ii];                    \
    for (_k=0; _k<_ncols; _k++) {                                    \
      _kk = _jptr_a[_iptr_a[_ii]+_k]-_jlo_a;                         \
      if (trans) {                                                   \
        for (_j=_lo_b; _j<=_hi_b; _j++) {                            \
          _type _ra, _ia, _rb, _ib, _rc, _ic;                        \
          _ra = ((_type*)_ptr_a)[2*(_iptr_a[_ii]+_k)];               \
          _ia = ((_type*)_ptr_a)[2*(_iptr_a[_ii]+_k)+1];             \
          _rb = ((_type*)_ptr_b)[2*(_kk*_ld_b+(_j-_lo_b))];          \
          _ib = ((_type*)_ptr_b)[2*(_kk*_ld_b+(_j-_lo_b))+1];        \
          ((_type*)_ptr_c)[2*((_j-_lo_b) + (_i-_ilo_a)*_ld_c)]       \
            += _ra*_rb-_ia*_ib;                                      \
          ((_type*)_ptr_c)[2*((_j-_lo_b) + (_i-_ilo_a)*_ld_c)+1]     \
            += _ia*_rb+_ra*_ib;                                      \
        }                                                            \
      } else {                                                       \
        for (_j=_lo_b; _j<=_hi_b; _j++) {                            \
          _type _ra, _ia, _rb, _ib, _rc, _ic;                        \
          _ra = ((_type*)_ptr_a)[2*(_iptr_a[_ii]+_k)];               \
          _ia = ((_type*)_ptr_a)[2*(_iptr_a[_ii]+_k)+1];             \
          _rb = ((_type*)_ptr_b)[2*(_kk+(_j-_lo_b)*_ld_b)];          \
          _ib = ((_type*)_ptr_b)[2*(_kk+(_j-_lo_b)*_ld_b)+1];        \
          ((_type*)_ptr_c)[2*((_j-_lo_b)*_ld_c + (_i-_ilo_a))]       \
            += _ra*_rb-_ia*_ib;                                      \
          ((_type*)_ptr_c)[2*((_j-_lo_b)*_ld_c + (_i-_ilo_a))+1]     \
            += _ia*_rb+_ra*_ib;                                      \
        }                                                            \
      }                                                              \
    }                                                                \
  }                                                                  \
}

/**
 * Multiply sparse matrix A times dense matrix B to get dense matrix C
 * C = A.B
 * @param s_a sparse array A
 * @param g_b dense array B
 * @param trans dense matrices are transposed. Needed for C-interface
 * @return handle of dense array C
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_sprsdns_multiply =  pnga_sprs_array_sprsdns_multiply
#endif
Integer pnga_sprs_array_sprsdns_multiply(Integer s_a, Integer g_b, Integer trans)
{
  Integer lcnt;
  Integer icnt;
  Integer hdl_a = s_a+GA_OFFSET;
  Integer hdl_b = g_b+GA_OFFSET;
  Integer hdl_c;
  Integer elemsize;
  Integer idim, jdim;
  Integer i, j, k, l, m, n, nn;
  Integer nprocs = pnga_pgroup_nnodes(SPA[hdl_a].grp);
  Integer me = pnga_pgroup_nodeid(SPA[hdl_a].grp);
  Integer longidx;
  Integer type;
  Integer jhi, jlo;
  Integer gdims[2], blocks[2];
  Integer rowdim;
  Integer *count;
  Integer nblocks;
  Integer g_c;
  Integer two = 2;
  Integer ilen;
  Integer max_nnz_a, max_nnz_b;
  Integer *row_nnz;
  int64_t max_nnz;
  Integer *map, *size;
  /* Do some initial verification to see if matrix multiply is possible */
  if (SPA[hdl_a].type != GA[hdl_b].type) {
    pnga_error("(ga_sprs_array_sprsdns_multiply) types of sparse matrices"
    " A and B must match",0);
  }
  type = SPA[hdl_a].type;
  if (SPA[hdl_a].grp != GA[hdl_b].p_handle) {
    pnga_error("(ga_sprs_array_sprsdns_multiply) matrices A and B must"
    " be on the same group",0);
  }
  if (GA[hdl_b].ndim != 2) {
    pnga_error("(ga_sprs_array_sprsdns_multiply) matrix B must be"
    " of dimension 2",GA[hdl_b].ndim);
  }
  if ((trans && SPA[hdl_a].jdim != GA[hdl_b].dims[1]) ||
      (!trans && SPA[hdl_a].jdim != GA[hdl_b].dims[0])) {
    pnga_error("(ga_sprs_array_sprsdns_multiply) column dimension of"
      " A must match row dimension of B",0);
  }
  if (SPA[hdl_a].idx_size == sizeof(int64_t)) {
    longidx = 1;
  } else {
    longidx = 0;
  }
  elemsize = SPA[hdl_a].size;
  idim = SPA[hdl_a].idim;
  if (trans) {
    jdim = GA[hdl_b].dims[0];
  } else {
    jdim = GA[hdl_b].dims[1];
  }
  /* Construct product array C*/
  g_c = pnga_create_handle();
  hdl_c = g_c + GA_OFFSET;
  if (trans) {
    gdims[0] = jdim;
    gdims[1] = idim;
  } else {
    gdims[0] = idim;
    gdims[1] = jdim;
  }
  pnga_set_data(g_c,two,gdims,type);
  map = (Integer*)malloc((nprocs+1)*sizeof(Integer));
  size = (Integer*)malloc(nprocs*sizeof(Integer));
  for (i=0; i<nprocs; i++) size[i] = 0;
  size[me] = SPA[hdl_a].ihi-SPA[hdl_a].ilo+1;
  if (sizeof(Integer) == 8) {
    pnga_pgroup_gop(SPA[hdl_a].grp,C_LONG,size,nprocs,"+");
  } else {
    pnga_pgroup_gop(SPA[hdl_a].grp,C_INT,size,nprocs,"+");
  }
  if (trans) {
    map[0] = 1;
    map[1] = 1;
    for (i=1; i<nprocs; i++) {
      map[i+1] = map[i] + size[i-1];
    }
    blocks[0] = 1;
    blocks[1] = nprocs;
  } else {
    map[0] = 1;
    for (i=1; i<nprocs; i++) {
      map[i] = map[i-1] + size[i-1];
    }
    map[nprocs] = 1;
    blocks[0] = nprocs;
    blocks[1] = 1;
  }
  pnga_set_irreg_distr(g_c,map,blocks);
  pnga_set_pgroup(g_c,SPA[hdl_a].grp);
  if (!pnga_allocate(g_c)) {
    pnga_error("(ga_sprs_array_sprsdns_multiply) could not allocate"
      " product array C",0);
  }
  pnga_zero(g_c);
  /* loop over processors in row to get target block and then loop over
   * processors to get all block pairs that contribute to target block.
   * Multiply block pairs */
  for (l=0; l<nprocs; l++) {
    /* Find bounding dimensions (unit-based) of target block in row */
    void *data_c;
    void *data_a, *data_b;
    Integer lo_c[2], hi_c[2], ld_c[2];
    Integer tlo_c[2], thi_c[2];
    lo_c[0] = SPA[hdl_a].ilo+1;
    hi_c[0] = SPA[hdl_a].ihi+1;
    pnga_sprs_array_column_distribution(s_a,l,&lo_c[1],&hi_c[1]);
    lo_c[1]++;
    hi_c[1]++;
    /* loop over all sub-blocks in column and row block that contribute
     * to target block */
    if (trans) {
      tlo_c[0] = lo_c[1];
      thi_c[0] = hi_c[1];
      tlo_c[1] = lo_c[0];
      thi_c[1] = hi_c[0];
    } else {
      tlo_c[0] = lo_c[0];
      thi_c[0] = hi_c[0];
      tlo_c[1] = lo_c[1];
      thi_c[1] = hi_c[1];
    }
    /* loop over non-zero blocks in row in sparse matrix A */
    for (nn=0; nn<SPA[hdl_a].nblocks; nn++) {
      void *iptr_a, *jptr_a;
      Integer ilo_a, ihi_a, jlo_a, jhi_a;
      Integer lo_b[2], hi_b[2], ld_b[2];
      Integer tlo_b[2], thi_b[2];
      void *val_a;
      void *buf_b;
      Integer nelem;
      Integer ii,jlen;
      n = SPA[hdl_a].blkidx[nn];
      /* calculate values of jlo and jhi for block in g_b */
      if (trans) {
        ld_c[0] = jdim;
        ld_c[1] = thi_c[1]-tlo_c[1]+1;
      } else {
        ld_c[0] = thi_c[0]-tlo_c[0]+1;
        ld_c[1] = jdim;
      }
      pnga_access_ptr(g_c,tlo_c,thi_c,&data_c,ld_c);
      /* Find block of A for multiplication */
      ilo_a = lo_c[0]-1;
      ihi_a = hi_c[0]-1;
      pnga_sprs_array_column_distribution(s_a, n, &jlo_a, &jhi_a);
      /* get pointer to sparse block */
      pnga_sprs_array_access_col_block(s_a, n, &iptr_a, &jptr_a, &val_a);
      lo_b[0] = jlo_a+1;
      hi_b[0] = jhi_a+1;
      lo_b[1] = lo_c[1];
      hi_b[1] = hi_c[1];
      if (trans) {
        tlo_b[0] = lo_b[1];
        thi_b[0] = hi_b[1];
        tlo_b[1] = lo_b[0];
        thi_b[1] = hi_b[0];
      } else {
        tlo_b[0] = lo_b[0];
        thi_b[0] = hi_b[0];
        tlo_b[1] = lo_b[1];
        thi_b[1] = hi_b[1];
      }
      /* Copy block from dense array into buffer */
      nelem = (thi_b[0]-tlo_b[0]+1)*(thi_b[1]-tlo_b[1]+1);
      buf_b = malloc(nelem*SPA[hdl_a].size);
      ld_b[0] = (thi_b[0]-tlo_b[0]+1);
      ld_b[1] = (thi_b[1]-tlo_b[1]+1);
      pnga_get(g_b,tlo_b,thi_b,buf_b,ld_b);
      if (SPA[hdl_a].idx_size == 4) {
        int *idx_a = (int*)iptr_a;
        int *jdx_a = (int*)jptr_a;
        data_a = val_a;
        /* now have access to blocks from A, B, and C. Perform block
         * multiplication */
        if (type == C_INT) {
          REAL_SPRSDNS_MULTIPLY_M(int,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],thi_b[0],
              idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_LONG) {
          REAL_SPRSDNS_MULTIPLY_M(long,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],thi_b[0],
              idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_LONGLONG) {
          REAL_SPRSDNS_MULTIPLY_M(long long,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],
              thi_b[0],idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_FLOAT) {
          REAL_SPRSDNS_MULTIPLY_M(float,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],thi_b[0],
              idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_DBL) {
          REAL_SPRSDNS_MULTIPLY_M(double,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],thi_b[0],
              idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_SCPL) {
          COMPLEX_SPRSDNS_MULTIPLY_M(float,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],
              thi_b[0],idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_DCPL) {
          COMPLEX_SPRSDNS_MULTIPLY_M(double,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],
              thi_b[0],idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        }
      } else {
        int64_t *idx_a = (int64_t*)iptr_a;
        int64_t *jdx_a = (int64_t*)jptr_a;
        data_a = val_a;
        /* now have access to blocks from A, B, and C. Perform block
         * multiplication */
        if (type == C_INT) {
          REAL_SPRSDNS_MULTIPLY_M(int,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],thi_b[0],
              idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_LONG) {
          REAL_SPRSDNS_MULTIPLY_M(long,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],thi_b[0],
              idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_LONGLONG) {
          REAL_SPRSDNS_MULTIPLY_M(long long,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],
              thi_b[0],idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_FLOAT) {
          REAL_SPRSDNS_MULTIPLY_M(float,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],thi_b[0],
              idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_DBL) {
          REAL_SPRSDNS_MULTIPLY_M(double,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],
              thi_b[0],idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_SCPL) {
          COMPLEX_SPRSDNS_MULTIPLY_M(float,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],
              thi_b[0],idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        } else if (type == C_DCPL) {
          COMPLEX_SPRSDNS_MULTIPLY_M(double,ilo_a,ihi_a,jlo_a,jhi_a,tlo_b[0],
              thi_b[0],idx_a,jdx_a,data_a,buf_b,data_c,ld_c[0]);
        }
      }
      free(buf_b);
    }
  }
  return g_c;
}
#undef REAL_SPRSDNS_MULTIPLY_M
#undef COMPLEX_SPRSDNS_MULTIPLY_M

#define REAL_DNSSPRS_MULTIPLY_M(_type,_lo_a,_hi_a,_ilo_b,_ihi_b,     \
    _jlo_b,_jhi_b,_iptr_b,_jptr_b,_ptr_a,_ptr_b,_ptr_c,_ld_c)        \
{                                                                    \
  Integer _i, _j, _k, _ii, _jj, _kk;                                 \
  Integer _ld_a, _alo, _ahi;                                         \
  if (trans) {                                                       \
    _ld_a = _hi_a[0]-_lo_a[0]+1;                                     \
    _alo = _lo_a[1];                                                 \
    _ahi = _hi_a[1];                                                 \
  } else {                                                           \
    _ld_a = _hi_a[0]-_lo_a[0]+1;                                     \
    _alo = _lo_a[0];                                                 \
    _ahi = _hi_a[0];                                                 \
  }                                                                  \
  for (_i=_alo; _i<=_ahi; _i++) {                                    \
    _ii = _i - _alo;                                                 \
    for (_k=_ilo_b; _k<=_ihi_b; _k++) {                              \
      _kk = _k - _ilo_b;                                             \
      Integer _ncols = _iptr_b[_kk+1]-_iptr_b[_kk];                  \
      if (trans) {                                                   \
        for (_j=0; _j<_ncols; _j++) {                                \
          _jj = _jptr_b[_iptr_b[_kk]+_j]-_jlo_b+1;                   \
          ((_type*)_ptr_c)[_jj + _ii*_ld_c]                          \
          += ((_type*)_ptr_a)[_ii*_ld_a+_kk]                         \
          *((_type*)_ptr_b)[_iptr_b[_kk]+_j];                        \
        }                                                            \
      } else {                                                       \
        for (_j=0; _j<_ncols; _j++) {                                \
          _jj = _jptr_b[_iptr_b[_kk]+_j]-_jlo_b+1;                   \
          ((_type*)_ptr_c)[_jj*_ld_c + _ii]                          \
          += ((_type*)_ptr_a)[_ii+_kk*_ld_a]                         \
          *((_type*)_ptr_b)[_iptr_b[_kk]+_j];                        \
        }                                                            \
      }                                                              \
    }                                                                \
  }                                                                  \
}

#define COMPLEX_DNSSPRS_MULTIPLY_M(_type,_lo_a,_hi_a,_ilo_b,_ihi_b,  \
    _jlo_b,_jhi_b,_iptr_b,_jptr_b,_ptr_a,_ptr_b,_ptr_c,_ld_c)        \
{                                                                    \
  Integer _i, _j, _k, _ii, _jj, _kk;                                 \
  Integer _ld_a, _alo, _ahi;                                         \
  if (trans) {                                                       \
    _ld_a = _hi_a[0]-_lo_a[0]+1;                                     \
    _alo = _lo_a[1];                                                 \
    _ahi = _hi_a[1];                                                 \
  } else {                                                           \
    _ld_a = _hi_a[0]-_lo_a[0]+1;                                     \
    _alo = _lo_a[0];                                                 \
    _ahi = _hi_a[0];                                                 \
  }                                                                  \
  for (_i=_alo; _i<=_ahi; _i++) {                                    \
    _ii = _i - _alo;                                                 \
    for (_k=_ilo_b; _k<=_ihi_b; _k++) {                              \
      _kk = _k - _ilo_b;                                             \
      Integer _ncols = _iptr_b[_kk+1]-_iptr_b[_kk];                  \
      if (trans) {                                                   \
        for (_j=0; _j<_ncols; _j++) {                                \
          _type _ra, _ia, _rb, _ib, _rc, _ic;                        \
          _jj = _jptr_b[_iptr_b[_kk]+_j]-_jlo_b+1;                   \
          _ra = ((_type*)_ptr_a)[2*(_ii*_ld_a+_kk)];                 \
          _ia = ((_type*)_ptr_a)[2*(_ii*_ld_a+_kk)+1];               \
          _rb = ((_type*)_ptr_b)[2*(_iptr_b[_kk]+_j)];               \
          _ib = ((_type*)_ptr_b)[2*(_iptr_b[_kk]+_j)+1];             \
          ((_type*)_ptr_c)[2*(_jj + _ii*_ld_c)]                      \
          += _ra*_rb-_ia*_ib;                                        \
          ((_type*)_ptr_c)[2*(_jj + _ii*_ld_c)+1]                    \
          += _ia*_rb+_ra*_ib;                                        \
        }                                                            \
      } else {                                                       \
        for (_j=0; _j<_ncols; _j++) {                                \
          _type _ra, _ia, _rb, _ib, _rc, _ic;                        \
          _jj = _jptr_b[_iptr_b[_kk]+_j]-_jlo_b+1;                   \
          _ra = ((_type*)_ptr_a)[2*(_ii+_kk*_ld_a)];                 \
          _ia = ((_type*)_ptr_a)[2*(_ii*_kk*_ld_a)+1];               \
          _rb = ((_type*)_ptr_b)[2*(_iptr_b[_kk]+_j)];               \
          _ib = ((_type*)_ptr_b)[2*(_iptr_b[_kk]+_j)+1];             \
          ((_type*)_ptr_c)[2*(_jj*_ld_c + _ii)]                      \
          += _ra*_rb-_ia*_ib;                                        \
          ((_type*)_ptr_c)[2*(_jj*_ld_c + _ii)+1]                    \
          += _ia*_rb+_ra*_ib;                                        \
        }                                                            \
      }                                                              \
    }                                                                \
  }                                                                  \
}

/**
 * Multiply sparse matrix A times dense matrix B to get dense matrix C
 * C = A.B
 * @param g_a dense array A
 * @param s_b sparse array B
 * @param trans dense matrices are transposed. Needed for C-interface
 * @return handle of dense array C
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_dnssprs_multiply =  pnga_sprs_array_dnssprs_multiply
#endif
Integer pnga_sprs_array_dnssprs_multiply(Integer g_a, Integer s_b, Integer trans)
{
  Integer lcnt;
  Integer icnt;
  Integer hdl_a = g_a+GA_OFFSET;
  Integer hdl_b = s_b+GA_OFFSET;
  Integer hdl_c;
  Integer elemsize;
  Integer idim, jdim;
  Integer i, j, k, l, m, n, nn;
  Integer nprocs = pnga_pgroup_nnodes(SPA[hdl_b].grp);
  Integer me = pnga_pgroup_nodeid(SPA[hdl_b].grp);
  Integer longidx;
  Integer type;
  Integer ihi, ilo, jhi, jlo;
  Integer gdims[2], blocks[2];
  Integer rowdim;
  Integer *count;
  Integer nblocks;
  Integer g_c;
  Integer two = 2;
  Integer ilen;
  Integer max_nnz_a, max_nnz_b;
  Integer *row_nnz;
  int64_t max_nnz;
  Integer *map, *size;
  /* Do some initial verification to see if matrix multiply is possible */
  if (GA[hdl_a].type != SPA[hdl_b].type) {
    pnga_error("(ga_sprs_array_dnssprs_multiply) types of sparse matrices"
    " A and B must match",0);
  }
  type = SPA[hdl_b].type;
  if (GA[hdl_a].p_handle != SPA[hdl_b].grp) {
    pnga_error("(ga_sprs_array_dnssprs_multiply) matrices A and B must"
    " be on the same group",0);
  }
  if (GA[hdl_a].ndim != 2) {
    pnga_error("(ga_sprs_array_dnssprs_multiply) matrix A must be"
    " of dimension 2",GA[hdl_a].ndim);
  }
  if ((trans && GA[hdl_a].dims[0] != SPA[hdl_b].jdim) ||
      (!trans && GA[hdl_a].dims[1] != SPA[hdl_b].jdim)) {
    pnga_error("(ga_sprs_array_dnssprs_multiply) column dimension of"
      " A must match row dimension of B",0);
  }
  if (SPA[hdl_b].idx_size == sizeof(int64_t)) {
    longidx = 1;
  } else {
    longidx = 0;
  }
  elemsize = SPA[hdl_b].size;
  if (trans) {
    idim = GA[hdl_a].dims[1];
  } else {
    idim = GA[hdl_a].dims[0];
  }
  jdim = SPA[hdl_b].jdim;
  /* Construct product array C*/
  g_c = pnga_create_handle();
  hdl_c = g_c + GA_OFFSET;
  if (trans) {
    gdims[0] = jdim;
    gdims[1] = idim;
  } else {
    gdims[0] = idim;
    gdims[1] = jdim;
  }
  pnga_set_data(g_c,two,gdims,type);
  map = (Integer*)malloc((nprocs+1)*sizeof(Integer));
  size = (Integer*)malloc(nprocs*sizeof(Integer));
  /* Create a partition for dense C matrix */
  ilo = (idim*me)/nprocs;
  while ((ilo*nprocs)/idim < me) {
    ilo++;
  }
  while ((ilo*nprocs)/idim > me) {
    ilo--;
  }
  if ((ilo*nprocs)/idim != me) {
    ilo++;
  }
  ilo++;
  if (me < nprocs-1) {
    ihi = (idim*(me+1))/nprocs;
    while ((ihi*nprocs)/idim < me+1) {
      ihi++;
    }
    while ((ihi*nprocs)/idim > me+1) {
      ihi--;
    }
    ihi--;
  } else {
    ihi = idim-1;
  }
  ihi++;

  for (i=0; i<nprocs; i++) size[i] = 0;
  size[me] = ihi-ilo+1;
  if (sizeof(Integer) == 8) {
    pnga_pgroup_gop(SPA[hdl_b].grp,C_LONG,size,nprocs,"+");
  } else {
    pnga_pgroup_gop(SPA[hdl_b].grp,C_INT,size,nprocs,"+");
  }
  if (trans) {
    map[0] = 1;
    map[1] = 1;
    for (i=1; i<nprocs; i++) {
      map[i+1] = map[i] + size[i-1];
    }
    blocks[0] = 1;
    blocks[1] = nprocs;
  } else {
    map[0] = 1;
    for (i=1; i<nprocs; i++) {
      map[i] = map[i-1] + size[i-1];
    }
    map[nprocs] = 1;
    blocks[0] = nprocs;
    blocks[1] = 1;
  }
  pnga_set_irreg_distr(g_c,map,blocks);
  pnga_set_pgroup(g_c,SPA[hdl_b].grp);
  if (!pnga_allocate(g_c)) {
    pnga_error("(ga_sprs_array_sprsdns_multiply) could not allocate"
      " product array C",0);
  }
  pnga_zero(g_c);
  /* loop over processors in row to get target block and then loop over
   * processors to get all block pairs that contribute to target block.
   * Multiply block pairs */
  for (l=0; l<nprocs; l++) {
    /* Find bounding dimensions (unit-based) of target block in row */
    void *data_c;
    void *data_a, *data_b, *iptr, *jptr;
    Integer lo_c[2], hi_c[2], ld_c[2];
    Integer tlo_c[2], thi_c[2];
    lo_c[0] = ilo;
    hi_c[0] = ihi;
    pnga_sprs_array_row_distribution(s_b,l,&lo_c[1],&hi_c[1]);
    lo_c[1]++;
    hi_c[1]++;
    if (trans) {
      tlo_c[0] = lo_c[1];
      thi_c[0] = hi_c[1];
      tlo_c[1] = lo_c[0];
      thi_c[1] = hi_c[0];
    } else {
      tlo_c[0] = lo_c[0];
      thi_c[0] = hi_c[0];
      tlo_c[1] = lo_c[1];
      thi_c[1] = hi_c[1];
    }
    /* loop over all sub-blocks in row block of A and column block of B
     * that contribute to target block */
    for (n=0; n<nprocs; n++) {
      void *iptr_b, *jptr_b;
      Integer lo_a[2], hi_a[2], ld_a[2];
      Integer tlo_a[2], thi_a[2];
      Integer ilo_b, ihi_b, jlo_b, jhi_b;
      void *buf_a;
      void *val_b;
      Integer nelem;
      Integer ii,jlen;
      /* calculate values of jlo and jhi for block in g_b */
      if (trans) {
        ld_c[0] = jdim;
        ld_c[1] = thi_c[1]-tlo_c[1]+1;
      } else {
        ld_c[0] = thi_c[0]-tlo_c[0]+1;
        ld_c[1] = jdim;
      }
      pnga_access_ptr(g_c,tlo_c,thi_c,&data_c,ld_c);
      /* Get block of B for multiplication */
      if (!pnga_sprs_array_get_block(s_b,n,l,&iptr_b,&jptr_b,&val_b,
            &ilo_b,&ihi_b,&jlo_b,&jhi_b)) continue;
      /* get pointers to dense blocks on A and C */
      lo_a[0] = lo_c[0];
      hi_a[0] = hi_c[0];
      lo_a[1] = ilo_b;
      hi_a[1] = ihi_b;
      if (trans) {
        tlo_a[0] = lo_a[1];
        thi_a[0] = hi_a[1];
        tlo_a[1] = lo_a[0];
        thi_a[1] = hi_a[0];
      } else {
        tlo_a[0] = lo_a[0];
        thi_a[0] = hi_a[0];
        tlo_a[1] = lo_a[1];
        thi_a[1] = hi_a[1];
      }
      /* Copy block from dense array into buffer */
      nelem = (thi_a[0]-tlo_a[0]+1)*(thi_a[1]-tlo_a[1]+1);
      buf_a = malloc(nelem*SPA[hdl_b].size);
      ld_a[0] = (thi_a[0]-tlo_a[0]+1);
      ld_a[1] = (thi_a[1]-tlo_a[1]+1);
      pnga_get(g_a,tlo_a,thi_a,buf_a,ld_a);
      if (SPA[hdl_b].idx_size == 4) {
        int *idx_b = (int*)iptr_b;
        int *jdx_b = (int*)jptr_b;
        data_b = val_b;

        /* now have access to blocks from A, B, and C. Perform block
         * multiplication */
        if (type == C_INT) {
          REAL_DNSSPRS_MULTIPLY_M(int,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,jhi_b,
              idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_LONG) {
          REAL_DNSSPRS_MULTIPLY_M(long,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,jhi_b,
              idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_LONGLONG) {
          REAL_DNSSPRS_MULTIPLY_M(long long,tlo_a,thi_a,ilo_b,ihi_b,
              jlo_b,jhi_b,idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_FLOAT) {
          REAL_DNSSPRS_MULTIPLY_M(float,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,jhi_b,
              idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_DBL) {
          REAL_DNSSPRS_MULTIPLY_M(double,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,jhi_b,
              idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_SCPL) {
          COMPLEX_DNSSPRS_MULTIPLY_M(float,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,
              jhi_b,idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_DCPL) {
          COMPLEX_DNSSPRS_MULTIPLY_M(double,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,
              jhi_b,idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        }
      } else {
        int64_t *idx_b = (int64_t*)iptr_b;
        int64_t *jdx_b = (int64_t*)jptr_b;
        data_b = val_b;
        /* now have access to blocks from A, B, and C. Perform block
         * multiplication */
        if (type == C_INT) {
          REAL_DNSSPRS_MULTIPLY_M(int,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,jhi_b,
              idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
#if 0
{                                                                    
  Integer _i, _j, _k, _ii, _jj, _kk;                                 
  Integer _ld_a = thi_a[0]-tlo_a[0]+1;                                     
  for (_i=tlo_a[1]; _i<=thi_a[1]; _i++) {                                  
    _ii = _i - tlo_a[1];                                                
    for (_k=ilo_b; _k<=ihi_b; _k++) {                              
      _kk = _k - ilo_b;                                             
      Integer _ncols = idx_b[_kk+1]-idx_b[_kk];                  
      if (trans) {                                                   
        for (_j=0; _j<_ncols; _j++) {                                
          _jj = jdx_b[idx_b[_kk]+_j]-jlo_b+1;                   
          if (_ii*_ld_a+_kk >= nelem || _ii*_ld_a+_kk < 0) {
            printf("p[%d] OOB: ii: %d ld_a: %d kk: %d nelem: %d alo[0]: %d ahi[0]: %d alo[1]: %d ahi[1]: %d\n",
                me,_ii,_ld_a,_kk,nelem,tlo_a[0],thi_a[0],tlo_a[1],thi_a[1]);
          }
          ((int*)data_c)[_jj + _ii*ld_c[0]]                          
          += ((int*)buf_a)[_ii*_ld_a+_kk]                         
          *((int*)data_b)[idx_b[_kk]+_j];                        
        }                                                            
      } else {                                                       
        for (_j=0; _j<_ncols; _j++) {                                
          _jj = jdx_b[idx_b[_kk]+_j]-jlo_b+1;                   
          ((int*)data_c)[_jj*ld_c[0] + _ii]                          
          += ((int*)buf_a)[_ii+_kk*_ld_a]                         
          *((int*)data_b)[idx_b[_kk]+_j];                        
        }                                                            
      }                                                              
    }                                                                
  }                                                                  
}
#endif
        } else if (type == C_LONG) {
          REAL_DNSSPRS_MULTIPLY_M(long,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,jhi_b,
              idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_LONGLONG) {
          REAL_DNSSPRS_MULTIPLY_M(long long,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,
              jhi_b,idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_FLOAT) {
          REAL_DNSSPRS_MULTIPLY_M(float,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,jhi_b,
              idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_DBL) {
          REAL_DNSSPRS_MULTIPLY_M(double,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,jhi_b,
              idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_SCPL) {
          COMPLEX_DNSSPRS_MULTIPLY_M(float,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,
              jhi_b,idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        } else if (type == C_DCPL) {
          COMPLEX_DNSSPRS_MULTIPLY_M(double,tlo_a,thi_a,ilo_b,ihi_b,jlo_b,
              jhi_b,idx_b,jdx_b,buf_a,data_b,data_c,ld_c[0]);
        }
      }
      free(buf_a);
    }
  }
  return g_c;
}
#undef REAL_DNSSPRS_MULTIPLY_M
#undef COMPLEX_DNSSPRS_MULTIPLY_M

#define SPRS_REAL_ELEMENTWISE_MULTIPLY_M(_type)                       \
{                                                                     \
  Integer _i, _j, _k, _ii, _jj;                                       \
  for (_i=ilo; _i<=ihi; _i++) {                                       \
    /* create hash map to store j-indices of matrix B row */          \
    Integer *_jtop;                                                   \
    Integer *_jlist;                                                  \
    Integer _ncols_a, _ncols_b;                                       \
    _ii = _i-ilo;                                                     \
    _ncols_a = idx_a[_ii+1]-idx_a[_ii];                               \
    _ncols_b = idx_b[_ii+1]-idx_b[_ii];                               \
    if (_ncols_b == 0 || _ncols_a == 0) continue;                     \
    _jtop = (Integer*)malloc(_ncols_b*sizeof(Integer));               \
    _jlist = (Integer*)malloc(_ncols_b*sizeof(Integer));              \
    for (_j=0; _j<_ncols_b; _j++) _jtop[_j] = -1;                     \
    for (_j=0; _j<_ncols_b; _j++) {                                   \
      _jj = jdx_b[idx_b[_ii]+_j];                                     \
      _k = _jj%_ncols_b;                                              \
      _jlist[_j] = _jtop[_k];                                         \
      _jtop[_k] = _j;                                                 \
    }                                                                 \
    /* scan through all non-zero elements of row in A and see if      \
     * there is a corresponding j-index for B */                      \
    for (_j=0; _j<_ncols_a; _j++) {                                   \
      Integer _kk = -1;                                               \
      _jj = jdx_a[idx_a[_ii]+_j];                                     \
      /* Find out if there is a corresponding non-zero entry in B */  \
      _k = _jj%_ncols_b;                                              \
      _k = _jtop[_k];                                                 \
      _kk = -1;                                                       \
      while (_k > -1) {                                               \
        if (jdx_b[idx_b[_ii]+_k] == _jj) {                            \
          _kk = _k;                                                   \
          break;                                                      \
        }                                                             \
        _k = _jlist[_k];                                              \
      }                                                               \
      /* Found matching j-indices. Do multiplication and store product\
       * in list */                                                   \
      if (_kk > -1) {                                                 \
        _type _val_a = ((_type*)data_a)[idx_a[_ii]+_j];               \
        _type _val_b = ((_type*)data_b)[idx_b[_ii]+_kk];              \
        ((_type*)data)[lcnt] = _val_a*_val_b;                         \
        idx[lcnt] = _i;                                               \
        jdx[lcnt] = _jj;                                              \
        ldx = (_i*jdim+_jj)%bufsize;                                  \
        if (lcnt >= bufsize) {                                        \
          printf("p[%d] LCNT out of range: %d bufsize: %d\n",me,      \
              lcnt,bufsize);                                          \
        }                                                             \
        list[lcnt] = top[ldx];                                        \
        top[ldx] = lcnt;                                              \
        lcnt++;                                                       \
      }                                                               \
    }                                                                 \
    free(_jtop);                                                      \
    free(_jlist);                                                     \
  }                                                                   \
}

#define SPRS_COMPLEX_ELEMENTWISE_MULTIPLY_M(_type)                    \
{                                                                     \
  Integer _i, _j, _k, _ii, _jj;                                       \
  for (_i=ilo; _i<=ihi; _i++) {                                       \
    /* create hash map to store j-indices of matrix B row */          \
    Integer *_jtop;                                                   \
    Integer *_jlist;                                                  \
    Integer _ncols_a, _ncols_b;                                       \
    _ii = _i-ilo;                                                     \
    _ncols_a = idx_a[_ii+1]-idx_a[_ii];                               \
    _ncols_b = idx_b[_ii+1]-idx_b[_ii];                               \
    if (_ncols_b == 0 || _ncols_a == 0) continue;                     \
    _jtop = (Integer*)malloc(_ncols_b*sizeof(Integer));               \
    _jlist = (Integer*)malloc(_ncols_b*sizeof(Integer));              \
    for (_j=0; _j<_ncols_b; _j++) _jtop[_j] = -1;                     \
    for (_j=0; _j<_ncols_b; _j++) {                                   \
      _jj = jdx_b[idx_b[_ii]+_j];                                     \
      _k = _jj%_ncols_b;                                              \
      _jlist[_j] = _jtop[_k];                                         \
      _jtop[_k] = _j;                                                 \
    }                                                                 \
    /* scan through all non-zero elements of row in A and see if      \
     * there is a corresponding j-index for B */                      \
    for (_j=0; _j<_ncols_a; _j++) {                                   \
      Integer _kk = -1;                                               \
      _jj = jdx_a[idx_a[_ii]+_j];                                     \
      /* Find out if there is a corresponding non-zero entry in B */  \
      _k = _jj%_ncols_b;                                              \
      _k = _jtop[_k];                                                 \
      _kk = -1;                                                       \
      while (_k > -1) {                                               \
        if (jdx_b[idx_b[_ii]+_k] == _jj) {                            \
          _kk = _k;                                                   \
          break;                                                      \
        }                                                             \
        _k = _jlist[_k];                                              \
      }                                                               \
      /* Found matching j-indices. Do multiplication and store product\
       * in list */                                                   \
      if (_kk > -1) {                                                 \
        _type _rval_a = ((_type*)data_a)[2*(idx_a[_ii]+_j)];          \
        _type _ival_a = ((_type*)data_a)[2*(idx_a[_ii]+_j)+1];        \
        _type _rval_b = ((_type*)data_b)[2*(idx_b[_ii]+_kk)];         \
        _type _ival_b = ((_type*)data_b)[2*(idx_b[_ii]+_kk)+1];       \
        ((_type*)data)[2*lcnt] = _rval_a*_rval_b-_ival_a*_ival_b;     \
        ((_type*)data)[2*lcnt+1] = _rval_a*_ival_b+_ival_a*_rval_b;   \
        idx[lcnt] = _i;                                               \
        jdx[lcnt] = _jj;                                              \
        ldx = (_i*jdim+_jj)%bufsize;                                  \
        if (lcnt >= bufsize) {                                        \
          printf("p[%d] LCNT out of range: %d bufsize: %d\n",me,      \
              lcnt,bufsize);                                          \
        }                                                             \
        list[lcnt] = top[ldx];                                        \
        top[ldx] = lcnt;                                              \
        lcnt++;                                                       \
      }                                                               \
    }                                                                 \
    free(_jtop);                                                      \
    free(_jlist);                                                     \
  }                                                                   \
}

/**
 * Multiply sparse matrices A and B elementwise to get sparse matrix C
 * @param s_a sparse array A
 * @param s_b sparse array B
 * @return handle of sparse array C
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_elementwise_multiply = pnga_sprs_array_elementwise_multiply
#endif
Integer pnga_sprs_array_elementwise_multiply(Integer s_a, Integer s_b)
{
  Integer *top;
  Integer *list;
  Integer *idx;
  Integer *jdx;
  Integer lcnt;
  Integer icnt;
  Integer hdl_a = s_a+GA_OFFSET;
  Integer hdl_b = s_b+GA_OFFSET;
  Integer hdl_c;
  Integer bufsize;
  Integer elemsize;
  Integer idim, jdim;
  Integer i, j, k, l, m, n, ldx;
  void *data;
  Integer nprocs = pnga_pgroup_nnodes(SPA[hdl_a].grp);
  Integer me = pnga_pgroup_nodeid(SPA[hdl_a].grp);
  Integer longidx;
  Integer type;
  Integer ihi, ilo;
  Integer rowdim;
  Integer *count;
  Integer nblocks;
  Integer s_c;
  Integer ilen;
  Integer max_nnz_a, max_nnz_b;
  Integer *row_nnz;
  int64_t max_nnz;
  /* Do some initial verification to see if elementwise multiply is possible */
  if (SPA[hdl_a].type != SPA[hdl_b].type) {
    pnga_error("(ga_sprs_array_elementwise_multiply) types of sparse matrices"
    " A and B must match",0);
  }
  type = SPA[hdl_a].type;
  if (SPA[hdl_a].grp != SPA[hdl_b].grp) {
    pnga_error("(ga_sprs_array_elementwise_multiply) matrices A and B must"
    " be on the same group",0);
  }
  if (SPA[hdl_a].idim != SPA[hdl_b].idim || SPA[hdl_a].jdim != SPA[hdl_b].jdim) {
    pnga_error("(ga_sprs_array_elementwise_multiply) dimensions of"
      " A must match dimensions of B",0);
  }
  if (SPA[hdl_a].idx_size != SPA[hdl_b].idx_size) {
    pnga_error("(ga_sprs_array_elementwise_multiply) size of integer"
      " indices of A and B must match",0);
  }
  if (SPA[hdl_a].idx_size == sizeof(int64_t)) {
    longidx = 1;
  } else {
    longidx = 0;
  }
  /* Allocate initial buffers to hold elements of product matrix. Product
   * matrix must be at least as sparse as A and B. */
//#if ENABLE_PREALLOC
  {
    Integer nrows = SPA[hdl_a].ihi-SPA[hdl_a].ilo + 1;
    Integer nnz;
    max_nnz_a = SPA[hdl_a].max_nnz;
    max_nnz_b = SPA[hdl_b].max_nnz;
    nnz = max_nnz_a > max_nnz_b ? max_nnz_a : max_nnz_b;
    bufsize = nnz*(SPA[hdl_a].ihi-SPA[hdl_a].ilo+1);
  }
//#else
//  bufsize = INIT_BUF_SIZE;
//#endif
  elemsize = SPA[hdl_a].size;
  idim = SPA[hdl_a].idim;
  jdim = SPA[hdl_a].jdim;
  top = (Integer*)malloc(bufsize*sizeof(Integer));
  list = (Integer*)malloc(bufsize*sizeof(Integer));
  idx = (Integer*)malloc(bufsize*sizeof(Integer));
  jdx = (Integer*)malloc(bufsize*sizeof(Integer));
  data = malloc(bufsize*elemsize);
  for (i=0; i<bufsize; i++) top[i] = -1;
  for (i=0; i<bufsize; i++) list[i] = -1;
  lcnt = 0;
  /* loop over processors in row and then do elementwise
   * multiply of block pairs */
  for (l=0; l<nprocs; l++) {
    Integer *idx_a, *jdx_a, *idx_b, *jdx_b;
    void *data_a, *data_b, *iptr, *jptr;
    Integer ilo = SPA[hdl_a].ilo;
    Integer ihi = SPA[hdl_a].ihi;
    pnga_sprs_array_access_col_block(s_a, l, &iptr, &jptr, &data_a);
    idx_a = (Integer*)iptr;
    jdx_a = (Integer*)jptr;
    if (idx_a == NULL || jdx_a == NULL || data_a == NULL) continue;
    pnga_sprs_array_access_col_block(s_b, l, &iptr, &jptr, &data_b);
    idx_b = (Integer*)iptr;
    jdx_b = (Integer*)jptr;
    if (idx_b == NULL || jdx_b == NULL || data_b == NULL) continue;
    if (type == C_INT) {
      SPRS_REAL_ELEMENTWISE_MULTIPLY_M(int);
#if 0
      for (i=ilo; i<=ihi; i++) {
        /* create hash map to store j-indices of matrix B row */
        Integer *jtop;
        Integer *jlist;
        ii = i-ilo;
        ncols_a = idx_a[ii+1]-idx_a[ii];
        ncols_b = idx_b[ii+1]-idx_b[ii];
        if (ncols_b == 0 || ncols_a == 0) continue;
        jtop = (Integer*)malloc(ncols_b*sizeof(Integer));
        jlist = (Integer*)malloc(ncols_b*sizeof(Integer));
        for (j=0; j<ncols_b; j++) jtop[j] = -1;
        for (j=0; j<ncols_b; j++) {
          jj = jdx_b[idx_b[ii]+j];
          k = jj%ncols_b;
          jlist[j] = jtop[k];
          jtop[k] = j;
        }
        /* scan through all non-zero elements of row in A and see if
         * there is a corresponding j-index for B */
        for (j=0; j<ncols_a; j++) {
          Integer kk = -1;
          jj = jdx_a[idx_a[ii]+j];
          /* Find out if there is a corresponding non-zero entry in B */
          k = jj%ncols_b;
          k = jtop[k];
          kk = -1;
          while (k > -1) {
          printf("p[%d] l: %d i: %d j: %d j_a: %d j_b: %d\n",me,l,i,j,jj,jdx_b[idx_b[ii]+k]);
            if (jdx_b[idx_b[ii]+k] == jj) {
              kk = k;
              break;
            }
            k = jlist[k];
          }
          /* Found matching j-indices. Do multiplication and store product
           * in list */
          if (kk > -1) {
            int val_a = ((int*)data_a)[idx_a[ii]+j];
            int val_b = ((int*)data_b)[idx_b[ii]+kk];
            /* add new value to list */
//            if (lcnt == bufsize)
//              update_map(&top, &list, &idx, &jdx, &data, idim,
//                  jdim, elemsize, &bufsize, &lcnt);
            ((int*)data)[lcnt] = val_a*val_b;
            idx[lcnt] = i;
            jdx[lcnt] = jj;
            ldx = (i*jdim+jj)%bufsize;
            printf("p[%d] store lcnt: %d idx: %d jdx: %d data:%d\n",me,
                lcnt,idx[lcnt],jdx[lcnt],((int*)data)[lcnt]);
            if (ldx >= bufsize) {
              printf("p[%d] LDX out of range: %d bufsize: %d\n",me,ldx,bufsize);
            }
            if (lcnt >= bufsize) {
              printf("p[%d] LCNT out of range: %d bufsize: %d\n",me,lcnt,bufsize);
            }
            list[lcnt] = top[ldx];
            top[ldx] = lcnt;
            lcnt++;
          }
        }
        free(jtop);
        free(jlist);
      }
#endif
    } else if (type == C_LONG) {
      SPRS_REAL_ELEMENTWISE_MULTIPLY_M(long);
    } else if (type == C_LONGLONG) {
      SPRS_REAL_ELEMENTWISE_MULTIPLY_M(long long);
    } else if (type == C_FLOAT) {
      SPRS_REAL_ELEMENTWISE_MULTIPLY_M(float);
    } else if (type == C_DBL) {
      SPRS_REAL_ELEMENTWISE_MULTIPLY_M(double);
    } else if (type == C_SCPL) {
      SPRS_COMPLEX_ELEMENTWISE_MULTIPLY_M(float);
    } else if (type == C_DCPL) {
      SPRS_COMPLEX_ELEMENTWISE_MULTIPLY_M(double);
    }
  }

  /* At this point all blocks have been multiplied and the resulting
   * values are stored in the link list defined by top, list, idx,
   * jdx, data. No data needs to be moved, but all elements need to
   * be resorted into blocks and then into rows. Start by binning
   * data into column and row blocks. */
  free(top);
  for (i=0; i<bufsize; i++) list[i] = -1;
  ilo = SPA[hdl_a].ilo;
  ihi = SPA[hdl_a].ihi;
  rowdim = ihi-ilo+1;
  top = (Integer*)malloc(nprocs*sizeof(Integer));
  count = (Integer*)malloc(nprocs*sizeof(Integer));
  for (i=0; i<nprocs; i++) top[i] = -1;
  for (i=0; i<nprocs; i++) count[i] = 0;
  /* bin up all data elements into column blocks */
  for (i=0; i<lcnt; i++) {
    /* jdx is unit based so need to subtract 1 */
    Integer np = ((jdx[i]-1)*nprocs/jdim);
    if (np >= nprocs) np = nprocs-1;
    list[i] = top[np];
    top[np] = i;
    count[np]++;
  }
  /* count up number of column blocks with data */
  nblocks = 0;
  j = 0;
  for (i=0; i<nprocs; i++) {
    if (count[i] > 0) nblocks++;
    j += count[i];
  }

  /* create a new sparse array to hold product array */
  s_c = pnga_sprs_array_create(idim,jdim,type,SPA[hdl_a].idx_size);
  hdl_c = GA_OFFSET + s_c;
  free(SPA[hdl_c].idx);
  free(SPA[hdl_c].jdx);
  free(SPA[hdl_c].val);

  /* set up array of offsets */
  SPA[hdl_c].blkidx = (Integer*)malloc(nblocks*sizeof(Integer));
  SPA[hdl_c].blksize = (Integer*)malloc(nblocks*sizeof(Integer));
  SPA[hdl_c].offset = (Integer*)malloc(nblocks*sizeof(Integer));
  nblocks = 0;
  for (i=0; i<nprocs; i++) {
    if (count[i] > 0) {
      SPA[hdl_c].blkidx[nblocks] = i;
      SPA[hdl_c].blksize[nblocks] = count[i];
      nblocks++;
    }
  }
  SPA[hdl_c].offset[0] = 0;
  for (i=1; i<nblocks; i++) {
    SPA[hdl_c].offset[i] = SPA[hdl_c].offset[i-1]+SPA[hdl_c].blksize[i-1];
  }

  SPA[hdl_c].nblocks = nblocks;
  SPA[hdl_c].ilo = ilo;
  SPA[hdl_c].ihi = ihi;
  SPA[hdl_c].type = type;
  SPA[hdl_c].nprocs = SPA[hdl_a].nprocs;
  SPA[hdl_c].idx = NULL;
  SPA[hdl_c].jdx = NULL;
  SPA[hdl_c].val = NULL;
  SPA[hdl_c].nval = lcnt;
  SPA[hdl_c].maxval = bufsize;
  SPA[hdl_c].size = SPA[hdl_a].size;
  /* sort data into column blocks */
  SPA[hdl_c].idx = (Integer*)malloc(lcnt*sizeof(Integer));
  SPA[hdl_c].jdx = (Integer*)malloc(lcnt*sizeof(Integer));
  SPA[hdl_c].val = (Integer*)malloc(lcnt*SPA[hdl_c].size);
  icnt = 0;
  for (n=0; n<nprocs; n++) {
    Integer elemsize = SPA[hdl_c].size;
    i = top[n];
    while (i >= 0) {
      SPA[hdl_c].idx[icnt] = idx[i];
      SPA[hdl_c].jdx[icnt] = jdx[i];
      memcpy(&((char*)SPA[hdl_c].val)[icnt*elemsize],
          &((char*)data)[i*elemsize],elemsize);
      i = list[i];
      icnt++;
    }
  }
  free(idx);
  free(jdx);
  free(data);

  /* Set up global arrays to hold distributed indices and non-zero values */
  {
    int64_t isize = (rowdim+1)*nblocks;
    int64_t totalsize = 0;
    Integer ndim = 1;
    Integer *offset = (Integer*)malloc(nprocs*sizeof(Integer));
    Integer *tmp = (Integer*)malloc(nprocs*sizeof(Integer));
    Integer *map = (Integer*)malloc(nprocs*sizeof(Integer));
    /* set up array to hold row indices. Evaluate offsets for rows */
    for (i=0; i<nprocs; i++) {
      offset[i] = 0;
      tmp[i] = 0;
    }
    tmp[me] = isize;
    if (sizeof(Integer) == 8) {
      pnga_pgroup_gop(SPA[hdl_a].grp,C_LONG,tmp,nprocs,"+");
    } else {
      pnga_pgroup_gop(SPA[hdl_a].grp,C_INT,tmp,nprocs,"+");
    }
    offset[0] = 0;
    for (i=0; i<nprocs; i++) {
      totalsize += tmp[i];
      if (i>0) offset[i] = offset[i-1]+tmp[i-1];
      map[i] = offset[i]+1; /* 1-based indexing for map array */
    }

    SPA[hdl_c].g_i = pnga_create_handle();
    if (longidx) {
      pnga_set_data(SPA[hdl_c].g_i,ndim,&totalsize,C_LONG);
    } else {
      pnga_set_data(SPA[hdl_c].g_i,ndim,&totalsize,C_INT);
    }
    pnga_set_irreg_distr(SPA[hdl_c].g_i,map,&nprocs);
    pnga_allocate(SPA[hdl_c].g_i);
    /* set up arrays to hold column indices and data. Evaluate offsets
     * for row blocks*/
    for (i=0; i<nprocs; i++) {
      offset[i] = 0;
      tmp[i] = 0;
    }
    tmp[me] = lcnt;
    if (sizeof(Integer) == 8) {
      pnga_pgroup_gop(SPA[hdl_a].grp,C_LONG,tmp,nprocs,"+");
    } else {
      pnga_pgroup_gop(SPA[hdl_a].grp,C_INT,tmp,nprocs,"+");
    }
    offset[0] = 0;
    totalsize = 0;
    for (i=0; i<nprocs; i++) {
      totalsize += tmp[i];
      if (i>0) offset[i] = offset[i-1]+tmp[i-1];
      map[i] = offset[i]+1; /* 1-based indexing for map array */
    }
    SPA[hdl_c].g_j = pnga_create_handle();
    SPA[hdl_c].g_data = pnga_create_handle();
    if (longidx) {
      pnga_set_data(SPA[hdl_c].g_j,ndim,&totalsize,C_LONG);
    } else {
      pnga_set_data(SPA[hdl_c].g_j,ndim,&totalsize,C_INT);
    }
    pnga_set_data(SPA[hdl_c].g_data,ndim,&totalsize,SPA[hdl_c].type);
    pnga_set_irreg_distr(SPA[hdl_c].g_j,map,&nprocs);
    pnga_set_irreg_distr(SPA[hdl_c].g_data,map,&nprocs);
    pnga_allocate(SPA[hdl_c].g_j);
    pnga_allocate(SPA[hdl_c].g_data);
    free(map);
    free(tmp);
    free(offset);
  }
  /* organize row block into column blocks in CSR format */
  {
    int *ti, *tj;
    int64_t *lti, *ltj;
    void *tdata;
    Integer tlo, thi, tld;
    char *cdata, *ctdata;
    /* Get pointers to global arrays */
    pnga_distribution(SPA[hdl_c].g_i,me,&tlo,&thi);
    if (longidx) {
      pnga_access_ptr(SPA[hdl_c].g_i,&tlo,&thi,&lti,&tld);
    } else {
      pnga_access_ptr(SPA[hdl_c].g_i,&tlo,&thi,&ti,&tld);
    }
    pnga_distribution(SPA[hdl_c].g_j,me,&tlo,&thi);
    if (longidx) {
      pnga_access_ptr(SPA[hdl_c].g_j,&tlo,&thi,&ltj,&tld);
    } else {
      pnga_access_ptr(SPA[hdl_c].g_j,&tlo,&thi,&tj,&tld);
    }
    pnga_distribution(SPA[hdl_c].g_data,me,&tlo,&thi);
    pnga_access_ptr(SPA[hdl_c].g_data,&tlo,&thi,&tdata,&tld);
    idx = SPA[hdl_c].idx;
    jdx = SPA[hdl_c].jdx;
    cdata = (char*)SPA[hdl_c].val;
    ctdata = (char*)tdata;
    ilen = ihi - ilo + 1;
    row_nnz = (Integer*)malloc(ilen*sizeof(Integer));
    for (n=0; n<ilen; n++) row_nnz[n] = 0;
    /* loop over column blocks */
    for (n=0; n<nblocks; n++) {
      Integer *rowtop;
      Integer *rowlist;
      Integer irow;
      Integer offset_i;
      Integer offset_j;
      j = SPA[hdl_c].blkidx[n];
      /* sort blocks into rows */
      rowtop = (Integer*)malloc(ilen*sizeof(Integer));
      rowlist = (Integer*)malloc(count[j]*sizeof(Integer));
      for (i=0; i<ilen; i++) rowtop[i] = -1;
      for (i=0; i<count[j]; i++) rowlist[i] = -1;
      offset_i = n*(ilen+1);
      offset_j = SPA[hdl_c].offset[n];
      /* sort block into rows */
      icnt = 0;
      while (icnt < SPA[hdl_c].blksize[n]) {
        /* idx and ilo are zero based. Use offset_j since
         * data in idx is partitioned the same way as for jdx and data */
        Integer id = idx[icnt+offset_j]-ilo;
        if (id >= ilen || id < 0) {
          printf("p[%d] ID: %d ILEN: %d\n",me,id,ilen);
        }
        if (icnt >= count[j]) {
          printf("p[%d] ICNT: %d COUNT: %d\n",me,icnt,count[j]);
        }
        rowlist[icnt] = rowtop[id];
        rowtop[id] = icnt;
        icnt++;
      }

      /* now organize data in g_i, g_j, g_data */
      icnt = 0;
      if (longidx) {
        for (irow=0; irow<ilen; irow++) {
          Integer jd = rowtop[irow];
          lti[offset_i+irow] = icnt;
          while (jd >= 0) {
            ltj[offset_j+icnt] = (int64_t)jdx[offset_j+jd];
            memcpy(&ctdata[elemsize*(offset_j+icnt)],
                &cdata[(offset_j+jd)*elemsize],elemsize);
            row_nnz[irow]++;
            icnt++;
            jd = rowlist[jd];
          }
        }
        lti[offset_i+ilen] = icnt;
      } else {
        for (irow=0; irow<ilen; irow++) {
          Integer jd = rowtop[irow];
          ti[offset_i+irow] = icnt;
          while (jd >= 0) {
            tj[offset_j+icnt] = (int)jdx[offset_j+jd];
            memcpy(&ctdata[elemsize*(offset_j+icnt)],
                &cdata[(offset_j+jd)*elemsize],elemsize);
            icnt++;
            row_nnz[irow]++;
            jd = rowlist[jd];
          }
        }
        ti[offset_i+ilen] = icnt;
      }
      /* clean up arrays */
      free(rowtop);
      free(rowlist);
    }
    /* Find maximum number of non-zeros per row on this processors */
    max_nnz = 0;
    for (i=0; i<ilen; i++) {
      if (max_nnz < (int64_t)row_nnz[i]) max_nnz = (int64_t)row_nnz[i];
    }
    free(row_nnz);
    pnga_pgroup_gop(SPA[hdl_c].grp,C_LONG,&max_nnz,1,"max");
    SPA[hdl_c].max_nnz = (Integer)max_nnz;

    pnga_distribution(SPA[hdl_c].g_i,me,&tlo,&thi);
    pnga_release(SPA[hdl_c].g_i,&tlo,&thi);
    pnga_distribution(SPA[hdl_c].g_data,me,&tlo,&thi);
    pnga_release(SPA[hdl_c].g_data,&tlo,&thi);
    pnga_release(SPA[hdl_c].g_j,&tlo,&thi);
  }
  free(top);
  free(list);
  free(SPA[hdl_c].idx);
  free(SPA[hdl_c].jdx);
  free(SPA[hdl_c].val);
  SPA[hdl_c].idx = NULL;
  SPA[hdl_c].jdx = NULL;
  SPA[hdl_c].val= NULL;
  free(count);

  /* Create global array to store information on sparse blocks */
  {
    Integer dims[3], chunk[3];
    Integer three = 3;
    Integer g_blk;
    int64_t *row_info;
    Integer jlo, jhi;
    dims[0] = 7;
    dims[1] = nprocs;
    dims[2] = nprocs;
    chunk[0] = 7;
    chunk[1] = -1;
    chunk[2] = -1;
    /* g_blk contains information about how data for each sparse
     * block is laid out in g_j and g_data. The last two dimensions
     * describe location of sparse block in nproc X nproc array
     * of sparse blocks corresponding to original sparse matrix.
     *
     * First dimension contains the following information on each
     * block
     *    ilo: lowest row index of block
     *    ihi: highest row index of block
     *    jlo: lowest column index of block
     *    jhi: highest column index of block
     *    offset: offset in g_j and g_data for column indices and data
     *            values for block
     *    blkend: last index  g_j and g_data for block
     */
    g_blk = pnga_create_handle();
    pnga_set_pgroup(g_blk,SPA[hdl_c].grp);
    pnga_set_data(g_blk,three,dims,C_LONG);
    pnga_set_chunk(g_blk,chunk);
    if (!pnga_allocate(g_blk)) {
      pnga_error("(pnga_sprs_elementwise_multiply) Failure allocating g_blk",0);
    }
    SPA[hdl_c].g_blk = g_blk;
    row_info = (int64_t*)malloc(7*nprocs*sizeof(int64_t));
    /* set up g_blk */
    for (i=0; i<nprocs; i++) {
      Integer jlo, jhi;
      Integer jbot, jtop;
      int iblk;
      /* find offset for row block on this processor
       * in g_j (should be the same for g_data */
      pnga_distribution(SPA[hdl_c].g_j,me,&jbot,&jtop);

      /* calculate column limits for processor i */
      jlo = (SPA[hdl_c].jdim*i)/nprocs;
      while ((jlo*nprocs)/jdim < i) {
        jlo++;
      }
      while ((jlo*nprocs)/jdim > i) {
        jlo--;
      }
      if (i < nprocs-1) {
        jhi = (SPA[hdl_c].jdim*(i+1))/nprocs;
        while ((jhi*nprocs)/jdim < i+1) {
          jhi++;
        }
        while ((jhi*nprocs)/jdim > i+1) {
          jhi--;
        }
        jhi--;
      } else {
        jhi = SPA[hdl_c].jdim-1;
      }
      /* set indices to fortran indexing */
      jlo++;
      jhi++;
      /* find index for block in blksize and offset arrays */
      iblk = -1;
      for (j=0; j<SPA[hdl_c].nblocks; j++) {
        if (SPA[hdl_c].blkidx[j] == i) {
          iblk = j;
          break;
        }
      }
      if (iblk != -1) {
        /* block has data */
        row_info[i*7  ] = SPA[hdl_c].ilo+1;
        row_info[i*7+1] = SPA[hdl_c].ihi+1;
        row_info[i*7+2] = jlo;
        row_info[i*7+3] = jhi;
        row_info[i*7+4] = jbot+SPA[hdl_c].offset[iblk];
        row_info[i*7+5] = row_info[i*7+4]+SPA[hdl_c].blksize[iblk]-1;
      } else {
        /* block contains no data */
        row_info[i*7  ] = SPA[hdl_c].ilo+1;
        row_info[i*7+1] = SPA[hdl_c].ihi+1;
        row_info[i*7+2] = jlo;
        row_info[i*7+3] = jhi;
        if (i == 0) {
          row_info[4] = 1;
          row_info[5] = 0;
        } else {
          row_info[i*7+4] = row_info[(i-1)*7+4]+1;
          row_info[i*7+5] = row_info[(i-1)*7+4]-1;
        }
      }
      row_info[i*7+6] = iblk;
    }
    /* copy data in row info to g_blk */
    {
      Integer tlo[3], thi[3], tld[2];
      tlo[0] = 1;
      tlo[1] = me+1;
      tlo[2] = 1;
      thi[0] = 7;
      thi[1] = me+1;
      thi[2] = nprocs;
      tld[0] = 7;
      tld[1] = 1;
      pnga_put(g_blk,tlo,thi,row_info,tld);
      pnga_pgroup_sync(SPA[hdl_c].grp);
    }
    free(row_info);
  }
  SPA[hdl_c].ready = 1;
  return s_c;
}

#define SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(_type,_idx_type)                  \
{                                                                             \
  /* Loop over rows */                                                        \
  for (i=ilo_c; i<=ihi_c; i++) {                                              \
    _idx_type *_idx_c = (_idx_type*)iptr;                                     \
    _idx_type *_jdx_c = (_idx_type*)jptr;                                     \
    ncol = _idx_c[i+1-ilo_c]-_idx_c[i-ilo_c];                                 \
    /* loop over elements in row */                                           \
    for (j=0; j<ncol; j++) {                                                  \
      Integer _idx, _jdx, _jabs, _kdim;                                       \
      _idx = i-ilo_c;                                                         \
      _jdx = _jdx_c[_idx_c[_idx] + j];                                        \
      if (trans) {                                                            \
        _kdim = ihi_a-ilo_a+1;                                                \
        _jabs = _jdx - jlo_c;                                                 \
        /* now have coordinates of C in block. Calculate contribution from */ \
        /* multiplying corresponding rows in blocks of A and B */             \
        if (transB) {                                                         \
          for (k=0; k<_kdim; k++) {                                           \
            ((_type*)data_c)[_idx_c[_idx]+j] += ((_type*)data_b)[_jabs*ldb+k] \
            * ((_type*)data_a)[k+lda*_idx];                                   \
          }                                                                   \
        } else {                                                              \
          for (k=0; k<_kdim; k++) {                                           \
            ((_type*)data_c)[_idx_c[_idx]+j] += ((_type*)data_b)[_jabs+k*ldb] \
            * ((_type*)data_a)[k+lda*_idx];                                   \
          }                                                                   \
        }                                                                     \
      } else {                                                                \
        _kdim = jhi_a-jlo_a+1;                                                \
        _jabs = _jdx - jlo_c;                                                 \
        /* now have coordinates of C in block. Calculate contribution from */ \
        /* multiplying corresponding rows in blocks of A and B */             \
        for (k=0; k<_kdim; k++) {                                             \
          ((_type*)data_c)[_idx_c[_idx]+j] += ((_type*)data_a)[_jabs+k*lda]   \
          * ((_type*)data_b)[k+ldb*_idx];                                     \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }                                                                           \
}

#define SPRS_COMPLEX_SAMPLED_MATMAT_MULTIPLY_M(_type,_idx_type)               \
{                                                                             \
  /* Loop over rows */                                                        \
  for (i=ilo_c; i<=ihi_c; i++) {                                              \
    _idx_type *_idx_c = (_idx_type*)iptr;                                     \
    _idx_type *_jdx_c = (_idx_type*)jptr;                                     \
    ncol = _idx_c[i+1-ilo_c]-_idx_c[i-ilo_c];                                 \
    /* loop over elements in row */                                           \
    for (j=0; j<ncol; j++) {                                                  \
      Integer _idx, _jdx, _jabs, _kdim;                                       \
      _idx = i-ilo_c;                                                         \
      _jdx = _jdx_c[_idx_c[_idx] + j];                                        \
      if (trans) {                                                            \
        _kdim = ihi_a-ilo_a+1;                                                \
        _jabs = _jdx - jlo_c;                                                 \
        /* now have coordinates of C in block. Calculate contribution from */ \
        /* multiplying corresponding rows in blocks of A and B */             \
        if (transB) {                                                         \
          for (k=0; k<_kdim; k++) {                                           \
            _type _ra, _rb, _ia, _ib;                                         \
            _ra = ((_type*)data_a)[2*(k+lda*_idx)];                           \
            _ia = ((_type*)data_a)[2*(k+lda*_idx)+1];                         \
            _rb = ((_type*)data_b)[2*(_jabs*ldb+k)];                          \
            _ib = ((_type*)data_b)[2*(_jabs*ldb+k)+1];                        \
            ((_type*)data_c)[2*(_idx_c[_idx]+j)] += _ra*_rb-_ia*_ib;          \
            ((_type*)data_c)[2*(_idx_c[_idx]+j)+1] += _ra*_ib+_ia*_rb;        \
          }                                                                   \
        } else {                                                              \
          for (k=0; k<_kdim; k++) {                                           \
            _type _ra, _rb, _ia, _ib;                                         \
            _ra = ((_type*)data_a)[2*(k+lda*_idx)];                           \
            _ia = ((_type*)data_a)[2*(k+lda*_idx)+1];                         \
            _rb = ((_type*)data_b)[2*(_jabs+ldb*k)];                          \
            _ib = ((_type*)data_b)[2*(_jabs+ldb*k)+1];                        \
            ((_type*)data_c)[2*(_idx_c[_idx]+j)] += _ra*_rb-_ia*_ib;          \
            ((_type*)data_c)[2*(_idx_c[_idx]+j)+1] += _ra*_ib+_ia*_rb;        \
          }                                                                   \
        }                                                                     \
      } else {                                                                \
        _kdim = jhi_a-jlo_a+1;                                                \
        _jabs = _jdx - jlo_c;                                                 \
        /* now have coordinates of C in block. Calculate contribution from */ \
        /* multiplying corresponding rows in blocks of A and B */             \
        for (k=0; k<_kdim; k++) {                                             \
          _type _ra, _rb, _ia, _ib;                                           \
          _ra = ((_type*)data_a)[2*(_jabs+lda*k)];                            \
          _ia = ((_type*)data_a)[2*(_jabs+lda*k)+1];                          \
          _rb = ((_type*)data_b)[2*(k+ldb*_idx)];                             \
          _ib = ((_type*)data_b)[2*(k+ldb*_idx)+1];                           \
          ((_type*)data_c)[2*(_idx_c[_idx]+j)] += _ra*_rb-_ia*_ib;            \
          ((_type*)data_c)[2*(_idx_c[_idx]+j)+1] += _ra*_ib+_ia*_rb;          \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }                                                                           \
}

/**
 * Evaluate the product of dense matrices A and B at the non-zero values
 * of C. Values of C are overwritten
 * @param g_a dense array A
 * @param g_b dense array B
 * @param s_c sparse array C
 * @param transB use the transpose of B in evaluating the product
 * @param trans all matrices are transposed. Needed for C-interface
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_sampled_multiply = pnga_sprs_array_sampled_multiply
#endif
void pnga_sprs_array_sampled_multiply(Integer g_a, Integer g_b, Integer s_c,
    Integer transB, Integer trans)
{
  Integer hdl_a = g_a+GA_OFFSET;
  Integer hdl_b = g_b+GA_OFFSET;
  Integer hdl_c = s_c+GA_OFFSET;
  Integer elemsize;
  Integer idim, jdim;
  Integer i, j, k, l, m, n;
  Integer iblock;
  Integer nprocs = pnga_pgroup_nnodes(SPA[hdl_c].grp);
  Integer me = pnga_pgroup_nodeid(SPA[hdl_c].grp);
  Integer longidx;
  Integer type;
  Integer ncol;
  long *column_info;
  /* Do some initial verification to see if matrix multiply is possible */
  if (GA[hdl_a].type != GA[hdl_b].type) {
    pnga_error("(ga_sprs_array_sampled_multiply) types of dense matrices"
    " A and B must match",0);
  }
  if (SPA[hdl_c].type != GA[hdl_b].type) {
    pnga_error("(ga_sprs_array_sampled_multiply) types of sparse matrix"
    " C and dense matrix B must match",0);
  }
  type = GA[hdl_a].type;
  if (GA[hdl_a].p_handle != GA[hdl_b].p_handle) {
    pnga_error("(ga_sprs_array_sampled_multiply) matrices A and B must"
    " be on the same group",0);
  }
  if (SPA[hdl_c].grp != GA[hdl_b].p_handle) {
    pnga_error("(ga_sprs_array_sampled_multiply) matrices C and B must"
    " be on the same group",0);
  }
  if (GA[hdl_a].ndim != 2 || GA[hdl_b].ndim != 2) {
    pnga_error("(ga_sprs_array_sampled_multiply) dimension of matrices"
    " A and B must be 2",0);
  }
  if (trans) {
    if (transB) {
      if (GA[hdl_b].dims[0] != GA[hdl_a].dims[0]) {
        pnga_error("(ga_sprs_array_sampled_multiply) column dimension of"
            " A must match row dimension of B transpose",0);
      }
      if (SPA[hdl_c].idim != GA[hdl_a].dims[1] ||
          SPA[hdl_c].jdim != GA[hdl_b].dims[1]) {
        pnga_error("(ga_sprs_array_sampled_multiply) row dimension of"
            " C must match row dimension of A and column dimension of C must"
            " match column dimension of B transpose",0);
      }
    } else {
      if (GA[hdl_b].dims[1] != GA[hdl_a].dims[0]) {
        pnga_error("(ga_sprs_array_sampled_multiply) column dimension of"
            " A must match row dimension of B",0);
      }
      if (SPA[hdl_c].idim != GA[hdl_a].dims[1] ||
          SPA[hdl_c].jdim != GA[hdl_b].dims[0]) {
        pnga_error("(ga_sprs_array_sampled_multiply) row dimension of"
            " C must match row dimension of A and column dimension of C must"
            " match column dimension of B",0);
      }
    }
  } else {
    if (transB) {
      if (GA[hdl_a].dims[1] != GA[hdl_b].dims[1]) {
        pnga_error("(ga_sprs_array_sampled_multiply) column dimension of"
            " A must match row dimension of B transpose",0);
      }
      if (SPA[hdl_c].idim != GA[hdl_a].dims[0] ||
          SPA[hdl_c].jdim != GA[hdl_b].dims[0]) {
        pnga_error("(ga_sprs_array_sampled_multiply) row dimension of"
            " C must match row dimension of A and column dimension of C must"
            " match column dimension of B transpose",0);
      }
    } else {
      if (GA[hdl_a].dims[1] != GA[hdl_b].dims[0]) {
        pnga_error("(ga_sprs_array_sampled_multiply) column dimension of"
            " A must match row dimension of B",0);
      }
      if (SPA[hdl_c].idim != GA[hdl_a].dims[0] ||
          SPA[hdl_c].jdim != GA[hdl_b].dims[1]) {
        pnga_error("(ga_sprs_array_sampled_multiply) row dimension of"
            " C must match row dimension of A and column dimension of C must"
            " match column dimension of B",0);
      }
    }
  }
  if (SPA[hdl_c].idx_size == sizeof(int64_t)) {
    longidx = 1;
  } else {
    longidx = 0;
  }
  elemsize = GA[hdl_a].elemsize;
  idim = SPA[hdl_c].idim;
  jdim = SPA[hdl_c].jdim;

  /* get information on row partitions from g_blk */
  column_info = (long*)malloc(7*nprocs*sizeof(long));
  {
    Integer tlo[3], thi[3], tld[2];
    tlo[0] = 1;
    thi[0] = 7;
    tlo[1] = 1;
    thi[1] = nprocs;
    tlo[2] = me+1;
    thi[2] = me+1;
    tld[0] = 7;
    tld[1] = nprocs;
    pnga_get(SPA[hdl_c].g_blk,tlo,thi,column_info,tld);
  }
  /* initalize all data in C to zero */
  pnga_zero(SPA[hdl_c].g_data);
  /* loop over blocks in row that contain data and then loop over block pairs. */
  for (iblock=0; iblock<SPA[hdl_c].nblocks; iblock++) {
    Integer ilo_c, ihi_c, jlo_c, jhi_c;
    void *iptr, *jptr, *data_c;
    int64_t *lidx_c, *ljdx_c;
    int *sidx_c, *sjdx_c;
    n = SPA[hdl_c].blkidx[iblock];
    ilo_c = SPA[hdl_c].ilo;
    ihi_c = SPA[hdl_c].ihi;
    pnga_sprs_array_column_distribution(s_c,n,&jlo_c,&jhi_c);
    /* get pointers to data in C */
    pnga_sprs_array_access_col_block(s_c,n,&iptr,&jptr,&data_c);
    for (l=0; l<nprocs; l++) { /* loop over block pairs */
      Integer alo[2], ahi[2], blo[2], bhi[2], lda, ldb;
      void *data_a, *data_b;
      Integer asize, bsize;
      Integer ilo_a, ihi_a, jlo_a, jhi_a;
      Integer ilo_b, ihi_b, jlo_b, jhi_b;
      /* calculate bounding indices for A and B blocks */
      if (trans) {
        ilo_a = (Integer)(column_info[7*l]);
        ihi_a = (Integer)(column_info[7*l+1]);
        jlo_a = ilo_c;
        jhi_a = ihi_c;
        if (transB) {
          ilo_b = ilo_a;
          ihi_b = ihi_a;
          jlo_b = jlo_c;
          jhi_b = jhi_c;
        } else {
          ilo_b = jlo_c;
          ihi_b = jhi_c;
          jlo_b = ilo_a;
          jhi_b = ihi_a;
        }
      } else {
        ilo_a = ilo_c;
        ihi_a = ihi_c;
        jlo_a = (Integer)(column_info[7*l]);
        jhi_a = (Integer)(column_info[7*l+1]);
        if (transB) {
          ilo_b = jlo_c;
          ihi_b = jhi_c;
          jlo_b = jlo_a;
          jhi_b = jhi_a;
        } else {
          ilo_b = jlo_a;
          ihi_b = jhi_a;
          jlo_b = jlo_c;
          jhi_b = jhi_c;
        }
      }
      asize = (ihi_a-ilo_a+1)*(jhi_a-jlo_a+1);
      bsize = (ihi_b-ilo_b+1)*(jhi_b-jlo_b+1);
      if (trans) {
        alo[0] = ilo_a;
        ahi[0] = ihi_a;
        alo[1] = jlo_a+1;
        ahi[1] = jhi_a+1;
        if (transB) {
          blo[0] = ilo_b;
          bhi[0] = ihi_b;
          blo[1] = jlo_b+1;
          bhi[1] = jhi_b+1;
        } else {
          blo[0] = ilo_b+1;
          bhi[0] = ihi_b+1;
          blo[1] = jlo_b;
          bhi[1] = jhi_b;
        }
      } else {
        alo[0] = ilo_a+1;
        ahi[0] = ihi_a+1;
        alo[1] = jlo_a;
        ahi[1] = jhi_a;
        blo[0] = ilo_b;
        bhi[0] = ihi_b;
        blo[1] = jlo_b+1;
        bhi[1] = jhi_b+1;
      }
      lda = ahi[0]-alo[0]+1;
      ldb = bhi[0]-blo[0]+1;
      /* get data blocks from A and B */
      data_a = malloc(asize*elemsize);
      data_b = malloc(bsize*elemsize);
      pnga_get(g_a, alo, ahi, data_a, &lda);
      pnga_get(g_b, blo, bhi, data_b, &ldb);
      if (longidx) {
        if (type == C_INT) {
          SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(int,int64_t);
        } else if (type == C_LONG) {
          SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(long,int64_t);
        } else if (type == C_LONGLONG) {
          SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(long long,int64_t);
        } else if (type == C_FLOAT) {
          SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(float,int64_t);
        } else if (type == C_DBL) {
          SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(double,int64_t);
        } else if (type == C_SCPL) {
          SPRS_COMPLEX_SAMPLED_MATMAT_MULTIPLY_M(float,int64_t);
        } else if (type == C_DCPL) {
          SPRS_COMPLEX_SAMPLED_MATMAT_MULTIPLY_M(double,int64_t);
        }
      } else {
        if (type == C_INT) {
          SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(int,int);
        } else if (type == C_LONG) {
          SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(long,int);
        } else if (type == C_LONGLONG) {
          SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(long long,int);
        } else if (type == C_FLOAT) {
          SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(float,int);
        } else if (type == C_DBL) {
          SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M(double,int);
        } else if (type == C_SCPL) {
          SPRS_COMPLEX_SAMPLED_MATMAT_MULTIPLY_M(float,int);
        } else if (type == C_DCPL) {
          SPRS_COMPLEX_SAMPLED_MATMAT_MULTIPLY_M(double,int);
        }
      }
      free(data_a);
      free(data_b);
    }
  }
  free(column_info);
  pnga_pgroup_sync(SPA[hdl_c].grp);
}
#undef SPRS_REAL_SAMPLED_MATMAT_MULTIPLY_M
#undef SPRS_COMPLEX_SAMPLED_MATMAT_MULTIPLY_M

/**
 * Scale all values to be integers in the range [0,255].
 * @param s_a sparse array A
 * @param max scale all values by 256/max
 * @return sparse array of same type as s_a but with values scaled
 *         to be integers in the range [0,255]
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_quantize256 = pnga_sprs_array_quantize256
#endif
Integer pnga_sprs_array_quantize256(Integer s_a, void *max)
{
  Integer hdl = GA_OFFSET+s_a;
  Integer type = SPA[hdl].type;
  Integer s_n, hdl_n;
  Integer ok = 1;
  int *sidx, *sjdx;
  int64_t *lidx, *ljdx;
  int *isptr, *jsptr;
  int64_t *ilptr, *jlptr;
  Integer grp = SPA[hdl].grp;
  Integer me = pnga_pgroup_nodeid(grp);
  Integer nprocs = pnga_pgroup_nnodes(grp);
  Integer max_nnz, innz;
  void *data;
  void *dptr;
  Integer size, lo, hi, ld;
  Integer i, j, ilo, ihi;
  Integer idx_size = SPA[hdl].idx_size;
  Integer nblocks = SPA[hdl].nblocks;
  Integer *blkidx, *blksize, *offset;
  Integer blkcnt, ncnt, iblk, itot;
  int64_t *isum, *nsum;
  Integer *mapc;
  char plus[2];
  Integer one = 1;
  int64_t *row_info;
  Integer jbot, jtop;
  if (type == C_SCPL || type == C_DCPL) {
    pnga_error("(ga_sprs_array_quantize256) complex data types not allowed",
        type);
  }
  if (type == C_INT) {
    if (*((int*)max) <= 0) ok = 0;
  } else if (type == C_LONG) {
    if (*((long*)max) <= 0) ok = 0;
  } else if (type == C_LONGLONG) {
    if (*((long long*)max) <= 0) ok = 0;
  } else if (type == C_FLOAT) {
    if (*((float*)max) <= 0.0) ok = 0;
  } else if (type == C_DBL) {
    if (*((double*)max) <= 0.0) ok = 0;
  }
  if (!ok) {
    pnga_error("(ga_sprs_array_quantize256) illegal value for max",0);
  }
  size = SPA[hdl].size;
  /* create arrays to hold indices and quantized values. These cannot be any
   * larger than the locally held portions of s_a, so use these sizes to
   * allocate memory
   */
  pnga_distribution(SPA[hdl].g_i,me,&lo,&hi);
  if (idx_size == 4 && hi >= lo) {
    sidx = (int*)malloc((hi-lo+1)*idx_size);
    pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&isptr,&ld);
  } else if (hi >= lo) {
    lidx = (int64_t*)malloc((hi-lo+1)*idx_size);
    pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&ilptr,&ld);
  }
  pnga_distribution(SPA[hdl].g_j,me,&lo,&hi);
  if (idx_size == 4 && hi >= lo) {
    sjdx = (int*)malloc((hi-lo+1)*idx_size);
    pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jsptr,&ld);
  } else if (hi >= lo) {
    ljdx = (int64_t*)malloc((hi-lo+1)*idx_size);
    pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jlptr,&ld);
  }
  pnga_distribution(SPA[hdl].g_data,me,&lo,&hi);
  data = malloc((hi-lo+1)*size);
  pnga_access_ptr(SPA[hdl].g_data,&lo,&hi,&dptr,&ld);
  /* Loop through blocks on this array and apply quantization transformation to
   * them
   */
  ilo = SPA[hdl].ilo;
  ihi = SPA[hdl].ihi;
  blkcnt = 0;
  ncnt = 0;
  /* set up arrays containing block information */
  blkidx = (Integer*)malloc(nblocks*sizeof(Integer));
  blksize = (Integer*)malloc(nblocks*sizeof(Integer));
  offset = (Integer*)malloc(nblocks*sizeof(Integer));
  max_nnz = 0;
  for (iblk=0; iblk<nblocks; iblk++) {
    Integer istart = iblk*(ihi-ilo+2);
    Integer nvals;
    Integer jstart;
    Integer joffset = SPA[hdl].offset[iblk];
    offset[blkcnt] = ncnt;
    for (i=ilo; i<=ihi; i++) {
      if (idx_size == 4) {
        jstart = (Integer)isptr[i-ilo+istart]+joffset;
        nvals = (Integer)(isptr[i-ilo+1+istart]-isptr[i-ilo+istart]);
      } else {
        jstart = (Integer)ilptr[i-ilo+istart]+joffset;
        nvals = (Integer)(ilptr[i-ilo+1+istart]-ilptr[i-ilo+istart]);
      }
      if (idx_size == 4) {
        sidx[blkcnt*(ihi-ilo+2)+i-ilo] = (int)ncnt-offset[blkcnt];
      } else {
        lidx[blkcnt*(ihi-ilo+2)+i-ilo] = (int64_t)ncnt-offset[blkcnt];
      }
      innz = ncnt;
      for (j=0; j<nvals; j++) {
        Integer jj;
        if (idx_size == 4) {
          jj = (Integer)jsptr[j+jstart];
          sjdx[ncnt] = jsptr[j+jstart];
        } else {
          jj = (Integer)jlptr[j+jstart];
          ljdx[ncnt] = jlptr[j+jstart];
        } 
        if (type == C_INT) {
          double dmax = (double)(*((int*)max));
          int val = ((int*)dptr)[j+jstart];
          double dval = (double)val;
          if (val >= *((int*)max)) {
           ((int*)data)[ncnt] = 255;
           ncnt++;
          } else if (val > 0) {
            int ival = (int)((dval/dmax)*255.0);
            if (ival > 0 || i == jj) {
              ((int*)data)[ncnt] = ival;
              ncnt++;
            }
          } else {
            if (i == jj) {
              ((int*)data)[ncnt] = 0;
              ncnt++;
            }
          }
        } else if (type == C_LONG) {
          double dmax = (double)(*((long*)max));
          long val = ((long*)dptr)[j+jstart];
          double dval = (double)val;
          if (val >= *((long*)max)) {
           ((long*)data)[ncnt] = 255;
           ncnt++;
          } else if (val > 0) {
            long ival = (long)((dval/dmax)*255.0);
            if (ival > 0 || i == jj) {
              ((long*)data)[ncnt] = ival;
              ncnt++;
            }
          } else {
            if (i == jj) {
              ((long*)data)[ncnt] = 0;
              ncnt++;
            }
          }
        } else if (type == C_LONGLONG) {
          double dmax = (double)(*((long long*)max));
          long long val = ((long long*)dptr)[j+jstart];
          double dval = (double)val;
          if (val >= *((long long*)max)) {
           ((long long*)data)[ncnt] = 255;
           ncnt++;
          } else if (val > 0) {
            long long ival = (long long)((dval/dmax)*255.0);
            if (ival > 0 || i == jj) {
              ((long long*)data)[ncnt] = ival;
              ncnt++;
            }
          } else {
            if (i == jj) {
              ((long long*)data)[ncnt] = 0;
              ncnt++;
            }
          }
        } else if (type == C_FLOAT) {
          float val = ((float*)dptr)[j+jstart];
          if (val >= *((float*)max)) {
            ((float*)data)[ncnt] = 255.0;
            ncnt++;
          } else if (val > 0.0) {
            float fval = (float)((int)((val/(*((float*)max)))*255.0));
            if (fval > 0.0 || i == jj) {
              ((float*)data)[ncnt] = fval;
              ncnt++;
            }
          } else {
            if (i == jj) {
              ((float*)data)[ncnt] = 0.0;
              ncnt++;
            }
          }
        } else if (type == C_DBL) {
          double val = ((double*)dptr)[j+jstart];
          if (val >= *((double*)max)) {
            ((double*)data)[ncnt] = 255.0;
            ncnt++;
          } else if (val > 0.0) {
            double dval = (double)((int)((val/(*((double*)max)))*255.0));
            if (dval > 0.0 || i == jj) {
              ((double*)data)[ncnt] = dval;
              ncnt++;
            }
          } else {
            if (i == jj) {
              ((double*)data)[ncnt] = 0.0;
              ncnt++;
            }
          }
        }
      } /* loop over j */
      innz = ncnt - innz;
      if (innz > max_nnz) max_nnz = innz;
    } /* loop over i */
    if (idx_size == 4) {
      sidx[blkcnt*(ihi-ilo+2)+ihi-ilo+1] = (int)ncnt-offset[blkcnt];
    } else {
      lidx[blkcnt*(ihi-ilo+2)+ihi-ilo+1] = (int64_t)ncnt-offset[blkcnt];
    }
    /* only include block if it has non-zero values */
    if (ncnt-offset[blkcnt] > 0) {
      blkidx[blkcnt] = SPA[hdl].blkidx[iblk];
      blksize[blkcnt] = ncnt-offset[blkcnt];
      blkcnt++;
    }
  } /* loop over iblk */
  /* Calculate size of internal arrays in new sparse array */
  isum = (int64_t*)malloc(nprocs*sizeof(int64_t));
  nsum = (int64_t*)malloc(nprocs*sizeof(int64_t));
  mapc = (int64_t*)malloc(nprocs*sizeof(Integer));
  for (i=0; i<nprocs; i++) {
    isum[i] = 0;
    nsum[i] = 0;
  }
  isum[me] = blkcnt*(ihi-ilo+2);
  nsum[me] = ncnt;
  plus[0] = '+';
  plus[1] = '\0';
  pnga_pgroup_gop(grp,C_LONG,isum,nprocs,plus);
  pnga_pgroup_gop(grp,C_LONG,nsum,nprocs,plus);
  /* begin constructing new sparse array */
  s_n = pnga_sprs_array_create(SPA[hdl].idim,SPA[hdl].jdim,
      SPA[hdl].type,idx_size);
  hdl_n = GA_OFFSET + s_n;
  /* create arrays to hold data */
  mapc[0] = 1;
  for (i=1; i<nprocs; i++) mapc[i] = mapc[i-1]+(Integer)isum[i-1];
  itot = 0;
  for (i=0; i<nprocs; i++) itot += isum[i];
  SPA[hdl_n].g_i = pnga_create_handle();
  if (idx_size == 4) {
    pnga_set_data(SPA[hdl_n].g_i,one,&itot,C_INT);
  }  else {
    pnga_set_data(SPA[hdl_n].g_i,one,&itot,C_LONG);
  }
  pnga_set_irreg_distr(SPA[hdl_n].g_i,mapc,&nprocs);
  if (!pnga_allocate(SPA[hdl_n].g_i)) {
    pnga_error("(ga_sprs_array_quantize256) could not allocate i-index",0);
  }
  mapc[0] = 1;
  for (i=1; i<nprocs; i++) mapc[i] = mapc[i-1]+(Integer)nsum[i-1];
  itot = 0;
  for (i=0; i<nprocs; i++) itot += nsum[i];
  SPA[hdl_n].g_j = pnga_create_handle();
  if (idx_size == 4) {
    pnga_set_data(SPA[hdl_n].g_j,one,&itot,C_INT);
  }  else {
    pnga_set_data(SPA[hdl_n].g_j,one,&itot,C_LONG);
  }
  pnga_set_irreg_distr(SPA[hdl_n].g_j,mapc,&nprocs);
  if (!pnga_allocate(SPA[hdl_n].g_j)) {
    pnga_error("(ga_sprs_array_quantize256) could not allocate j-index",0);
  }
  SPA[hdl_n].g_data = pnga_create_handle();
  pnga_set_data(SPA[hdl_n].g_data,one,&itot,type);
  pnga_set_irreg_distr(SPA[hdl_n].g_data,mapc,&nprocs);
  if (!pnga_allocate(SPA[hdl_n].g_data)) {
    pnga_error("(ga_sprs_array_quantize256) could not allocate value array",0);
  }
  /* copy data to new arrays */
  ld = one;
  pnga_distribution(SPA[hdl_n].g_i,me,&lo,&hi);
  if (idx_size == 4) {
    pnga_put(SPA[hdl_n].g_i,&lo,&hi,sidx,&ld);
  } else {
    pnga_put(SPA[hdl_n].g_i,&lo,&hi,lidx,&ld);
  }
  pnga_distribution(SPA[hdl_n].g_j,me,&lo,&hi);
  if (idx_size == 4) {
    pnga_put(SPA[hdl_n].g_j,&lo,&hi,sjdx,&ld);
  } else {
    pnga_put(SPA[hdl_n].g_j,&lo,&hi,ljdx,&ld);
  }
  pnga_distribution(SPA[hdl_n].g_data,me,&lo,&hi);
  pnga_put(SPA[hdl_n].g_data,&lo,&hi,data,&ld);
  /* copy info on data organization */
  SPA[hdl_n].nblocks = blkcnt;
  SPA[hdl_n].nprocs = nprocs;
  SPA[hdl_n].idx_size = idx_size;
  SPA[hdl_n].ilo = ilo;
  SPA[hdl_n].ihi = ihi;
  SPA[hdl_n].type = type;
  SPA[hdl_n].max_nnz = max_nnz;
  SPA[hdl_n].blkidx = (Integer*)malloc(blkcnt*sizeof(Integer));
  SPA[hdl_n].blksize = (Integer*)malloc(blkcnt*sizeof(Integer));
  SPA[hdl_n].offset = (Integer*)malloc(blkcnt*sizeof(Integer));
  for (i=0; i<blkcnt; i++) {
    SPA[hdl_n].blkidx[i] = blkidx[i];
    SPA[hdl_n].blksize[i] = blksize[i];
    SPA[hdl_n].offset[i] = offset[i];
  }
  /* duplicate array containing info on data blocks */
  if (!pnga_duplicate(SPA[hdl].g_blk,&SPA[hdl_n].g_blk,"blk_copy")) {
    pnga_error("(ga_sprs_array_quantize256) unable to duplicate g_blk",0);
  }
  /* set up g_blk */
  row_info = (int64_t*)malloc(3*nprocs*sizeof(int64_t));
  pnga_distribution(SPA[hdl_n].g_j,me,&jbot,&jtop);
  for (i=0; i<nprocs; i++) {
    int iblk;
    /* find index for block in blksize and offset arrays */
    iblk = -1;
    for (j=0; j<SPA[hdl_n].nblocks; j++) {
      if (SPA[hdl_n].blkidx[j] == i) {
        iblk = j;
        break;
      }
    }
    if (iblk != -1) {
      /* block has data */
      row_info[i*3  ] = jbot+SPA[hdl].offset[iblk];
      row_info[i*3+1] = row_info[i*3]+SPA[hdl].blksize[iblk]-1;
    } else {
      /* block contains no data */
      if (i == 0) {
        row_info[0] = 1;
        row_info[1] = 0;
      } else {
        row_info[i*3  ] = row_info[(i-1)*3+1]+1;
        row_info[i*3+1] = row_info[i*3]-1;
      }
    }
    row_info[i*3+2] = iblk;
  }
  /* copy data in row info to g_blk */
  {
    Integer tlo[3], thi[3], tld[2];
    pnga_distribution(SPA[hdl_n].g_blk,me,tlo,thi);
    tlo[0] = 5;
    tlo[1] = me+1;
    tlo[2] = 1;
    thi[0] = 7;
    thi[1] = me+1;
    thi[2] = nprocs;
    tld[0] = 3;
    tld[1] = 1;
    pnga_put(SPA[hdl_n].g_blk,tlo,thi,row_info,tld);
    pnga_pgroup_sync(SPA[hdl_n].grp);
  }
  free(row_info);
  SPA[hdl_n].ready = 1;
  if (idx_size == 4) {
    free(sidx);
    free(sjdx);
  } else {
    free(lidx);
    free(ljdx);
  }
  free(data);
  free(blkidx);
  free(blksize);
  free(offset);
  free(isum);
  free(nsum);
  free(mapc);
  return s_n;
}
