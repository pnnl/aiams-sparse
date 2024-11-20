/**
 * @file capi.c
 *
 * Implements the C interface.
 * These calls forward to the (possibly) weak symbols of the internal
 * implementations.
 */
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif
#include <assert.h>

#include "armci.h"
#include "ga.h"
#include "globalp.h"
#include "ga-papi.h"

#if ENABLE_PROFILING
#   include "ga-wapi.h"
#else
#   include "ga-wapidefs.h"
#endif

#define USE_GATSCAT_NEW

int *_ga_argc=NULL;
char ***_ga_argv=NULL;
int _ga_initialize_args=0;
int _ga_initialize_c=0;

short int _ga_irreg_flag = 0;

static Integer* copy_map(int block[], int block_ndim, int map[]);
static Integer* copy_map64(int64_t block[], int block_ndim, int64_t map[]);

#ifdef USE_FAPI
#  define COPYC2F(carr, farr, n){\
   int i; for(i=0; i< (n); i++)(farr)[i]=(Integer)(carr)[i];} 
#  define COPYF2C(farr, carr, n){\
   int i; for(i=0; i< (n); i++)(carr)[i]=(int)(farr)[i];} 
#  define COPYF2C_64(farr, carr, n){\
   int i; for(i=0; i< (n); i++)(carr)[i]=(int64_t)(farr)[i];} 
#  define COPYINDEX_F2C     COPYF2C
#  define COPYINDEX_F2C_64  COPYF2C_64
#else
#  define COPYC2F(carr, farr, n){\
   int i; for(i=0; i< (n); i++)(farr)[n-i-1]=(Integer)(carr)[i];} 
#  define COPYF2C(farr, carr, n){\
   int i; for(i=0; i< (n); i++)(carr)[n-i-1]=(int)(farr)[i];} 
#  define COPYF2C_64(farr, carr, n){\
   int i; for(i=0; i< (n); i++)(carr)[n-i-1]=(int64_t)(farr)[i];} 
#  define COPYINDEX_C2F(carr, farr, n){\
   int i; for(i=0; i< (n); i++)(farr)[n-i-1]=(Integer)(carr)[i]+1;}
#  define COPYINDEX_F2C(farr, carr, n){\
   int i; for(i=0; i< (n); i++)(carr)[n-i-1]=(int)(farr)[i] -1;}
#  define COPYINDEX_F2C_64(farr, carr, n){\
   int i; for(i=0; i< (n); i++)(carr)[n-i-1]=(int64_t)(farr)[i] -1;}
#define BASE_0
#endif

#define COPY(CAST,src,dst,n) {\
   int i; for(i=0; i< (n); i++)(dst)[i]=(CAST)(src)[i];} 
#define COPY_INC(CAST,src,dst,n) {\
   int i; for(i=0; i< (n); i++)(dst)[i]=(CAST)(src)[i]+1;} 
#define COPY_DEC(CAST,src,dst,n) {\
   int i; for(i=0; i< (n); i++)(dst)[i]=(CAST)(src)[i]-1;} 

int GA_Uses_fapi(void)
{
#ifdef USE_FAPI
return 1;
#else
return 0;
#endif
}


void GA_Initialize_ltd(size_t limit)
{
  Integer lim = (Integer)limit;
  _ga_initialize_c = 1;
  wnga_initialize_ltd(lim);
}

void NGA_Initialize_ltd(size_t limit)
{
  Integer lim = (Integer)limit;
  _ga_initialize_c = 1;
  wnga_initialize_ltd(lim);
}

void GA_Initialize_args(int *argc, char ***argv)
{
  _ga_argc = argc;
  _ga_argv = argv;
  _ga_initialize_c = 1;
  _ga_initialize_args = 1;

  wnga_initialize();
}

void GA_Initialize()
{
  _ga_initialize_c = 1;
  wnga_initialize();
}

void NGA_Initialize()
{
  _ga_initialize_c = 1;
  wnga_initialize();
}

int GA_Initialized()
{
    return wnga_initialized();
}

int NGA_Initialized()
{
    return wnga_initialized();
}

void GA_Terminate() 
{
    wnga_terminate();
    
    _ga_argc = NULL;
    _ga_argv = NULL;
    _ga_initialize_args = 0;
    _ga_initialize_c = 0;
}

void NGA_Terminate() 
{
    wnga_terminate();
    
    _ga_argc = NULL;
    _ga_argv = NULL;
    _ga_initialize_args = 0;
    _ga_initialize_c = 0;
}

int NGA_Create(int type, int ndim, int dims[], char *name, int *chunk)
{
    Integer *ptr, g_a; 
    logical st;
    Integer _ga_work[MAXDIM];
    Integer _ga_dims[MAXDIM];
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    if(!chunk)ptr=(Integer*)0;  
    else {
         COPYC2F(chunk,_ga_work, ndim);
         ptr = _ga_work;
    }
    st = wnga_create((Integer)type, (Integer)ndim, _ga_dims, name, ptr, &g_a);
    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create64(int type, int ndim, int64_t dims[], char *name, int64_t *chunk)
{
    Integer *ptr, g_a; 
    logical st;
    Integer _ga_dims[MAXDIM];
    Integer _ga_work[MAXDIM];
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    if(!chunk)ptr=(Integer*)0;  
    else {
         COPYC2F(chunk,_ga_work, ndim);
         ptr = _ga_work;
    }
    st = wnga_create((Integer)type, (Integer)ndim, _ga_dims, name, ptr, &g_a);
    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create_config(int type, int ndim, int dims[], char *name, int chunk[],
                      int p_handle)
{
    Integer *ptr, g_a; 
    logical st;
    Integer _ga_dims[MAXDIM];
    Integer _ga_work[MAXDIM];
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    if(!chunk)ptr=(Integer*)0;  
    else {
         COPYC2F(chunk,_ga_work, ndim);
         ptr = _ga_work;
    }
    st = wnga_create_config((Integer)type, (Integer)ndim, _ga_dims, name, ptr,
                    (Integer)p_handle, &g_a);
    if(st==TRUE) return (int) g_a;
    else return 0;
}


int NGA_Create_config64(int type, int ndim, int64_t dims[], char *name, int64_t chunk[], int p_handle)
{
    Integer *ptr, g_a; 
    logical st;
    Integer _ga_dims[MAXDIM];
    Integer _ga_work[MAXDIM];
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    if(!chunk)ptr=(Integer*)0;  
    else {
         COPYC2F(chunk,_ga_work, ndim);
         ptr = _ga_work;
    }
    st = wnga_create_config((Integer)type, (Integer)ndim, _ga_dims, name, ptr,
                    (Integer)p_handle, &g_a);
    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create_irreg(int type,int ndim,int dims[],char *name,int block[],int map[])
{
    Integer g_a;
    logical st;
    Integer _ga_dims[MAXDIM];
    Integer _ga_work[MAXDIM];
    Integer *_ga_map_capi;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    COPYC2F(block,_ga_work, ndim);
    _ga_map_capi = copy_map(block, ndim, map);

    _ga_irreg_flag = 1; /* set this flag=1, to indicate array is irregular */
    st = wnga_create_irreg(type, (Integer)ndim, _ga_dims, name, _ga_map_capi,
            _ga_work, &g_a);
    _ga_irreg_flag = 0; /* unset it after creating the array */

    free(_ga_map_capi);
    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create_irreg64(int type,int ndim,int64_t dims[],char *name,int64_t block[],int64_t map[])
{
    Integer g_a;
    logical st;
    Integer _ga_dims[MAXDIM];
    Integer _ga_work[MAXDIM];
    Integer *_ga_map_capi;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    COPYC2F(block,_ga_work, ndim);
    _ga_map_capi = copy_map64(block, ndim, map);

    _ga_irreg_flag = 1; /* set this flag=1, to indicate array is irregular */
    st = wnga_create_irreg(type, (Integer)ndim, _ga_dims, name, _ga_map_capi,
            _ga_work, &g_a);
    _ga_irreg_flag = 0; /* unset it after creating the array */

    free(_ga_map_capi);
    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create_irreg_config(int type,int ndim,int dims[],char *name,int block[],
                            int map[], int p_handle)
{
    Integer g_a;
    logical st;
    Integer _ga_dims[MAXDIM];
    Integer _ga_work[MAXDIM];
    Integer *_ga_map_capi;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    COPYC2F(block,_ga_work, ndim);
    _ga_map_capi = copy_map(block, ndim, map);

    _ga_irreg_flag = 1; /* set this flag=1, to indicate array is irregular */
    st = wnga_create_irreg_config(type, (Integer)ndim, _ga_dims, name,
            _ga_map_capi, _ga_work, (Integer)p_handle, &g_a);
    _ga_irreg_flag = 0; /* unset it, after creating array */

    free(_ga_map_capi);
    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create_irreg_config64(int type,int ndim,int64_t dims[],char *name,int64_t block[], int64_t map[], int p_handle)
{
    Integer g_a;
    logical st;
    Integer _ga_dims[MAXDIM];
    Integer _ga_work[MAXDIM];
    Integer *_ga_map_capi;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    COPYC2F(block,_ga_work, ndim);
    _ga_map_capi = copy_map64(block, ndim, map);

    _ga_irreg_flag = 1; /* set this flag=1, to indicate array is irregular */
    st = wnga_create_irreg_config(type, (Integer)ndim, _ga_dims, name,
            _ga_map_capi, _ga_work, (Integer)p_handle, &g_a);
    _ga_irreg_flag = 0; /* unset it, after creating array */

    free(_ga_map_capi);
    if(st==TRUE) return (int) g_a;
    else return 0;
}

int GA_Create_handle()
{
    Integer g_a;
    g_a = wnga_create_handle();
    return (int)g_a;
}

int NGA_Create_handle()
{
    Integer g_a;
    g_a = wnga_create_handle();
    return (int)g_a;
}

void GA_Set_data(int g_a, int ndim, int dims[], int type)
{
    Integer aa, nndim, ttype;
    Integer _ga_dims[MAXDIM];
    COPYC2F(dims,_ga_dims, ndim);
    aa = (Integer)g_a;
    nndim = (Integer)ndim;
    ttype = (Integer)type;
    wnga_set_data(aa, nndim, _ga_dims, ttype);
}

void GA_Set_data64(int g_a, int ndim, int64_t dims[], int type)
{
    Integer aa, nndim, ttype;
    Integer _ga_dims[MAXDIM];
    COPYC2F(dims,_ga_dims, ndim);
    aa = (Integer)g_a;
    nndim = (Integer)ndim;
    ttype = (Integer)type;
    wnga_set_data(aa, nndim, _ga_dims, ttype);
}

void NGA_Set_data(int g_a, int ndim, int dims[], int type)
{
    Integer aa, nndim, ttype;
    Integer _ga_dims[MAXDIM];
    COPYC2F(dims,_ga_dims, ndim);
    aa = (Integer)g_a;
    nndim = (Integer)ndim;
    ttype = (Integer)type;
    wnga_set_data(aa, nndim, _ga_dims, ttype);
}

void NGA_Set_data64(int g_a, int ndim, int64_t dims[], int type)
{
    Integer aa, nndim, ttype;
    Integer _ga_dims[MAXDIM];
    COPYC2F(dims,_ga_dims, ndim);
    aa = (Integer)g_a;
    nndim = (Integer)ndim;
    ttype = (Integer)type;
    wnga_set_data(aa, nndim, _ga_dims, ttype);
}

void GA_Set_chunk(int g_a, int chunk[])
{
    Integer aa, *ptr, ndim;
    Integer _ga_work[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    if(!chunk)ptr=(Integer*)0;  
    else {
      COPYC2F(chunk,_ga_work, ndim);
      ptr = _ga_work;
    }
    wnga_set_chunk(aa, ptr);
}

void GA_Set_chunk64(int g_a, int64_t chunk[])
{
    Integer aa, *ptr, ndim;
    Integer _ga_work[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    if(!chunk)ptr=(Integer*)0;  
    else {
      COPYC2F(chunk,_ga_work, ndim);
      ptr = _ga_work;
    }
    wnga_set_chunk(aa, ptr);
}

void NGA_Set_chunk(int g_a, int chunk[])
{
    Integer aa, *ptr, ndim;
    Integer _ga_work[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    if(!chunk)ptr=(Integer*)0;  
    else {
      COPYC2F(chunk,_ga_work, ndim);
      ptr = _ga_work;
    }
    wnga_set_chunk(aa, ptr);
}

void NGA_Set_chunk64(int g_a, int64_t chunk[])
{
    Integer aa, *ptr, ndim;
    Integer _ga_work[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    if(!chunk)ptr=(Integer*)0;  
    else {
      COPYC2F(chunk,_ga_work, ndim);
      ptr = _ga_work;
    }
    wnga_set_chunk(aa, ptr);
}

void GA_Set_array_name(int g_a, char *name)
{
    Integer aa;
    aa = (Integer)g_a;
    wnga_set_array_name(aa, name);
}

void NGA_Set_array_name(int g_a, char *name)
{
    Integer aa;
    aa = (Integer)g_a;
    wnga_set_array_name(aa, name);
}

void GA_Get_array_name(int g_a, char *name)
{
    Integer aa;
    aa = (Integer)g_a;
    wnga_get_array_name(aa, name);
}

void NGA_Get_array_name(int g_a, char *name)
{
    Integer aa;
    aa = (Integer)g_a;
    wnga_get_array_name(aa, name);
}

void GA_Set_pgroup(int g_a, int p_handle)
{
  Integer aa, pp;
  aa = (Integer)g_a;
  pp = (Integer)p_handle;
  wnga_set_pgroup(aa, pp);
}

void NGA_Set_pgroup(int g_a, int p_handle)
{
  Integer aa, pp;
  aa = (Integer)g_a;
  pp = (Integer)p_handle;
  wnga_set_pgroup(aa, pp);
}

void GA_Set_block_cyclic(int g_a, int dims[])
{
    Integer aa, ndim;
    Integer _ga_dims[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(dims,_ga_dims, ndim);
    wnga_set_block_cyclic(aa, _ga_dims);
}

void NGA_Set_block_cyclic(int g_a, int dims[])
{
    Integer aa, ndim;
    Integer _ga_dims[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(dims,_ga_dims, ndim);
    wnga_set_block_cyclic(aa, _ga_dims);
}

void GA_Set_block_cyclic64(int g_a, int64_t dims[])
{
    Integer aa, ndim;
    Integer _ga_dims[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(dims,_ga_dims, ndim);
    wnga_set_block_cyclic(aa, _ga_dims);
}

void NGA_Set_block_cyclic64(int g_a, int64_t dims[])
{
    Integer aa, ndim;
    Integer _ga_dims[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(dims,_ga_dims, ndim);
    wnga_set_block_cyclic(aa, _ga_dims);
}

void GA_Set_restricted(int g_a, int list[], int size)
{
    Integer aa;
    Integer asize = (Integer)size;
    int i;
    Integer *_ga_map_capi;
    aa = (Integer)g_a;
    _ga_map_capi = (Integer*)malloc(size * sizeof(Integer));
    for (i=0; i<size; i++)
       _ga_map_capi[i] = (Integer)list[i];
    wnga_set_restricted(aa,_ga_map_capi,asize);
    free(_ga_map_capi);
}

void NGA_Set_restricted(int g_a, int list[], int size)
{
    Integer aa;
    Integer asize = (Integer)size;
    int i;
    Integer *_ga_map_capi;
    aa = (Integer)g_a;
    _ga_map_capi = (Integer*)malloc(size * sizeof(Integer));
    for (i=0; i<size; i++)
       _ga_map_capi[i] = (Integer)list[i];
    wnga_set_restricted(aa,_ga_map_capi,asize);
    free(_ga_map_capi);
}

void GA_Set_restricted_range(int g_a, int lo_proc, int hi_proc)
{
    Integer aa, lo, hi;
    aa = (Integer)g_a;
    lo = (Integer)lo_proc;
    hi = (Integer)hi_proc;
    wnga_set_restricted_range(aa,lo,hi);
}

void NGA_Set_restricted_range(int g_a, int lo_proc, int hi_proc)
{
    Integer aa, lo, hi;
    aa = (Integer)g_a;
    lo = (Integer)lo_proc;
    hi = (Integer)hi_proc;
    wnga_set_restricted_range(aa,lo,hi);
}

void GA_Set_property(int g_a, char* property)
{
    Integer aa;
    aa = (Integer)g_a;
    wnga_set_property(aa,property);
}

void NGA_Set_property(int g_a, char* property)
{
    Integer aa;
    aa = (Integer)g_a;
    wnga_set_property(aa,property);
}

void GA_Unset_property(int g_a)
{
    Integer aa;
    aa = (Integer)g_a;
    wnga_unset_property(aa);
}

void NGA_Unset_property(int g_a)
{
    Integer aa;
    aa = (Integer)g_a;
    wnga_unset_property(aa);
}

void GA_Set_memory_dev(int g_a, char *device)
{
    Integer aa;
    aa = (Integer)g_a;
    wnga_set_memory_dev(aa,device);
}

void NGA_Set_memory_dev(int g_a, char *device)
{
    Integer aa;
    aa = (Integer)g_a;
    wnga_set_memory_dev(aa,device);
}

int GA_Total_blocks(int g_a)
{
    Integer aa;
    aa = (Integer)g_a;
    return (int)wnga_total_blocks(aa);
}

int NGA_Total_blocks(int g_a)
{
    Integer aa;
    aa = (Integer)g_a;
    return (int)wnga_total_blocks(aa);
}

void GA_Get_proc_index(int g_a, int iproc, int index[])
{
     Integer aa, proc, ndim;
     Integer _ga_work[MAXDIM];
     aa = (Integer)g_a;
     proc = (Integer)iproc;
     ndim = wnga_get_dimension(aa);
     wnga_get_proc_index(aa, proc, _ga_work);
     COPYF2C(_ga_work,index, ndim);
}

void NGA_Get_proc_index(int g_a, int iproc, int index[])
{
     Integer aa, proc, ndim;
     Integer _ga_work[MAXDIM];
     aa = (Integer)g_a;
     proc = (Integer)iproc;
     ndim = wnga_get_dimension(aa);
     wnga_get_proc_index(aa, proc, _ga_work);
     COPYF2C(_ga_work,index, ndim);
}

void GA_Get_block_info(int g_a, int num_blocks[], int block_dims[])
{
     Integer aa, ndim;
     Integer _ga_work[MAXDIM], _ga_lo[MAXDIM];
     aa = (Integer)g_a;
     ndim = wnga_get_dimension(aa);
     wnga_get_block_info(aa, _ga_work, _ga_lo);
     COPYF2C(_ga_work,num_blocks, ndim);
     COPYF2C(_ga_lo,block_dims, ndim);
}

void NGA_Get_block_info(int g_a, int num_blocks[], int block_dims[])
{
     Integer aa, ndim;
     Integer _ga_work[MAXDIM], _ga_lo[MAXDIM];
     aa = (Integer)g_a;
     ndim = wnga_get_dimension(aa);
     wnga_get_block_info(aa, _ga_work, _ga_lo);
     COPYF2C(_ga_work,num_blocks, ndim);
     COPYF2C(_ga_lo,block_dims, ndim);
}

int GA_Uses_proc_grid(int g_a)
{
     Integer aa = (Integer)g_a;
     return (int)wnga_uses_proc_grid(aa);
}

int NGA_Uses_proc_grid(int g_a)
{
     Integer aa = (Integer)g_a;
     return (int)wnga_uses_proc_grid(aa);
}

int GA_Valid_handle(int g_a)
{
     Integer aa = (Integer)g_a;
     return (int)wnga_valid_handle(aa);
}

int NGA_Valid_handle(int g_a)
{
     Integer aa = (Integer)g_a;
     return (int)wnga_valid_handle(aa);
}

int GA_Verify_handle(int g_a)
{
     Integer aa = (Integer)g_a;
     return (int)wnga_valid_handle(aa);
}

int NGA_Verify_handle(int g_a)
{
     Integer aa = (Integer)g_a;
     return (int)wnga_valid_handle(aa);
}

void GA_Set_block_cyclic_proc_grid(int g_a, int block[], int proc_grid[])
{
    Integer aa, ndim;
    Integer _ga_dims[MAXDIM];
    Integer _ga_lo[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_ga_dims, ndim);
    COPYC2F(proc_grid,_ga_lo, ndim);
    wnga_set_block_cyclic_proc_grid(aa, _ga_dims, _ga_lo);
}

void NGA_Set_block_cyclic_proc_grid(int g_a, int block[], int proc_grid[])
{
    Integer aa, ndim;
    Integer _block[MAXDIM];
    Integer _proc_grid[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_block, ndim);
    COPYC2F(proc_grid, _proc_grid, ndim);
    wnga_set_block_cyclic_proc_grid(aa, _block, _proc_grid);
}

void GA_Set_block_cyclic_proc_grid64(int g_a, int64_t block[], int64_t proc_grid[])
{
    Integer aa, ndim;
    Integer _ga_dims[MAXDIM];
    Integer _ga_lo[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_ga_dims, ndim);
    COPYC2F(proc_grid,_ga_lo, ndim);
    wnga_set_block_cyclic_proc_grid(aa, _ga_dims, _ga_lo);
}

void NGA_Set_block_cyclic_proc_grid64(int g_a, int64_t block[], int64_t proc_grid[])
{
    Integer aa, ndim;
    Integer _block[MAXDIM];
    Integer _proc_grid[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_block, ndim);
    COPYC2F(proc_grid, _proc_grid, ndim);
    wnga_set_block_cyclic_proc_grid(aa, _block, _proc_grid);
}

void GA_Set_tiled_proc_grid(int g_a, int block[], int proc_grid[])
{
    Integer aa, ndim;
    Integer _ga_dims[MAXDIM];
    Integer _ga_lo[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_ga_dims, ndim);
    COPYC2F(proc_grid,_ga_lo, ndim);
    wnga_set_tiled_proc_grid(aa, _ga_dims, _ga_lo);
}

void NGA_Set_tiled_proc_grid(int g_a, int block[], int proc_grid[])
{
    Integer aa, ndim;
    Integer _block[MAXDIM];
    Integer _proc_grid[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_block, ndim);
    COPYC2F(proc_grid, _proc_grid, ndim);
    wnga_set_tiled_proc_grid(aa, _block, _proc_grid);
}

void GA_Set_tiled_proc_grid64(int g_a, int64_t block[], int64_t proc_grid[])
{
    Integer aa, ndim;
    Integer _ga_dims[MAXDIM];
    Integer _ga_lo[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_ga_dims, ndim);
    COPYC2F(proc_grid,_ga_lo, ndim);
    wnga_set_tiled_proc_grid(aa, _ga_dims, _ga_lo);
}

void NGA_Set_tiled_proc_grid64(int g_a, int64_t block[], int64_t proc_grid[])
{
    Integer aa, ndim;
    Integer _block[MAXDIM];
    Integer _proc_grid[MAXDIM];
    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_block, ndim);
    COPYC2F(proc_grid, _proc_grid, ndim);
    wnga_set_tiled_proc_grid(aa, _block, _proc_grid);
}

void GA_Set_tiled_irreg_proc_grid(int g_a, int mapc[], int nblocks[],
    int proc_grid[])
{
    Integer aa, ndim;
    Integer *_ga_map_capi;
    Integer _nblocks[MAXDIM];
    Integer _proc_grid[MAXDIM];

    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    _ga_map_capi = copy_map(nblocks, (int)ndim, mapc);
    COPYC2F(nblocks,_nblocks, ndim);
    COPYC2F(proc_grid, _proc_grid, ndim);

    wnga_set_tiled_irreg_proc_grid(aa, _ga_map_capi, _nblocks, _proc_grid);
    free(_ga_map_capi);
}

void NGA_Set_tiled_irreg_proc_grid(int g_a, int mapc[], int nblocks[],
    int proc_grid[])
{
    Integer aa, ndim;
    Integer *_ga_map_capi;
    Integer _nblocks[MAXDIM];
    Integer _proc_grid[MAXDIM];

    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    _ga_map_capi = copy_map(nblocks, (int)ndim, mapc);
    COPYC2F(nblocks,_nblocks, ndim);
    COPYC2F(proc_grid, _proc_grid, ndim);

    wnga_set_tiled_irreg_proc_grid(aa, _ga_map_capi, _nblocks, _proc_grid);
    free(_ga_map_capi);
}

void GA_Set_tiled_irreg_proc_grid64(int g_a, int64_t mapc[], int64_t nblocks[],
    int64_t proc_grid[])
{
    Integer aa, ndim;
    Integer *_ga_map_capi;
    Integer _nblocks[MAXDIM];
    Integer _proc_grid[MAXDIM];

    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    _ga_map_capi = copy_map64(nblocks, (int)ndim, mapc);
    COPYC2F(nblocks,_nblocks, ndim);
    COPYC2F(proc_grid, _proc_grid, ndim);

    wnga_set_tiled_irreg_proc_grid(aa, _ga_map_capi, _nblocks, _proc_grid);
    free(_ga_map_capi);
}
void NGA_Set_tiled_irreg_proc_grid64(int g_a, int64_t mapc[], int64_t nblocks[],
    int64_t proc_grid[])
{
    Integer aa, ndim;
    Integer *_ga_map_capi;
    Integer _nblocks[MAXDIM];
    Integer _proc_grid[MAXDIM];

    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    _ga_map_capi = copy_map64(nblocks, (int)ndim, mapc);
    COPYC2F(nblocks,_nblocks, ndim);
    COPYC2F(proc_grid, _proc_grid, ndim);

    wnga_set_tiled_irreg_proc_grid(aa, _ga_map_capi, _nblocks, _proc_grid);
    free(_ga_map_capi);
}

int GA_Get_pgroup(int g_a)
{
    Integer aa;
    aa = (Integer)g_a;
    return (int)wnga_get_pgroup(aa);
}

int NGA_Get_pgroup(int g_a)
{
    Integer aa;
    aa = (Integer)g_a;
    return (int)wnga_get_pgroup(aa);
}

int GA_Get_pgroup_size(int grp_id)
{
    Integer aa;
    aa = (Integer)grp_id;
    return (int)wnga_get_pgroup_size(aa);
}

int NGA_Get_pgroup_size(int grp_id)
{
    Integer aa;
    aa = (Integer)grp_id;
    return (int)wnga_get_pgroup_size(aa);
}

void GA_Set_irreg_distr(int g_a, int map[], int block[])
{
    Integer aa, ndim;
    Integer _ga_work[MAXDIM];
    Integer *_ga_map_capi;

    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_ga_work, ndim);
    _ga_map_capi = copy_map(block, (int)ndim, map);

    wnga_set_irreg_distr(aa, _ga_map_capi, _ga_work);
    free(_ga_map_capi);
}

void GA_Set_irreg_distr64(int g_a, int64_t map[], int64_t block[])
{
    Integer aa, ndim;
    Integer _ga_work[MAXDIM];
    Integer *_ga_map_capi;

    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_ga_work, ndim);
    _ga_map_capi = copy_map64(block, (int)ndim, map);

    wnga_set_irreg_distr(aa, _ga_map_capi, _ga_work);
    free(_ga_map_capi);
}

void NGA_Set_irreg_distr(int g_a, int map[], int block[])
{
    Integer aa, ndim;
    Integer _ga_work[MAXDIM];
    Integer *_ga_map_capi;

    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_ga_work, ndim);
    _ga_map_capi = copy_map(block, (int)ndim, map);

    wnga_set_irreg_distr(aa, _ga_map_capi, _ga_work);
    free(_ga_map_capi);
}

void NGA_Set_irreg_distr64(int g_a, int64_t map[], int64_t block[])
{
    Integer aa, ndim;
    Integer _ga_work[MAXDIM];
    Integer *_ga_map_capi;

    aa = (Integer)g_a;
    ndim = wnga_get_dimension(aa);
    COPYC2F(block,_ga_work, ndim);
    _ga_map_capi = copy_map64(block, (int)ndim, map);

    wnga_set_irreg_distr(aa, _ga_map_capi, _ga_work);
    free(_ga_map_capi);
}

void GA_Set_irreg_flag(int g_a, int flag)
{
  Integer aa;
  logical fflag;
  aa = (Integer)g_a;
  fflag = (logical)flag;
  wnga_set_irreg_flag(aa, fflag);
}

void NGA_Set_irreg_flag(int g_a, int flag)
{
  Integer aa;
  logical fflag;
  aa = (Integer)g_a;
  fflag = (logical)flag;
  wnga_set_irreg_flag(aa, fflag);
}

int GA_Get_dimension(int g_a)
{
  Integer aa;
  aa = (Integer)g_a;
  return (int)wnga_get_dimension(aa);
}

int NGA_Get_dimension(int g_a)
{
  Integer aa;
  aa = (Integer)g_a;
  return (int)wnga_get_dimension(aa);
}

int GA_Allocate(int g_a)
{
  Integer aa;
  aa = (Integer)g_a;
  return (int)wnga_allocate(aa);
}

int NGA_Allocate(int g_a)
{
  Integer aa;
  aa = (Integer)g_a;
  return (int)wnga_allocate(aa);
}

int GA_Deallocate(int g_a)
{
  Integer aa;
  aa = (Integer)g_a;
  return (int)wnga_deallocate(aa);
}

int NGA_Deallocate(int g_a)
{
  Integer aa;
  aa = (Integer)g_a;
  return (int)wnga_deallocate(aa);
}

int GA_Overlay(int g_a, int g_p)
{
  Integer aa, bb;
  aa = (Integer)g_a;
  bb = (Integer)g_p;
  return (int)wnga_overlay(aa, bb);
}

int NGA_Overlay(int g_a, int g_p)
{
  Integer aa, bb;
  aa = (Integer)g_a;
  bb = (Integer)g_p;
  return (int)wnga_overlay(aa, bb);
}

int GA_Pgroup_nodeid(int grp_id)
{
    Integer agrp_id = (Integer)grp_id;
    return (int)wnga_pgroup_nodeid(agrp_id);
}

int NGA_Pgroup_nodeid(int grp_id)
{
    Integer agrp_id = (Integer)grp_id;
    return (int)wnga_pgroup_nodeid(agrp_id);
}

int GA_Pgroup_nnodes(int grp_id)
{
    Integer agrp_id = (Integer)grp_id;
    return (int)wnga_pgroup_nnodes(agrp_id);
}

int NGA_Pgroup_nnodes(int grp_id)
{
    Integer agrp_id = (Integer)grp_id;
    return (int)wnga_pgroup_nnodes(agrp_id);
}

int GA_Pgroup_create(int *list, int count)
{
    Integer acount = (Integer)count;
    int i;
    int grp_id;
    Integer *_ga_map_capi;
    _ga_map_capi = (Integer*)malloc(count * sizeof(Integer));
    for (i=0; i<count; i++)
       _ga_map_capi[i] = (Integer)list[i];
    grp_id = (int)wnga_pgroup_create(_ga_map_capi,acount);
    free(_ga_map_capi);
    return grp_id;
}

int NGA_Pgroup_create(int *list, int count)
{
    Integer acount = (Integer)count;
    int i;
    int grp_id;
    Integer *_ga_map_capi;
    _ga_map_capi = (Integer*)malloc(count * sizeof(Integer));
    for (i=0; i<count; i++)
       _ga_map_capi[i] = (Integer)list[i];
    grp_id = (int)wnga_pgroup_create(_ga_map_capi,acount);
    free(_ga_map_capi);
    return grp_id;
}

int GA_Pgroup_duplicate(int grp)
{
  Integer pgrp = (Integer)grp;
  return (int)wnga_pgroup_duplicate(pgrp);
}

int NGA_Pgroup_duplicate(int grp)
{
  Integer pgrp = (Integer)grp;
  return (int)wnga_pgroup_duplicate(pgrp);
}

int GA_Pgroup_self()
{
  return (int)wnga_pgroup_self();
}

int NGA_Pgroup_self()
{
  return (int)wnga_pgroup_self();
}

int GA_Pgroup_destroy(int grp)
{
    Integer grp_id = (Integer)grp;
    return (int)wnga_pgroup_destroy(grp_id);
}

int NGA_Pgroup_destroy(int grp)
{
    Integer grp_id = (Integer)grp;
    return (int)wnga_pgroup_destroy(grp_id);
}

int GA_Pgroup_split(int grp_id, int num_group)
{
    Integer anum = (Integer)num_group;
    Integer grp  = (Integer)grp_id;
    return (int)wnga_pgroup_split(grp, anum);
}

int NGA_Pgroup_split(int grp_id, int num_group)
{
    Integer anum = (Integer)num_group;
    Integer grp  = (Integer)grp_id;
    return (int)wnga_pgroup_split(grp, anum);
}

int GA_Pgroup_split_irreg(int grp_id, int color)
{
    Integer acolor = (Integer)color;
    Integer grp  = (Integer)grp_id;
    return (int)wnga_pgroup_split_irreg(grp, acolor);
}

int NGA_Pgroup_split_irreg(int grp_id, int color)
{
    Integer acolor = (Integer)color;
    Integer grp  = (Integer)grp_id;
    return (int)wnga_pgroup_split_irreg(grp, acolor);
}

void GA_Merge_mirrored(int g_a)
{
    Integer a=(Integer)g_a;
    wnga_merge_mirrored(a);
}

void NGA_Merge_mirrored(int g_a)
{
    Integer a=(Integer)g_a;
    wnga_merge_mirrored(a);
}

void GA_Nblock(int g_a, int *nblock)
{
    Integer aa, ndim;
    Integer _ga_work[MAXDIM];
    aa = (Integer)g_a;
    wnga_nblock(aa, _ga_work);
    ndim = wnga_get_dimension(aa);
    COPYF2C(_ga_work,nblock,ndim);
}

void NGA_Nblock(int g_a, int *nblock)
{
    Integer aa, ndim;
    Integer _ga_work[MAXDIM];
    aa = (Integer)g_a;
    wnga_nblock(aa, _ga_work);
    ndim = wnga_get_dimension(aa);
    COPYF2C(_ga_work,nblock,ndim);
}

int GA_Is_mirrored(int g_a)
{
    Integer a=(Integer)g_a;
    return (int)wnga_is_mirrored(a);
}

int NGA_Is_mirrored(int g_a)
{
    Integer a=(Integer)g_a;
    return (int)wnga_is_mirrored(a);
}

void GA_List_nodeid(int *nlist, int nprocs)
{
  Integer i, procs;
  Integer *list;
  procs = (Integer)(nprocs);
  list = malloc(procs*sizeof(int));
  wnga_list_nodeid(list, procs);
  for (i=0; i<procs; i++) {
    nlist[i] = (int)list[i];
  }
  free(list);
}

void NGA_List_nodeid(int *nlist, int nprocs)
{
  Integer i, procs;
  Integer *list;
  procs = (Integer)(nprocs);
  list = malloc(procs*sizeof(int));
  wnga_list_nodeid(list, procs);
  for (i=0; i<procs; i++) {
    nlist[i] = (int)list[i];
  }
  free(list);
}

void NGA_Merge_distr_patch(int g_a, int *alo, int *ahi,
                          int g_b, int *blo, int *bhi)
{
    Integer a = (Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_merge_distr_patch(a, _ga_alo, _ga_ahi, b, _ga_blo, _ga_bhi);
}

void NGA_Merge_distr_patch64(int g_a, int64_t *alo, int64_t *ahi,
                             int g_b, int64_t *blo, int64_t *bhi)
{
    Integer a = (Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_merge_distr_patch(a, _ga_alo, _ga_ahi, b, _ga_blo, _ga_bhi);
}

void GA_Mask_sync(int first, int last)
{
    Integer ifirst = (Integer)first;
    Integer ilast = (Integer)last;
    wnga_mask_sync(ifirst,ilast);
}

void NGA_Mask_sync(int first, int last)
{
    Integer ifirst = (Integer)first;
    Integer ilast = (Integer)last;
    wnga_mask_sync(ifirst,ilast);
}

int GA_Duplicate(int g_a, char* array_name)
{
    logical st;
    Integer a=(Integer)g_a, b;
    st = wnga_duplicate(a, &b, array_name);
    if(st==TRUE) return (int) b;
    else return 0;
}

int NGA_Duplicate(int g_a, char* array_name)
{
    logical st;
    Integer a=(Integer)g_a, b;
    st = wnga_duplicate(a, &b, array_name);
    if(st==TRUE) return (int) b;
    else return 0;
}

void GA_Destroy(int g_a)
{
    logical st;
    Integer a=(Integer)g_a;
    st = wnga_destroy(a);
    if(st==FALSE)GA_Error("GA (c) destroy failed",g_a);
}

void NGA_Destroy(int g_a)
{
    logical st;
    Integer a=(Integer)g_a;
    st = wnga_destroy(a);
    if(st==FALSE)GA_Error("GA (c) destroy failed",g_a);
}

void GA_Set_memory_limit(size_t limit)
{
Integer lim = (Integer)limit;
     wnga_set_memory_limit(lim);
}

void NGA_Set_memory_limit(size_t limit)
{
Integer lim = (Integer)limit;
     wnga_set_memory_limit(lim);
}

void GA_Zero(int g_a)
{
    Integer a=(Integer)g_a;
    wnga_zero(a);
}

void NGA_Zero(int g_a)
{
    Integer a=(Integer)g_a;
    wnga_zero(a);
}

int GA_Pgroup_get_default()
{
    int value = (int)wnga_pgroup_get_default();
    return value;
}

int NGA_Pgroup_get_default()
{
    int value = (int)wnga_pgroup_get_default();
    return value;
}

void GA_Pgroup_set_default(int p_handle)
{
    Integer grp = (Integer)p_handle;
    wnga_pgroup_set_default(grp);
}

void NGA_Pgroup_set_default(int p_handle)
{
    Integer grp = (Integer)p_handle;
    wnga_pgroup_set_default(grp);
}

int GA_Pgroup_get_mirror()
{
    int value = (int)wnga_pgroup_get_mirror();
    return value;
}

int NGA_Pgroup_get_mirror()
{
    int value = (int)wnga_pgroup_get_mirror();
    return value;
}

int GA_Pgroup_get_world()
{
    int value = (int)wnga_pgroup_get_world();
    return value;
}

int NGA_Pgroup_get_world()
{
    int value = (int)wnga_pgroup_get_world();
    return value;
}

int GA_Idot(int g_a, int g_b)
{
    int value;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    wnga_dot(C_INT, a, b, &value);
    return value;
}


long GA_Ldot(int g_a, int g_b)
{
    long value;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    wnga_dot(C_LONG, a, b, &value);
    return value;
}


long long GA_Lldot(int g_a, int g_b)
{
    long long value;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    wnga_dot(C_LONGLONG, a, b, &value);
    return value;
}

     
double GA_Ddot(int g_a, int g_b)
{
    double value;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    wnga_dot(C_DBL, a, b, &value);
    return value;
}


DoubleComplex GA_Zdot(int g_a, int g_b)
{
    DoubleComplex value;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    wnga_dot(pnga_type_f2c(MT_F_DCPL),a,b,&value);
    return value;
}


SingleComplex GA_Cdot(int g_a, int g_b)
{
    SingleComplex value;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    wnga_dot(pnga_type_f2c(MT_F_SCPL),a,b,&value);
    return value;
}


float GA_Fdot(int g_a, int g_b)
{
    float sum;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    wnga_dot(C_FLOAT, a, b, &sum);
    return sum;
}    


void GA_Randomize(int g_a, void *value)
{
    Integer a=(Integer)g_a;
    wnga_randomize(a, value);
}

void NGA_Randomize(int g_a, void *value)
{
    Integer a=(Integer)g_a;
    wnga_randomize(a, value);
}

void GA_Fill(int g_a, void *value)
{
    Integer a=(Integer)g_a;
    wnga_fill(a, value);
}

void NGA_Fill(int g_a, void *value)
{
    Integer a=(Integer)g_a;
    wnga_fill(a, value);
}

void GA_Scale(int g_a, void *value)
{
    Integer a=(Integer)g_a;
    wnga_scale(a,value);
}


void GA_Add(void *alpha, int g_a, void* beta, int g_b, int g_c)
{
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    Integer c=(Integer)g_c;
    wnga_add(alpha, a, beta, b, c);
}


void GA_Copy(int g_a, int g_b)
{
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    wnga_copy(a, b);
}


void NGA_Get(int g_a, int lo[], int hi[], void* buf, int ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_get(a, _ga_lo, _ga_hi, buf, _ga_work);
}

void NGA_Get64(int g_a, int64_t lo[], int64_t hi[], void* buf, int64_t ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_get(a, _ga_lo, _ga_hi, buf, _ga_work);
}

void NGA_NbGet(int g_a, int lo[], int hi[], void* buf, int ld[],
               ga_nbhdl_t* nbhandle)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_nbget(a, _ga_lo, _ga_hi, buf, _ga_work,(Integer *)nbhandle);
}

void NGA_NbGet64(int g_a, int64_t lo[], int64_t hi[], void* buf, int64_t ld[],
               ga_nbhdl_t* nbhandle)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_nbget(a, _ga_lo, _ga_hi, buf, _ga_work,(Integer *)nbhandle);
}

void NGA_Put(int g_a, int lo[], int hi[], void* buf, int ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_put(a, _ga_lo, _ga_hi, buf, _ga_work);
}    

void NGA_Put64(int g_a, int64_t lo[], int64_t hi[], void* buf, int64_t ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_put(a, _ga_lo, _ga_hi, buf, _ga_work);
}    

void NGA_NbPut(int g_a, int lo[], int hi[], void* buf, int ld[],
               ga_nbhdl_t* nbhandle)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_nbput(a, _ga_lo, _ga_hi, buf, _ga_work,(Integer *)nbhandle);
}

void NGA_NbPut64(int g_a, int64_t lo[], int64_t hi[], void* buf, int64_t ld[],
                 ga_nbhdl_t* nbhandle)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_nbput(a, _ga_lo, _ga_hi, buf, _ga_work,(Integer *)nbhandle);
}

int NGA_NbTest(ga_nbhdl_t* nbhandle)
{
    return(wnga_nbtest((Integer *)nbhandle));
}

void NGA_NbWait(ga_nbhdl_t* nbhandle)
{
    wnga_nbwait((Integer *)nbhandle);
}

void NGA_Strided_acc(int g_a, int lo[], int hi[], int skip[],
                     void* buf, int ld[], void *alpha)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM], _ga_skip[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    COPYC2F(skip, _ga_skip, ndim);
    wnga_strided_acc(a, _ga_lo, _ga_hi, _ga_skip, buf, _ga_work, alpha);
}    

void NGA_Strided_acc64(int g_a, int64_t lo[], int64_t hi[], int64_t skip[],
                     void* buf, int64_t ld[], void *alpha)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_skip[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    COPYC2F(skip, _ga_skip, ndim);
    wnga_strided_acc(a, _ga_lo, _ga_hi, _ga_skip, buf, _ga_work, alpha);
}

void NGA_Strided_get(int g_a, int lo[], int hi[], int skip[],
                     void* buf, int ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_skip[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    COPYC2F(skip, _ga_skip, ndim);
    wnga_strided_get(a, _ga_lo, _ga_hi, _ga_skip, buf, _ga_work);
}    

void NGA_Strided_get64(int g_a, int64_t lo[], int64_t hi[], int64_t skip[],
                     void* buf, int64_t ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_skip[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    COPYC2F(skip, _ga_skip, ndim);
    wnga_strided_get(a, _ga_lo, _ga_hi, _ga_skip, buf, _ga_work);
}

void NGA_Strided_put(int g_a, int lo[], int hi[], int skip[],
                     void* buf, int ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_skip[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    COPYC2F(skip, _ga_skip, ndim);
    wnga_strided_put(a, _ga_lo, _ga_hi, _ga_skip, buf, _ga_work);
}    

void NGA_Strided_put64(int g_a, int64_t lo[], int64_t hi[], int64_t skip[],
                     void* buf, int64_t ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_skip[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    COPYC2F(skip, _ga_skip, ndim);
    wnga_strided_put(a, _ga_lo, _ga_hi, _ga_skip, buf, _ga_work);
}

void NGA_Acc(int g_a, int lo[], int hi[], void* buf,int ld[], void* alpha)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_acc(a, _ga_lo, _ga_hi, buf, _ga_work, alpha);
}    

void NGA_Acc64(int g_a, int64_t lo[], int64_t hi[], void* buf, int64_t ld[], void* alpha)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_acc(a, _ga_lo, _ga_hi, buf, _ga_work, alpha);
}    

void NGA_NbAcc(int g_a, int lo[], int hi[], void* buf,int ld[], void* alpha,
               ga_nbhdl_t* nbhandle)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_nbacc(a, _ga_lo,_ga_hi,buf,_ga_work,alpha,(Integer *)nbhandle);
}

void NGA_NbAcc64(int g_a, int64_t lo[], int64_t hi[], void* buf,int64_t ld[], void* alpha,
               ga_nbhdl_t* nbhandle)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    Integer _ga_work[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    wnga_nbacc(a, _ga_lo,_ga_hi,buf,_ga_work,alpha,(Integer *)nbhandle);
}

long NGA_Read_inc(int g_a, int subscript[], long inc)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer in=(Integer)inc;
    Integer _ga_lo[MAXDIM];
    COPYINDEX_C2F(subscript, _ga_lo, ndim);
    return (long)wnga_read_inc(a, _ga_lo, in);
}

long NGA_Read_inc64(int g_a, int64_t subscript[], long inc)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer in=(Integer)inc;
    Integer _ga_lo[MAXDIM];
    COPYINDEX_C2F(subscript, _ga_lo, ndim);
    return (long)wnga_read_inc(a, _ga_lo, in);
}

void NGA_Distribution(int g_a, int iproc, int lo[], int hi[])
{
     Integer a=(Integer)g_a;
     Integer p=(Integer)iproc;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
     wnga_distribution(a, p, _ga_lo, _ga_hi);
     COPYINDEX_F2C(_ga_lo,lo, ndim);
     COPYINDEX_F2C(_ga_hi,hi, ndim);
}

void NGA_Distribution64(int g_a, int iproc, int64_t lo[], int64_t hi[])
{
     Integer a=(Integer)g_a;
     Integer p=(Integer)iproc;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
     wnga_distribution(a, p, _ga_lo, _ga_hi);
     COPYINDEX_F2C_64(_ga_lo,lo, ndim);
     COPYINDEX_F2C_64(_ga_hi,hi, ndim);
}

int GA_Compare_distr(int g_a, int g_b)
{
    logical st;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    st = wnga_compare_distr(a,b);
    if(st == TRUE) return 0;
    else return 1;
}

int NGA_Compare_distr(int g_a, int g_b)
{
    logical st;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    st = wnga_compare_distr(a,b);
    if(st == TRUE) return 0;
    else return 1;
}

void NGA_Access(int g_a, int lo[], int hi[], void *ptr, int ld[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
     Integer _ga_work[MAXDIM];
     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);

     wnga_access_ptr(a,_ga_lo, _ga_hi, ptr, _ga_work);
     COPYF2C(_ga_work,ld, ndim-1);
}

void NGA_Access64(int g_a, int64_t lo[], int64_t hi[], void *ptr, int64_t ld[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
     Integer _ga_work[MAXDIM];
     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);

     wnga_access_ptr(a,_ga_lo, _ga_hi, ptr, _ga_work);
     COPYF2C_64(_ga_work,ld, ndim-1);
}

void NGA_Access_block(int g_a, int idx, void *ptr, int ld[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer iblock = (Integer)idx;
     Integer _ga_work[MAXDIM];
     wnga_access_block_ptr(a,iblock,ptr,_ga_work);
     COPYF2C(_ga_work,ld, ndim-1);
}

void NGA_Access_block64(int g_a, int64_t idx, void *ptr, int64_t ld[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer iblock = (Integer)idx;
     Integer _ga_work[MAXDIM];
     wnga_access_block_ptr(a,iblock,ptr,_ga_work);
     COPYF2C_64(_ga_work,ld, ndim-1);
}

void NGA_Access_block_grid(int g_a, int index[], void *ptr, int ld[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer _ga_work[MAXDIM], _ga_lo[MAXDIM];
     COPYC2F(index, _ga_lo, ndim);
     wnga_access_block_grid_ptr(a,_ga_lo,ptr,_ga_work);
     COPYF2C(_ga_work,ld, ndim-1);
}

void NGA_Access_block_grid64(int g_a, int64_t index[], void *ptr, int64_t ld[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM], _ga_work[MAXDIM];
     COPYC2F(index, _ga_lo, ndim);
     wnga_access_block_grid_ptr(a,_ga_lo,ptr,_ga_work);
     COPYF2C_64(_ga_work,ld, ndim-1);
}

void NGA_Access_block_segment(int g_a, int proc, void *ptr, int *len)
{
     Integer a=(Integer)g_a;
     Integer iblock = (Integer)proc;
     Integer ilen = (Integer)(*len);
     wnga_access_block_segment_ptr(a,iblock,ptr,&ilen);
     *len = (int)ilen;
}

void NGA_Access_block_segment64(int g_a, int proc, void *ptr, int64_t *len)
{
     Integer a=(Integer)g_a;
     Integer iblock = (Integer)proc;
     Integer ilen = (Integer)(*len);
     wnga_access_block_segment_ptr(a,iblock,ptr,&ilen);
     *len = (int64_t)ilen;
}

void NGA_Release(int g_a, int lo[], int hi[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);

     wnga_release(a,_ga_lo, _ga_hi);
}

void NGA_Release64(int g_a, int64_t lo[], int64_t hi[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);

     wnga_release(a,_ga_lo, _ga_hi);
}

void NGA_Release_block(int g_a, int idx)
{
     Integer a=(Integer)g_a;
     Integer iblock = (Integer)idx;

     wnga_release_block(a, iblock);
}

void NGA_Release_block_grid(int g_a, int index[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM];
     COPYINDEX_C2F(index,_ga_lo,ndim);

     wnga_release_block_grid(a, _ga_lo);
}

void NGA_Release_block_segment(int g_a, int idx)
{
     Integer a=(Integer)g_a;
     Integer iproc = (Integer)idx;

     wnga_release_block_segment(a, iproc);
}

int NGA_Locate(int g_a, int subscript[])
{
    logical st;
    Integer a=(Integer)g_a, owner;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM];
    COPYINDEX_C2F(subscript,_ga_lo,ndim);

    st = wnga_locate(a,_ga_lo,&owner);
    if(st == TRUE) return (int)owner;
    else return -1;
}

int NGA_Locate64(int g_a, int64_t subscript[])
{
    logical st;
    Integer a=(Integer)g_a, owner;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM];
    COPYINDEX_C2F(subscript,_ga_lo,ndim);

    st = wnga_locate(a,_ga_lo,&owner);
    if(st == TRUE) return (int)owner;
    else return -1;
}


int NGA_Locate_nnodes(int g_a, int lo[], int hi[])
{
     /* logical st; */
     Integer a=(Integer)g_a, np;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];

     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);
     /* st = wnga_locate_nnodes(a, _ga_lo, _ga_hi, &np); */
     (void)wnga_locate_nnodes(a, _ga_lo, _ga_hi, &np);
     return (int)np;
}


int NGA_Locate_nnodes64(int g_a, int64_t lo[], int64_t hi[])
{
     /* logical st; */
     Integer a=(Integer)g_a, np;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];

     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);
     /* st = wnga_locate_nnodes(a, _ga_lo, _ga_hi, &np); */
     (void)wnga_locate_nnodes(a, _ga_lo, _ga_hi, &np);
     return (int)np;
}


int NGA_Locate_region(int g_a,int lo[],int hi[],int map[],int procs[])
{
     logical st;
     Integer a=(Integer)g_a, np_guess, np_actual;
     Integer ndim = wnga_ndim(a);
     Integer *tmap;
     int i;
     Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
     Integer *_ga_map_capi;
     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);
     st = wnga_locate_nnodes(a, _ga_lo, _ga_hi, &np_guess);
     tmap = (Integer *)malloc( (int)(np_guess*2*ndim *sizeof(Integer)));
     if(!map)GA_Error("NGA_Locate_region: unable to allocate memory",g_a);
     _ga_map_capi = (Integer*)malloc(np_guess*sizeof(Integer));

     st = wnga_locate_region(a,_ga_lo, _ga_hi, tmap, _ga_map_capi, &np_actual);
     assert(np_guess == np_actual);
     if(st==FALSE){
       free(tmap);
       free(_ga_map_capi);
       return 0;
     }

     COPY(int,_ga_map_capi,procs, np_actual);

        /* might have to swap lo/hi when copying */

     for(i=0; i< np_actual*2; i++){
        Integer *ptmap = tmap+i*ndim;
        int *pmap = map +i*ndim;
        COPYINDEX_F2C(ptmap, pmap, ndim);  
     }
     free(tmap);
     free(_ga_map_capi);
     return (int)np_actual;
}

int NGA_Locate_region64(int g_a,int64_t lo[],int64_t hi[],int64_t map[],int procs[])
{
     logical st;
     Integer a=(Integer)g_a, np_guess, np_actual;
     Integer ndim = wnga_ndim(a);
     Integer *tmap;
     int i;
     Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
     Integer *_ga_map_capi;
     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);
     st = wnga_locate_nnodes(a, _ga_lo, _ga_hi, &np_guess);
     tmap = (Integer *)malloc( (int)(np_guess*2*ndim *sizeof(Integer)));
     if(!map)GA_Error("NGA_Locate_region: unable to allocate memory",g_a);
     _ga_map_capi = (Integer*)malloc(np_guess*sizeof(Integer));

     st = wnga_locate_region(a,_ga_lo, _ga_hi, tmap, _ga_map_capi, &np_actual);
     assert(np_guess == np_actual);
     if(st==FALSE){
       free(tmap);
       free(_ga_map_capi);
       return 0;
     }

     COPY(int,_ga_map_capi,procs, np_actual);

        /* might have to swap lo/hi when copying */

     for(i=0; i< np_actual*2; i++){
        Integer *ptmap = tmap+i*ndim;
        int64_t *pmap = map +i*ndim;
        COPYINDEX_F2C_64(ptmap, pmap, ndim);  
     }
     free(tmap);
     free(_ga_map_capi);
     return (int)np_actual;
}

int NGA_Locate_num_blocks(int g_a, int *lo, int *hi)
{
  Integer ret;
  Integer a = (Integer)g_a;
  Integer ndim = wnga_ndim(a);
  Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
  COPYINDEX_C2F(lo,_ga_lo,ndim);
  COPYINDEX_C2F(hi,_ga_hi,ndim);
  ret = wnga_locate_num_blocks(a, _ga_lo, _ga_hi);
  return (int)ret;
}


void NGA_Inquire(int g_a, int *type, int *ndim, int dims[])
{
     Integer a=(Integer)g_a;
     Integer ttype, nndim;
     Integer _ga_dims[MAXDIM];
     wnga_inquire(a,&ttype, &nndim, _ga_dims);
     COPYF2C(_ga_dims, dims,nndim);  
     *ndim = (int)nndim;
     *type = (int)ttype;
}

void NGA_Inquire64(int g_a, int *type, int *ndim, int64_t dims[])
{
     Integer a=(Integer)g_a;
     Integer ttype, nndim;
     Integer _ga_dims[MAXDIM];
     wnga_inquire(a,&ttype, &nndim, _ga_dims);
     COPYF2C_64(_ga_dims, dims,nndim);  
     *ndim = (int)nndim;
     *type = (int)ttype;
}

char* GA_Inquire_name(int g_a)
{
     Integer a=(Integer)g_a;
     char *ptr;
     wnga_inquire_name(a, &ptr);
     return(ptr);
}

char* NGA_Inquire_name(int g_a)
{
     Integer a=(Integer)g_a;
     char *ptr;
     wnga_inquire_name(a, &ptr);
     return(ptr);
}

size_t GA_Memory_avail(void)
{
    return (size_t)wnga_memory_avail();
}

size_t NGA_Memory_avail(void)
{
    return (size_t)wnga_memory_avail();
}

int GA_Memory_limited(void)
{
    if(wnga_memory_limited() ==TRUE) return 1;
    else return 0;
}

int NGA_Memory_limited(void)
{
    if(wnga_memory_limited() ==TRUE) return 1;
    else return 0;
}

void NGA_Proc_topology(int g_a, int proc, int coord[])
{
     Integer a=(Integer)g_a;
     Integer p=(Integer)proc;
     Integer ndim = wnga_ndim(a);
     Integer _ga_work[MAXDIM];
     wnga_proc_topology(a, p, _ga_work);
     COPY(int,_ga_work, coord,ndim);  
}

void GA_Get_proc_grid(int g_a, int dims[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer _ga_work[MAXDIM];
     wnga_get_proc_grid(a, _ga_work);
     COPY(int,_ga_work, dims ,ndim);  
}

void NGA_Get_proc_grid(int g_a, int dims[])
{
     Integer a=(Integer)g_a;
     Integer _ga_work[MAXDIM];
     Integer ndim = wnga_ndim(a);
     wnga_get_proc_grid(a, _ga_work);
     COPY(int,_ga_work, dims ,ndim);  
}

void GA_Check_handle(int g_a, char* string)
{
     Integer a=(Integer)g_a;
     wnga_check_handle(a,string);
}

int GA_Create_mutexes(int number)
{
     Integer n = (Integer)number;
     if(wnga_create_mutexes(n) == TRUE)return 1;
     else return 0;
}

int NGA_Create_mutexes(int number)
{
     Integer n = (Integer)number;
     if(wnga_create_mutexes(n) == TRUE)return 1;
     else return 0;
}

int GA_Destroy_mutexes(void) {
  if(wnga_destroy_mutexes() == TRUE) return 1;
  else return 0;
}

int NGA_Destroy_mutexes(void) {
  if(wnga_destroy_mutexes() == TRUE) return 1;
  else return 0;
}

void GA_Lock(int mutex)
{
     Integer m = (Integer)mutex;
     wnga_lock(m);
}

void NGA_Lock(int mutex)
{
     Integer m = (Integer)mutex;
     wnga_lock(m);
}

void GA_Unlock(int mutex)
{
     Integer m = (Integer)mutex;
     wnga_unlock(m);
}

void NGA_Unlock(int mutex)
{
     Integer m = (Integer)mutex;
     wnga_unlock(m);
}

void GA_Brdcst(void *buf, int lenbuf, int root)
{
  Integer type=GA_TYPE_BRD;
  Integer len = (Integer)lenbuf;
  Integer orig = (Integer)root;
  wnga_msg_brdcst(type, buf, len, orig);
}
   
void GA_Pgroup_brdcst(int grp_id, void *buf, int lenbuf, int root)
{
    Integer type=GA_TYPE_BRD;
    Integer len = (Integer)lenbuf;
    Integer orig = (Integer)root;
    Integer grp = (Integer)grp_id;
    wnga_pgroup_brdcst(grp, type, buf, len, orig);
}

void GA_Pgroup_sync(int grp_id)
{
    Integer grp = (Integer)grp_id;
    wnga_pgroup_sync(grp);
}

void NGA_Pgroup_sync(int grp_id)
{
    Integer grp = (Integer)grp_id;
    wnga_pgroup_sync(grp);
}

void GA_Gop(int type, void *x, int n, char *op)
{ wnga_gop(type, x, n, op); }

void GA_Igop(int x[], int n, char *op)
{ wnga_gop(C_INT, x, n, op); }

void GA_Lgop(long x[], int n, char *op)
{ wnga_gop(C_LONG, x, n, op); }

void GA_Llgop(long long x[], int n, char *op)
{ wnga_gop(C_LONGLONG, x, n, op); }

void GA_Fgop(float x[], int n, char *op)
{ wnga_gop(C_FLOAT, x, n, op); }       

void GA_Dgop(double x[], int n, char *op)
{ wnga_gop(C_DBL, x, n, op); }

void GA_Cgop(SingleComplex x[], int n, char *op)
{ wnga_gop(C_SCPL, x, n, op); }

void GA_Zgop(DoubleComplex x[], int n, char *op)
{ wnga_gop(C_DCPL, x, n, op); }

void NGA_Gop(int type, void *x, int n, char *op)
{ wnga_gop(type, x, n, op); }

void NGA_Igop(int x[], int n, char *op)
{ wnga_gop(C_INT, x, n, op); }

void NGA_Lgop(long x[], int n, char *op)
{ wnga_gop(C_LONG, x, n, op); }

void NGA_Llgop(long long x[], int n, char *op)
{ wnga_gop(C_LONGLONG, x, n, op); }

void NGA_Fgop(float x[], int n, char *op)
{ wnga_gop(C_FLOAT, x, n, op); }       

void NGA_Dgop(double x[], int n, char *op)
{ wnga_gop(C_DBL, x, n, op); }

void NGA_Cgop(SingleComplex x[], int n, char *op)
{ wnga_gop(C_SCPL, x, n, op); }

void NGA_Zgop(DoubleComplex x[], int n, char *op)
{ wnga_gop(C_DCPL, x, n, op); }

void GA_Pgroup_gop(int grp_id, int type, double x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, type, x, n, op); }

void GA_Pgroup_igop(int grp_id, int x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_INT, x, n, op); }
 
void GA_Pgroup_lgop(int grp_id, long x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_LONG, x, n, op); }
 
void GA_Pgroup_llgop(int grp_id, long long x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_LONGLONG, x, n, op); }
 
void GA_Pgroup_fgop(int grp_id, float x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_FLOAT, x, n, op); }
 
void GA_Pgroup_dgop(int grp_id, double x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_DBL, x, n, op); }
 
void GA_Pgroup_cgop(int grp_id, SingleComplex x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_SCPL, x, n, op); }
 
void GA_Pgroup_zgop(int grp_id, DoubleComplex x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_DCPL, x, n, op); }
 
void NGA_Pgroup_gop(int grp_id, int type, double x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, type, x, n, op); }

void NGA_Pgroup_igop(int grp_id, int x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_INT, x, n, op); }
 
void NGA_Pgroup_lgop(int grp_id, long x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_LONG, x, n, op); }
 
void NGA_Pgroup_llgop(int grp_id, long long x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_LONGLONG, x, n, op); }
 
void NGA_Pgroup_fgop(int grp_id, float x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_FLOAT, x, n, op); }
 
void NGA_Pgroup_dgop(int grp_id, double x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_DBL, x, n, op); }
 
void NGA_Pgroup_cgop(int grp_id, SingleComplex x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_SCPL, x, n, op); }
 
void NGA_Pgroup_zgop(int grp_id, DoubleComplex x[], int n, char *op)
{ wnga_pgroup_gop(grp_id, C_DCPL, x, n, op); }
 
void NGA_Alloc_gatscat_buf(int nelems)
{
  Integer elems = (Integer)nelems;
  wnga_alloc_gatscat_buf(elems);
}

void NGA_Free_gatscat_buf()
{
  wnga_free_gatscat_buf();
}

void NGA_Scatter(int g_a, void *v, int* subsArray[], int n)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
#ifndef USE_GATSCAT_NEW
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx][i] + 1;
    wnga_scatter(a, v, _subs_array, 0, nv);
    free(_subs_array);
#else
    wnga_scatter(a, v, subsArray, 1, nv);
#endif
}

void NGA_Scatter_flat(int g_a, void *v, int subsArray[], int n)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);

    /* adjust the indices for fortran interface */
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx*ndim+i] + 1;
    
    wnga_scatter(a, v, _subs_array, 0, nv);
    
    free(_subs_array);
}


void NGA_Scatter64(int g_a, void *v, int64_t* subsArray[], int64_t n)
{
    int64_t idx;
    int i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
#ifndef USE_GATSCAT_NEW
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx][i] + 1;
    wnga_scatter(a, v, _subs_array, 0, nv);
    free(_subs_array);
#else
    wnga_scatter(a, v, subsArray, 1, nv);
#endif
}

void NGA_Scatter_flat64(int g_a, void *v, int64_t subsArray[], int64_t n)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);

    /* adjust the indices for fortran interface */
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx*ndim+i] + 1;
    
    wnga_scatter(a, v, _subs_array, 0, nv);
    
    free(_subs_array);
}


void NGA_Scatter_acc(int g_a, void *v, int* subsArray[], int n, void *alpha)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
#ifndef USE_GATSCAT_NEW
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx][i] + 1;
    wnga_scatter_acc(a, v, _subs_array, 0, nv, alpha);
    free(_subs_array);
#else
    wnga_scatter_acc(a, v, subsArray, 1, nv, alpha);
#endif
}

void NGA_Scatter_acc_flat(int g_a, void *v, int subsArray[], int n, void *alpha)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);

    /* adjust the indices for fortran interface */
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx*ndim+i] + 1;
    
    wnga_scatter_acc(a, v, _subs_array, 0, nv, alpha);
    
    free(_subs_array);
}

void NGA_Scatter_acc64(int g_a, void *v, int64_t* subsArray[], int64_t n, void *alpha)
{
    int64_t idx;
    int i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
#ifndef USE_GATSCAT_NEW
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx][i] + 1;
    wnga_scatter_acc(a, v, _subs_array, 0, nv, alpha);
    free(_subs_array);
#else
    wnga_scatter_acc(a, v, subsArray, 1, nv, alpha);
#endif
}

void NGA_Scatter_acc_flat64(int g_a, void *v, int64_t subsArray[], int64_t n, void *alpha)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);

    /* adjust the indices for fortran interface */
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx*ndim+i] + 1;
    
    wnga_scatter_acc(a, v, _subs_array, 0, nv, alpha);
    
    free(_subs_array);
}

void NGA_Gather(int g_a, void *v, int* subsArray[], int n)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
#ifndef USE_GATSCAT_NEW
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);

    /* adjust the indices for fortran interface */
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx][i] + 1;
    wnga_gather(a, v, _subs_array, 0, nv);
    free(_subs_array);
#else
    wnga_gather(a, v, subsArray, 1, nv);
#endif
}


void NGA_Gather_flat(int g_a, void *v, int subsArray[], int n)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);

    /* adjust the indices for fortran interface */
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx*ndim+i] + 1;
    
    wnga_gather(a, v, _subs_array, 0, nv);
    
    free(_subs_array);
}


void NGA_Gather64(int g_a, void *v, int64_t* subsArray[], int64_t n)
{
    int64_t idx;
    int i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
#ifndef USE_GATSCAT_NEW
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);

    /* adjust the indices for fortran interface */
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx][i] + 1;
    wnga_gather(a, v, _subs_array, 0, nv);
    free(_subs_array);
#else
    wnga_gather(a, v, subsArray, 1, nv);
#endif
}


void NGA_Gather_flat64(int g_a, void *v, int64_t subsArray[], int64_t n)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
    Integer ndim = wnga_ndim(a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);

    /* adjust the indices for fortran interface */
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+(ndim-i-1)] = subsArray[idx*ndim+i] + 1;
    
    wnga_gather(a, v, _subs_array, 0, nv);
    
    free(_subs_array);
}

/* Patch related */

void NGA_Copy_patch(char trans, int g_a, int alo[], int ahi[],
                                int g_b, int blo[], int bhi[])
{
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_copy_patch(&trans, a, _ga_alo, _ga_ahi, b, _ga_blo, _ga_bhi);
}

void NGA_Copy_patch64(char trans, int g_a, int64_t alo[], int64_t ahi[],
                                  int g_b, int64_t blo[], int64_t bhi[])
{
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_copy_patch(&trans, a, _ga_alo, _ga_ahi, b, _ga_blo, _ga_bhi);
}


int NGA_Idot_patch(int g_a, char t_a, int alo[], int ahi[],
                   int g_b, char t_b, int blo[], int bhi[])
{
    int res;
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);

    return res;
}

long NGA_Ldot_patch(int g_a, char t_a, int alo[], int ahi[],
                    int g_b, char t_b, int blo[], int bhi[])
{
    long res;
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);

    return res;
}

long long NGA_Lldot_patch(int g_a, char t_a, int alo[], int ahi[],
                    int g_b, char t_b, int blo[], int bhi[])
{
    long res;
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);

    return res;
}

double NGA_Ddot_patch(int g_a, char t_a, int alo[], int ahi[],
                   int g_b, char t_b, int blo[], int bhi[])
{
    double res;
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);

    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);

    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);

    return res;
}

DoubleComplex NGA_Zdot_patch(int g_a, char t_a, int alo[], int ahi[],
                             int g_b, char t_b, int blo[], int bhi[])
{
    DoubleComplex res;
    
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);
    
    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    
    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);
    
    return res;
}

SingleComplex NGA_Cdot_patch(int g_a, char t_a, int alo[], int ahi[],
                             int g_b, char t_b, int blo[], int bhi[])
{
    SingleComplex res;
    
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);
    
    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    
    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);
    
    return res;
}

float NGA_Fdot_patch(int g_a, char t_a, int alo[], int ahi[],
                   int g_b, char t_b, int blo[], int bhi[])
{
    float res;
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);
 
    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
 
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
 
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
 
    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);
 
    return res;
}                                           

int NGA_Idot_patch64(int g_a, char t_a, int64_t alo[], int64_t ahi[],
                     int g_b, char t_b, int64_t blo[], int64_t bhi[])
{
    int res;
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);

    return res;
}

long NGA_Ldot_patch64(int g_a, char t_a, int64_t alo[], int64_t ahi[],
                     int g_b, char t_b, int64_t blo[], int64_t bhi[])
{
    long res;
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);

    return res;
}

long long NGA_Lldot_patch64(int g_a, char t_a, int64_t alo[], int64_t ahi[],
                     int g_b, char t_b, int64_t blo[], int64_t bhi[])
{
    long long res;
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);

    return res;
}

double NGA_Ddot_patch64(int g_a, char t_a, int64_t alo[], int64_t ahi[],
                        int g_b, char t_b, int64_t blo[], int64_t bhi[])
{
    double res;
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);

    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);

    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);

    return res;
}

DoubleComplex NGA_Zdot_patch64(int g_a, char t_a, int64_t alo[], int64_t ahi[],
                               int g_b, char t_b, int64_t blo[], int64_t bhi[])
{
    DoubleComplex res;
    
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);
    
    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    
    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);
    
    return res;
}

SingleComplex NGA_Cdot_patch64(int g_a, char t_a, int64_t alo[], int64_t ahi[],
                               int g_b, char t_b, int64_t blo[], int64_t bhi[])
{
    SingleComplex res;
    
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);
    
    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
    
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    
    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);
    
    return res;
}

float NGA_Fdot_patch64(int g_a, char t_a, int64_t alo[], int64_t ahi[],
                       int g_b, char t_b, int64_t blo[], int64_t bhi[])
{
    float res;
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);
 
    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);
 
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
 
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
 
    wnga_dot_patch(a, &t_a, _ga_alo, _ga_ahi,
                   b, &t_b, _ga_blo, _ga_bhi, &res);
 
    return res;
}


void NGA_Fill_patch(int g_a, int lo[], int hi[], void *val)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);

    wnga_fill_patch(a, _ga_lo, _ga_hi, val);
}

void NGA_Fill_patch64(int g_a, int64_t lo[], int64_t hi[], void *val)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);

    wnga_fill_patch(a, _ga_lo, _ga_hi, val);
}

void NGA_Zero_patch(int g_a, int lo[], int hi[])
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);

    wnga_zero_patch(a, _ga_lo, _ga_hi);
}

void NGA_Zero_patch64(int g_a, int64_t lo[], int64_t  hi[])
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);

    wnga_zero_patch(a, _ga_lo, _ga_hi);
}

void NGA_Scale_patch(int g_a, int lo[], int hi[], void *alpha)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);

    wnga_scale_patch(a, _ga_lo, _ga_hi, alpha);
}

void NGA_Scale_patch64(int g_a, int64_t lo[], int64_t hi[], void *alpha)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);

    wnga_scale_patch(a, _ga_lo, _ga_hi, alpha);
}

void NGA_Add_patch(void * alpha, int g_a, int alo[], int ahi[],
                   void * beta,  int g_b, int blo[], int bhi[],
                   int g_c, int clo[], int chi[])
{
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);

    Integer c=(Integer)g_c;
    Integer cndim = wnga_ndim(c);
    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    Integer _ga_clo[MAXDIM], _ga_chi[MAXDIM];

    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    COPYINDEX_C2F(clo,_ga_clo, cndim);
    COPYINDEX_C2F(chi,_ga_chi, cndim);

    wnga_add_patch(alpha, a, _ga_alo, _ga_ahi, beta, b, _ga_blo, _ga_bhi,
                   c, _ga_clo, _ga_chi);
}

void NGA_Add_patch64(void * alpha, int g_a, int64_t alo[], int64_t ahi[],
                     void * beta,  int g_b, int64_t blo[], int64_t bhi[],
                                   int g_c, int64_t clo[], int64_t chi[])
{
    Integer a=(Integer)g_a;
    Integer andim = wnga_ndim(a);

    Integer b=(Integer)g_b;
    Integer bndim = wnga_ndim(b);

    Integer c=(Integer)g_c;
    Integer cndim = wnga_ndim(c);

    Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
    Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
    Integer _ga_clo[MAXDIM], _ga_chi[MAXDIM];
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    COPYINDEX_C2F(clo,_ga_clo, cndim);
    COPYINDEX_C2F(chi,_ga_chi, cndim);

    wnga_add_patch(alpha, a, _ga_alo, _ga_ahi, beta, b, _ga_blo, _ga_bhi,
                   c, _ga_clo, _ga_chi);
}


void GA_Print_patch(int g_a,int ilo,int ihi,int jlo,int jhi,int pretty)
{
    Integer a = (Integer)g_a;
    Integer lo[2];
    Integer hi[2];
    Integer p = (Integer) pretty;
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];

    lo[0] = ilo; lo[1] = jlo;
    hi[0] = ihi; lo[1] = jhi;
    COPYINDEX_C2F(lo,_ga_lo,2);
    COPYINDEX_C2F(hi,_ga_hi,2);
    wnga_print_patch(a, _ga_lo, _ga_hi, p);
}


void NGA_Print_patch(int g_a, int lo[], int hi[], int pretty)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer p = (Integer)pretty;
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];

    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    wnga_print_patch(a, _ga_lo, _ga_hi, p);
}

void NGA_Print_patch64(int g_a, int64_t lo[], int64_t hi[], int pretty)
{
    Integer a=(Integer)g_a;
    Integer ndim = wnga_ndim(a);
    Integer p = (Integer)pretty;
    Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
  
    wnga_print_patch(a, _ga_lo, _ga_hi, p);
}

void GA_Print(int g_a)
{
    Integer a=(Integer)g_a;
    wnga_print(a);
}

void GA_Print_file(FILE *file, int g_a)
{
  Integer G_a = g_a;
  wnga_print_file(file, G_a);
}

void GA_Summarize(int verbose)
{
    Integer v = (Integer)verbose;

    wnga_summarize(v);
}

void GA_Transpose(int g_a, int g_b)
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;

    wnga_transpose(a, b);
}


void GA_Print_distribution(int g_a)
{
#ifdef USE_FAPI
    wnga_print_distribution(1,(Integer)g_a);
#else
    wnga_print_distribution(0,(Integer)g_a);
#endif
}


void NGA_Release_update(int g_a, int lo[], int hi[])
{
  Integer a = (Integer)g_a;
  Integer ndim = wnga_ndim(a);
  Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
  COPYINDEX_C2F(lo,_ga_lo,ndim);
  COPYINDEX_C2F(hi,_ga_hi,ndim);

  wnga_release_update(a,_ga_lo, _ga_hi);
}

void NGA_Release_update64(int g_a, int64_t lo[], int64_t hi[])
{
  Integer a = (Integer)g_a;
  Integer ndim = wnga_ndim(a);
  Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
  COPYINDEX_C2F(lo,_ga_lo,ndim);
  COPYINDEX_C2F(hi,_ga_hi,ndim);

  wnga_release_update(a,_ga_lo, _ga_hi);
}

void NGA_Release_update_block(int g_a, int idx)
{
     Integer a=(Integer)g_a;
     Integer iblock = (Integer)idx;

     wnga_release_update_block(a, iblock);
}

void NGA_Release_update_block_grid(int g_a, int index[])
{
     Integer a=(Integer)g_a;
     Integer ndim = wnga_ndim(a);
     Integer _ga_lo[MAXDIM];
     COPYINDEX_C2F(index,_ga_lo,ndim);
     wnga_release_update_block_grid(a, _ga_lo);
}

void NGA_Release_update_block_segment(int g_a, int idx)
{
     Integer a=(Integer)g_a;
     Integer iproc = (Integer)idx;

     wnga_release_update_block_segment(a, iproc);
}

int GA_Ndim(int g_a)
{
    Integer a = (Integer)g_a;
    return (int)wnga_ndim(a);
}

int NGA_Ndim(int g_a)
{
    Integer a = (Integer)g_a;
    return (int)wnga_ndim(a);
}

void GA_Elem_multiply(int g_a, int g_b, int g_c)
{
  Integer a = (Integer)g_a;
  Integer b = (Integer)g_b;
  Integer c = (Integer)g_c;
  wnga_elem_multiply(a, b, c);
}

void GA_Norm1(int g_a, double *nm){
  Integer a = (Integer )g_a;
  wnga_norm1(a, nm);
}

void GA_Norm_infinity(int g_a, double *nm){
  Integer a = (Integer )g_a;
  wnga_norm_infinity(a, nm);
}

/* return number of nodes being used in a cluster */
int GA_Cluster_nnodes()
{
    return wnga_cluster_nnodes();
} 

/* returns ClusterNode id of the calling process */
int GA_Cluster_nodeid() 
{
    return wnga_cluster_nodeid();
}

/* returns ClusterNode id of the specified process */
int GA_Cluster_proc_nodeid(int proc)
{
    Integer aproc = proc;
    return wnga_cluster_proc_nodeid(aproc);
}

/* return number of processes being used on the specified node */
int GA_Cluster_nprocs(int x) 
{
    Integer ax = x;
    return wnga_cluster_nprocs(ax);
}

/* global id of the calling process */
int GA_Cluster_procid(int node, int loc_proc)
{
    Integer anode = node;
    Integer aloc_proc = loc_proc;
    return wnga_cluster_procid(anode, aloc_proc);
}

/* wrapper for timer routines */
double GA_Wtime()
{
    return (double)wnga_wtime();
}

double NGA_Wtime()
{
    return (double)wnga_wtime();
}

void GA_Set_debug(int flag)
{
    Integer aa;
    aa = (Integer)flag;
    wnga_set_debug(aa);
}

void NGA_Set_debug(int flag)
{
    Integer aa;
    aa = (Integer)flag;
    wnga_set_debug(aa);
}

int GA_Get_debug()
{
    return (int)wnga_get_debug();
}

int NGA_Get_debug()
{
    return (int)wnga_get_debug();
}

#ifdef ENABLE_CHECKPOINT
void GA_Checkpoint(int* gas, int num)
{
    wnga_checkpoint_arrays(gas,num);
}
#endif

int GA_Pgroup_absolute_id(int grp_id, int pid) {
  Integer agrp_id = (Integer)grp_id;
  Integer apid = (Integer) pid;
  return (int)wnga_pgroup_absolute_id(agrp_id, apid);
}

int NGA_Pgroup_absolute_id(int grp_id, int pid) {
  Integer agrp_id = (Integer)grp_id;
  Integer apid = (Integer) pid;
  return (int)wnga_pgroup_absolute_id(agrp_id, apid);
}

void GA_Error(char *str, int code)
{
    Integer icode = code;
    wnga_error(str, icode);
}

void NGA_Error(char *str, int code)
{
    Integer icode = code;
    wnga_error(str, icode);
}

size_t GA_Inquire_memory()
{
    return (size_t)wnga_inquire_memory();
}

size_t NGA_Inquire_memory()
{
    return (size_t)wnga_inquire_memory();
}

void GA_Sync()
{
    wnga_sync();
}

void NGA_Sync()
{
    wnga_sync();
}

int GA_Uses_ma()
{
    return wnga_uses_ma();
}

int NGA_Uses_ma()
{
    return wnga_uses_ma();
}

void GA_Print_stats()
{
    wnga_print_stats();
}

void GA_Init_fence()
{
    wnga_init_fence();
}

void NGA_Init_fence()
{
    wnga_init_fence();
}

void GA_Fence()
{
    wnga_fence();
}

void NGA_Fence()
{
    wnga_fence();
}

int GA_Nodeid()
{
    return wnga_nodeid();
}

int NGA_Nodeid()
{
    return wnga_nodeid();
}

int GA_Nnodes()
{
    return wnga_nnodes();
}

int NGA_Nnodes()
{
    return wnga_nnodes();
}

static Integer* copy_map(int block[], int block_ndim, int map[])
{
    int d;
    int i,sum=0,capi_offset=0,map_offset=0;
    Integer *_ga_map_capi;

    for (d=0; d<block_ndim; d++) {
        sum += block[d];
    }

    _ga_map_capi = (Integer*)malloc(sum * sizeof(Integer));

    capi_offset = sum;
    for (d=0; d<block_ndim; d++) {
        capi_offset -= block[d];
        for (i=0; i<block[d]; i++) {
            _ga_map_capi[capi_offset+i] = map[map_offset+i] + 1;
        }
        map_offset += block[d];
    }

    return _ga_map_capi;
}

static Integer* copy_map64(int64_t block[], int block_ndim, int64_t map[])
{
    int d;
    int64_t i,sum=0,capi_offset=0,map_offset=0;
    Integer *_ga_map_capi;

    for (d=0; d<block_ndim; d++) {
        sum += block[d];
    }

    _ga_map_capi = (Integer*)malloc(sum * sizeof(Integer));

    capi_offset = sum;
    for (d=0; d<block_ndim; d++) {
        capi_offset -= block[d];
        for (i=0; i<block[d]; i++) {
            _ga_map_capi[capi_offset+i] = map[map_offset+i] + 1;
        }
        map_offset += block[d];
    }

    return _ga_map_capi;
}

int NGA_Register_type(size_t bytes) {
  return wnga_register_type(bytes);
}

int NGA_Deregister_type(int type) {
  return wnga_deregister_type(type);
}


void NGA_Get_field(int g_a, int *lo, int *hi, int foff, int fsize,
		   void *buf, int *ld) {
  Integer a = (Integer)g_a;
  Integer andim = wnga_ndim(a);
  Integer _alo[MAXDIM], _ahi[MAXDIM];
  Integer _ld[MAXDIM];
  COPYINDEX_C2F(lo,_alo, andim);
  COPYINDEX_C2F(hi,_ahi, andim);
  COPYINDEX_C2F(ld, _ld, andim-1);

  wnga_get_field(a, _alo, _ahi, foff, fsize, buf, _ld);
}

void NGA_Nbget_field(int g_a, int *lo, int *hi, int foff, int fsize,
		     void *buf, int *ld, ga_nbhdl_t *nbhandle) {
  Integer a = (Integer)g_a;
  Integer andim = wnga_ndim(a);
  Integer _alo[MAXDIM], _ahi[MAXDIM];
  Integer _ld[MAXDIM];
  COPYINDEX_C2F(lo,_alo, andim);
  COPYINDEX_C2F(hi,_ahi, andim);
  COPYINDEX_C2F(ld, _ld, andim-1);

  wnga_nbget_field(a, _alo, _ahi, foff, fsize, buf, _ld, (Integer*)nbhandle);
}

void NGA_Nbput_field(int g_a, int *lo, int *hi, int foff, int fsize,
		     void *buf, int *ld, ga_nbhdl_t *nbhandle) {
  Integer a = (Integer)g_a;
  Integer andim = wnga_ndim(a);
  Integer _alo[MAXDIM], _ahi[MAXDIM];
  Integer _ld[MAXDIM];
  COPYINDEX_C2F(lo,_alo, andim);
  COPYINDEX_C2F(hi,_ahi, andim);
  COPYINDEX_C2F(ld, _ld, andim-1);

  wnga_nbput_field(a, _alo, _ahi, foff, fsize, buf, _ld, (Integer*)nbhandle);
}

void NGA_Put_field(int g_a, int *lo, int *hi, int foff, int fsize,
		   void *buf, int *ld) {
  Integer a = (Integer)g_a;
  Integer andim = wnga_ndim(a);
  Integer _alo[MAXDIM], _ahi[MAXDIM];
  Integer _ld[MAXDIM];
  COPYINDEX_C2F(lo,_alo, andim);
  COPYINDEX_C2F(hi,_ahi, andim);
  COPYINDEX_C2F(ld, _ld, andim-1);

  wnga_put_field(a, _alo, _ahi, foff, fsize, buf, _ld);
}

void GA_Version(int *major, int *minor, int *patch)
{
  Integer maj, min, ptch;
  wnga_version(&maj,&min,&ptch);
  *major = (int)maj;
  *minor = (int)min;
  *patch = (int)ptch;
}

void NGA_Version(int *major, int *minor, int *patch)
{
  Integer maj, min, ptch;
  wnga_version(&maj,&min,&ptch);
  *major = (int)maj;
  *minor = (int)min;
  *patch = (int)ptch;
} 

int NGA_Sprs_array_create(int idim, int jdim, int type)
{
  Integer i = (Integer)idim;
  Integer j = (Integer)jdim;
  return (int)wnga_sprs_array_create(i,j,type,sizeof(int));
}

int NGA_Sprs_array_create64(int64_t idim, int64_t jdim, int type)
{
  Integer i = (Integer)idim;
  Integer j = (Integer)jdim;
  return (int)wnga_sprs_array_create(i,j,type,sizeof(int64_t));
}

int NGA_Sprs_array_create_from_dense(int g_a)
{
  Integer ga = (Integer)g_a;
  return (int)wnga_sprs_array_create_from_dense(ga,sizeof(int),1);
}

int NGA_Sprs_array_create_from_dense64(int g_a)
{
  Integer ga = (Integer)g_a;
  return (int)wnga_sprs_array_create_from_dense(ga,sizeof(int64_t),1);
}

int NGA_Sprs_array_create_from_sparse(int s_a)
{
  Integer sa = (Integer)s_a;
  return (int)wnga_sprs_array_create_from_sparse(sa,1);
}

int NGA_Sprs_array_quantize256(int s_a, void *max)
{
  Integer sa = (Integer)s_a;
  return (int)wnga_sprs_array_quantize256(sa, max);
}

void NGA_Sprs_array_add_element(int s_a, int idx, int jdx, void *val)
{
  Integer sa = (Integer)s_a;
  Integer i = (Integer)idx;
  Integer j = (Integer)jdx;
  wnga_sprs_array_add_element(sa,i,j,val);
}

void NGA_Sprs_array_add_element64(int s_a, int64_t idx, int64_t jdx, void *val)
{
  Integer sa = (Integer)s_a;
  Integer i = (Integer)idx;
  Integer j = (Integer)jdx;
  wnga_sprs_array_add_element(sa,i,j,val);
}

int NGA_Sprs_array_assemble(int s_a)
{
  Integer sa = (Integer)s_a;
  wnga_sprs_array_assemble(sa);
}

void NGA_Sprs_array_row_distribution(int s_a, int iproc, int *lo, int *hi)
{
  Integer sa = (Integer)s_a;
  Integer ip = (Integer)iproc;
  Integer ilo, ihi;
  wnga_sprs_array_row_distribution(sa,ip,&ilo,&ihi);
  *lo = (int)ilo;
  *hi = (int)ihi;
}

void NGA_Sprs_array_row_distribution64(int s_a, int iproc, int64_t *lo, int64_t *hi)
{
  Integer sa = (Integer)s_a;
  Integer ip = (Integer)iproc;
  Integer ilo, ihi;
  wnga_sprs_array_row_distribution(sa,ip,&ilo,&ihi);
  *lo = (int64_t)ilo;
  *hi = (int64_t)ihi;
}

void NGA_Sprs_array_column_distribution(int s_a, int iproc, int *lo, int *hi)
{
  Integer sa = (Integer)s_a;
  Integer ip = (Integer)iproc;
  Integer ilo, ihi;
  wnga_sprs_array_column_distribution(sa,ip,&ilo,&ihi);
  *lo = (int)ilo;
  *hi = (int)ihi;
}

void NGA_Sprs_array_column_distribution64(int s_a, int iproc, int64_t *lo, int64_t *hi)
{
  Integer sa = (Integer)s_a;
  Integer ip = (Integer)iproc;
  Integer ilo, ihi;
  wnga_sprs_array_column_distribution(sa,ip,&ilo,&ihi);
  *lo = (int64_t)ilo;
  *hi = (int64_t)ihi;
}

int NGA_Sprs_array_get_block(int s_a, int irow, int icol, int **idx,
    int **jdx, void **data, int *ilo, int *ihi, int *jlo, int *jhi)
{
  Integer sa = (Integer)s_a;
  Integer ir = (Integer)irow;
  Integer ic = (Integer)icol;
  Integer il, ih, jl, jh;
  void *id, *jd;
  int ret = wnga_sprs_array_get_block(sa, ir, ic, &id, &jd, data,
      &il, &ih, &jl, &jh);
  *ilo = (int)il-1;
  *ihi = (int)ih-1;
  *jlo = (int)jl-1;
  *jhi = (int)jh-1;
  *idx = (int*)id;
  *jdx = (int*)jd;
  return ret;
}

int NGA_Sprs_array_get_block64(int s_a, int64_t irow, int64_t icol,
    int64_t **idx, int64_t **jdx, void **data, int64_t *ilo,
    int64_t *ihi, int64_t *jlo, int64_t *jhi)
{
  Integer sa = (Integer)s_a;
  Integer ir = (Integer)irow;
  Integer ic = (Integer)icol;
  Integer il, ih, jl, jh;
  void *id, *jd;
  int ret = wnga_sprs_array_get_block(sa, ir, ic, &id, &jd, data,
      &il, &ih, &jl, &jh);
  *ilo = (int64_t)il-1;
  *ihi = (int64_t)ih-1;
  *jlo = (int64_t)jl-1;
  *jhi = (int64_t)jh-1;
  *idx = (int64_t*)id;
  *jdx = (int64_t*)jd;
  return ret;
}

void NGA_Sprs_array_access_col_block(int s_a, int icol, int **idx, int **jdx,
    void **val)
{
  Integer sa = (Integer)s_a;
  Integer ic = (Integer)icol;
  wnga_sprs_array_access_col_block(sa,ic,idx,jdx,val);
}

void NGA_Sprs_array_access_col_block64(int s_a, int icol, int64_t **idx,
    int64_t **jdx, void **val)
{
  Integer sa = (Integer)s_a;
  Integer ic = (Integer)icol;
  wnga_sprs_array_access_col_block(sa,ic,idx,jdx,val);
}

void NGA_Sprs_array_col_block_list(int s_a, int **idx, int *n)
{
  Integer *Idx;
  Integer i,N;
  Integer sa = s_a;
  wnga_sprs_array_col_block_list(sa, &Idx, &N);
  *n = N;
  *idx = (int*)malloc(N*sizeof(int));
  for (i=0; i<N; i++) {
    (*idx)[i] = (int)(Idx[i]);
  }
  free(Idx);
}

void NGA_Sprs_array_matvec_multiply(int s_a, int g_a, int g_v)
{
  Integer sa = (Integer)s_a;
  Integer ga = (Integer)g_a;
  Integer gv = (Integer)g_v;
  wnga_sprs_array_matvec_multiply(sa, ga, gv);
}

int NGA_Sprs_array_destroy(int s_a)
{
  Integer sa = (Integer)s_a;
  return wnga_sprs_array_destroy(sa);
}

void NGA_Sprs_array_export(int s_a, const char* file)
{
  Integer sa = (Integer)s_a;
  wnga_sprs_array_export(sa, file);
}

void NGA_Sprs_array_get_diag(int s_a, int *g_d)
{
  Integer sa = (Integer)s_a;
  Integer gd;
  wnga_sprs_array_get_diag(sa, &gd);
  *g_d = (int)gd;
}

void NGA_Sprs_array_diag_right_multiply(int s_a, int g_d)
{
  Integer sa = (Integer)s_a;
  Integer gd = (Integer)g_d;
  wnga_sprs_array_diag_right_multiply(sa, gd);
}

void NGA_Sprs_array_diag_left_multiply(int s_a, int g_d)
{
  Integer sa = (Integer)s_a;
  Integer gd = (Integer)g_d;
  wnga_sprs_array_diag_left_multiply(sa, gd);
}

void NGA_Sprs_array_shift_diag(int s_a, void *shift)
{
  Integer sa = (Integer)s_a;
  wnga_sprs_array_shift_diag(sa, shift);
}

int NGA_Sprs_array_duplicate(int s_a)
{
  Integer sa = (Integer)s_a;
  return (int)wnga_sprs_array_duplicate(sa);
}

int NGA_Sprs_array_matmat_multiply(int s_a, int s_b)
{
  int s_c;
  Integer sa = (Integer)s_a;
  Integer sb = (Integer)s_b;
  return (int)wnga_sprs_array_matmat_multiply(sa, sb);
}

int NGA_Sprs_array_sprsdns_multiply(int s_a, int g_b)
{
  int g_c;
  Integer sa = (Integer)s_a;
  Integer gb = (Integer)g_b;
  return (int)wnga_sprs_array_sprsdns_multiply(sa, gb, 1);
}

int NGA_Sprs_array_dnssprs_multiply(int g_a, int s_b)
{
  int g_c;
  Integer ga = (Integer)g_a;
  Integer sb = (Integer)s_b;
  return (int)wnga_sprs_array_dnssprs_multiply(ga, sb, 1);
}

int NGA_Sprs_array_elementwise_multiply(int s_a, int s_b)
{
  int s_c;
  Integer sa = (Integer)s_a;
  Integer sb = (Integer)s_b;
  return (int)wnga_sprs_array_elementwise_multiply(sa, sb);
}

int NGA_Sprs_array_activate_ReLU(int s_a, void *bias, int **categories,
    int *features)
{
  void **tcategories = (void**)categories;
  void **tfeatures = (void**)features;
  int sa = (int)s_a;
  return (int)wnga_sprs_array_activate_relu(sa, bias, tcategories,
      tfeatures);
}

int NGA_Sprs_array_activate_ReLU64(int s_a, void *bias, int64_t **categories,
    int64_t *features)
{
  void **tcategories = (void**)categories;
  void **tfeatures = (void**)features;
  int sa = (int)s_a;
  return (int)wnga_sprs_array_activate_relu(sa, bias, tcategories,
      tfeatures);
}

void NGA_Sprs_array_sampled_multiply(int g_a, int g_b, int s_c, int transB)
{
  /* Switch order of multiply and set transpose flag for C-interface */
  Integer ga = (Integer)g_a;
  Integer gb = (Integer)g_b;
  Integer sc = (Integer)s_c;
  Integer tb = (Integer)transB;
  wnga_sprs_array_sampled_multiply(gb, ga, sc, tb, 1);
}
