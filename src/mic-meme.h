#ifndef MIC_MEME
#define MIC_MEME

#include <omp.h>
#include "meme.h"
#define MY_DEBUG_SUBSEQ7 
//#define MY_DEBUG_MEME 
//#define MY_DEBUG_STARTS 
#define MIC_NUM_THREADS 224 //number of threads on MIC 
#define MIC_NUM_GROUPS 1 //number of thead group on MIC
#define MIC_RATIO_TASK 1 //ratio of workload on MIC

#define CPU_NUM_THREADS 24 //number of threads on CPU
#define CPU_NUM_GROUPS 1 //number of thead group on CPU
#define CPU_RATIO_TASK 1 //ratio of workload on CPU

#define MAX_NUM_THREADS 24 //maximum number of threads on CPU

#define MY_NUM_SAMPLES 200 //maximum number of sequences to handle in one iteration (to control the size of memory in one iteration)

typedef struct max_pair{
  int max; //maximum P{S_i,j , S_k,l}
  int max_j; //position of maxinum value  
  char pYic; //for revcomp
}MaxPair;

typedef struct taskinfo
{
  int flag; //whether or not valid
  int index; //index of task, reference to max_pair
  int j_start; //starting searching position of sequence  
  int j_end; //ending searching position of sequence 
  int k_end; //for revcomp
  int k_start; //for revcomp
}TaskInfo;

#pragma offload_attribute(push,target(mic))

  int micLmap[MAXALPH][MAXALPH];	/* consensus letter vs. log frequency matrix */

  int* py_offset; //offset of py (length - w + 1)
  uint8_t *res_buf; 
  uint8_t *resic_buf;
  int *log_not_o_buf;
  bool *skipArray; //equal to dataset->skip[samples[iseq]->group]

  int *res_offset; //offset of res_buf (length)
#pragma offload_attribute(pop)
  /*craete task for each thread*/
TaskInfo * createTask(  
  DATASET *dataset, 
  int totalLengthW, 
  int numThreads, 
  int w, 
  int ic, 
  int *taskCount);
#endif 
