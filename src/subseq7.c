/***********************************************************************
*								       *
*	MEME++							       *
*	Copyright 1994-2015, the Regents of the University of California*
*	Author: Timothy L. Bailey				       *
*								       *
***********************************************************************/
/* subseq7.c */
/* 5-23-00 tlb; change objective function to be log likelihood ratio for
  all models */
/* 7-12-99 tlb; move not_o out of pY calculation and move to local/global_max */
/* 7-12-99 tlb; multiply sw * not_o in m_step of align_top_subsequences so
  that erased starts will have low likelihood */
/* 7-01-99 tlb; multiply background counts by motif width for Tcm model */
/* 6-29-99 tlb; add reverse complement DNA strand support */

#include "calculate_p_y.h"
#include "heap.h"
#include "meme.h"
#include "psp.h"
#include "seed.h"
#include "sp_matrix.h"
#include "macros.h"
#include <assert.h>
#include <math.h> // For the "log" function; temporary.
#include <omp.h>
#include "mic-meme.h"
#define trace(Y)     fprintf(stderr,Y)

/* minimum probability of NOT overlapping a previous site for starting points */
#define MIN_NOT_O .1

/* use logs and integers */
#define LOG_THETA_TYPE(V)		int *V[2][MAXSITE]
#define LOG_THETAG_TYPE(V)		int *V[MAXSITE]



int my_mic_global_max(
  DATASET *dataset, /* the dataset */
  BOOLEAN negative, // control dataset?
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int thread_num,
  int w,    /* length of sites */ 
  P_PROB maxima,  /* array of encoded site starts of local maxima */
  BOOLEAN ic    /* use reverse complement, too */
);

__attribute__((target(mic))) void my_mic_get_pY(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * ioffRes,
  int w,        /* width of motif */
  int thread_rank,
  int thread_num,
  int n_samples
);

__attribute__((target(mic))) void my_mic_get_pYic(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * ioffRes,
  int w,        /* width of motif */
  int thread_rank,
  int thread_num,
  int n_samples
);



__attribute__((target(mic))) void my_mic_next_pY(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  int* my_last_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * ioffRes,
  int w,        /* width of motif */
  int thread_rank,
  int thread_num,
  int n_samples
);

__attribute__((target(mic))) void mic_next_pY_not_ic(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  int* my_last_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * res,
  int w,        /* width of motif */
  int ioff,
  int thread_rank,
  int thread_num,
  int n_samples
) ;

__attribute__((target(mic))) void my_mic_next_pYic(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  int* my_last_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * ioffRes,
  int w,        /* width of motif */
  int thread_rank,
  int thread_num,
  int n_samples
) ; 
__attribute__((target(mic))) void my_cpu_next_pYic(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  int* my_last_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * ioffRes,
  int w,        /* width of motif */
  int thread_rank,
  int thread_num,
  int n_samples
);

/* local functions */
static void next_pY(
  DATASET *dataset,			/* the dataset */
  LOG_THETAG_TYPE(theta_1),		/* integer log theta_1 */
  int w,				/* width of motif */
  int *theta_0,				/* first column of previous theta_1 */
  int pYindex				/* which pY array to use */
);
static void init_theta_1(
  int w,			/* width of site */
  uint8_t *res,			/* (encoded) letters of subsequence */
  LOG_THETAG_TYPE(theta_1),	/* theta_1 */
  int lmap[MAXALPH][MAXALPH] 	/* matrix of frequency vectors */ 
);
static int global_max(
  DATASET *dataset,	/* the dataset */
  BOOLEAN negative,	// control dataset?
  int w,		/* length of sites */ 
  P_PROB maxima, 	/* array of encoded site starts of local maxima */
  BOOLEAN ic 		/* use reverse complement, too */
);
static int local_max(
  DATASET *dataset,	/* the dataset */
  BOOLEAN negative,	// control dataset?
  int w,		/* length of sites */ 
  P_PROB maxima,  	/* array of encoded site starts of local maxima */
  BOOLEAN ic 		/* use reverse complement, too */
);

__attribute__((target(mic))) void funcheck()
{
#ifdef __MIC__
	printf("Exec on MIC\n");
#else
	printf("Exec on CPU\n");
#endif
}
/*calculate py*/
__attribute__((target(mic))) void mic_calculate_py(
  MaxPair *my_max_buf,
  TaskInfo* myThreadTasks,
  int lmap[MAXALPH][MAXALPH],
  int totalLengthW, 
  int myNumThreads,
  int myTaskCount,
  int maxLength,
  int last_seed_seq,
  int last_seed_pos,
  int init_off,
  int max_off,
  int my_init_iseq,
  int my_max_iseq,
  int n_samples,
  int ic,
  int w,
  int mic_flag,
  int my_num_threads,
  int my_num_groups
  );
/*calculate py, appropriate for datasets with small number of long sequences*/
__attribute__((target(mic))) void mic_calculate_py_loop_reverse(
  MaxPair *my_max_buf,
  TaskInfo* myThreadTasks,
  int lmap[MAXALPH][MAXALPH],
  int totalLengthW, 
  int myNumThreads,
  int myTaskCount,
  int maxLength,
  int last_seed_seq,
  int last_seed_pos,
  int init_off,
  int max_off,
  int my_init_iseq,
  int my_max_iseq,
  int n_samples,
  int ic,
  int w,
  int mic_flag,
  int my_num_threads,
  int my_num_groups
  );
/**********************************************************************/
/*
	subseq7	

	Try subsequences as starting points and choose the
	one which yields the highest score.
	Score is computed by:
		1) computing log p(Y | theta_1)
		2) finding the (sorted) postions of maximum log pY
		3) aligning the top NSITES0 scores for each value of NSITES0
		4) computing the expected likelihood for each value of NSITES0

	The computing of p(Y | theta_1) for successive
	subsequences (values of theta_1) is optimized by
	using p_i to calculate p_{i+1}.

	Returns number of starting points updated in s_points array.

	Updates s_points, array of starting points, one
	for each value of NSITES0 tried-- finds one \theta for each
	value of nsites0 specified in the input.

        NOTE: (22/11/06)
        The approach used to facilitate dynamic programming was left unchanged.
        HOWEVER, it could be "unified" with the dynamic programming approach
        used in branching_search, in which "SEED_DIFF" objects are used.
        This has not yet been done because "if it ain't broke, don't fix it".
*/
/**********************************************************************/
void subseq7(
  MODEL *model,			// the model
  DATASET *dataset,		/* the dataset */
  int w,			// w to use
  int n_nsites0,		/* number of nsites0 values to try */
  S_POINT s_points[],           /* array of starting points: 1 per nsites0 */
  HASH_TABLE evaluated_seed_ht 	/* A hash table used for remembering which seeds
                                   have been evaluated previously */
)
{
  MOTYPE mtype = model->mtype;		/* type of model */
  BOOLEAN ic = model->invcomp;		/* use reverse complement strand of DNA, too */
  LOG_THETA_TYPE(ltheta);   /* integer encoded log theta */
  int istep, iseq, ioff, bufIndex, i, j, k;
  int n_samples = dataset->n_samples;	/* number of samples in dataset */
  SAMPLE **samples = dataset->samples;	/* samples in dataset */
  int n_starts = 0;			/* number of sampled start subseq */
  //int n_maxima = ps(dataset, w);	/* upper bound on # maxima */
  /* the local maxima positions */
  //P_PROB maxima = (P_PROB) mymalloc(n_maxima * sizeof(p_prob));

  char *str_seed;                       // A string representation of a seed.
#ifdef MY_DEBUG_SUBSEQ7
  double my_time=0.0; 
#endif 
  // PRECONDITIONS:

  // 1. If the sequence model is oops, then n_nsites0 is exactly 1:
  if (mtype == Oops) {
    assert(n_nsites0 == 1);
  }
  int totalLengthW = 0;

  int micNumThreads = MIC_NUM_THREADS / MIC_NUM_GROUPS; //number of thread in one group on MIC
  int cpuNumThreads = CPU_NUM_THREADS / CPU_NUM_GROUPS; //number of thread in one group on CPU

  int maxLength = 0;
  for (iseq = 0; iseq < n_samples; iseq++)
  {
    if(samples[iseq]->length > maxLength)
      maxLength = samples[iseq]->length;
      py_offset[iseq] = totalLengthW;
      totalLengthW += samples[iseq]->length - w + 1;
  }
  py_offset[iseq] = totalLengthW;
#pragma offload_transfer target(mic:0) in(py_offset: length(n_samples + 1) alloc_if(0) free_if(0))

  int micTaskCount, cpuTaskCount; //number of tasks on MIC and CPU
  TaskInfo * micThreadTasks =  createTask(dataset, totalLengthW, micNumThreads, w, ic, &micTaskCount);
#pragma offload_transfer target(mic:0) in(micThreadTasks: length(micNumThreads * n_samples) alloc_if(1) free_if(0))
  TaskInfo * cpuThreadTasks =  createTask(dataset, totalLengthW, cpuNumThreads, w, ic, &cpuTaskCount);

  if (TRACE) { printf("w= %d\n", w); }
  /* get the probability that a site starting at position x_ij would
     NOT overlap a previously found motif.
  */
  get_not_o(dataset, w);

  // Set up log_not_o: log_not_o[site] is:
  // log ( Pr(site not overlapped) * scaled_to_one_Pr(site) )
  if (model->mtype != Tcm) {
    add_psp_to_log_not_o(dataset, w, model->invcomp, model->mtype);
  }
  
  for (iseq = 0; iseq < n_samples; iseq++)
  {
    memcpy(log_not_o_buf + res_offset[iseq], samples[iseq]->log_not_o, samples[iseq]->length * sizeof(int));
  }
#pragma offload_transfer target(mic:0) in(log_not_o_buf: length(res_offset[n_samples]) alloc_if(0) free_if(0))

  for(istep = 0; istep < n_samples; istep += MY_NUM_SAMPLES)
  {

    int max_off, init_off, max_iseq, init_iseq, seq_len;
    int mic_max_iseq, mic_init_iseq, mic_seq_len;
    int cpu_max_iseq, cpu_init_iseq, cpu_seq_len;
    init_iseq = istep;
    max_iseq = init_iseq +  MY_NUM_SAMPLES;
    if(max_iseq > n_samples)
      max_iseq = n_samples;
    printf(" %d, %d\n", init_iseq, max_iseq);

    init_off = 0;
    max_off = maxLength - w;
	/*distribute task to MIC and CPU*/
    int tmp_len = (max_iseq - init_iseq) * CPU_RATIO_TASK / (MIC_RATIO_TASK + CPU_RATIO_TASK);
    cpu_init_iseq = init_iseq;
    cpu_max_iseq = init_iseq + tmp_len;
    mic_init_iseq = cpu_max_iseq;
    mic_max_iseq = max_iseq;



    printf("mic %d, %d\n", mic_init_iseq, mic_max_iseq);
    printf("cpu %d, %d\n", cpu_init_iseq, cpu_max_iseq);

    seq_len = py_offset[max_iseq] - py_offset[init_iseq];
    mic_seq_len = py_offset[mic_max_iseq] - py_offset[mic_init_iseq];
    MaxPair * mic_max_buf = (MaxPair *)malloc(mic_seq_len* micTaskCount * sizeof(MaxPair));
#pragma offload_transfer target(mic:0) nocopy(mic_max_buf: length(mic_seq_len* micTaskCount) alloc_if(1) free_if(0))
    cpu_seq_len = py_offset[cpu_max_iseq] - py_offset[cpu_init_iseq];
    MaxPair * cpu_max_buf = (MaxPair *)malloc(cpu_seq_len* cpuTaskCount * sizeof(MaxPair));

  //printf("threadtask: %d * %d * %d\n", avglength, n_samples,  (mic_max_iseq - mic_init_iseq)/MIC_NUM_GROUPS);

  #ifdef MY_DEBUG_SUBSEQ7
    my_time = omp_get_wtime();
  #endif
    /* score all the sampled positions saving the best position for
       each value of NSITES0 */


      /*
        Loop over all subsequences in the current sequence testing them
        each as "starting points" (inital values) for theta
      */
  int last_seed_seq = dataset->last_seed_seq;
  int last_seed_pos = dataset->last_seed_pos;
  
 #pragma offload target (mic:0) mandatory\
     nocopy(micLmap, py_offset, res_buf, log_not_o_buf,  skipArray, micThreadTasks,\
      res_offset, resic_buf)\
     out(mic_max_buf: length(mic_seq_len* micTaskCount) alloc_if(0)  free_if(1))\
     signal(mic_max_buf)
{

if(n_samples < MAX_NUM_THREADS)//check the number of sequence of datasets and decide the algorithm
{
  printf("loop_reverse\n");
  mic_calculate_py_loop_reverse(
    mic_max_buf,
    micThreadTasks,
    micLmap,
    totalLengthW, 
    micNumThreads,
    micTaskCount,
    maxLength,
    last_seed_seq,
    last_seed_pos,
    init_off,
    max_off,
    mic_init_iseq,
    mic_max_iseq,
    n_samples,
    ic,
    w,
    1,
    MIC_NUM_THREADS,
    MIC_NUM_GROUPS
    );  
}
else  
{

  mic_calculate_py(
    mic_max_buf,
    micThreadTasks,
    micLmap,
    totalLengthW, 
    micNumThreads,
    micTaskCount,
    maxLength,
    last_seed_seq,
    last_seed_pos,
    init_off,
    max_off,
    mic_init_iseq,
    mic_max_iseq,
    n_samples,
    ic,
    w,
    1,
    MIC_NUM_THREADS,
    MIC_NUM_GROUPS
    );
}
}      

  #ifdef MY_DEBUG_SUBSEQ7
    //printf(" \n/************ mic loop %f *************/\n ", omp_get_wtime() - my_time);
    //my_time = omp_get_wtime();

  #endif
if(n_samples < MAX_NUM_THREADS)//check the number of sequence of datasets and decide the algorithm
{
  printf("loop_reverse\n");
  mic_calculate_py_loop_reverse(
    cpu_max_buf,
    cpuThreadTasks,
    micLmap,
    totalLengthW, 
    cpuNumThreads,
    cpuTaskCount,
    maxLength,
    last_seed_seq,
    last_seed_pos,
    init_off,
    max_off,
    cpu_init_iseq,
    cpu_max_iseq,
    n_samples,
    ic,
    w,
    0,
    CPU_NUM_THREADS,
    CPU_NUM_GROUPS
    );
}
else
{
  mic_calculate_py(
    cpu_max_buf,
    cpuThreadTasks,
    micLmap,
    totalLengthW, 
    cpuNumThreads,
    cpuTaskCount,
    maxLength,
    last_seed_seq,
    last_seed_pos,
    init_off,
    max_off,
    cpu_init_iseq,
    cpu_max_iseq,
    n_samples,
    ic,
    w,
    0,
    CPU_NUM_THREADS,
    CPU_NUM_GROUPS
    );
}

#pragma offload_wait target(mic:0) wait(mic_max_buf)

  #ifdef MY_DEBUG_SUBSEQ7
    printf(" \n/************ parallel loop %f *************/\n ", omp_get_wtime() - my_time);//print time cost 
  #endif

   #ifdef MY_DEBUG_SUBSEQ7
    my_time = omp_get_wtime();
  #endif   




    int * my_n_starts = (int *)malloc(seq_len  * sizeof(int));
    S_POINT * my_s_points = (S_POINT *)malloc(seq_len * n_nsites0 * sizeof(S_POINT));
    for (ioff = 0; ioff < seq_len ; ioff++) 
      memcpy(my_s_points + ioff * n_nsites0, s_points, n_nsites0 * sizeof(S_POINT));
    memset(my_n_starts, 0, seq_len  * sizeof(int));


  int num_threads2 = max_iseq - init_iseq;
  if(num_threads2 > MAX_NUM_THREADS) //check and decide the number of threads of parallelization  part 2
    num_threads2 = MAX_NUM_THREADS;
#pragma omp parallel  for num_threads(num_threads2) default(none) \
  shared(w, ic, dataset, mtype,  n_nsites0, my_n_starts, my_s_points, micNumThreads, mic_max_buf,\
    micTaskCount, micThreadTasks, mic_init_iseq, mic_max_iseq, py_offset, samples, NO_STATUS, stderr,\
    cpuNumThreads, cpu_max_buf,cpuTaskCount, cpuThreadTasks, cpu_init_iseq, cpu_max_iseq,\
    max_iseq, init_iseq)\
  private(iseq, ioff) schedule(dynamic)
    for (iseq = init_iseq; iseq < max_iseq; iseq++) {
      SAMPLE *s = samples[iseq];
      int lseq = s->length;
      uint8_t *res = s->res;        /* left to right */
      char *name = s->sample_name;
      double *not_o = s->not_o;
      int max_off, init_off;
      MaxPair * max_buf;
      int taskCount, numThreads;
      TaskInfo * threadTasks;
     if(iseq >= cpu_init_iseq & iseq <  cpu_max_iseq)//check if the task is handled by CPU 
      {
        taskCount = cpuTaskCount;
        threadTasks = cpuThreadTasks;
        numThreads = cpuNumThreads;
        max_buf = cpu_max_buf + (py_offset[iseq] - py_offset[cpu_init_iseq]) * taskCount;
      }
      else if(iseq >= mic_init_iseq & iseq <  mic_max_iseq)//check if the task is handled by MIC
      {
        taskCount = micTaskCount;
        threadTasks = micThreadTasks;
        numThreads = micNumThreads;
        max_buf = mic_max_buf + (py_offset[iseq] - py_offset[mic_init_iseq]) * taskCount;
      }


      int n_maxima = ps(dataset, w);  /* upper bound on # maxima */
      double col_scores[MAXSITE];   /* not used */
      P_PROB maxima = (P_PROB) mymalloc(n_maxima * sizeof(p_prob));

      if (lseq < w) continue;     /* shorter than motif */
      skip_sample_if_required(dataset, s);
      if (iseq > dataset->last_seed_seq) continue;  // limit seed words

      if ((!NO_STATUS) && ((iseq % 5) == 0)) {
        fprintf(stderr, "starts: w=%d, seq=%d, l=%d          \r", w, iseq, lseq); 
      }
      init_off = 0;
      max_off = lseq - w;
      /* the local maxima positions */

      for (ioff = init_off; ioff <= max_off; ioff+= 1) {/* subsequence */ 
        // limit seed words
        if (iseq == dataset->last_seed_seq && ioff > dataset->last_seed_pos) break; 
        if (not_o[ioff] < MIN_NOT_O) continue;

        S_POINT * ptr = my_s_points + (py_offset[iseq] - py_offset[init_iseq] +ioff) * n_nsites0;
		n_maxima = my_mic_global_max(dataset, FALSE, max_buf + ioff * taskCount, threadTasks, numThreads, w, maxima, ic); //select the highest - scoring substring S_k,maxk
        qsort((char *) maxima, n_maxima, sizeof(p_prob), pY_compare);//Sort the nsites0 highest-scoring substrings {S_k,maxk} in decreasing order of scores 
        /* "fake out" align_top_subsequences by setting each of the scores in
           the s_points objects to LITTLE, thereby forcing
           align_top_subsequences to record the attributes for the current seed
           in the s_points, rather than the seed with the highest respective
           scores: */
        int sp_idx;
        for (sp_idx = 0; sp_idx < n_nsites0; sp_idx++) {
          ptr[sp_idx].score = LITTLE;
        }
		//determine the potential starting points
        my_n_starts[py_offset[iseq] - py_offset[init_iseq] +ioff] = align_top_subsequences(
          mtype,
          w,
          dataset,
          iseq,
          ioff, 
          res+ioff,
          name,
          n_nsites0,
          n_maxima,
          maxima,
          col_scores,
          ptr
		  ); 

      } /* subsequence */
      free(maxima);
    } /* sequence */

  #ifdef MY_DEBUG_SUBSEQ7
    printf(" \n/************ parallel loop2 %f *************/\n ", omp_get_wtime() - my_time);
  #endif

   #ifdef MY_DEBUG_SUBSEQ7
    my_time = omp_get_wtime();
  #endif   
	//Update the hash map and starting point heap serially
    for (iseq = init_iseq; iseq < max_iseq; iseq++) {
      SAMPLE *s = samples[iseq];
      int lseq = s->length;
      uint8_t *res = s->res;        /* left to right */
      double *not_o = s->not_o;
      int init_off = 0;
      int max_off = lseq - w;
      for (ioff = init_off; ioff <= max_off; ioff++) {/* subsequence */ 
        // limit seed words
        if (iseq == dataset->last_seed_seq && ioff > dataset->last_seed_pos) break; 
        if (not_o[ioff] < MIN_NOT_O) continue;
        
        str_seed = to_str_seed(dataset->alph, res+ioff, w);
        hash_insert_str(str_seed, evaluated_seed_ht);
        S_POINT * ptr = my_s_points + (py_offset[iseq] - py_offset[init_iseq] +ioff) * n_nsites0;

        update_s_point_heaps(ptr, str_seed, n_nsites0);
        myfree(str_seed);
        
        n_starts += my_n_starts[py_offset[iseq] - py_offset[init_iseq] +ioff];
      }
    }
//#pragma offload_transfer target(mic:0) nocopy(mic_max_buf: alloc_if(0)  free_if(1))  



  #ifdef MY_DEBUG_SUBSEQ7
    printf(" \n/************ serial loop %f *************/\n ", omp_get_wtime() - my_time);
  #endif  
 
    free(my_s_points);
    free(my_n_starts);
    free(mic_max_buf);
    free(cpu_max_buf);

  }


#ifdef PARALLEL
  reduce_across_heaps(s_points, n_nsites0);
#endif // PARALLEL 

  //free(max_buf);

  // Print the sites predicted using the seed after subsequence search, for
  // each of the starting points, if requested:
  if (dataset->print_pred) {
    int sp_idx;
    for (sp_idx = 0; sp_idx < n_nsites0; sp_idx++) {
      // Retrieve the best seed, from the heap:
      HEAP *heap = s_points[sp_idx].seed_heap;
      // Only print sites for the s_point if its heap was non-empty:
      if (get_num_nodes(heap) > 0) {
        SEED *best_seed = (SEED *)get_node(heap, get_best_node(heap));
        char *seed = get_str_seed(best_seed);

        /* Print the sites predicted using the motif corresponding to that seed,
           according to the sequence model being used:
        */
        int nsites0 = s_points[sp_idx].nsites0;
        fprintf(stdout,
                "PREDICTED SITES AFTER SUBSEQUENCE SEARCH WITH W = %i "
                "NSITES = %i MOTIF = %i\n", w, nsites0, dataset->imotif);
        int n_maxima = ps(dataset, w); // upper bound on number of maxima
        P_PROB psites = (P_PROB) mymalloc(n_maxima * sizeof(p_prob));
        n_maxima = get_pred_sites(psites, mtype, w, seed, ltheta[1], micLmap,
                                  dataset, ic);
        print_site_array(psites, nsites0, stdout, w, dataset);
        myfree(psites);
      } // get_num_nodes > 0
    } //sp_idx
  } // print_pred

  if (TRACE){
    printf("Tested %d possible starts...\n", n_starts);
  }
#pragma offload_transfer target(mic:0) nocopy(micThreadTasks: alloc_if(0) free_if(1))

  free(micThreadTasks);
  free(cpuThreadTasks);

  //myfree(maxima);
} // subseq7

__attribute__((target(mic))) void mic_calculate_py(
  MaxPair *my_max_buf,
  TaskInfo* myThreadTasks,
  int lmap[MAXALPH][MAXALPH],
  int totalLengthW, 
  int myNumThreads,
  int myTaskCount,
  int maxLength,
  int last_seed_seq,
  int last_seed_pos,
  int init_off,
  int max_off,
  int my_init_iseq,
  int my_max_iseq,
  int n_samples,
  int ic,
  int w,
  int mic_flag,
  int my_num_threads,
  int my_num_groups
  )
{
  funcheck();
  int iseq, ioff, i;
  int* my_py_buf, * my_last_py_buf;
  my_py_buf = (int *)malloc(3 * (my_max_iseq - my_init_iseq) * totalLengthW * sizeof(int));
  my_last_py_buf = (int *)malloc(3 * (my_max_iseq - my_init_iseq) * totalLengthW * sizeof(int));

  memset(my_py_buf, 0, 3 * (my_max_iseq - my_init_iseq) * totalLengthW* sizeof(int));
  memset(my_last_py_buf, 0, 3 * (my_max_iseq - my_init_iseq) * totalLengthW* sizeof(int));
  int * in_pY_buf = my_last_py_buf;
  int * out_pY_buf = my_py_buf;
  bool *sense = (bool *)malloc( my_num_groups * sizeof(bool));
  omp_lock_t *mutex = (omp_lock_t *)malloc( my_num_groups * sizeof(omp_lock_t));
  int *currentNbThread = (int *)malloc( my_num_groups * sizeof(int));
  memset(currentNbThread, 0, my_num_groups * sizeof(int));
  for(i = 0; i < my_num_groups; i ++)
    omp_init_lock( &mutex[i]);
#pragma omp parallel num_threads(my_num_threads) default(none)  \
  shared(n_samples, lmap, ic, myNumThreads, my_max_buf, w, myThreadTasks, res_offset, myTaskCount, skipArray,\
    res_buf, maxLength, in_pY_buf, out_pY_buf, py_offset, totalLengthW, mic_flag,\
    my_max_iseq, my_init_iseq, last_seed_seq, last_seed_pos, my_num_groups, sense, mutex, currentNbThread)\
  private(ioff, iseq, init_off, max_off)
{
    int thread_rank = omp_get_thread_num();
    int my_rank = thread_rank / my_num_groups;
    for (iseq = thread_rank % my_num_groups + my_init_iseq; iseq < my_max_iseq; iseq+= my_num_groups)
    {
      int lseq = res_offset[iseq + 1] - res_offset[iseq];
      uint8_t *res = res_buf + res_offset[iseq];        /* left to right */
      int * out_py = out_pY_buf + (iseq - my_init_iseq) * 3 * totalLengthW;
      int * in_py = in_pY_buf + (iseq - my_init_iseq) * 3 * totalLengthW; 
      int *tmpy;
      MaxPair * max_buf = my_max_buf + (py_offset[iseq] - py_offset[my_init_iseq]) * myTaskCount;
      if (lseq < w) continue;     /* shorter than motif */
      if(skipArray[iseq]) continue;
      if (iseq > last_seed_seq) continue;  // limit seed words
      init_off = 0;
      max_off = lseq - w;
      for (ioff = init_off; ioff <= max_off; ioff++) {/* subsequence */ 
        // limit seed words
      

        if (iseq == last_seed_seq && ioff > last_seed_pos) break; 



        /* warning: always do the next step; don't ever
           "continue" or the value of pY will not be correct since
           it is computed based the previous value 
        */
        /* convert subsequence in dataset to starting point for EM */
        if (ioff == init_off) {       /* new sequence */

          /* Compute p(Y_ij | theta_1^0) */
          if (!ic) {
            my_mic_get_pY(out_py, max_buf + ioff * myTaskCount, myThreadTasks, lmap, res, w, my_rank, myNumThreads, n_samples);
          } else {
            my_mic_get_pYic(out_py, max_buf + ioff * myTaskCount, myThreadTasks, lmap, res, w, my_rank, myNumThreads, n_samples);
          }
        } else {          /* same sequence */
          /* get theta[0][0]^{k-1} */
          /* compute p(Y_ij | theta_1^k) */
          if (!ic) {
            mic_next_pY_not_ic(out_py, out_py, max_buf + ioff * myTaskCount, myThreadTasks, lmap, res, w, ioff, my_rank, myNumThreads, n_samples);
            //my_mic_next_pY(dataset, out_py, in_py, max_buf + ioff * micTaskCount, micThreadTasks, lmap, res+ioff, w, my_rank, numThreads);
          } else {
            if(mic_flag)
              my_mic_next_pYic(out_py, in_py, max_buf + ioff * myTaskCount, myThreadTasks, lmap, res+ioff, w, my_rank, myNumThreads, n_samples);
            else
              my_cpu_next_pYic(out_py, in_py, max_buf + ioff * myTaskCount, myThreadTasks, lmap, res+ioff, w, my_rank, myNumThreads, n_samples);

          }
        } /* same sequence */

        /* skip if there is a high probability that this subsequence
           is part of a site which has already been found 
        */

		/*synchronize and swap the input and output score vector*/
      if(ic)
      {
        int my_group = thread_rank % my_num_groups;

        const bool mySense = sense[my_group];
        omp_set_lock( &mutex[my_group] );
        const int nbThreadsArrived = (++currentNbThread[my_group]);
        omp_unset_lock( &mutex[my_group] );
        
 
        if(nbThreadsArrived == myNumThreads) {

            currentNbThread[my_group] = 0;
            sense[my_group] = !sense[my_group];

            #pragma omp flush(sense)
        }
        else {
            volatile const bool* const ptSense = &sense[my_group];
            while( (*ptSense) == mySense){
            }
        }
        tmpy  = in_py ;
        in_py = out_py ;
        out_py = tmpy ;  
      }
      
      
      }
    }
  }
  free(my_py_buf);
  free(my_last_py_buf);   
  for(i = 0; i < my_num_groups; i ++)
    omp_init_lock( &mutex[i]);   
  free(sense);
  free(mutex);
  free(currentNbThread);
}


__attribute__((target(mic))) void mic_calculate_py_loop_reverse(
  MaxPair *my_max_buf,
  TaskInfo* myThreadTasks,
  int lmap[MAXALPH][MAXALPH],
  int totalLengthW, 
  int myNumThreads,
  int myTaskCount,
  int maxLength,
  int last_seed_seq,
  int last_seed_pos,
  int init_off,
  int max_off,
  int my_init_iseq,
  int my_max_iseq,
  int n_samples,
  int ic,
  int w,
  int mic_flag,
  int my_num_threads,
  int my_num_groups
  )
{
  funcheck();
  int iseq, ioff;
  int* my_py_buf, * my_last_py_buf;
  my_py_buf = (int *)malloc(3 * (my_max_iseq - my_init_iseq) * totalLengthW * sizeof(int));
  my_last_py_buf = (int *)malloc(3 * (my_max_iseq - my_init_iseq) * totalLengthW * sizeof(int));

  memset(my_py_buf, 0, 3 * (my_max_iseq - my_init_iseq) * totalLengthW* sizeof(int));
  memset(my_last_py_buf, 0, 3 * (my_max_iseq - my_init_iseq) * totalLengthW* sizeof(int));
  int * in_pY_buf = my_last_py_buf;
  int * out_pY_buf = my_py_buf;
  int * tmp_pY_buf;
#pragma omp parallel num_threads(my_num_threads) default(none)  \
  shared(n_samples, lmap, ic, myNumThreads, my_max_buf, w, myThreadTasks, res_offset, myTaskCount, skipArray,\
    res_buf, maxLength, init_off, max_off, in_pY_buf, out_pY_buf, tmp_pY_buf, py_offset, totalLengthW, mic_flag,\
    my_max_iseq, my_init_iseq, last_seed_seq, last_seed_pos, my_num_groups)\
  private(ioff, iseq)
{
    int thread_rank = omp_get_thread_num();
    int my_rank = thread_rank / my_num_groups;
    for (ioff = init_off; ioff <= max_off; ioff++) {/* subsequence */ 
      // limit seed words
    for (iseq = thread_rank % my_num_groups + my_init_iseq; iseq < my_max_iseq; iseq+= my_num_groups)
    {
      int lseq = res_offset[iseq + 1] - res_offset[iseq];
      if(ioff > lseq-w) continue;
      if (lseq < w) continue;     /* shorter than motif */
      if(skipArray[iseq]) continue;
      if (iseq > last_seed_seq) continue;  // limit seed words
      if (iseq == last_seed_seq && ioff > last_seed_pos) break; 

      uint8_t *res = res_buf + res_offset[iseq];        /* left to right */
      int * out_py = out_pY_buf + (iseq - my_init_iseq) * 3 * totalLengthW;
      int * in_py = in_pY_buf + (iseq - my_init_iseq) * 3 * totalLengthW; 
      MaxPair * max_buf = my_max_buf + (py_offset[iseq] - py_offset[my_init_iseq]) * myTaskCount;

      /* warning: always do the next step; don't ever
         "continue" or the value of pY will not be correct since
         it is computed based the previous value 
      */
      /* convert subsequence in dataset to starting point for EM */
      if (ioff == init_off) {       /* new sequence */

        /* Compute p(Y_ij | theta_1^0) */
        if (!ic) {
          my_mic_get_pY(out_py, max_buf + ioff * myTaskCount, myThreadTasks, lmap, res, w, my_rank, myNumThreads, n_samples);
        } else {
          my_mic_get_pYic(out_py, max_buf + ioff * myTaskCount, myThreadTasks, lmap, res, w, my_rank, myNumThreads, n_samples);
        }
      } else {          /* same sequence */
        /* get theta[0][0]^{k-1} */
        /* compute p(Y_ij | theta_1^k) */
        if (!ic) {
          mic_next_pY_not_ic(out_py, out_py, max_buf + ioff * myTaskCount, myThreadTasks, lmap, res, w, ioff, my_rank, myNumThreads, n_samples);
          //my_mic_next_pY(dataset, out_py, in_py, max_buf + ioff * micTaskCount, micThreadTasks, lmap, res+ioff, w, my_rank, numThreads);
        } else {
          if(mic_flag)
            my_mic_next_pYic(out_py, in_py, max_buf + ioff * myTaskCount, myThreadTasks, lmap, res+ioff, w, my_rank, myNumThreads, n_samples);
          else
            my_cpu_next_pYic(out_py, in_py, max_buf + ioff * myTaskCount, myThreadTasks, lmap, res+ioff, w, my_rank, myNumThreads, n_samples);
        }
      } /* same sequence */

      /* skip if there is a high probability that this subsequence
         is part of a site which has already been found 
      */

    }
	/*synchronize and swap the input and output score vector*/
    if(ic)
    {
      #pragma omp barrier
      #pragma omp single
      {
        tmp_pY_buf  = in_pY_buf ;
        in_pY_buf  = out_pY_buf ;
        out_pY_buf  = tmp_pY_buf ;
      }
    }
    
    
    }
  }
  free(my_py_buf);
  free(my_last_py_buf);      
}
  /*select the highest-scoring substring S_k,maxk for each S_k depending on the local highest-scoring substrings {S_k,local_maxk }*/
int my_mic_global_max(
  DATASET *dataset, /* the dataset */
  BOOLEAN negative, // control dataset?
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int thread_num,
  int w,    /* length of sites */ 
  P_PROB maxima,  /* array of encoded site starts of local maxima */
  BOOLEAN ic    /* use reverse complement, too */
)
{
  int i, j, bufIndex;
  SAMPLE **samples = dataset->samples;    /* datset samples */
  int n_samples = dataset->n_samples;   /* number samples in dataset */
  int n_maxima = 0;       /* number of maxima found */
  /* find the position with maximum pY in each sequence */
   for(i = 0; i < n_samples; i++)
   {
     MaxPair * max_buf_ptr =  max_buf;
     int lseq = samples[i]->length;
     int max;
     int max_j;
     char pYic;
     int flag = 0;
     if (lseq < w) continue;    
//FIXME
//printf("GLOBAL i %d group %d skip %d\n", i, s->group, dataset->skip[s->group]);
     skip_sample_if_required(dataset, samples[i]);
     for(j = 0; j < thread_num; j ++)
     {
       TaskInfo* my_task = threadTasks + j* n_samples;
        if(!flag  && my_task[i].flag )
        {
           max = max_buf_ptr[my_task[i].index].max;
           max_j = max_buf_ptr[my_task[i].index].max_j;
           pYic = max_buf_ptr[my_task[i].index].pYic;   
           flag = 1;       
        }
       if(my_task[i].flag && max_buf_ptr[my_task[i].index].max > max){
         max = max_buf_ptr[my_task[i].index].max;
         max_j = max_buf_ptr[my_task[i].index].max_j;
         pYic = max_buf_ptr[my_task[i].index].pYic;
       }
       else if(my_task[i].flag && max_buf_ptr[my_task[i].index].max == max && max_buf_ptr[my_task[i].index].max_j < max_j)
       {
         max_j = max_buf_ptr[my_task[i].index].max_j;      
         pYic = max_buf_ptr[my_task[i].index].pYic;       
       }
     }
     maxima[n_maxima].x = i;
     maxima[n_maxima].y = max_j;
     maxima[n_maxima].ic = ic && pYic;  
     maxima[n_maxima].negative = FALSE;
     maxima[n_maxima].rank = -1;
     maxima[n_maxima].prob = max;
     n_maxima++;
  }
  return n_maxima;
} /* global_max */


__attribute__((target(mic))) void my_mic_get_pY(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * ioffRes,
  int w,        /* width of motif */
  int thread_rank,
  int thread_num,
  int n_samples
)
{
  int i, j, k;
  TaskInfo* myTask = threadTasks + thread_rank * n_samples;
  //#pragma omp parallel  for num_threads(4)\
  default(none) shared(samples, n_samples, pYindex, theta_1, w) private(i, j, k)
  for (i=0; i<n_samples; i++) {  /* sequence */
    if(!myTask[i].flag) continue;
    MaxPair * max_buf_ptr = &max_buf[myTask[i].index];
    int lseq = res_offset[i + 1] - res_offset[i];
    int *log_not_o = log_not_o_buf + res_offset[i];	
    uint8_t *res = res_buf + res_offset[i] ;  /* integer sequence */
    //int *pY = s->pY[pYindex];   /* p(Y_j | theta_1) */
    int *pY = my_py + py_offset[i];  
    if (lseq < w) continue;   /* skip if sequence too short */
    for (j=myTask[i].j_start; j<=myTask[i].j_end; j+=1) {   
      uint8_t *r = res + j;
      int p = 0;
      //#pragma ivdep
      for (k=0; k<w; k++) {   
        p += lmap[ioffRes[k]][(int)(*r++)];//theta_1[k][(int) (*r++)];
      }
      pY[j] = p;
	  //selects the local highest - scoring substring S_k, local_maxk
      if(j == myTask[i].j_start)
      {
        max_buf_ptr->max_j = j;
        max_buf_ptr->max = pY[j] + log_not_o[j]; 	
      }
      else if (pY[j] + log_not_o[j] > max_buf_ptr->max) {		
        max_buf_ptr->max_j = j;
        max_buf_ptr->max = pY[j] + log_not_o[j]; 		// log (pY * Pr(site) * Pr(no overlap))
      } 
    }
    //for (j=lseq-w+1; j<lseq; j++) pY[j] = 0;  /* impossible positions */
  }
}

__attribute__((target(mic))) void my_mic_get_pYic(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * ioffRes,
  int w,        /* width of motif */
  int thread_rank,
  int thread_num,
  int n_samples
)
{
  int i, j, k, kk;

  TaskInfo* myTask = threadTasks + thread_rank * n_samples;
  //#pragma omp parallel  for num_threads(4)\
  default(none) shared(samples, n_samples, pYindex, theta_1, w) private(i, j, k)
  for (i=0; i<n_samples; i++) {  /* sequence */
    if(!myTask[i].flag) continue;
    MaxPair * max_buf_ptr = &max_buf[myTask[i].index];
    int lseq = res_offset[i + 1] - res_offset[i];
    int *log_not_o = log_not_o_buf + res_offset[i]; 
    uint8_t *res = res_buf + res_offset[i] ;  /* integer sequence */
    uint8_t *resic = resic_buf + res_offset[i] ;  /* integer sequence */
    //int *pY = s->pY[pYindex];   /* p(Y_j | theta_1) */
    int *pY = my_py + py_offset[i];  
    int *pY1 = my_py + py_offset[i] + py_offset[n_samples];  
    int *pY2 = my_py + py_offset[i]+ 2 * py_offset[n_samples]; 
    char pYic_j,  pYic_k;//= my_pyic + i * maxLength;  
    if (lseq < w) continue;   /* skip if sequence too short */
    for (j=myTask[i].j_start, kk = myTask[i].k_start; j <= myTask[i].j_end && kk >= myTask[i].k_end; j+=1, kk -= 1) 
    {   /* site start */
      uint8_t *r = res + j;
      uint8_t *ric = resic + j;
      uint8_t *r_k = res + kk;
      uint8_t *ric_k = resic + kk;

      int p = 0;
      int pic = 0;
      int p_k = 0;
      int pic_k = 0;
      //#pragma ivdep
      for (k=0; k<w; k++) {   
        p += lmap[ioffRes[k]][(int)(*r++)];//theta_1[k][(int) (*r++)];
        pic += lmap[ioffRes[k]][(int)(*ric++)];
        p_k += lmap[ioffRes[k]][(int)(*r_k++)];//theta_1[k][(int) (*r++)];
        pic_k += lmap[ioffRes[k]][(int)(*ric_k++)];
      }
      pY1[j] = p;
      pY2[j] = pic;
      pY1[kk] = p_k;
      pY2[kk] = pic_k;
      //for revcomp
      if (pY2[kk] > pY1[j]) {    
        pYic_j = '\1'; pY[j] = pY2[kk];
      } else {       
        pYic_j = '\0'; pY[j] = pY1[j]; 
      }
      if (pY2[j] > pY1[kk]) {    
        pYic_k = '\1'; pY[kk] = pY2[j];
      } else {       
        pYic_k = '\0'; pY[kk] = pY1[kk]; 
      }
	  //selects the local highest - scoring substring S_k, local_maxk

      if(j == myTask[i].j_start)
      {
        max_buf_ptr->max_j = j;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[j] + log_not_o[j]; 
        max_buf_ptr->pYic = pYic_j;	 
      }
      else if (pY[j] + log_not_o[j] > max_buf_ptr->max) {		
        max_buf_ptr->max_j = j;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[j] + log_not_o[j]; 		// log (pY * Pr(site) * Pr(no overlap))
        max_buf_ptr->pYic = pYic_j;
      } 
      else if (pY[j] + log_not_o[j] == max_buf_ptr->max && j < max_buf_ptr->max_j) {	
        max_buf_ptr->max_j = j;
        max_buf_ptr->pYic = pYic_j;
      }   
      if (pY[kk] + log_not_o[kk] > max_buf_ptr->max) {		
        max_buf_ptr->max_j = kk;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[kk] + log_not_o[kk]; 		// log (pY * Pr(site) * Pr(no overlap))
        max_buf_ptr->pYic = pYic_k;
      } 
      else if (pY[kk] + log_not_o[kk] == max_buf_ptr->max && kk < max_buf_ptr->max_j) {		
        max_buf_ptr->max_j = kk;
        max_buf_ptr->pYic = pYic_k;
      }      

    }
    //for (j=lseq-w+1; j<lseq; j++) pY[j] = 0;  /* impossible positions */


  }
}

/**********************************************************************/
/*
	next_pY

	Compute the value of p(Y_ij | theta_1^{k+1})
	from p(Y_ij | theta_1^{k} and the probability
	of first letter of Y_ij given theta_1^k,
	p(Y_ij^0 | theta_1^k).
*/
/**********************************************************************/
static void next_pY(
  DATASET *dataset,			/* the dataset */
  LOG_THETAG_TYPE(theta_1),		/* integer log theta_1 */
  int w,				/* width of motif */
  int *theta_0,				/* first column of previous theta_1 */
  int pYindex				/* which pY array to use */
) {
  int i, k;
  int *theta_last = theta_1[w-1];	/* last column of theta_1 */
  int n_samples = dataset->n_samples;
  SAMPLE **samples = dataset->samples;
  
  for (i=0; i < n_samples; i++) { 	/* sequence */
    SAMPLE *s = samples[i];		/* sequence */
    int lseq = s->length;		/* length of sequence */
    uint8_t *res = pYindex<2 ? s->res : s->resic;	/* integer sequence */
    int *pY = s->pY[pYindex];		/* log p(Y_j | theta_1) */
    uint8_t *r = res+lseq-1;		/* last position in sequence */
    uint8_t *r0 = res+lseq-w-1;	        /* prior to start of last subsequence */
    int j, p;

    if (lseq < w) continue;		/* skip if sequence too short */
    skip_sample_if_required(dataset, s);

    /* calculate p(Y_ij | theta_1) */
    int *pY_shifted_1 = pY - 1;
    for (j=lseq-w; j>0; j--) {
      pY[j] = pY_shifted_1[j] + theta_last[(*r--)] - theta_0[(*r0--)];
    }

    /* calculate log p(Y_i0 | theta_1) */
    p = 0;
    r = res;
    for (k=0; k<w; k++) {     		/* position in site */
      p += theta_1[k][(*r++)];
    }
    pY[0] = p;
  }
}


__attribute__((target(mic))) void my_mic_next_pY(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  int* my_last_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * ioffRes,
  int w,        /* width of motif */
  int thread_rank,
  int thread_num,
  int n_samples
) {
  int i, k;
  int *theta_last = lmap[ioffRes[w-1]];//theta_1[w-1];  /* last column of theta_1 */
  int *theta_0 = lmap[(ioffRes - 1)[0]];

  TaskInfo* myTask = threadTasks + thread_rank * n_samples;
  for (i=0; i < n_samples; i++) {  /* sequence */
    if(!myTask[i].flag) continue;
    MaxPair * max_buf_ptr = &max_buf[myTask[i].index];

    int lseq = res_offset[i + 1] - res_offset[i];
    int *log_not_o = log_not_o_buf + res_offset[i]; 
    uint8_t *res = res_buf + res_offset[i] ;  /* integer sequence */
    int *pY = my_py + py_offset[i];  
    int *last_pY = my_last_py + py_offset[i];  

    int j, p;
    if (lseq < w) continue;   /* skip if sequence too short */
    if(skipArray[i]) continue;

    /* calculate p(Y_ij | theta_1) */
    int *pY_shifted_1 = last_pY - 1;
    for (j=myTask[i].j_start; j<=myTask[i].j_end; j+= 1){
      if(j == 0)
      {
        p = 0;
        uint8_t *r = res;
        for (k=0; k<w; k++) {         /* position in site */
          p += lmap[ioffRes[k]][(int)(*r++)];
        }
        pY[0] = p;
		//selects the local highest - scoring substring S_k, local_maxk

       if (pY[j] + log_not_o[j] > max_buf_ptr->max) {   /* new maximum found */
          max_buf_ptr->max_j = j;
          // FIXME: We are assumming that priors are always symmetrical here.
          max_buf_ptr->max = pY[j] + log_not_o[j];    // log (pY * Pr(site) * Pr(no overlap))
        } /* new maximum */
      }
      else
        pY[j] = pY_shifted_1[j] + theta_last[(*(res + w + j - 1))] - theta_0[(*(res + j - 1))];
	  //selects the local highest - scoring substring S_k, local_maxk
      if(j == myTask[i].j_start)
      {
        max_buf_ptr->max_j = j;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[j] + log_not_o[j]; 	   
      }
      else if (pY[j] + log_not_o[j] > max_buf_ptr->max) {		/* new maximum found */
        max_buf_ptr->max_j = j;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[j] + log_not_o[j]; 		// log (pY * Pr(site) * Pr(no overlap))
      } /* new maximum */
    }

    /* calculate log p(Y_i0 | theta_1) */

  }
}

/*improved iteration updating strategy*/
__attribute__((target(mic))) void mic_next_pY_not_ic(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  int* my_last_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * res,
  int w,        /* width of motif */
  int ioff,
  int thread_rank,
  int thread_num,
  int n_samples
) {
  int i, k;
  uint8_t *ioffRes = res + ioff;
  int *theta_last = lmap[ioffRes[w-1]];//theta_1[w-1];  /* last column of theta_1 */
  int *theta_0 = lmap[(ioffRes - 1)[0]];

  TaskInfo* myTask = threadTasks + thread_rank * n_samples;
  for (i=0; i < n_samples; i++) {  /* sequence */
    if(!myTask[i].flag) continue;
    MaxPair * max_buf_ptr = &max_buf[myTask[i].index];

    int lseq = res_offset[i + 1] - res_offset[i];
    int *log_not_o = log_not_o_buf + res_offset[i]; 
    uint8_t *res = res_buf + res_offset[i] ;  /* integer sequence */
    int *pY = my_py + py_offset[i];  
    int *last_pY = my_last_py + py_offset[i];  
    int j_start = (myTask[i].j_start + ioff) % (lseq - w + 1);
    int j_end = (myTask[i].j_end + ioff) % (lseq - w + 1);

    int j, p, index;
    if (lseq < w) continue;   /* skip if sequence too short */
    if(skipArray[i]) continue;

    /* calculate p(Y_ij | theta_1) */
    int *pY_shifted_1 = last_pY;
    if(j_start < j_end)
    {
      if(j_start == 0)
      {
        index = myTask[i].j_start;
        p = 0;
        uint8_t *r = res;
        //#pragma ivdep
        for (k=0; k<w; k++) {         /* position in site */
          p += lmap[ioffRes[k]][(int)(*r++)];
        }
        pY[index] = p;
        #pragma ivdep
        for (j = j_start + 1, index = myTask[i].j_start + 1; j<=j_end; j+= 1, index += 1){

            pY[index] = pY_shifted_1[index] + theta_last[(*(res + w + j - 1))] - theta_0[(*(res + j - 1))];
        }
      }
      else{
        #pragma ivdep
        for (j = j_start, index = myTask[i].j_start; j<=j_end; j+= 1, index += 1){

            pY[index] = pY_shifted_1[index] + theta_last[(*(res + w + j - 1))] - theta_0[(*(res + j - 1))];
        }
      }
      for (j = j_start, index = myTask[i].j_start; j<=j_end; j+= 1, index += 1){
		//selects the local highest - scoring substring S_k, local_maxk
        if(j == j_start)
        {
          max_buf_ptr->max_j = j;
          // FIXME: We are assumming that priors are always symmetrical here.
          max_buf_ptr->max = pY[index] + log_not_o[j];     
        }
        else if (pY[index] + log_not_o[j] > max_buf_ptr->max) {   /* new maximum found */
          max_buf_ptr->max_j = j;
          // FIXME: We are assumming that priors are always symmetrical here.
          max_buf_ptr->max = pY[index] + log_not_o[j];    // log (pY * Pr(site) * Pr(no overlap))
        } /* new maximum */
      }
    }
    else
    {
      #pragma ivdep
      for (j = j_start, index = myTask[i].j_start; j<=lseq-w; j+= 1, index += 1){

        pY[index] = pY_shifted_1[index] + theta_last[(*(res + w + j - 1))] - theta_0[(*(res + j - 1))];
      }
      for (j = j_start, index = myTask[i].j_start; j<=lseq-w; j+= 1, index += 1){
		//selects the local highest - scoring substring S_k, local_maxk
        if(j == j_start)
        {
          max_buf_ptr->max_j = j;
          // FIXME: We are assumming that priors are always symmetrical here.
          max_buf_ptr->max = pY[index] + log_not_o[j];     
        }
        else if (pY[index] + log_not_o[j] > max_buf_ptr->max) {   /* new maximum found */
          max_buf_ptr->max_j = j;
          // FIXME: We are assumming that priors are always symmetrical here.
          max_buf_ptr->max = pY[index] + log_not_o[j];    // log (pY * Pr(site) * Pr(no overlap))
        } /* new maximum */
      }
      #pragma ivdep
      for (j = j_end, index = myTask[i].j_end; j>0; j -= 1, index -= 1){
          pY[index] = pY_shifted_1[index] + theta_last[(*(res + w + j - 1))] - theta_0[(*(res + j - 1))];
      }
        //if(j == 0)
        {
          p = 0;
          uint8_t *r = res;
          //#pragma ivdep
          for (k=0; k<w; k++) {         /* position in site */
            p += lmap[ioffRes[k]][(int)(*r++)];
          }
          pY[index] = p;
        }
      for (j = j_end, index = myTask[i].j_end; j>=0; j -= 1, index -= 1){
		 //selects the local highest - scoring substring S_k, local_maxk
        if (pY[index] + log_not_o[j] > max_buf_ptr->max) {   /* new maximum found */
          max_buf_ptr->max_j = j;
          // FIXME: We are assumming that priors are always symmetrical here.
          max_buf_ptr->max = pY[index] + log_not_o[j];    // log (pY * Pr(site) * Pr(no overlap))
        } /* new maximum */
      }      
    }

    /* calculate log p(Y_i0 | theta_1) */

  }
}

__attribute__((target(mic))) void my_cpu_next_pYic(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  int* my_last_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * ioffRes,
  int w,        /* width of motif */
  int thread_rank,
  int thread_num,
  int n_samples
) {
  int i, k, kk;
  int *theta_last = lmap[ioffRes[w-1]];//theta_1[w-1];  /* last column of theta_1 */
  int *theta_0 = lmap[(ioffRes - 1)[0]];

  TaskInfo* myTask = threadTasks + thread_rank * n_samples;
  for (i=0; i < n_samples; i++) {  /* sequence */
    if(!myTask[i].flag) continue;
    MaxPair * max_buf_ptr = &max_buf[myTask[i].index];

    int lseq = res_offset[i + 1] - res_offset[i];
    int *log_not_o = log_not_o_buf + res_offset[i]; 
    uint8_t *res = res_buf + res_offset[i] ;  /* integer sequence */
    uint8_t *resic = resic_buf + res_offset[i] ; 
    int *pY = my_py + py_offset[i]; 
    int *pY1 = my_py + py_offset[i] + py_offset[n_samples]; 
    int *pY2 = my_py + py_offset[i] + 2 * py_offset[n_samples];  
    char pYic_j, pYic_k;// = my_pyic + i * maxLength;  
    int *last_pY1 = my_last_py + py_offset[i] + py_offset[n_samples]; 
    int *last_pY2 = my_last_py + py_offset[i] + 2 * py_offset[n_samples];  
    int j, p, pic;
    if (lseq < w) continue;   /* skip if sequence too short */
    if(skipArray[i]) continue;

    /* calculate p(Y_ij | theta_1) */
    int *pY_shifted_1_1 = last_pY1 - 1;
    int *pY_shifted_1_2 = last_pY2 - 1;
    for (j=myTask[i].j_start, kk = myTask[i].k_start; j <= myTask[i].j_end && kk >= myTask[i].k_end; j+=1, kk -= 1) 
    {
        pY1[kk] = pY_shifted_1_1[kk] + theta_last[(*(res + w + kk - 1))] - theta_0[(*(res + kk - 1))];
        pY2[kk] = pY_shifted_1_2[kk] + theta_last[(*(resic + w + kk - 1))] - theta_0[(*(resic + kk - 1))];

      if(j == 0)
      {
        uint8_t * _r = res;
        uint8_t * _ric = resic;
        p = 0;
        pic = 0;

        for (k=0; k<w; k++) {         /* position in site */
          p += lmap[ioffRes[k]][(int)(*_r++)];
          pic += lmap[ioffRes[k]][(int)(*_ric++)];
        }
        pY1[0] = p;
        pY2[0] = pic;

      }
      else{

        pY1[j] = pY_shifted_1_1[j] + theta_last[(*(res + w + j - 1))] - theta_0[(*(res + j - 1))];
        pY2[j] = pY_shifted_1_2[j] + theta_last[(*(resic + w + j - 1))] - theta_0[(*(resic + j - 1))];
      }
      if (pY2[kk] > pY1[j]) {    
        pYic_j = '\1'; pY[j] = pY2[kk];
      } else {       
        pYic_j = '\0'; pY[j] = pY1[j]; 
      }
      if (pY2[j] > pY1[kk]) {    
        pYic_k = '\1'; pY[kk] = pY2[j];
      } else {       
        pYic_k = '\0'; pY[kk] = pY1[kk]; 
      }
	  //selects the local highest - scoring substring S_k, local_maxk
      if(j == myTask[i].j_start)
      {
        max_buf_ptr->max_j = j;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[j] + log_not_o[j];
        max_buf_ptr->pYic = pYic_j;
      }
      else if (pY[j] + log_not_o[j] > max_buf_ptr->max) {   
        max_buf_ptr->max_j = j;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[j] + log_not_o[j];    // log (pY * Pr(site) * Pr(no overlap))
        max_buf_ptr->pYic = pYic_j;
      } 
      else if(pY[j] + log_not_o[j] == max_buf_ptr->max && j < max_buf_ptr->max_j) { 
        max_buf_ptr->max_j = j;
        max_buf_ptr->pYic = pYic_j;
      } 
      if (pY[kk] + log_not_o[kk] > max_buf_ptr->max) {    
        max_buf_ptr->max_j = kk;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[kk] + log_not_o[kk];    // log (pY * Pr(site) * Pr(no overlap))
        max_buf_ptr->pYic = pYic_k;
      }    
      else if(pY[kk] + log_not_o[kk] == max_buf_ptr->max && kk < max_buf_ptr->max_j) {    
        max_buf_ptr->max_j = kk;
        max_buf_ptr->pYic = pYic_k;
      } 

    }

    /* calculate log p(Y_i0 | theta_1) */

  }
}
__attribute__((target(mic))) void my_mic_next_pYic(
  //LOG_THETAG_TYPE(theta_1),   /* integer log theta_1 */
  int* my_py,
  int* my_last_py,
  MaxPair *max_buf,
  TaskInfo* threadTasks,
  int lmap[MAXALPH][MAXALPH],   /* integer log theta_1 */
  uint8_t * ioffRes,
  int w,        /* width of motif */
  int thread_rank,
  int thread_num,
  int n_samples
) {
  int i, k, kk;
  int *theta_last = lmap[ioffRes[w-1]];//theta_1[w-1];  /* last column of theta_1 */
  int *theta_0 = lmap[(ioffRes - 1)[0]];

  TaskInfo* myTask = threadTasks + thread_rank * n_samples;
  for (i=0; i < n_samples; i++) {  /* sequence */
    if(!myTask[i].flag) continue;
    MaxPair * max_buf_ptr = &max_buf[myTask[i].index];

    int lseq = res_offset[i + 1] - res_offset[i];
    int *log_not_o = log_not_o_buf + res_offset[i]; 
    uint8_t *res = res_buf + res_offset[i] ;  /* integer sequence */
    uint8_t *resic = resic_buf + res_offset[i] ; 
    int *pY = my_py + py_offset[i]; 
    int *pY1 = my_py + py_offset[i] + py_offset[n_samples]; 
    int *pY2 = my_py + py_offset[i] + 2 * py_offset[n_samples];  
    char pYic_j, pYic_k;// = my_pyic + i * maxLength;  
    int *last_pY1 = my_last_py + py_offset[i] + py_offset[n_samples]; 
    int *last_pY2 = my_last_py + py_offset[i] + 2 * py_offset[n_samples];  
    int j, p, pic;
    if (lseq < w) continue;   /* skip if sequence too short */
    if(skipArray[i]) continue;

    /* calculate p(Y_ij | theta_1) */
    int *pY_shifted_1_1 = last_pY1 - 1;
    int *pY_shifted_1_2 = last_pY2 - 1;
    if(myTask[i].j_start == 0)
    {
        uint8_t * _r = res;
        uint8_t * _ric = resic;
        p = 0;
        pic = 0;

        for (k=0; k<w; k++) {         /* position in site */
          p += lmap[ioffRes[k]][(int)(*_r++)];
          pic += lmap[ioffRes[k]][(int)(*_ric++)];
        }
        pY1[0] = p;
        pY2[0] = pic;
        kk = myTask[i].k_start;
        pY1[kk] = pY_shifted_1_1[kk] + theta_last[(*(res + w + kk - 1))] - theta_0[(*(res + kk - 1))];
        pY2[kk] = pY_shifted_1_2[kk] + theta_last[(*(resic + w + kk - 1))] - theta_0[(*(resic + kk - 1))];
        #pragma ivdep
        for (j=myTask[i].j_start + 1; j <= myTask[i].j_end; j+=1) 
        {

            pY1[j] = pY_shifted_1_1[j] + theta_last[(*(res + w + j - 1))] - theta_0[(*(res + j - 1))];
            pY2[j] = pY_shifted_1_2[j] + theta_last[(*(resic + w + j - 1))] - theta_0[(*(resic + j - 1))];
        }

        #pragma ivdep
        for (kk = myTask[i].k_start - 1; kk >= myTask[i].k_end; kk -= 1) 
        {
            pY1[kk] = pY_shifted_1_1[kk] + theta_last[(*(res + w + kk - 1))] - theta_0[(*(res + kk - 1))];
            pY2[kk] = pY_shifted_1_2[kk] + theta_last[(*(resic + w + kk - 1))] - theta_0[(*(resic + kk - 1))];


        }
    }
    else{
      #pragma ivdep
      for (j=myTask[i].j_start; j <= myTask[i].j_end; j+=1) 
      {


          pY1[j] = pY_shifted_1_1[j] + theta_last[(*(res + w + j - 1))] - theta_0[(*(res + j - 1))];
          pY2[j] = pY_shifted_1_2[j] + theta_last[(*(resic + w + j - 1))] - theta_0[(*(resic + j - 1))];
      }
      #pragma ivdep
      for (kk = myTask[i].k_start;kk >= myTask[i].k_end;kk -= 1) 
      {
          pY1[kk] = pY_shifted_1_1[kk] + theta_last[(*(res + w + kk - 1))] - theta_0[(*(res + kk - 1))];
          pY2[kk] = pY_shifted_1_2[kk] + theta_last[(*(resic + w + kk - 1))] - theta_0[(*(resic + kk - 1))];

      }
    }
    for (j=myTask[i].j_start, kk = myTask[i].k_start; j <= myTask[i].j_end && kk >= myTask[i].k_end; j+=1, kk -= 1) 
    {
	  //for revcomp	
      if (pY2[kk] > pY1[j]) {    
        pYic_j = '\1'; pY[j] = pY2[kk];
      } else {       
        pYic_j = '\0'; pY[j] = pY1[j]; 
      }
      if (pY2[j] > pY1[kk]) {    
        pYic_k = '\1'; pY[kk] = pY2[j];
      } else {       
        pYic_k = '\0'; pY[kk] = pY1[kk]; 
      }
	  //selects the local highest - scoring substring S_k, local_maxk
      if(j == myTask[i].j_start)
      {
        max_buf_ptr->max_j = j;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[j] + log_not_o[j];
        max_buf_ptr->pYic = pYic_j;
      }
      else if (pY[j] + log_not_o[j] > max_buf_ptr->max) {		
        max_buf_ptr->max_j = j;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[j] + log_not_o[j]; 		// log (pY * Pr(site) * Pr(no overlap))
        max_buf_ptr->pYic = pYic_j;
      } 
      else if(pY[j] + log_not_o[j] == max_buf_ptr->max && j < max_buf_ptr->max_j) {	
        max_buf_ptr->max_j = j;
        max_buf_ptr->pYic = pYic_j;
      } 
      if (pY[kk] + log_not_o[kk] > max_buf_ptr->max) {		
        max_buf_ptr->max_j = kk;
        // FIXME: We are assumming that priors are always symmetrical here.
        max_buf_ptr->max = pY[kk] + log_not_o[kk]; 		// log (pY * Pr(site) * Pr(no overlap))
        max_buf_ptr->pYic = pYic_k;
      }    
      else if(pY[kk] + log_not_o[kk] == max_buf_ptr->max && kk < max_buf_ptr->max_j) {		
        max_buf_ptr->max_j = kk;
        max_buf_ptr->pYic = pYic_k;
      } 

    }

    /* calculate log p(Y_i0 | theta_1) */

  }
}


/**********************************************************************/
/*
	get_max

	Find the erased maxima of pY.  
	If add_psp_to_log_not_o() has been called, "erasing" includes
	the scaled_to_one PSP.

	Returns the length of the (sorted) list of maxima.

	For the oops and zoops models, one maximum is found per sequence.

	For the tcm model, all non-overlapping local maxima are found.
*/
/**********************************************************************/
int get_max(
  MOTYPE mtype,		/* the model type */
  DATASET *dataset,	/* the dataset */
  BOOLEAN negative,	// control dataset?
  int w,		/* width of sites */ 
  P_PROB maxima, 	/* array of encoded site starts of local maxima */
  BOOLEAN ic, 		/* use reverse complement, too */
  BOOLEAN sort		/* sort the maxima */
)
{
  int n_maxima;

  /* find the maxima */
  if (mtype == Oops || mtype == Zoops) {
    n_maxima = global_max(dataset, negative, w, maxima, ic);
  } else {
    n_maxima = local_max(dataset, negative, w, maxima, ic);
  }

  /* sort the local maxima of pY[1] */
  if (sort) qsort((char *) maxima, n_maxima, sizeof(p_prob), pY_compare);

  return n_maxima;
} /* get_max */

/**********************************************************************/
/*
	global_max	

	Find the position in each sequence with the globally maximal
	value of log pY + log_not_o.

	Returns the number of maxima found and
 	the updated array of maxima positions.
*/
/**********************************************************************/
static int global_max(
  DATASET *dataset,	/* the dataset */
  BOOLEAN negative,	// control dataset?
  int w,		/* length of sites */ 
  P_PROB maxima, 	/* array of encoded site starts of local maxima */
  BOOLEAN ic 		/* use reverse complement, too */
)
{
  int i, j;
  SAMPLE **samples = dataset->samples;		/* datset samples */
  int n_samples = dataset->n_samples;		/* number samples in dataset */
  int n_maxima = 0;				/* number of maxima found */

  /* find the position with maximum pY in each sequence */
  for (i=0; i<n_samples; i++) {			/* sequence */
    SAMPLE *s = samples[i];
    int lseq = s->length;
    int last_j = lseq-w;			/* start of last subseq */
    int *pY = s->pY[0];				/* log p(Y_j | theta_1) */
    char *pYic = s->pYic;			/* site on - strand */
    int *log_not_o = s->log_not_o;		// Pr(site) * Pr(no overlap)
    int max_j = 0;				/* best offset */
    int max = pY[0] + log_not_o[0]; 		/* initial maximum */

    if (lseq < w) continue;			/* skip if too short */
//FIXME
//printf("GLOBAL i %d group %d skip %d\n", i, s->group, dataset->skip[s->group]);
    skip_sample_if_required(dataset, s);

    for (j=0; j<=last_j; j++) {			/* subsequence */
      if (pY[j] + log_not_o[j] > max) {		/* new maximum found */
        max_j = j;
        // FIXME: We are assumming that priors are always symmetrical here.
	max = pY[j] + log_not_o[j]; 		// log (pY * Pr(site) * Pr(no overlap))
      } /* new maximum */

    } /* subsequence */
    /* record the maximum for this sequence */
    maxima[n_maxima].x = i;
    maxima[n_maxima].y = max_j;
    maxima[n_maxima].ic = ic && pYic[max_j];	/* on - strand */
    maxima[n_maxima].negative = negative;
    maxima[n_maxima].rank = -1;
    maxima[n_maxima].prob = max;
    n_maxima++;
  } /* sequence */

  return n_maxima;
} /* global_max */


/**********************************************************************/
/*
	local_max

	Find the local maxima of pY * log_not_o 
	subject to the constraint that they are separated by at 
	least w positions. 

	Returns the number of local maxima found and the
	updated array of maxima positions.
*/
/**********************************************************************/
static int local_max(
  DATASET *dataset,	/* the dataset */
  BOOLEAN negative,	// control dataset?
  int w,		/* length of sites */ 
  P_PROB maxima,  	/* array of encoded site starts of local maxima */
  BOOLEAN ic		/* use reverse complement, too */
)
{
  int i, j, k, next_j, n_maxima;
  SAMPLE **samples = dataset->samples;		/* datset samples */
  int n_samples = dataset->n_samples;		/* number samples in dataset */

  /* Find the non-overlapping local maxima of p(Y_ij | theta_1) */
  n_maxima = 0;
  for (i=0; i<n_samples; i++) {			/* sequence */
    SAMPLE *s = samples[i];
    int lseq = s->length;			/* length of sequence */
    int *pY = s->pY[0];				/* log p(Y_j | theta_1) */
    int *log_not_o = s->log_not_o;		// Pr(site) * Pr(no overlap)
    int last_j = lseq-w;			/* last possible site */
    int max = pY[0]+log_not_o[0]; 		/* initial maximum */

    if (lseq < w) continue;			/* skip if too short */
    skip_sample_if_required(dataset, s);

    maxima[n_maxima].x = i;			/* candidate */
    maxima[n_maxima].y = 0;			/* candidate site */
    maxima[n_maxima].prob = max;
    next_j = MIN(w, last_j+1);			/* next possible maximum */

    for (j=0; j<=last_j; j++) {			/* subsequence */
      // FIXME: We are assumming that priors are always symmetrical here.
      int prob = pY[j]+log_not_o[j]; 		/* log (pY * Pr(site) * Pr(no overlap)) */
      if (j==next_j) n_maxima++;		/* candidate not exceeded */
      if (j==next_j || prob>max) {		/* create/overwrite */
        max = prob;				/* new max */
        maxima[n_maxima].x = i;			/* overwrite the candidate */
        maxima[n_maxima].y = j;			/* site */
        maxima[n_maxima].prob = max;		/* max */
        next_j = MIN(j+w, last_j+1);		/* next possible maximum */
      } /* create/overwrite candidate */
    }
    n_maxima++;					/* record last maxima */
  }

  /* set the strand and position */
  for (k=0; k<n_maxima; k++) {
    int i = maxima[k].x;			/* site position */
    int j = maxima[k].y;			/* site position */
    SAMPLE *s = samples[i];			/* sequence record */
    maxima[k].ic = ic && s->pYic[j];		/* on - strand */
    maxima[k].negative = negative;
    maxima[k].rank = -1;
  } /* n_maxima */

  return n_maxima;
} /* local_max */

/**********************************************************************/
/*
        pY_compare

        Compare the pY of two start sequences.  Return <0 0 >0
        if the second pY is <, =, > than the first pY.
*/
/**********************************************************************/
int pY_compare(
  const void *v1, 
  const void *v2 
)
{
  double result;

  const struct p_prob * s1 = (const struct p_prob *) v1; 
  const struct p_prob * s2 = (const struct p_prob *) v2; 

  if ((result = s2->prob - s1->prob) != 0) {
    return (result<0) ? -1 : +1;
  } else if ((result = s2->x - s1->x) != 0) {
    return result;
  } else {
    return s2->y - s1->y;
  }
}

/**********************************************************************/
/*
	init_theta_1

	Convert a subsequence to a motif.

	Uses globals:
*/
/**********************************************************************/
static void init_theta_1(
  int w,			/* width of site */
  uint8_t *res,			/* (encoded) letters of subsequence */
  LOG_THETAG_TYPE(theta_1),	/* theta_1 */
  int lmap[MAXALPH][MAXALPH]  	/* matrix of frequency vectors */ 
)
{
  int m;
  for (m=0; m<w; m++) {
    theta_1[m] = lmap[res[m]];
  }
} /* init_theta_1 */

/**********************************************************************/
/*
	align_top_subsequences

     	Align the top nsites0 subsequences for each value
	of nsites0 and save the alignments with the highest 
        product of p-values of log likelihood ratio (classic)
        or highest LLR (non-classic) of the columns.
	Saves LLR or (-log_pop if classic and not use_llr) 
        as the score for the start.

	Returns number of values of nsites0 tried.
*/ 
/**********************************************************************/
int align_top_subsequences(
  MOTYPE mtype,				/* type of model */
  int w,				/* width of motif */
  DATASET *dataset,			/* the dataset */
  int iseq,				/* sequence number of starting point */
  int ioff,				/* sequence offset of starting point */
  uint8_t *eseq,			/* integer encoded subsequence */
  char *name,				/* name of sequence */
  int n_nsites0,			/* number of nsites0 values to try */
  int n_maxima,				/* number of local maxima */
  P_PROB maxima,			/* sorted local maxima indices */
  double *col_scores,			/* column scores for last start point */
  S_POINT s_points[]			/* array of starting points */
)
{
  int i, j, k, i_nsites0;
  int next_seq;				/* index of next subsequence to align */
  int n_starts = 0;			/* number of nsites0 tried */
  int nsites0;				/* starting nsites rounded down */
  int alength = alph_size_core(dataset->alph);       /* length of alphabet */
  ARRAY_T *back = dataset->back;	/* background frequencies */
  SAMPLE **samples = dataset->samples;	/* the sequences */
  double counts[MAXSITE][MAXALPH];	/* array to hold observed counts */
  double wN;				/* weighted number of sites */
  double score;				/* score for start */
  BOOLEAN classic = (dataset->objfun==Classic);	/* use classic MEME algorithm */

  /* initialize letter counts to 0 */
  wN = 0;				/* weighted number of sites */
  for (i=0; i<w; i++) 
    for (j=0; j < alength; j++) { counts[i][j] = 0; }

  /* calculate the product of p-values of information content
     of the top nsite0 probability positions 
  */
  for (i_nsites0=0, next_seq=0; i_nsites0 < n_nsites0; i_nsites0++) {

    /* don't score this start if not enough maxima found */
    nsites0 = (int) s_points[i_nsites0].nsites0;	/* round down */
    if (n_maxima < nsites0) {
      continue;
    }
    n_starts++;					/* number of nsites0 tried */

    /* Align the next highest probability sites 
	1) count the number of occurrences of each letter in each column 
	   of the motif and, 
        2) compute the log likelihood of the sites under the background model
    */
    for (k=next_seq; k<nsites0; k++) {		/* site */
      int jj;
      BOOLEAN ic = maxima[k].ic;		/* on - strand */
      int y = maxima[k].y;			/* position of site */
      SAMPLE *s = samples[maxima[k].x];		/* sequence */
      int off = ic ? s->length-w-y : y;		/* - strand offset from rgt. */
      uint8_t *res = ic ? s->resic+off : s->res+off;	/* integer sequence */
      double sw = s->sw;			/* sequence weight */
      //
      // TLB: Note that log_not_o contains Pr(site) scaled to have max=1
      // when called from subseq7() but not when called from discretize().
      //
      // Why not revert to not_o[y] here?  TLB: Because the other one works
      // much better, although its kind of a hack.
      //double esw = sw * s->not_o[y];		// Pr(site not overlapped)
      //
      // FIXME: We are assumming that priors are always symmetrical here.
      double esw = sw * INT_DELOG(s->log_not_o[y]);	// Pr(site not overlapped) * Pr(site) 
      wN += esw;				/* total sequence wgt */

      /* residue counts */
      for (j=0; j<w; j++) {			/* position in sequence */
        int c = res[j];
        if (c < alength) {/* normal letter */
          counts[j][c] += esw;
	} else {				/* wildcard : esw * back[letter] */
          for (jj=0; jj < alength; jj++) 
            counts[j][jj] += esw * get_array_item(jj, back);
	}
        
      } /* position */

    } /* site */
    next_seq = k;				/* next site to align */
    
    /* 
      For DNA palindromes, combine the counts in symmetrically opposing columns
    */
    if (dataset->pal) palindrome(counts, counts, w, dataset->alph);

    // Updated on 13-12-06: Only calculate objective function score if the
    // current s_point is supposed to be evaluated:
    if (s_points[i_nsites0].evaluate) {
      /* 
	convert COUNTS to FREQUENCIES and calculate log likelihood ratio
      */
      score = 0;				/* score for start */
      for (i=0; i<w; i++) {			/* position in site */
	double llr = 0;				/* log-like-ratio of column */
	double log_pv;				/* log of column p-value */
	double ic;

	/* compute log likelihood for position i */
	for (j=0; j < alength; j++) {		/* letter */
	  double f = wN ? counts[i][j] / wN : 1;/* observed letter frequency */
	  double p = get_array_item(j, back);	/* backgrnd letter frequency */
	  double log_f = LOGL(f);
	  double log_p = LOGL(p);
	  double llr_ij = (f&&p) ? f*(log_f - log_p) : 0;
	  llr += llr_ij;
	} /* letter */
	RND(llr/0.6934, RNDDIG, ic);		/* info content in bits */
	llr *= wN;				/* convert entropy to ll */ 
	RND(llr, RNDDIG, llr);			/* round to RNDDIG places */
        if (classic) {
          log_pv = get_llr_pv(llr, wN, 1, LLR_RANGE, 1.0, alength, back); 
        } else {
          log_pv = 0;
        }

	if (!classic || dataset->use_llr) {
	  // Using llr instead of pop:
	  col_scores[i] = llr;			// LLR of column
	  score -= llr;
	} else {
	  score += col_scores[i] = log_pv;	// log_pv of column
	}
      } /* position in site */
      RND(score, RNDDIG, score);

      /* print the start sequence and other stuff */
      if (TRACE) {
	if (eseq) {
	  char seq[MAXSITE+1];
          for (i = 0; i < w; i++) {
            seq[i] = alph_char(dataset->alph, eseq[i]);
          }
          seq[i] = '\0';
	  fprintf(stdout, 
	    "( %3d %3d ) ( %*.*s ) %.*s score %.17f nsites0 %6d\n",
	    iseq+1, ioff+1, MSN, MSN, name, w, seq, -score, nsites0);
	} else {
	  fprintf(stdout, 
	    "l_off %3d w %d score %.17f nsites0 %6d\n",
	    iseq, w, -score, nsites0);
	}
      }

      /* save the best start */
      if (-score > s_points[i_nsites0].score) {
	/* Save the starting point and offset so we can re-calculate
	   eseq later. */
	s_points[i_nsites0].iseq = iseq;
	s_points[i_nsites0].ioff = ioff;
	s_points[i_nsites0].e_cons0 = eseq;
	s_points[i_nsites0].wgt_nsites = wN;
	s_points[i_nsites0].score = -score;
      }
    } // Evaluating only if told to do so.
  } /* nsites0 */

  return n_starts;
} /* align_top_subsequences */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 2
 * End:
 */
