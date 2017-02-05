#include "mic-meme.h"
#include "meme.h"
/*
#pragma offload_attribute(push,target(mic))
  int micLmap[MAXALPH][MAXALPH]; 

  int* py_offset;
  uint8_t *res_buf;
  uint8_t *resic_buf;
  int *log_not_o_buf;
  bool *skipArray;

  int *res_offset;

#pragma offload_attribute(pop)
*/

/*craete task for each thread*/
TaskInfo * createTask(  
  DATASET *dataset, 
  int totalLengthW, 
  int numThreads, 
  int w, 
  int ic, 
  int *taskCount)
{
  int n_samples = dataset->n_samples; /* number of samples in dataset */
  SAMPLE **samples = dataset->samples;  /* samples in dataset */
  TaskInfo* threadTasks = (TaskInfo *)malloc(numThreads * n_samples * sizeof(TaskInfo));
  int* length_buf = (int *)malloc(n_samples * sizeof(int));
  memset(length_buf, 0, n_samples * sizeof(int));
  memset(threadTasks, 0, numThreads * n_samples * sizeof(TaskInfo));
  int avglength = totalLengthW / numThreads;//average workload for each thread 
  int remainder = totalLengthW%numThreads;
  int i, iseq = 0;
  int  subtractor = 0;
  int taskIndex = 0;
  for ( i = 0; i < numThreads ; i ++)
  {
    int mylength = avglength;
    if( i < remainder) mylength ++;
    mylength -= subtractor;
    if(subtractor > 0) subtractor = 0;
    TaskInfo* myTask = threadTasks + i * n_samples;
    while(mylength > 0)
    {
      if(ic)
      {
        if(mylength >=  (samples[iseq]->length - w + 1 - length_buf[iseq]))
        {
          mylength -= (samples[iseq]->length - w + 1 - length_buf[iseq]);
          myTask[iseq].flag = 1;
          myTask[iseq].index = taskIndex;
          taskIndex ++;
          myTask[iseq].j_start = length_buf[iseq]/2;
          myTask[iseq].k_start = samples[iseq]->length - w - length_buf[iseq]/2;
          int halflength = (samples[iseq]->length - w - length_buf[iseq])/2;
          if( (samples[iseq]->length - w - length_buf[iseq])%2 != 0) halflength ++;
          myTask[iseq].j_end =  myTask[iseq].j_start +halflength ;
          myTask[iseq].k_end = myTask[iseq].k_start - halflength;
          length_buf[iseq] = samples[iseq]->length - w + 1;

          iseq ++;
        }
        else 
        {
          if(mylength % 2 != 0)
          {
            mylength ++;
            subtractor ++;
            continue;
          }
          else{
            myTask[iseq].flag = 1;
            myTask[iseq].index = taskIndex;
            taskIndex ++;
            myTask[iseq].j_start = length_buf[iseq]/2;
            myTask[iseq].j_end = length_buf[iseq]/2 + mylength/2 - 1;
            myTask[iseq].k_start =  samples[iseq]->length - w - length_buf[iseq]/2;
            myTask[iseq].k_end = samples[iseq]->length - w - length_buf[iseq]/2 - mylength/2 + 1;
            length_buf[iseq] += mylength;
            mylength = 0;
          }

        }
      }
      else
      {
        if(mylength >=  (samples[iseq]->length - w + 1 - length_buf[iseq]))
        {
          mylength -= (samples[iseq]->length - w + 1 - length_buf[iseq]);
          myTask[iseq].flag = 1;
          myTask[iseq].index = taskIndex;
          taskIndex ++;
          myTask[iseq].j_start = length_buf[iseq];
          myTask[iseq].j_end = samples[iseq]->length - w;
          length_buf[iseq] = samples[iseq]->length - w + 1;
          iseq ++;
        }
        else
        {
          myTask[iseq].flag = 1;
          myTask[iseq].index = taskIndex;
          taskIndex ++;
          myTask[iseq].j_start = length_buf[iseq];
          myTask[iseq].j_end = length_buf[iseq] + mylength - 1;
          length_buf[iseq] += mylength;
          mylength = 0;
        }
      }
    }
  }
  free(length_buf);
  *taskCount = taskIndex; 
  return threadTasks;
}