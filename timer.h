#include <time.h>

#ifndef TIMER
#define TIMER

#define timedifference_msec(start,stop) (((double)((stop) - (start))) / CLOCKS_PER_SEC * 1000.0)

#endif