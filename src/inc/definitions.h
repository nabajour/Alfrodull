
// switch between double and single precision (currently, single prec. provides no speed-up and thus appears to be useless)
// TODO: should get rid of this
/***
#define USE_SINGLE
***/
#ifdef USE_SINGLE
typedef float utype;
#else
typedef double utype;
#endif

