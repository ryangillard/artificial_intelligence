#ifndef l_bfgs_b_h
#define l_bfgs_b_h

#include <stdio.h>
#include <math.h>
#include <time.h>

#define START 1 
#define NEW_X 2
#define ABNORMAL 3 /* message: ABNORMAL_TERMINATION_IN_LNSRCH. */
#define RESTART 4 /* message: RESTART_FROM_LNSRCH. */

#define FG      10
#define FG_END  15
#define IS_FG(x) ( ((x)>=FG) ?  ( ((x)<=FG_END) ? 1 : 0 ) : 0 )
#define FG_LN   11
#define FG_LNSRCH FG_LN
#define FG_ST   12
#define FG_START FG_ST

#define CONVERGENCE 20
#define CONVERGENCE_END  25
#define IS_CONVERGED(x) ( ((x)>=CONVERGENCE) ?  ( ((x)<=CONVERGENCE_END) ? 1 : 0 ) : 0 )
#define CONV_GRAD   21 /* message: CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL. */
#define CONV_F      22 /* message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH. */

#define STOP  30  
#define STOP_END 40
#define IS_STOP(x) ( ((x)>=STOP) ?  ( ((x)<=STOP_END) ? 1 : 0 ) : 0 )
#define STOP_CPU  31 /* message: STOP: CPU EXCEEDING THE TIME LIMIT. */
#define STOP_ITER 32 /* message: STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIM.  */
#define STOP_GRAD 33 /* message: STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL. */

#define WARNING 100
#define WARNING_END 110
#define IS_WARNING(x) ( ((x)>=WARNING) ?  ( ((x)<=WARNING_END) ? 1 : 0 ) : 0 )
#define WARNING_ROUND 101  /* WARNING: ROUNDING ERRORS PREVENT PROGRESS */
#define WARNING_XTOL  102  /* WARNING: XTOL TEST SATISIED */
#define WARNING_STPMAX 103 /* WARNING: STP = STPMAX */
#define WARNING_STPMIN 104 /* WARNING: STP = STPMIN */

#define ERROR 200
#define ERROR_END 240
#define IS_ERROR(x) ( ((x)>=ERROR) ?  ( ((x)<=ERROR_END) ? 1 : 0 ) : 0 )
/* More specific conditions below */
#define ERROR_SMALLSTP 201 /* message: ERROR: STP .LT. STPMIN  */
#define ERROR_LARGESTP 202 /* message: ERROR: STP .GT. STPMAX  */
#define ERROR_INITIAL  203 /* message: ERROR: INITIAL G .GE. ZERO */
#define ERROR_FTOL     204 /* message: ERROR: FTOL .LT. ZERO   */
#define ERROR_GTOL     205 /* message: ERROR: GTOL .LT. ZERO   */
#define ERROR_XTOL     206 /* message: ERROR: XTOL .LT. ZERO   */
#define ERROR_STP0     207 /* message: ERROR: STPMIN .LT. ZERO */
#define ERROR_STP1     208 /* message: ERROR: STPMAX .LT. STPMIN */
#define ERROR_N0       209 /* ERROR: N .LE. 0 */
#define ERROR_M0       210 /* ERROR: M .LE. 0 */
#define ERROR_FACTR    211 /* ERROR: FACTR .LT. 0 */
#define ERROR_NBD      212 /* ERROR: INVALID NBD */
#define ERROR_FEAS     213 /* ERROR: NO FEASIBLE SOLUTION */


/* and "word" was a char that was one of these: */
#define WORD_DEFAULT 0 /* aka "---".  */
#define WORD_CON 1 /*  the subspace minimization converged. */
#define WORD_BND 2 /* the subspace minimization stopped at a bound. */
#define WORD_TNT 3 /*  the truncated Newton step has been used. */


/* Some linesearch parameters. They are used in dcsrch()  */
#ifndef FTOL
#define FTOL .001
#endif
#ifndef GTOL
#define GTOL .9
#endif
#ifndef XTOL
#define XTOL .1
#endif
#ifndef STEPMIN
#define STEPMIN 0.
#endif


/* If we want machine precision in a nice fashion, do this: */
#include <float.h>
#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2e-16
#endif

/* Function prototypes */
void setulb(long int n, long int m, double* x, double* l, double* u, long int* nbd, double* f, double* g, double factr, double pgtol, double* wa, long int* iwa, long int* task, long int iprint, long int* csave, long int* lsave, long int* isave, double* dsave);

void mainlb(long int n, long int m, double* x, double* l, double* u, long int* nbd, double* f, double* g, double factr, double pgtol, double* ws, double* wy, double* sy, double* ss, double* wt, double* wn, double* snd, double* z__, double* r__, double* d__, double* t, double* xp, double* wa, long int* index, long int* iwhere, long int* indx2, long int* task, long int iprint, long int* csave, long int* lsave, long int* isave, double* dsave);

void active(long int n, double* l, double* u, long int* nbd, double* x, long int* iwhere, long int iprint, long int* prjctd, long int* cnstnd, long int* boxed);

void bmv(long int m, double* sy, double* wt, long int col, double* v, double* p, long int* info);

void cauchy(long int n, double* x, double* l, double* u, long int* nbd, double* g, long int* iorder, long int* iwhere, double* t, double* d__, double* xcp, long int m, double* wy, double* ws, double* sy, double* wt, double theta, long int col, long int head, double* p, double* c__, double* wbp, double* v, long int* nseg, long int iprint, double sbgnrm, long int* info, double epsmch);

void cmprlb(long int n, long int m, double* x, double* g, double* ws, double* wy, double* sy, double* wt, double* z__, double* r__, double* wa, long int* index, double theta, long int col, long int* head, long int nfree, long int cnstnd, long int* info);

void errclb(long int n, long int m, double factr, double *l, double *u, long int* nbd, long int* task, long int* info, long int* k);

void formk(long int n, long int nsub, long int* ind, long int nenter, long int ileave, long int* indx2, long int iupdat, long int updatd, double* wn, double* wn1, long int m, double* ws, double* wy, double* sy, double theta, long int col, long int head, long int* info);

void formt(long int m, double* wt, double* sy, double* ss, long int col, double theta, long int* info);

void freev(long int n, long int* nfree, long int* index, long int* nenter, long int* ileave, long int* indx2, long int* iwhere, long int* wrk, long int updatd, long int cnstnd, long int iprint, long int iter);

void hpsolb(long int n, double* t, long int* iorder, long int iheap);

void lnsrlb(long int n, double* l, double* u, long int* nbd, double* x, double* f, double* fold, double* gd, double* gdold, double* g, double* d__, double* r__, double* t, double* z__, double* stp, double* dnorm, double* dtd, double* xstep, double* stpmx, long int iter, long int* ifun, long int* iback, long int* nfgv, long int* info, long int* task, long int boxed, long int cnstnd, long int* csave, long int* isave, double* dsave);

void matupd(long int n, long int m, double* ws, double* wy, double* sy, double* ss, double* d__, double* r__, long int* itail, long int iupdat, long int* col, long int* head, double* theta, double rr, double dr, double stp, double dtd);

void prn1lb(long int n, long int m, double* l, double* u, double* x, long int iprint, double epsmch);

void prn2lb(long int n, double* x, double f, double* g, long int iprint, long int iter, double sbgnrm, long int* word, long int iword, long int iback, double xstep);

void prn3lb(long int n, double* x, double f, long int task, long int iprint, long int info, long int iter, long int nfgv, long int nintol, long int nskip, long int nact, double sbgnrm, double time, double stp, double xstep, long int k, double cachyt, double sbtime, double lnscht);

void projgr(long int n, double* l, double* u, long int* nbd, double* x, double* g, double* sbgnrm);

void subsm(long int n, long int m, long int nsub, long int* ind, double* l, double* u, long int* nbd, double* x, double* d__, double* xp, double* ws, double* wy, double theta, double* xx, double* gg, long int col, long int head, long int* iword, double* wv, double* wn, long int iprint, long int* info);

void dcsrch(double* f, double* g, double* stp, double ftol, double gtol, double xtol, double stpmin, double stpmax, long int* task, long int* isave, double* dsave);

void dcstep(double* stx, double* fx, double* dx, double* sty, double* fy, double* dy, double* stp, double* fp, double* dp, long int* brackt, double* stpmin, double* stpmax);

void dpofa(double* a, long int lda, long int n, long int* info);

void dtrsl(double* t, long int ldt, long int n, double* b, long int job, long int* info);

void daxpy(long int n, double da, double* dx, long int incx, double* dy, long int incy);

void dcopy(long int n, double* dx, long int incx, double* dy, long int incy);

double ddot(long int n, double* dx, long int incx, double* dy, long int incy);

void dscal(long int n, double da, double* dx, long int incx);

void timer(double *ttime);

#endif /* l_bfgs_b_h */