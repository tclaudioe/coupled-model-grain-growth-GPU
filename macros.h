#ifndef SRC_MACROS_H
#define SRC_MACROS_H

// @@@@@@@@@@@@@ DURATION AND SNAPS @@@@@@@@@@@@@

// Time limit of the simulation (use negative to infinity).
#define MAX_TIME 100

// Limit of steps of the simulation (use negative to infinity).
#define MAX_STEPS -1

// Time after a new snapshot of the grains is made (use negative to infinity).
#define SAVE_N_ITER 70
#define SAVE_T_MIN -1
#define SAVE_T_MAX -1

// Minimal value for the time delta after inhibition happens.
// Big values may stuck the simulation,
// (considerate the number of grains)
// very small values (<0.000001) may also do it if using floats.
#define MIN_DELTA_TOL 0.0000005

// Fraction of grain decay (from 0.0 to 1.0) after a new snapshot of the grains is made
//  (use negative for infinity). This is calculated counting the number of disabled boundaries.
#define GRAIN_DECAY_DELTA -1

// Beware that a snap of 1000000 grains can use up to 5GB of memory.

// @@@@@@@@@@@@@ OUTPUT PARAMETERS @@@@@@@@@@@@@

// Size of output images (use negative or 0 to not save images).
#define IMAGE_SIZE -1

// Sends the steps and iterations duration to "debug.txt"
// (enable with 1, disable with 0. With 2 the file is opened and closed
// for each line written, slows a lot, but is useful when running on the HPC).
#define DEBUG_FILE 0

// @@@@@@@@@@@@@ PRECISION PARAMETERS @@@@@@@@@@@@@

// Number of inner points on each frontier (can be possitive or 0).
#define INNER_POINTS 2

// Number of iterations on the bisection method used when reparametrizing frontiers.
#define BISECTION_ITERS 12

// Order of the guassian quadrature when integrating (from 1 to 64 is supported).
#define QUAD_ORDER 20

// Use doubles instead of floats (enable with 1, disable with 0), is highly recommended.
#define DOUBLE_PRECISION 1

// Uses the linear interpolation of inverse arclen function to reparametrizate
// instead of the integral over the line (enable with 1, disable with 0).
// When its enabled, BISECTION_ITERS is not used.
#define INVERSE_ARCLEN_FUNCTION_REPARAMETRIZATION 1
#define INVERSE_REPARAMETRIZATION_LENGTH 20

// When a frontier of length 0 is detected, it's expanded a little, determinated by this constant.
#define SMALL_EXPAND 1e-8

// Tolerance of negative results on the calculation of grain areas
// (they may happen, not because of a folded structure due to a big time delta,
// but because precision limitations).
// If calculated value is negative, but larger than the tolerance,
// it will be changed to 0.0 to prevent warnings.
#define NEGATIVE_AREA_TOLERANCE -1e-11 //-0.00000000001

// @@@@@@@@@@@@@ OTHERS EXPERIMENTAL PARAMETERS @@@@@@@@@@@@@

#endif
