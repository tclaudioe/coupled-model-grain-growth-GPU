#ifndef SRC_CALCULUS_H
#define SRC_CALCULUS_H

#include <math.h>

#include "utils.h"
#include "geometry.h"

#include "precomp_gauss.h"

const coord pi = 3.14159265358979323846;

double host_chebyshev[INNER_POINTS+2];
__constant__ double chebyshev[INNER_POINTS+2];
// ^ Chebyshev points from 0 to 1.

double host_diffMatrix[(INNER_POINTS+2)*(INNER_POINTS+2)];
__constant__ double diffMatrix[(INNER_POINTS+2)*(INNER_POINTS+2)];
// ^ Differentiation matrix for chebyhev points.

double host_interp_legendre_w_cheb_coefs[QUAD_ORDER*(INNER_POINTS+2)];
__constant__ double interp_legendre_w_cheb_coefs[QUAD_ORDER*(INNER_POINTS+2)];
// ^ Values of phi for each legendre point from 0 to 1.

double host_wj_chebyshev[INNER_POINTS+2];
__constant__ double wj_chebyshev[INNER_POINTS+2];

double host_chebyshev_phi_denominators[(INNER_POINTS+2)];
__constant__ double chebyshev_phi_denominators[(INNER_POINTS+2)];
// ^ Denominators of the phi values for the chebyshev points.

__constant__ double legendre_points[QUAD_ORDER];
__constant__ double gauss_weights[QUAD_ORDER];

#define MTRXIDX(I,J) ((I)*(INNER_POINTS+2)+(J))
#define MTRXIDX2(I,J) ((I)*(INNER_POINTS)+(J))

inline void precompute() {
    for (int i = 0; i < INNER_POINTS+2; i++)
        host_chebyshev[i] = 0.5 - 0.5*cos(pi*i/(double)(INNER_POINTS+1));

    // Precompute Differentiation matrix.
    double tmp_host_diffMatrix[(INNER_POINTS+2)*(INNER_POINTS+2)];
    for (int y = 0; y < INNER_POINTS+2; y++) {
        for (int x = 0; x < INNER_POINTS+2; x++) {
            double c = (1+(x==0)+(x==INNER_POINTS+1))*(1-2*(x%2));
            double ct = (1+(y==0)+(y==INNER_POINTS+1))*(1-2*(y%2));
            tmp_host_diffMatrix[MTRXIDX(y,x)] = c/ct/(host_chebyshev[x]-host_chebyshev[y]+(x==y));
        }
    }
    for (int i = 0; i < INNER_POINTS+2; i++) {
        double sum=0;
        for (int y = 0; y < INNER_POINTS+2; y++)
            sum+= tmp_host_diffMatrix[MTRXIDX(y,i)];
        tmp_host_diffMatrix[MTRXIDX(i,i)] -= sum;
    }

    for (int i = 0; i < INNER_POINTS+2; i++) {
        for (int j = 0; j < INNER_POINTS+2; j++)
            host_diffMatrix[MTRXIDX(i,j)] = tmp_host_diffMatrix[MTRXIDX(j,i)];
    }

    /* BEGIN: Precompute all the INNER_POINTS+2 phi values for each
    legendre node using Barycentric Interpolation.
    This is computed using the Barycentric Interpolation (See Trefethen).
    Following Trefethen's notation, they consider
    wj=(-1)^j \delta_j, where \delta_0=1/2, \delta_n=1/2,
    \delta_j=1 otherwise, for j=0 ... n.
    So, in our case n=INNER_POINTS+2-1=INNER_POINTS+1.
    */
    host_wj_chebyshev[0] = 0.5;
    host_wj_chebyshev[INNER_POINTS+1] = 0.5*pow(-1,INNER_POINTS+1);
    for (int i = 0; i < INNER_POINTS; i++)
        host_wj_chebyshev[i+1]=pow(-1,i+1);
    
    double den_barycentric; // denominator of barycentric interpolation
    double xk_tilde; // The k-th quadrature point
    double phi_tmp;
    for (int k = 0; k < QUAD_ORDER; k++) {
        xk_tilde = 0.5 + host_legendre_points[k]/2;
        den_barycentric=0.0;
        for (int i = 0; i < INNER_POINTS+2; i++) {
            phi_tmp=host_wj_chebyshev[i]/(xk_tilde - host_chebyshev[i]);
            den_barycentric += phi_tmp;
            // This line computes the numerator wj/(xk-x_j)
            host_interp_legendre_w_cheb_coefs[MTRXIDX(k,i)] = phi_tmp;
        }

        for (int i = 0; i < INNER_POINTS+2; i++) {
            // Here we divide wj/(xk-x_j) by sum_{i}^n wi/(xk-x_i)
            host_interp_legendre_w_cheb_coefs[MTRXIDX(k,i)] /= den_barycentric;
        }
    }

    // Pass the calculated values to constant memory:
    HERR(cudaMemcpyToSymbol(chebyshev, host_chebyshev, sizeof(double)*(INNER_POINTS+2)));
    HERR(cudaMemcpyToSymbol(diffMatrix, host_diffMatrix, sizeof(double)*(INNER_POINTS+2)*(INNER_POINTS+2)));
    HERR(cudaMemcpyToSymbol(interp_legendre_w_cheb_coefs, host_interp_legendre_w_cheb_coefs, sizeof(double)*QUAD_ORDER*(INNER_POINTS+2)));
    HERR(cudaMemcpyToSymbol(chebyshev_phi_denominators, host_chebyshev_phi_denominators, sizeof(double)*(INNER_POINTS+2)));
    HERR(cudaMemcpyToSymbol(wj_chebyshev, host_wj_chebyshev, sizeof(double)*(INNER_POINTS+2)));
    
    // Pass the precomputed constants to constant memory:
    HERR(cudaMemcpyToSymbol(legendre_points, host_legendre_points, sizeof(double)*(QUAD_ORDER)));
    HERR(cudaMemcpyToSymbol(gauss_weights, host_gauss_weights, sizeof(double)*(QUAD_ORDER)));
}

/**
 * Spectral derivation of a vector at Chebyshev points.
 *
 * @param a   Input vector
 * @param res Output vector
 */
__host__ __device__ inline void derivate_interpolators(coord *a, coord *res) {
    // Does a spectral derivation doing a dot product with the differentiation matrix.
    for (int i = 0; i < INNER_POINTS+2; i++) {
        double sum = 0;
        for (int j = 0; j < INNER_POINTS+2; j++) {
            #ifdef __CUDA_ARCH__
            sum += diffMatrix[MTRXIDX(i,j)]*a[j];
            #else
            sum += host_diffMatrix[MTRXIDX(i,j)]*a[j];
            #endif
        }
        res[i] = (coord)sum;
    }
}

/**
 * Given an vector of elements at Chebyshev nodes, this function interpolates the data
 * and evaluates the interpolator at Legendre nodes to give data for Gaussian quadrature-
 *
 * @param points vector of elements of size INNER_POINTS+2
 * @param res    output vector of size QUAD_ORDER
 */
__host__ __device__ inline void interpolate_for_legendre(const coord *points, coord *res) {
    for (int j = 0; j < QUAD_ORDER; j++) {
        double current = 0.0;
        for (int i = 0; i < INNER_POINTS+2; i++) {
            #ifdef __CUDA_ARCH__
            current += points[i]*interp_legendre_w_cheb_coefs[MTRXIDX(j,i)];
            #else
            current += points[i]*host_interp_legendre_w_cheb_coefs[MTRXIDX(j,i)];
            #endif
        }
        res[j] = current;
    }
}

/**
 * Integrate function with data given at Legrende nodes
 * @param  points [description]
 * @return        [description]
 */
__host__ __device__ inline coord integrate_legendre(const coord *points) {
    // Integrates a function using the values on the legendre points.
    double total = 0.0;
    for (int u = 0; u < QUAD_ORDER; u++) {
        #ifdef __CUDA_ARCH__
        total += gauss_weights[u]*points[u];
        #else
        total += host_gauss_weights[u]*points[u];
        #endif
    }
    // divide by two because it uses half of the [-1,1] domain.
    return (coord)(total/2.0);
}

/**
 * Compute derivation of unit vector at Gaussian quadrature nodes given the Chebyshev data.
 * The exponent "exp" of the denominator (lx^2 + ly^2)^exp is a parameter.
 * If exp is 2.0, then an aditional norm of l is dividing, useful for computing curvature.
 * For normal unit derivation, exp must be set to 1.5
 */
__device__ inline void derivate_unit_vector_gauss(coord *xpos, coord *ypos, coord *resx, coord *resy) {
    coord lx[INNER_POINTS+2];
    coord ly[INNER_POINTS+2];
    derivate_interpolators(xpos, lx);
    derivate_interpolators(ypos, ly);
    
    coord dlxds[INNER_POINTS+2];
    coord dlyds[INNER_POINTS+2];
    derivate_interpolators(lx, dlxds);
    derivate_interpolators(ly, dlyds);
    
    coord lxgauss[QUAD_ORDER];
    coord lygauss[QUAD_ORDER];
    coord dlxdsgauss[QUAD_ORDER];
    coord dlydsgauss[QUAD_ORDER];
    interpolate_for_legendre(lx, lxgauss);
    interpolate_for_legendre(ly, lygauss);
    interpolate_for_legendre(dlxds, dlxdsgauss);
    interpolate_for_legendre(dlyds, dlydsgauss);
    
    vector2 tmp;
    coord tmp_norm;
    for (int i = 0; i < QUAD_ORDER; i++) {
        // This is the case when the boundary is a straight line.
        if (lxgauss[i] == 0.0 && lygauss[i] == 0.0) {
            resx[i] = 0.0;
            resy[i] = 0.0;
        }
        // This case is when we actually have a curved boundary.
        else {
            coord interm = (lygauss[i] * dlxdsgauss[i] - lxgauss[i] * dlydsgauss[i]);
            tmp.x = lxgauss[i];
            tmp.y = lygauss[i];
            tmp_norm = vector2_mag(tmp);
            interm /= tmp_norm*tmp_norm*tmp_norm;
            resx[i] = lygauss[i] * interm;
            resy[i] = -lxgauss[i] * interm;
        }
    }
}

/**
 * Compute unit derivative of a vector at Gauss nodes given Chebyshev data.
 */
__device__ inline void derivate_unit_vector(coord *xpos, coord *ypos, coord *resx, coord *resy) {
    derivate_unit_vector_gauss(xpos, ypos, resx, resy);
}


/**
 * Compute the curvature of a given curve. The data is at Chebyshev points
 * and all the calculus are performed on Gauss quadrature nodes.
 */
__device__ inline coord compute_curvature(coord *xpos, coord *ypos) {
    coord Nx[QUAD_ORDER];
    coord Ny[QUAD_ORDER];
    coord curvatures[QUAD_ORDER];
    derivate_unit_vector_gauss(xpos, ypos, Nx, Ny);
    
    coord lx[INNER_POINTS+2];
    coord ly[INNER_POINTS+2];
    derivate_interpolators(xpos, lx);
    derivate_interpolators(ypos, ly);
    
    coord lxgauss[QUAD_ORDER];
    coord lygauss[QUAD_ORDER];
    interpolate_for_legendre(lx, lxgauss);
    interpolate_for_legendre(ly, lygauss);
    
    vector2 tmp;
    vector2 tmp_l;
    for (int i = 0; i < QUAD_ORDER; i++) {
        tmp.x = Nx[i];
        tmp.y = Ny[i];
        
        // This term adds the corrextion of the exponent in the denominator, it computes dTds/||l||
        tmp_l.x=lxgauss[i];
        tmp_l.y=lygauss[i];
        curvatures[i] = vector2_mag(tmp)/vector2_mag(tmp_l);
    }
    return integrate_legendre(curvatures);
}

/**
 * Integrate a function evaluated at Legendre nodes and computes the integral of
 * this function times the Lagrange $\phi_i$ function, i.e.:
 *
 * $res = \int_0^1 f(s) \phi_i(s)\, ds$
 *
 * @param  points Vector of elements at Legendre nodes
 * @param  i      index for $phi_i$ Lagrange interpolating function
 * @return        [description]
 */
__device__ inline coord integrate_legendre_mult_by_cheb(const coord *points, int i) {
    // Integrates a function (multiplied by phi_i) using the values on the legendre points.
    coord pointsp[QUAD_ORDER];
    for (int u = 0; u < QUAD_ORDER; u++)
        pointsp[u] = points[u]*interp_legendre_w_cheb_coefs[MTRXIDX(u,i)];
    return integrate_legendre(pointsp);
}

__device__ inline coord evaluate_w_cheb_interpolators(const coord *points, coord s) {
    // Evaluate a Lagrange polynomial given its evaluations on chebyshev points using barycentric interpolation.
    // This verify if the evaluation point "s" is a collocation point
    for (int i = 0; i < INNER_POINTS+2; i++) {
        if (abs(s-chebyshev[i]) < 1e-12)
            return points[i];
    }

    // Barycentric interpolation, see Trefethen 2004.
    double result = 0.0;
    double tmp = 0.0;
    double den = 0.0;
    for (int i = 0; i < INNER_POINTS+2; i++) {
        tmp     = wj_chebyshev[i]/(s-chebyshev[i]);
        result += tmp*points[i];
        den    += tmp;
    }
    return ((coord)result)/den;
}

__device__ inline coord integrate_upto(const coord *points, coord s) {
    // Integrates from 0 to s, usign the legendre points on that interval which are interpolated
    // from the input evaluations on the chebyshev points from the interval 0 to 1.
    coord sum = 0.0;
    for (int u = 0; u < QUAD_ORDER; u++) {
        coord st = (coord)((legendre_points[u] + 1.0)*s/2.0);
        sum += evaluate_w_cheb_interpolators(points,st)*gauss_weights[u];
    }
    return sum*s/2.0;
}

/**
 * Evaluates the following integral:
 *
 *  L(s) = \int_{0}^{s} ||l(u)|| du
 *
 * @param  lx x-component of the derivative of the boundary
 * @param  ly y-component of the derivative of the boundary
 * @param  s  Upper integration limit
 * @return    Integral L(s)
 */
__device__ inline coord integrate_upto2(const coord *lx, const coord *ly, coord s) {
    vector2 tmp;
    coord st;
    coord sum = 0.0;
    for (int u = 0; u < QUAD_ORDER; u++) {
        st = (coord)((legendre_points[u]+1.0)*s/2.0);
        tmp.x = evaluate_w_cheb_interpolators(lx, st);
        tmp.y = evaluate_w_cheb_interpolators(ly, st);
        sum += vector2_mag(tmp)*gauss_weights[u];
    }
    return sum*s/2.0;
}

__device__ inline coord find_s_where_integral_reaches(const coord *points, coord inte) {
    coord a = 0.0;
    coord b = 1.0;
    coord c = 0.5;
    for (int t = 0; t < BISECTION_ITERS; t++) {
        coord res = integrate_upto(points,c);
        if (res < inte)
            a = c;
        else
            b = c;
        c = (a+b)/2.0;
    }
    return c;
}


__device__ inline void interpolate_inverse_lineal(const coord *xs, coord *res){
    // Uses the x possition of INNER_POINTS+2 points knowing that their y values are the chebyshev points, to set an array of INNER_POINTS equidistant points on the interval [xs[0],xs[INNER_POINTS+1]].
    coord advance = (xs[INNER_POINTS+1]-xs[0])/((coord)INNER_POINTS+1.0);
    coord pos = xs[0];
    int ival = 0;
    for (int i = 0; i < INNER_POINTS; i++) {
        pos += advance;
        
        while (xs[ival+1] < pos)
            ival+=1;
        
        coord alpha = 0.5;
        if ((xs[ival+1] - xs[ival]) != 0)
            alpha = (pos - xs[ival])/(xs[ival+1] - xs[ival]);
        res[i] = chebyshev[ival]*(1.0-alpha)+chebyshev[ival+1]*alpha;
    }
}

/**
 * Uses the x possition of INNER_POINTS+2 points knowing that their y values are
 * the chebyshev points, to set an array of INNER_POINTS equidistant points on
 * the interval [xs[0],xs[INNER_POINTS+1]].
 *
 * @param AL  Values of arclength of boundary at given equispaced points
 * @param res Stores the values of s where the areclength are found
 */
__device__ inline void interpolate_inverse_lineal2(const coord *AL, coord *res) {
    coord delta_L = (AL[INVERSE_REPARAMETRIZATION_LENGTH - 1])/((coord)INNER_POINTS + 1.0);
    coord pos = AL[0];
    coord s_ival0;
    coord s_ival1;
    for(int i = 0; i < INNER_POINTS; i++) {
        int ival = 0;
        pos = delta_L*(i+1);
        
        while ((ival+1) < INVERSE_REPARAMETRIZATION_LENGTH && AL[ival+1] < pos)
            ival+=1;
        
        s_ival0 = ival/((coord)INVERSE_REPARAMETRIZATION_LENGTH-1.0);
        if (abs(AL[ival+1] - AL[ival]) != 0) {
            s_ival1 = (ival+1)/((coord)INVERSE_REPARAMETRIZATION_LENGTH-1);
            res[i]  = s_ival0+(s_ival1-s_ival0)*(pos-AL[ival])/(AL[ival+1]-AL[ival]);
        }
        else{
            res[i] = s_ival0;
        }
    }
}

#endif
