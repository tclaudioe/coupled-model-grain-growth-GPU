#ifndef SRC_MOVEMENT_H
#define SRC_MOVEMENT_H

#include "topological.h"
#include "calculus.h"

#define MIN_FLIP_STEPS 10
#define INTERVAL_TOLERANCE 1e-12

/* This function determines if a boundary is convex by
 checking if the tangent vector dot the normal vector rotated
 in 90 degrees keep the sign alog the curve, if they switch
 signs, it means it is not convex.
*/
__device__ __host__ inline bool is_boundary_convex(boundary *bnd) {
    vector2 T_unit, N_unit_rotated;
    int positives_values=0;

    coord xpos[INNER_POINTS+2];
    coord ypos[INNER_POINTS+2];
    boundary_adjusted_points(bnd,xpos,ypos);
    
    // Spectral derivation: dx/ds and dy/ds
    coord lx[INNER_POINTS+2];
    coord ly[INNER_POINTS+2];
    derivate_interpolators(xpos, lx);
    derivate_interpolators(ypos, ly);

    coord Txinterp[QUAD_ORDER];
    coord Tyinterp[QUAD_ORDER];

    // Get unit tangent T at Legendre nodes
    interpolate_for_legendre(lx, Txinterp);
    interpolate_for_legendre(ly, Tyinterp);

    for (int i = 0; i < QUAD_ORDER; i++) {
        N_unit_rotated = vector2_rotate90(vector2_unitary(bnd->dTds[i]));
        T_unit.x = Txinterp[i];
        T_unit.y = Tyinterp[i];
        T_unit = vector2_unitary(T_unit);

        if (vector2_dot(T_unit,N_unit_rotated) > 0.0)
            positives_values++;
    }

    // If all dot products have the same sign, it is convex.
    if(positives_values==0 || positives_values==QUAD_ORDER)
        return true;
    return false;
}

/**
 * Checks if a boundary is stable, that is if, from the initial vertex,
 * no inner point (or the opposite vertex is nearer another if,
 * on the boundary curve, it's after.
 *
 * @param  bnd Pointer to boundary
 * @return     true if the boundary is stable, false otherwise
 */
__device__ __host__ inline bool is_boundary_stable(boundary *bnd) {
    coord max_dist = 0.0;
    coord dist;

    for (int ii = 0; ii < INNER_POINTS; ii++) {
        dist = vector2_mag(vector2_delta_to(bnd->inners[ii],bnd->ini->pos));
        if(dist < max_dist) return false;
        max_dist = dist;
    }

    // This is checking if the distance from each triple juctions
    // is lower than the distance to any interior point.
    dist = vector2_mag(vector2_delta_to(bnd->end->pos,bnd->ini->pos));
    if(dist < max_dist)
        return false;

    // Adding this second part due to symmetry
    max_dist = 0.0;
    
    // Adding computation respect to end point as well.
    for(int ii = INNER_POINTS - 1; ii >= 0; ii--) {
        dist = vector2_mag(vector2_delta_to(bnd->inners[ii],bnd->end->pos));
        if(dist < max_dist) 
            return false;
        max_dist = dist;
    }
    
    // This is the same as before, but now the max_dist variable has a
    // different value.
    dist = vector2_mag(vector2_delta_to(bnd->end->pos, bnd->ini->pos));
    if(dist < max_dist)
        return false;
    return true;
}

/**
 * Boundary Energy Function
 *
 * @param  bnd       Pointer to boundary
 * @param  graindata Array of grain data
 * @return           Energy
 */
__device__ double compute_boundary_energy(boundary *bnd, gdata *graindata, double eps, double delta_energy) {
    #if USE_OLD_ENERGY == 1
    double ori1 = graindata[bnd->grains[0]].orientation;
    double ori2 = graindata[bnd->grains[1]].orientation;
    double dalpha = ori1 - ori2;
    return 1 + 0.5*eps*(1 - pow(cos(4*dalpha),3));
    #else
    double ori1 = graindata[bnd->grains[0]].orientation;
    double ori2 = graindata[bnd->grains[1]].orientation;
    double dalpha = acos(cos(4*(ori1 - ori2)))/4;
    double t_delta = delta_energy;
    double theta_s = 3.14159265358979323846/16;
    
    if (fabs(dalpha) < theta_s) {
        if (fabs(dalpha) < 1e-10)
                return t_delta;
        else
            return (1-t_delta)*(fabs(dalpha)/theta_s)*(1-log(fabs(dalpha)/theta_s))+t_delta;
    }
    else{
        return 1;
    }
    #endif
}
/**
 * Compute boundary energy function per boundary with a given fixed epsilon
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 * @param  graindata   Array of grain data
 */
__global__ void compute_boundaries_energies(boundary* boundaries, int n_boundaries, gdata* graindata, double eps, double delta_energy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            // Compute energy
            bnd->energy = compute_boundary_energy(bnd, graindata, eps, delta_energy);
        }
        tid += gridDim.x * blockDim.x;
    }
}


/**
 * Count within a block the number of boundaries.
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 * @param dev_buffer   Buffer to store the results per block
 */
__global__ void boundaries_per_block(const boundary* boundaries, int n_boundaries, int *dev_buffer) {
    __shared__ int tmp_nboundaries[N_TRDS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    int tmp = 0;
    while (tid < n_boundaries) {
        // Check if the vertex is enabled before add
        if (boundaries[tid].enabled)
            tmp++;
        tid += gridDim.x * blockDim.x;
    }

    tmp_nboundaries[cacheIndex] = tmp;
    __syncthreads();
    
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            tmp_nboundaries[cacheIndex] += tmp_nboundaries[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0)
        dev_buffer[blockIdx.x] = tmp_nboundaries[0];
}

/**
 * Count unstable boundaries in the simulation. The unstability is
 * read from the boundary stable state.
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 * @param dev_buffer   Buffer to store the results per block
 */
__global__ void unstable_boundaries_per_block(const boundary* boundaries, int n_boundaries, int *dev_buffer) {
    __shared__ int tmp_nboundaries[N_TRDS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    int tmp = 0;
    while (tid < n_boundaries) {
        // Check if the vertex is enabled before add
        if(boundaries[tid].enabled && (!boundaries[tid].stable || boundaries[tid].reparam))
            tmp++;
        tid += gridDim.x * blockDim.x;
    }

    tmp_nboundaries[cacheIndex] = tmp;
    __syncthreads();
    
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            tmp_nboundaries[cacheIndex] += tmp_nboundaries[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0)
        dev_buffer[blockIdx.x] = tmp_nboundaries[0];
}

/**
 * Count the number of grains from the number of boundaries.
 * The number of boundaries is three times the number of grains in the system
 * under this periodic condition, thus
 *
 *          n_grains = n_boundaries / 3
 *
 *
 * @param  boundaries   Boundary array
 * @param  n_boundaries Number of boundaries
 * @return              Number of grains
 */
inline int count_grains(const boundary *boundaries, int n_boundaries, int *dev_buffer, int *buffer) {
    // Buffers to store number of boundaries per block
    int tmp_n_boundaries = 0;
    boundaries_per_block<<<N_BLKS, N_TRDS>>>(boundaries, n_boundaries, dev_buffer);
    HERR(cudaMemcpy(buffer, dev_buffer, sizeof(int) * N_BLKS, cudaMemcpyDeviceToHost));
    
    // Add up results
    for(int i = 0; i < N_BLKS; i++)
        tmp_n_boundaries += buffer[i];
    return tmp_n_boundaries / 3;
}

/**
 * Count unstable boundaries
 * @param  boundaries   Boundary array
 * @param  n_boundaries Number of boundaries
 * @return              Number of unstable boundaries
 */
inline int count_unstable_and_reparam_boundaries(const boundary *boundaries, int n_boundaries,
    int *dev_buffer, int *buffer, int step, double time) {
    // Buffers to store number of unstable boundaries per block
    int tmp_n_unstable_boundaries = 0;
    unstable_boundaries_per_block<<<N_BLKS, N_TRDS>>>(boundaries, n_boundaries, dev_buffer);
    HERR(cudaMemcpy(buffer, dev_buffer, sizeof(int) * N_BLKS, cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < N_BLKS; i++)
        tmp_n_unstable_boundaries += buffer[i];

    cudaDeviceSynchronize();
    return tmp_n_unstable_boundaries;
}

/**
 * Expand the boundaries by a specific factor
 *
 * @param boundaries    Boundary array
 * @param n_boundaries  Number of boundaries
 */
__global__ void expand_small_boundaries(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries){
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            // Expand only if the boundary numerically has length zero
            if (bnd->ini->pos.x == bnd->end->pos.x && bnd->ini->pos.y == bnd->end->pos.y) {
                boundary_expand_little_boundary(bnd);
                boundary_set_equidistant_inners(bnd);
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

// Check stability of a boundary
__global__ void set_stability(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled)
            bnd->stable = is_boundary_stable(bnd);
        tid += gridDim.x * blockDim.x;
    }
}

// Check convexity of a boundary
__global__ void check_convexity(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            if ((bnd->t_steps_flip_applied < MIN_FLIP_STEPS) && !is_boundary_convex(bnd)) {
                //bnd->stable = false;
                bnd->inhibited_flip = true;
                boundary_set_equidistant_inners(bnd);
                bnd->stable = true;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Backup the boundary curvature.
 *
 * @param boundaries    Boundaries array
 * @param n_boundaries  Number of boundaries
 */
__global__ void backup_curvatures(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled)
            bnd->prev_curvature = bnd->curvature;
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Compute boundary arclength.
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 */
__global__ void compute_arclen(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            // Create arrays for all the points of the boundary.
            coord xpos[INNER_POINTS+2];
            coord ypos[INNER_POINTS+2];
            boundary_adjusted_points(bnd,xpos,ypos);
            
            // Calculate first derivates:
            coord lx[INNER_POINTS+2];
            coord ly[INNER_POINTS+2];
            derivate_interpolators(xpos,lx);
            derivate_interpolators(ypos,ly);
            
            // Get first derivatives at quadrature nodes
            coord lxgauss[QUAD_ORDER], lygauss[QUAD_ORDER];
            interpolate_for_legendre(lx, lxgauss);
            interpolate_for_legendre(ly, lygauss);
            coord integrand[QUAD_ORDER];
            for (int j = 0; j < QUAD_ORDER; j++) {
                vector2 mag;
                mag.x = lxgauss[j];
                mag.y = lygauss[j];
                integrand[j] = vector2_mag(mag);
            }

            // Backup arclen
            bnd->prev_arclen = bnd->arclen;
            bnd->arclen = integrate_legendre(integrand);
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Compute mean curvatures of boundaries and set the flat flag.
 * If using_tmp is True, the curvature is computed and stored in a tmp variable
 * inside boundary. Otherwise the curvature is set as the real mean curvature
 * and the flatness is set
 *
 * @param boundary          Boundary array
 * @param n_boundaries      Number of boundaries
 * @param using_tmp         If true, store as temporal, otherwise, replace current value
 * @param kappa0            Curvature threshold
 */
__global__ void compute_curvatures(boundary *boundaries, int n_boundaries, bool using_tmp, double kappa0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            // Create arrays for all the points of the frontier.
            coord xpos[INNER_POINTS+2];
            coord ypos[INNER_POINTS+2];
            boundary_adjusted_points(bnd,xpos,ypos);
            coord curv = compute_curvature(xpos, ypos);
            
            // Store the curvature in a temp variable
            if (using_tmp)
                bnd->tmp_curvature = curv;
            // Store the curvature as the real boundary curvature and update flat
            else
                bnd->curvature = curv;
        }
        tid += gridDim.x * blockDim.x;
    }
}


/**
 * Reparametrize a boundary along itself
 * @param bnd Pointer to boundary struct.
 */
__device__ void boundary_reparametrize(boundary* bnd) {
    coord xpos[INNER_POINTS+2];
    coord ypos[INNER_POINTS+2];
    boundary_adjusted_points(bnd,xpos,ypos);
    
    // Calculate first derivates:
    coord lx[INNER_POINTS+2];
    coord ly[INNER_POINTS+2];
    derivate_interpolators(xpos,lx);
    derivate_interpolators(ypos,ly);
    
    if (bnd->arclen > 0) {
        if (!bnd->stable) {
            boundary_set_equidistant_inners(bnd);
        } 
        else {
            // Set each inner point to it's equidistant possition over the curve.
            #if INVERSE_ARCLEN_FUNCTION_REPARAMETRIZATION == 1
                // here we store the arclength samples of a boundary at equispaced s
                coord arclens[INVERSE_REPARAMETRIZATION_LENGTH];
                coord s_tmp;
                
                // L(0) = 0 and L(1) = total arclen
                arclens[0] = 0;
                arclens[INVERSE_REPARAMETRIZATION_LENGTH-1] = bnd->arclen;
                
                for (int ii = 1; ii < INVERSE_REPARAMETRIZATION_LENGTH - 1; ii++) {
                    s_tmp = ((coord)ii) / (INVERSE_REPARAMETRIZATION_LENGTH-1);
                    arclens[ii] = integrate_upto2(lx,ly, s_tmp);
                }

                coord svalues[INNER_POINTS];
                interpolate_inverse_lineal2(arclens, svalues);
                
                for (int ii = 0; ii < INNER_POINTS; ii++) {
                    vector2 relpos;
                    relpos.x = evaluate_w_cheb_interpolators(xpos, svalues[ii]);
                    relpos.y = evaluate_w_cheb_interpolators(ypos, svalues[ii]);
                    bnd->inners[ii] = vector2_adjust(vector2_sum(relpos, bnd->ini->pos));
                }
            #else
                // Calculate the arclen apportation:
                coord apports[INNER_POINTS+2];
                for (int k = 0; k < INNER_POINTS + 2; k++) {
                    vector2 app;
                    app.x = lx[k];
                    app.y = ly[k];
                    apports[k] = vector2_mag(app);
                }

                for (int ii = 0; ii < INNER_POINTS; ii++) {
                    vector2 relpos;
                    coord s_trgt = find_s_where_integral_reaches(apports, (ii+1)*bnd->arclen/(INNER_POINTS+1));
                    relpos.x = evaluate_w_cheb_interpolators(xpos, s_trgt);
                    relpos.y = evaluate_w_cheb_interpolators(ypos, s_trgt);
                    bnd->inners[ii] = vector2_adjust(vector2_sum(relpos, bnd->ini->pos));
                }
            #endif
        }
        // Always unset the reparam flag
        bnd->reparam = false;
    }
}

/**
 * Perform reparametrization procedure.
 * The points are moved along the curve boundary equispaced in arclength.
 *
 * @param boundaries       Boundary array
 * @param n_boundaries     Number of boundaries
 */
__global__ void reparametrize(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            boundary_reparametrize(bnd);
            bnd->stable = true;
        }
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void reparametrize_unstable_and_after_flip(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            if (!bnd->stable) {
                boundary_set_equidistant_inners(bnd);
                bnd->stable = true;
            }
            if (bnd->reparam) {
                boundary_reparametrize(bnd);
                bnd->stable = true;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Reset the flip and inhibited state of boundaries.
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 */
__global__ void reset_boundaries_flip_state(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            // Reset boundary state
            bnd->inhibited_flip = false;
            bnd->to_flip = false;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Compute the boundaries velocities.
 * Also computes one of the three terms for vertices velocities
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 * @param mu
 * @param lambda
 */
__global__ void compute_boundaries_velocities(boundary *boundaries, int n_boundaries, double mu, double lambda) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            // Create arrays for all the points of the boundary
            coord xpos[INNER_POINTS+2];
            coord ypos[INNER_POINTS+2];
            boundary_adjusted_points(bnd,xpos,ypos);
            
            // Calculate first derivates:
            coord lx[INNER_POINTS+2];
            coord ly[INNER_POINTS+2];
            derivate_interpolators(xpos,lx);
            derivate_interpolators(ypos,ly);
            
            // Calculate unitary vectors of the first derivate
            // to calculate ini and end vertices velocities
            vector2 iniv,endv;
            iniv.x = lx[0];
            iniv.y = ly[0];
            endv.x = lx[INNER_POINTS+1];
            endv.y = ly[INNER_POINTS+1];
            iniv = vector2_unitary(iniv);
            endv = vector2_unitary(vector2_rotate180(endv));

            vector2 tmp_nl;
            coord norm_l;

            // Calculate the derivate of the unitary vector.
            // CHECK THIS... use the spectral derivation formula
            // 0. Compute l and dl/ds at Chebyshev nodes, use D^{(2)}_N
            // 1. Compute l(s,t) at gauss nodes
            // 2. Compute dl/ds at gauss nodes
            // 3. use Algorithm 2 to find derivative of unitary vector
            coord xinterp[QUAD_ORDER];
            coord yinterp[QUAD_ORDER];
            derivate_unit_vector(xpos, ypos, xinterp, yinterp);
            
            // DEBUG: Output dT/ds at quadrature nodes to output file
            for (int i = 0; i < QUAD_ORDER; i++) {
                bnd->dTds[i].x = xinterp[i];
                bnd->dTds[i].y = yinterp[i];
            }

            // Set velocities
            bnd->inivel.x = bnd->energy * lambda * (iniv.x+integrate_legendre_mult_by_cheb(xinterp,0));
            bnd->inivel.y = bnd->energy * lambda * (iniv.y+integrate_legendre_mult_by_cheb(yinterp,0));

            // DEBUG: FOR LATER PRINTF
            bnd->ini_int.x = integrate_legendre_mult_by_cheb(xinterp,0);
            bnd->ini_int.y = integrate_legendre_mult_by_cheb(yinterp,0);
            bnd->raw_inivel.x = iniv.x;
            bnd->raw_inivel.y = iniv.y;
            //END DEBUG

            for (int ii = 0; ii < INNER_POINTS; ii++) {
                tmp_nl.x=lx[ii+1];
                tmp_nl.y=ly[ii+1];
                norm_l = vector2_mag(tmp_nl);
                bnd->vels[ii].x = bnd->energy * mu * (integrate_legendre_mult_by_cheb(xinterp, ii+1) / norm_l);
                bnd->vels[ii].y = bnd->energy * mu * (integrate_legendre_mult_by_cheb(yinterp, ii+1) / norm_l);
                
                // Save the normal velocity for future use and debugging
                bnd->normal_vels[ii].x = bnd->vels[ii].x;
                bnd->normal_vels[ii].y = bnd->vels[ii].y;

                // DEBUG
                bnd->raw_normal_vels[ii].x = integrate_legendre_mult_by_cheb(xinterp, ii+1);
                bnd->raw_normal_vels[ii].y = integrate_legendre_mult_by_cheb(yinterp, ii+1);
                //END DEBUG
            }

            bnd->endvel.x = bnd->energy * lambda * (endv.x+integrate_legendre_mult_by_cheb(xinterp,INNER_POINTS+1));
            bnd->endvel.y = bnd->energy * lambda * (endv.y+integrate_legendre_mult_by_cheb(yinterp,INNER_POINTS+1));

            // DEBUG: FOR LATER PRINTF
            bnd->end_int.x = integrate_legendre_mult_by_cheb(xinterp,INNER_POINTS+1);
            bnd->end_int.y = integrate_legendre_mult_by_cheb(yinterp,INNER_POINTS+1);
            bnd->raw_endvel.x = endv.x;
            bnd->raw_endvel.y = endv.y;
            //END DEBUG
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Set all checked flags to false
 */
__global__ void set_checked_false(boundary *boundaries, int n_boundaries, int iteration)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        // Check if boundary it is active
        if (bnd->enabled) {
            bnd->checked[0] = false;
            bnd->checked[1] = false;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Check intersection of boundaries
 */
__global__ void check_intersections(boundary *boundaries, int n_boundaries, int iteration) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        // Check if boundary it is active
        if (bnd->enabled && bnd->arclen > 1e-4) {
            // Apply Newton's method to each grain
            for (int t = 0; t < 2; ++t) {
                // t is the current grain

                // Start from ini in the current boundary
                vertex *ini = bnd->ini;

                // Second boundary set to NULL
                boundary *bnd2 = NULL;
                boundary *bnd_prev = bnd;

                // Iterate over each boundary of grain t until we reach the current boundary bnd
                while (bnd != bnd2) {
                    // Choose boundary connected to this vertex and member of the same grain, different from the current boundary

                    // Boolean used to check if there are more boundaries not visited
                    bool changed = false;

                    // Itearte over each boundary connected to current ini vertex of current boundary
                    for (int i = 0; i < 3; ++i) {

                        // Choose i-th boundary
                        bnd2 = ini->boundaries[i];

                        // If bnd share the current t grain with bnd2 and it is different of bnd and bnd2 has a sufficient large arclen
                        if ((bnd2->arclen > 1e-4) && (bnd != bnd2 && bnd2 != bnd_prev) && ((bnd2->grains[0] == bnd->grains[t]  && !bnd2->checked[0]) || (bnd2->grains[1] == bnd->grains[t] && !bnd2->checked[1])) ) {
                            // A boundary was choosen, so set changed to true
                            changed = true;
                            bnd_prev = bnd2;

                            // Use Newton's method to detect collision
                            // Get boundary coordinates
                            coord xpos_k[INNER_POINTS+2];
                            coord ypos_k[INNER_POINTS+2];

                            // Get second boundary coordinates
                            coord xpos_l[INNER_POINTS+2];
                            coord ypos_l[INNER_POINTS+2];

                            // Fill boundary coordinates

                            // Initial point
                            xpos_k[0] = bnd->ini->pos.x;
                            ypos_k[0] = bnd->ini->pos.y;
                            xpos_l[0] = bnd2->ini->pos.x;
                            ypos_l[0] = bnd2->ini->pos.y;

                            // position 0
                            if (xpos_k[0] - xpos_l[0] > DOMAIN_BOUND/2.0)
                                xpos_l[0] += DOMAIN_BOUND;
                            else if (xpos_k[0] - xpos_l[0] < -DOMAIN_BOUND/2.0)
                                xpos_l[0] -= DOMAIN_BOUND;
                            if (ypos_k[0] - ypos_l[0] > DOMAIN_BOUND/2.0)
                                ypos_l[0] += DOMAIN_BOUND;
                            else if (ypos_k[0] - ypos_l[0] < -DOMAIN_BOUND/2.0)
                                ypos_l[0] -= DOMAIN_BOUND;

                            for (int kk = 0; kk < INNER_POINTS; ++kk) {
                                xpos_k[kk+1] = bnd->inners[kk].x;
                                ypos_k[kk+1] = bnd->inners[kk].y;
                                xpos_l[kk+1] = bnd2->inners[kk].x;
                                ypos_l[kk+1] = bnd2->inners[kk].y;

                                // Correction for pos_k
                                if (xpos_k[0] - xpos_k[kk+1] > DOMAIN_BOUND/2.0)
                                    xpos_k[kk+1] += DOMAIN_BOUND;
                                else if (xpos_k[0] - xpos_k[kk+1] < -DOMAIN_BOUND/2.0)
                                    xpos_k[kk+1] -= DOMAIN_BOUND;
                                
                                if (ypos_k[0] - ypos_k[kk+1] > DOMAIN_BOUND/2.0)
                                    ypos_k[kk+1] += DOMAIN_BOUND;
                                else if (ypos_k[0] - ypos_k[kk+1] < -DOMAIN_BOUND/2.0)
                                    ypos_k[kk+1] -= DOMAIN_BOUND;

                                // Correction for pos_l
                                if (xpos_k[0] - xpos_l[kk+1] > DOMAIN_BOUND/2.0)
                                    xpos_l[kk+1] += DOMAIN_BOUND;
                                else if (xpos_k[0] - xpos_l[kk+1] < -DOMAIN_BOUND/2.0)
                                    xpos_l[kk+1] -= DOMAIN_BOUND;
                                if (ypos_k[0] - ypos_l[kk+1] > DOMAIN_BOUND/2.0)
                                    ypos_l[kk+1] += DOMAIN_BOUND;
                                else if (ypos_k[0] - ypos_l[kk+1] < -DOMAIN_BOUND/2.0)
                                    ypos_l[kk+1] -= DOMAIN_BOUND;
                            }

                            xpos_k[3] = bnd->end->pos.x;
                            ypos_k[3] = bnd->end->pos.y;
                            xpos_l[3] = bnd2->end->pos.x;
                            ypos_l[3] = bnd2->end->pos.y;

                            // Correction for pos_k in position end-1
                            if (xpos_k[0] - xpos_k[3] > DOMAIN_BOUND/2.0)
                                xpos_k[3] += DOMAIN_BOUND;
                            else if (xpos_k[0] - xpos_k[3] < -DOMAIN_BOUND/2.0)
                                xpos_k[3] -= DOMAIN_BOUND;
                            
                            if (ypos_k[0] - ypos_k[3] > DOMAIN_BOUND/2.0)
                                ypos_k[3] += DOMAIN_BOUND;
                            else if (ypos_k[0] - ypos_k[3] < -DOMAIN_BOUND/2.0)
                                ypos_k[3] -= DOMAIN_BOUND;

                            // Correction for pos_l in position end-1
                            if (xpos_k[0] - xpos_l[3] > DOMAIN_BOUND/2.0)
                                xpos_l[3] += DOMAIN_BOUND;
                            else if (xpos_k[0] - xpos_l[3] < -DOMAIN_BOUND/2.0)
                                xpos_l[3] -= DOMAIN_BOUND;
                            if (ypos_k[0] - ypos_l[3] > DOMAIN_BOUND/2.0)
                                ypos_l[3] += DOMAIN_BOUND;
                            else if (ypos_k[0] - ypos_l[3] < -DOMAIN_BOUND/2.0)
                                ypos_l[3] -= DOMAIN_BOUND;

                            // Derivatives
                            coord lk_x[INNER_POINTS+2];
                            coord lk_y[INNER_POINTS+2];
                            coord ll_x[INNER_POINTS+2];
                            coord ll_y[INNER_POINTS+2];

                            // Get non-unitary tangent vector by spectral derivation
                            derivate_interpolators(xpos_k, lk_x);
                            derivate_interpolators(ypos_k, lk_y);
                            derivate_interpolators(xpos_l, ll_x);
                            derivate_interpolators(ypos_l, ll_y);

                            // Use Newton's Method
                            double sk[5] = {0.5, 0.25, 0.25, 0.75, 0.75};
                            double sl[5] = {0.5, 0.25, 0.75, 0.25, 0.75};

                            // Jacobian matrix and vector
                            double J[2][2];
                            double F[2];

                            // Iterate for each initial guess
                            for (int j = 0; j < 5; ++j) {
                                // Initial guess
                                double s[2] = { sk[j], sl[j] };
                                int MAX_IT = 10;

                                for (int it = 0; it < MAX_IT; ++it) {
                                    // Fill the Jacobian matrix
                                    J[0][0] = evaluate_w_cheb_interpolators(lk_x, s[0]);
                                    J[1][0] = evaluate_w_cheb_interpolators(lk_y, s[0]);

                                    J[0][1] = -evaluate_w_cheb_interpolators(ll_x, s[1]);
                                    J[1][1] = -evaluate_w_cheb_interpolators(ll_y, s[1]);

                                    // Compute the vector
                                    F[0] = evaluate_w_cheb_interpolators(xpos_k, s[0]) - evaluate_w_cheb_interpolators(xpos_l, s[1]);
                                    F[1] = evaluate_w_cheb_interpolators(ypos_k, s[0]) - evaluate_w_cheb_interpolators(ypos_l, s[1]);

                                    // Compute the next solution
                                    double det = J[0][0]*J[1][1] - J[1][0]*J[0][1];
                                    double det_x = F[0]*J[1][1] - F[1]*J[0][1];
                                    double det_y = J[0][0]*F[1] - J[1][0]*F[0];

                                    s[0] = s[0] - det_x/det;
                                    s[1] = s[1] - det_y/det;
                                }

                                // Check if a solution was found
                                // Compute the vector
                                F[0] = evaluate_w_cheb_interpolators(xpos_k, s[0]) - evaluate_w_cheb_interpolators(xpos_l, s[1]);
                                F[1] = evaluate_w_cheb_interpolators(ypos_k, s[0]) - evaluate_w_cheb_interpolators(ypos_l, s[1]);
                                
                                if (sqrt(F[0]*F[0] + F[1]*F[1]) < 1e-12) {
                                    if (
                                        ( (INTERVAL_TOLERANCE < s[0] && s[0] + INTERVAL_TOLERANCE < 1.0) && (INTERVAL_TOLERANCE < s[1] && s[1] + INTERVAL_TOLERANCE < 1.0) ) ||
                                        ( (INTERVAL_TOLERANCE < s[0] && s[0] + INTERVAL_TOLERANCE < 1.0) && (0.0 <= s[1] && s[1] <= 1.0) ) ||
                                        ( (INTERVAL_TOLERANCE < s[1] && s[1] + INTERVAL_TOLERANCE < 1.0) && (0.0 <= s[0] && s[0] <= 1.0) )
                                    ) {
                                        // Test if boundary is a neighboor
                                        if (bnd->ini == bnd2->ini || bnd->ini == bnd2->end || bnd->end == bnd2->ini || bnd->end == bnd2->end)
                                            printf("(NEIGHBOR BOUNDARIES) ");
                                        printf("Iteration %d . Boundaries: %d (energy = %f , arclen = %f ), %d (energy = %f , arclen = %f ) . s[0] = %.15f , s[1] = %.15f residual = %.15f . \n", iteration, bnd->id, bnd->energy, bnd->arclen, bnd2->id, bnd2->energy, bnd2->arclen, s[0], s[1], sqrt(F[0]*F[0] + F[1]*F[1]));
                                    }
                                }
                            }
                            if (bnd2->ini == ini)
                                ini = bnd2->end;
                            else if (bnd2->end == ini)
                                ini = bnd2->ini;
                            break;
                        }
                    }

                    if (!changed)
                        bnd2 = bnd;
                }

                // Mark this boundary as checked
                bnd->checked[t] = true;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Compute tangential component of velocity of boundary.
 *
 * @require compute_boundaries_velocities and compute_vertices_velocities executed before!
 * @param boundaries   [description]
 * @param n_boundaries [description]
 */
__global__ void compute_tangential_velocities(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            // Get boundary coordinates
            coord xpos[INNER_POINTS+2];
            coord ypos[INNER_POINTS+2];
            coord Tx[INNER_POINTS+2];
            coord Ty[INNER_POINTS+2];
            coord lx[INNER_POINTS+2];
            coord ly[INNER_POINTS+2];
            boundary_adjusted_points(bnd,xpos,ypos);

            // The Ti-hat vectors, which are orthogonal to Ni-hat
            coord TxH[INNER_POINTS];
            coord TyH[INNER_POINTS];
            vector2 tmp;

            // Spectral derivation: dx/ds and dy/ds
            derivate_interpolators(xpos, lx);
            derivate_interpolators(ypos, ly);

            // Compute norms of xres and yres
            coord norm_l[INNER_POINTS];
            for (int i = 0; i < INNER_POINTS; i++) {
                tmp.x=lx[i+1];
                tmp.y=ly[i+1];
                norm_l[i] = vector2_mag(tmp);
            }

            // Get unit tangent T at Chebyshev nodes
            for (int i = 0; i < INNER_POINTS + 2; i++) {
                tmp.x = lx[i];
                tmp.y = ly[i];
                tmp = vector2_unitary(tmp);
                Tx[i] = tmp.x; Ty[i] = tmp.y;
            }

            coord Txinterp[QUAD_ORDER];
            coord Tyinterp[QUAD_ORDER];

            // Get unit tangent T at Legendre nodes
            interpolate_for_legendre(lx, Txinterp);
            interpolate_for_legendre(ly, Tyinterp);
            for (int i = 0; i < QUAD_ORDER; i++) {
                tmp.x = Txinterp[i]; tmp.y = Tyinterp[i];
                tmp = vector2_unitary(tmp);
                Txinterp[i] = tmp.x; Tyinterp[i] = tmp.y;
            }

            // Compute the integral of the tangent vector times phi_k
            for (int k = 0; k < INNER_POINTS + 2; k++) {
                xpos[k] = integrate_legendre_mult_by_cheb(Txinterp, k);
                ypos[k] = integrate_legendre_mult_by_cheb(Tyinterp, k);
                bnd->Wk[k].x = xpos[k];
                bnd->Wk[k].y = ypos[k];
            }

            // Create matrices A and B, ba, bb such that (A-B)alpha=ba+bb
            coord A[INNER_POINTS * INNER_POINTS];
            coord b[INNER_POINTS], v[2];

            // Compute spectral derivative of current velocities for every point on boundary
            // Notice that vel[0] and vel[INNER_POINTS+1] are TJ velocities and the rest.
            coord velx[INNER_POINTS+2], vely[INNER_POINTS+2];
            velx[0] = bnd->ini->vel.x;
            vely[0] = bnd->ini->vel.y;
            
            for (int i = 0; i < INNER_POINTS; i++) {
                velx[i+1] = bnd->vels[i].x;
                vely[i+1] = bnd->vels[i].y;
            }

            velx[INNER_POINTS+1] = bnd->end->vel.x;
            vely[INNER_POINTS+1] = bnd->end->vel.y;


            // Building TiH, an orthogonal vector to normal velocity
            for (int i = 0; i < INNER_POINTS; i++) {
                tmp.x=velx[i+1];
                tmp.y=vely[i+1];
                
                if (vector2_mag(tmp) < 1e-10) {
                    TxH[i] = Tx[i+1];
                    TyH[i] = Ty[i+1];
                }
                else {
                    tmp = vector2_unitary(tmp);
                    TxH[i] = -tmp.y;
                    TyH[i] = tmp.x;
                }
            }
            
            coord Ydotx[INNER_POINTS+2], Ydoty[INNER_POINTS+2];
            derivate_interpolators(velx, Ydotx);
            derivate_interpolators(vely, Ydoty);

            // only have a "normal" component.
            for (int i = 0; i < INNER_POINTS; i++) {
                // Compute A - B
                v[0] = Tx[i+1] * bnd->arclen / norm_l[i];
                v[1] = Ty[i+1] * bnd->arclen / norm_l[i];
                
                for (int j = 0; j < INNER_POINTS; j++)
                    A[MTRXIDX2(i,j)] = diffMatrix[MTRXIDX(i+1, j+1)]*(v[0]*TxH[j] + v[1]*TyH[j]);

                for (int j = 0; j < INNER_POINTS; j++) {
                    for (int k = 0; k < INNER_POINTS + 2; k++)
                        A[MTRXIDX2(i, j)] -= diffMatrix[MTRXIDX(k, j+1)]*(bnd->Wk[k].x*TxH[j] + bnd->Wk[k].y*TyH[j]);
                }

                // Compute ba + bb
                b[i] = -(v[0]*Ydotx[i+1] + v[1]*Ydoty[i+1]);
                for (int k = 0; k < INNER_POINTS + 2; k++)
                    b[i] += bnd->Wk[k].x * Ydotx[k] + bnd->Wk[k].y * Ydoty[k];
            }

            for (int j = 0; j < INNER_POINTS-1; j++) {
               for (int i = j+1; i < INNER_POINTS; i++) {
                   A[MTRXIDX2(i, j)] = A[MTRXIDX2(i, j)]/A[MTRXIDX2(j, j)];
                   for (int h = j+1; h < INNER_POINTS; h++)
                       A[MTRXIDX2(i, h)] = A[MTRXIDX2(i, h)] - A[MTRXIDX2(i, j)]*A[MTRXIDX2(j, h)];
               }
            }

            coord c[INNER_POINTS];
            c[0] = b[0];
            for (int i = 1; i < INNER_POINTS; i++) {
                tmp.x = 0;

                for (int z = 0; z < i; z++)
                    tmp.x = tmp.x + A[i*INNER_POINTS + z] * c[z];

                c[i] = b[i]-tmp.x;
            }

            b[INNER_POINTS-1] = (1.0/A[INNER_POINTS*INNER_POINTS-1]) * c[INNER_POINTS-1];
            for (int i = INNER_POINTS-2; i > -1; i--) {
                tmp.x = 0;
                for (int z = i+1; z < INNER_POINTS; z++)
                    tmp.x = tmp.x + A[i*INNER_POINTS + z] * b[z];
                b[i] = (1.0/A[i*INNER_POINTS+i])* (c[i]-tmp.x);
            }

            // Add tangential component
            for (int i = 0; i < INNER_POINTS; i++) {
                vector2 tvel;
                tvel.x = b[i]*TxH[i];
                tvel.y = b[i]*TyH[i];
                
                bnd->tangent_vels[i] = tvel;
                bnd->raw_tangent_vels[i] = tvel;
                bnd->vels[i].x += tvel.x;
                bnd->vels[i].y += tvel.y;

                // Store alphas
                bnd->alphas[i] = b[i];
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Compute vertices velocities. This kernel assumes that the velocity terms
 * were computed using the compute_boundary_velocities kernel.
 *
 * @param vertices   Vertices array
 * @param n_vertices Number of vertices
 */
__global__ void compute_vertices_velocities(vertex *vertices, int n_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_vertices) {
        vertex *vrt = &vertices[tid];
        if (vrt->enabled) {
            // Sum all the velocitties terms for each vertex.
            vector2 totvel;
            totvel.x = 0;
            totvel.y = 0;

            for (int i = 0; i < 3; i++) {
                boundary *bnd = vrt->boundaries[i];
                if (bnd->ini == vrt)
                    totvel = vector2_sum(totvel, bnd->inivel);
                else
                    totvel = vector2_sum(totvel, bnd->endvel);
            }
            vrt->vel = totvel;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Correct the total velocities of the boundaries under the assumption
 * that the displacement shall not exceed the theoretical steady state
 * of the boundary, which is a straight line between vertices.
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 */
__global__ void correct_inner_velocities(boundary *boundaries, int n_boundaries, coord delta_t) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            // Vertices of boundary
            vector2 A = bnd->ini->pos;
            vector2 B = bnd->end->pos;
            
            // Matrix of the problem
            coord M[4] = {0.0, A.x - B.x,
                          0.0, A.y - B.y};
            // RHS
            coord b[2] = {0.0, 0.0};
            coord det = 0.0, w = 0.0;
            
            // Correct each interior point
            for (int i = 0; i < INNER_POINTS; i++) {
                M[0] = bnd->vels[i].x;
                M[2] = bnd->vels[i].y;
                b[0] = A.x - bnd->inners[i].x;
                b[1] = A.y - bnd->inners[i].y;
                det = M[0] * M[3] - M[1] * M[2];

                // Check if we can solve the system
                if (abs(det) >= 1e-12) {
                    // Only get the first component of the solution
                    w = (-b[1]*M[1] + b[0]*M[3]) / det;
                    w /= delta_t;
                    if (w < 1.0) {
                        if (w > 0.0) {
                            bnd->vels[i].x *= w;
                            bnd->vels[i].y *= w;
                        } 
                        else {
                            bnd->vels[i].x = 0.0;
                            bnd->vels[i].y = 0.0;
                            // Set the boundary as unstable
                            bnd->stable = false;
                        }
                        // Always reparam if we are modifying velocities
                        bnd->reparam = true;
                    }
                }
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Detect the flippings. If the boundary is flat, the extinction time
 * is estimated using the vertex model formula. Otherwise an integral formula
 * is used.
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 * @param delta_t      Delta t of simulation
 */
__global__ void detect_flips(boundary *boundaries, int n_boundaries, coord delta_t) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            if (bnd->t_ext <= delta_t && bnd->t_ext >= 0) {
                if (bnd->to_flip){
                    // We already know that bnd->t_ext it is positive
                    if (bnd->t_ext < bnd->t_ext_to_flip)
                        bnd->t_ext_to_flip=bnd->t_ext;
                }
                else {
                    bnd->to_flip = true;
                    bnd->t_ext_to_flip=bnd->t_ext;
                }

            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void compute_extinction_times(boundary *boundaries, int n_boundaries, double lambda) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Velocity of each boundary point
    coord velx[INNER_POINTS+2], vely[INNER_POINTS+2];
    
    // Derivative of l^k with respect to time
    coord dldtx[INNER_POINTS+2], dldty[INNER_POINTS+2];
    
    // Coordinates of each boundary point
    coord xpos[INNER_POINTS+2], ypos[INNER_POINTS+2];
    
    // l^k for each boundary point
    coord lx[INNER_POINTS+2], ly[INNER_POINTS+2];
    
    // Same arrays at gaussian quadrature nodes
    coord dldtxgauss[QUAD_ORDER], dldtygauss[QUAD_ORDER];
    coord unitlxgauss[QUAD_ORDER], unitlygauss[QUAD_ORDER];
    coord f[QUAD_ORDER];

    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            /********** Coupled model extinction time **********/
            // Build derivative of each velocity with respect to s (chebyshev parameter)
            velx[0] = bnd->ini->vel.x;
            vely[0] = bnd->ini->vel.y;
            for (int ii = 0; ii < INNER_POINTS; ii++) {
                velx[ii+1] = bnd->vels[ii].x;
                vely[ii+1] = bnd->vels[ii].y;
            }
            velx[INNER_POINTS+1] = bnd->end->vel.x;
            vely[INNER_POINTS+1] = bnd->end->vel.y;
            
            derivate_interpolators(velx, dldtx);
            derivate_interpolators(vely, dldty);
            
            // Build l^k / ||l^k||
            boundary_adjusted_points(bnd, xpos, ypos);
            derivate_interpolators(xpos, lx);
            derivate_interpolators(ypos, ly);
            
            // Obtain dl^k / dt and l^k / ||l^k|| at quadrature nodes
            interpolate_for_legendre(dldtx,dldtxgauss);
            interpolate_for_legendre(dldty,dldtygauss);
            
            //ext_times[tid] = dldtxgauss[0];
            interpolate_for_legendre(lx,unitlxgauss);
            interpolate_for_legendre(ly,unitlygauss);
            
            // Integrate to obtain dL / dt
            for (int ii = 0; ii < QUAD_ORDER; ii++) {
                vector2 unitl_ii;
                unitl_ii.x = unitlxgauss[ii];
                unitl_ii.y = unitlygauss[ii];
                unitl_ii = vector2_unitary(unitl_ii);
                unitlxgauss[ii] = unitl_ii.x;
                unitlygauss[ii] = unitl_ii.y;
                f[ii] = dldtxgauss[ii] * unitl_ii.x + dldtygauss[ii] * unitl_ii.y;
            }
            coord dLdt = integrate_legendre(f);
            bnd->dLdt = dLdt;

            // Compute extinction time
            bnd->t_ext_curv = -bnd->arclen/dLdt;

            /********** Vertex code extinction time **********/
            // Get velocities of triple junctions
            vector2 inivel, endvel;
            inivel.x = bnd->ini->vel.x;
            inivel.y = bnd->ini->vel.y;
            endvel.x = bnd->end->vel.x;
            endvel.y = bnd->end->vel.y;
            
            // Get the unit tangent vector of boundary
            coord xps[2], yps[2];
            xps[0] = bnd->ini->pos.x; xps[1] = bnd->end->pos.x;
            yps[0] = bnd->ini->pos.y; yps[1] = bnd->end->pos.y;
            adjust_origin_for_points(xps, yps, 2);
            
            vector2 diff;
            diff.x = xps[1] - xps[0];
            diff.y = yps[1] - yps[0];
            coord norm = vector2_mag(diff);
            vector2 T = vector2_unitary(diff);
            
            // Compute vector P = Vj - Vi + 2\lambda T
            vector2 P;
            P.x = endvel.x - inivel.x + 2.0*lambda*T.x;
            P.y = endvel.y - inivel.y + 2.0*lambda*T.y;
            
            // Compute extinction time
            bnd->t_ext_vert = norm * (vector2_dot(P, T) + 2.0) / (4.0 - vector2_mag2(P));
            bnd->t_ext = bnd->t_ext_curv;
        }
        tid += gridDim.x * blockDim.x;
    }
}

// This seems deprecated
__global__ void check_consistency(vertex *vertices, int jlen, bool *ans) {
    __shared__ bool cache[N_TRDS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool consistent = true;

    while (tid < jlen) {
        vertex *vrt = &vertices[tid];
        if (vrt->enabled) {
            consistent = vertex_are_flips_around_consistent(vrt, ans == NULL);
            if (ans!=NULL && !consistent) break;
        }
        tid += gridDim.x * blockDim.x;
    }

    // Converge the block to one answer.
    if (ans != NULL) {
        cache[threadIdx.x] = consistent;
        __syncthreads();
        int i = blockDim.x/2;

        while (i > 0) {
            if (threadIdx.x < i)
                cache[threadIdx.x] = cache[threadIdx.x] && cache[threadIdx.x+i];
            __syncthreads();
            i /= 2;
        }
        if (threadIdx.x == 0)
            ans[blockIdx.x] = cache[0];
    }
}

/**
 * Apply flipping to boundaries marked to flip this time-step
 * and not inhibited to do so. Boundary energy must be recomputed.
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 * @param graindata    Array of grain data
 * @param delta_t      Delta t of simulation
 * @param victim_bnd   If valid value, forces the flip
 */
__global__ void apply_flips(boundary *boundaries, int n_boundaries, gdata *graindata, coord delta_t, int victim_bnd) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool status;

    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            // Check if current boundary is being forced
            // No kernel should execute this if victim_bnd = -1;
            if (tid == victim_bnd && tid > 0) {
                boundary_force_flip(bnd);
                printf("Forcing flip of boundary %d\n", bnd->id);
                printf("to_flip = %d\n", bnd->to_flip);
                printf("inhibited_flip = %d\n", bnd->inhibited_flip);
            }

            // The flip
            if (bnd->to_flip && !bnd->inhibited_flip && bnd->n_votes == 2 && bnd->t_steps_flip_applied > MIN_FLIP_STEPS) {
                // Perform flip
                status = boundary_apply_flip(bnd, delta_t);
                bnd->t_steps_flip_applied=0;
                if (tid == victim_bnd)
                    printf("Boundary %d has flipped\n", bnd->id);
                // TEMPORAL PRINT: REMOVE IN THE FINAL VERSION
                printf("FLIPPING: BOUNDARY %d HAS BEEN FLIPPED\n", bnd->id);
                // END DEBUG
            }
            else
                bnd->t_steps_flip_applied++;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Check which set of double boundaries must be deleted.
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 * @param ans          Output array
 */
__global__ void prepare_double_boundary_deletion(boundary *boundaries, int n_boundaries, bool *ans) {
    __shared__ bool cache[N_TRDS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool has_doubles = false;

    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            bool would_destroy= boundary_delete_double_boundaries(bnd, false);
            bnd->autorized_to_delete = would_destroy;
            if (would_destroy)
                has_doubles = true;
        }
        tid += gridDim.x * blockDim.x;
    }

    // Converge the block to one answer.
    if (ans != NULL) {
        cache[threadIdx.x] = has_doubles;
        __syncthreads();
        int i = blockDim.x/2;

        while (i > 0) {
            if (threadIdx.x < i)
                cache[threadIdx.x] = cache[threadIdx.x] || cache[threadIdx.x+i];
            __syncthreads();
            i /= 2;
        }
        if (threadIdx.x == 0)
            ans[blockIdx.x] = cache[0];
    }
}

/**
 * Delete the double boundaries.
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 */
__global__ void delete_double_boundaries(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled && bnd->autorized_to_delete)
            boundary_delete_double_boundaries(bnd, true);
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Evolve boundaries given the velocity and the current delta_t.
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 * @param delta_t      Delta t of simulation
 */
__global__ void evolve_boundaries(boundary *boundaries, int n_boundaries, coord delta_t) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while(tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            // If a boundary is not stable, it does not move.
            if (bnd->stable) {
                for (int ii = 0; ii < INNER_POINTS; ii++) {
                    vector2 disp;
                    disp.x = bnd->vels[ii].x * delta_t;
                    disp.y = bnd->vels[ii].y * delta_t;
                    bnd->inners[ii] = vector2_adjust(vector2_sum(bnd->inners[ii], disp));
                }
            }
            else {
                boundary_set_equidistant_inners(bnd);
                bnd->stable = true;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Evolve vertices given the velocity and the current delta_t.

 */
__global__ void evolve_vertices(vertex *vertices, int n_vertices, coord delta_t) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_vertices) {
        vertex *vrt = &vertices[tid];
        if (vrt->enabled) {
            vector2 disp;
            disp.x = vrt->vel.x * delta_t;
            disp.y = vrt->vel.y * delta_t;
            vrt->pos = vector2_adjust(vector2_sum(vrt->pos, disp));
        }
        tid += gridDim.x * blockDim.x;
    }
}


/**
 * [RK2_backup_vertices_positions description]
 * @param dev_vrtX   [description]
 * @param vertices   [description]
 * @param n_vertices [description]
 */
__global__ void RK2_backup_vertices_positions(double *dev_vrtX, vertex *vertices, int n_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_vertices) {
        vertex *vrt = &vertices[tid];
        if (vrt->enabled) {
            dev_vrtX[2*tid] = vrt->pos.x;
            dev_vrtX[2*tid+1] = vrt->pos.y;
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * [RK2_backup_boundaries_positions description]
 * @param dev_bndX     [description]
 * @param boundaries   [description]
 * @param n_boundaries [description]
 */
__global__ void RK2_backup_boundaries_positions(double *dev_bndX, boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            for (int ii = 0; ii < INNER_POINTS; ii++) {
                dev_bndX[2*tid*INNER_POINTS + ii] = bnd->inners[ii].x;
                dev_bndX[(2*tid+1)*INNER_POINTS + ii] = bnd->inners[ii].y;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Store the current positions of the system.
 *
 * @param dev_vrtX       Array to store vertices positions
 * @param dev_vertices   Vertex array
 * @param n_vertices     Number of vertices
 * @param dev_bndX       Array to store boundaries positions
 * @param dev_boundaries Boundary array
 * @param n_boundaries   Number of boundaries
 */
void RK2_backup_positions(double *dev_vrtX, vertex *vertices, int n_vertices,
                          double *dev_bndX, boundary *boundaries, int n_boundaries) {
    RK2_backup_boundaries_positions<<<N_BLKS, N_TRDS>>>(dev_bndX, boundaries, n_boundaries);
    RK2_backup_vertices_positions<<<N_BLKS, N_TRDS>>>(dev_vrtX, vertices, n_vertices);
}

__global__ void RK2_compute_k1_vertices(vertex *vertices, int n_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_vertices) {
        vertex *vrt = &vertices[tid];
        if (vrt->enabled) {
            vrt->vel.x *= 0.5;
            vrt->vel.y *= 0.5;
        }
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void RK2_compute_k1_boundaries(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled) {
            for (int ii = 0; ii < INNER_POINTS; ii++) {
                bnd->vels[ii].x *= 0.5;
                bnd->vels[ii].y *= 0.5;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

void RK2_compute_k1(vertex *dev_vrtX, int n_vertices, boundary *dev_bndX, int n_boundaries) {
    RK2_compute_k1_vertices<<<N_BLKS, N_TRDS>>>(dev_vrtX, n_vertices);
    RK2_compute_k1_boundaries<<<N_BLKS, N_TRDS>>>(dev_bndX, n_boundaries);
}

__global__ void RK2_evolve_vertices(double *dev_vrtX, vertex *dev_vertices, int n_vertices, double delta_t) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_vertices) {
        vertex *vrt = &dev_vertices[tid];
        if (vrt->enabled) {
            vector2 pos, disp;
            pos.x = dev_vrtX[2*tid];
            pos.y = dev_vrtX[2*tid+1];
            disp.x = vrt->vel.x * delta_t;
            disp.y = vrt->vel.y * delta_t;
            vrt->pos = vector2_adjust(vector2_sum(pos, disp));
        }
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void RK2_evolve_boundaries(double *dev_bndX, boundary *dev_boundaries, int n_boundaries, double delta_t) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_boundaries) {
        boundary *bnd = &dev_boundaries[tid];
        if (bnd->enabled) {
            if (bnd->stable) {
                for (int ii = 0; ii < INNER_POINTS; ii++) {
                    vector2 pos, disp;
                    pos.x = dev_bndX[2*tid*INNER_POINTS + ii];
                    pos.y = dev_bndX[(2*tid+1)*INNER_POINTS + ii];
                    disp.x = bnd->vels[ii].x * delta_t;
                    disp.y = bnd->vels[ii].y * delta_t;
                    bnd->inners[ii] = vector2_adjust(vector2_sum(pos, disp));
                }
            }
            else {
                boundary_set_equidistant_inners(bnd);
                bnd->stable = true;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

void RK2_evolve(double *dev_vrtX, vertex *dev_vertices, int n_vertices,
                double *dev_bndX, boundary *dev_boundaries, int n_boundaries, double delta_t) {
    RK2_evolve_vertices<<<N_BLKS, N_TRDS>>>(dev_vrtX, dev_vertices, n_vertices, delta_t);
    RK2_evolve_boundaries<<<N_BLKS, N_TRDS>>>(dev_bndX, dev_boundaries, n_boundaries, delta_t);
}

/**
 * Reset the votes obtained by each boundary
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 */
__global__ void reset_boundaries_votes(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled)
            bnd->n_votes = 0;
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Reset the votes stored in each vertex
 *
 * @param vertices   Vertices array
 * @param n_vertices Number of vertices
 */
__global__ void reset_vertices_votes(vertex* vertices, int n_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_vertices) {
        vertex *vrt = &vertices[tid];
        if (vrt->enabled)
            vrt->voted = NULL;
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Perform the polling where each vertex votes for the adjacent boundary
 * with lowest extinction time.
 *
 * @param vertices   Vertices array
 * @param n_vertices Number of vertices
 */
__global__ void vertices_vote(vertex* vertices, int n_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_vertices) {
        vertex *vrt = &vertices[tid];
        if (vrt->enabled) {
            double min_t_ext = 99999999;
            int argmin = -1;
            for (int i = 0; i < 3; i++) {
                boundary *bnd = vrt->boundaries[i];
                
                // Check if boundary meets requirements
                if (bnd->to_flip && !bnd->inhibited_flip && (bnd->t_ext_to_flip <= min_t_ext) && (bnd->t_steps_flip_applied>MIN_FLIP_STEPS)) {
                    min_t_ext = bnd->t_ext_to_flip;
                    argmin = i;
                }
            }

            // Vertex didn't vote
            if (argmin == -1)
                vrt->voted = NULL; 
            else
                vrt->voted = vrt->boundaries[argmin];
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Count the votes obtained for each boundary given by the polling made over vertices.

 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 */
__global__ void count_boundaries_votes(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        bnd->n_votes = 0;
        if (bnd->enabled) {
            if (bnd->to_flip && !bnd->inhibited_flip && (bnd->t_steps_flip_applied>MIN_FLIP_STEPS)) {
                if(bnd->ini->voted == bnd)
                    bnd->n_votes++;
                if(bnd->end->voted == bnd)
                    bnd->n_votes++;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Inhibit flips around a boundary if the current boundary is:
 *     - enabled
 *     - marked as flip
 *     - it has 2 votes
 *     - is not inhibited
 *
 * @param boundaries   Boundary array
 * @param n_boundaries Number of boundaries
 */
__global__ void inhibit_flips_around(boundary *boundaries, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled && bnd->to_flip && bnd->n_votes == 2 && !bnd->inhibited_flip && (bnd->t_steps_flip_applied>MIN_FLIP_STEPS))  {
            for (int i = 0; i < 3; i++) {
                if (bnd != bnd->ini->boundaries[i])
                    bnd->ini->boundaries[i]->inhibited_flip = true;
                if (bnd != bnd->end->boundaries[i])
                    bnd->end->boundaries[i]->inhibited_flip = true;
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Reduction kernel to compare the inhibited state between boundaries
 * and stored state.
 *
 * @param boundaries       Boundary array
 * @param n_boundaries     Number of boundaries
 * @param candidate_buffer Buffer to store inhibited state
 * @param dev_buffer       Buffer to store results per block
 */
__global__ void flips_per_block(boundary *boundaries, int n_boundaries,
                                bool *candidate_buffer, bool *dev_buffer) {
    // Shared buffer per block with capability for 32 threads
    __shared__ bool tmp_compare[N_TRDS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    bool tmp = true;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled)
            tmp *= (bnd->inhibited_flip == candidate_buffer[tid]);
        tid += gridDim.x * blockDim.x;
    }
    tmp_compare[cacheIndex] = tmp;
    __syncthreads();

    int i = blockDim.x/2;
    
    while (i != 0) {
        if (cacheIndex < i)
            tmp_compare[cacheIndex] += tmp_compare[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0)
        dev_buffer[blockIdx.x] = tmp_compare[0];
}

/**
 * Compare the current state of boundaries with the previous state stored.
 *
 * @param  boundaries       Boundary array
 * @param  n_boundaries     Number of boundaries
 * @param  candidate_buffer Buffer to store inhibited state
 * @return                  True if the configuration is equal, false otherwise
 */
inline bool compare_candidates(boundary *boundaries, int n_boundaries, bool *candidate_buffer,
    bool *buffer, bool *dev_buffer) {
    // Buffer to store results per block
    flips_per_block<<<N_BLKS, N_TRDS>>>(boundaries, n_boundaries, candidate_buffer, dev_buffer);
    HERR(cudaMemcpy(buffer, dev_buffer, sizeof(bool) * N_BLKS, cudaMemcpyDeviceToHost));
    
    bool tmp =true;
    for (int i = 0; i < N_BLKS; i++)
        tmp *= buffer[i];
    
    return tmp;
}

/**
 * Write the inhibited state of each boundary to a buffer.
 *
 * @param boundaries       Boundary array
 * @param n_boundaries     Number of boundaries
 * @param candidate_buffer Buffer to store inhibited state
 */
__global__ void write_candidate_results(boundary *boundaries, int n_boundaries, bool *candidate_buffer) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        boundary *bnd = &boundaries[tid];
        if (bnd->enabled)
            candidate_buffer[tid] = bnd->inhibited_flip;
        else
            candidate_buffer[tid] = false;
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void init_candidate_buffer(bool *candidate_buffer, int n_boundaries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < n_boundaries) {
        candidate_buffer[tid] = false;
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Polling system to unlabel conflicting boundaries.
 * The algorithm stores the inhibited state of each boundary and resolves
 * conflicts until two consecutive states are equal.
 */
inline void polling_system(vertex *vertices, int n_vertices, boundary *boundaries, int n_boundaries,
    bool *candidate_buffer, bool *buffer_polling, bool *dev_buffer_polling) {
    // Init candidate buffer
    init_candidate_buffer<<<N_BLKS, N_TRDS>>>(candidate_buffer, n_boundaries);
    
    // Convergence implies two configurations are the same over time.
    bool convergence = false;

    while (!convergence) {
        // Reset the boundary total number of votes
        reset_boundaries_votes<<<N_BLKS, N_TRDS>>>(boundaries, n_boundaries);
        
        // Reset the vertices votes
        reset_vertices_votes<<<N_BLKS, N_TRDS>>>(vertices, n_vertices);
        
        // Make the vertices vote over the remaining candidate boundaries
        vertices_vote<<<N_BLKS, N_TRDS>>>(vertices, n_vertices);
        
        // Count each candidate boundary votes
        count_boundaries_votes<<<N_BLKS, N_TRDS>>>(boundaries, n_boundaries);
        cudaDeviceSynchronize();
        
        // inhibit flips arround the boundaries with enough votes
        inhibit_flips_around<<<N_BLKS, N_TRDS>>>(boundaries, n_boundaries);
        
        // Check if the current candidates are the same as before, stored in candidate_buffer
        convergence = compare_candidates(boundaries, n_boundaries, candidate_buffer, buffer_polling, dev_buffer_polling);
        write_candidate_results<<<N_BLKS, N_TRDS>>>(boundaries, n_boundaries, candidate_buffer);
    }
}


/**
 * Updating state variables
 */
inline void update_state_variables(vertex *dev_vertices, int n_vertices,
                                    boundary *dev_boundaries, int n_boundaries,
                                    gdata* dev_graindata, options opt) {
    
    // Compute arclengths
    compute_arclen<<<N_BLKS, N_TRDS>>>(dev_boundaries,n_boundaries);
    
    // Compute grain boundary energies
    compute_boundaries_energies<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries, dev_graindata, opt.eps, opt.delta_energy);
    
    // Compute velocities
    compute_boundaries_velocities<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries, opt.mu, opt.lambda);
    compute_vertices_velocities<<<N_BLKS, N_TRDS>>>(dev_vertices, n_vertices);
    compute_tangential_velocities<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries);
    compute_extinction_times<<<N_BLKS, N_TRDS>>>(dev_boundaries,n_boundaries, opt.lambda);
}

/**
 * Detect and apply topological changes
 */
inline void detect_and_apply_topological_changes(vertex *dev_vertices, int n_vertices,
                                    boundary *dev_boundaries, int n_boundaries,
                                    gdata* dev_graindata, bool* boolarr, bool* dev_boolarr,
                                    bool *candidate_buffer, bool *buffer_polling,
                                    bool *dev_buffer_polling, options *opt, coord delta_t) {
    // Compute extinction times
    // DETECT_FLIPS NEEDS TO BE CALLED AFTER compute_extinction_times
    detect_flips<<<N_BLKS, N_TRDS>>>(dev_boundaries,n_boundaries, delta_t);
    
    // Poll which boundaries should flip
    polling_system(dev_vertices, n_vertices, dev_boundaries, n_boundaries,  candidate_buffer,
                  buffer_polling, dev_buffer_polling);
    
    // Apply the flipping procedure
    apply_flips<<<N_BLKS,N_TRDS>>>(dev_boundaries, n_boundaries, dev_graindata, delta_t, opt->victim_bnd);
    cudaDeviceSynchronize();
    opt->victim_bnd = -1;
    
    while (true) {
        prepare_double_boundary_deletion<<<N_BLKS, N_TRDS>>>(dev_boundaries,n_boundaries,dev_boolarr);
        HERR(cudaMemcpy(boolarr,dev_boolarr,sizeof(bool)*N_BLKS,cudaMemcpyDeviceToHost));
        
        bool has_doubles = false;
        for (int i = 0; i < N_BLKS; i++) {
            has_doubles = boolarr[i];
            if (has_doubles) break;
        }

        if (!has_doubles) break;
        delete_double_boundaries<<<N_BLKS,N_TRDS>>>(dev_boundaries,n_boundaries);
    }
    reset_boundaries_flip_state<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries);
}

#endif
