#ifndef SRC_GRAINS_H
#define SRC_GRAINS_H

#include <assert.h>
#include "geometry.h"

/**
 * Grain structure useful for output results to files
 * Contains list of reference to vertices, number of vertices
 * and total memory size requested.
 */
struct grain {
    vertex **vrtxs;
    int vsize;
    int vlen;
};

/**
 * Init empty grain with enough memory for one vertex.
 * Logically there is no vertex yet.
 *
 * @param grain Pointer to grain struct
 */
inline void init_grain(grain *grain) {
    grain->vsize= 1;
    grain->vrtxs = (vertex**) malloc(grain->vsize*sizeof(vertex*));
    grain->vlen= 0;
}

/**
 * Add a vertex reference to a grain. If the memory limit is reached
 * we ask for more.
 * @param grain Pointer to grain struct
 * @param vrt   Pointer to vertex struct
 */
inline void grain_add_vertex(grain *grain, vertex *vrt) {
    if (grain->vlen == grain->vsize) {
        grain->vsize *= 2;
        grain->vrtxs = (vertex**) realloc((void *)grain->vrtxs, grain->vsize*sizeof(vertex*));
    }
    grain->vrtxs[grain->vlen] = vrt;
    grain->vlen += 1;
}

/**
 * Delete grain by calling free on every vertex reference and grain itself.
 * @param grains Pointer to grain struct
 * @param leng   Number of vertices being destroyed
 */
inline void delete_grains(grain *grains, int leng) {
    for (int i = 0; i < leng; i++)
        free(grains[i].vrtxs);
    free(grains);
}

/**
 * Build grain structure by checking vertices and boundaries.
 * If label_grain_on_bnds is true, The operation to build grain structure
 * labels each boundary associating absolute grain indices,
 * thus nout_grns and out_lengrns are not used and must be passed as NULL pointers.
 *
 * @param vertices    Pointer to vertex array
 * @param n_vertices  Number of vertices
 * @param out_grns    Pointer to grains array
 * @param out_lengrns Number of grains
 */
inline void obtain_grains(vertex *vertices, int n_vertices, grain **out_grns, int *out_lengrns, bool label_grains_on_bnds) {
    // Memory size and number of grains
    int sgrains = 1;
    int lengrains = 0;
    grain *grains;

    // Build local grain array
    if (!label_grains_on_bnds)
        grains = (grain *) malloc(sgrains*sizeof(grain));

    // Create an array of considerated sides of vertices that have already been considerated
    bool *considerated= (bool*) malloc(3*n_vertices*sizeof(bool));
    for (int k = 0; k < 3 * n_vertices; k++)
        considerated[k] = false;

    // For each enabled vertices, check on each not yet considerated side for grains.
    for (int j = 0; j < n_vertices; j++) {
        vertex *start_vrt = &vertices[j];
        if (start_vrt->enabled) {
            // Backtracking of boundaries of a vertex, so we visit them all.
            for (int g = 0; g < 3; g++) {
                if (!considerated[j*3+g]) {
                    grain grn;
                    init_grain(&grn);

                    // Start moving through vertices at this side of the grain.
                    vertex *current_vrt = start_vrt;
                    int current_side = g;
                    do{
                        int jindx = current_vrt-vertices;
                        if (considerated[jindx*3+current_side] || !current_vrt->enabled) {
                            printf("Illegal grains were detected!\n");
                            exit(1);
                        }

                        // Mark the current vertex and side as considerated.
                        considerated[jindx * 3 + current_side] = true;
                        
                        //Check the boundary of the grain.
                        boundary *bnd = current_vrt->boundaries[current_side];
                        if (label_grains_on_bnds) {
                            // Perform grain labeling without associating grains geometric structure.
                            if (bnd->grains[0] == -1) {
                                bnd->grains[0] = lengrains;
                            } 
                            else if (bnd->grains[1] < 0) {
                                bnd->grains[1] = lengrains;
                            } 
                            else {
                                printf("Illegal grains were detected while assigning grain to boundary!\n");
                                exit(1);
                            }
                        } 
                        else {
                            // Otherwise allow the grain to be added to vertex
                            grain_add_vertex(&grn,current_vrt);
                        }

                        // Advance to the next vertex and get the index of the boundary on the next vertex.
                        if (bnd->ini == current_vrt)
                            current_vrt = bnd->end;
                        else
                            current_vrt = bnd->ini;
                        
                        int i=0;
                        while (true) {
                            if (current_vrt->boundaries[i] == bnd) break;
                            i++;
                        }
                        current_side = (i+1)%3;
                    } while (current_vrt != start_vrt);

                    // Save the built grains
                    if (!label_grains_on_bnds) {
                        if (lengrains == sgrains) {
                            sgrains *= 2;
                            grains = (grain *) realloc(grains,sgrains*sizeof(grain));
                        }
                        memcpy(&grains[lengrains],&grn,sizeof(grain));
                    }
                    lengrains+=1;
                }
            }
        }
    }
    free(considerated);
    if (!label_grains_on_bnds) {
        *out_grns = grains;
        *out_lengrns = lengrains;
    }
}

/**
 * Compute grain area.
 *
 * @param  gra      Pointer to grain structure
 * @param  id       Unused variable
 * @param  vertices Vertices array
 * @return          Total grain area
 */
inline double grain_area(grain *gra, int id, vertex* vertices) {
    // Get the points of each vertex of the grain, with origin on the first one.
    double *xpts = (double *) malloc(sizeof(double)*gra->vlen);
    double *ypts = (double *) malloc(sizeof(double)*gra->vlen);
    
    for (int k = 0; k < gra->vlen; k++) {
        xpts[k]= gra->vrtxs[k]->pos.x;
        ypts[k]= gra->vrtxs[k]->pos.y;
    }
    adjust_origin_for_points(xpts, ypts, gra->vlen);
    
    // Sum the apport to the integral of each boundary.
    double total= 0.0;
    vertex *current = gra->vrtxs[0];
    for (int k = 0; k < gra->vlen; k++) {
        // Check the next vertex.
        vertex *next= gra->vrtxs[(k+1)%gra->vlen];
        
        // Find the boundary between this vertex and the next.
        bool reversed;
        int i=0;
        while (true) {
            if(current->boundaries[i]->end == next) {
                reversed = false;
                break;
            }
            if (current->boundaries[i]->ini == next) {
                reversed = true;
                break;
            }
            i++;
            assert(i < 3);
        }
        boundary *bnd = current->boundaries[i];
        double area = 0;
        // Calculate the intergral for the boundary with origin in the ini.
        {
            // The boundary points are adjusted to the origin (0,0)
            // A correction is applied later to obtain correct apportation of area
            double xpos[INNER_POINTS+2];
            double ypos[INNER_POINTS+2];
            boundary_adjusted_points(bnd, xpos, ypos);

            // Calculate the derivate of the y axis.
            double yprime[INNER_POINTS+2];
            derivate_interpolators(ypos, yprime);
            
            // Get the values of the multiplication of both on legendre nodes.
            double legendre_xpos[QUAD_ORDER];
            double legendre_yprime[QUAD_ORDER];
            interpolate_for_legendre(xpos,legendre_xpos);
            interpolate_for_legendre(yprime,legendre_yprime);
            
            for (int r = 0; r < QUAD_ORDER; r++)
                legendre_yprime[r] *= legendre_xpos[r];
            
            // Integrate them
            if (reversed) {
                area= -integrate_legendre(legendre_yprime);
                // Adding constant value since we moved the boundary to the origin
                area += xpts[(k+1)%gra->vlen]*(ypts[(k+1)%gra->vlen] - ypts[k]);
            }
            else {
                area= integrate_legendre(legendre_yprime);
                // Adding constant value since we moved the boundary to the origin
                area += xpts[k]*(ypts[(k+1)%gra->vlen] - ypts[k]);
            }
        }
        // Add to the total integral
        total+= area;

        // Advance the current vertex.
        current = next;
    }
    free(xpts);
    free(ypts);
    return total;
}


/**
Considering:
A_i(t)=\sum_{k\in \Gamma_i} \int_0^1 x_k(s,t)\,\frac{\partial y_k(s,t)}{\partial s}\,ds,
where \Gamma_i is set of boundaries fo grain i,
we theoretical compute:
\frac{d A_i(t)}{dt} = \sum_{k\in \Gamma_i}
                        \int_0^1 \left[
                            \frac{\partial x_k(s,t)}{\partial t}\,\frac{\partial y_k(s,t)}{\partial s}
                            +
                            x_k(s,t)\,\frac{\partial }{\partial s} \frac{\partial y_k(s,t)}{\partial t}
                            // NOTE: In the previous line we switch the order of the mix derivative.
                        \right]\,ds
However, since we are shifting to the origin the grain boundary
x_k(s,t)=xhat(s,t)+x0k
y_k(s,t)=yhat(s,t)+y0k
so the actual integral is:
\frac{d A_i(t)}{dt} = \sum_{k\in \Gamma_i}
                        \int_0^1 \left[
                            \frac{\partial xhat_k(s,t)}{\partial t}\,\frac{\partial y_k(s,t)}{\partial s}
                            +
                            (xhat(s,t)+x0k)\,\frac{\partial }{\partial s} \frac{\partial y_k(s,t)}{\partial t}
                        \right]\,ds
                    = \sum_{k\in \Gamma_i}
                        \int_0^1 \left[
                            \frac{\partial xhat_k(s,t)}{\partial t}\,\frac{\partial y_k(s,t)}{\partial s}
                            +
                            xhat(s,t)\,\frac{\partial }{\partial s} \frac{\partial y_k(s,t)}{\partial t}
                        \right]\,ds
                        +x0k\,\left.\frac{\partial y_k(s,t)}{\partial t}\right|_0^1
                    = \sum_{k\in \Gamma_i}
                        \int_0^1 \left[
                            \frac{\partial xhat_k(s,t)}{\partial t}\,\frac{\partial y_k(s,t)}{\partial s}
                            +
                            xhat(s,t)\,\frac{\partial }{\partial s} \frac{\partial y_k(s,t)}{\partial t}
                        \right]\,ds
                        +x0k\,(Vy_k(1,t)-Vy_k(0,t)).
*/
/**
 * Compute the variation of area.
 *
 * @param  gra      Pointer to grain structure
 * @param  k        Unused variable
 * @return          Return te variation of area
 */
inline double grain_dAdt(grain *gra, int k) {
    // Sum the contribution to the integral of each boundary
    double total = 0.0;
    vertex *current = gra->vrtxs[0];

    for (int k = 0; k < gra->vlen; k++) {
        // Check the next vertex
        vertex *next = gra->vrtxs[(k+1) % gra->vlen];
        
        // Find the boundary between this vertex and the next
        bool reversed;
        int i = 0;
        while (true) {
            if (current->boundaries[i]->end == next) {
                reversed = false;
                break;
            }
            if (current->boundaries[i]->ini == next) {
                reversed = true;
                break;
            }
            i++;
            assert(i < 3);
        }
        
        // Get the current boundary
        boundary *bnd = current->boundaries[i];
        
        // Adjust positions to (0,0)
        double xpos[INNER_POINTS + 2];
        double ypos[INNER_POINTS + 2];
        boundary_adjusted_points(bnd, xpos, ypos);
        
        // Get the boundary velocities
        double velx[INNER_POINTS + 2];
        double vely[INNER_POINTS + 2];
        velx[0] = bnd->ini->vel.x;
        vely[0] = bnd->ini->vel.y;
        for (int ii = 0; ii < INNER_POINTS; ii++) {
            velx[ii+1] = bnd->vels[ii].x;
            vely[ii+1] = bnd->vels[ii].y;
        }
        velx[INNER_POINTS + 1] = bnd->end->vel.x;
        vely[INNER_POINTS + 1] = bnd->end->vel.y;
        
        // Perform estimation of dA/dt
        double x0, dAdt_bnd = 0.0;
        x0 = bnd->ini->pos.x;
        
        // Calculate the spectral derivative of the y axis.
        double yprime[INNER_POINTS+2];
        derivate_interpolators(ypos, yprime);
        
        // Calculate the spectral derivative of velocity y axis
        double velyprime [INNER_POINTS + 2];
        derivate_interpolators(vely, velyprime);
        
        // Interpolate for legendre all the values
        double legendre_xpos[QUAD_ORDER];
        double legendre_yprime[QUAD_ORDER];
        double legendre_velx[QUAD_ORDER];
        double legendre_velyprime[QUAD_ORDER];
        interpolate_for_legendre(xpos, legendre_xpos);
        interpolate_for_legendre(yprime, legendre_yprime);
        interpolate_for_legendre(velx, legendre_velx);
        interpolate_for_legendre(velyprime, legendre_velyprime);
        
        // Multiply element-wise  Vx and dyds, dVyds and x
        // Now legendre_xpos is the integrand!
        for (int q = 0; q < QUAD_ORDER; q++)
            legendre_xpos[q] = (legendre_velx[q] * legendre_yprime[q]) + (legendre_xpos[q] * legendre_velyprime[q]);
        
        dAdt_bnd = integrate_legendre(legendre_xpos) + (x0 * (vely[INNER_POINTS + 1] - vely[0]));
        
        if (reversed)
            dAdt_bnd *= -1;

        total += dAdt_bnd;
        current = next;
    }
    return total;
}

#endif
