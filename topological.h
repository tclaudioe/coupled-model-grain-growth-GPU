#ifndef SRC_TOPOLOGICAL_H
#define SRC_TOPOLOGICAL_H

#include <math.h>
#include "utils.h"
#include "geometry.h"

/**
 * Force the boundary flip by setting the state variables.
 * This function to work properly must be called AFTER deciding
 * which boundarys will flip.
 * @param bnd [description]
 */
__device__ inline void boundary_force_flip(boundary *bnd) {
    bnd->to_flip = true;
    bnd->inhibited_flip = false;
}

/** From SE code
 * Apply a flipping to a boundary.
 *
 *                                     C   G2  F
 *                                      \     /
 *      C    G2    F                     \   /
 *       \        /                        B
 *   G3   A------B   G1    =flip=>   G3    |   G1
 *       /        \                        A
 *      D    G4    E                     /   \
 *                                      /     \
 *                                     D   G4  E
 *
 * @param bnd Pointer to boundary struct
 */
__device__ inline bool boundary_apply_flip(boundary *bnd, coord delta_t) {
    // Get the boundary after front on the ini.
    vertex *A = bnd->ini;
    vertex *B = bnd->end;

    // The boundaries are ordered CLOCKWISE at each vertex.
    int i = 0;
    while (true) {
        if (A->boundaries[i] == bnd) break;
        i++;
    }
    i = (i+2)%3;
    boundary *bnd_AC = A->boundaries[i];
    
    // Get the boundary after front on the end.
    int j = 0;
    while (true) {
        if (B->boundaries[j] == bnd) break;
        j++;
    }
    j = (j+2)%3;
    boundary *bnd_BE = B->boundaries[j];
    
    // Swap the frontier pointers on front ends
    A->boundaries[i] = bnd_BE;
    B->boundaries[j] = bnd_AC;
    
    // Connect the swapping frontiers to the other ends.
    // Building BC
    if (bnd_AC->ini == A)
        bnd_AC->ini = B;
    else
        bnd_AC->end = B;

    if (bnd_BE->ini == B)
        bnd_BE->ini = A;
    else
        bnd_BE->end = A;
    
    // Change grains of flipping bundary
    // Finding G3
    int G3 = -1;
    int changed;
    
    if (bnd->grains[0] == bnd_AC->grains[0]) {
        bnd->grains[0] = bnd_AC->grains[1];
        changed = 0;
        G3 = bnd->grains[0];
    } 
    else if (bnd->grains[0] == bnd_AC->grains[1]) {
        bnd->grains[0] = bnd_AC->grains[0];
        changed = 0;
        G3 = bnd->grains[0];
    }
    else if (bnd->grains[1] == bnd_AC->grains[0]) {
        bnd->grains[1] = bnd_AC->grains[1];
        changed = 1;
        G3 = bnd->grains[1];
    }
    else if (bnd->grains[1] == bnd_AC->grains[1]) {
        bnd->grains[1] = bnd_AC->grains[0];
        changed = 1;
        G3 = bnd->grains[1];
    }
    else {
        printf("In %s: boundary %d grain G3 not found.\n", __func__, bnd->id);
        print_boundary(bnd);
        return false;
    }

    // Change with BE
    // Finding G1
    if (bnd->grains[(changed+1)%2] == bnd_BE->grains[0]) {
        bnd->grains[(changed+1)%2] = bnd_BE->grains[1];
    }
    else if (bnd->grains[(changed+1)%2] == bnd_BE->grains[1]) {
        bnd->grains[(changed+1)%2] = bnd_BE->grains[0];
    }
    else{
        printf("In %s: boundary %d, grain G1 not found.\n", __func__, bnd->id);
        print_boundary(bnd);
        return false;
    }

    // Readjust the boundary order of the vertices to get them CLOCKWISE
    vertex_invert_boundary_order(A);
    vertex_invert_boundary_order(B);

    // Defining new locations of vertices A and B.
    coord xAB[2] = {A->pos.x, B->pos.x};
    coord yAB[2] = {A->pos.y, B->pos.y};
    
    // Adjusting the vertices such that the origin is at vertex A
    adjust_origin_for_points(xAB, yAB, 2);
    vector2 Ap, Bp, Pp;
    Ap.x = xAB[0];
    Ap.y = yAB[0];
    Bp.x = xAB[1];
    Bp.y = yAB[1];
    
    // Compute just a normal midpoint since xAB and yAB are not in the periodic domain
    Pp.x = 0.5*(Ap.x + Bp.x);
    Pp.y = 0.5*(Ap.y + Bp.y);
    
    vector2 V1;
    V1.x = Ap.x - Pp.x;
    V1.y = Ap.y - Pp.y;
    
    vector2 V2;
    V2.x = Bp.x - Pp.x;
    V2.y = Bp.y - Pp.y;
    
    // At this point we could scale these vectors
    V1 = vector2_rotate90(V1);
    V2 = vector2_rotate90(V2);
    
    // After flipping, boundary is smaller
    V1.x *= 0.5;
    V1.y *= 0.5;
    V2.x *= 0.5;
    V2.y *= 0.5;

    V1.x += (Pp.x + A->pos.x);
    V1.y += (Pp.y + A->pos.y);
    V2.x += (Pp.x + A->pos.x);
    V2.y += (Pp.y + A->pos.y);

    A->pos=vector2_adjust(V1);
    B->pos=vector2_adjust(V2);

    // assert(approach.x != 0 || approach.y != 0);
    // Set the velocity to 0 because it was already used.
    vector2 nullv; 
    nullv.x = 0;
    nullv.y = 0;
    A->vel = nullv;
    B->vel = nullv;

    // Reparametrize neighbors of boundary
    for (int i =0; i < 3; i++) {
        if(A->boundaries[i] != bnd)
            A->boundaries[i]->reparam = true;

        if (B->boundaries[i] != bnd)
            B->boundaries[i]->reparam = true;
    }

    // Recalculate inner points of frontier.
    boundary_set_equidistant_inners(bnd);
    return true;
}


/* Detects if, in the case that on any side of a frontier there is a duplicated frontier,
if the frontier is candidate to eliminate and replace the sequence double frontiers.
This may only happen if the frontier is the one with biggest pointer on all the sequence of double frontiers.
After that, this function can be called with the destroy flag to delete all the double frontier sequence.
*/
__device__ inline bool boundary_delete_double_boundaries(boundary *bnd, bool destroy) {
    boundary *front1;
    boundary *front2;
    vertex *current_junct;
    boundary *current_front;
    int i;
    bool candidate = true;
    bool would_destroy = false;

    // Check that this frontier isn't part of a double frontier
    vertex *neighs[3];
    vertex_neighbors(bnd->ini,neighs);
    for (int k = 0; k < 3; k++) {
        if (bnd->ini->boundaries[k] != bnd && neighs[k] == bnd->end)
            candidate = false;
    }

    vertex_neighbors(bnd->end,neighs);
    for (int k = 0; k < 3; k++) {
        if (bnd->end->boundaries[k] != bnd && neighs[k] == bnd->ini)
            candidate = false;
    }

    // Check for duplicated neighbors.
    for (int t = 0; t < 2; t++) {
        if (!candidate) break;

        current_junct = bnd->ini;
        if (t == 1) current_junct = bnd->end;
        current_front = bnd;
        
        // Advance on the duplicated sequence
        while (true) {
            // Get the frontiers on the ending side:
            i = 0;
            while (true) {
                if (current_junct->boundaries[i] == current_front) break;
                i++;
            }

            front1 = current_junct->boundaries[(i+1)%3];
            front2 = current_junct->boundaries[(i+2)%3];
            
            // Check if they are the duplicated:
            if (boundary_is_duplicate(front1,front2)) {
                would_destroy= true;

                // Advance the junction pointer
                if (front1->ini == current_junct)
                    current_junct = front1->end;
                else
                    current_junct = front1->ini;

                // Advance the frontier pointer
                i = 0;
                while (true) {
                    if (current_junct->boundaries[i] != front1 && current_junct->boundaries[i] != front2) break;
                    i++;
                }
                current_front = current_junct->boundaries[i];
                
                // Advance the junction pointer again.
                vertex *cpoint = current_junct;
                if (current_front->ini == current_junct)
                    current_junct = current_front->end;
                else
                    current_junct = current_front->ini;

                // | If the frontier is autorized to destroy, terminate with the 3 frontiers and 2 junctions.
                if (destroy) {
                    // Connect the current_junct with junct.
                    i = 0;
                    while (true) {
                        if (current_junct->boundaries[i] == current_front) break;
                        i++;
                    }
                    current_junct->boundaries[i] = bnd;
                    
                    // Connect front with the current_junct and save the starting possition of the double frontier
                    vector2 ex_double_ini_pos;
                    if (t == 0) {
                        ex_double_ini_pos.x = bnd->ini->pos.x;
                        ex_double_ini_pos.y = bnd->ini->pos.y;
                        bnd->ini = current_junct;
                    }
                    else {
                        ex_double_ini_pos.x = bnd->end->pos.x;
                        ex_double_ini_pos.y = bnd->end->pos.y;
                        bnd->end = current_junct;
                    }

                    // Disable the junctions
                    front1->ini->enabled = false;
                    front1->end->enabled = false;
                    
                    // Disable the frontiers
                    front1->enabled = false;
                    front2->enabled = false;
                    current_front->enabled = false;
                    
                    // Try to match the curvature of the opossing frontier and the double frontier.
                    vector2 doublefront_len = vector2_delta_to(cpoint->pos,ex_double_ini_pos);
                    coord arclenI = bnd->arclen;
                    coord arclenA = vector2_mag(doublefront_len);
                    coord arclenB = current_front->arclen;
                    coord totalArclen = arclenI+arclenA+arclenB;
                    bool opp_normal_order = (current_front->ini==current_junct) ^ (t==1);
                    
                    for (int k = 0; k < INNER_POINTS; k++) {
                        coord carrclen = totalArclen*(k+1.0)/(INNER_POINTS+1.0);
                        if (carrclen <= arclenI) {
                            // Take some points of the same frontier
                            int on_same_k = 0;
                            if (arclenI != 0.0)
                                on_same_k = (int)floor((INNER_POINTS)*(carrclen)/arclenI);
                            if (on_same_k >= INNER_POINTS)
                                on_same_k = INNER_POINTS-1;
                            
                            bnd->inners[k].x = current_front->inners[on_same_k].x;
                            bnd->inners[k].y = current_front->inners[on_same_k].y;
                        }
                        else if (carrclen <= arclenI + arclenA) {
                            // Assume that the double frontier is straight.
                            vector2 point = vector2_adjust(vector2_sum(
                                ex_double_ini_pos,vector2_float_portion(
                                    doublefront_len,(carrclen-arclenI)/arclenA)
                            ));
                            bnd->inners[k].x = point.x;
                            bnd->inners[k].y = point.y;
                        }
                        else {
                            // Take some of the points on the opossing front.
                            int on_opposing_k = (int)floor((INNER_POINTS)*(carrclen-arclenA-arclenI)/arclenB);
                            if (on_opposing_k >= INNER_POINTS)
                                on_opposing_k = INNER_POINTS-1;
                            
                            if (opp_normal_order) {
                                bnd->inners[k].x = current_front->inners[on_opposing_k].x;
                                bnd->inners[k].y = current_front->inners[on_opposing_k].y;
                            }
                            else {
                                bnd->inners[k].x = current_front->inners[INNER_POINTS-on_opposing_k-1].x;
                                bnd->inners[k].y = current_front->inners[INNER_POINTS-on_opposing_k-1].y;
                            }
                        }
                    }
                    bnd->arclen = totalArclen;
                    
                    // This may set the inner points equalspaced
                    boundary_set_equidistant_inners(bnd);
                    
                    // Set front to current front so the next step works
                    current_front = bnd;
                }
                else {
                    // Check if this other junction has bigger pointer.
                    if (bnd <= current_front) { // It's a curious case when it meets itself.
                        candidate = false;
                        break;
                    }
                }
            }
            else {
                break;
            }
        }
    }
    return (candidate && would_destroy);
}

/* For a given junction, checks if 2 of the 3 adjacent frontiers will flip.
   If inhibit is true, then this call will inhibit all the frontiers except
   the one with higher pointer.
*/
__device__ inline bool vertex_are_flips_around_consistent(
        vertex *vrt, bool inhibit) {
    
    bool consistent = true;
    boundary *to_flip = NULL;

    for (int i = 0; i < 3; i++) {
        // Check if the frontier between is going to flip.
        if (vrt->boundaries[i]->to_flip) {
            // If it is, register it.
            if (to_flip == NULL) {
                to_flip = vrt->boundaries[i];
            }
            else if (vrt->boundaries[i] != to_flip) {
                // ^ If there's already one registered and it's not this one...
                consistent = false; // It's not consistent.
                
                // to_flip must hold the higher pointer.
                if (vrt->boundaries[i] > to_flip) {
                    if (inhibit)
                        boundary_inhibit(to_flip);
                    to_flip = vrt->boundaries[i];
                }
                else {
                    if(inhibit)
                        boundary_inhibit(vrt->boundaries[i]);
                }
            }
        }
    }
    return consistent;
}

/* Called for frontiers of length 0, it pushes a little both endings, using the directions of the opossing junctions.
*/
__device__ inline void boundary_expand_little_boundary(boundary *bnd) {
    int i;

    for (int t = 0; t < 2; t++) {
        boundary *front1;
        boundary *front2;
        
        // Get the frontiers at this side.
        vertex *end = bnd->ini;
        if (t==1)
            end = bnd->end;
        i = 0;
        while (true) {
            if (end->boundaries[i] == bnd) break;
            i++;
        }
        front1 = end->boundaries[(i+1)%3];
        front2 = end->boundaries[(i+2)%3];
        
        // Find the vector:
        vector2 opposite1;
        vector2 opposite2;
        
        if (front1->ini == end)
            opposite1 = front1->end->pos;
        else
            opposite1 = front1->ini->pos;
        
        if (front2->ini == end)
            opposite2 = front2->end->pos;
        else
            opposite2 = front2->ini->pos;
        
        vector2 delta1 = vector2_delta_to(opposite1, end->pos);
        vector2 delta2 = vector2_delta_to(opposite2, end->pos);
        vector2 dir = vector2_unitary(vector2_prom(delta1,delta2));
        end->pos = vector2_adjust(vector2_sum(vector2_float_portion(dir,SMALL_EXPAND),end->pos));
    }
}

#endif
