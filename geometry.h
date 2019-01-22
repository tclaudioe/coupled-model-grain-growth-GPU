#ifndef SRC_GEOMETRY_H
#define SRC_GEOMETRY_H

#include "macros.h"

#define DOMAIN_BOUND 1.0

#if DOUBLE_PRECISION == 0
typedef float coord;
#elif DOUBLE_PRECISION == 1
typedef double coord;
#endif

__device__ __host__ inline coord dom_mod(coord a) {
    // Adjust a given coordinate if it is outside the domain.
    // Assuming it doesn't wrap arround more that one time.
    if (a < 0) return a+DOMAIN_BOUND;
    else if (a >= DOMAIN_BOUND) return a-DOMAIN_BOUND;
    return a;
}

__device__ __host__ inline coord wrap_dist(coord dest, coord orig) {
    // Gives the minor distance from a coordinate to another on the wrapping domain.
    coord delta = dest - orig;
    if(delta > +(DOMAIN_BOUND/2)) return (delta - DOMAIN_BOUND);
    if(delta < -(DOMAIN_BOUND/2)) return (DOMAIN_BOUND + delta);
    return delta;
}

struct vector2 {
    coord x;
    coord y;
};

__device__ __host__ inline vector2 vector2_delta_to(const vector2 dest, const vector2 orig) {
    vector2 res;
    res.x = wrap_dist(dest.x,orig.x);
    res.y = wrap_dist(dest.y,orig.y);
    return res;
}

__device__ __host__ inline vector2 vector2_sum(const vector2 a, const vector2 b) {
    vector2 res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
}

__device__ __host__ inline coord vector2_mag2(const vector2 a) {
    coord p = max(abs(a.x), abs(a.y));
    coord q = min(abs(a.x), abs(a.y));
    return (p*p) * (1.0 + (q/p)*(q/p));
}

__device__ __host__ inline coord vector2_mag(const vector2 a) {
    coord p = max(abs(a.x), abs(a.y));
    coord q = min(abs(a.x), abs(a.y));
    return p * sqrt(1.0 + (q/p)*(q/p));
}

__device__ __host__ inline coord vector2_dot(const vector2 a, const vector2 b) {
    return (a.x*b.x+a.y*b.y);
}

__device__ __host__ inline vector2 vector2_portion(const vector2 v, int num, int den) {
    vector2 res;
    res.x = (num*v.x)/den;
    res.y = (num*v.y)/den;
    return res;
}

__device__ __host__ inline vector2 vector2_float_portion(const vector2 v, coord prt) {
    vector2 res;
    if (prt < 0.0) prt=0.0;
    if (prt > 1.0) prt=1.0;
    res.x = (coord)(prt*v.x);
    res.y = (coord)(prt*v.y);
    return res;
}

__device__ __host__ inline vector2 vector2_unitary(const vector2 v) {
    vector2 ret;
    double angle = atan2(v.y,v.x);
    ret.x = cos(angle);
    ret.y = sin(angle);
    return ret;
}

__device__ __host__ inline vector2 vector2_adjust(const vector2 v) {
    vector2 aux;
    aux.x = dom_mod(v.x);
    aux.y = dom_mod(v.y);
    return aux;
}

__device__ __host__ inline vector2 vector2_rotate90(const vector2 v) {
    vector2 aux;
    aux.x = -v.y;
    aux.y = v.x;
    return aux;
}

__device__ __host__ inline vector2 vector2_rotate180(const vector2 v) {
    vector2 aux;
    aux.x = -v.x;
    aux.y = -v.y;
    return aux;
}

__device__ __host__ inline vector2 vector2_rotate270(const vector2 v) {
    vector2 aux;
    aux.x = v.y;
    aux.y = -v.x;
    return aux;
}

__device__ __host__ inline vector2 vector2_prom(const vector2 v1, const vector2 v2) {
    vector2 v1_cpy = v1;
    vector2 delta = vector2_delta_to(v2,v1);
    v1_cpy.x += delta.x/2.0;
    v1_cpy.y += delta.y/2.0;
    return vector2_adjust(v1_cpy);
}

struct vertex;
struct boundary {
    int id;
    vector2 inners[INNER_POINTS];
    vector2 vels[INNER_POINTS];

    // Normal and tangential velocity
    vector2 tangent_vels[INNER_POINTS];
    vector2 normal_vels[INNER_POINTS];

    vector2 inivel;
    vector2 endvel;
    vertex *ini;
    vertex *end;
    bool enabled;
    bool to_flip; // Is the vertex about to flip?
    bool inhibited_flip; // Is the vertex flip inhibited?
    bool autorized_to_delete; // Can the vertex delete all the sequence of double vertices arround.
    double energy; // Grain boundary energy as function of misorientation
    double arclen; // Current arclen
    double prev_arclen; // Arclen at previous iteration
    double t_ext; // Extinction time used in simulation
    double t_ext_vert; // Extinction time used in vertex model
    double t_ext_curv; // Extinction time used in curvature model
    double t_ext_to_flip; // Extinction time used to activate the flipping, it is updated if a lower value is found.
    double dLdt; // Rate of change of arclen
    double prev_curvature; // Mean curvature at previous iteration
    double curvature; // Mean curvature
    double tmp_curvature; // Temporal curvature after flippings
    bool stable; // Is the boundary stable or unstable
    bool reparam;
    int n_votes;
    int t_steps_flip_applied; // Counts how many time steps ago a flipping was applied,
    
    // Grain data:
    int grains[2];

    // FOR DEBUGGING
    vector2 Wk[INNER_POINTS+2];
    vector2 dTds[QUAD_ORDER];

    // Values of alphas
    coord alphas[INNER_POINTS];

    // DEBUG: FOR DEBUG OPTIONS
    vector2 ini_int;
    vector2 end_int;
    vector2 raw_tangent_vels[INNER_POINTS];
    vector2 raw_normal_vels[INNER_POINTS];
    vector2 raw_inivel;
    vector2 raw_endvel;

    // Check intersection flag
    bool checked[2];
};

/* Special structure which holds loadable data of grains */
struct gdata {
    double orientation;
};

struct vertex {
    vector2 pos;
    vector2 vel;
    boundary *boundaries[3];
    bool enabled;
    int id;
    boundary *voted;
};



__device__ __host__ inline void init_vertex(vertex *vrt, const vector2 pos, int id){
    vrt->id = id;
    vrt->pos = pos;
    vrt->boundaries[0] = NULL;
    vrt->boundaries[1] = NULL;
    vrt->boundaries[2] = NULL;
    vrt->vel.x = 0;
    vrt->vel.y = 0;
    vrt->enabled = true;
}

__device__ __host__ inline void vertex_neighbors(vertex *vrt, vertex **neighs){
    // Sets the 3 neighbor vertices of the given one.
    // IN THE SAME ORDER THAT boundaries.
    for (int i = 0; i < 3; i++) {
        if (vrt->boundaries[i]->ini == vrt)
            neighs[i] = vrt->boundaries[i]->end;
        else
            neighs[i] = vrt->boundaries[i]->ini;
    }
}

__device__ __host__ inline void vertex_add_boundary(vertex *vrt, boundary *bnd) {
    // Put the boundary pointer on the next possition that isn't NULL.
    if (vrt->boundaries[0] == NULL)
        vrt->boundaries[0] = bnd;
    else if (vrt->boundaries[1] == NULL)
        vrt->boundaries[1] = bnd;
    else
        vrt->boundaries[2] = bnd;
}

__device__ __host__ inline void vertex_set_boundaries_clockwise(vertex *vrt) {
    // Reorders the boundaries of a vertex on clockwise order.
    coord angle[3];

    for (int i = 0; i < 3; i++) {
        vector2 delta;
        #if INNER_POINTS == 0
            if (vrt->boundaries[i]->ini == vrt)
                delta = vector2_delta_to(vrt->boundaries[i]->end->pos, vrt->pos);
            else
                delta = vector2_delta_to(vrt->boundaries[i]->ini->pos, vrt->pos);
        #else
            if (vrt->boundaries[i]->ini == vrt)
                delta = vector2_delta_to(vrt->boundaries[i]->inners[0], vrt->pos);
            else
                delta = vector2_delta_to(vrt->boundaries[i]->inners[INNER_POINTS-1], vrt->pos);
        #endif
        angle[i] = atan2(delta.y,delta.x);
    }
    
    // Sort the boundaries
    for (int k = 0;k < 2; k++) {
        for (int p = 1;p < 3-k; p++) {
            if (angle[p-1] < angle[p]) {
                coord aux = angle[p-1];
                angle[p-1] = angle[p];
                angle[p] = aux;
                boundary *auxb = vrt->boundaries[p-1];
                vrt->boundaries[p-1] = vrt->boundaries[p];
                vrt->boundaries[p] = auxb;
            }
        }
    }
}

__device__ __host__ inline void vertex_invert_boundary_order(vertex *vrt) {
    // The order can be reversed easily, just swapping two frontiers.
    boundary *auxb = vrt->boundaries[1];
    vrt->boundaries[1] = vrt->boundaries[0];
    vrt->boundaries[0] = auxb;
}

__device__ __host__ inline void boundary_adjusted_points(boundary *bnd, coord *xpos, coord *ypos);

__device__ __host__ inline void boundary_set_equidistant_inners(boundary *bnd) {
    // Place the inner points equidistantly between the ini and end,
    // sets all the velocities to 0.
    vector2 nullv; nullv.x = 0.0; nullv.y = 0.0;
    vector2 delt = vector2_delta_to(bnd->end->pos, bnd->ini->pos);

    for (int k = 0; k < INNER_POINTS; k++) {
        vector2 place = vector2_sum(bnd->ini->pos, vector2_portion(delt,k+1,INNER_POINTS+1));
        
        // | If the point happens to be outside the domain.
        place = vector2_adjust(place);
        bnd->inners[k] = place;
        
        // | Set the velocity to 0
        bnd->vels[k] = nullv;
        bnd->tangent_vels[k] = nullv;
        bnd->normal_vels[k] = nullv;
    }
    bnd->inivel = nullv;
    bnd->endvel = nullv;
}

__device__ __host__ inline void init_boundary(boundary *bnd, vertex *ini, vertex *end, int id) {
    // Update the pointers to and from both ends.
    bnd->id = id;
    bnd->ini = ini;
    vertex_add_boundary(ini, bnd);
    bnd->end = end;
    vertex_add_boundary(end, bnd);
    boundary_set_equidistant_inners(bnd);
    bnd->to_flip = false;
    bnd->inhibited_flip = false;
    bnd->autorized_to_delete = false;
    bnd->enabled = true;
    bnd->prev_arclen = 0;
    bnd->arclen = 0;
    bnd->prev_curvature = 0;
    bnd->curvature = 0;
    bnd->reparam = false;
    bnd->t_steps_flip_applied = 0;
    
    // grains
    bnd->grains[0] = -1;
    bnd->grains[1] = -1;
}

__device__ __host__ inline void boundary_inhibit(boundary *bnd) {
    // Makes the boundary unable of flip.
    bnd->inhibited_flip = true;
    vector2 nullv;
    nullv.x=0;
    nullv.y=0;
    bnd->ini->vel = nullv;
    bnd->end->vel = nullv;
}


__device__ __host__ inline bool boundary_is_duplicate(boundary *bnd1, boundary *bnd2) {
    return (bnd1->ini == bnd2->ini && bnd1->end == bnd2->end) || (bnd1->ini == bnd2->end && bnd1->end == bnd2->ini);
}

__device__ __host__ inline void adjust_origin_for_points(coord *xpts, coord *ypts, int plen) {
    // Adjust the points on a curve to new coordinates with the origin on the first one.
    if (plen > 0) {
        vector2 prev;
        prev.x=xpts[0]; prev.y=ypts[0];
        xpts[0] = 0; ypts[0] = 0;
        for (int i = 1;i < plen; i++) {
            // Calculate the delta with the previous point
            vector2 current;
            current.x = xpts[i];
            current.y = ypts[i];
            vector2 delta = vector2_delta_to(current, prev);

            // Save the point before changing it.
            prev.x=xpts[i]; prev.y=ypts[i];
            
            // Update the current point to the new axis.
            vector2 adjusted_prev;
            adjusted_prev.x = xpts[i-1];
            adjusted_prev.y = ypts[i-1];
            current = vector2_sum(adjusted_prev, delta);
            xpts[i] = current.x; ypts[i] = current.y;
        }
    }
}

__device__ __host__ inline void boundary_adjusted_points(boundary *bnd, coord *xpos, coord *ypos) {
    // From a boundary, sets 2 arrays with the x and y coordinates of the INNER_POINTS+2
    // points that define it (vertex possitions and inner possitions) with the origin on the first one.
    // Create arrays for all the points of the boundary.
    xpos[0] = bnd->ini->pos.x;
    ypos[0] = bnd->ini->pos.y;
    for (int ii = 0; ii < INNER_POINTS; ii++) {
        xpos[ii+1]= bnd->inners[ii].x;
        ypos[ii+1]= bnd->inners[ii].y;
    }
    xpos[INNER_POINTS+1] = bnd->end->pos.x;
    ypos[INNER_POINTS+1] = bnd->end->pos.y;
    adjust_origin_for_points(xpos, ypos, INNER_POINTS+2);
}


#endif
