#ifndef SRC_UTILS_H
#define SRC_UTILS_H

#include <stdio.h>
#include "geometry.h"

// Default number of blocks.
#define N_BLKS 32

// Default number of threads. Must be a power of 2.
#define N_TRDS 32

inline void HandleError(cudaError_t err, const char *file, int line) {
    // As from the book, "CUDA by example".
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HERR( err ) (HandleError( err, __FILE__, __LINE__ ))

inline void CheckError(const char *file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CERR(err) (CheckError(__FILE__, __LINE__ ))

inline void CheckNull(void *ptr, const char *file, int line) {
    if (ptr == NULL) {
        printf("NULL pointer in %s at line %d\n", file, line);
        exit(EXIT_FAILURE);
    }
}

#define CNULL(ptr) (CheckNull((void *)(ptr),__FILE__, __LINE__ ))

__device__ __host__ void print_vertex(vertex *vrt);
__device__ __host__ void print_boundary(boundary *bnd);

__device__ __host__ void print_boundary(boundary *bnd) {
    printf("(bnd %d) Boundary Info\n",bnd->id);
    printf("(bnd %d) Vertices: [%d %d]\n",bnd->id, bnd->ini->id, bnd->end->id);
    printf("(bnd %d) Grains: [%d %d]\n",bnd->id, bnd->grains[0], bnd->grains[1]);
    printf("(bnd %d) Number of votes: %d\n",bnd->id, bnd->n_votes);
    printf("(bnd %d) To flip: %d\n",bnd->id, bnd->to_flip);
    printf("(bnd %d) inhibited: %d\n",bnd->id, bnd->inhibited_flip);
    print_vertex(bnd->ini);
    print_vertex(bnd->end);
}

__device__ __host__ void print_vertex(vertex *vrt) {
    int id[3];

    printf("(vrt %d) Vertex information\n", vrt->id, vrt);
    printf("(vrt %d) Neighbor boundaries: [%d %d %d]\n", vrt->id,
        vrt->boundaries[0]->id, vrt->boundaries[1]->id, vrt->boundaries[2]->id);

    for(int i = 0; i < 3; i++) {
        if (vrt->boundaries[i]->ini == vrt)
            id[i] = vrt->boundaries[i]->end->id;
        else
            id[i] = vrt->boundaries[i]->ini->id;
    }

    printf("(vrt %d) Neighbor vertices: [%d %d %d]\n", vrt->id, id[0], id[1], id[2]);
}

inline bool check_valid(vertex *vertices, int jlen, boundary *boundaries, int flen, bool check_duplicates) {
    bool isvalid = true;

    for (int f = 0; f < flen; f++) {
        boundary *bnd = &boundaries[f];
        
        if(!bnd->enabled) continue;

        if (bnd->ini == bnd->end) {
            printf("BOUNDARY STARTS AND ENDS ON SAME JUNCTION\n");
            print_boundary(bnd);
            isvalid= false;
        }

        if (!bnd->ini->enabled || !bnd->end->enabled) {
            printf("BOUNDARY HAS DISABLED JUNCTION\n");
            isvalid= false;
        }

        if((
                bnd->ini->boundaries[0]!=bnd &&
                bnd->ini->boundaries[1]!=bnd &&
                bnd->ini->boundaries[2]!=bnd ) || (
                bnd->end->boundaries[0]!=bnd &&
                bnd->end->boundaries[1]!=bnd &&
                bnd->end->boundaries[2]!=bnd )) {
            printf("VERTEX DOESN'T REFERENCE HIS BOUNDARY\n");
            isvalid= false;
        }

        vector2 dist = vector2_delta_to(bnd->end->pos, bnd->ini->pos);

        if (dist.x == 0 && dist.y == 0)
            printf("WARNING: BOUNDARY HAS LENGTH 0!\n");
    }

    for (int j = 0; j < jlen; j++) {
        vertex *vrt = &vertices[j];

        if(!vrt->enabled) continue;
        
        if(
                (vrt->boundaries[0] == vrt->boundaries[1]) ||
                (vrt->boundaries[1] == vrt->boundaries[2]) ||
                (vrt->boundaries[2] == vrt->boundaries[0]) ) {
            printf("VERTEX HAS REPEATED BOUNDARIES!\n");
            isvalid= false;
        }

        if (vrt->pos.x < 0.0 || vrt->pos.x > 1.0 || vrt->pos.y < 0.0 || vrt->pos.y > 1.0) {
            printf("VERTEX OUTSIDE DOMAIN!\n");
            isvalid= false;
        }

        for (int i = 0; i < 3; i++) {
            if (!vrt->boundaries[i]->enabled) {
                printf("VERTEX HAS DISABLED BOUNDARY\n");
                isvalid= false;
            }
            if (vrt->boundaries[i]->ini != vrt && vrt->boundaries[i]->end != vrt) {
                printf("BOUNDARY DOESN'T REFERENCE ITS VERTEX\n");
                isvalid= false;
            }
        }
    }

    if (check_duplicates) {
        for (int j = 0; j < jlen; j++) {
            vertex *vrt = &vertices[j];

            if (!vrt->enabled) continue;
            
            for (int i = 0; i < 3; i++) {
                boundary *bnd1 = vrt->boundaries[i];
                boundary *bnd2 = vrt->boundaries[(i+1)%3];
                if(
                        (bnd1->ini == bnd2->ini && bnd1->end == bnd2->end) ||
                        (bnd1->ini == bnd2->end && bnd1->end == bnd2->ini) ) {
                    printf("DUPLICATED BOUNDARY!\n");
                    print_vertex(vrt);
                    isvalid= false;
                }
            }
        }
    }
    return isvalid;
}

__global__ void debug_boundary(boundary *dev_boundaries, int n_boundaries, int bnd_id, int istep, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0) {
        boundary *bnd = &dev_boundaries[bnd_id];
        if(bnd->enabled) {
            printf("BEGIN debug boundary %d at inner step %d of step %d\n",
                bnd_id, istep, step);
            printf("arclength: %.16f\n", bnd->arclen);
            printf("dL/dt: %.16f\n", bnd->dLdt);
            printf("extinction time: %.16f\n", bnd->t_ext);
            printf("to flip: %d\n", bnd->to_flip);
            printf("inhibited: %d\n", bnd->inhibited_flip);

            printf("pos ini (%.16f, %.16f)\n", bnd->ini->pos.x, bnd->ini->pos.y);
            for (int i = 0; i < INNER_POINTS; i++)
                printf("pos %03d (%.16f, %.16f)\n", i, bnd->inners[i].x, bnd->inners[i].y);
            printf("pos end (%.16f, %.16f)\n", bnd->end->pos.x, bnd->end->pos.y);

            printf("vel ini (%.16f, %.16f)\n", bnd->ini->vel.x, bnd->ini->vel.y);
            for (int i = 0; i < INNER_POINTS; i++) {
                printf("vel Tangen %03d (%.16f, %.16f)\n", i, bnd->tangent_vels[i].x, bnd->tangent_vels[i].y);
                printf("vel Normal %03d (%.16f, %.16f)\n", i, bnd->normal_vels[i].x, bnd->normal_vels[i].y);
                printf("vel %03d (%.16f, %.16f)\n", i, bnd->vels[i].x, bnd->vels[i].y);
            }
            printf("vel end (%.16f, %.16f)\n", bnd->end->vel.x, bnd->end->vel.y);

            printf("END debug boundary %d\n", bnd_id);
        } 
        else {
            printf("You are looking for %d, which is a disabled boundary :(\n", bnd_id);
        }
    }
}

#endif
