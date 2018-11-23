#ifndef SRC_TRANSLATION_H
#define SRC_TRANSLATION_H

#include "geometry.h"
#include "utils.h"

/**
 * Adds a direction delta to all the pointer on a given place of the memory.
 */
__global__ void adjust_pointers(
        vertex *dev_juncts_start, vertex *juncts_start, int jlen,
        boundary *dev_fronts_start, boundary *fronts_start, int flen,
        bool rollback) {

    int stid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid;

    // Adjust pointers of vertices
    tid = stid;
    while (tid < jlen) {
        if (rollback) {
            for (int i = 0; i < 3; i++) {
                int fpos = dev_juncts_start[tid].boundaries[i]-dev_fronts_start;
                dev_juncts_start[tid].boundaries[i] = &fronts_start[fpos];
            }
        } 
        else {
            for (int i = 0; i < 3; i++) {
                int fpos = dev_juncts_start[tid].boundaries[i]-fronts_start;
                dev_juncts_start[tid].boundaries[i] = &dev_fronts_start[fpos];
            }
        }
        tid += gridDim.x * blockDim.x;
    }

    // Adjust pointers of boundaries
    tid = stid;
    while (tid < flen) {
        if (rollback) {
            int jpos_ini, jpos_end;
            jpos_ini = dev_fronts_start[tid].ini-dev_juncts_start;
            dev_fronts_start[tid].ini = &juncts_start[jpos_ini];
            jpos_end = dev_fronts_start[tid].end-dev_juncts_start;
            dev_fronts_start[tid].end = &juncts_start[jpos_end];
        }
        else {
            int jpos_ini, jpos_end;
            jpos_ini = dev_fronts_start[tid].ini-juncts_start;
            dev_fronts_start[tid].ini = &dev_juncts_start[jpos_ini];
            jpos_end = dev_fronts_start[tid].end-juncts_start;
            dev_fronts_start[tid].end = &dev_juncts_start[jpos_end];
        }
        tid += gridDim.x * blockDim.x;
    }
}

/**
 * Translates the memory of the vertices and boundaries to the device, also adjusts the pointer
 * so that they point to the struct in the device.
 */
inline void host_to_device( vertex *juncts, int jlen, boundary *fronts, int flen, vertex* &dev_juncts, boundary* &dev_fronts) {
    // Allocate memory on device:
    HERR(cudaMalloc(&dev_juncts, sizeof(vertex)*jlen));
    CNULL(dev_juncts);
    HERR(cudaMalloc(&dev_fronts, sizeof(boundary)*flen));
    CNULL(dev_fronts);

    // Copy memory:
    HERR(cudaMemcpy(dev_juncts,juncts, sizeof(vertex)*jlen, cudaMemcpyHostToDevice));
    HERR(cudaMemcpy(dev_fronts,fronts, sizeof(boundary)*flen, cudaMemcpyHostToDevice));
    
    // Adjust the pointers on device:
    adjust_pointers<<<N_BLKS,N_TRDS>>>(dev_juncts, juncts, jlen,
                                       dev_fronts, fronts, flen,
                                       false);
}

/**
 * Translate memory of grain data that is loaded at the beginning and the end
 */
inline void host_to_device_graindata(gdata *graindata, int glen, gdata *&dev_graindata) {
    HERR(cudaMalloc(&dev_graindata, sizeof(gdata)*glen));
    CNULL(dev_graindata);
    HERR(cudaMemcpy(dev_graindata, graindata, sizeof(gdata)*glen, cudaMemcpyHostToDevice));
}

/**
 * Sends the data back to the host, assuming that it has already reserved the memory in juncts and
 * fronts, frees the data on the device.
 */
inline void device_to_host(vertex *juncts, int jlen, boundary *fronts, int flen, vertex* dev_juncts, boundary* dev_fronts) {
    // Adjust the pointers for the Host on the device:
    adjust_pointers<<<N_BLKS,N_TRDS>>>(dev_juncts,juncts,jlen,dev_fronts,fronts,flen,true);
    
    // Copy memory:
    HERR(cudaMemcpy(juncts,dev_juncts, sizeof(vertex)*jlen, cudaMemcpyDeviceToHost));
    HERR(cudaMemcpy(fronts,dev_fronts, sizeof(boundary)*flen, cudaMemcpyDeviceToHost));
    
    // Free memory on device:
    HERR(cudaFree(dev_juncts));
    HERR(cudaFree(dev_fronts));
}

inline void device_to_host_graindata(gdata *graindata, int glen, gdata *dev_graindata) {
    HERR(cudaMemcpy(graindata, dev_graindata, sizeof(gdata)*glen, cudaMemcpyDeviceToHost));
    HERR(cudaFree(dev_graindata));
}

/* Reduces the structure on the host, deleting disabled boundarys and vertices */
inline void reduct_structure(vertex **juncts, int *jlen, boundary **fronts, int *flen) {
    // Alloc another space of memory to save the new structure
    vertex *juncts2 = (vertex *) malloc(sizeof(vertex)*(*jlen));
    boundary *fronts2 = (boundary *) malloc(sizeof(boundary)*(*flen));
    
    // Create transformations from current indexes to real indexes and start filling the new space
    int *jtransf = (int *) malloc(sizeof(int)*(*jlen));
    int *ftransf = (int *) malloc(sizeof(int)*(*flen));
    
    // Only save the good vertexs
    int goodj = 0;
    for (int k = 0; k < *jlen; k++) {
        vertex *jun = &(*juncts)[k];
        if (jun->enabled) {
            jtransf[k] = goodj;
            memcpy(&juncts2[goodj],&(*juncts)[k],sizeof(vertex));
            goodj+=1;
        }
        else{
            jtransf[k] = -1;
        }
    }

    // Only save the good boundarys
    int goodf = 0;
    for (int k = 0; k < *flen; k++) {
        boundary *fro = &(*fronts)[k];
        if (fro->enabled) {
            ftransf[k] = goodf;
            memcpy(&fronts2[goodf],&(*fronts)[k],sizeof(boundary));
            goodf+=1;
        }
        else{
            ftransf[k] = -1;
        }
    }

    // Update the pointers on the new structure
    for (int k=0; k < goodj; k++) {
        vertex *jun = &juncts2[k];
        
        if (ftransf[jun->boundaries[0]-*fronts]==-1 || ftransf[jun->boundaries[1]-*fronts]==-1 || ftransf[jun->boundaries[2]-*fronts]==-1)
            printf("DETECTEDERROR\n");
        
        jun->boundaries[0] = &fronts2[ftransf[jun->boundaries[0]-*fronts]];
        jun->boundaries[1] = &fronts2[ftransf[jun->boundaries[1]-*fronts]];
        jun->boundaries[2] = &fronts2[ftransf[jun->boundaries[2]-*fronts]];
    }
    for (int k = 0; k < goodf; k++) {
        boundary *fro = &fronts2[k];
        
        if (jtransf[fro->ini-*juncts]==-1 || jtransf[fro->end-*juncts]==-1)
            printf("ERRORDEE\n");
        
        fro->ini = &juncts2[jtransf[fro->ini-*juncts]];
        fro->end = &juncts2[jtransf[fro->end-*juncts]];
    }

    // Free the original memory and replace it.
    *jlen = goodj;
    *flen = goodf;
    free(*juncts);
    free(*fronts);
    free(jtransf);
    free(ftransf);
    *juncts = juncts2;
    *fronts = fronts2;
}

#endif
