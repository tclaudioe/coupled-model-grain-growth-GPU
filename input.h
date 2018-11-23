#ifndef SRC_INPUT_H
#define SRC_INPUT_H

#include <stdio.h>

#include "geometry.h"
#include "options.h"

#define PARSE_VERTICES 0
#define PARSE_BOUNDARIES 1
#define PARSE_GRAINS 2

/**
 *  Load initial condition from vertex and boundary files
 */
inline bool load_from_files(const char *jfname, const char *ffname, const char *oriname,
                            vertex* &vertices, int &vlen,
                            boundary * &boundaries, int &flen,
                            gdata * &graindata, int &glen) {
    vlen = 0; flen = 0; glen = 0;

    vector2 vec;

    FILE *junct_fil = fopen(jfname, "r");
    
    if (junct_fil == NULL) return false;
    int jsize = 1;
    vertices = (vertex*) malloc(jsize*sizeof(vertex));
    
    #if DOUBLE_PRECISION == 0
    while (fscanf(junct_fil,"%f %f\n",&(vec.x),&(vec.y)) != EOF) {
    #elif DOUBLE_PRECISION == 1
    while (fscanf(junct_fil,"%lf %lf\n",&(vec.x),&(vec.y)) != EOF) {
    #endif
        if (vlen == jsize) {
            jsize *= 2;
            vertices = (vertex*) realloc(vertices, jsize*sizeof(vertex));
        }
        init_vertex(&vertices[vlen], vec, vlen);
        vertices[vlen].id = vlen;
        vlen+=1;
    }
    fclose(junct_fil);
    printf("%d vertices loaded.\n", vlen);
    int a,b;

    FILE *front_fil = fopen(ffname, "r");
    if (front_fil == NULL) {
        printf("In %s: Could not open %s\n", __func__, ffname);
        free(vertices);
        return false;
    }

    int fsize = 1;
    int *frnts_ini = (int*) malloc(fsize*sizeof(int));
    int *frnts_end = (int*) malloc(fsize*sizeof(int));

    while (fscanf(front_fil,"%d %d\n",&a,&b) != EOF) {
        if (flen == fsize) {
            fsize *= 2;
            frnts_ini = (int*) realloc(frnts_ini, fsize*sizeof(int));
            frnts_end = (int*) realloc(frnts_end, fsize*sizeof(int));
        }
        frnts_ini[flen] = a;
        frnts_end[flen] = b;
        flen+=1;
    }

    boundaries = (boundary*) malloc(fsize*sizeof(boundary));
    for (int i = 0; i < flen; i++)
        init_boundary(&boundaries[i],&vertices[frnts_ini[i]],&vertices[frnts_end[i]], i);
    free(frnts_ini);
    free(frnts_end);

    fclose(front_fil);
    printf("%d boundaries loaded.\n", flen);

    // Set all the junctions's fronts clockwise.
    for (int j = 0; j < vlen; j++)
        vertex_set_boundaries_clockwise(&vertices[j]);

    // load orientations
    FILE *ori_fil = fopen(oriname, "r");
    if (ori_fil == NULL) {
        free(vertices);
        free(boundaries);
        return false;
    }

    double ori;
    int gsize = 1;
    graindata = (gdata*) malloc(gsize*sizeof(gdata));
    
    while (fscanf(ori_fil,"%lf\n", &ori) != EOF) {
        if (glen == gsize) {
            gsize *= 2;
            graindata = (gdata*) realloc(graindata, gsize*sizeof(gdata));
        }
        graindata[glen].orientation = 2.0 * M_PI * ori;
        glen++;
    }
    printf("%d grains loaded.\n", glen);
    return true;
}

bool load_result_file(options *opt, int *n_grains, int *n_vertices, int *n_boundaries,
                      vertex* &vertices, boundary* &boundaries, gdata * &graindata, coord *t) {
    FILE *resultfile = fopen(opt->input_file, "r");
    
    if (resultfile == NULL) return false;
    
    char *pch;
    char *line;
    char *tmp;
    char buffer[MAX_BUFFER_SIZE];
    char var[MAX_BUFFER_SIZE];
    int now_parsing = -1;
    int i = 0;
    int inner_points;
    
    // Parse DELTA T
    while (fgets(buffer, MAX_BUFFER_SIZE, resultfile)) {
        pch = strchr(buffer, ' ');
        
        if (pch != NULL) {
            strncpy(var, buffer,  pch - buffer);
            var[pch-buffer] = '\0';
            
            if (!strcmp(var, "DELTA_T")) {
                opt->delta_t = atof(pch+1);
            } 
            else if (!strcmp(var, "MOMENT")) {
                *t = atof(pch+1);
            } 
            else if (!strcmp(var, "LAMBDA")) {
                opt->lambda = atof(pch+1);
            } 
            else if (!strcmp(var, "MU")) {
                opt->mu = atof(pch+1);
            } 
            else if (!strcmp(var, "GB EPSILON")) {
                opt->eps = atof(pch+1);
            } 
            else if (!strcmp(var, "INIT_GRAINS")) {
                *n_grains = atoi(pch+1);
            } 
            else if (!strcmp(var, "INIT_VERTICES")) {
                *n_vertices = atoi(pch+1);
            } 
            else if (!strcmp(var, "INIT_BOUNDARIES")) {
                *n_boundaries = atoi(pch+1);
            } 
            else if (!strcmp(var, "INNER_POINTS")) {
                inner_points = atoi(pch+1);
                if (inner_points != INNER_POINTS) {
                    printf("In %s: Run this file with INNER_POINTS = %d.\n", __func__, INNER_POINTS);
                    printf("Don't forget to change the macro before compile!\n");
                    return false;
                }
            } 
            else if (!strcmp(var, "QUAD_ORDER")) {
                int qord = atoi(pch+1);
                if (qord != QUAD_ORDER) {
                    printf("In %s: Run this file with QUAD_ORDER = %d\n", __func__, QUAD_ORDER);
                    printf("Don't forget to change the macro before compile!\n");
                    return false;
                }
            } 
            else if(!strcmp(var, "#VERTICES")) {
                vertices = (vertex*) malloc((*n_vertices) * sizeof(vertex));
                
                for (int k = 0; k < *n_vertices; k++)
                    vertices[k].enabled = false;
                
                boundaries = (boundary*) malloc((*n_boundaries) * sizeof(boundary));
                
                for (int k = 0; k < *n_boundaries; k++)
                    boundaries[k].enabled = false;
                
                now_parsing = PARSE_VERTICES;
                continue;
            } 
            else if (!strcmp(var, "#BOUNDARIES")) {
                now_parsing = PARSE_BOUNDARIES;
                graindata = (gdata*) malloc((*n_grains) * sizeof(gdata));
                continue;
            } 
            else if(!strcmp(var, "#GRAINS")) {
                now_parsing = PARSE_GRAINS;
                continue;
            }
        }
        
        // Parse vertices
        if (now_parsing == PARSE_VERTICES) {
            line = strdup(buffer);
            tmp = strsep(&line, " ");
            
            // Get the id of the vertex, which is the position on the global array
            int id = atoi(tmp);
            vertices[id].id = id;
            tmp = strsep(&line, " ");
            vertices[id].pos.x = atof(tmp);
            tmp = strsep(&line, " ");
            vertices[id].pos.y = atof(tmp);
            tmp = strsep(&line, " ");
            vertices[id].vel.x = atof(tmp);
            tmp = strsep(&line, " ");
            vertices[id].vel.y = atof(tmp);
            
            // Join data of boundaries
            tmp = strsep(&line, " ");
            vertices[id].boundaries[0] = &boundaries[atoi(tmp)];
            tmp = strsep(&line, " ");
            vertices[id].boundaries[1] = &boundaries[atoi(tmp)];
            tmp = strsep(&line, " ");
            vertices[id].boundaries[2] = &boundaries[atoi(tmp)];
            
            // Enable current vertex
            vertices[id].enabled = true;
            i++;
            if (i-1 == (*n_vertices))
                i = 0;
        }

        // Parse boundaries
        if (now_parsing == PARSE_BOUNDARIES) {
            line = strdup(buffer);
            tmp = strsep(&line, " ");
            
            int id = atoi(tmp);
            boundaries[id].id = id;
            tmp = strsep(&line, " ");
            
            // Skip line suntil we found grains
            for (int k = 0; k < 15; k++) { strsep(&line, " "); }
            
            tmp = strsep(&line, " ");
            boundaries[id].grains[0] = atoi(tmp);
            tmp = strsep(&line, " ");
            boundaries[id].grains[1] = atoi(tmp);
            tmp = strsep(&line, " ");
            graindata[boundaries[id].grains[0]].orientation = atof(tmp);
            tmp = strsep(&line, " ");
            graindata[boundaries[id].grains[1]].orientation = atof(tmp);
            
            // Get vertices
            tmp = strsep(&line, " ");
            boundaries[id].ini = &vertices[atoi(tmp)];
            tmp = strsep(&line, " ");
            boundaries[id].end = &vertices[atoi(tmp)];
            
            // Parse inner points positions
            for (int k = 0; k < inner_points; k++) {
                tmp = strsep(&line, " ");
                boundaries[id].inners[k].x = atof(tmp);
                tmp = strsep(&line, " ");
                boundaries[id].inners[k].y = atof(tmp);
            }

            // Skip lines until we found inner points
            boundaries[id].enabled = true;
            i++;
            if (i-1 == (*n_boundaries)) i = 0;
        }
    }
    fclose(resultfile);
    return true;
}

#endif
