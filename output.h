#ifndef SRC_OUTPUT_H
#define SRC_OUTPUT_H

#include "macros.h"
#include "geometry.h"
#include "grains.h"
#include "options.h"
#include "translation.h"

/* Saves the grain structure to a file, it must be REDUCTED just before this function is called.
*/
inline void output(vertex *vertices, int n_vertices, boundary *boundaries, int n_boundaries,
                    gdata *graindata, int init_n_grains, coord time, coord delta_t, const char *fname, options opt) {
    FILE *fp;
    if(fp= fopen(fname,"w")) {
        // Get the grains
        grain *grains=NULL; int lengrains=0;
        obtain_grains(vertices, n_vertices, &grains, &lengrains, false);

        // Print DATA header
        fprintf(fp, "#DATA\n");
        fprintf(fp, "MOMENT %.16lf\n", (double)time);
        fprintf(fp, "DELTA_T %.16lf\n", (double)delta_t);
        fprintf(fp, "LAMBDA %.16lf\n", (double)opt.lambda);
        fprintf(fp, "MU %.16lf\n", (double)opt.mu);
        fprintf(fp, "GB EPSILON %.16f\n", (double)opt.eps);
        fprintf(fp, "INIT_GRAINS %d\n", init_n_grains);
        fprintf(fp, "INIT_VERTICES %d\n", n_vertices);
        fprintf(fp, "INIT_BOUNDARIES %d\n", n_boundaries);
        fprintf(fp, "VERTICES %d\n", lengrains*2);
        fprintf(fp, "BOUNDARIES %d\n", lengrains*3);
        fprintf(fp, "GRAINS %d\n", lengrains);

        fprintf(fp, "INNER_POINTS %d\n", INNER_POINTS);
        fprintf(fp, "QUAD_ORDER %d\n", QUAD_ORDER);
        fprintf(fp, "DOUBLE_PRECISION %d\n", DOUBLE_PRECISION);
        fprintf(fp, "INV_ARCLEN_REPARAM %d\n", INVERSE_ARCLEN_FUNCTION_REPARAMETRIZATION);
        fprintf(fp, "BISECTION_ITERS %d\n", BISECTION_ITERS);

        // Print VERTICES
        fprintf(fp, "#VERTICES x y velx vely bndId bndId bndId\n");
        for (int k = 0; k < n_vertices; k++){
            vertex *vrt = &vertices[k];
            if (vrt->enabled) {
                fprintf(fp, "%ld", vrt->id);
                fprintf(fp, " %.16lf %.16lf", (double)vrt->pos.x, (double)vrt->pos.y);
                fprintf(fp, " %.16lf %.16lf", (double)vrt->vel.x , (double)vrt->vel.y);
                fprintf(fp, " %ld", vrt->boundaries[0]->id);
                fprintf(fp, " %ld", vrt->boundaries[1]->id);
                fprintf(fp, " %ld", vrt->boundaries[2]->id);
                fprintf(fp, "\n");
            }
        }

        // Print BOUNDARIES
        fprintf(fp, "#BOUNDARIES energy prev_arclen arclen dLdt t_ext t_ext_vert t_ext_curv prev_curvature curvature tmp_curvature G1 G2 ori1 ori2 vrtId vrtId");

        for(int k = 0; k < INNER_POINTS; k++)
            fprintf(fp, " inner%dX inner%dY", k+1, k+1);
        
        for(int k = 0; k < INNER_POINTS; k++)
            fprintf(fp, " innerV%dX innerV%dY", k+1, k+1);
        
        fprintf(fp, " inivel_X inivel_Y endvel_X endvel_Y");
        fprintf(fp, "\n");
        for(int k = 0; k < n_boundaries; k++) {
            boundary *bnd = &boundaries[k];
            if (bnd->enabled) {
                fprintf(fp, "%d", bnd->id);
                fprintf(fp, " %.16lf", (double)bnd->energy);
                fprintf(fp, " %.16lf", (double)bnd->prev_arclen);
                fprintf(fp, " %.16lf", (double)bnd->arclen);
                fprintf(fp, " %.16lf", (double) bnd->dLdt);
                fprintf(fp, " %.16lf", (double) bnd->t_ext);
                fprintf(fp, " %.16lf", (double) bnd->t_ext_vert);
                fprintf(fp, " %.16lf", (double) bnd->t_ext_curv);
                fprintf(fp, " %.16lf", (double) bnd->prev_curvature);
                fprintf(fp, " %.16lf", (double) bnd->curvature);
                fprintf(fp, " %.16lf", (double) bnd->tmp_curvature);
                
                // Grain info for debugging
                fprintf(fp, " %d", (int)bnd->grains[0]);
                fprintf(fp, " %d", (int)bnd->grains[1]);
                fprintf(fp, " %.16lf", (double)graindata[bnd->grains[0]].orientation);
                fprintf(fp, " %.16lf", (double)graindata[bnd->grains[1]].orientation);
                fprintf(fp, " %ld %ld", bnd->ini->id, bnd->end->id);

                for(int p = 0; p < INNER_POINTS; p++)
                    fprintf(fp, " %.16lf %.16lf", (double)bnd->inners[p].x, (double)bnd->inners[p].y);
                
                for(int p = 0; p < INNER_POINTS; p++)
                    fprintf(fp, " %.16lf %.16lf", (double)bnd->vels[p].x, (double)bnd->vels[p].y);
                
                
                // Save inivel and endvel
                fprintf(fp, " %.16lf %.16lf %.16lf %.16lf", (double)bnd->inivel.x, (double)bnd->inivel.y, (double)bnd->endvel.x, (double)bnd->endvel.y);

                fprintf(fp,"\n");
            }
        }

        // Print GRAINS
        int negative_areas = 0;
        int too_negative_areas = 0;
        fprintf(fp, "#GRAINS area dAdt {vrtId bndId}\n");
        for(int k = 0; k < lengrains; k++) {
            grain *gra = &grains[k];
            
            // Compute grain area and dA/dt
            double g_area = grain_area(gra, k, vertices);
            double g_dAdt = grain_dAdt(gra, k);
            
            // Handle negative areas
            if(g_area < 0 && g_area < NEGATIVE_AREA_TOLERANCE)
                too_negative_areas++;
            
            // Count negative areas that are within tolerance,
            // This will be the new warning that the output will throw
            if(g_area < 0 &&  g_area >= NEGATIVE_AREA_TOLERANCE)
                negative_areas++;

            fprintf(fp, " %.16lf", g_area);
            fprintf(fp, " %.16lf", g_dAdt);
            for(int p = 0; p < gra->vlen; p++) {
                vertex *vrt = gra->vrtxs[p];
                vertex *vrt_next = gra->vrtxs[(p+1)%gra->vlen];
                boundary*bnd;
                int q = 0;
                while (true) {
                    bnd = vrt->boundaries[q];
                    if(bnd->ini == vrt_next || bnd->end == vrt_next) break;
                    q++;
                }
                fprintf(fp, " %ld %ld",vrt->id, bnd->id);
            }
            fprintf(fp, "\n");
        }
        
        // ----
        fclose(fp);
        delete_grains(grains,lengrains);
        if (negative_areas > 0) {
            printf("WARNING: %d GRAINS OF NEGATIVE AREA AT %.16lf, TIME DELTA TOO BIG?\n",
                negative_areas, (double)time);
        }
        if (too_negative_areas > 0) {
            printf("WARNING: %d GRAINS OF EXCESIVE NEGATIVE AREA AT %.16lf, TIME DELTA TOO BIG?\n",
                too_negative_areas, (double)time);
        }
    }
    else {
        printf("Error saving output!\n");
        exit(0);
    }
}


void save_structure(vertex* vertices, vertex* dev_vertices, int n_vertices,
                    boundary* boundaries, boundary* dev_boundaries, int n_boundaries,
                    gdata* graindata, int init_n_grains, int curr_n_grains, double &remaining,
                    coord t, coord delta_t, options opt, int steps, int istep) {
    if ((opt.save_each_percent > 0 && curr_n_grains <= remaining) ||
        ((opt.save_each_steps > 0) && (steps % opt.save_each_steps == 0)) ||
        (opt.min_timestep <= steps && opt.max_timestep >= steps))
    {
        char snap_filename[2048];
        sprintf(snap_filename, "%s_%d_%d.txt", opt.output_basename, steps, istep);
        
        device_to_host(vertices, n_vertices, boundaries, n_boundaries, dev_vertices, dev_boundaries);
        output(vertices, n_vertices, boundaries, n_boundaries, graindata,
               init_n_grains, t, delta_t, snap_filename, opt);
        host_to_device(vertices, n_vertices, boundaries, n_boundaries, dev_vertices, dev_boundaries);
        remaining -= (opt.save_each_percent * init_n_grains);
    }
}

#endif // OUTPUT_H
