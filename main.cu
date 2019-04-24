#include "utils.h"
#include "input.h"
#include "geometry.h"
#include "calculus.h"
#include "textdisplay.h"
#include "output.h"
#include "options.h"

#include "translation.h"
#include "movement.h"

#include "macros.h"

#ifndef VDEBUG
#define CHECK_VALIDITY(X)
#else
#define CHECK_VALIDITY(X) device_to_host(vertices, n_vertices, boundaries, n_boundaries, dev_vertices, dev_boundaries); \
    valid = check_valid(vertices, n_vertices, boundaries,n_boundaries, X); \
    if(!valid){ printf("INVALID!\n"); exit(0);} \
    host_to_device(vertices, n_vertices, boundaries, n_boundaries, dev_vertices, dev_boundaries);
#endif

#if DEBUG_FILE == 1
#define PRINT_TO_DEBUG_FILE(...) fprintf(debug_file,__VA_ARGS__);
#elif DEBUG_FILE == 2
#define PRINT_TO_DEBUG_FILE(...) fprintf(debug_file,__VA_ARGS__);\
fclose(debug_file);\
if(!(debug_file= fopen("debug.txt","a"))){\
    printf("Error opening debug file.\n");\
    exit(0);\
}
#else
#define PRINT_TO_DEBUG_FILE(...)
#endif


int main(int argc, char const *argv[])
{
    options opt;
    coord delta_t, t, delta_tau;
    int n_vertices, n_boundaries, init_n_grains, curr_n_grains;
    
    // Host vertices, boundaries and grain arrays
    vertex *vertices;
    boundary *boundaries;
    gdata *graindata;
    
    // Device vertices, boundaries and grain arrays
    boundary *dev_boundaries;
    vertex *dev_vertices;
    gdata *dev_graindata;

    // Buffer for counting unstable and reparam boundaries
    int *buffer_cnt_unstable, *dev_buffer_cnt_unstable;
    buffer_cnt_unstable = (int*) calloc(N_BLKS, sizeof(int));
    HERR(cudaMalloc(&dev_buffer_cnt_unstable, sizeof(int) * N_BLKS));
    
    // Buffer for counting grains
    int *buffer_cnt_grains, *dev_buffer_cnt_grains;
    buffer_cnt_grains = (int*) calloc(N_BLKS, sizeof(int));
    HERR(cudaMalloc(&dev_buffer_cnt_grains, sizeof(int) * N_BLKS));

    // Parse input and check if we are loading saved data or initial condition
    int input_state = parse_input(argc, argv, &opt);
    
    // Sate of loading
    bool state;
    if (input_state == OPT_LOAD_SAVED_DATA) {
        printf("Loading from saved file\n");
        state = load_result_file(&opt, &init_n_grains, &n_vertices, &n_boundaries,
                vertices, boundaries, graindata, &t);
        if (!state) {
            perror("Error trying to parse input files!");
            exit(0);
        }

        if (!check_valid(vertices, n_vertices, boundaries, n_boundaries, true)) {
            perror("Inconsistent input files!\n");
            exit(0);
        }
        printf("Starting at t = %.16f\n", t);
    }
    else if (input_state == OPT_LOAD_INITIAL_CONDITION) {
        printf("Loading %s\n", opt.ridge_file);
        printf("Loading %s\n", opt.vertex_file);
        printf("Loading %s\n", opt.orientation_file);
        t = 0.0;
    }
    delta_t = opt.delta_t;

    printf("dt:         %.16f\n", delta_t);
    printf("Mu:         %16f\n", opt.mu);
    printf("Lambda:     %16f\n", opt.lambda);
    printf("GB epsilon: %.16f\n", opt.eps);
    printf("Saving results to %s of removed grains\n", opt.output_basename);
    
    if (opt.save_each_percent > 0)
        printf("Saving each %f%\n", (opt.save_each_percent * 100));
    else if(opt.save_each_steps > 0)
        printf("Saving each %d iterations\n", opt.save_each_steps);
    
    if(opt.min_timestep > 0 && opt.max_timestep > 0)
        printf("Saving in timestep range [%d, %d]\n", opt.min_timestep, opt.max_timestep);
    printf("Reparametrizing each %d iterations\n", opt.reparam_each_steps);

    if (opt.solver == RK2){ 
        printf("Using Multistep Second Order Runge-Kutta with size %d.\n", opt.inner_res);
        perror("RK2 in opt.solver no longer supported");
        exit(0);

    }
    else if (opt.solver == MULTISTEP)
        printf("Using Multistep Euler with size %d.\n", opt.inner_res);
    delta_tau = delta_t / (double) opt.inner_res;

    printf("delta tau:\t%.16f\n", delta_tau);

    if (opt.victim_bnd != -1)
        printf("Boundary %d is forced to flip now.\n", opt.victim_bnd);
    int steps = 0;

    #ifdef VDEBUG
    bool valid;
    #endif

    #if DEBUG_FILE > 0
    // Open debug file.
    FILE *debug_file;
    if (!(debug_file = fopen("debug.txt","w"))) {
        printf("Error opening debug file.\n");
        exit(0);
    }
    #endif

    PRINT_TO_DEBUG_FILE("PRECOMPUTING CONSTANTS\n");
    precompute();

    if (input_state == OPT_LOAD_INITIAL_CONDITION) {
        // Load initial condition. Vertices positions, boundaries relations
        // and orientations for grains are loaded. The number of grains is estimated
        // from the loaded vertices and boundaries. This number must match the number
        // of loaded orientations.
        PRINT_TO_DEBUG_FILE("READING INPUT FILES\n");
        printf("Loading files...\n");
        state = load_from_files(opt.vertex_file, opt.ridge_file, opt.orientation_file,
                                         vertices, n_vertices, boundaries,n_boundaries, graindata, curr_n_grains);
        init_n_grains = curr_n_grains;
        printf("Files loaded.\n");
        if (!state) {
            perror("Error trying to parse input files!");
            exit(0);
        }
        if (!check_valid(vertices, n_vertices, boundaries, n_boundaries, true)) {
            perror("Inconsistent input files!\n");
            exit(0);
        }

        // Compute the right grain indices for the boundaries.
        // This indices will point to a global grain data device structure
        obtain_grains(vertices, n_vertices, NULL, NULL, true);
    }


    // Partial output for RK2
    double *dev_vrtX, *dev_bndX;
    cudaMalloc(&dev_vrtX, sizeof(double) * n_vertices * 2);
    cudaMalloc(&dev_bndX, sizeof(double) * n_boundaries * INNER_POINTS * 2);

    PRINT_TO_DEBUG_FILE("STARTING\n");
    host_to_device(vertices, n_vertices, boundaries, n_boundaries, dev_vertices, dev_boundaries);
    host_to_device_graindata(graindata, init_n_grains, dev_graindata);
    
    // MEMORY FOR BOOLEAN RESULTS
    bool boolarr[N_BLKS];
    bool *dev_boolarr;
    HERR(cudaMalloc(&dev_boolarr, sizeof(bool) * N_BLKS));
    CNULL(dev_boolarr);

    PRINT_TO_DEBUG_FILE("n_vertices:%d n_boundaries:%d\n", n_vertices, n_boundaries);

    PRINT_TO_DEBUG_FILE("\tEXPAND SMALL BOUNDARIES\n");
    expand_small_boundaries<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries);
    CERR(); CHECK_VALIDITY(true);

    // Grain counter update
    curr_n_grains = count_grains(dev_boundaries, n_boundaries, dev_buffer_cnt_grains, buffer_cnt_grains);
    printf("Loaded %d grains\n", curr_n_grains);
    if (input_state == OPT_LOAD_INITIAL_CONDITION)
        init_n_grains = curr_n_grains;
    
    double max_n_grains = (1 - opt.max_grains_percent) * curr_n_grains;
    double remaining;
    if (opt.save_each_steps > 0)
        remaining = 0;
    else
        remaining = curr_n_grains;

    // Buffers for polling system
    bool buffer_polling[N_BLKS];
    bool *candidate_buffer, *dev_buffer_polling;
    cudaMalloc(&candidate_buffer, sizeof(bool) * n_boundaries);
    HERR(cudaMalloc(&dev_buffer_polling, sizeof(int) * N_BLKS));

    /********** Prepare initial condition **********/

    /*************************************/
    /* BEGIN: UPDATE STATE VARIABLES
    (FLIP_STATE, ARCLENGTH, GRAIN_BOUNDARY_ENERGY,
    NORMAL, TANGENTIAL AND TOTAL VELOCITY,
    EXTINCTION TIMES) */
    /*************************************/
    update_state_variables(dev_vertices,n_vertices, dev_boundaries,
                            n_boundaries, dev_graindata, opt);
    /*************************************/
    /* END: UPDATE STATE VARIABLES */
    /*************************************/
    PRINT_TO_DEBUG_FILE("\tRESET STATE\n");
    reset_boundaries_flip_state<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries);
    CERR(); CHECK_VALIDITY(true);

    /*************************************/
    /* BEGIN: TOPOLOGICAL CHANGES */
    /*************************************/
    detect_and_apply_topological_changes(dev_vertices,n_vertices, dev_boundaries, n_boundaries,
                                         dev_graindata, boolarr, dev_boolarr, candidate_buffer,
                                         buffer_polling, dev_buffer_polling, &opt, delta_t);
    /*************************************/
    /* END: TOPOLOGICAL CHANGES */
    /*************************************/

    /*************************************/
    /* BEGIN: UPDATE STATE VARIABLES
    (FLIP_STATE, ARCLENGTH, GRAIN_BOUNDARY_ENERGY,
    NORMAL, TANGENTIAL AND TOTAL VELOCITY,
    EXTINCTION TIMES) */
    /*************************************/
    update_state_variables(dev_vertices,n_vertices, dev_boundaries,
                            n_boundaries, dev_graindata, opt);
    /*************************************/
    /* END: UPDATE STATE VARIABLES */
    /*************************************/


    /********************  MAIN LOOP  ********************/
    double ratio_to_increase_dt = 0.0;
    while (t <= MAX_TIME || steps <= MAX_STEPS) {
        // Increase dt as we remove grains
        if ( ((double)curr_n_grains / (double)init_n_grains) <= ratio_to_increase_dt) {
            delta_t *= 2.0;
            delta_tau *= 2.0;
            ratio_to_increase_dt -= 0.1;
            printf("Duplicating dt at %f removed grains", 100*(1 - ratio_to_increase_dt));
        }

        PRINT_TO_DEBUG_FILE("STARTING %.10lf + %.10lf\n",(double)t,(double)delta_t);
        
        // Check the end of loop
        if ((MAX_STEPS)>0 && steps>(MAX_STEPS)) {
            PRINT_TO_DEBUG_FILE("EXPIRED\n");
            break;
        }


        /******************** INNER LOOP ********************/
        for (int istep = 0; istep < opt.inner_res; istep++) {
            // Detect flippings
            PRINT_TO_DEBUG_FILE("\tDETECT FLIPPINGS\n");
            compute_extinction_times<<<N_BLKS, N_TRDS>>>(dev_boundaries,n_boundaries, opt.lambda);
            detect_flips<<<N_BLKS, N_TRDS>>>(dev_boundaries,n_boundaries, 2.0*delta_t-istep*delta_tau);
            CERR(); CHECK_VALIDITY(true);

            // Evolution
            PRINT_TO_DEBUG_FILE("\tVERTICES\n");
            evolve_vertices<<<N_BLKS, N_TRDS>>>(dev_vertices, n_vertices, delta_tau);
            CERR(); CHECK_VALIDITY(true);
            
            // Correct the evolution of boundaries given the current state of vertices
            correct_inner_velocities<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries, delta_tau);
            CERR(); CHECK_VALIDITY(true);
            PRINT_TO_DEBUG_FILE("\tBOUNDARIES\n");
            evolve_boundaries<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries, delta_tau);
            CERR(); CHECK_VALIDITY(true);
            
            t = t + delta_tau;

            update_state_variables(dev_vertices,n_vertices, dev_boundaries,
                            n_boundaries, dev_graindata, opt);
        }

        /******************** END INNER LOOP ********************/
        // now we are in time t + delta_t
        detect_and_apply_topological_changes(dev_vertices,n_vertices,  dev_boundaries, n_boundaries,
                                             dev_graindata, boolarr, dev_boolarr, candidate_buffer,
                                             buffer_polling, dev_buffer_polling, &opt, delta_t);
        update_state_variables(dev_vertices,n_vertices, dev_boundaries,
                                n_boundaries, dev_graindata, opt);

        // Reparametrize
        if (opt.reparam_each_steps > 0 && (steps % opt.reparam_each_steps) == 0) {
            PRINT_TO_DEBUG_FILE("\tREPARAMETRIZE BOUNDARIES\n");
            reparametrize<<<N_BLKS,N_TRDS>>>(dev_boundaries,n_boundaries);
            CERR(); CHECK_VALIDITY(true);
        }

        // Check stability - Hard test
        PRINT_TO_DEBUG_FILE("\tCHECK STABILITY\n");
        set_stability<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries);
        CERR(); CHECK_VALIDITY(true);
        
        // Show unstable grains prior to reparametrization
        int n_unstables = count_unstable_and_reparam_boundaries(dev_boundaries, n_boundaries,
            dev_buffer_cnt_unstable, buffer_cnt_unstable, steps, t);
        if (n_unstables > 0)
            reparametrize_unstable_and_after_flip<<<N_BLKS,N_TRDS>>>(dev_boundaries, n_boundaries);
        CERR(); CHECK_VALIDITY(true);

        /*************************************/
        /* BEGIN: UPDATE STATE VARIABLES
        (FLIP_STATE, ARCLENGTH, GRAIN_BOUNDARY_ENERGY,
        NORMAL, TANGENTIAL AND TOTAL VELOCITY,
        EXTINCTION TIMES) */
        /*************************************/
        update_state_variables(dev_vertices,n_vertices, dev_boundaries,
                                n_boundaries, dev_graindata, opt);

        // Check for intersection of boundaries
        cudaDeviceSynchronize();
        check_intersections<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries, steps);
        cudaDeviceSynchronize();
        set_checked_false<<<N_BLKS, N_TRDS>>>(dev_boundaries, n_boundaries, steps);
        
        /*************************************/
        /* BEGIN: UPDATE STATE VARIABLES */
        /*************************************/

        save_structure(vertices, dev_vertices, n_vertices, boundaries,
                       dev_boundaries, n_boundaries, graindata, init_n_grains,
                       curr_n_grains, remaining, t, delta_t, opt, steps, opt.inner_res+2);
        
        // Update number of grains
        if (curr_n_grains <= max_n_grains)
            break;
        curr_n_grains = count_grains(dev_boundaries, n_boundaries, dev_buffer_cnt_grains, buffer_cnt_grains);
        
        // Advance to next step
        steps++;
    }

    PRINT_TO_DEBUG_FILE("ENDING\n");
    
    // Free all the memory on the device.
    HERR(cudaFree(dev_vertices));
    HERR(cudaFree(dev_boundaries));
    HERR(cudaFree(dev_boolarr));
    HERR(cudaFree(dev_vrtX));
    HERR(cudaFree(dev_bndX));
    HERR(cudaFree(dev_buffer_cnt_grains));
    HERR(cudaFree(dev_buffer_cnt_unstable));
    HERR(cudaFree(dev_buffer_polling));
    HERR(cudaFree(candidate_buffer));
    free(vertices);
    free(boundaries);
    free(graindata);
    free(buffer_cnt_unstable);
    free(buffer_cnt_grains);
    #if DEBUG_FILE > 0
    // Close debug file.
    fclose(debug_file);
    #endif

    printf("Simulation done.\n");
    return 0;
}
