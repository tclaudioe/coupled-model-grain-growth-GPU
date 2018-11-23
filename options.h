#ifndef OPTIONS_H
#define OPTIONS_H

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_BUFFER_SIZE 100000

#define OPT_LOAD_INITIAL_CONDITION 1
#define OPT_LOAD_SAVED_DATA 2

enum solver_t { MULTISTEP, RK2 };

struct options {
    // Mobilities
    double lambda;
    double mu;
    double delta_t;
    
    // The solver, MULTISTEP or Runge-Kutta 2
    solver_t solver;
    
    // Multiscale parameter for inner iteration.
    // Using the value 1 implies classic resolution
    int inner_res;
    
    // Reparametrize each x steps
    int reparam_each_steps;
    
    // Grain boundary energy epsilon
    double eps;

    // Save each x percent of grains removed
    double save_each_percent;
    
    // Max number of grains to be removed (in percentage) (use negative to avoid)
    double max_grains_percent;
    
    // Save each x steps
    int save_each_steps;
    
    // Save between range of timesteps given by {min,max}_timestep
    int min_timestep;
    int max_timestep;
    
    // Input and output data
    char vertex_file[MAX_BUFFER_SIZE];
    char ridge_file[MAX_BUFFER_SIZE];
    char orientation_file[MAX_BUFFER_SIZE];
    char output_basename[MAX_BUFFER_SIZE];
    
    // Input saved data
    char input_file[MAX_BUFFER_SIZE];
    
    // Output snapshot or not
    bool save_image;
    
    // id of boundary to force the flip
    int victim_bnd;

    // Number to avoid reaching zero in energy function
    double delta_energy;
};


/**
 * Parse input
 * @param  argc Number of arguments
 * @param  argv
 * @param  opt  [description]
 * @return      [description]
 */
int parse_input(int argc, const char *argv[], options *opt) {
    int ret = OPT_LOAD_INITIAL_CONDITION;
    opt->inner_res = 1;
    opt->eps = 0.0;
    opt->save_image = false;
    opt->save_each_percent = 0;
    opt->save_each_steps = 0;
    opt->reparam_each_steps = 5;
    opt->min_timestep = -1;
    opt->max_timestep = -1;
    opt->victim_bnd = -1;
    opt->delta_energy = 0.1;

    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-de") && i+1 < argc)
            opt->delta_energy = atof(argv[i+1]);

        if (!strcmp(argv[i], "-dt") && i+1 < argc)
            opt->delta_t = atof(argv[i+1]);
        
        if (!strcmp(argv[i], "-output") && i+1 < argc) {
            char order[2048] = "rm -rf ";

            strcpy(opt->output_basename, argv[i+1]);
            strcat(order, opt->output_basename);
            
            printf("%s\n", order);
            system(order);
            
            mkdir(opt->output_basename, 0775);
            strcat(opt->output_basename, "/result_");
        }

        if (!strcmp(argv[i], "-vertex-file") && i+1 < argc)
            strcpy(opt->vertex_file, argv[i+1]);
        
        if (!strcmp(argv[i], "-ridge-file") && i+1 < argc)
            strcpy(opt->ridge_file, argv[i+1]);
        
        if (!strcmp(argv[i], "-orientation-file") && i+1 < argc)
            strcpy(opt->orientation_file, argv[i+1]);
        
        if (!strcmp(argv[i], "-save-each-percent") && i+1 < argc)
            opt->save_each_percent = atof(argv[i+1]);
        
        if (!strcmp(argv[i], "-max-grains-percent") && i+1 < argc) 
            opt->max_grains_percent = atof(argv[i+1]);
        
        if (!strcmp(argv[i], "-save-each-steps") && i+1 < argc) 
            opt->save_each_steps = atoi(argv[i+1]);
        
        if (!strcmp(argv[i], "-min-timestep") && i+1 < argc)
            opt->min_timestep = atoi(argv[i+1]);
       
        if (!strcmp(argv[i], "-max-timestep") && i+1 < argc) 
            opt->max_timestep = atoi(argv[i+1]);
        
        if (!strcmp(argv[i], "-reparam-each-steps") && i+1 < argc) 
            opt->reparam_each_steps = atoi(argv[i+1]);
        
        if (!strcmp(argv[i], "-solver") && i+1 < argc) {
            if (!strcmp(argv[i+1], "MULTISTEP"))
                opt->solver = MULTISTEP; 
            else if(!strcmp(argv[i+1], "RK2"))
                opt->solver = RK2;
        }

        if (!strcmp(argv[i], "-inner-resolution") && i+1 < argc)
            opt->inner_res = atof(argv[i+1]);
        
        if (!strcmp(argv[i], "-lambda") && i+1 < argc)
            opt->lambda = atof(argv[i+1]);
        
        if (!strcmp(argv[i], "-mu") && i+1 < argc)
            opt->mu = atof(argv[i+1]);
        
        if (!strcmp(argv[i], "-save-image"))
            opt->save_image = true;
        
        // Special option if loading file
        if (!strcmp(argv[i], "-load")) {
            // Get the input file
            strcpy(opt->input_file, argv[i+1]);
            ret = OPT_LOAD_SAVED_DATA;
        }

        if (!strcmp(argv[i], "-force-flip")) {
            // Get the input file
            opt->victim_bnd = atoi(argv[i+1]);
        }
        
        if (!strcmp(argv[i], "-gb-eps"))
            opt->eps = atof(argv[i+1]);
    }
    return ret;
}


#endif // OPTIONS_H