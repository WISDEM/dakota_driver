README.txt file for dakota_driver.

A suite of OpenMDAO drivers based on DAKOTA.
To view the Sphinx documentation for this distribution, type:

plugin docs dakota_driver

import driver using:
    from dakota_driver.driver import pydakdriver

There are four main configuration types for pydakdriver - UQ, Parameter_study, Optimization, and DOE.
==================================================================================================
==================================================================================================
==================================================================================================
    UQ
==================================================================================================
       usage: pydakdriver.UQ( UQ_type = 'sampling', sample_type = 'lhs', seed = __, samples = 100)
       description: uncertainty quantification driver configuration
       arguments:
           UQ_type = dakota uncertainty quantification procedure
              options: 
                 'sampling'
                    description: monte carlo sampling
           sample_type = random sampling approach
              options: 'lhs', 'random', incremental_lhs, incremental_random
           seed = seed for random number generator
           samples = number of samples to be taken
==================================================================================================
==================================================================================================
   Parameter_study
==================================================================================================
       usage: pydakdrive.Parameter_Study( study_type = 'vector')
       description: 
            explores the effect of parametric changes within simulation models by 
            computing re- sponse data sets at a selection of points in the parameter space
       study_type = type of parameter study
            options:
                'vector': 
                    description:
                       performs a parameter study along a line between any two points 
                       in an n-dimensional parameter space, where the user specifies the number of steps used in the study.
                    configured with 'final_point' and 'num_steps'

                'multi-dim':
                    description:
                       Forms a regular lattice or hypergrid in an n-dimensional parameter space, where the user 
                       specifies the number of intervals used for each parameter 
                    configured with: 'partitions'

                'list':
                    description:
                       The user supplies a list of points in an n-dimensional space where Dakota 
                       will evaluate response data from the simulation code.
                    configured with: 'list_of_points' 
                'centered'
                     description:
                         Given a point in an n-dimensional parameter space, this method evaluates nearby points
                         along the coordinate axes of the parameter space. The user selects the number of steps 
                         and the step size.
                     configured with: 'step_vector', 'steps_per_variable'  
       final_point
             description: 
                defines the final values for each variable on the vector to be used in the vector parameter study
                (the initial points are defined values variables are set to using openmdao API)
       num_steps:
             description: number of steps to be taken from initial point to final point 
       partitions: number of intervals for each parameter in multi-dim study
       list_of_points: List of variable values to evaluate in a list parameter study
       step_vector: Number of sampling steps along the vector in parameter study
       steps_per_variable: Number of steps to take in each dimension of a centered parameter study
==================================================================================================
==================================================================================================
   Optimization
==================================================================================================
       usage: pydakdrive.Optimization( opt_type='npsol_sqp', convergence_tolerance = '1.e-8', seed=__,
                                       max_iterations=__, max_function_evaluations=__, interval_type = 'forward')
       description: optimize one or multiple objectives
       opt_type = type of optimization
           options: 
              'npsol_sqp'
                   description: Sequential Quadratic Program
                   configured with: 'convergence_tolerance'

              'efficient_global'
                   description: Global Surrogate Based Optimization
                   configured with: 'seed'
 
              'conmin'
                   description:ONstrained function MINimization
                   configured with: 
                         'max_iterations', 'max_function_evaluations', 'convergence_tolerance'
                         'constraint_tolerance', 'interval_type'
       convergence_tolerance: Stopping criterion based on convergence of the objective function
       seed = random number generator seed
       max_iterations = Stopping criteria based on number of iterations (different than max_function_evaluations)
       interval_type = Specifies how to compute gradients and hessians
==================================================================================================
NOTES
    * __ arguments must be set by the use for propper configuration, when those arguments are 
     used to configure the desired mehtod
    * function arguments other than <analysis>_type are objects and can be also be set after configuration.
         ex.
         pydakdriver.UQ(UQ_type='sampling', seed = 11241)
         pydakdriver.samples = 1000 # (samples could have been specified in UQ function call)
