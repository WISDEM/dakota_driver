README.txt file for dakota_driver.

A suite of OpenMDAO drivers based on DAKOTA.
To view the Sphinx documentation for this distribution, type:

plugin docs dakota_driver

This can be installed using `plugin install` in an openmdao environment after
the succesful installation of pyDAKOTA

import driver using:
    from dakota_driver.driver import pydakdriver

example usage:
    dakDriver = pydakdriver()
    dakDriver.UQ()
    dakDriver.samples = 50
    dakDriver.seed = 4723
    driver = self.add('driver',dakDriver)

==================================================================================================
There are three main configuration types for pydakdriver - UQ, Parameter_study, and Optimization.
==================================================================================================
==================================================================================================
    UQ  ( Uncertainty Quantification )
==================================================================================================
       usage: pydakdriver.UQ( UQ_type = 'sampling', sample_type = 'lhs', seed = __, samples = 100)
       description: uncertainty quantification driver configuration
       arguments:
           UQ_type = dakota uncertainty quantification procedure
              options: 
                 'sampling'
                    description: monte carlo sampling
       Option Descriptions
       ------------------
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

       Option Descriptions
       ------------------
       final_point
             description: 
                list which defines the final values for each variable on the vector to be used in the vector parameter study
                (the initial points are defined values variables are set to using openmdao API)
       num_steps:
             description: number of steps to be taken from initial point to final point 
       partitions: list describin number of intervals for each parameter in multi-dim study
       list_of_points: List of variable values to evaluate in a list parameter study
                       Note: this is a "list of lists". ex: driver.list_of_points = [[2,3,4,5],[4,3,2,1]]
       step_vector: Number of sampling steps along the vector in parameter study
       steps_per_variable: Number of steps to take in each dimension of a centered parameter study
==================================================================================================
==================================================================================================
   Optimization
==================================================================================================
       usage: pydakdrive.Optimization( opt_type='optpp_newton', interval_type = 'forward')
       description: optimize one or multiple objectives
       opt_type = type of optimization
           defaults:
              convergence_tolerance = 1.e-8, max_iterations = 200, max_function_evaluations = 2000
           options: 
              'optpp_newton'
                   description: Newton method based optimization
                   Notes:
                        Expects analytical gradients in provideJ. Numerical gradiants can be set using
                        driver.numerical_gradients().
                        Computes Numerical Hessians.

              'efficient_global'
                   description: DAKOTA Global Optimization
                   configured with: seed
 
              'conmin'
                   description:ONstrained function MINimization
                   configured with: 
                         constraint_tolerance = 1.e-8

       Option Descriptions
       -------------------
       convergence_tolerance: Stopping criterion based on convergence of the objective function
       seed = random number generator seed
       max_iterations = Stopping criteria based on number of iterations (different than max_function_evaluations)
       constraint_tolerance: maximum allowable value of constraint violation still considered to be feasible

       Configurations Functions
       ----------------------- 
       interval_type:
             Specifies how to compute gradients and hessians 
             options are 'forward', 'central'
==================================================================================================
Notes for Future Development
----------------------------
- DAKOTA's Design of Experiment analysis could easily be incorporated into
  this driver
- A new class of driver for facilitating optimization under uncertainty and
  other nested DAKOTA approaches could be implemented.
- There is currently no support for analytica hessian evaluations.
   as fns is the keyword for function evaluations and fnGrads is the keyword
   for gradient evaluations, fnHessians is the keyword for hessian evaluations.
   Gradient evaluations have only been tested for single gradient evaluations,
   and may potentially fail under multiple gradient functions.
   These functionalities are implemented in the "dakota_callback" function in driver.py
