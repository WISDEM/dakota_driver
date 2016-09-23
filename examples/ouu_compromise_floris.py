from __future__ import print_function
from dakota_driver.driver import pydakdriver
from openmdao.api import Problem, ScipyOptimizer, ExecComp
#from openmdao.api import Problem, pyOptSparseDriver, ScipyOptimizer, ExecComp
from florisse.OptimizationGroups import OptAEP
from florisse import config
#from mpi4py import MPI

import time
import numpy as np
import pylab as plt

import cProfile


import sys

if __name__ == "__main__":

    config.floris_single_component = True

    ######################### for MPI functionality #########################
    from openmdao.core.mpi_wrap import MPI

    if MPI: # pragma: no cover
        # if you called this script with 'mpirun', then use the petsc data passing
        from openmdao.core.petsc_impl import PetscImpl as impl

    else:
        # if you didn't use 'mpirun', then use the numpy data passing
        from openmdao.api import BasicImpl as impl

    def mpi_print(prob, *args):
        """ helper function to only print on rank 0 """
        if prob.root.comm.rank == 0:
            print(*args)

    prob = Problem(impl=impl)

    size = 20 # number of processors (and number of wind directions to run)

    #########################################################################
    # define turbine size
    rotor_diameter = 126.4  # (m)

    opting = int(sys.argv[-1])

    # define turbine locations in global reference frame
    # original example case
    # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
    # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m

    # Scaling grid case
    nRows = int(sys.argv[1])     # number of rows and columns in grid
    spacing = 5     # turbine grid spacing in diameters

    ke = 0.065
    # Set up position arrays
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)
    if not opting:
        turbineX = np.array([7.9301380511e+02, 1.4057425864e+03, 1.8000000000e+03 , 6.5676596960e+02, 1.2994534268e+03, 1.6638190796e+03, 5.2426479858e+02, 1.1792879733e+03, 1.5385211821e+03])
        turbineY = np.array([5.2848848996e+02, 6.6248982811e+02, 7.9956939180e+02, 1.1713457607e+03, 1.2893950320e+03, 1.4030517066e+03, 1.5428096962e+03, 1.6656117632e+03, 1.8000000000e+03])
    
   # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    ratedPower = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)
    minSpacing = 2.                         # number of rotor diameters

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter      # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.
        ratedPower[turbI] = 5000.0  # rated power of each turbine in kW

    # Define flow properties
    wind_speed = 8        # m/s
    air_density = 1.1716    # kg/m^3
    windDirections = np.linspace(0, 270, size)
    windSpeeds = np.ones(size)*wind_speed
    windFrequencies = 0.2*np.ones(size)/(size-2)
    windFrequencies[0] = 0.4
    windFrequencies[size/3] = 0.4
    # initialize problem
    prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size,
                                          minSpacing=minSpacing, differentiable=True, use_rotor_components=False))

    # set up optimizer
    #prob.driver = ScipyOptimizer()
    prob.driver = pydakdriver(name='dakota.driver')

    prob.driver.stdout = 'dakota.out'
    if not opting:
     prob.driver.ortho = 10
     #prob.driver.dakota_hotstart = True
     prob.driver.UQ()#'fsu_quasi_mc')
     prob.driver.sample_type = 'lhs'
     prob.driver.samples=20
    else:
     prob.driver.n_sub_samples = 800#500 # 200 is tooo low
     prob.driver.n_sur_samples = 10
     prob.driver.max_iterations = 4#0#0

     # With compromise programming the mean at std_dev are added together. This addition
     # can be controlled using meanMult and stdMult. They are multiplied by their corresponding responses
     # so the mean objective response is scaled by meanMult and the std_dev of the objective is multiplied 
     # by stdMult
     prob.driver.population_size = 20#0
     prob.driver.meanMult = 1. 
     #prob.driver.dakota_hotstart = True
     prob.driver.stdMult = 1. 
     prob.driver.fd_gradient_step_size = 1e-2
     prob.driver.Optimization(opt_type='moga', ouu=1, compromise = True)
     #prob.driver.Optimization(opt_type='conmin', ouu=1)



    #prob.driver.options['optimizer'] = 'COBYLA'
    #prob.driver.options['maxiter'] = 1000
    #prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.add_objective('obj', scaler=1E-5)

    # set optimizer options
    #prob.driver.opt_settings['Verify level'] = 3
    #prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptAEP.out'
    #prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptAEP.out'
    #prob.driver.opt_settings['Major iterations limit'] = 1000

    # select design variables
    #for i in range(len(turbineX)):
    #  #prob.driver.add_desvar('turbineX[%d]'%i, lower=min(turbineX), upper=max(turbineX), scaler=1)
    #  prob.driver.add_desvar('turbineY[%d]'%i, lower=min(turbineY), upper=max(turbineY), scaler=1)

     #prob.driver.add_desvar('windSpeeds', lower=windSpeeds - 1e-8, upper=windSpeeds + 1e-8, scaler=1)
     #prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1)
     #prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1)



    #prob.driver.add_special_distribution('windSpeeds', 'normal', lower_bounds=5, upper_bounds=9, mean=7, std_dev=1)
    #prob.driver.add_special_distribution('windSpeeds', 'normal', lower_bounds=5, upper_bounds=9, mean=7, std_dev=1)

    # works
    if opting:
     prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*200, upper=np.ones(nTurbs)*1800, scaler=1)
     prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*200, upper=np.ones(nTurbs)*1800, scaler=1)


    for i in range(turbineX.size):
        #prob.driver.add_special_distribution('floris_params:ke[%d]'%i, 'normal', mean=ke, std_dev = 1e-11, lower_bounds = 0.02, upper_bounds =1.0)
        prob.driver.add_special_distribution('floris_params:ke[%d]'%i, 'normal', mean=ke, std_dev = 0.07, lower_bounds = 0.02, upper_bounds =1.0)
        if not opting:
            prob.driver.add_special_distribution('turbineX[%d]'%i, 'normal', lower_bounds=turbineX[i]-1e-8, upper_bounds=turbineX[i]+1e-8, mean = turbineX[i] , std_dev=1e-11)
            prob.driver.add_special_distribution('turbineY[%d]'%i, 'normal',lower_bounds=turbineY[i]-1e-8, upper_bounds=turbineY[i]+1e-8, mean = turbineY[i] , std_dev=1e-11)

    for i in range(size):
        #prob.driver.add_special_distribution('windSpeeds[%d]'%i, 'normal', lower_bounds=6., upper_bounds=15., mean=wind_speed, std_dev=1e-11)
        prob.driver.add_special_distribution('windSpeeds[%d]'%i, 'normal', lower_bounds=3., upper_bounds=17., mean=wind_speed, std_dev=.7)
    #prob.driver.add_special_distribution('air_density', 'normal', lower_bounds=0., upper_bounds=2., mean=1.1716, std_dev=.02)


    #prob.driver.add_special_distribution('wsm', 'normal', lower_bounds=-2, upper_bounds=2, mean=0, std_dev=1)

    #for direction_id in range(0, windDirections.size):
    #    prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)

    # add constraints
    prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0)
    prob.root.add('new_con', ExecComp('aep_con1 = -1.2e3 - obj'), promotes=['*'])
    prob.driver.add_constraint('aep_con1', upper=0.) 

    tic = time.time()
    prob.setup(check=False)
    toc = time.time()

    # print the results
    mpi_print(prob, ('FLORIS setup took %.03f sec.' % (toc-tic)))

    # time.sleep(10)
    # assign initial values to design variables

    #prob['turbineX'] = np.array([9.85276e2, 1.0138e3, 7.4383e2, 8.6856e3])
    #prob['turbineY'] = np.array([1.233e3, 1.1527e3, 9.9808e2, 6.7753e2])

    #prob['turbineX'] = np.array([1.0419e3, 1.2386e3, 9.064e3, 7.7887e2])
    #prob['turbineX'] = np.array([1.0419e3, 1.2386e3, 9.064e3, 7.7887e2])


    # ouu
    #prob['turbineX'] = np.array([ 6.5868268285e+02, 7.7440021896e+02, 9.6449457118e+02, 7.3171146728e+02, 
    #         1.4239115567e+03, 7.0622445984e+02, 8.0356539199e+02, 1.8291190158e+03, 1.8799010137e+03])
    #prob['turbineY'] = np.array([7.1869565674e+02 , 1.8798583010e+03, 1.3872804665e+03, 1.8039797090e+03, 1.8772847586e+03, 
    #         8.2107567818e+02, 6.7493181293e+02, 1.7874053173e+03 ,  1.7566516059e+03])

    # det
    #prob['turbineX'] = np.array([1.8596885915e+03, 9.9702503578e+02, 1.7154693853e+03, 1.8118952799e+03, 1.3444756985e+03,
    #         8.5087914400e+02, 1.0831552491e+03, 1.4975011511e+03, 6.9256271773e+02])
    #prob['turbineY'] = np.array([9.7945066268e+02, 7.3866785817e+02, 1.6628095542e+03, 1.2338737173e+03, 1.1071917915e+03,
    #         1.4837974880e+03, 6.5327332673e+02, 1.8357379200e+03, 1.6150482810e+03])

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['floris_params:ke'] = np.array([ke for i in range(nTurbs)])


    for direction_id in range(0, windDirections.size):
        prob['yaw%i' % direction_id] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    #prob['ratedPower'] = ratedPower
    prob['windSpeeds'] = windSpeeds
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    # set options
    # prob['floris_params:FLORISoriginal'] = True
    # prob['floris_params:CPcorrected'] = False
    # prob['floris_params:CTcorrected'] = False

    # run the problem
    mpi_print(prob, 'start FLORIS run')
    tic = time.time()
    # cProfile.run('prob.run()')
    prob.run()
    toc = time.time()

    # print the results
    mpi_print(prob, ('FLORIS Opt. calculation took %.03f sec.' % (toc-tic)))

    for direction_id in range(0, windDirections.size):
        mpi_print(prob,  'yaw%i (deg) = ' % direction_id, prob['yaw%i' % direction_id])
    # for direction_id in range(0, windDirections.size):
        # mpi_print(prob,  'velocitiesTurbines%i (m/s) = ' % direction_id, prob['velocitiesTurbines%i' % direction_id])
    # for direction_id in range(0, windDirections.size):
    #     mpi_print(prob,  'wt_power%i (kW) = ' % direction_id, prob['wt_power%i' % direction_id])

    mpi_print(prob,  'turbine X positions in wind frame (m): %s' % ', '.join([str(s) for s in prob['turbineX']]))
    mpi_print(prob,  'turbine Y positions in wind frame (m): %s' % ', '.join([str(s) for s in prob['turbineY']]))
    mpi_print(prob,  'turbine ke positions in wind frame (m): %s' % ', '.join([str(s) for s in prob['floris_params:ke']]))
    mpi_print(prob,  'speeds%s' % ', '.join([str(s) for s in prob['windSpeeds']]))
    mpi_print(prob,  'wind farm power in each direction (kW): %s' % prob['dirPowers'])
    mpi_print(prob,  'AEP (kWh): %s' % prob['AEP'])

    xbounds = [min(turbineX)/rotor_diameter, min(turbineX)/rotor_diameter, max(turbineX)/rotor_diameter, max(turbineX)/rotor_diameter, min(turbineX)/rotor_diameter]
    ybounds = [min(turbineY)/rotor_diameter, max(turbineY)/rotor_diameter, max(turbineY)/rotor_diameter, min(turbineY)/rotor_diameter, min(turbineX)/rotor_diameter]

    plt.figure()
    plt.plot(turbineX/rotor_diameter, turbineY/rotor_diameter, 'ok', label='Original')
    plt.plot(prob['turbineX']/rotor_diameter, prob['turbineY']/rotor_diameter, 'og', label='Optimized')
    plt.plot(xbounds, ybounds, ':k')
    for i in range(0, nTurbs):
        plt.plot([turbineX[i]/rotor_diameter, prob['turbineX'][i]/rotor_diameter], [turbineY[i]/rotor_diameter, prob['turbineY'][i]/rotor_diameter], '--k')
    plt.legend()
    plt.xlabel('Turbine X Position ($X/D_r$)')
    plt.ylabel('Turbine Y Position ($Y/D_r$)')
    plt.xlim([3, 12])
    plt.ylim([3, 12])
    plt.show()
