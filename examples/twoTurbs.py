from __future__ import print_function
from dakota_driver.driver import pydakdriver
from openmdao.api import Problem
from florisse.GeneralWindFarmComponents import calculate_boundary
from florisse.OptimizationGroups import OptAEP
from florisse import config

import time
import numpy as np
#import pylab as plt

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

    size = 1 # number of processors (and number of wind directions to run)

    #########################################################################
    # define turbine size
    rotor_diameter = 126.4  # (m)

    # define turbine locations in global reference frame
    # original example case
    # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
    # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m

    # Scaling grid case
    #nRows = int(sys.argv[1])     # number of rows and columns in grid
    spacing = 5     # turbine grid spacing in diameters

    # Set up position arrays
    #points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    #xpoints, ypoints = np.meshgrid(points, points)
    #locations = np.loadtxt('../../input_files/layout_amalia.txt')
    #boundaryVertices, boundaryNormals = calculate_boundary(locations)
    #turbineX = np.ndarray.flatten(xpoints)
    #turbineY = np.ndarray.flatten(ypoints)
    turbineX = np.array([300, 1300])
    turbineY = np.array([300, 300])
    #a = calculate_boundary([turbineX, turbineY])

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
        ratedPower[turbI] = 5000.0  # rated power of each turbine in kW

    # Define flow properties
    wind_speed = 7.0        # m/s
    air_density = 1.1716    # kg/m^3
    windDirections = np.array([90])
    size = windDirections.size
    windSpeeds = np.ones(size)*wind_speed
    windFrequencies = np.ones(size)/size

    # initialize problem
    prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size,
                                          minSpacing=minSpacing, differentiable=True, use_rotor_components=False))

    # set up optimizer
    prob.driver = pydakdriver('mydak')
    prob.driver.stdout = 'dak.out'
    prob.driver.add_objective('obj', scaler=-1e-6)

    # select design variables
    STD = 10.
    for direction_id in range(0, windDirections.size):
        prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)
        for n in range(nTurbs): prob.driver.add_special_distribution('yaw%i[%i]' % (direction_id,n), 'normal', mean=yaw[n], std_dev=STD, lower_bounds=-360.0, upper_bounds=360.0)
        #prob.driver.add_special_distribution("windDirections[0]", 'normal', mean=windDirections[0], std_dev = STD, lower_bounds=-99, upper_bounds=99)
        #for n in range(windDirections.size): prob.driver.add_special_distribution('windDirections[%i]' % (n), 'normal', mean=windDirections[n], std_dev=STD, lower_bounds=-9999, upper_bounds=9999)

    #if int(sys.argv[-1])==1:
    #   prob.driver.add_method('coliny_cobyla', gradients='analytic', model='nested', model_options={'primary_response_mapping':'1 0', 'secondary_variable_mapping':''})# , method_options={'max_iterations':300})
    #   prob.driver.add_method(response_type='r', model='single', method='sampling', method_options = {'sample_type':'lhs','samples':30000,'output':'silent'})
    prob.driver.add_method(response_type='r', model='nested', method='list_parameter_study', method_options = {'list_of_points':' '.join([str(s)+' '+str(s) for s in np.arange(0, 30, 5)]) }, variable_types=['custom'], variable_block=["continuous_design 2","  descriptors 'my_dev1' 'my_dev2'", "lower_bounds 0. 0.", "upper_bounds 40. 40."], model_options={'primary_response_mapping':'1 0', 'secondary_variable_mapping':"'std_deviation' 'std_deviation'"})
    #prob.driver.add_method(response_type='r', model='nested', method='list_parameter_study', method_options = {'list_of_points':' '.join([str(s) for s in np.arange(0, 30, 5)]) + ' 0 '}, variable_types=['custom'], variable_block=["continuous_design 1","  descriptors 'my_dev'", "lower_bounds 0", "upper_bounds 40"], model_options={'primary_variable_mapping': "'windDirections[0]'", 'primary_response_mapping':'1 0', 'secondary_variable_mapping':"'std_deviation'"})
    prob.driver.add_method(response_type='r', model='single', method='sampling', method_options = {'sample_type':'lhs','samples':40000,'output':'silent'}, variable_types=['uncertain'])
                                 
    #prob.driver.add_method(response_type='r', model='single', method='polynomial_chaos', method_options = {'quadrature_order':'5'})
    #else: prob.driver.add_method(response_type='r', model='single', method='sampling', method_options = {'sample_type':'lhs','samples':30000,'output':'silent'})
    tic = time.time()
    prob.setup(check=False)
    toc = time.time()

    # print the results
    mpi_print(prob, ('FLORIS setup took %.03f sec.' % (toc-tic)))

    # time.sleep(10)
    # assign initial values to design variables
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, windDirections.size):
        prob['yaw%i' % direction_id] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['ratedPower'] = ratedPower
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

    mpi_print(prob,  'turbine X positions in wind frame (m): %s' % prob['turbineX'])
    mpi_print(prob,  'turbine Y positions in wind frame (m): %s' % prob['turbineY'])
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
