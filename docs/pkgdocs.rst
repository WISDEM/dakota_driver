
================
Package Metadata
================

- **classifier**:: 

    Intended Audience :: Science/Research
    Topic :: Scientific/Engineering

- **description-file:** README.txt

- **entry_points**:: 

    [openmdao.component]
    dakota_driver.test.test_driver.VectorStudy=dakota_driver.test.test_driver:VectorStudy
    dakota_driver.driver.DakotaVectorStudy=dakota_driver.driver:DakotaVectorStudy
    dakota_driver.driver.DakotaCONMIN=dakota_driver.driver:DakotaCONMIN
    dakota_driver.test.test_driver.ConstrainedOptimization=dakota_driver.test.test_driver:ConstrainedOptimization
    dakota_driver.test.test_driver.Textbook=dakota_driver.test.test_driver:Textbook
    dakota_driver.test.test_driver.ParameterStudy=dakota_driver.test.test_driver:ParameterStudy
    dakota_driver.test.test_driver.SensitivityStudy=dakota_driver.test.test_driver:SensitivityStudy
    dakota_driver.driver.DakotaBase=dakota_driver.driver:DakotaBase
    dakota_driver.test.test_driver.Optimization=dakota_driver.test.test_driver:Optimization
    dakota_driver.test.test_driver.Rosenbrock=dakota_driver.test.test_driver:Rosenbrock
    dakota_driver.driver.DakotaGlobalSAStudy=dakota_driver.driver:DakotaGlobalSAStudy
    dakota_driver.driver.DakotaOptimizer=dakota_driver.driver:DakotaOptimizer
    dakota_driver.test.test_driver.Broken=dakota_driver.test.test_driver:Broken
    dakota_driver.driver.DakotaMultidimStudy=dakota_driver.driver:DakotaMultidimStudy
    [openmdao.driver]
    dakota_driver.driver.DakotaOptimizer=dakota_driver.driver:DakotaOptimizer
    dakota_driver.driver.DakotaVectorStudy=dakota_driver.driver:DakotaVectorStudy
    dakota_driver.driver.DakotaCONMIN=dakota_driver.driver:DakotaCONMIN
    dakota_driver.driver.DakotaBase=dakota_driver.driver:DakotaBase
    dakota_driver.driver.DakotaGlobalSAStudy=dakota_driver.driver:DakotaGlobalSAStudy
    dakota_driver.driver.DakotaMultidimStudy=dakota_driver.driver:DakotaMultidimStudy
    [openmdao.container]
    dakota_driver.driver.DakotaOptimizer=dakota_driver.driver:DakotaOptimizer
    dakota_driver.driver.DakotaVectorStudy=dakota_driver.driver:DakotaVectorStudy
    dakota_driver.driver.DakotaCONMIN=dakota_driver.driver:DakotaCONMIN
    dakota_driver.test.test_driver.ConstrainedOptimization=dakota_driver.test.test_driver:ConstrainedOptimization
    dakota_driver.test.test_driver.VectorStudy=dakota_driver.test.test_driver:VectorStudy
    dakota_driver.test.test_driver.SensitivityStudy=dakota_driver.test.test_driver:SensitivityStudy
    dakota_driver.driver.DakotaBase=dakota_driver.driver:DakotaBase
    dakota_driver.test.test_driver.Optimization=dakota_driver.test.test_driver:Optimization
    dakota_driver.driver.DakotaGlobalSAStudy=dakota_driver.driver:DakotaGlobalSAStudy
    dakota_driver.test.test_driver.Rosenbrock=dakota_driver.test.test_driver:Rosenbrock
    dakota_driver.test.test_driver.Textbook=dakota_driver.test.test_driver:Textbook
    dakota_driver.test.test_driver.ParameterStudy=dakota_driver.test.test_driver:ParameterStudy
    dakota_driver.test.test_driver.Broken=dakota_driver.test.test_driver:Broken
    dakota_driver.driver.DakotaMultidimStudy=dakota_driver.driver:DakotaMultidimStudy

- **home-page:** https://github.com/OpenMDAO-Plugins/dakota_driver

- **keywords:** openmdao

- **license:** Apache License, Version 2.0

- **name:** dakota_driver

- **requires-dist**:: 

    openmdao.main
    pyDAKOTA

- **requires-python**:: 

    >=2.6
    <3.0

- **static_path:** [ '_static' ]

- **summary:** 'OpenMDAO drivers using DAKOTA (Design Analysis Kit for Optimization and Terascale Applications)'

- **version:** 0.4

