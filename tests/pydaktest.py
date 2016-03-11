from dakota_driver.driver import pydakdriver
from openmdao.main.api import Component, Assembly
from openmdao.lib.datatypes.api import Float
import unittest

class rosen(Component):
    x1 = Float(0.0, iotype='in', desc = 'The variable x1' )
    x2 = Float(0.0, iotype='in', desc = 'The variable x2' )
    f = Float(0.0, iotype='out', desc='F(x,y)')

    def execute(self):

        x1 = self.x1
        x2 = self.x2

        self.f = (1-x1)**2 + 100*(x2-x1**2)**2 

class rosenDistTest(Assembly):
    def configure(self):
        self.add('rose', rosen())
        driver_obj = pydakdriver()
        driver_obj.UQ()
        driver = self.add('driver',driver_obj)
        driver.stdout = 'dakotaDist.out'
        driver.stderr = 'dakotaDist.err'
        driver.sample_type = 'random'
        driver.seed = 20
        driver.samples = 500 
        driver.add_special_distribution('rose.x2', "weibull", alpha = .5, beta = 0.2)
        driver.add_parameter('rose.x1', low=-1.5, high=1.5)
        driver.add_objective('rose.f')

class rosenTest(Assembly):
    def configure(self):
        self.add('rose', rosen())
        driver_obj = pydakdriver()
        driver_obj.Parameter_Study(study_type = 'centered')
        driver = self.add('driver',driver_obj)
        driver.stdout = 'dakotaPS.out'
        driver.stderr = 'dakotaPS.err'
        driver.step_vector = [.1,.1]
        driver.steps_per_variable = 9
        driver.add_parameter('rose.x1', low=-1.5, high=1.5)
        driver.add_parameter('rose.x2', low=-1.5, high=1.5)
        driver.add_objective('rose.f')

class rosenOptTest(Assembly):
    def configure(self):
        self.add('rose', rosen())
        driver_obj = pydakdriver()
        driver_obj.Optimization(opt_type='conmin', surrogate_model=False)
        driver = self.add('driver',driver_obj)
        driver.stdout = 'dakotaOPT.out'
        driver.stderr = 'dakotaOPT.err'
        driver.samples = 500 
        driver.add_parameter('rose.x1', low=-1.5, high=1.5)
        driver.add_parameter('rose.x2', low=-1.5, high=1.5)
        driver.add_objective('rose.f')

class RoseTest(unittest.TestCase):
    def pstudy(self):
        top = rosenTest()
        top.run()
        self.assertTrue(top.rose.x2 == 0.9)
    def opt_test(self):
        top = rosenOptTest()
        top.run()
        self.assertTrue(top.rose.x1 - 0.9956 < 0.02)

if __name__ == '__main__':
    top = rosenDistTest()
    top.run()
    unittest.main()

