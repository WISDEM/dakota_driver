from dakota_driver.driver import pydakdriver
from openmdao.main.api import Component, Assembly
from openmdao.lib.datatypes.api import Float
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
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.sample_type = 'random'
        driver.seed = 20
        driver.samples = 500 
        driver.add_special_distribution('rose.x2', "weibull", descriptor = "the x2 variable", alpha = .5, beta = 0.2)
        driver.add_parameter('rose.x1', low=-1.5, high=1.5)
        driver.add_objective('rose.f')

class rosentest(Assembly):
    def configure(self):
        self.add('rose', rosen())
        driver_obj = pydakdriver()
        driver_obj.UQ()
        driver = self.add('driver',driver_obj)
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.sample_type = 'random'
        driver.seed = 20
        driver.samples = 500 
        driver.add_parameter('rose.x1', low=-1.5, high=1.5)
        driver.add_parameter('rose.x2', low=-1.5, high=1.5)
        driver.add_objective('rose.f')

# top = rosenTest()
top = rosenDistTest()
top.run()
