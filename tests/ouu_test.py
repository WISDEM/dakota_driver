from dakota_driver.driver import pydakdriver
from openmdao.main.api import Component, Assembly
from openmdao.lib.drivers.api import SLSQPdriver, CONMINdriver, Genetic, COBYLAdriver, NEWSUMTdriver
from openmdao.lib.datatypes.api import Float
import unittest

dacout = 'dakota.out'

class get_dak_output(Component):
    mean_f = Float(0.0, iotype='out', desc='p50 cost of energy output from dakota')
    x1 = Float(-888, iotype = 'in')

    def execute(self):
       nam ='rose.f'

       # read data
       with open(dacout) as dak:
         daklines = dak.readlines()
       vals = {}
       for n, line in enumerate(daklines):
         if n<49 or n > len(daklines)-45: pass
         else:
           split_line = line.split()
           if len(split_line)==2:
               if split_line[1] not in vals: vals[split_line[1]] = [float(split_line[0])]
               else:vals[split_line[1]].append(float(split_line[0]))

       # Get mean value
       self.objective_vals = vals[nam]
       self.mean_f = sum(self.objective_vals)/len(self.objective_vals)
       print "GET_DAK X1 f", self.x1, self.mean_f

class rosen(Component):
    x1 = Float(0.0, iotype='in', desc = 'The variable x1' )
    x2 = Float(0.0, iotype='in', desc = 'The variable x2' )
    f = Float(0.0, iotype='out', desc='F(x,y)')

    def execute(self):
        x1 = self.x1
        x2 = self.x2
        self.f = (1-x1)**2 + 100*(x2-x1**2)**2
        print 'X1 X2', x1,x2

class rosenSA(Assembly):
    def configure(self):
        self.add('rose', rosen())
        self.add('x1', Float(-999, iotype = 'in'))
        self.connect('x1', 'rose.x1')

        driver_obj = pydakdriver()
        driver_obj.UQ()
        self.driver = self.add('driver',driver_obj)
        self.driver.stdout = dacout
        self.driver.stderr = 'dakota.err'
        self.driver.sample_type = 'random'
        self.driver.seed = 4
        self.driver.samples = 15
        self.driver.add_special_distribution('rose.x2', "normal", mean = .1, std_dev = 0.000000001)
        self.driver.add_objective('x1')
        self.driver.add_objective('rose.f')

class outer_opt(Assembly):
    def configure(self):
        self.add('driver',COBYLAdriver())

        self.add('roseSA',rosenSA())
        self.add('dakBak', get_dak_output())

        self.driver.workflow.add('roseSA')
        self.driver.workflow.add('dakBak')
        self.add('x1',Float(0.4, iotype = 'in'))
        self.driver.add_parameter('x1', low = -6, high = 6)
        self.connect('x1', ['roseSA.x1', 'dakBak.x1'])
        self.driver.add_objective('dakBak.mean_f')

class TestOUU(unittest.TestCase):
    def test(self):
        top = outer_opt()
        top.run()
        self.assertTrue(top.x1-0.33 < 0.01)

if __name__ == '__main__':
    unittest.main()
