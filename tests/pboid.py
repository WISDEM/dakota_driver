from __future__ import print_function
from dakota_driver.driver import pydakdriver
from openmdao.api import ScipyOptimizer
from openmdao.api import IndepVarComp, Component, Problem, Group

class Paraboloid(Component):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

    def __init__(self):
        super(Paraboloid, self).__init__()

        self.add_param('x', val=0.0)
        self.add_param('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        """

        x = params['x']
        y = params['y']

        unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def linearize(self, params, unknowns, resids):
        """ Jacobian for our paraboloid."""

        x = params['x']
        y = params['y']
        J = {}

        J['f_xy', 'x'] = 2.0*x - 6.0 + y
        J['f_xy', 'y'] = 2.0*y + 8.0 + x
        return J
top = Problem()

root = top.root = Group()

# Initial value of x and y set in the IndepVarComp.
root.add('p1', IndepVarComp('x', 13.0))
root.add('p2', IndepVarComp('y', -14.0))
root.add('p', Paraboloid())

root.connect('p1.x', 'p.x')
root.connect('p2.y', 'p.y')

#drives = pydakdriver()
#drives.Optimization()
#drives.UQ()
#top.driver = drives
top.driver = ScipyOptimizer()
top.driver.options['optimizer'] = 'SLSQP'

top.driver.add_desvar('p1.x', lower=-50, upper=50)
top.driver.add_desvar('p2.y', lower=-50, upper=50)
top.driver.add_objective('p.f_xy')

top.setup()

# You can also specify initial values post-setup
top['p1.x'] = 3.0
top['p2.y'] = -4.0

top.run()

print('\n')
print('Minimum of %f found at (%f, %f)' % (top['p.f_xy'], top['p.x'], top['p.y']))

