""" Test DAKOTA-based drivers. """

import csv
import glob
import logging
import nose
import os.path
import sys
import unittest

from openmdao.main.api import Component, Assembly, set_as_top
from openmdao.main.datatypes.api import Array, Float
from openmdao.util.testutil import assert_rel_error, assert_raises

from dakota_driver import DakotaCONMIN, DakotaMultidimStudy, \
                          DakotaVectorStudy, DakotaGlobalSAStudy


class Rosenbrock(Component):
    """ Standard two-dimensional Rosenbrock function. """

    x = Array([0., 0.], iotype='in')
    f = Float(iotype='out')

    def execute(self):
        """ Just evaluate the function. """
        x1 = self.x[0]
        x2 = self.x[1]
        self.f = 100 * (x2 - x1**2)**2 + (1 - x1)**2


class Textbook(Component):
    """ DAKOTA 'text_book' function. """

    x1 = Float(iotype='in')
    x2 = Float(iotype='in')
    f  = Float(iotype='out')

    def execute(self):
        """ Just evaluate the function. """
        self.f = (self.x1 - 1)**4 + (self.x2 - 1)**4


class Broken(Component):
    """ Always raises an exception. """

    x1 = Float(iotype='in')
    x2 = Float(iotype='in')
    f  = Float(iotype='out')

    def execute(self):
        """ Just raise a RuntimeError. """
        raise RuntimeError('Evaluating x1=%s, x2=%s' % (self.x1, self.x2))


class Optimization(Assembly):
    """ Use DAKOTA to perform an optimization. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('rosenbrock', Rosenbrock())

        driver = self.add('driver', DakotaCONMIN())
        driver.workflow.add('rosenbrock')
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.tabular_graphics_data = True
        driver.max_iterations = 100
        driver.convergence_tolerance = 1e-4
        driver.interval_type = 'forward'
        driver.fd_gradient_step_size = 1e-5

        driver.add_parameter('rosenbrock.x', low=-2, high=2, start=(-1.2, 1))
        driver.add_objective('rosenbrock.f')


class ConstrainedOptimization(Assembly):
    """ Use DAKOTA to perform a constrained optimization. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('textbook', Textbook())

        driver = self.add('driver', DakotaCONMIN())
        driver.workflow.add('textbook')
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.tabular_graphics_data = True
        driver.max_iterations = 50
        driver.convergence_tolerance = 1e-4
        driver.interval_type = 'central'
        driver.fd_gradient_step_size = 1e-4

        driver.add_parameter('textbook.x1', low=0.5, high=5.8, start=0.9)
        driver.add_parameter('textbook.x2', low=-2.9, high=2.9, start=1.1)
        driver.add_objective('textbook.f')
        driver.add_constraint('textbook.x1**2 - textbook.x2/2 <= 0', name='g1')
        driver.add_constraint('textbook.x2**2 - textbook.x1/2 <= 0', name='g2')


class ParameterStudy(Assembly):
    """ Use DAKOTA to run a multidimensional parameter study. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('rosenbrock', Rosenbrock())

        driver = self.add('driver', DakotaMultidimStudy())
        driver.workflow.add('rosenbrock')
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.partitions = [8, 8]

        driver.add_parameter('rosenbrock.x', low=-2, high=2)
        driver.add_objective('rosenbrock.f')


class VectorStudy(Assembly):
    """ Use DAKOTA to run a vector study. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('rosenbrock', Rosenbrock())

        driver = self.add('driver', DakotaVectorStudy())
        driver.workflow.add('rosenbrock')
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.final_point = [1.1, 1.3]
        driver.num_steps = 10

        driver.add_parameter('rosenbrock.x', start=(-0.3, 0.2))
        driver.add_objective('rosenbrock.f')


class SensitivityStudy(Assembly):
    """ Use DAKOTA to run a global sensitivity study. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('rosenbrock', Rosenbrock())

        driver = self.add('driver', DakotaGlobalSAStudy())
        driver.workflow.add('rosenbrock')
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.tabular_graphics_data = True

        driver.add_parameter('rosenbrock.x', low=-2, high=2)
        driver.add_objective('rosenbrock.f')


class TestCase(unittest.TestCase):
    """ Test DAKOTA-based drivers. """

    def tearDown(self):
        """ Cleanup files. """
        for pattern in ('LHS*', 'S4', 'dakota.out', 'dakota.err',
                        'dakota.rst', 'dakota_tabular.dat', 'driver.in'):
            for name in glob.glob(pattern):
                try:
                    os.remove(name)
                except WindowsError as exc:
                    # Currently no way to release DAKOTA streams.
                    logging.debug("Can't remove %s: %s", name, exc)

    def test_optimization(self):
        # Test DakotaCONMIN driver.
        logging.debug('')
        logging.debug('test_optimization')

        top = Optimization()
        top.run()
        # Current state isn't optimium,
        # probably left over from CONMIN line search.
        assert_rel_error(self, top.rosenbrock.x[0], 0.99401209, 0.00001)
        assert_rel_error(self, top.rosenbrock.x[1], 0.98869321, 0.00001)
        assert_rel_error(self, top.rosenbrock.f,  7.59464541e-05, 0.00001)

        with open('dakota_tabular.dat', 'rb') as inp:
            reader = csv.reader(inp, delimiter=' ', skipinitialspace=True)
            count = 0
            for row in reader:
#                print >>sys.stderr, row
                count += 1

        self.assertEqual(count, 83)
        self.assertEqual(row[0], '82')
        assert_rel_error(self, float(row[1]), 0.99401209, 0.00001)
        assert_rel_error(self, float(row[2]), 0.98869321, 0.00001)
        assert_rel_error(self, float(row[3]), 7.59464541e-05, 0.00001)

    def test_constrained_optimization(self):
        # Test DakotaCONMIN driver.
        logging.debug('')
        logging.debug('test_constrained_optimization')

        top = set_as_top(ConstrainedOptimization())
        top.run()
        # Current state isn't optimium,
        # probably left over from CONMIN line search.
        assert_rel_error(self, top.textbook.x1, 0.5, 0.0004)
        assert_rel_error(self, top.textbook.x2, 0.43167254, 0.0004)
        assert_rel_error(self, top.textbook.f,  0.16682649, 0.0007)

        with open('dakota_tabular.dat', 'rb') as inp:
            reader = csv.reader(inp, delimiter=' ', skipinitialspace=True)
            count = 0
            for row in reader:
#                print >>sys.stderr, row
                count += 1

        self.assertEqual(count, 31)
        self.assertEqual(row[0], '30')
        assert_rel_error(self, float(row[1]), 0.5, 0.0004)
        assert_rel_error(self, float(row[2]), 0.43167254, 0.0004)
        assert_rel_error(self, float(row[3]),  0.16682649, 0.0007)

        top.driver.clear_objectives()
        assert_raises(self, 'top.run()', globals(), locals(), ValueError,
                      'driver: No objectives, run aborted')

        top.driver.clear_parameters()
        assert_raises(self, 'top.run()', globals(), locals(), ValueError,
                      'driver: No parameters, run aborted')

    def test_broken_optimization(self):
        # Test exception handling. This requires a modified version of
        # DAKOTA that can be configured to not exit on analysis failure.
        logging.debug('')
        logging.debug('test_broken_optimization')

        top = set_as_top(ConstrainedOptimization())
        top.replace('textbook', Broken())
        try:
            top.run()
        except RuntimeError as exc:
            print exc
            self.assertTrue('driver: Evaluating x1=0.9, x2=1.1' in str(exc))
        else:
            self.fail('Expected RuntimeError')

    def test_multidim(self):
        # Test DakotaMultidimStudy driver.
        logging.debug('')
        logging.debug('test_multidim')

        top = ParameterStudy()
        top.run()
        self.assertEqual(top.rosenbrock.x[0], 2)
        self.assertEqual(top.rosenbrock.x[1], 2)
        self.assertEqual(top.rosenbrock.f,  401)

        top.driver.partitions = [8, 8, 999]
        assert_raises(self, 'top.run()', globals(), locals(), ValueError,
                      'driver: #partitions (3) != #parameters (2)')

    def test_vector(self):
        # Test DakotaVectorStudy driver.
        logging.debug('')
        logging.debug('test_vector')

        top = VectorStudy()
        top.run()
        assert_rel_error(self, top.rosenbrock.x[0], 1.1, 0.00001)
        assert_rel_error(self, top.rosenbrock.x[1], 1.3, 0.00001)
        assert_rel_error(self, top.rosenbrock.f,  0.82, 0.00001)

        top.driver.final_point = [1.1, 1.3, 999]
        assert_raises(self, 'top.run()', globals(), locals(), ValueError,
                      'driver: #final_point (3) != #parameters (2)')

    def test_sensitivity(self):
        # Test DakotaGlobalSAStudy driver.
        logging.debug('')
        logging.debug('test_sensitivity')

        top = set_as_top(SensitivityStudy())
        top.run()
        assert_rel_error(self, top.rosenbrock.x[0],  1.091489532, 0.00001)
        assert_rel_error(self, top.rosenbrock.x[1], -1.415779759, 0.00001)
        assert_rel_error(self, top.rosenbrock.f,   679.7206145, 0.00001)

        with open('dakota_tabular.dat', 'rb') as inp:
            reader = csv.reader(inp, delimiter=' ', skipinitialspace=True)
            count = 0
            for row in reader:
#                print >>sys.stderr, row
                count += 1
        self.assertEqual(count, 101)

    def test_errors(self):
        # Test base error responses.
        logging.debug('')
        logging.debug('test_errors')

        top = set_as_top(SensitivityStudy())

        top.driver.clear_objectives()
        assert_raises(self, 'top.run()', globals(), locals(), ValueError,
                      'driver: No objectives, run aborted')

        top.driver.clear_parameters()
        assert_raises(self, 'top.run()', globals(), locals(), ValueError,
                      'driver: No parameters, run aborted')


if __name__ == '__main__':
    sys.argv.append('--cover-package=dakota_driver')
    sys.argv.append('--cover-erase')
    nose.runmodule()

