===========
Usage Guide
===========

The following code is the equivalent of the gradient based constrained
optimization example from section 2.4.2.2 of the DAKOTA User's Manual:

::

    import csv

    from openmdao.main.api import Component, Assembly
    from openmdao.main.datatypes.api import Float
    from dakota_driver import DakotaCONMIN

    class Textbook(Component):
        """ DAKOTA 'text_book' function. """

        x1 = Float(iotype='in')
        x2 = Float(iotype='in')
        f  = Float(iotype='out')

        def execute(self):
            """ Just evaluate the function. """
            self.f = (self.x1 - 1)**4 + (self.x2 - 1)**4

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

    if __name__ == '__main__':
        top = ConstrainedOptimization()
        top.run()
        with open('dakota_tabular.dat', 'rb') as inp:
            reader = csv.reader(inp, delimiter=' ', skipinitialspace=True)
            for row in reader:
                print row

Consult the :ref:`dakota_driver_src_label` section for more detail.

