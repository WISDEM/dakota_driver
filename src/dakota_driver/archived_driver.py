"""
A collection of drivers using DAKOTA to exercise the workflow.
The general scheme is to have a separate class for each separate DAKOTA
method type.

Currently these drivers simply run the workflow, they do not parse any
DAKOTA results.
"""

from numpy import array

from dakota import DakotaInput, run_dakota

from openmdao.main.datatypes.api import Bool, Enum, Float, Int, List, Str
from openmdao.main.driver import Driver
from openmdao.main.hasparameters import HasParameters
from openmdao.main.hasconstraints import HasIneqConstraints
from openmdao.main.hasobjective import HasObjectives
from openmdao.main.interfaces import IHasParameters, IHasIneqConstraints, \
                                     IHasObjectives, IOptimizer, implements
from openmdao.util.decorators import add_delegate

__all__ = ['DakotaCONMIN', 'DakotaMultidimStudy', 'DakotaVectorStudy',
           'DakotaGlobalSAStudy', 'DakotaOptimizer', 'DakotaBase']


@add_delegate(HasParameters, HasObjectives)
class DakotaBase(Driver):
    """
    Base class for common DAKOTA operations, adds :class:`DakotaInput` instance.
    The ``method`` and ``responses`` sections of `input` must be set
    directly.  :meth:`set_variables` is typically used to set the ``variables``
    section.
    """

    implements(IHasParameters, IHasObjectives)

    output = Enum('normal', iotype='in', desc='Output verbosity',
                  values=('silent', 'quiet', 'normal', 'verbose', 'debug'))
    stdout = Str('', iotype='in', desc='DAKOTA stdout filename')
    stderr = Str('', iotype='in', desc='DAKOTA stderr filename')
    tabular_graphics_data = \
             Bool(iotype='in',
                  desc="Record evaluations to 'dakota_tabular.dat'")

    def __init__(self):
        super(DakotaBase, self).__init__()

        # Set baseline input, don't touch 'interface'.
        self.input = DakotaInput(environment=[],
                                 method=[],
                                 model=['single'],
                                 variables=[],
                                 responses=[])

    def check_config(self, strict=False):
        """ Verify valid configuration. """
        super(DakotaBase, self).check_config(strict=strict)

        parameters = self.get_parameters()
        if not parameters:
            self.raise_exception('No parameters, run aborted', ValueError)

        objectives = self.get_objectives()
        if not objectives:
            self.raise_exception('No objectives, run aborted', ValueError)

    def configure_input(self):
        """ Configures input specification, must be overridden. """
        self.raise_exception('configure_input', NotImplementedError)

    def execute(self):
        """ Write DAKOTA input and run. """
        self.configure_input()
        self.run_dakota()

    def set_variables(self, need_start, uniform=False, need_bounds=True):
        """ Set :class:`DakotaInput` ``variables`` section. """
        parameters = self.get_parameters()

        if uniform:
            self.input.variables = [
                'uniform_uncertain = %s' % self.total_parameters()]
        else:
            self.input.variables = [
                'continuous_design = %s' % self.total_parameters()]

        if need_start:
            initial = [str(val) for val in self.eval_parameters(dtype=None)]
            self.input.variables.append(
                '  initial_point %s' % ' '.join(initial))

        if need_bounds:
            lbounds = [str(val) for val in self.get_lower_bounds(dtype=None)]
            ubounds = [str(val) for val in self.get_upper_bounds(dtype=None)]
            self.input.variables.extend([
                '  lower_bounds %s' % ' '.join(lbounds),
                '  upper_bounds %s' % ' '.join(ubounds)])

        names = []
        for param in parameters.values():
            for name in param.names:
                names.append('%r' % name)

        self.input.variables.append(
            '  descriptors  %s' % ' '.join(names)
        )

    def run_dakota(self):
        """
        Call DAKOTA, providing self as data, after enabling or disabling
        tabular graphics data in the ``environment`` section.
        DAKOTA will then call our :meth:`dakota_callback` during the run.
        """
        if not self.input.method:
            self.raise_exception('Method not set', ValueError)
        if not self.input.variables:
            self.raise_exception('Variables not set', ValueError)
        if not self.input.responses:
            self.raise_exception('Responses not set', ValueError)

        for i, line in enumerate(self.input.environment):
            if 'tabular_graphics_data' in line:
                if not self.tabular_graphics_data:
                    self.input.environment[i] = \
                        line.replace('tabular_graphics_data', '')
                break
        else:
            if self.tabular_graphics_data:
                self.input.environment.append('tabular_graphics_data')

        infile = self.get_pathname() + '.in'
        self.input.write_input(infile, data=self)
        try:
            run_dakota(infile, stdout=self.stdout, stderr=self.stderr)
        except Exception:
            self.reraise_exception()

    def dakota_callback(self, **kwargs):
        """
        Return responses from parameters.  `kwargs` contains:

        ========== ==============================================
        Key        Definition
        ========== ==============================================
        functions  number of functions (responses, constraints)
        ---------- ----------------------------------------------
        variables  total number of variables
        ---------- ----------------------------------------------
	cv         list/array of continuous variable values
        ---------- ----------------------------------------------
        div        list/array of discrete integer variable values
        ---------- ----------------------------------------------
        drv        list/array of discrete real variable values
        ---------- ----------------------------------------------
        av         single list/array of all variable values
        ---------- ----------------------------------------------
        cv_labels  continuous variable labels
        ---------- ----------------------------------------------
        div_labels discrete integer variable labels
        ---------- ----------------------------------------------
        drv_labels discrete real variable labels
        ---------- ----------------------------------------------
        av_labels  all variable labels
        ---------- ----------------------------------------------
        asv        active set vector (bit1=f, bit2=df, bit3=d^2f)
        ---------- ----------------------------------------------
        dvv        derivative variables vector
        ---------- ----------------------------------------------
        currEvalId current evaluation ID number
        ========== ==============================================

        """
        cv = kwargs['cv']
        asv = kwargs['asv']
        self._logger.debug('cv %s', cv)
        self._logger.debug('asv %s', asv)

        self.set_parameters(cv)
        self.run_iteration()

        expressions = self.get_objectives().values()
        if hasattr(self, 'get_eq_constraints'):
            expressions.extend(self.get_eq_constraints().values())
        if hasattr(self, 'get_ineq_constraints'):
            expressions.extend(self.get_ineq_constraints().values())

        fns = []
        for i, expr in enumerate(expressions):
            if asv[i] & 1:
                val = expr.evaluate(self.parent)
                if isinstance(val, list):
                    fns.extend(val)
                else:
                    fns.append(val)
            if asv[i] & 2:
                self.raise_exception('Gradients not supported yet',
                                     NotImplementedError)
            if asv[i] & 4:
                self.raise_exception('Hessians not supported yet',
                                     NotImplementedError)

        retval = dict(fns=array(fns))
        self._logger.debug('returning %s', retval)
        return retval


class DakotaOptimizer(DakotaBase):
    """ Base class for optimizers using the DAKOTA Python interface. """
    # Currently only a 'marker' class.

    implements(IOptimizer)


@add_delegate(HasIneqConstraints)
class DakotaCONMIN(DakotaOptimizer):
    """ CONMIN optimizer using DAKOTA.  """

    implements(IHasIneqConstraints)

    max_iterations = Int(100, low=1, iotype='in',
                         desc='Max number of iterations to execute')
    max_function_evaluations = Int(1000, low=1, iotype='in',
                                   desc='Max number of function evaluations')
    convergence_tolerance = Float(1.e-7, low=1.e-10, iotype='in',
                                  desc='Convergence tolerance')
    constraint_tolerance = Float(1.e-7, low=1.e-10, iotype='in',
                                 desc='Constraint tolerance')
    fd_gradient_step_size = Float(1.e-5, low=1.e-10, iotype='in',
                                  desc='Relative step size for gradients')
    interval_type = Enum(values=('forward', 'central'), iotype='in',
                         desc='Type of finite difference for gradients')

    def __init__(self):
        super(DakotaCONMIN, self).__init__()
        # DakotaOptimizer leaves _max_objectives at 0 (unlimited).
        self._hasobjectives._max_objectives = 1

    def configure_input(self):
        """ Configures input specification. """
        objectives = self.get_objectives()
        ineq_constraints = self.total_ineq_constraints()

        method = 'conmin_mfd' if ineq_constraints else 'conmin_frcg'
        self.input.method = [
            '%s' % method,
            '  output = %s' % self.output,
            '  max_iterations = %s' % self.max_iterations,
            '  max_function_evaluations = %s' % self.max_function_evaluations,
            '  convergence_tolerance = %s' % self.convergence_tolerance]
        if ineq_constraints:
            self.input.method.append(
                '  constraint_tolerance = %s' % self.constraint_tolerance)

        self.set_variables(need_start=True)

        self.input.responses = [
            'objective_functions = %s' % len(objectives)]

        if ineq_constraints:
            self.input.responses.append(
                'nonlinear_inequality_constraints = %s' % ineq_constraints)

        self.input.responses.extend([
            'numerical_gradients',
            '  method_source dakota',
            '  interval_type %s' % self.interval_type,
            '  fd_gradient_step_size = %s' % self.fd_gradient_step_size,
            '  no_hessians',
        ])


class DakotaMultidimStudy(DakotaBase):
    """ Multidimensional parameter study using DAKOTA. """

    partitions = List(Int, low=1, iotype='in',
                      desc='List giving # of partitions for each parameter')

    def configure_input(self):
        """ Configures input specification. """
        if len(self.partitions) != self.total_parameters():
            self.raise_exception('#partitions (%s) != #parameters (%s)'
                                 % (len(self.partitions), self.total_parameters()),
                                 ValueError)

        partitions = [str(partition) for partition in self.partitions]
        objectives = self.get_objectives()

        self.input.method = [
            'multidim_parameter_study',
            '  output = %s' % self.output,
            '  partitions = %s' % ' '.join(partitions)]

        self.set_variables(need_start=False)

        self.input.responses = [
            'objective_functions = %s' % len(objectives),
            'no_gradients',
            'no_hessians']


class DakotaVectorStudy(DakotaBase):
    """ Vector parameter study using DAKOTA. """

    final_point = List(Float, iotype='in',
                       desc='List of final parameter values')
    num_steps = Int(1, low=1, iotype='in',
                    desc='Number of steps along path to evaluate')

    def __init__(self):
        super(DakotaVectorStudy, self).__init__()
        for dname in self._delegates_:
            delegate = getattr(self, dname)
            if isinstance(delegate, HasParameters):
                delegate._allowed_types.append('unbounded')
                break

    def configure_input(self):
        """ Configures the input specification. """
        n_params = self.total_parameters()
        if len(self.final_point) != n_params:
            self.raise_exception('#final_point (%s) != #parameters (%s)'
                                 % (len(self.final_point), n_params),
                                 ValueError)

        final_point = [str(point) for point in self.final_point]
        objectives = self.get_objectives()

        self.input.method = [
            'output = %s' % self.output,
            'vector_parameter_study',
            '  final_point = %s' % ' '.join(final_point),
            '  num_steps = %s' % self.num_steps]

        self.set_variables(need_start=False, need_bounds=False)

        self.input.responses = [
            'objective_functions = %s' % len(objectives),
            'no_gradients',
            'no_hessians']


class DakotaGlobalSAStudy(DakotaBase):
    """ Global sensitivity analysis using DAKOTA. """

    sample_type = Enum('lhs', iotype='in', values=('random', 'lhs'),
                       desc='Type of sampling')
    seed = Int(52983, iotype='in', desc='Seed for random number generator')
    samples = Int(100, iotype='in', low=1, desc='# of samples to evaluate')

    def configure_input(self):
        """ Configures input specification. """
        objectives = self.get_objectives()

        self.input.method = [
            'sampling',
            '  output = %s' % self.output,
            '  sample_type = %s' % self.sample_type,
            '  seed = %s' % self.seed,
            '  samples = %s' % self.samples]

        self.set_variables(need_start=False, uniform=True)

        names = ['%r' % name for name in objectives.keys()]
        self.input.responses = [
            'num_response_functions = %s' % len(objectives),
            'response_descriptors = %s' % ' '.join(names),
            'no_gradients',
            'no_hessians']

