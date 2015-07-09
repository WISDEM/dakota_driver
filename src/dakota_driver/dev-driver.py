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
        ##########
        # method #
        ##########
        n_params = self.total_parameters()
        ineq_constraints = self.total_ineq_constraints()
        for key in self.input.method:

            if key == 'output': self.input.method[key] = self.output
            if key == 'max_iterations': self.input.method[key] = self.max_iterations
            if key == 'max_function_evaluations': self.input.method[key] = self.max_function_evaluations
            if key == 'convergence_tolerance': self.input.method[key] = self.convergence_tolerance
            if key == 'constraint_tolerance': 
               if ineq_constraints: self.input.method[key] = self.constraint_tolerance
               else: self.input.method.pop(key) 

            if key == 'conmin':
                if ineq_constraints: self.input.method['conmin_mfd'] = ''
                else: self.input.method['conmin_frcg'] = ''
                self.input.method.pop(key)
            
            if key == 'partitions':
                if len(self.partitions) != self.total_parameters():
                    self.raise_exception('#partitions (%s) != #parameters (%s)'
                                 % (len(self.partitions), self.total_parameters()),
                                 ValueError)
                self.input.method[key] = ' '.join(partitions)

           if key == 'vector_parameter_study':
                if len(self.final_point) != n_params:
                    self.raise_exception('#final_point (%s) != #parameters (%s)'
                                 % (len(self.final_point), n_params),
                                 ValueError)
           final_point = [str(point) for point in self.final_point]
           if key == 'final_point': self.input.method[key] = ' '.join(final_point)
           if key == 'num_steps': self.input.method[key] = self.num_steps

           if key == 'sample_type': self.input.method[key] = self.sample_type
           if key == 'seed': self.input.method[key] = self.seed
           if key == 'samples': self.input.method[key] = self.samples

        #############
        # responses #
        #############
        objectives = self.get_objectives()
        for key in self.input.responses:
             if key =='objective_functions': self.input.responses[key] = len(objectives)
             if key == 'nonlinear_inequality_constraints' :
                 if ineq_constraints: self.input.responses[key] = ineq_constraints
                 else: self.input.responses.pop(key)
             if key == 'interval_type': self.input.responses[key] = self.interval_type
             if key == 'fd_gradient_step_size': self.input.responses[key] = self.fd_gradient_step_size

             if key == 'num_response_functions': self.input.responses[key] = len(objectives)
           if key == 'response_descriptors': 
                names = ['%r' % name for name in objectives.keys()]
                self.input.method[key] = ' '.join(names)

         ################################################################
        # method and response mapping from ordered dictionaries to lists #
         ################################################################

         temp_list = []
         for key in self.input.methods:
             if self.input.methods[key]:
                 temp_list.append(str(key) + ' = '+str(self.input.methods[key]))
             else: temp_list.append(key)
         self.input.methods = temp_list

         temp_list = []
         for key in self.input.responses:
             if self.input.responses[key]:
                 temp_list.append(str(key) + ' = '+str(self.input.responses[key]))
             else: temp_list.append(key)
         self.input.methods = temp_list

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

    """ Configures input specification. """

    self.input.method = collections.OrderedDict()
    self.input.method["conmin"] = ''
    self.input.method["output"] = ''
    self.input.method["max_iterations"] = -1
    self.input.method["max_function_evaluations"] = -1
    self.input.method["convergence_tolerance"] = -1
    self.input.method["constraint_tolerance"] = -1

    self.set_variables(need_start=True)

    self.input.responses = collections.OrderedDict()
    self.input.responses['objective_functions'] = -1
    self.input.responses['nonlinear_inequality_constraints'] = -1
    self.input.responses['numerical_gradients'] = ''
    self.input.responses['method_source dakota'] = ''
    self.input.responses['interval_type'] = 'default'
    self.input.responses['fd_gradient_step_size'] = 'default'
    self.input.responses['no_hessians'] = ''


class DakotaMultidimStudy(DakotaBase):
    """ Multidimensional parameter study using DAKOTA. """

    partitions = List(Int, low=1, iotype='in',
                      desc='List giving # of partitions for each parameter')

        """ Configures input specification. """

    partitions = [str(partition) for partition in self.partitions]
    objectives = self.get_objectives()

    self.input.method = collections.OrderedDict()
    self.input.method['multidim_parameter_study'] = ''
    self.input.method['output'] = 'default'
    self.input.method['partitions'] = 'default'

    self.set_variables(need_start=False)

    self.input.responses = collections.OrderedDict()
    self.input.responses['objective_functions'] = 'default'
    self.input.responses['no_gradients'] = ''
    self.input.responses['no_hessians'] = ''

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

        """ Configures the input specification. """

    self.input.method = collections.OrderedDict()
    self.input.method['output'] = 'default'
    self.input.method['vector_parameter_study'] = ""
    self.input.method['final_point'] = 'default'
    self.input.method['num_steps'] = 'default'

    self.set_variables(need_start=False, need_bounds=False)

    self.input.responses = collections.OrderedDict()
    self.input.responses['objective_functions'] = 'default'
    self.input.responses['no_gradients'] = ''
    self.input.responses['no_hessians'] = ''


class DakotaGlobalSAStudy(DakotaBase):
    """ Global sensitivity analysis using DAKOTA. """

    sample_type = Enum('lhs', iotype='in', values=('random', 'lhs'),
                       desc='Type of sampling')
    seed = Int(52983, iotype='in', desc='Seed for random number generator')
    samples = Int(100, iotype='in', low=1, desc='# of samples to evaluate')

        """ Configures input specification. """

    self.input.method = collections.OrderedDict()
    self.input.method['sampling'] = ''
    self.input.method['output' = 'default'
    self.input.method['sample_type'] = 'default'
    self.input.method['seed'] = 'default'
    self.input.method['samples'] = 'default'

    self.set_variables(need_start=False, uniform=True)

    self.input.responses = collections.OrderedDict()
    self.input.responses['num_response_functions'] = 'default'
    self.input.responses['response_descriptors'] = 'default'
    self.input.responses['no_gradients'] = ''
    self.input.responses['no_hessians'] = ''

