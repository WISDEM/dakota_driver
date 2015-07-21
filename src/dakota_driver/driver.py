"""
A collection of drivers using DAKOTA to exercise the workflow.
The general scheme is to have a separate class for each separate DAKOTA
method type.

Currently these drivers simply run the workflow, they do not parse any
DAKOTA results.
"""

from numpy import array
import collections

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

_NOT_SET = "SPECIFICATION DECLARED BUT NOT DEFINED"


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


    def configure_input(self):
        """ Configures input specification, must be overridden. """

        ######## 
       # method #
        ######## 
        n_params = self.total_parameters()
        if hasattr(self, 'get_ineq_constraints'): ineq_constraints = self.total_ineq_constraints()
        for key in self.input.method:
            if key == 'output': self.input.method[key] = self.output
            if key == 'max_iterations': self.input.method[key] = self.max_iterations
            if key == 'max_function_evaluations': self.input.method[key] = self.max_function_evaluations
            if key == 'convergence_tolerance': self.input.method[key] = self.convergence_tolerance
            if key == 'constraint_tolerance': 
               if ineq_constraints: self.input.method[key] = self.constraint_tolerance
               else: self.input.method.pop(key) 
            if key == 'seed': self.input.method[key] = self.seed

            # optimization
            if key == 'conmin':
                self.set_variables(need_start=True)
                if ineq_constraints: self.input.method['conmin_mfd'] = ''
                else: self.input.method['conmin_frcg'] = ''
                self.input.method.pop(key)

            
            # parameter studies
            if key == 'partitions':
                if len(self.partitions) != self.total_parameters():
                    self.raise_exception('#partitions (%s) != #parameters (%s)'
                                 % (len(self.partitions), self.total_parameters()),
                                 ValueError)
                partitions = [str(partition) for partition in self.partitions]
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
            if key == 'samples': self.input.method[key] = self.samples
        ########### 
       # variables #
        ########### 
        self.set_variables(need_start=self.need_start,
                           uniform=self.uniform,
                           need_bounds=self.need_bounds)
        ########### 
       # responses #
        ########### 
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
                self.input.responses[key] = ' '.join(names)

        ##################################################
       # Verify that all input fields have been adressed #
        ##################################################
        def assignment_enforcemer(tag,val):
             if val == _NOT_SET: raise ValueError(str(tag)+ " NOT DEFINED")
        for key in self.input.method: assignment_enforcemer(key,self.input.method[key])
        for key in self.input.responses: assignment_enforcemer(key,self.input.responses[key])

        #############################################################
       # map method and response from ordered dictionaries to lists  #
       #                                                             #
       # convention is if the value is an empty string there will be #
       #    no equals sign. Otherwise, data will be inoyt to dakota  #
       #    as "{key} = {associated value}"                          #
        #############################################################
        temp_list = []
        for key in self.input.method:
            if self.input.method[key]:
                temp_list.append(str(key) + ' = '+str(self.input.method[key]))
            else: temp_list.append(key)
        self.input.method = temp_list

        temp_list = []
        for key in self.input.responses:
            if self.input.responses[key]:
                temp_list.append(str(key) + ' = '+str(self.input.responses[key]))
            else: temp_list.append(key)
        self.input.responses = temp_list

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


class DakotaOptimizer(DakotaBase):
    """ Base class for optimizers using the DAKOTA Python interface. """
    # Currently only a 'marker' class.

    implements(IOptimizer)

################################################################################
########################## Hierarchical Driver ################################
class pydakdriver(DakotaBase):

    def __init__(self):
        super(pydakdriver, self).__init__()
        self.input.method = collections.OrderedDict()
        self.input.responses = collections.OrderedDict()

        # default definitions for set_variables
        self.need_start = False 
        self.uniform=False
        self.need_bounds=True
 
    def analytical_gradients(self):
         self.interval_typ = 'formed'
         self.fd_gradient_step_size = '1.e-4'
         for key in self.input.responses:
             if key == 'no_gradients':
                  self.input.responses.pop(key)
         self.input.responses['numerical_gradients'] = ''
         self.input.responses['method_source dakota'] = ''
         self.input.responses['interval_type '+interval_type] = ''
         self.input.responses['fd_gradient_step_size'] = _NOT_SET

    def numerical_gradients(self, method_source='dakota',
                            interval_type= Enum(values=('forward', 'central'), iotype='in',
                                            desc='Type of finite difference for gradients'),
                            fd_gradient_step_size=_NOT_SET):
         for key in self.input.responses:
             if key == 'no_gradients':
                  self.input.responses.pop(key)
         if method_source=='dakota':self.input.responses['method_source dakota']
         self.input.responses['interval_type'] = interval_type
         self.input.responses['fd_gradient_step_size'] = fd_gradient_step_size
         self.fd_gradient_step_size = fd_gradient_step_size

    def hessians(self):
         for key in self.input.responses:
             if key == 'no_hessians':
                  self.input.responses.pop(key)
         # todo: Create Hessian default with options

    def Optimization(self,opt_type='npsol_sqp', interval_typer = 'forward'):
        self.convergence_tolerance = '1.e-8'
        self.seed = _NOT_SET
        self.max_iterations = '200'
        self.max_function_evaluations = '2000'

        self.input.responses['objective_functions']=_NOT_SET
        self.input.responses['no_gradients'] = ''
        self.input.responses['no_hessians'] = ''
        if opt_type == 'npsol_sqp':
            self.need_start=True
            self.need_bounds=True
            self.input.method[opt_type] = ""
            self.input.method['convergence_tolerance'] = convergence_tolerance # please double check this kj 
        if opt_type == 'efficient_global':
            self.input.method["efficiency_global"] = ""
            self.input.method["seed"] = seed
        if opt_type == 'conmin':
            self.need_start=True           

            self.input.method["conmin"] = ''
            self.input.method["output"] = ''
            self.input.method['max_iterations'] = max_iterations
            self.input.method['max_function_evaluations'] = max_function_evaluations
            self.input.method['convergence_tolerance'] = convergence_tolerance
            self.input.method['constraint_tolerance'] = constraint_tolerance

            self.input.responses['nonlinear_inequality_constraints'] = _NOT_SET
            if interval_type in ['central','forward']:
               self.input.method['interval_type'+interval_type]=''
            else: self.raise_exception('invalid interval_type'+str(interval_type), ValueError)
            self.numerical_gradients()
            

    def Parameter_Study(self,study_type = 'vector'):
        self.study_type = study_type
        if study_type == 'vector':
            self.need_start=True
            self.need_bounds=False
            # why was this false? legacy was self.set_variables(need_start=False, need_bounds=False)
            self.input.method['vector_parameter_study'] = ""
            self.input.method['final_point'] = _NOT_SET 
            self.input.method['num_steps'] = _NOT_SET 
            self.final_point = _NOT_SET
            self.num_steps = _NOT_SET
        if study_type == 'multi-dim':
            self.need_start=False
            self.input.method['multidim_parameter_study'] = ""
            self.input.method['partitions'] = _NOT_SET 
            self.partitions =  _NOT_SET
        if study_type == 'list': #todo: specifiy instructions for set_variables  
            self.input.method['list_parameter_study'] = "" # todo
            self.input.method['list_of_points'] = _NOT_SET  # todo
            self.input.responses['response_functions']=_NOT_SET  # todo: add responsens_not_objectives()
        else: self.input.responses['objective_functions']=_NOT_SET 
        if study_type == 'centered': #todo: specifiy instructions for set_variables
            self.input.method['centered_parameter_study'] = ""
            self.input.method['step_vector'] = _NOT_SET  # todo
            self.input.method['steps_per_variable'] = _NOT_SET  # todo
        self.input.responses['no_gradients']=''
        self.input.responses['no_hessians']=''

    def UQ(self,UQ_type = 'sampling'):
            self.sample_type =  'lhs'
            self.seed = _NOT_SET
            self.samples=100
            
            if UQ_type == 'sampling':
                self.need_start = False
                self.uniform = True
                self.input.method = collections.OrderedDict()
                self.input.method['sampling'] = ''
                self.input.method['output'] = _NOT_SET
                self.input.method['sample_type'] = _NOT_SET
                self.input.method['seed'] = _NOT_SET
                self.input.method['samples'] = _NOT_SET
        
                self.input.responses = collections.OrderedDict()
                self.input.responses['num_response_functions'] = _NOT_SET
                self.input.responses['response_descriptors'] = _NOT_SET
            self.input.responses['no_gradients'] = ''
            self.input.responses['no_hessians'] = ''
################################################################################
