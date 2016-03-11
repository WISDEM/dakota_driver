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

from openmdao.core.driver import Driver
from openmdao.util.record_util import create_local_meta, update_local_meta
#import sys
#from openmdao.main.hasparameters import HasParameters
#from openmdao.main.hasconstraints import HasIneqConstraints
#from openmdao.main.hasobjective import HasObjectives
#from openmdao.main.interfaces import IHasParameters, IHasIneqConstraints, \
#                                     IHasObjectives, IOptimizer, implements
#from openmdao.util.decorators import add_delegate

__all__ = ['DakotaCONMIN', 'DakotaMultidimStudy', 'DakotaVectorStudy',
           'DakotaGlobalSAStudy', 'DakotaOptimizer', 'DakotaBase']

_SET_AT_RUNTIME = "SPECIFICATION DECLARED BUT NOT DEFINED"


#@add_delegate(HasParameters, HasObjectives)
class DakotaBase(Driver):
    """
    Base class for common DAKOTA operations, adds :class:`DakotaInput` instance.
    The ``method`` and ``responses`` sections of `input` must be set
    directly.  :meth:`set_variables` is typically used to set the ``variables``
    section.
    """

#    implements(IHasParameters, IHasObjectives)

    output = 'normal',
    #output = Enum('normal', iotype='in', desc='Output verbosity',
    #              values=('silent', 'quiet', 'normal', 'verbose', 'debug'))
    stdout = ''
    stderr = ''
    tabular_graphics_data = True
            

    def __init__(self):
        super(DakotaBase, self).__init__()

        # allow for special variable distributions
        self.special_distribution_variables = []
        self.clear_special_variables()
 
        self.configured = None
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
        if not parameters and not self.special_distribution_variables:
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
        #parameters = self.get_parameters()
        parameters = self._desvars
        if not parameters:
            self.raise_exception('No parameters, run aborted', ValueError)

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

        #infile = self.get_pathname() + '.in'
        infile = self.name+ '.in'
        self.input.write_input(infile, data=self)
        run_dakota(infile, stdout=self.stdout, stderr=self.stderr)
        #try:
        #    run_dakota(infile, stdout=self.stdout, stderr=self.stderr)
        #except Exception:
        #    print sys.exc_info()
        #    exc_type, exc_value, exc_traceback = sys.exc_info()
        #    raise type('%s' % exc_type), exc_value, exc_traceback

           # self.reraise_exception()

    def dakota_callback(self, **kwargs):
        print 'HELLO'
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
        #self._logger.debug('cv %s', cv)
        #self._logger.debug('asv %s', asv)

        for i, var  in enumerate(self._desvars.keys()):
            print i, cv
            self.set_desvar(var, cv[i])
        #self.set_parameters(cv)
        #self.run_iteration()
        system = self.root
        with system._dircontext:
            system.solve_nonlinear()

        expressions = self.get_objectives()#.values()
        print 'expr',expressions
        if hasattr(self, 'get_eq_constraints'):
            expressions.extend(self.get_eq_constraints().values()) # revisit - won't work with ordereddict
        if hasattr(self, 'get_ineq_constraints'):
            expressions.extend(self.get_ineq_constraints().values())

        fns = []
        fnGrads = []
        for i, val in enumerate(expressions.values()):
            if asv[i] & 1:
                #val = expr.evaluate(self.parent)
                #if isinstance(val, list):
                #if isinstance(val, array):
                fns.extend(val)
                #else:
                #    fns.append(val)
            if asv[i] & 2:
               #val = expr.evaluate_gradient(self.parent)
               fnGrads.append(val)
               # self.raise_exception('Gradients not supported yet',
               #                      NotImplementedError)
            if asv[i] & 4:
                self.raise_exception('Hessians not supported yet',
                                     NotImplementedError)

        retval = dict(fns=array(fns), fnGrads = array(fnGrads))
        #self._logger.debug('returning %s', retval)
        print 'BYE',retval
        return retval


    def configure_input(self, problem):
        """ Configures input specification, must be overridden. """

        ######## 
       # method #
        ######## 
        print 'yo! ps are ',self._desvars
        n_params = len(self._desvars.keys())
        if hasattr(self, 'get_ineq_constraints'): ineq_constraints = self.total_ineq_constraints()
        else: ineq_constraints = False
        for key in self.input.method:
            if key == 'output': self.input.method[key] = self.output[0]
            if key == 'max_iterations': self.input.method[key] = self.max_iterations
            if key == 'max_function_evaluations': self.input.method[key] = self.max_function_evaluations
            if key == 'convergence_tolerance': self.input.method[key] = self.convergence_tolerance
            if key == 'fd_gradient_step_size': self.input.method[key] = self.fd_gradient_step_size
            if key == 'constraint_tolerance': 
               if ineq_constraints: self.input.method[key] = self.constraint_tolerance
               else: self.input.method.pop(key) 
            if key == 'seed': self.input.method[key] = self.seed

            # optimization
            if key == 'conmin':
                self.set_variables(need_start=True)
                if ineq_constraints: 
                    conmeth = 'conmin_mfd'
                else: 
                     conmeth = 'conmin_frcg'
                self.input.method = collections.OrderedDict([
                     (conmeth, v) if k == key else (k, v) for k, v in self.input.method.items()])

            
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
            if key == 'step_vector': self.input.method[key] = ' '.join([str(s) for s in self.step_vector])
            if key == 'num_steps': self.input.method[key] = self.num_steps
            if key == 'steps_per_variable': self.input.method[key] = self.steps_per_variable
            if key == 'list_of_points': self.input.method[key] = '\n'.join(' '.join(str(j) for j in i) for i in self.list_of_points)

            if key == 'sample_type': self.input.method[key] = self.sample_type
            if key == 'samples': self.input.method[key] = self.samples
        
            # surrogate model
            if key == "surrogate": 
               self.input.method["model_pointer"] = "'surr'\n"
               self.input.method["method"] = "\n\t\tid_method 'dace'\n\t\tsampling\n\t\tsamples %i\n\t\tmodel_pointer 'sim'"%self.samples
               self.input.model = ["id_model 'surr'", "surrogate global gaussian_process surfpack",
                                "dace_method_pointer 'dace'\n", "model", "id_model 'sim'","\tsingle"]
               self.input.method.pop(key)

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
            if key =='response_functions': self.input.responses[key] = len(objectives)
            if key == 'nonlinear_inequality_constraints' :
               if ineq_constraints: self.input.responses[key] = ineq_constraints
               else: self.input.responses.pop(key)
            if key == 'interval_type': 
               self.input.responses = collections.OrderedDict([(key+' '+self.interval_type, v) if k == key else (k, v) for k, v in self.input.responses.items()])
            if key == 'fd_gradient_step_size': self.input.responses[key] = self.fd_gradient_step_size

            if key == 'num_response_functions': self.input.responses[key] = len(objectives)
            if key == 'response_descriptors': 
                names = ['%r' % name for name in objectives.keys()]
                self.input.responses[key] = ' '.join(names)

        ##################################################
       # Verify that all input fields have been adressed #
        ##################################################
        def assignment_enforcemer(tag,val):
             if val == _SET_AT_RUNTIME: raise ValueError(str(tag)+ " NOT DEFINED")
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

        self.configured = 1

    #def execute(self):
    def run(self, problem):
        print 'RUNNING DAK'
        """ Write DAKOTA input and run. """
        if not self.configured: self.configure_input(problem) # this limits configuration to one time
        self.run_dakota()

    def set_variables(self, need_start, uniform=False, need_bounds=True):
        """ Set :class:`DakotaInput` ``variables`` section. """

        dvars = self.get_desvars()
        parameters = [] # [ [name, value], ..]
        for param in dvars.keys():
            print dvars[param]
            if len( dvars[param]) == 1:
                parameters.append( [param, dvars[param][0]])
            else:
                print dvars[param]
                for i, val in enumerate(dvars[param]):
                    parameters.append([param+str(i+1),val])
        print 'hey! ps are ',parameters
        #parameters = self.get_parameters()
        if parameters:
            if uniform:
                self.input.variables = [
                    'uniform_uncertain = %s' % len(parameters)]
                    #'uniform_uncertain = %s' % self.total_parameters()]
            else:
                self.input.variables = [
                    'continuous_design = %s' % len(parameters)]
                    #'continuous_design = %s' % self.total_parameters()]
    
            if need_start:
                print 'yo,',self.get_desvars()
                #initial = [str(val[0] for val in self.get_desvars().values()]
                initial = []
                for val in self.get_desvars().values():
                    if isinstance(val, collections.Iterable):
                        initial.extend(val)
                    else: initial.append(val)
                #initial = [str(val) for val in self.eval_parameters(dtype=None)]
                self.input.variables.append(
                    '  initial_point %s' % ' '.join(str(s) for s in initial))
    
            if need_bounds:
                #lbounds = [str(val) for val in self.get_lower_bounds(dtype=None)]
                #ubounds = [str(val) for val in self.get_upper_bounds(dtype=None)]
                lbounds = []
                for val in self._desvars.values():
                    if isinstance(val["lower"], collections.Iterable):
                        lbounds.extend(val["lower"])
                    else: lbounds.append(val["lower"])
                #lbounds = [str(val['lower']) for val in parameters.values()]
                #ubounds = [str(val['upper']) for val in parameters.values()]
                ubounds = []
                for val in self._desvars.values():
                    if isinstance(val["upper"], collections.Iterable):
                        ubounds.extend(val["upper"])
                    else: ubounds.append(val["upper"])
                self.input.variables.extend([
                    '  lower_bounds %s' % ' '.join(str(bnd) for bnd in lbounds),
                    '  upper_bounds %s' % ' '.join(str(bnd) for bnd in ubounds)])
    
            names = [s[0] for s in parameters]
            #names = []
            #for param in parameters.values():
            #    for name in param.names:
            #        names.append('%r' % name)
    
            self.input.variables.append(
                '  descriptors  %s' % ' '.join( "'"+str(nam)+"'" for nam in names)
            )
        # ------------ special distributions cases ------- -------- #
        for var in self.special_distribution_variables:
             if var in parameters: self.remove_parameter(var)
             self.add_parameter(var,low= -999, high = 999)


        if self.normal_descriptors:
            self.input.variables.extend([
                'normal_uncertain =  %s' % len(self.normal_means),
                '  means  %s' % ' '.join(self.normal_means),
                '  std_deviations  %s' % ' '.join(self.normal_std_devs),
                "  descriptors  '%s'" % "' '".join(self.normal_descriptors)
                ])
                   
        if self.lognormal_descriptors:
            self.input.variables.extend([
                'lognormal_uncertain = %s' % len(self.lognormal_means),
                '  means  %s' % ' '.join(self.lognormal_means),
                '  std_deviations  %s' % ' '.join(self.lognormal_std_devs),
                "  descriptors  '%s'" % "' '".join(self.lognormal_descriptors)
                ])
                   
        if self.exponential_descriptors:
            self.input.variables.extend([
                'exponential_uncertain = %s' % len(self.exponential_descriptors),
                '  betas  %s' % ' '.join(self.exponential_betas),
                "  descriptors ' %s'" % "' '".join(self.exponential_descriptors)
                ])
                   
        if self.beta_descriptors:
            self.input.variables.extend([
                'beta_uncertain = %s' % len(self.beta_descriptors),
                '  betas = %s' % ' '.join(self.beta_betas),
                '  alphas = %s' % ' '.join(self.beta_alphas),
                "  descriptors = '%s'" % "' '".join(self.beta_descriptors),
                '  lower_bounds = %s' % ' '.join(self.beta_lower_bounds),
                '  upper_bounds = %s' % ' '.join(self.beta_upper_bounds)
                ])

        if self.gamma_descriptors:
            self.input.variables.extend([
                'beta_uncertain = %s' % len(self.gamma_descriptors),
                '  betas = %s' % ' '.join(self.gamma_betas),
                '  alphas = %s' % ' '.join(self.gamma_alphas),
                "  descriptors = '%s'" % "' '".join(self.gamma_descriptors)
                ])

        if self.weibull_descriptors:
            self.input.variables.extend([
                'weibull_uncertain = %s' % len(self.weibull_descriptors),
                '  betas  %s' % ' '.join(self.weibull_betas),
                '  alphas  %s' % ' '.join(self.weibull_alphas),
                "  descriptors  '%s'" % "' '".join(self.weibull_descriptors)
                ])
        

# ---------------------------  special distributions ---------------------- #
 
    def clear_special_variables(self):
       for var in self.special_distribution_variables:
          try: self.remove_parameter(var)
          except AttributeError:
             print var +' is being cleared but its not declared'
             pass
       self.special_distribution_variables = []

       self.normal_means = []
       self.normal_std_devs = []
       self.normal_descriptors = []
       #normal_lower_bounds = []
       #normal_upper_bounds = []
   
       self.lognormal_means= []
       self.lognormal_std_devs = []
       self.lognormal_descriptors = []
   
       self.exponential_betas = []
       self.exponential_descriptors = []
   
       self.beta_betas = []
       self.beta_alphas = []
       self.beta_descriptors = []
       self.beta_lower_bounds = []
       self.beta_upper_bounds = []

       self.gamma_alphas = []
       self.gamma_betas = []
       self.gamma_descriptors = []

       self.weibull_alphas = []
       self.weibull_betas = []
       self.weibull_descriptors = []

    def add_special_distribution(self, var, dist, alpha = _SET_AT_RUNTIME, beta = _SET_AT_RUNTIME, 
                                 mean = _SET_AT_RUNTIME, std_dev = _SET_AT_RUNTIME,
                                 lower_bounds = _SET_AT_RUNTIME, upper_bounds = _SET_AT_RUNTIME ):
        def check_set(option):
            if option == _SET_AT_RUNTIME: raise ValueError("INCOMPLETE DEFINITION FOR VARIABLE "+str(var))

        if dist == 'normal':
            check_set(std_dev)
            check_set(mean)
           # check_set(lower_bounds)
           # check_set(upper_bounds)
          #  self.normal_lower_bounds.append(str(lower_bounds))
          #  self.normal_upper_bounds.append(str(upper_bounds))
            self.normal_means.append(str(mean))
            self.normal_std_devs.append(str(std_dev))
            self.normal_descriptors.append(var)
               
        elif dist == 'lognormal':
            check_set(std_dev)
            check_set(mean)
            self.lognormal_means.append(str(mean))
            self.lognormal_std_devs.append(str(std_dev))
            self.lognormal_descriptors.append(descriptor)
               
        elif dist == 'exponential':
            check_set(beta)
            check_set(descriptor)
            self.exponential_betas.append(str(beta))
            self.exponential_descriptors.append(descriptor)

        elif dist == 'beta':
            check_set(beta)
            check_set(alpha)
            check_set(lower_bounds)
            check_set(upper_bounds)

            self.beta_betas.append(str(beta))
            self.beta_alphas.append(str(alpha))
            self.beta_descriptors.append(var)
            self.beta_lower_bounds.append(str(lower_bounds))
            self.beta_upper_bounds.append(str(upper_bounds))
            
        elif dist == "gamma":
            check_set(beta)
            check_set(alpha)

            self.gamma_alphas.append(str(alpha))
            self.gamma_betas.append(str(beta))
            self.gamma_descriptors.append(var)

        elif dist == "weibull":
            check_set(beta)
            check_set(alpha)

            self.weibull_alphas.append(str(alpha))
            self.weibull_betas.append(str(beta))
            self.weibull_descriptors.append(var)
       
        else: 
            raise ValueError(str(dist)+" is not a defined distribution")

        self.special_distribution_variables.append(var)

################################################################################
########################## Hierarchical Driver ################################
class pydakdriver(DakotaBase):
    #implements(IOptimizer) # Not sure what this does

    def __init__(self, name=None):
        super(pydakdriver, self).__init__()
        self.input.method = collections.OrderedDict()
        self.input.responses = collections.OrderedDict()

        # default definitions for set_variables
        self.need_start = False 
        self.uniform=False
        self.need_bounds=True
 
        if name: self.name = name
        else: self.name = 'dakota_'+str(id(self))

    # How DAKOTA input file options are set:
    #    1. user sets options either using function calls or setting objects 
    #            The design allows both options to have the same effect
    #            (These functions are shown below)
    #    2. When the driver is run(), the self.input objects are searched and used to build input file
    #            (This code is above)
    #    - The items in self.input are stored as orderedDict so order matters
    #    - If an object has '' as it's value, There is no corresponding value it is just a command (eg. no_gradients)
    #    - If an object has _SET_AT_RUNTIME as it's value, then the user must set this. _SET_AT_RUNTIME is a placeholder
    #      which allows the value to be set until runtime. if a key is associated with a value besides '' 
    #      or _SET_AT_RUNTIME. the value is effectively hardwired.


    def analytical_gradients(self):
         self.interval_type = 'forward'
         self.fd_gradient_step_size = '1.e-4'
         for key in self.input.responses:
             if key == 'no_gradients':
                  self.input.responses.pop(key)
         self.input.responses['numerical_gradients'] = ''
         self.input.responses['method_source dakota'] = ''
         self.input.responses['interval_type'] = ''
         self.input.responses['fd_gradient_step_size'] = self.fd_gradient_step_size

    def numerical_gradients(self, method_source='dakota'):
         for key in self.input.responses:
             if key == 'no_gradients': self.input.responses.pop(key)
         self.input.responses['numerical_gradients'] = ''
         if method_source=='dakota':self.input.responses['method_source dakota']=''
         self.fd_gradient_step_size = '1e-5'
         self.interval_type = 'forward'
         self.input.responses['interval_type'] = ''
         self.input.responses['fd_gradient_step_size'] = self.fd_gradient_step_size

    def hessians(self):
         self.input.responses['numerical_hessians']=''
         for key in self.input.responses:
             if key == 'no_hessians':
                  self.input.responses.pop(key)
         # todo: Create Hessian default with options

    def Optimization(self,opt_type='optpp_newton', interval_type = 'forward', surrogate_model=False):
        self.convergence_tolerance = '1.e-8'
        self.seed = _SET_AT_RUNTIME
        self.max_iterations = '200'
        self.max_function_evaluations = '2000'
        self.fd_gradient_step_size = 1e-6

        self.input.responses['objective_functions']=_SET_AT_RUNTIME
        self.input.responses['no_gradients'] = ''
        self.input.responses['no_hessians'] = ''
        if opt_type == 'optpp_newton':
            self.need_start=True
            self.need_bounds=True
            self.input.method[opt_type] = ""
            self.analytical_gradients()
            self.hessians()
        if opt_type == 'efficient_global':
            self.input.method["efficient_global"] = ""
            self.input.method["seed"] = _SET_AT_RUNTIME
            self.numerical_gradients()
        if opt_type == 'conmin':
            self.need_start=True           

            self.input.method["conmin"] = ''
            self.input.method["output"] = ''
            self.input.method['constraint_tolerance'] = '1.e-8'

            self.input.responses['nonlinear_inequality_constraints'] = _SET_AT_RUNTIME
            self.numerical_gradients()
            
        if surrogate_model: 
            self.input.method["surrogate"] = _SET_AT_RUNTIME
            self.samples = 100
            #self.input.interface.append("\tfork")

    def Parameter_Study(self,study_type = 'vector'):
        self.study_type = study_type
        if study_type == 'vector':
            self.need_start=True
            self.need_bounds=False
            # why was this false? legacy was self.set_variables(need_start=False, need_bounds=False)
            self.input.method['vector_parameter_study'] = ""
            self.input.method['final_point'] = _SET_AT_RUNTIME 
            self.input.method['num_steps'] = _SET_AT_RUNTIME 
            self.final_point = _SET_AT_RUNTIME
            self.num_steps = _SET_AT_RUNTIME
        if study_type == 'multi-dim':
            self.need_start=False
            self.input.method['multidim_parameter_study'] = ""
            self.input.method['partitions'] = _SET_AT_RUNTIME 
            self.partitions =  _SET_AT_RUNTIME
        if study_type == 'list':
            self.input.method['list_parameter_study'] = ""
            self.input.method['list_of_points'] = _SET_AT_RUNTIME 
            self.input.responses['response_functions']=_SET_AT_RUNTIME
        else: self.input.responses['objective_functions']=_SET_AT_RUNTIME 
        if study_type == 'centered':
            self.input.method['centered_parameter_study'] = ""
            self.input.method['step_vector'] = _SET_AT_RUNTIME
            self.input.method['steps_per_variable'] = _SET_AT_RUNTIME
        self.input.responses['no_gradients']=''
        self.input.responses['no_hessians']=''

    def UQ(self,UQ_type = 'sampling'):
            self.sample_type =  'random' #'lhs'
            #self.seed = _SET_AT_RUNTIME
            self.samples=100
            
            if UQ_type == 'sampling':
                self.need_start = False
                self.uniform = True
                self.input.method = collections.OrderedDict()
                self.input.method['sampling'] = ''
                self.input.method['output'] = _SET_AT_RUNTIME
                self.input.method['sample_type'] = _SET_AT_RUNTIME
                #self.input.method['seed'] = _SET_AT_RUNTIME
                self.input.method['samples'] = _SET_AT_RUNTIME
        
                self.input.responses = collections.OrderedDict()
                self.input.responses['num_response_functions'] = _SET_AT_RUNTIME
                self.input.responses['response_descriptors'] = _SET_AT_RUNTIME
            self.input.responses['no_gradients'] = ''
            self.input.responses['no_hessians'] = ''
################################################################################
