# ---------------------------------------------------------------------------------
# Pyomo Solver Wrapper
# Language - Python
# https://github.com/judejeh/PyomoSolverWrapper
# Licensed under MIT license
# Copyright 2021 The Pyomo Solver Wrapper authors <https://github.com/judejeh>
# ---------------------------------------------------------------------------------

import re, sys, platform, textwrap, subprocess
from os import path, makedirs, system, environ
from datetime import datetime
from itertools import chain

from pyomo import version as pyoversion
from numpy import array, zeros
from pyomo.opt import SolverFactory, SolverManagerFactory


class SolverWrapper:
    class __SolversInfo:
        """
        Default info for solvers
        """

        def __init__(self):
            self.configured_solvers = {
                # solver_name: ['windows executable', 'unix executable', 'solver_io']
                'cbc': ['cbc', 'cbc', 'lp'],
                'cplex': ['cplex', 'cplex', 'lp'],
                'glpk': ['glpk', 'glpk', 'lp'],
                'gurobi': ['gurobi', 'gurobi.sh', 'python'],    # Configure gurobi to use python api?
                'baron': ['baron', 'baron', 'nl'],
                'ipopt': ['ipopt', 'ipopt', 'lp']
                }
            self.neos_compatible = ['bonmin', 'cbc', 'conopt', 'couenne', 'cplex', 'filmint', 'filter', 'ipopt',
                                    'knitro', 'l-bfgs-b', 'lancelot', 'lgo', 'loqo', 'minlp', 'minos', 'minto',
                                    'mosek', 'ooqp', 'path', 'raposa', 'snopt']

    class __Constants:
        """
        Default constants for use throughout the class
        """

        def __init__(self):
            self.var_defaults = {
                'solver_name': 'gurobi',
                'solver_path': False,
                'time_limit': 1200,
                'threads': 2,
                'neos': False,
                'verbosity': False,
                'debug_mode': False,
                'solver_progress': True,
                'write_solution': True,
                'write_solution_to_stdout': True,
                'return_solution': True,
                'rel_gap': 0.0,
                'result_precision': 6
            }
            self.var_types = {
                'time_limit': [int, float],
                'threads': [int],
                'neos': [bool],
                'verbosity': [bool],
                'debug_mode': [bool],
                'solver_progress': [bool],
                'write_solution': [bool],
                'write_solution_to_stdout': [bool],
                'return_solution': [bool],
                'rel_gap': [int, float],
                'result_precision': [int]
            }
            self.os_name = platform.system()

    def __init__(self, solver_name=None, solver_path=None, time_limit=None, threads=None, neos=None, verbosity=None,
                 debug_mode=None, solver_progress=None, write_solution=None, write_solution_to_stdout=None,
                 return_solution=None, rel_gap=None, result_precision=None, neos_registered_email=None):
        # Set methods defaults
        self.solver_info = self.__SolversInfo()
        self.constants = self.__Constants()
        self.solver_name = self.__apattr(solver_name, self.constants.var_defaults['solver_name'])
        self.solver_path = self.__apattr(solver_path, self.constants.var_defaults['solver_path'])

        # Set solver defaults
        self.time_limit = self.__apattr(time_limit, self.constants.var_defaults['time_limit'])
        self.threads = self.__apattr(threads, self.constants.var_defaults['threads'])
        self.neos = self.__apattr(neos, self.constants.var_defaults['neos'])
        self.verbosity = self.__apattr(verbosity, self.constants.var_defaults['verbosity'])
        self.debug_mode = self.__apattr(debug_mode, self.constants.var_defaults['debug_mode'])
        self.verbose_debug_mode = False
        self.solver_progress = self.__apattr(solver_progress, self.constants.var_defaults['solver_progress'])
        self.write_solution = self.__apattr(write_solution, self.constants.var_defaults['write_solution'])
        self.write_solution_to_stdout = self.__apattr(write_solution_to_stdout,
                                                     self.constants.var_defaults['write_solution_to_stdout'])
        self.return_solution = self.__apattr(return_solution, self.constants.var_defaults['return_solution'])
        self.rel_gap = self.__apattr(rel_gap, self.constants.var_defaults['rel_gap'])
        self.result_precision = self.__apattr(result_precision, self.constants.var_defaults['result_precision'])

        # Set other defaults
        self.__model_name_str = None
        self.__current_datetime_str = None
        self.__pyomo_version = pyoversion.version
        self.__pyutilib_version = self.__get_pkg_version("PyUtilib")
        self.__dependency_check_count = 1
        self.__dependency_check = self.__pyutilib_dependency_check()
        self.__DEF_registered_email = "pyosolvewrapper@gmail.com"
        self.__DEF_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.neos_registered_email = neos_registered_email
        self.neos_registered_email = self.__sg_neos_registered_email(defaults=False)

    def __sg_neos_registered_email(self, defaults=True):
        if self.neos_registered_email is None:
            if defaults:
                self.__msg("NB: An email address can be passed to 'neos_registered_email' instead of using defaults.")
            self.neos_registered_email = self.__DEF_registered_email
            return self.neos_registered_email
        else:
            if re.match(self.__DEF_REGEX, self.neos_registered_email):
                return self.neos_registered_email
            else:
                self.__msg("An invalid email address was passed. Resorting to default...")
                self.neos_registered_email = None
                self.__sg_neos_registered_email(defaults=False)   # Tiny way to avoid repeated warnings to users
                return self.neos_registered_email


    def __apattr(self, attrib, value):
        """
        Set value to an attrib
        :param attrib:
        :param value:
        :return: None
        """
        if attrib is None:
            return value
        else:
            return attrib

    def __chkattrt(self, attrib):
        """
        Check type of attrib against requirement and set to default else
        :param attrib:
        :return: None
        """

        # Get attrib name as string
        c_var_list = [vars for vars in locals().keys() if "_" not in vars[:2]]

        attrib_str = None
        for var_l in c_var_list:
            if id(attrib) == id(var_l):
                attrib_str = var_l
                break
            else:
                pass

        if attrib_str is not None and attrib_str in self.constants.var_types.keys():
            if type(attrib) in self.constants.var_types[attrib_str]:
                pass
            else:
                self.__psmsg('Value given to ' + attrib_str + ' is invalid',
                            'The following value types are acceptable: ' + str(self.constants.var_types[attrib_str]))
                self.__psmsg('Setting default value. . .')
                setattr(self, attrib_str, self.constants.var_defaults[attrib_str])
        else:
            pass

    def __msg(self, *msg, text_indent=4):
        text_indent = " " * int(text_indent)
        # Text wrapper function
        wrapper = textwrap.TextWrapper(width=60, initial_indent=text_indent, subsequent_indent=text_indent)
        # Print message
        print("\n")
        for message in msg:
            message = wrapper.wrap(text=str(message))
            for element in message:
                print(element)
        return text_indent

    def __pemsg(self, *msg, exit=True):
        """
        Custom error messages to print to stdout and stop execution
        :param message: Error message to be printed
        """

        if self.debug_mode:
            exit = self.debug_mode
        else:
            exit = exit

        text_indent = self.__msg(*msg, text_indent=4)

        if exit:  # Stop run
            print(text_indent + "Exiting . . .")
            sys.exit(1)
        else:
            pass

    def __psmsg(self, *msg):
        """
        Custom status messages to print to stdout and stop execution
        :param message: Error message to be printed
        """
        if self.verbosity:
            self.__msg("INFO:", *msg, text_indent=1)
        else:
            pass

    def __page_borders(self, bottom=False):
        if self.verbosity:
            if bottom:
                self.__msg("- - " * 15, "\n", text_indent=0)
            else:
                self.__msg("- - " * 15, text_indent=0)
        else:
            pass

    def __run_ext_command(self, cmd=[" "]):
        return subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]

    def __get_solver_path(self, solver_exct):
        if self.constants.os_name == 'Windows':
            path = self.__run_ext_command(['where', solver_exct]).decode('utf-8')
        else:
            path = self.__run_ext_command(['which', solver_exct]).decode('utf-8')

        if path != '':
            return path, True
        else:
            return None, False

    def __get_pkg_version(self, pkg):
        parent_cmd = subprocess.Popen(['pip', 'freeze'], stdout=subprocess.PIPE)
        if self.constants.os_name == 'Windows':
            value = subprocess.check_output(['findstr', str(pkg)], stdin=parent_cmd.stdout).decode('utf-8')
        else:
            value = subprocess.check_output(['grep', str(pkg)], stdin=parent_cmd.stdout).decode('utf-8')

        if value == '':
            return value
        elif len(value) > len(pkg):
            value = value[len(pkg)+2:]
            return value
        else:
            return None

    def __pyutilib_dependency_check(self):
        if self.__dependency_check_count <= 3:
            if self.__pyomo_version == '5.6.8':
                if '5.7.3' not in self.__pyutilib_version:
                    system('pip install pyutilib==5.7.3')
                    self.__pyutilib_version = self.__get_pkg_version("PyUtilib")
                    self.__dependency_check_count += 1
                    self.__pyutilib_dependency_check()
                else:
                    self.__dependency_check = True
            elif self.__pyomo_version == '5.6.9':
                if '5.8.0' not in self.__pyutilib_version:
                    system('pip install pyutilib==5.8.0')
                    self.__pyutilib_version = self.__get_pkg_version("PyUtilib")
                    self.__dependency_check_count += 1
                    self.__pyutilib_dependency_check()
                else:
                    self.__dependency_check = True
            elif self.__pyomo_version > '5.6.9':
                if '6.0.0' not in self.__pyutilib_version:
                    system('pip install pyutilib==6.0.0')
                    self.__pyutilib_version = self.__get_pkg_version("PyUtilib")
                    self.__dependency_check_count += 1
                    self.__pyutilib_dependency_check()
                else:
                    self.__dependency_check = True
        else:
            self.__pemsg("The right version of pyutilib could not be installed.")

    def __set_tempdir(self, folder):
        """
        Set temporary solver folder
        :param folder: path to folder
        :return:
        """
        if self.__pyomo_version <= '5.7.1':
            from pyutilib.services import TempfileManager
            TempfileManager.tempdir = folder
        else:
            from pyomo.common.tempfiles import TempfileManager
            TempfileManager.tempdir = folder

    def __get_set_list(self, pyo_obj):
        if self.__pyomo_version < '5.7.0'[:len(self.__pyomo_version)]:
            return [set for set in pyo_obj._index.set_tuple]
        else:
            return [set.name for set in pyo_obj._index.subsets()]

    def __get_set_list_alt(self, pyo_obj):
        if self.__pyomo_version < '5.7.0'[:len(self.__pyomo_version)]:
            return [set for set in pyo_obj.domain.set_tuple]
        else:
            return [set.name for set in pyo_obj.domain.subsets()]

    def __solvers_compatibility_check(self):
        """
        NEOS vs local solvers: Check solvers are recognised/available
        :return:
        """
        if self.neos:  # using NEOS
            if self.solver_name not in self.solver_info.neos_compatible:
                self.__pemsg("NEOS server does not seem to be configure for " + str(self.solver_name),
                             "If you must used this solver, install a local version and set option 'neos' to 'False'",
                             exit=False)
                self.__pemsg("Attempting to use a local solver instance . . .", exit=False)
                self.neos = False
                self.__solvers_path_check()
                self.__solvers_compatibility_check()
            else:
                environ['NEOS_EMAIL'] = self.__sg_neos_registered_email()
                if self.verbosity:
                    self.__psmsg("Using NEOS server to solve model . . .")
                else:
                    pass
        else:  # using a locally installed solver
            if self.solver_name not in self.solver_info.configured_solvers.keys():
                self.__pemsg(self.solver_name + " is not amongst those currently configured by this package")
            elif not self.solver_avail:
                self.__pemsg(self.solver_name + " is not installed or at the path specified")
            else:
                if self.verbosity:
                    self.__psmsg("Solver located in {}".format(self.solver_path))
                else:
                    pass

    def __solvers_path_check(self):
        """
        # Confirm solver paths and thus availability
        :return:
        """
        if not self.neos:
            if self.solver_path is False:
                if self.constants.os_name == 'Windows':
                    self.solver_path, self.solver_avail = \
                        self.__get_solver_path(self.solver_info.configured_solvers[self.solver_name][0])
                else:
                    self.solver_path, self.solver_avail = \
                        self.__get_solver_path(self.solver_info.configured_solvers[self.solver_name][1])
            else:
                self.solver_avail = path.exists(self.solver_path)
        else:
            self.solver_avail = False


    def solve_model(self, model):
        """
        Method to solve an optimization model using a specified solver
        Returns:
            :dict - solver statistics and model solution
        """

        # Set a few default values
        attr_to_check = ['solver_name', 'neos', 'write_solution', 'return_solution', 'verbosity', 'solver_progress']
        for attr in attr_to_check:
            self.__chkattrt(attr)

        # Get model name
        model_name = model.name
        self.__model_name_str = str(re.sub(" ", "_", model_name))

        # Solver name to lower case characters
        self.solver_name = self.solver_name.lower()

        # NEOS vs local solvers: check solvers are recognised/available
        self.__solvers_compatibility_check()   # Run solvers check

        # Confirm solver paths and thus availability
        self.__solvers_path_check

        # Call solver factory
        opt_solver = SolverFactory(self.solver_name)

        # Change solver temporary folder path
        log_folder = path.join('_log', '')
        if self.solver_name in ['gurobi', 'baron', 'cplex']:
            if not path.exists(log_folder):
                makedirs(log_folder)
            self.__set_tempdir(log_folder)
        else:
            pass

        # Include solver-compatible options
        if self.solver_name in ['cplex', 'gurobi']:
            opt_solver.options['threads'] = self.threads
            opt_solver.options['mipgap'] = self.rel_gap
            opt_solver.options['timelimit'] = self.time_limit
        elif self.solver_name in ['baron']:
            opt_solver.options['threads'] = self.threads
            opt_solver.options['LPSol'] = 3
            opt_solver.options['EpsR'] = self.rel_gap
            opt_solver.options['MaxTime'] = self.time_limit
            # For Unix systems ensure "libcplexxxxx.dylib" is in the system PATH for baron to use CPLEX for MIPs
        elif self.solver_name in ['cbc']:
            opt_solver.options['threads'] = self.threads
            opt_solver.options['ratio'] = self.rel_gap
            opt_solver.options['seconds'] = self.time_limit
            opt_solver.options['log'] = int(self.solver_progress) * 2
            opt_solver.options['mess'] = 'on'
            opt_solver.options['timeM'] = "elapsed"
            opt_solver.options['preprocess'] = "equal"
        elif self.solver_name in ['glpk']:
            opt_solver.options['mipgap'] = self.rel_gap
            opt_solver.options['tmlim'] = self.time_limit
        else:
            pass

        # Write log to file named <model_name>/DD_MM_YY_HH_MM_xx.log
        # Create (if it does not exist) the '_log' folder
        log_store_folder = path.join(log_folder, self.__model_name_str, '')
        if not path.exists(log_store_folder):
            makedirs(log_store_folder)

        self.__current_datetime_str = datetime.now().strftime("%d_%m_%y_%H_%M_")
        file_suffix = 0
        # Results filename
        if self.__model_name_str == 'Unknown' or len(self.__model_name_str) <= 10:
            log_filename = self.__model_name_str + self.__current_datetime_str + str(file_suffix) + ".log"
        else:
            log_filename = self.__model_name_str[:4] + '..' + self.__model_name_str[-4:] + \
                           self.__current_datetime_str + str(file_suffix) + ".log"
        while path.exists(log_store_folder + log_filename):
            file_suffix += 1
            log_filename = self.__current_datetime_str + str(file_suffix) + ".log"
        log_filename = self.__current_datetime_str + str(file_suffix) + ".log"

        # Solve <model> with/without writing final solution to stdout
        processed_results = None
        self.__page_borders()   # Headers for page
        try:
            if self.neos:
                self.solver_results = SolverManagerFactory('neos').solve(model, opt=opt_solver,
                                                                         tee=self.solver_progress)
            else:
                self.solver_results = opt_solver.solve(model, tee=self.solver_progress,
                                                       logfile=log_store_folder + log_filename)

            # Process results obtained
            if self.neos:
                self.__msg("Currently, results are not post-processed for NEOS server runs")  #FIXME Find a work around
            else:
                processed_results = self._process_solver_results(model)
        except ValueError:
            self.__psmsg("Something went wrong with the solver")

        # Return model solution and solver statistics
        self.final_results = processed_results

        self.__page_borders(bottom=True)   # Footer for page

    # Method for post processing solver results
    def _process_solver_results(self, model):
        """
        Method to post process results from 'solve_model' method
        :param model: solved model
        :return: dictionary of solver results
        """

        from pyomo.environ import Set, RealSet, RangeSet, Param, Var

        # Write solution to stdout/file
        if self.write_solution and (str(self.solver_results.solver.Status) in ['ok']
                               or str(self.solver_results.solver.Termination_condition) in ['maxTimeLimit']):

            if self.write_solution_to_stdout:
                # Write solution to screen
                self.solver_results.write()
            else:
                pass

            # Write solution to file named <model_name>/DD_MM_YY_HH_MM_xx.json
            # Create (if it does not exist) the '_results_store' folder
            results_store_folder = path.join('_results_store', self.__model_name_str, '')
            if not path.exists(results_store_folder):
                makedirs(results_store_folder)

            if self.__pyomo_version <= '5.7.1':
                model.solutions.store_to(self.solver_results)  # define solutions storage folder
            else:
                pass
            self.__current_datetime_str = datetime.now().strftime("%d_%m_%y_%H_%M_")
            file_suffix = 0
            # Results filename
            if self.__model_name_str == 'Unknown' or len(self.__model_name_str) <= 10:
                result_filename = self.__model_name_str + self.__current_datetime_str + str(
                    file_suffix) + ".json"
            else:
                result_filename = self.__model_name_str[:4] + '..' + self.__model_name_str[-4:] + \
                                  self.__current_datetime_str + str(file_suffix) + ".json"
            while path.exists(results_store_folder + result_filename):
                file_suffix += 1
                result_filename = self.__current_datetime_str + str(file_suffix) + ".json"
            result_filename = self.__current_datetime_str + str(file_suffix) + ".json"
            self.solver_results.write(filename=results_store_folder + result_filename, format="json")
        else:
            pass

        # Create dictionary to for solver statistics and solution
        final_result = dict()
        # Include the default solver results from opt_solver.solve & current state of model
        final_result['solver_results_def'] = self.solver_results
        final_result['model'] = model  # copy.deepcopy(model)   # include all model attributes

        # _include solver statistics
        acceptable_termination_conditions = ['maxTimeLimit', 'maxIterations', 'locallyOptimal', 'globallyOptimal',
                                             'optimal', 'other']
        if str(self.solver_results.solver.Status) == 'ok' or (
                str(self.solver_results.solver.Status) == 'aborted' and
                str(self.solver_results.solver.Termination_condition)
                in acceptable_termination_conditions):
            final_result['solver'] = dict()  # Create dictionary for solver statistics
            final_result['solver'] = {
                'status': str(self.solver_results.solver.Status),
                'solver_message': str(self.solver_results.solver.Message),
                'termination_condition': str(self.solver_results.solver.Termination_condition)
            }
            try:
                final_result['solver']['wall_time'] = self.solver_results.solver.wall_time
            except AttributeError:
                final_result['solver']['wall_time'] = None

            try:
                final_result['solver']['wall_time'] = self.solver_results.solver.wall_time
            except AttributeError:
                final_result['solver']['wall_time'] = None

            try:
                final_result['solver']['cpu_time'] = self.solver_results.solver.time
            except AttributeError:
                final_result['solver']['cpu_time'] = None

            try:
                final_result['solver']['gap'] = round(
                    (self.solver_results.problem.Upper_bound - self.solver_results.problem.Lower_bound) \
                    * 100 / self.solver_results.problem.Upper_bound, 2)
            except AttributeError:
                final_result['solver']['gap'] = None

            # Check state of available solution
            if self.__pyomo_version < '5.7.1':
                try:
                    for key, value in final_result['solver_results_def']['Solution'][0]['Objective'].items():
                        objective_value = value['Value']
                    final_result['solution_status'] = True
                except:
                    final_result['solution_status'] = False
                    objective_value = 'Unk'
            else:
                try:
                    objective_value = model.solutions.solutions[0]._entry['objective'][
                        list(model.solutions.solutions[0]._entry['objective'].keys())[0]][1]['Value']
                    final_result['solution_status'] = True
                except:
                    final_result['solution_status'] = False
                    objective_value = 'Unk'

            if self.return_solution and final_result['solution_status']:
                # True: include values of all model objects in 'final_result'
                # write list of sets, parameters and variables
                final_result['sets_list'] = [str(i) for i in chain(model.component_objects(Set),
                                                                   model.component_objects(RealSet),
                                                                   model.component_objects(RangeSet))
                                             if (re.split("_", str(i))[-1] != 'index')
                                             if (re.split("_", str(i))[-1] != 'domain')]
                final_result['parameters_list'] = [str(i) for i in model.component_objects(Param)]
                final_result['variables_list'] = [str(i) for i in model.component_objects(Var)]

                # Populate final results for sets, parameters and variables
                # Create method to return array
                def indexed_value_extract(index, object):
                    return array([value for value in object[index].value])

                # Sets
                final_result['sets'] = dict()
                for set in final_result['sets_list']:
                    set_object = getattr(model, str(set))
                    final_result['sets'][set] = array(list(set_object))  # save array of set elements

                # Parameters
                final_result['parameters'] = dict()
                if self.verbosity:
                    print('\nProcessing parameters . . . ')
                else:
                    pass
                for par in final_result['parameters_list']:
                    if self.verbosity:
                        print(par, ' ', end="")
                    else:
                        pass
                    par_object = getattr(model, str(par))
                    par_object_dim = par_object.dim()  # get dimension of parameter
                    if par_object_dim == 0:
                        final_result['parameters'][par] = par_object.value
                    elif par_object_dim == 1:
                        final_result['parameters'][par] = array([value for value in par_object.values()])
                    else:
                        try:
                            par_set_list = self.__get_set_list(par_object)
                            par_set_lens = [len(getattr(model, str(set))) for set in par_set_list]
                        except AttributeError:
                            par_set_list = [str(par_object._index.name)]
                            temp_par_set = getattr(model, par_set_list[0])
                            if temp_par_set.dimen == 1:
                                par_set_lens = [len(temp_par_set)]
                            else:
                                par_set_lens = [len(set) for set in self.__get_set_list_alt(temp_par_set)]

                        # print(par_set_lens)
                        final_result['parameters'][par] = zeros(shape=par_set_lens, dtype=float)
                        if par_object_dim == 2:
                            if len(par_set_list) == par_object_dim:
                                for ind_i, i in enumerate(getattr(model, str(par_set_list[0]))):
                                    for ind_j, j in enumerate(getattr(model, str(par_set_list[1]))):
                                        final_result['parameters'][par][ind_i][ind_j] = par_object[i, j]
                            elif len(par_set_list) == 1:
                                for set in par_set_list:
                                    for ind, (i, j) in enumerate(getattr(model, str(set))):
                                        # print(type(final_result['parameters'][par]),final_result['parameters'][par])
                                        # print(i,j,final_result['parameters'][par][i-1][j-1])
                                        # print(par_set_lens)
                                        # print(par_set_list)
                                        # print(par_object_dim)
                                        if self.__pyomo_version < '5.7':  # FIXME: Better way to do this?
                                            final_result['parameters'][par][i - 1][j - 1] = par_object[i, j]
                                        else:
                                            final_result['parameters'][par][ind] = par_object[i, j]
                            else:
                                pass

                        else:
                            pass  # FIXME 3-dimensional variables are not considered yet

                # Variables
                final_result['variables'] = dict()
                # Include objective functionv value
                final_result['variables']['Objective'] = objective_value
                if self.verbosity:
                    print('\nProcessing results of variables . . . ')
                else:
                    pass
                for variable in final_result['variables_list']:
                    try:
                        if self.verbosity:
                            print(variable, ' ', end="")
                        else:
                            pass
                        variable_object = getattr(model, str(variable))
                        variable_object_dim = variable_object.dim()  # get dimension of variable
                        if variable_object_dim == 0:
                            final_result['variables'][variable] = variable_object.value
                        elif variable_object_dim == 1:
                            final_result['variables'][variable] = array([value.value for value in variable_object.values()])
                        else:
                            try:
                                variable_set_list = self.__get_set_list(variable_object)
                                variable_set_lens = [len(getattr(model, str(set))) for set in variable_set_list]
                            except AttributeError:
                                variable_set_list = [str(variable_object._index.name)]
                                temp_variable_set = getattr(model, variable_set_list[0])
                                if temp_variable_set.dimen == 1:
                                    variable_set_lens = [len(temp_variable_set)]
                                else:
                                    variable_set_lens = [len(set) for set in self.__get_set_list_alt(temp_variable_set)]

                            # print(variable_set_lens)
                            final_result['variables'][variable] = zeros(shape=variable_set_lens, dtype=float)
                            if variable_object_dim == 2:
                                if len(variable_set_list) == variable_object_dim:
                                    for ind_i, i in enumerate(getattr(model, str(variable_set_list[0]))):
                                        for ind_j, j in enumerate(
                                                getattr(model, str(variable_set_list[1]))):
                                            final_result['variables'][variable][ind_i][ind_j] = variable_object[i, j].value
                                elif len(variable_set_list) == 1:
                                    for set in variable_set_list:
                                        for ind, (i, j) in enumerate(getattr(model, str(set))):
                                            # print(type(final_result['variables'][variable]),final_result['variables'][variable])
                                            # print(i,j,final_result['variables'][variable][i-1][j-1])
                                            if self.__pyomo_version < '5.7':  # FIXME: Better way to do this?
                                                final_result['variables'][variable][i - 1][j - 1] = variable_object[i, j].value
                                            else:
                                                final_result['variables'][variable][ind] = variable_object[i, j].value
                                else:
                                    pass

                            else:
                                pass  # FIXME 3-dimensional variables are not considered yet

                    except AttributeError:
                        pass

                print('\n')

            else:  # if solver_status != 'ok' or amongst acceptable termination conditions
                if self.debug_mode:
                    if self.verbose_debug_mode:  # Print troublesome constraints
                        from pyomo.util.infeasible import log_infeasible_constraints
                        log_infeasible_constraints(model)
                    else:
                        pass
                    self.__psmsg('An optimal solution could not be processed')  # leave program running to debug
                else:
                    self.__pemsg('An optimal solution could not be processed')

        else:  # if solver_status != ok
            self.__pemsg('An optimal solution was NOT found')

        return final_result


if __name__ == '__main__':
    print('This is a wrapper for solving Pyomo models, auto-processing variables and parameters')
