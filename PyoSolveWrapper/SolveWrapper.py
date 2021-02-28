# ---------------------------------------------------------------------------------
# Pyomo Solver Wrapper
# Language - Python
# https://github.com/judejeh/PyomoSolverWrapper
# Licensed under MIT license
# Copyright 2021 The Pyomo Solver Wrapper authors <https://github.com/judejeh>
# ---------------------------------------------------------------------------------

import os, re, sys, copy, platform, textwrap, subprocess
from datetime import datetime
from itertools import chain

import pyomo.environ as env
from numpy import array, zeros
from pyomo.opt import SolverFactory, SolverManagerFactory
import pyutilib.services


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
                'verbosity': True,
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
                 return_solution=None, rel_gap=None, result_precision=None):
        # Set methods defaults
        self.solver_info = self.__SolversInfo()
        self.constants = self.__Constants()
        self.solver_name = self._apattr(solver_name, self.constants.var_defaults['solver_name'])
        self.solver_path = self._apattr(solver_path, self.constants.var_defaults['solver_path'])

        # Set solver defaults
        self.time_limit = self._apattr(time_limit, self.constants.var_defaults['time_limit'])
        self.threads = self._apattr(threads, self.constants.var_defaults['threads'])
        self.neos = self._apattr(neos, self.constants.var_defaults['neos'])
        self.verbosity = self._apattr(verbosity, self.constants.var_defaults['verbosity'])
        self.debug_mode = self._apattr(debug_mode, self.constants.var_defaults['debug_mode'])
        self.verbose_debug_mode = False
        self.solver_progress = self._apattr(solver_progress, self.constants.var_defaults['solver_progress'])
        self.write_solution = self._apattr(write_solution, self.constants.var_defaults['write_solution'])
        self.write_solution_to_stdout = self._apattr(write_solution_to_stdout,
                                                     self.constants.var_defaults['write_solution_to_stdout'])
        self.return_solution = self._apattr(return_solution, self.constants.var_defaults['return_solution'])
        self.rel_gap = self._apattr(rel_gap, self.constants.var_defaults['rel_gap'])
        self.result_precision = self._apattr(result_precision, self.constants.var_defaults['result_precision'])

        # Set other defaults
        self.model_name_str = None
        self.current_datetime_str = None

    def _apattr(self, attrib, value):
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

    def _chkattrt(self, attrib):
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
                self._psmsg('Value given to ' + attrib_str + ' is invalid',
                            'The following value types are acceptable: ' + str(self.constants.var_types[attrib_str]))
                self._psmsg('Setting default value. . .')
                setattr(self, attrib_str, self.constants.var_defaults[attrib_str])
        else:
            pass

    def __msg(self, *msg, text_indent):
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

    def _pemsg(self, *msg, exit=True):
        """
        Custom error messages to print to stdout and stop execution
        :param message: Error message to be printed
        """

        text_indent = self.__msg(*msg, text_indent=4)

        if exit:  # Stop run
            print(text_indent + "Exiting . . .")
            sys.exit(1)
        else:
            pass

    def _psmsg(self, *msg):
        """
        Custom status messages to print to stdout and stop execution
        :param message: Error message to be printed
        """
        text_indent = self.__msg(*msg, text_indent=1)

    def _run_ext_command(self, cmd=[" "]):
        return subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]

    def _get_solver_path(self, solver_exct):
        if self.constants.os_name == 'Windows':
            path = self._run_ext_command(['where', solver_exct]).decode('utf-8')
        else:
            path = self._run_ext_command(['which', solver_exct]).decode('utf-8')

        if path != '':
            return path, True
        else:
            return None, False

    def solve_model(self, model):
        """
        Method to solve an optimization model using a specified solver
        Returns:
            :dict - solver statistics and model solution
        """
        # Set a few default values
        vars_to_check = ['solver_name', 'neos', 'write_solution', 'return_solution', 'verbosity', 'solver_progress']
        for var in vars_to_check:
            self._chkattrt(var)

        # Get model name
        model_name = model.name
        self.model_name_str = str(re.sub(" ", "_", model_name))

        # Solver name to lower case characters
        self.solver_name = self.solver_name.lower()

        # Confirm solver paths and thus availability
        if self.solver_path is False:
            if self.constants.os_name == 'Windows':
                self.solver_path, self.solver_avail = \
                    self._get_solver_path(self.solver_info.configured_solvers[self.solver_name][0])
            else:
                self.solver_path, self.solver_avail = \
                    self._get_solver_path(self.solver_info.configured_solvers[self.solver_name][1])
        else:
            self.solver_avail = os.path.exists(self.solver_path)

        # NEOS vs local solvers: check solvers are recognised/available
        if self.neos:  # using NEOS
            if self.solver_name not in self.solver_info.neos_compatible:
                self._pemsg("NEOS server does not seem to be configure for " + str(self.solver_name),
                            "If you must used this solver, install a local version and set option 'neos' to 'False'")
            else:
                if self.verbosity:
                    self._psmsg("Using NEOS server to solve model . . .")
                else:
                    pass
        else:  # using a locally installed solver
            if self.solver_name not in self.solver_info.configured_solvers.keys():
                self._pemsg(self.solver_name + " is not amongst those currently configured by this package")
            elif not self.solver_avail:
                self._pemsg(self.solver_name + " is not installed or at the path specified")
            else:
                if self.verbosity:
                    self._psmsg("Solver located in {}".format(self.solver_path))
                else:
                    pass

        # Call solver factory
        opt_solver = SolverFactory(self.solver_name)

        # Change solver temporary folder path
        log_folder = os.path.join('_log', '')
        if self.solver_name in ['gurobi', 'baron', 'cplex']:
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            pyutilib.services.TempfileManager.tempdir = log_folder
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
        log_store_folder = os.path.join(log_folder, self.model_name_str, '')
        if not os.path.exists(log_store_folder):
            os.makedirs(log_store_folder)

        self.current_datetime_str = datetime.now().strftime("%d_%m_%y_%H_%M_")
        file_suffix = 0
        # Results filename
        if self.model_name_str == 'Unknown' or len(self.model_name_str) <= 10:
            log_filename = self.model_name_str + self.current_datetime_str + str(file_suffix) + ".log"
        else:
            log_filename = self.model_name_str[:4] + '..' + self.model_name_str[-4:] + \
                           self.current_datetime_str + str(file_suffix) + ".log"
        while os.path.exists(log_store_folder + log_filename):
            file_suffix += 1
            log_filename = self.current_datetime_str + str(file_suffix) + ".log"
        log_filename = self.current_datetime_str + str(file_suffix) + ".log"

        # Solve <model> with/without writing final solution to stdout
        processed_results = None
        try:
            if self.neos:
                self.solver_results = SolverManagerFactory('neos').solve(model, opt=opt_solver,
                                                                         tee=self.solver_progress)
            else:
                self.solver_results = opt_solver.solve(model, tee=self.solver_progress,
                                                       logfile=log_store_folder + log_filename)

            # Process results obtained
            processed_results = self._process_solver_results(model)
        except ValueError:
            self._psmsg("Something went wrong with the solver")

        # Return model solution and solver statistics
        self.final_results = processed_results

    # Method for post processing solver results
    def _process_solver_results(self, model):
        """
        Method to post process results from 'solve_model' method
        :param model: solved model
        :return: dictionary of solver results
        """

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
            results_store_folder = os.path.join('_results_store', self.model_name_str, '')
            if not os.path.exists(results_store_folder):
                os.makedirs(results_store_folder)

            model.solutions.store_to(self.solver_results)  # define solutions storage folder
            self.current_datetime_str = datetime.now().strftime("%d_%m_%y_%H_%M_")
            file_suffix = 0
            # Results filename
            if self.model_name_str == 'Unknown' or len(self.model_name_str) <= 10:
                result_filename = self.model_name_str + self.current_datetime_str + str(
                    file_suffix) + ".json"
            else:
                result_filename = self.model_name_str[:4] + '..' + self.model_name_str[-4:] + \
                                  self.current_datetime_str + str(file_suffix) + ".json"
            while os.path.exists(results_store_folder + result_filename):
                file_suffix += 1
                result_filename = self.current_datetime_str + str(file_suffix) + ".json"
            result_filename = self.current_datetime_str + str(file_suffix) + ".json"
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
            try:
                for key, value in final_result['solver_results_def']['Solution'][0]['Objective'].items():
                    objective_value = value['Value']
                final_result['solution_status'] = True
            except:
                final_result['solution_status'] = False

            if self.return_solution and final_result['solution_status']:
                # True: include values of all model objects in 'final_result'
                # write list of sets, parameters and variables
                final_result['sets_list'] = [str(i) for i in chain(model.component_objects(env.Set),
                                                                   model.component_objects(env.RealSet),
                                                                   model.component_objects(
                                                                       env.RangeSet))
                                             if (re.split("_", str(i))[-1] != 'index')
                                             if (re.split("_", str(i))[-1] != 'domain')]
                final_result['parameters_list'] = [str(i) for i in model.component_objects(env.Param)]
                final_result['variables_list'] = [str(i) for i in model.component_objects(env.Var)]

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
                            par_set_list = [set for set in par_object._index.set_tuple]
                            par_set_lens = [len(getattr(model, str(set))) for set in par_set_list]
                        except AttributeError:
                            par_set_list = [str(par_object._index.name)]
                            temp_par_set = getattr(model, par_set_list[0])
                            if temp_par_set.dimen == 1:
                                par_set_lens = [len(temp_par_set)]
                            else:
                                par_set_lens = [len(set) for set in temp_par_set.domain.set_tuple]

                        # print(par_set_lens)
                        final_result['parameters'][par] = zeros(shape=par_set_lens, dtype=float)
                        if par_object_dim == 2:
                            if len(par_set_list) == par_object_dim:
                                for ind_i, i in enumerate(getattr(model, str(par_set_list[0]))):
                                    for ind_j, j in enumerate(getattr(model, str(par_set_list[1]))):
                                        final_result['parameters'][par][ind_i][ind_j] = par_object[i, j]
                            elif len(par_set_list) == 1:
                                for set in par_set_list:
                                    for (i, j) in getattr(model, str(set)):
                                        # print(type(final_result['parameters'][par]),final_result['parameters'][par])
                                        # print(i,j,final_result['parameters'][par][i-1][j-1])
                                        final_result['parameters'][par][i - 1][j - 1] = par_object[i, j]
                            else:
                                pass

                        else:
                            pass  # FIXME 3-dimensional variables are not considered yet

                # Variables
                final_result['variables'] = dict()
                # Include objective functionv value
                for key, value in final_result['solver_results_def']['Solution'][0]['Objective'].items():
                    final_result['variables']['Objective'] = value['Value']
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
                            final_result['variables'][variable] = array(
                                [value.value for value in variable_object.values()])
                        else:
                            try:
                                variable_set_list = [set for set in variable_object._index.set_tuple]
                                variable_set_lens = [len(getattr(model, str(set))) for set in
                                                     variable_set_list]
                            except AttributeError:
                                variable_set_list = [str(variable_object._index.name)]
                                temp_variable_set = getattr(model, variable_set_list[0])
                                if temp_variable_set.dimen == 1:
                                    variable_set_lens = [len(temp_variable_set)]
                                else:
                                    variable_set_lens = [len(set) for set in
                                                         temp_variable_set.domain.set_tuple]

                            # print(variable_set_lens)
                            final_result['variables'][variable] = zeros(shape=variable_set_lens,
                                                                        dtype=float)
                            if variable_object_dim == 2:
                                if len(variable_set_list) == variable_object_dim:
                                    for ind_i, i in enumerate(getattr(model, str(variable_set_list[0]))):
                                        for ind_j, j in enumerate(
                                                getattr(model, str(variable_set_list[1]))):
                                            final_result['variables'][variable][ind_i][ind_j] = \
                                                variable_object[i, j].value
                                elif len(variable_set_list) == 1:
                                    for set in variable_set_list:
                                        for (i, j) in getattr(model, str(set)):
                                            # print(type(final_result['variables'][variable]),final_result['variables'][variable])
                                            # print(i,j,final_result['variables'][variable][i-1][j-1])
                                            final_result['variables'][variable][i - 1][j - 1] = \
                                                variable_object[i, j].value
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
                    self._psmsg('An optimal solution could not be processed')  # leave program running to debug
                else:
                    self._pemsg('An optimal solution could not be processed')

        else:  # if solver_status != ok
            self._pemsg('An optimal solution was NOT found')

        return final_result


if __name__ == '__main__':
    print('This is a wrapper for solving Pyomo models, auto-processing variables and parameters')
