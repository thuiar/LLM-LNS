import os
import copy
import numpy as np
import json
import random
import time
import sys
import types
import re
import warnings
import http.client
import ast
from gurobipy import GRB, read, Model
import concurrent.futures
import heapq
from typing import Sequence, Tuple
from joblib import Parallel, delayed
from pathlib import Path
import traceback

from selection import prob_rank,equal,roulette_wheel,tournament
from management import pop_greedy,ls_greedy,ls_sa

# Define some necessary classes
class Paras():
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'eoh'                # Method selected for use
        self.problem = 'milp_construct'     # Problem selected for solving
        self.selection = None              # Method for individual selection (how to select individuals for evolution from the population)
        self.management = None             # Method for population management

        #####################
        ###  EC settings  ###
        #####################
        self.ec_pop_size = 5  # Number of algorithms in each population, default = 10
        self.ec_n_pop = 5 # Number of populations, default = 10
        self.ec_operators = None # Evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
        self.ec_m = 2  # Number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = None  # Weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]
        
        #####################
        ### LLM settings  ###
        ##################### 
        self.llm_api_endpoint = None # Endpoint for remote LLM, e.g., api.deepseek.com
        self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = None  # Model type for remote LLM, e.g., deepseek-chat

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False  # If debug mode is enabled
        self.exp_output_path = "./ACP_SC/"  # Default folder for AEL outputs
        self.exp_n_proc = 1
        
        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 5 * 300
        self.prompt_eva_timeout = 30
        self.eva_numba_decorator = False


    def set_parallel(self):
        #################################
        ### Set number of threads to maximum available threads ###
        #################################
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes}.")
    
    def set_ec(self):    
        ###########################################################
        ### Set population management strategy, parent selection strategy, and evolution strategy with corresponding weights ###
        ###########################################################
        if self.management == None:
            if self.method in ['ael','eoh']:
                self.management = 'pop_greedy'
            elif self.method == 'ls':
                self.management = 'ls_greedy'
            elif self.method == 'sa':
                self.management = 'ls_sa'
        
        if self.selection == None:
            self.selection = 'prob_rank'
            
        
        if self.ec_operators == None:
            if self.method == 'eoh':
                self.ec_operators  = ['e1','e2','m1','m2']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1, 1, 1]
            elif self.method == 'ael':
                self.ec_operators  = ['crossover','mutation']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1]
            elif self.method == 'ls':
                self.ec_operators  = ['m1']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1]
            elif self.method == 'sa':
                self.ec_operators  = ['m1']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1]
                    
        if self.method in ['ls','sa'] and self.ec_pop_size >1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> single-point-based, set pop size to 1.")
            
    def set_evaluation(self):
        #################################
        ### Set evaluation parameters related to the problem ###
        #################################
        if self.problem == 'bp_online':
            self.eva_timeout = 20
            self.eva_numba_decorator  = True
        elif self.problem == 'milp_construct':
            self.eva_timeout = 350 * 5
                
    def set_paras(self, *args, **kwargs):
        #################################
        ### Set multi-threading, population strategy, and evaluation parameters ###
        #################################
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
              
        # Identify and set parallel 
        self.set_parallel()
        
        # Initialize method and ec settings
        self.set_ec()
        
        # Initialize evaluation settings
        self.set_evaluation()

#######################################
#######################################
### All set! Time to configure prompts. ###
#######################################
#######################################

def create_folders(results_path):
    #####################################################
    ### Create result folders and subfolders for history, pops, and pops_best ###
    #####################################################
    folder_path = os.path.join(results_path, "results")

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Create the main folder "results"
        os.makedirs(folder_path)

    # Create subfolders inside "results"
    subfolders = ["history", "pops", "pops_best"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

class GetPrompts():
    #####################################################
    ### Prompt class, defining various prompts and their related returns ###
    #####################################################
    def __init__(self):
        # Task description prompt
        
        self.prompt_task = "Given an initial feasible solution and a current solution to a Mixed-Integer Linear Programming (MILP) problem, with a complete description of the constraints and objective function.\
        We want to improve the current solution using Large Neighborhood Search (LNS). \
        The task can be solved step-by-step by starting from the current solution and iteratively selecting a subset of decision variables to relax and re-optimize. \
        In each step, most decision variables are fixed to their values in the current solution, and only a small subset is allowed to change. \
        You need to score all the decision variables based on the information I give you, and I will choose the decision variables with high scores as neighborhood selection.\
        To avoid getting stuck in local optima, the choice of the subset can incorporate a degree of randomness.\
        You can also consider the correlation between decision variables, for example, assigning similar scores to variables involved in the same constraint, which often exhibit high correlation. This will help me select decision variables from the same constraint.\
        Of course, I also welcome other interesting strategies that you might suggest."
        # Prompt function name: select next point
        self.prompt_func_name = "select_neighborhood"
        # Prompt function inputs: current point, destination point, unvisited points, distance matrix
        self.prompt_func_inputs = ["n", "m", "k", "site", "value", "constraint", "initial_solution", "current_solution", "objective_coefficient"]
        # Prompt function output: next point
        self.prompt_func_outputs = ["neighbor_score"]
        # Description of prompt function inputs and outputs
        self.prompt_inout_inf = "'n': Number of decision variables in the problem instance. 'n' is a integer number. \
        'm': Number of constraints in the problem instance. 'm' is a integer number.\
        'k': k[i] indicates the number of decision variables involved in the ith constraint. 'k' is a Numpy array with length m.\
        'site': site[i][j] indicates which decision variable is involved in the jth position of the ith constraint. 'site' is a list of Numpy arrays. The length of the list is m.\
        'value': value[i][j] indicates the coefficient of the jth decision variable in the ith constraint. 'value' is a list of Numpy arrays. The length of the list is m.\
        'constraint': constraint[i] indicates the right-hand side value of the ith constraint. 'constraint' is a Numpy array with length m.\
        'initial_solution': initial_solution[i] indicates the initial value of the i-th decision variable. initial_solution is a Numpy array with length n\
        'current_solution': current_solution[i] indicates the current value of the i-th decision variable. current_solution is a Numpy array with length n.\
        'objective_coefficient': objective_coefficient[i] indicates the objective function coefficient corresponding to the i-th decision variable. objective_coefficient is a Numpy array with length n.\
        'initial_solution', 'current_solution', and 'objective_coefficient' are numpy arrays with length n. The i-th element of the arrays corresponds to the i-th decision variable. \
        This corresponds to the Set Cover MILP problem, where all decision variables are binary (0-1 variables), and all constraints are in the form of LHS >= RHS.\
        'neighbor_score' is also a numpy array that you need to create manually. The i-th element of the arrays corresponds to the i-th decision variable."
        # Other prompt descriptions
        self.prompt_other_inf = "All are Numpy arrays. I don't give you 'neighbor_score' so that you need to create it manually. The length of the 'neighbor_score' array is also 'n'."

    def get_task(self):
        #################################
        ### Get task description ###
        #################################
        return self.prompt_task
    
    def get_func_name(self):
        #################################
        ### Get prompt function name ###
        #################################
        return self.prompt_func_name
    
    def get_func_inputs(self):
        #################################
        ### Get prompt function inputs ###
        #################################
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        #################################
        ### Get prompt function outputs ###
        #################################
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        #################################
        ### Get prompt function input/output description ###
        #################################
        return self.prompt_inout_inf

    def get_other_inf(self):
        #################################
        ### Get other prompt descriptions ###
        #################################
        return self.prompt_other_inf

#######################################
#######################################
### Time to create problem instances! ###
#######################################
#######################################

class GetData():
    ########################################################
    ### Given number of instances and cities, obtain coordinates and distance matrix for each point in each instance ###
    ########################################################
    def generate_instances(self, lp_path): #'./test'
        sample_files = [str(path) for path in Path(lp_path).glob("*.lp")]
        instance_data = []
        for f in sample_files: # Randomly generate for each instance
            model = read(f)
            value_to_num = {}
            value_to_type = {}
            value_num = 0
            # n: number of decision variables
            # m: number of constraints
            # k[i]: number of decision variables in the i-th constraint
            # site[i][j]: which decision variable is in the j-th position of the i-th constraint
            # value[i][j]: coefficient of the j-th decision variable in the i-th constraint
            # constraint[i]: right-hand side value of the i-th constraint
            # constraint_type[i]: type of the i-th constraint, 1 for <=, 2 for >=, 3 for ==
            # coefficient[i]: coefficient of the i-th decision variable in the objective function
            n = model.NumVars
            m = model.NumConstrs
            k = []
            site = []
            value = []
            constraint = []
            constraint_type = []
            for cnstr in model.getConstrs():
                if(cnstr.Sense == '<'):
                    constraint_type.append(1)
                elif(cnstr.Sense == '>'):
                    constraint_type.append(2) 
                else:
                    constraint_type.append(3) 
                
                constraint.append(cnstr.RHS)


                now_site = []
                now_value = []
                row = model.getRow(cnstr)
                k.append(row.size())
                for i in range(row.size()):
                    if(row.getVar(i).VarName not in value_to_num.keys()):
                        value_to_num[row.getVar(i).VarName] = value_num
                        value_num += 1
                    now_site.append(value_to_num[row.getVar(i).VarName])
                    now_value.append(row.getCoeff(i))
                site.append(now_site)
                value.append(now_value)

            coefficient = {}
            lower_bound = {}
            upper_bound = {}
            value_type = {}
            for val in model.getVars():
                if(val.VarName not in value_to_num.keys()):
                    value_to_num[val.VarName] = value_num
                    value_num += 1
                coefficient[value_to_num[val.VarName]] = val.Obj
                lower_bound[value_to_num[val.VarName]] = val.LB
                upper_bound[value_to_num[val.VarName]] = val.UB
                value_type[value_to_num[val.VarName]] = val.Vtype

            # 1 for minimization, -1 for maximization
            obj_type = model.ModelSense
            model.setObjective(0, GRB.MAXIMIZE)
            model.optimize()
            new_sol = {}
            for val in model.getVars():
                if(val.VarName not in value_to_num.keys()):
                    value_to_num[val.VarName] = value_num
                    value_num += 1
                new_sol[value_to_num[val.VarName]] = val.x
            
            # Post-process
            new_site = []
            new_value = []
            new_constraint = np.zeros(m)
            new_constraint_type = np.zeros(m, int)
            for i in range(m):
                new_site.append(np.zeros(k[i], int))
                new_value.append(np.zeros(k[i]))
                for j in range(k[i]):
                    new_site[i][j] = site[i][j]
                    new_value[i][j] = value[i][j]
                new_constraint[i] = constraint[i]
                new_constraint_type[i] = constraint_type[i]
            
            new_coefficient = np.zeros(n)
            new_lower_bound = np.zeros(n)
            new_upper_bound = np.zeros(n)
            new_value_type = np.zeros(n, int)
            new_new_sol = np.zeros(n)
            for i in range(n):
                new_coefficient[i] = coefficient[i]
                new_lower_bound[i] = lower_bound[i]
                new_upper_bound[i] = upper_bound[i]
                if(value_type[i] == 'B'):
                    new_value_type[i] = 0
                elif(value_type[i] == 'C'):
                    new_value_type[i] = 1
                else:
                    new_value_type[i] = 2
                new_new_sol[i] = new_sol[i]


            instance_data.append((n, m, k, new_site, new_value, new_constraint, new_constraint_type, new_coefficient, obj_type, new_lower_bound, new_upper_bound, new_value_type, new_new_sol))
        return instance_data
    
class PROBLEMCONST():
    ###########################################
    ### Create brand new TSP problem instances ###
    ###########################################
    def __init__(self) -> None:
        self.path = "./SC_easy_instance/LP"
        self.set_time = 100
        self.n_p = 5 # Test time
        self.epsilon = 1e-3

        self.prompts = GetPrompts()
        # Call defined GetData() to get randomly generated problem instances
        getData = GetData()
        self.instance_data = getData.generate_instances(self.path)
        #print(self.instance_data[0])
    
    def Gurobi_solver(self, n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, value_type, now_sol, now_col):
        '''
        Function description:
        Solves the given problem instance using the Gurobi solver.

        Parameter description:
        - n: Number of decision variables in the problem instance.
        - m: Number of constraints in the problem instance.
        - k: k[i] indicates the number of decision variables in the i-th constraint.
        - site: site[i][j] indicates which decision variable is in the j-th position of the i-th constraint.
        - value: value[i][j] indicates the coefficient of the j-th decision variable in the i-th constraint.
        - constraint: constraint[i] indicates the right-hand side value of the i-th constraint.
        - constraint_type: constraint_type[i] indicates the type of the i-th constraint, 1 for <=, 2 for >=, 3 for ==
        - coefficient: coefficient[i] indicates the coefficient of the i-th decision variable in the objective function.
        - time_limit: Maximum solving time.
        - obj_type: Whether the problem is maximization or minimization.
        '''
        # Get start time
        begin_time = time.time()
        # Define the solver model
        model = Model("Gurobi")
        # Set variable mapping
        site_to_new = {}
        new_to_site = {}
        new_num = 0
        x = []
        for i in range(n):
            if(now_col[i] == 1):
                site_to_new[i] = new_num
                new_to_site[new_num] = i
                new_num += 1
                if(value_type[i] == 0):
                    x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY))
                if(value_type[i] == 1):
                    x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
                else:
                    x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER))
        
        # Set objective function and optimization goal (maximization/minimization)
        coeff = 0
        for i in range(n):
            if(now_col[i] == 1):
                coeff += x[site_to_new[i]] * coefficient[i]
            else:
                coeff += now_sol[i] * coefficient[i]
        if(obj_type == -1):
            model.setObjective(coeff, GRB.MAXIMIZE)
        else:
            model.setObjective(coeff, GRB.MINIMIZE)
        # Add m constraints
        for i in range(m):
            constr = 0
            flag = 0
            for j in range(k[i]):
                if(now_col[site[i][j]] == 1):
                    constr += x[site_to_new[site[i][j]]] * value[i][j]
                    flag = 1
                else:
                    constr += now_sol[site[i][j]] * value[i][j]

            if(flag == 1):
                if(constraint_type[i] == 1):
                    model.addConstr(constr <= constraint[i])
                elif(constraint_type[i] == 2):
                    model.addConstr(constr >= constraint[i])
                else:
                    model.addConstr(constr == constraint[i])
            else:
                if(constraint_type[i] == 1):
                    if(constr > constraint[i]):
                        print("QwQ")
                        print(constr,  constraint[i])
                        print(now_col)
                else:
                    if(constr < constraint[i]):
                        print("QwQ")
                        print(constr,  constraint[i])
                        print(now_col)
        # Set maximum solving time
        model.setParam('OutputFlag', 0)
        if(time_limit - (time.time() - begin_time) <= 0):
            return -1, -1, -1
        model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
        # Optimize
        model.optimize()
        try:
            new_sol = np.zeros(n)
            for i in range(n):
                if(now_col[i] == 0):
                    new_sol[i] = now_sol[i]
                else:
                    if(value_type[i] == 'C'):
                        new_sol[i] = x[site_to_new[i]].X
                    else:
                        new_sol[i] = (int)(x[site_to_new[i]].X)
                
            return new_sol, model.ObjVal, 1
        except:
            return -1, -1, -1
    
    def eval(self, n, coefficient, new_sol):
        ans = 0
        for i in range(n):
            ans += coefficient[i] * new_sol[i]
        return(ans)
    
    def greedy_one(self, now_instance_data, eva):
        n = now_instance_data[0]
        m = now_instance_data[1]
        k = now_instance_data[2]
        site = now_instance_data[3]
        value = now_instance_data[4] 
        constraint = now_instance_data[5]
        constraint_type = now_instance_data[6] 
        coefficient = now_instance_data[7]
        obj_type = now_instance_data[8]
        lower_bound = now_instance_data[9]
        upper_bound = now_instance_data[10]
        value_type = now_instance_data[11]
        initial_sol = now_instance_data[12]

        parts = 10
        begin_time = time.time()
        turn_ans = [self.eval(n, coefficient, initial_sol)]
        
        now_sol = initial_sol
        try:
            while(time.time() - begin_time <= self.set_time):

                #print("before", parts, time.time() - begin_time)
                #"n", "m", "k", "site", "value", "constraint", "initial_solution", "current_solution", "objective_coefficient"
                neighbor_score = eva.select_neighborhood(
                                    n, 
                                    m, 
                                    copy.deepcopy(k),
                                    copy.deepcopy(site), 
                                    copy.deepcopy(value), 
                                    copy.deepcopy(constraint), 
                                    copy.deepcopy(initial_sol), 
                                    copy.deepcopy(now_sol), 
                                    copy.deepcopy(coefficient)
                                )
                #print("after", parts, time.time() - begin_time)
                indices = np.argsort(neighbor_score)[::-1]
                color = np.zeros(n)
                for i in range(n // parts):
                    color[indices[i]] = 1
                if(self.set_time - (time.time() - begin_time) <= 0):
                    break
                new_sol, now_val, now_flag = self.Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(self.set_time - (time.time() - begin_time), self.set_time / 5), obj_type, lower_bound, upper_bound, value_type, now_sol, color)
                if(now_flag == -1):
                    continue
                now_sol = new_sol
                turn_ans.append(now_val)
                if(len(turn_ans) > 3 and abs(turn_ans[-1] - turn_ans[-3]) <= self.epsilon *  turn_ans[-1] and parts >= 3):
                    parts -= 1
            return(turn_ans[-1])
        except Exception as e:
            print(f"MILP Error: {e}")
            traceback.print_exc()
            return(1e9)
    
    def run_with_timeout(self, time_limit, func, *args, **kwargs):
        """Run function func within time_limit seconds, return None if timeout"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                # Wait for result, at most time_limit seconds
                return future.result(timeout=time_limit)
            except concurrent.futures.TimeoutError:
                # Timeout, return None
                print(f"Function {func.__name__} timed out after {time_limit} seconds.")
                return None

    def greedy(self, eva):
        ###############################################################################
        ### Use a greedy-like method, selecting the next point in each step via eva.select_next_node ###
        ### Run multiple instances, and return the average of the results from these instances ###
        ###############################################################################
        results = []
        try:
            num = 0
            for now_instance_data in self.instance_data:
                num += 1
                #print("QWQ!", num)
                result = self.run_with_timeout(150, self.greedy_one, now_instance_data, eva)
                #result = self.greedy_one(now_instance_data, eva)
                
                if result is not None:
                    results.append(result)
                else:
                    results.append(1e9)
                #print("QAQ!", num, result)
                #print("QAQ!")
            # Generate self.pop_size offspring individuals. Results are stored in the 'results' list, each element being a (p, off) tuple, where 'p' is the parent and 'off' is the generated offspring.
            #results = Parallel(n_jobs=self.n_p,timeout=self.set_time+30)(delayed(self.greedy_one)(now_instance_data, eva) for now_instance_data in self.instance_data)
        except Exception as e:
            print(f"Parallel MILP Error: {e}")
            traceback.print_exc()
            results = [1e9]
        
        return sum(results) / len(results)


    def evaluate(self, code_string):
        ###############################################################################
        ### Call greedy to evaluate the fitness of the current strategy ###
        ### Key question: How to transform the current strategy (a string) into executable code? ###
        ###############################################################################
        try:
            # Use warnings.catch_warnings() to capture and control warnings produced during code execution
            with warnings.catch_warnings():
                # Set captured warnings to ignore mode. This means any warnings generated within this code block will be ignored and not displayed to the user.
                warnings.simplefilter("ignore")
                
                # Create a new module object, named "heuristic_module". types.ModuleType creates a new empty module, similar to a container for the code to be executed.
                heuristic_module = types.ModuleType("heuristic_module")
                
                # Use the exec function to execute the code_string in the namespace of the heuristic_module module.
                # exec can dynamically execute code in string form and store the execution results in the specified namespace.
                exec(code_string, heuristic_module.__dict__)

                # Add the newly created module to sys.modules, making it accessible like a regular module in the program. sys.modules is a dictionary that stores all imported modules.
                # By adding heuristic_module to this dictionary, other parts of the code can access it using import heuristic_module.
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Call a method self.greedy from the class, passing heuristic_module as an argument.
                # This line of code will return a fitness value based on the logic defined in the passed heuristic_module.
                fitness = self.greedy(heuristic_module)
                # If no exception occurs, return the calculated fitness value.
                return fitness
        except Exception as e:
            # Return None if an exception occurs, indicating that code execution failed or fitness value could not be successfully calculated.
            print(f"Greedy MILP Error: {e}")
            return None


class Probs():
    ###########################################
    ### Read problem instances or call PROBLEMCONST() to create problem instances ###
    ###########################################
    def __init__(self,paras):
        if not isinstance(paras.problem, str):
            # Read existing problem instances
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "milp_construct":
            # Create new problem instances
            self.prob = PROBLEMCONST()
            print("- Prob "+paras.problem+" loaded ")
        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):
        # Return problem instances
        return self.prob


#######################################
#######################################
### Finally, the section for interacting with large language models! ###
#######################################
#######################################
class InterfaceAPI:
    #######################################
    ### Call API to communicate with large language models ###
    #######################################
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint # API endpoint address
        self.api_key = api_key           # API key
        self.model_LLM = model_LLM       # Name of the large language model used
        self.debug_mode = debug_mode     # Whether debug mode is enabled
        self.n_trial = 5                 # Maximum 5 attempts to get a response

    def get_response(self, prompt_content):
        # Create a JSON formatted string payload_explanation, containing the model name and message content. This JSON will be used as the request payload.
        payload_explanation = json.dumps(
            {
                # Specify the model to use.
                "model": self.model_LLM,  
                # Message content, representing the user's input.
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
            }
        )
        # Define request headers
        headers = {
            "Authorization": "Bearer " + self.api_key,           # Contains API key for authentication.
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",   # Identifies client information for the request.
            "Content-Type": "application/json",                  # Specifies content type of the request as JSON.
            "x-api2d-no-cache": 1,                               # Custom header to control caching behavior.
        }
        
        response = None   # Initialize response variable to None, used to store API response content.
        n_trial = 1       # Initialize attempt count n_trial to 1, indicating the first attempt will begin.
        
        # Start an infinite loop to repeatedly attempt to get API response until successful or maximum attempts reached.
        while True:
            n_trial += 1
            # Check if current attempt count exceeds maximum allowed attempts self.n_trial (5 times). If exceeded, return current response (possibly None) and exit function.
            if n_trial > self.n_trial:
                return response
            try:
                # Create an HTTPS connection to the API endpoint.
                conn = http.client.HTTPSConnection(self.api_endpoint)
                # Send a POST request to the /v1/chat/completions endpoint, passing request payload and headers.
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                # Get the response to the request.
                res = conn.getresponse()
                # Read the response content.
                data = res.read()
                # Convert response content from JSON format to Python dictionary.
                json_data = json.loads(data)
                # Extract actual model reply content from JSON response and store it in the response variable.
                response = json_data["choices"][0]["message"]["content"]

                #server_b_url = "http://43.134.189.32:5000/openai"
                #response = requests.post(server_b_url, json={"prompt": prompt_content}).json()['response']
                break
            except:
                if self.debug_mode:  # If debug mode is enabled, output debug information.
                    print("Error in API. Restarting the process...")
                continue
            
        return response


class InterfaceLLM:
    #######################################
    ### Call InterfaceAPI class to communicate with large language models ###
    #######################################
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint # API endpoint URL for communication with the language model
        self.api_key = api_key           # API key for authentication
        self.model_LLM = model_LLM       # Name of the language model used
        self.debug_mode = debug_mode     # Whether debug mode is enabled

        print("- check LLM API")

        print('remote llm api is used ...')
        # If default settings are not changed, provide a reminder.
        if self.api_key == None or self.api_endpoint ==None or self.api_key == 'xxx' or self.api_endpoint == 'xxx':
            print(">> Stop with wrong API setting: Set api_endpoint (e.g., api.chat...) and api_key (e.g., kx-...) !")
            exit()
        # Create an instance of the InterfaceAPI class and assign it to self.interface_llm. InterfaceAPI is a class defined above for actual API request processing.
        self.interface_llm = InterfaceAPI(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            self.debug_mode,
        )

        # Call the get_response method of the InterfaceAPI instance, sending a simple request "1+1=?" to test if the API connection and configuration are correct.
        res = self.interface_llm.get_response("1+1=?")

        # Check if the response is None, which means API request failed or configuration error.
        if res == None:
            print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
            exit()

    def get_response(self, prompt_content):
        ##############################################################################
        # Defines a method get_response to obtain LLM's response to given content.
        # It accepts one parameter prompt_content, which represents the user-provided prompt content.
        ##############################################################################

        # Call the get_response method of the InterfaceAPI instance, send prompt content, and get the response.
        response = self.interface_llm.get_response(prompt_content)

        # Return the response obtained from InterfaceAPI.
        return response


#######################################
#######################################
### Outer layer! Seems everything is ready? Let's start preparing for evolution! ###
#######################################
#######################################
class Evolution_Prompt():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, problem_type, **kwargs):
        # problem_type: minimization/maximization

        self.prompt_task = "We are working on solving a " + problem_type + " problem." + \
        " Our objective is to leverage the capabilities of the Language Model (LLM) to generate heuristic algorithms that can efficiently tackle this problem." + \
        " We have already developed a set of initial prompts and observed the corresponding outputs." + \
        " However, to improve the effectiveness of these algorithms, we need your assistance in carefully analyzing the existing prompts and their results." + \
        " Based on this analysis, we ask you to generate new prompts that will help us achieve better outcomes in solving the " + problem_type + " problem."

        # Set large language model parameters
        self.api_endpoint = api_endpoint      # LLM API endpoint for interacting with external services.
        self.api_key = api_key                # API key for authentication and authorization.
        self.model_LLM = model_LLM            # Name or identifier of the language model used.
        self.debug_mode = debug_mode          # Debug mode flag

        # Set LLM using the defined InterfaceLLM. Now its get_response(self, prompt_content) function can be used for communication.
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)
    
        
    def get_prompt_cross(self,prompts_indivs):
        ##################################################
        ### Generate a crossover method for prompts ###
        ##################################################
        
        # Combine the ideas and corresponding code of algorithms in indivs to form statements like "Algorithm No. 1's idea and code are..., Algorithm No. 2's idea and code are..."
        prompt_indiv = ""
        for i in range(len(prompts_indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" prompt's tasks assigned to LLM, and objective function value are: \n" + prompts_indivs[i]['prompt']+"\n" + str(prompts_indivs[i]['objective']) +"\n"
        # 1. Describe the task
        # 2. Tell LLM how many algorithms we're providing, and what they are like (combined with prompt_indiv)
        # 3. Request that LLM create an algorithm completely different from the ones provided
        # 4. Instruct LLM to first describe its new algorithm and main steps in one sentence, enclosed in parentheses.
        # 5. Then, implement it in Python as a function named self.prompt_func_name.
        # 6. Tell LLM how many inputs and outputs this function has, and what they are (corresponding to the processed self.prompt_func_inputs and self.prompt_func_outputs).
        # 7. Describe some properties of the input and output data.
        # 8. Describe some other supplementary properties.
        # 9. Finally, emphasize no other explanations, just output as required.
        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(prompts_indivs))+" existing prompt with objective function value as follows: \n"\
+prompt_indiv+\
"Please help me create a new prompt that has a totally different form from the given ones but can be motivated from them. \n" +\
"Please describe your new prompt and main steps in one sentences."\
+"\n"+"Do not give additional explanations!!! Just one sentences." \
+"\n"+"Do not give additional explanations!!! Just one sentences."
        return prompt_content
    
    
    def get_prompt_variation(self,prompts_indivs):
        ##################################################
        ### Prompt to modify a heuristic to improve performance ###
        ##################################################
        
        # 1. Describe the task
        # 2. Tell LLM about 1 algorithm, what it's like, and provide both its idea and code
        # 3. Request LLM to create a new algorithm with a different form, but which can be a modified version of the provided algorithm
        # 4. Instruct LLM to first describe its new algorithm and main steps in one sentence, enclosed in parentheses.
        # 5. Then, implement it in Python as a function named self.prompt_func_name.
        # 6. Tell LLM how many inputs and outputs this function has, and what they are (corresponding to the processed self.prompt_func_inputs and self.prompt_func_outputs).
        # 7. Describe some properties of the input and output data.
        # 8. Describe some other supplementary properties.
        # 9. Finally, emphasize no other explanations, just output as required.
        prompt_content = self.prompt_task+"\n"\
"I have one prompt with its objective function value as follows." + \
"prompt description: " + prompts_indivs[0]['prompt'] + "\n" + \
"objective function value:\n" +\
str(prompts_indivs[0]['objective'])+"\n" +\
"Please assist me in creating a new prompt that has a different form but can be a modified version of the algorithm provided. \n" +\
"Please describe your new prompt and main steps in one sentences." \
+"\n"+"Do not give additional explanations!!! Just one sentences." \
+"\n"+"Do not give additional explanations!!! Just one sentences."
        return prompt_content
    
    def initialize(self, prompt_type):
        if(prompt_type == 'cross'):
            prompt_content = ['Please help me create a new algorithm that has a totally different form from the given ones.', \
                              'Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.']
        else:
            prompt_content = ['Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.', \
                              'Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.']            
        return prompt_content
    
    def cross(self,parents):
        #print("Begin: 4")
        ##################################################
        ### Obtain algorithm description and code for generating new heuristics as different as possible from parent heuristics ###
        ##################################################

        # Get prompt for LLM to help create new heuristics as different as possible from parent heuristics
        prompt_content = self.get_prompt_cross(parents)
        #print("???", prompt_content)
        response = self.interface_llm.get_response(prompt_content)

        # Output prompt for LLM to help create new heuristics as different as possible from parent heuristics in debug mode.
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ cross ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            

        return response
    
    def variation(self,parents):
        ###########################################################
        ### Obtain algorithm description and code for new heuristics by modifying current heuristics to improve performance ###
        ###########################################################
        prompt_content = self.get_prompt_variation(parents)
        response = self.interface_llm.get_response(prompt_content)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ variation ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            
    
        return response

#######################################
#######################################
### Prompt: How to communicate and process algorithms are ready, let's start! ###
#######################################
#######################################
class InterfaceEC_Prompt():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, select,n_p,timeout, problem_type, **kwargs):

        # Set information needed by LLM
        self.pop_size = pop_size                   # Define population size

        self.evol = Evolution_Prompt(api_endpoint, api_key, llm_model, debug_mode, problem_type , **kwargs)  # Evolution type, including i1, e1, e2, m1, m2, can be used for algorithm evolution
        self.m = m                                  # Number of parent algorithms for prompt's cross operation
        self.debug = debug_mode                     # Whether debug mode is enabled

        # If debug mode is not enabled, no warnings are displayed
        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select                        # Parent selection method
        self.n_p = n_p                              # Number of processes/parallel jobs
        
        self.timeout = timeout                      # Timeout definition
    
    # Write text code to file ./prompt.txt
    def prompt2file(self,prompt):
        with open("./prompt.txt", "w") as file:
        # Write the code to the file
            file.write(prompt)
        return 
    
    # Add a new individual (offspring) to an existing population,
    # provided that the new individual does not have a duplicate objective value in the population.
    # If there is no duplicate objective function value, add it and return True; otherwise, return False.
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['prompt'] == offspring['prompt']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def extract_first_quoted_string(self, text):
        # Use regex to match content within the first double quotes
        match = re.search(r'"(.*?)"', text)
        if match:
            text =  match.group(1)  # Extract the content of the first match
        prefix = "Prompt: "
        if text.startswith(prefix):
            return text[len(prefix):].strip()  # Remove prefix and leading/trailing spaces
        return text  # Return original string if no match
    
    
    # Used to generate offspring individuals based on the specified evolutionary operator.
    def _get_alg(self,pop,operator):
        #print("Begin: 3")
        # Initialize offspring: create an offspring dictionary
        offspring = {
            'prompt': None,
            'objective': None,
            'number': None
        }
        off_set = []
        # Get initial prompt
        if operator == "initial_cross":
            parents = []
            prompt_list =  self.evol.initialize("cross")
            for prompt in prompt_list:
                offspring = {
                    'prompt': None,
                    'objective': None,
                    'number': None
                }
                offspring["prompt"] = prompt
                offspring["objective"] = 1e9
                offspring["number"] = []
                off_set.append(offspring)
        elif operator == "initial_variation":
            parents = []
            prompt_list =  self.evol.initialize("variation")   
            for prompt in prompt_list:
                offspring = {
                    'prompt': None,
                    'objective': None,
                    'number': None
                }
                offspring["prompt"] = prompt
                offspring["objective"] = 1e9
                offspring["number"] = []
                off_set.append(offspring)
        # Generate new prompt through crossover      
        elif operator == "cross":
            parents = self.select.parent_selection(pop,self.m)
            prompt_now = self.evol.cross(parents)
            try:
                prompt_new = self.extract_first_quoted_string(prompt_now)
            except Exception as e:
                print("Prompt cross", e)
            offspring["prompt"] = prompt_new
            offspring["objective"] = 1e9
            offspring["number"] = []
        # Generate new prompt through mutation
        elif operator == "variation":
            parents = self.select.parent_selection(pop,1)
            #print(parents)
            prompt_now = self.evol.variation(parents)
            try:
                prompt_new = self.extract_first_quoted_string(prompt_now)
            except Exception as e:
                print("Prompt variation", e)

            offspring["prompt"] = prompt_new
            offspring["objective"] = 1e9
            offspring["number"] = [] 
        # No such operation!
        else:
            print(f"Prompt operator [{operator}] has not been implemented ! \n") 

        # Return selected parent algorithms and generated offspring
        return parents, offspring, off_set

    # Used to generate offspring individuals and evaluate their fitness
    def get_offspring(self, pop, operator):
        try:
            #print("Begin: 2")
            # Call _get_alg method to generate offspring individual 'offspring' from 'pop' based on 'operator' (i1, m1, etc.), and return parent individual 'p' and offspring individual 'offspring'.
            #print(operator)
            p, offspring, off_set = self._get_alg(pop, operator)
            
        # If an exception occurs, set 'offspring' to a dictionary containing all None values, and set 'p' to None.
        except Exception as e:
            print("get_offspring", e)
            offspring = {
                'prompt': None,
                'objective': None,
                'number': None
            }
            p = None
            off_set = None

        # Return parent individual 'p' and generated offspring individual 'offspring'.
        return p, offspring, off_set
    
    def get_algorithm(self, pop, operator):
        # results: Create an empty list results to store generated offspring individuals.
        results = []
        try:
            # Generate self.pop_size offspring individuals. Results are stored in the 'results' list, each element being a (p, off) tuple, where 'p' is the parent and 'off' is the generated offspring.
            if(operator == 'cross' or operator == 'variation'):
                #print("Begin: 1")
                results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size))
            else:
                results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator) for _ in range(1))
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel time out .")
            
        time.sleep(2)


        out_p = []   # All parent individuals
        out_off = [] # All offspring individuals

        for p, off, off_set in results:
            out_p.append(p)
            if(operator == 'cross' or operator == 'variation'):
                out_off.append(off)
            else:
                for now_off in off_set:
                    out_off.append(now_off)
            # If in debug mode, output offspring individuals
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off

    def population_generation(self, initial_type):
        # Set to 2, indicating that 2 rounds of individuals are to be generated.
        n_create = 1
        # Create an empty list to store generated initial population individuals.
        population = []
        # Loop to generate individuals
        for i in range(n_create):
            _,pop = self.get_algorithm([], initial_type)
            #print(pop)
            for p in pop:
                population.append(p)
             
        return population

    
#######################################
#######################################
### Inner layer! Seems everything is ready? Let's start preparing for evolution! ###
#######################################
####################################### 

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode,prompts, **kwargs):

        # set prompt interface
        #getprompts = GetPrompts()
        self.prompt_task         = prompts.get_task()         #"Given a set of nodes with their coordinates, \
                                                              #you need to find the shortest route that visits each node once and returns to the starting node. \
                                                              #The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. \
                                                              #Help me design a novel algorithm that is different from the algorithms in literature to select the next node in each step."
        self.prompt_func_name    = prompts.get_func_name()    #"select_next_node"
        self.prompt_func_inputs  = prompts.get_func_inputs()  #["current_node","destination_node","univisited_nodes","distance_matrix"]
        self.prompt_func_outputs = prompts.get_func_outputs() #["next_node"]
        self.prompt_inout_inf    = prompts.get_inout_inf()    #"'current_node', 'destination_node', 'next_node', and 'unvisited_nodes' are node IDs. 'distance_matrix' is the distance matrix of nodes."
        self.prompt_other_inf    = prompts.get_other_inf()    #"All are Numpy arrays."
        
        # ["current_node","destination_node","univisited_nodes","distance_matrix"] -> "'current_node','destination_node','univisited_nodes','distance_matrix'"
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        # ["next_node"] -> "'next_node'"
        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # Set large language model parameters
        self.api_endpoint = api_endpoint      # LLM API endpoint for interacting with external services.
        self.api_key = api_key                # API key for authentication and authorization.
        self.model_LLM = model_LLM            # Name or identifier of the language model used.
        self.debug_mode = debug_mode          # Debug mode flag

        # Set LLM using the defined InterfaceLLM. Now its get_response(self, prompt_content) function can be used for communication.
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

    def get_prompt_initial(self):
        #######################################
        ### Generate prompt for initial strategy ###
        #######################################

        # First describe the task, then describe what LLM needs to do:
        # 1. First, describe your new algorithm and main steps in one sentence, enclosed in parentheses.
        # 2. Then, implement it in Python as a function named self.prompt_func_name.
        # 3. Tell LLM how many inputs and outputs this function has, and what they are (corresponding to the processed self.prompt_func_inputs and self.prompt_func_outputs).
        # 4. Describe some properties of the input and output data.
        # 5. Describe some other supplementary properties.
        # 6. Finally, emphasize no other explanations, just output as required.
        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content

        
    def get_prompt_cross(self,indivs, prompt):
        ##################################################
        ### Generate prompt for new heuristics that are as different as possible from parent heuristics ###
        ##################################################
        
        # Combine the ideas and corresponding code of algorithms in indivs to form statements like "Algorithm No. 1's idea and code are..., Algorithm No. 2's idea and code are..."
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm's thought, objective function value, and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" + str(indivs[i]['objective']) +"\n" +indivs[i]['code']+"\n"
        # 1. Describe the task
        # 2. Tell LLM how many algorithms we're providing, and what they are like (combined with prompt_indiv)
        # 3. Request LLM to create an algorithm completely different from the ones provided
        # 4. Instruct LLM to first describe its new algorithm and main steps in one sentence, enclosed in parentheses.
        # 5. Then, implement it in Python as a function named self.prompt_func_name.
        # 6. Tell LLM how many inputs and outputs this function has, and what they are (corresponding to the processed self.prompt_func_inputs and self.prompt_func_outputs).
        # 7. Describe some properties of the input and output data.
        # 8. Describe some other supplementary properties.
        # 9. Finally, emphasize no other explanations, just output as required.
        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithm's thought, objective function value with their codes as follows: \n"\
+prompt_indiv+ prompt + "\n" +\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    
    def get_prompt_variation(self,indiv1, prompt):
        ##################################################
        ### Prompt to modify a heuristic to improve performance ###
        ##################################################
        
        # 1. Describe the task
        # 2. Tell LLM about 1 algorithm, what it's like, and provide both its idea and code
        # 3. Request LLM to create a new algorithm with a different form, but which can be a modified version of the provided algorithm
        # 4. Instruct LLM to first describe its new algorithm and main steps in one sentence, enclosed in parentheses.
        # 5. Then, implement it in Python as a function named self.prompt_func_name.
        # 6. Tell LLM how many inputs and outputs this function has, and what they are (corresponding to the processed self.prompt_func_inputs and self.prompt_func_outputs).
        # 7. Describe some properties of the input and output data.
        # 8. Describe some other supplementary properties.
        # 9. Finally, emphasize no other explanations, just output as required.
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n" + \
prompt + "\n" + \
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    


    def _get_alg(self,prompt_content):
        # Obtain response for given prompt_content via LLM interface.
        #print("QwQ~!")
        response = self.interface_llm.get_response(prompt_content)

        if self.debug_mode:
            print("\n >>> check response for creating algorithm using [ i1 ] : \n", response )
            print(">>> Press 'Enter' to continue")
            
        # Use regex re.findall(r"\{(.*)\}", response, re.DOTALL) to attempt to extract algorithm description enclosed in curly braces {}.
        # The re.DOTALL option allows the regex to match newline characters.
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        # If no algorithm description found within curly braces, use alternative patterns.
        if len(algorithm) == 0:
            # If the response contains the word 'python', extract the part from the beginning up to the 'python' keyword for the algorithm description.
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
            # If 'import' is contained, extract the part from the beginning up to 'import' for the algorithm description.
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
            # Otherwise, extract the part from the beginning up to 'def' for the algorithm description.
            else:
                algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

        # Attempt to extract the code part using regex re.findall(r"import.*return", response, re.DOTALL), where the code part starts with 'import' and ends with 'return'.
        code = re.findall(r"import.*return", response, re.DOTALL)
        # If no matching code block is found, try the code block starting with 'def' and ending with 'return'.
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        # If initial extraction of algorithm description or code fails, retry.
        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            # Call get_response method again to get a new response, and repeat attempts to extract algorithm description and code.
            response = self.interface_llm.get_response(prompt_content)
            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)
            
            # If retry count exceeds 3 times (n_retry > 3), exit the loop.
            if n_retry > 3:
                break
            n_retry += 1

        # Assuming algorithm description and code have been successfully extracted, extract them from the list (i.e., take only the first match).
        algorithm = algorithm[0]
        code = code[0] 

        # The extracted code goes up to 'return', we append the rest. Concatenate the extracted code with output variables (stored in self.prompt_func_outputs) to form a complete code string.
        code_all = code+" "+", ".join(s for s in self.prompt_func_outputs) 


        return [code_all, algorithm]


    def initial(self):
        ##################################################
        ### Obtain algorithm description and code for creating initial population ###
        ##################################################
        
        # Obtain prompt for LLM to help create initial population.
        prompt_content = self.get_prompt_initial()

        # Output prompt for LLM to help create initial population in debug mode.
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            
        #print("QAQ~!")
        # Call _get_alg, input prompt to LLM, and split the returned text into code and algorithm description.
        [code_all, algorithm] = self._get_alg(prompt_content)

        # Output the split text (code and algorithm description) in debug mode.
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            

        return [code_all, algorithm]
    
    def cross(self, parents, prompt):
        ##################################################
        ### Obtain algorithm description and code for generating new heuristics as different as possible from parent heuristics ###
        ##################################################

        # Obtain prompt for LLM to help create new heuristics as different as possible from parent heuristics.
        prompt_content = self.get_prompt_cross(parents, prompt)

        # Output prompt for LLM to help create new heuristics as different as possible from parent heuristics in debug mode.
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            
        # Call _get_alg, input prompt to LLM, and split the returned text into code and algorithm description.
        [code_all, algorithm] = self._get_alg(prompt_content)

        # Output the split text (code and algorithm description) in debug mode.
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            

        return [code_all, algorithm]
    
    def variation(self,parents, prompt):
        ###########################################################
        ### Obtain algorithm description and code for new heuristics by modifying current heuristics to improve performance ###
        ###########################################################
        prompt_content = self.get_prompt_variation(parents, prompt)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            

        return [code_all, algorithm]
    
############################################################################################################
### Used to add import statements to given Python code.
### This function's purpose is to insert an import package_name as as_name statement into the code, if the package has not already been imported.
############################################################################################################
def add_import_package_statement(program: str, package_name: str, as_name=None, *, check_imported=True) -> str:
    """Add 'import package_name as as_name' in the program code.
    """
    # Use ast.parse() method to parse the input Python code string into an Abstract Syntax Tree (AST).
    tree = ast.parse(program)

    # If check_imported is True, iterate through each node in the AST to check if a statement to import the specified package already exists.
    if check_imported:
        # check if 'import package_name' code exists
        package_imported = False
        for node in tree.body:
            # The first part checks if the node is an import statement.
            # The second part checks if the package name in the import statement is the same as package_name.
            if isinstance(node, ast.Import) and any(alias.name == package_name for alias in node.names):
                package_imported = True
                break
        # If the package is found to be already imported, the package_imported flag will be set to True, and the unmodified code is returned directly.
        if package_imported:
            return ast.unparse(tree)

    # Create a new import node. Use ast.Import to create a new import statement node, with package_name as the name and as_name as the alias (if any).
    import_node = ast.Import(names=[ast.alias(name=package_name, asname=as_name)])
    # Insert the new import node at the very top of the AST.
    tree.body.insert(0, import_node)
    # Use ast.unparse(tree) to convert the modified AST back into a Python code string and return it.
    program = ast.unparse(tree)
    return program


############################################################################################################
### Used to add the @numba.jit(nopython=True) decorator for NumPy to a given Python code.
### The decorator is added above a specified function to improve its execution efficiency.
### Numba is a JIT (Just-In-Time) compiler used to accelerate numerical computations.
############################################################################################################
def _add_numba_decorator(
        program: str,
        function_name: str
) -> str:
    # Use ast.parse() method to parse the input Python code string into an Abstract Syntax Tree (AST).
    tree = ast.parse(program)

    # Iterate through each node in the AST to check if an 'import numba' statement already exists.
    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break

    # If numba is not yet imported, create an 'import numba' node and insert it at the very top of the AST.
    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)

    
    for node in ast.walk(tree):
        # Use ast.walk(tree) to traverse all nodes in the AST and find function definitions that match the function_name.
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Create an @numba.jit(nopython=True) decorator node.
            # ast.Call creates a call node, ast.Attribute represents attribute access (i.e., numba.jit), ast.keyword creates a keyword node with named arguments.
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='numba', ctx=ast.Load()),
                    attr='jit',
                    ctx=ast.Load()
                ),
                args=[],  # args do not have argument name
                keywords=[ast.keyword(arg='nopython', value=ast.NameConstant(value=True))]
                # keywords have argument name
            )
            # Add it to the decorator_list attribute of the target function.
            node.decorator_list.append(decorator)

    # Use ast.unparse(tree) to convert the modified AST back into a Python code string and return it.
    modified_program = ast.unparse(tree)
    return modified_program


def add_numba_decorator(
        program: str,
        function_name: str | Sequence[str],
) -> str:
    # If function_name is a string, it means only one function needs the decorator. In this case, call the helper function _add_numba_decorator and return its result as the final result.
    if isinstance(function_name, str):
        return _add_numba_decorator(program, function_name)
    # If function_name is a sequence (e.g., list or tuple), iterate through each function name. For each function name, call _add_numba_decorator and update the program.
    for f_name in function_name:
        program = _add_numba_decorator(program, f_name)
    return program


############################################################################################################
##   Add a fixed random seed at the beginning, within the function.
############################################################################################################
# Used to insert a statement to set the random seed (np.random.seed(...)) into the specified Python code.
# If the numpy module (i.e., import numpy as np statement) has not yet been imported into the code, this function first adds that import statement.
def add_np_random_seed_below_numpy_import(program: str, seed: int = 2024) -> str:
    # This line calls the add_import_package_statement function to ensure that 'import numpy as np' is included in the program.
    program = add_import_package_statement(program, 'numpy', 'np')
    # Use Python's ast (Abstract Syntax Tree) module to parse the code into a syntax tree.
    tree = ast.parse(program)

    # find 'import numpy as np'
    found_numpy_import = False

    # find 'import numpy as np' statement
    for node in tree.body:
        # Loop through the nodes of the syntax tree to find the 'import numpy as np' statement.
        if isinstance(node, ast.Import) and any(alias.name == 'numpy' and alias.asname == 'np' for alias in node.names):
            found_numpy_import = True
            # Insert the 'np.random.seed(seed)' statement immediately after the found 'import numpy as np' statement. This is done by creating a new AST node representing the call to the np.random.seed function.
            node_idx = tree.body.index(node)
            seed_node = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id='np', ctx=ast.Load()),
                            attr='random',
                            ctx=ast.Load()
                        ),
                        attr='seed',
                        ctx=ast.Load()
                    ),
                    args=[ast.Num(n=seed)],
                    keywords=[]
                )
            )
            tree.body.insert(node_idx + 1, seed_node)
    # If the 'import numpy as np' statement is not found in the code, raise a ValueError exception. This step ensures that numpy is imported before np.random.seed(seed) is inserted.
    if not found_numpy_import:
        raise ValueError("No 'import numpy as np' found in the code.")
    # Use the ast.unparse method to convert the modified syntax tree back into a Python code string, and return that string.
    modified_code = ast.unparse(tree)
    return modified_code

# Used to add the np.random.seed(seed) statement within a specified Python function to set the random number generator's seed.
# This operation is typically used to ensure reproducibility of random number generation across different runs. The following is a detailed explanation of this code.
def add_numpy_random_seed_to_func(program: str, func_name: str, seed: int = 2024) -> str:
    # This line parses the input code string into an Abstract Syntax Tree (AST).
    tree = ast.parse(program)

    # Insert the new np.random.seed(seed) statement at the beginning of the target function's body.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            node.body = [ast.parse(f'np.random.seed({seed})').body[0]] + node.body
    
    # Convert the entire syntax tree into a new code string that includes the seed setting.
    modified_code = ast.unparse(tree)
    return modified_code

############################################################################################################
### Replaces the division operator (/) in Python code with a custom protected division function _protected_div,
### and optionally uses the numba library to accelerate calculations.
############################################################################################################
# First defines the _CustomDivisionTransformer class, which inherits from ast.NodeTransformer.
# Its purpose is to traverse the Abstract Syntax Tree (AST), find all division operations (/), and replace them with calls to a custom division function.
class _CustomDivisionTransformer(ast.NodeTransformer):
    # Accepts a parameter custom_divide_func_name, which represents the name of the custom function to be used to replace the division operator. Here, the name is _protected_div.
    def __init__(self, custom_divide_func_name: str):
        super().__init__()
        self._custom_div_func = custom_divide_func_name

    # Used to visit all binary operator nodes. If a division operator (/) is detected, it is replaced with the custom function.
    def visit_BinOp(self, node):
        self.generic_visit(node)  # recur visit child nodes
        if isinstance(node.op, ast.Div):
            # self-defined node
            custom_divide_call = ast.Call(
                func=ast.Name(id=self._custom_div_func, ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
            return custom_divide_call
        return node

# Its purpose is to replace all division operators in the input code string with a custom protected division function named _protected_div, which avoids division by zero.
def replace_div_with_protected_div(code_str: str, delta=1e-5, numba_accelerate=False) -> Tuple[str, str]:
    # Define the protected division function _protected_div.
    protected_div_str = f'''
def _protected_div(x, y, delta={delta}):
    return x / (y + delta)
    '''
    # Parse the input code string into an AST tree.
    tree = ast.parse(code_str)

    # Create an instance of _CustomDivisionTransformer and traverse the AST, finding and replacing division operations.
    transformer = _CustomDivisionTransformer('_protected_div')
    modified_tree = transformer.visit(tree)

    # Convert the modified AST back into a code string. Here, the modified code and the definition of the protected division function are returned together.
    modified_code = ast.unparse(modified_tree)
    modified_code = '\n'.join([modified_code, '', '', protected_div_str])

    # If numba_accelerate is true, add the @numba.jit() decorator to the _protected_div function to accelerate computations.
    if numba_accelerate:
        modified_code = add_numba_decorator(modified_code, '_protected_div')
    # Return the modified code string and the name of the custom division function.
    return modified_code, '_protected_div'
    
#######################################
#######################################
### How to communicate and process algorithms are ready, let's start! ###
#######################################
#######################################
class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # Set information needed by LLM
        self.pop_size = pop_size                    # Define population size
        self.interface_eval = interface_prob        # PROBLEMCONST() type, can call evaluate function to assess algorithm code
        prompts = interface_prob.prompts            # Problem description, input/output information prompt, can be used to generate subsequent prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode,prompts, **kwargs)  # Evolution type, including i1, e1, e2, m1, m2, can be used for algorithm evolution
        self.m = m                                  # Number of parent algorithms for 'e1' and 'e2' operations
        self.debug = debug_mode                     # Whether debug mode is enabled

        # If debug mode is not enabled, no warnings are displayed
        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select                        # Parent selection method
        self.n_p = n_p                              # Number of processes/parallel jobs
        
        self.timeout = timeout                      # Timeout definition
        #self.timeout = 400
        self.use_numba = use_numba                  # Whether to use the numba library to accelerate generated functions
    
    # Write text code to file ./ael_alg.py
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    # Add a new individual (offspring) to an existing population,
    # provided that the new individual's objective value does not duplicate any in the population.
    # If there is no duplicate objective function value, add it and return True; otherwise, return False.
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    # Used to check if a given code snippet already exists in any individual within the population.
    # By checking for duplicate code snippets, redundant additions of the same individual to the population can be avoided.
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False
    
    # Used to generate offspring individuals based on the specified evolutionary operator.
    def _get_alg(self,pop,operator, prompt):
        # Initialize offspring: create an offspring dictionary
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        # Obtain initial algorithm
        if operator == "initial":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.initial()    
        # Generate algorithms that are dissimilar to parents        
        elif operator == "cross":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.cross(parents, prompt)
        # Generate new algorithms by improving current ones
        elif operator == "variation":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.variation(parents[0], prompt)    
        # No such operation!
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        # Return selected parent algorithms and generated offspring simultaneously.
        return parents, offspring

    # Used to generate offspring individuals and evaluate their fitness.
    def get_offspring(self, pop, operator, prompt):

        try:
            # Call _get_alg method to generate offspring individual 'offspring' from 'pop' based on 'operator' (i1, m1, etc.), and return parent individual 'p' and offspring individual 'offspring'.
            p, offspring = self._get_alg(pop, operator, prompt)
            
            # Whether to use Numba
            if self.use_numba:
                # Use regex r"def\s+(\w+)\s*\(.*\):" to match function definitions.
                pattern = r"def\s+(\w+)\s*\(.*\):"
                # Extract function name from offspring['code'].
                match = re.search(pattern, offspring['code'])
                function_name = match.group(1)
                # Call add_numba_decorator method to add Numba decorator to the function.
                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            # Handle duplicate code
            n_retry= 1
            while self.check_duplicate(pop, offspring['code']):
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                
                # If generated code duplicates existing code in the current population, regenerate offspring.
                p, offspring = self._get_alg(pop, operator, prompt)

                # Whether to use Numba
                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring['code'])
                    function_name = match.group(1)
                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']
                
                # Try at most once
                if n_retry > 1:
                    break
                
            # Create thread pool: Use ThreadPoolExecutor to execute evaluation tasks.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit self.interface_eval.evaluate method for evaluation, passing the generated code.
                future = executor.submit(self.interface_eval.evaluate, code)
                # Get evaluation result 'fitness', round it to 5 decimal places, and store it in offspring['objective'].
                fitness = future.result(timeout=self.timeout)
                offspring['objective'] = np.round(fitness, 5)
                # Cancel task to release resources.
                future.cancel()              

        # If an exception occurs, set offspring to a dictionary containing all None values, and set p to None.
        except Exception as e:
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            p = None

        # Return parent individual 'p' and generated offspring individual 'offspring'.
        return p, offspring
    
    def get_algorithm(self, pop, operator, prompt):
        # results: Create an empty list results to store generated offspring individuals.
        results = []
        try:
            # Generate self.pop_size offspring individuals. Results are stored in the 'results' list, each element being a (p, off) tuple, where 'p' is the parent and 'off' is the generated offspring.
            results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator, prompt) for _ in range(self.pop_size))
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel time out .")
            
        time.sleep(2)


        out_p = []   # All parent individuals
        out_off = [] # All offspring individuals

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            # If in debug mode, output offspring individuals.
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off

    def population_generation(self):
        # Set to 2, indicating that 2 rounds of individuals are to be generated.
        n_create = 2
        # Create an empty list to store generated initial population individuals.
        population = []
        # Loop to generate individuals
        for i in range(n_create):
            _,pop = self.get_algorithm([],'initial', [])
            for p in pop:
                population.append(p)
             
        return population
    
    # Used to generate a population based on seeds (recorded algorithms), where the fitness of each individual is obtained through parallel evaluation.
    def population_generation_seed(self,seeds,n_p):
        # Create an empty list to store generated population individuals.
        population = []
        # Evaluate each seed's code using the self.interface_eval.evaluate method and calculate its fitness.
        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)
        # Iterate through each seed and its corresponding fitness.
        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population


class EOH:

    # Initialization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem      # Define the problem
        self.select = select     # Define parent selection method
        self.manage = manage     # Define population management method
        
        # LLM settings
        self.api_endpoint = paras.llm_api_endpoint  # Define API endpoint URL for communication with the language model
        self.api_key = paras.llm_api_key            # API private key
        self.llm_model = paras.llm_model            # Define the large language model to use

        # prompt
        self.pop_size_cross = 2
        self.pop_size_variation = 2
        self.problem_type = "minimization"

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # Population size
        self.n_pop = paras.ec_n_pop        # Number of generations to run

        self.operators = paras.ec_operators   # Define number of operations, default is e1, e2, m1, m2
        
        self.operator_weights = paras.ec_operator_weights    # Define operator weights [0, 1]. Higher weight means higher probability of using the operator.
        if paras.ec_m > self.pop_size or paras.ec_m == 1:    # Number of parents required for e1 and e2 operations must be at least two but not exceed population size.
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m                                  # Set number of parents required for e1 and e2 operations.

        self.debug_mode = paras.exp_debug_mode               # Whether debug mode is enabled
        self.ndelay = 1  # default

        self.output_path = paras.exp_output_path             # Population result saving path

        self.exp_n_proc = paras.exp_n_proc                   # Number of processes set
        
        self.timeout = paras.eva_timeout                     # Timeout definition

        self.prompt_timeout = paras.prompt_eva_timeout

        self.use_numba = paras.eva_numba_decorator           # Whether to use the numba library for acceleration

        print("- EoH parameters loaded -")

        # Set random seed
        random.seed(2024)

    # Add newly generated offspring to the population. If debug mode is enabled, compare each offspring with current individuals in the population to check for redundancy.
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)
    
    def add2pop_prompt(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['prompt'] == off['prompt']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)
    

    # Run EOH
    def run(self):

        print("- Evolution Start -")
        # Record start time
        time_start = time.time()

        # Set problem evaluation window
        interface_prob = self.prob
        
        # Set prompt evolution
        interface_promt_cross = InterfaceEC_Prompt(self.pop_size_cross, self.m, self.api_endpoint, self.api_key, self.llm_model, self.debug_mode, self.select, self.exp_n_proc, self.prompt_timeout, self.problem_type)
        interface_promt_variation = InterfaceEC_Prompt(self.pop_size_variation, self.m, self.api_endpoint, self.api_key, self.llm_model, self.debug_mode, self.select, self.exp_n_proc, self.prompt_timeout, self.problem_type)
        # Set evolution mode, including initialization, evolution, and management.
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba
                                   )

        # Initialize population
        cross_operators = []
        variation_operators = []
        print("creating initial prompt:")
        cross_operators = interface_promt_cross.population_generation("initial_cross")
        #cross_operators = self.manage.population_management(cross_operators, self.pop_size_cross)
        variation_operators = interface_promt_variation.population_generation("initial_variation")
        #variation_operators = self.manage.population_management(variation_operators, self.pop_size_variation)
        print(f"Prompt initial: ")
        
        for prompt in cross_operators:
            print("Cross Prompt: ", prompt['prompt'])
        for prompt in variation_operators:
            print("Variation Prompt: ", prompt['prompt'])
        print("initial population has been created!")


        print("=======================================")
        population = []
        print("creating initial population:")
        population = interface_ec.population_generation()
        population = self.manage.population_management(population, self.pop_size)

        print(f"Pop initial: ")
        for off in population:
            print(" Obj: ", off['objective'], end="|")
        print()
        print("initial population has been created!")
        # Save generated population to file
        filename = self.output_path + "/results/pops/population_generation_0.json"
        with open(filename, 'w') as f:
            json.dump(population, f, indent=5)
        n_start = 0

        print("=======================================")

        # n_op: number of evolution operations
        n_op = len(self.operators)
        worst = []
        delay_turn = 3
        change_flag = 0
        last = -1
        max_k = 4
        # n_pop: number of rounds to run
        for pop in range(n_start, self.n_pop):  
            #print(f" [{na + 1} / {self.pop_size}] ", end="|")    
            if(change_flag):
                change_flag -= 1
                if(change_flag == 0):
                    cross_operators = self.manage.population_management(cross_operators, self.pop_size_cross)
                    for prompt in cross_operators:
                        print("Cross Prompt: ", prompt['prompt'])

                    variation_operators = self.manage.population_management(variation_operators, self.pop_size_variation)
                    for prompt in variation_operators:
                        print("Variation Prompt: ", prompt['prompt'])

            if(len(worst) >= delay_turn and worst[-1] == worst[-delay_turn] and pop - last > delay_turn):
                parents, offsprings = interface_promt_cross.get_algorithm(cross_operators, 'cross')
                #print(offsprings)
                self.add2pop_prompt(cross_operators, offsprings)
                parents, offsprings = interface_promt_cross.get_algorithm(cross_operators, 'variation')
                self.add2pop_prompt(cross_operators, offsprings)
                for prompt in cross_operators:
                    print("Cross Prompt: ", prompt['prompt'])
                    prompt["objective"] = 1e9
                    prompt["number"] = []

                parents, offsprings = interface_promt_variation.get_algorithm(variation_operators, 'cross')
                self.add2pop_prompt(variation_operators, offsprings)
                parents, offsprings = interface_promt_variation.get_algorithm(variation_operators, 'variation')
                self.add2pop_prompt(variation_operators, offsprings)
                for prompt in variation_operators:
                    print("Variation Prompt: ", prompt['prompt'])
                    prompt["objective"] = 1e9
                    prompt["number"] = []
                
                change_flag = 2
                last = pop

            # First, check crossover operations
            for i in range(len(cross_operators)):
                promot = cross_operators[i]["prompt"]
                print(f" OP: cross, [{i + 1} / {len(cross_operators)}] ", end="|") 
                parents, offsprings = interface_ec.get_algorithm(population, "cross", promot)
                # Add newly generated offspring to the population. If debug mode is enabled, compare each offspring with current individuals in the population to check for redundancy.
                self.add2pop(population, offsprings)  
                for off in offsprings:
                    print(" Obj: ", off['objective'], end="|")
                    if(off['objective'] is None):
                        continue

                    if len(cross_operators[i]["number"]) < max_k:
                        heapq.heappush(cross_operators[i]["number"], -off['objective'])
                    else:
                        # If heap is full, and current element is smaller than heap top, replace heap top element
                        if off['objective'] < -cross_operators[i]["number"][0]:
                            heapq.heapreplace(cross_operators[i]["number"], -off['objective'])  # Replace heap top element
                        
                    cross_operators[i]["objective"] = -sum(cross_operators[i]["number"]) / len(cross_operators[i]["number"])
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)
                
                # After adding new generation, if population size exceeds limit, manage population to keep it at most pop_size.
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f"Cross {i + 1}, objective: {cross_operators[i]['objective']}", end = "|")
                print()
            
            # Then, check variation operations
            for i in range(len(cross_operators)):
                promot = variation_operators[i]["prompt"]
                print(f" OP: variation, [{i + 1} / {len(variation_operators)}] ", end="|") 
                parents, offsprings = interface_ec.get_algorithm(population, "variation", promot)
                # Add newly generated offspring to the population. If debug mode is enabled, compare each offspring with current individuals in the population to check for redundancy.
                self.add2pop(population, offsprings)  
                for off in offsprings:
                    print(" Obj: ", off['objective'], end="|")
                    if(off['objective'] is None):
                        continue
                    if len(variation_operators[i]["number"]) < max_k:
                        heapq.heappush(variation_operators[i]["number"], -off['objective'])
                    else:
                        # If heap is full, and current element is smaller than heap top, replace heap top element
                        if off['objective'] < -variation_operators[i]["number"][0]:
                            heapq.heapreplace(variation_operators[i]["number"], -off['objective'])  # Replace heap top element
                        
                    variation_operators[i]["objective"] = -sum(variation_operators[i]["number"]) / len(variation_operators[i]["number"])
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)
                
                # After adding new generation, if population size exceeds limit, manage population to keep it at most pop_size.
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f"variation {i + 1}, objective: {variation_operators[i]['objective']}", end = "|")
                print()

            ''' 
            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|") 
                op_w = self.operator_weights[i]
                # If random number is less than weight (weight range from 0 to 1), then run.
                # In other words, for each operation, its operator_weights is the probability of running this operation.
                if (np.random.rand() < op_w):
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                # Add newly generated offspring to the population. If debug mode is enabled, compare each offspring with current individuals in the population to check for redundancy.
                self.add2pop(population, offsprings)  
                for off in offsprings:
                    print(" Obj: ", off['objective'], end="|")
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)
                
                # After adding new generation, if population size exceeds limit, manage population to keep it at most pop_size.
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()
            '''  

            # Save population to file, each generation has its own file
            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save best individual of population to file, each generation has its own file
            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            # Output time in minutes
            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            # Output objective function values of the managed population
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            worst.append(population[-1]['objective'])
            print()


class Methods():
    # Set parent selection method and population management method, which is quite magical to map strings to function methods.
    def __init__(self,paras,problem) -> None:
        self.paras = paras      
        self.problem = problem
        if paras.selection == "prob_rank":
            self.select = prob_rank
        elif paras.selection == "equal":
            self.select = equal
        elif paras.selection == 'roulette_wheel':
            self.select = roulette_wheel
        elif paras.selection == 'tournament':
            self.select = tournament
        else:
            print("selection method "+paras.selection+" has not been implemented !")
            exit()

        if paras.management == "pop_greedy":
            self.manage = pop_greedy
        elif paras.management == 'ls_greedy':
            self.manage = ls_greedy
        elif paras.management == 'ls_sa':
            self.manage = ls_sa
        else:
            print("management method "+paras.management+" has not been implemented !")
            exit()

        
    def get_method(self):
        # Must run EOH
        if self.paras.method == "eoh":   
            return EOH(self.paras,self.problem,self.select,self.manage)
        else:
            print("method "+self.method+" has not been implemented!")
            exit()

class EVOL:
    # Initialization
    def __init__(self, paras, prob=None, **kwargs):

        print("----------------------------------------- ")
        print("---              Start EoH            ---")
        print("-----------------------------------------")
        # Create folder #
        create_folders(paras.exp_output_path)
        print("- output folder created -")

        self.paras = paras

        print("-  parameters loaded -")

        self.prob = prob

        # Set a random seed
        random.seed(2024)

        
    # Run methods
    def run(self):

        problemGenerator = Probs(self.paras)

        problem = problemGenerator.get_problem()

        methodGenerator = Methods(self.paras,problem)

        method = methodGenerator.get_method()

        method.run()

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")



# Parameter initialization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = "milp_construct", #['milp_construct','bp_online']
                llm_api_endpoint = "your_llm_endpoint", # set your LLM endpoint
                llm_api_key = "your_api_key",   # set your key
                llm_model = "gpt-4o-mini",
                ec_pop_size = 4, # number of samples in each population
                ec_n_pop = 20,  # number of populations
                exp_n_proc = 4,  # multi-core parallel
                exp_debug_mode = False)

# Initialization
evolution = EVOL(paras)

# Run 
evolution.run()