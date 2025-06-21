import os
import shutil
import numpy as np
import json
import random
import time
import pickle
import sys
import types
import re
import warnings
import http.client
import requests
import ast
import concurrent.futures
import heapq
from typing import Sequence, Tuple
from joblib import Parallel, delayed
from gls.gls_run import solve_instance

from selection import prob_rank, equal, roulette_wheel, tournament
from management import pop_greedy, ls_greedy, ls_sa

# Define some necessary classes
class Paras():
    def __init__(self):
        #####################
        ### General settings ###
        #####################
        self.method = 'eoh'                # Selected method
        self.problem = 'tsp_construct'     # Selected problem to solve
        self.selection = None              # Selected individual selection method (how to select individuals from population for evolution)
        self.management = None             # Selected population management method

        #####################
        ### EC settings ###
        #####################
        self.ec_pop_size = 5  # Number of algorithms in each population, default = 10
        self.ec_n_pop = 5 # Number of populations, default = 10
        self.ec_operators = None # Evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
        self.ec_m = 2  # Number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = None  # Weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]

        #####################
        ### LLM settings ###
        #####################
        self.llm_api_endpoint = None # Endpoint for remote LLM, e.g., api.deepseek.com
        self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = None  # Model type for remote LLM, e.g., deepseek-chat

        #####################
        ### Exp settings ###
        #####################
        self.exp_debug_mode = False  # If debug
        self.exp_output_path = "./TSP/rebuttal1"  # Default folder for AEL outputs
        self.exp_n_proc = 1

        #####################
        ### Evaluation settings ###
        #####################
        self.eva_timeout = 600
        self.eva_numba_decorator = False

    def set_parallel(self):
        #########################################
        ### Set the number of threads to the maximum number of machine threads ###
        #########################################
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes}.")

    def set_ec(self):
        ##############################################################
        ### Set population management strategy, parent selection strategy, and evolution strategy with corresponding weights ###
        ##############################################################
        if self.management is None:
            if self.method in ['ael','eoh']:
                self.management = 'pop_greedy'
            elif self.method == 'ls':
                self.management = 'ls_greedy'
            elif self.method == 'sa':
                self.management = 'ls_sa'

        if self.selection is None:
            self.selection = 'prob_rank'

        if self.ec_operators is None:
            if self.method == 'eoh':
                self.ec_operators  = ['e1','e2','m1','m2']
                if self.ec_operator_weights is None:
                    self.ec_operator_weights = [1, 1, 1, 1]
            elif self.method == 'ael':
                self.ec_operators  = ['crossover','mutation']
                if self.ec_operator_weights is None:
                    self.ec_operator_weights = [1, 1]
            elif self.method == 'ls':
                self.ec_operators  = ['m1']
                if self.ec_operator_weights is None:
                    self.ec_operator_weights = [1]
            elif self.method == 'sa':
                self.ec_operators  = ['m1']
                if self.ec_operator_weights is None:
                    self.ec_operator_weights = [1]

        if self.method in ['ls','sa'] and self.ec_pop_size > 1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> Single-point-based, set pop size to 1.")

    def set_evaluation(self):
        #################################
        ### Set evaluation parameters (problem-based) ###
        #################################
        if self.problem == 'bp_online':
            self.eva_timeout = 600
            self.eva_numba_decorator  = True
        elif self.problem == 'tsp_construct':
            self.eva_timeout = 600

    def set_paras(self, *args, **kwargs):
        #################################
        ### Set multi-threading, population strategy, and evaluation ###
        #################################
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Identify and set parallel
        self.set_parallel()

        # Initialize method and EC settings
        self.set_ec()

        # Initialize evaluation settings
        self.set_evaluation()

#######################################
#######################################
### Basic settings are done, now for prompt settings ###
#######################################
#######################################

def create_folders(results_path):
    #####################################################
    ### Create results folder and subfolders for history, pops, and pops_best ###
    #####################################################
    folder_path = os.path.join(results_path, "results")

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Remove the existing folder and its contents (kept commented as in original code)
        # shutil.rmtree(folder_path)

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
    ### Prompt class, defines various prompts and their related returns ###
    #####################################################
    def __init__(self):
        # Task description prompt
        self.prompt_task = "Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. \
You should create a heuristic for me to update the edge distance matrix."
        self.prompt_func_name = "update_edge_distance"
        self.prompt_func_inputs = ['edge_distance', 'local_opt_tour', 'edge_n_used']
        self.prompt_func_outputs = ['updated_edge_distance']
        self.prompt_inout_inf = "'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrices, 'edge_n_used' includes the number of each edge used during permutation."
        self.prompt_other_inf = "All are Numpy arrays."

    def get_task(self):
        #################################
        ### Get task description ###
        #################################
        return self.prompt_task

    def get_func_name(self):
        #################################
        ### Get the name of the prompt function ###
        #################################
        return self.prompt_func_name

    def get_func_inputs(self):
        #################################
        ### Get the inputs of the prompt function ###
        #################################
        return self.prompt_func_inputs

    def get_func_outputs(self):
        #################################
        ### Get the outputs of the prompt function ###
        #################################
        return self.prompt_func_outputs

    def get_inout_inf(self):
        #################################
        ### Get the description of the prompt function's inputs and outputs ###
        #################################
        return self.prompt_inout_inf

    def get_other_inf(self):
        #################################
        ### Get other descriptions of the prompt function ###
        #################################
        return self.prompt_other_inf

#######################################
#######################################
### Time to create problem instances! ###
#######################################
#######################################

class GetData():
    ########################################################
    ### Given instance count and number of cities, get coordinates and distance matrix for each instance ###
    ########################################################
    def __init__(self,n_instance,n_cities):
        self.n_instance = n_instance # Number of instances
        self.n_cities = n_cities     # Number of cities in each instance

    def generate_instances(self):
        np.random.seed(2024)
        instance_data = []
        for _ in range(self.n_instance): # Generate randomly for each instance
            # Randomly generate city coordinates
            coordinates = np.random.rand(self.n_cities, 2)
            # Generate distance matrix between cities
            distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
            instance_data.append((coordinates,distances))
        return instance_data

#@jit(nopython=True) # Kept commented as in original code
def read_coordinates(instance_path, file_name):
    coordinates = []
    optimal_distance = 1E10

    with open(os.path.join(instance_path, file_name), 'r') as file:
        lines = file.readlines()
    
    index = -1
    for i, line in enumerate(lines):
        if line.startswith('NODE_COORD_SECTION'):
            index = i + 1
            break
    
    if index == -1: # Fallback if section not found
        index = 0

    for i in range(index, len(lines)):
        parts = lines[i].split()
        if (parts[0]=='EOF'): break
        coordinates.append([float(parts[1]), float(parts[2])])

    with open(os.path.join(instance_path, "solutions"), 'r') as sol:
        lines = sol.readlines()
    for line in lines:
        if line.startswith(file_name.removesuffix(".tsp")):
            optimal_distance = float(line.split()[2])
            break

    return np.array(coordinates), optimal_distance

#@jit(nopython=True) # Kept commented as in original code
def create_distance_matrix(coordinates):
    distance_matrix = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
    return distance_matrix

#@jit(nopython=True) # Kept commented as in original code
def read_instance(instance_path, filename):
    coord, opt_cost = read_coordinates(instance_path, filename)
    instance = create_distance_matrix(coord)

    return coord, instance, opt_cost

def read_instance_all(instances_path):
    file_names = os.listdir(instances_path)
    coords = []
    instances = []
    opt_costs = []
    names = []
    for filename in file_names:
        if filename.endswith('.tsp'):
            coord, instance, opt_cost = read_instance(instances_path, filename)
            coords.append(coord)
            instances.append(instance)
            opt_costs.append(opt_cost)
            names.append(filename)
    return coords, instances, opt_costs, names

class TSPGLS():
    ###########################################
    ### Create a new TSP problem instance ###
    ###########################################
    def __init__(self) -> None:
        self.n_inst_eva = 3 # A small value for test only
        self.time_limit = 10 # Maximum 10 seconds for each instance
        self.ite_max = 1000 # Maximum number of local searches in GLS for each instance
        self.perturbation_moves = 1 # Moves of each edge in each perturbation
        #path = os.path.dirname(os.path.abspath(__file__)) # Kept commented as in original
        self.instance_path = './tsplib' # ,instances=None,instances_name=None,instances_scale=None (Kept commented as in original)
        self.debug_mode=False

        self.coords,self.instances,self.opt_costs,self.names = read_instance_all(self.instance_path)

        self.prompts = GetPrompts()

    def tour_cost(self,instance, solution, problem_size):
        cost = 0
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def generate_neighborhood_matrix(self,instance):
        instance = np.array(instance)
        n = len(instance)
        neighborhood_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)
            sorted_indices = np.argsort(distances)  # sort indices based on distances
            neighborhood_matrix[i] = sorted_indices

        return neighborhood_matrix

    def evaluateGLS(self,heuristic):
        gaps = np.zeros(self.n_inst_eva)

        for i in range(self.n_inst_eva):
            gap = solve_instance(i, self.opt_costs[i],
                                 self.instances[i],
                                 self.coords[i],
                                 self.time_limit,
                                 self.ite_max,
                                 self.perturbation_moves,
                                 heuristic)
            gaps[i] = gap

        return np.mean(gaps)

    def evaluate(self, code_string):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")

                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                #print(code_string) # Kept commented as in original
                fitness = self.evaluateGLS(heuristic_module)

                return fitness

        except Exception as e:
            #print("Error:", str(e)) # Kept commented as in original
            return None

class Probs():
    ###########################################
    ### Read problem instances or call TSPGLS() to create problem instances ###
    ###########################################
    def __init__(self,paras):
        if not isinstance(paras.problem, str):
            # Read existing problem instances
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "tsp_construct":
            # Create new problem instances
            self.prob = TSPGLS()
            print("- Prob "+paras.problem+" loaded ")
        else:
            print("Problem "+paras.problem+" not found!")

    def get_problem(self):
        # Return problem instance
        return self.prob

#######################################
#######################################
### Now for interaction with the large language model! ###
#######################################
#######################################
class InterfaceAPI:
    #######################################
    ### Call API to communicate with the large language model ###
    #######################################
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint # API endpoint URL
        self.api_key = api_key           # API key
        self.model_LLM = model_LLM       # Name of the large language model to use
        self.debug_mode = debug_mode     # Enable debug mode
        self.n_trial = 5                 # Maximum 5 attempts to get a response

    def get_response(self, prompt_content):
        # Create a JSON formatted string payload_explanation, which includes the model name and message content.
        # This JSON will be used as the request payload.
        payload_explanation = json.dumps(
            {
                # Specify the model to use.
                "model": self.model_LLM,
                # Message content, representing the user's input.
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."}, # Kept commented as in original
                    {"role": "user", "content": prompt_content}
                ],
            }
        )
        # Define request headers
        headers = {
            "Authorization": "Bearer " + self.api_key,           # Contains API key for request authentication.
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",   # Identifies client information for the request.
            "Content-Type": "application/json",                  # Specifies request content type as JSON.
            "x-api2d-no-cache": 1,                               # Custom header to control caching behavior.
        }

        response = None   # Initialize response variable to None
        n_trial = 0       # Initialize attempt count n_trial to 0

        # Start an infinite loop to repeatedly try to get API response until successful or maximum attempts reached
        while True:
            n_trial += 1
            # Check if current attempt count exceeds maximum allowed attempts self.n_trial (5 times).
            # If so, return current response (possibly None) and exit function.
            if n_trial > self.n_trial:
                return response
            try:
                # Create an HTTPS connection to the API endpoint
                conn = http.client.HTTPSConnection(self.api_endpoint)
                # Send a POST request to the /v1/chat/completions endpoint, passing the request payload and headers
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                # Get the response
                res = conn.getresponse()
                # Read the response content
                data = res.read()
                # Convert response content from JSON format to Python dictionary
                json_data = json.loads(data)
                # Extract actual model reply content from JSON response and store it in response variable
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if self.debug_mode:  # If debug mode is enabled, output debug information.
                    print(f"Error in API: {e}. Retrying process...")
                continue

        return response

class InterfaceLLM:
    #######################################
    ### Call InterfaceAPI class to communicate with the large language model ###
    #######################################
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint # API endpoint URL for communication with the language model
        self.api_key = api_key           # API key for authentication
        self.model_LLM = model_LLM       # Name of the language model to use
        self.debug_mode = debug_mode     # Enable debug mode

        print("- Checking LLM API")

        print('Remote LLM API is used ...')
        # Remind if default settings are not changed
        if self.api_key is None or self.api_endpoint is None or self.api_key == 'xxx' or self.api_endpoint == 'xxx':
            print(">> Stop with wrong API setting: Set api_endpoint (e.g., api.chat...) and api_key (e.g., kx-...) !")
            exit()
        # Create an instance of InterfaceAPI class and assign it to self.interface_llm.
        # InterfaceAPI is a class defined above used to handle actual API requests.
        self.interface_llm = InterfaceAPI(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            self.debug_mode,
        )

        # Call get_response method of InterfaceAPI instance, sending a simple request "1+1=?" to test if API connection and configuration are correct
        res = self.interface_llm.get_response("1+1=?")

        # Check if response is None, meaning API request failed or configuration error
        if res is None:
            print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
            exit()

    def get_response(self, prompt_content):
        ##############################################################################
        # Define a method get_response to get LLM response for given content.
        # It accepts one parameter prompt_content, which represents the user-provided prompt content.
        ##############################################################################

        # Call get_response method of InterfaceAPI instance, sending prompt content and getting response.
        response = self.interface_llm.get_response(prompt_content)

        # Return response obtained from InterfaceAPI
        return response

#######################################
#######################################
### Outer layer! Seems ready? Time to prepare for evolution! ###
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
        self.api_endpoint = api_endpoint      # LLM API endpoint for external service interaction.
        self.api_key = api_key                # API key for authentication and authorization.
        self.model_LLM = model_LLM            # Name or identifier of the language model to use.
        self.debug_mode = debug_mode          # Debug mode flag

        # Set LLM using defined InterfaceLLM. Now its get_response(self, prompt_content) function can be used for communication.
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)


    def get_prompt_cross(self,prompts_indivs):
        ##################################################
        ### Generate prompt for crossover method ###
        ##################################################

        # Combine the ideas and corresponding code of algorithms in indivs, forming statements like "The thought and code of the 1st algorithm are..., the 2nd algorithm are..."
        prompt_indiv = ""
        for i in range(len(prompts_indivs)):
            prompt_indiv = prompt_indiv + "No." + str(i+1) + " prompt's tasks assigned to LLM, and objective function value are: \n" + prompts_indivs[i]['prompt'] + "\n" + str(prompts_indivs[i]['objective']) + "\n"
        # 1. Describe the task.
        # 2. Tell the LLM how many algorithms we are providing, and what they are like (combined with prompt_indiv).
        # 3. Request that the LLM create a new prompt that is totally different from the previously given ones but can be motivated from them.
        # 4. Ask the LLM to describe its new prompt and main steps in one sentence, the description must be in parentheses.
        # 5. Emphasize not to provide other explanations, just output as required.
        prompt_content = self.prompt_task + "\n"\
"I have " + str(len(prompts_indivs)) + " existing prompt with objective function value as follows: \n"\
+ prompt_indiv + \
"Please help me create a new prompt that has a totally different form from the given ones but can be motivated from them. \n" + \
"Please describe your new prompt and main steps in one sentence."\
+ "\n" + "Do not give additional explanations!!! Just one sentence." \
+ "\n" + "Do not give additional explanations!!! Just one sentence."
        return prompt_content


    def get_prompt_variation(self,prompts_indivs):
        ##################################################
        ### Prompt to modify a heuristic to improve performance ###
        ##################################################

        # 1. Describe the task.
        # 2. Tell the LLM about 1 algorithm, what it is like, provide both its idea and code.
        # 3. Request that the LLM create a new prompt whose form is different but can be a modified version of the provided algorithm.
        # 4. Ask the LLM to describe its new prompt and main steps in one sentence, the description must be in parentheses.
        # 5. Emphasize not to provide other explanations, just output as required.
        prompt_content = self.prompt_task + "\n"\
"I have one prompt with its objective function value as follows." + \
"prompt description: " + prompts_indivs[0]['prompt'] + "\n" + \
"objective function value:\n" +\
str(prompts_indivs[0]['objective']) + "\n" +\
"Please assist me in creating a new prompt that has a different form but can be a modified version of the algorithm provided. \n" + \
"Please describe your new prompt and main steps in one sentence." \
+ "\n" + "Do not give additional explanations!!! Just one sentence." \
+ "\n" + "Do not give additional explanations!!! Just one sentence."
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
        #print("Begin: 4") # Kept commented as in original
        ##################################################
        ### Get algorithm description and code to generate new heuristics as different as possible from parent heuristics ###
        ##################################################

        # Get the prompt to help LLM create algorithms as different as possible from parent heuristics
        prompt_content = self.get_prompt_cross(parents)
        #print("???", prompt_content) # Kept commented as in original
        response = self.interface_llm.get_response(prompt_content)

        # In debug mode, output the prompt to help LLM create algorithms as different as possible from parent heuristics
        if self.debug_mode:
            print("\n >>> Check prompt for creating algorithm using [ cross ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()

        return response

    def variation(self,parents):
        ###########################################################
        ### Get algorithm description and code for new heuristics that modify current heuristics to improve performance ###
        ###########################################################
        prompt_content = self.get_prompt_variation(parents)
        response = self.interface_llm.get_response(prompt_content)

        if self.debug_mode:
            print("\n >>> Check prompt for creating algorithm using [ variation ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()

        return response

#######################################
#######################################
### Prompt: How to communicate, how to process algorithms, all ready, let's start! ###
#######################################
#######################################
class InterfaceEC_Prompt():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, select,n_p,timeout, problem_type, **kwargs):

        # Set information needed by LLM
        self.pop_size = pop_size                   # Define population size

        self.evol = Evolution_Prompt(api_endpoint, api_key, llm_model, debug_mode, problem_type , **kwargs)  # Evolution type, including i1, e1, e2, m1, m2, can be used for algorithm evolution
        self.m = m                                  # Number of parent algorithms for prompt cross operation
        self.debug = debug_mode                     # Enable debug mode

        # If not in debug mode, suppress warnings
        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select                        # Parent selection method
        self.n_p = n_p                              # Number of processes/threads for parallel execution

        self.timeout = timeout                      # Timeout definition

    # Add a new individual (offspring) to an existing population,
    # provided that this new individual's objective value is not duplicated with other individuals in the population.
    # If no duplicate objective function value, add it and return True, otherwise return False.
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['prompt'] == offspring['prompt']:
                if self.debug:
                    print("Duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def extract_first_quoted_string(self, text):
        # Use regex to match content within the first double quotes
        match = re.search(r'"(.*?)"', text)
        if match:
            text =  match.group(1)  # Extract the first matched content
        prefix = "Prompt: "
        if text.startswith(prefix):
            return text[len(prefix):].strip()  # Remove prefix and leading/trailing spaces
        return text  # If no match, return original string

    # Used to generate offspring individuals based on specified evolutionary operators
    def _get_alg(self,pop,operator):
        #print("Begin: 3") # Kept commented as in original
        # Initialize offspring: Create an offspring dictionary
        offspring = {
            'prompt': None,
            'objective': None,
            'number': None
        }
        off_set = []
        # Get initial prompt
        if operator == "initial_cross":
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
                print("Prompt cross Error:", e)
                prompt_new = None
            offspring["prompt"] = prompt_new
            offspring["objective"] = 1e9
            offspring["number"] = []
        # Generate new prompt through mutation (variation)
        elif operator == "variation":
            parents = self.select.parent_selection(pop,1)
            #print(parents) # Kept commented as in original
            prompt_now = self.evol.variation(parents)
            try:
                prompt_new = self.extract_first_quoted_string(prompt_now)
            except Exception as e:
                print("Prompt variation Error:", e)
                prompt_new = None
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
            #print("Begin: 2") # Kept commented as in original
            # Call _get_alg method to generate offspring individuals from pop based on operator (i1, m1...), and return parent individuals p and offspring individuals offspring
            #print(operator) # Kept commented as in original
            p, offspring, off_set = self._get_alg(pop, operator)

        # If an exception occurs, set offspring to a dictionary containing all None values, and set p to None
        except Exception as e:
            print("get_offspring Error:", e)
            offspring = {
                'prompt': None,
                'objective': None,
                'number': None
            }
            p = None
            off_set = None

        # Return parent individuals p and generated offspring individuals offspring
        return p, offspring, off_set

    def get_algorithm(self, pop, operator):
        # results: Create an empty list results to store generated offspring individuals
        results = []
        try:
            # Generate self.pop_size offspring individuals. Results are stored in the results list, each element is a (p, off) tuple, where p is the parent individual and off is the generated offspring individual.
            if(operator == 'cross' or operator == 'variation'):
                #print("Begin: 1") # Kept commented as in original
                results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size))
            else:
                results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator) for _ in range(1))
        except Exception as e:
            if self.debug:
                print(f"Error in get_algorithm: {e}")
            print("Parallel timeout.")

        time.sleep(2)

        out_p = []   # All parent individuals
        out_off = [] # All offspring individuals

        for p, off, off_set in results:
            out_p.append(p)
            if(operator == 'cross' or operator == 'variation'):
                out_off.append(off)
            else:
                if off_set:
                    for now_off in off_set:
                        out_off.append(now_off)
            # If in debug mode, output offspring individuals
            if self.debug:
                print(f">>> Check offsprings: \n {off}")
        return out_p, out_off

    def population_generation(self, initial_type):
        # Set to 1, meaning 1 round of individuals will be generated
        n_create = 1
        # Create an empty list to store generated initial population individuals
        population = []
        # Loop to generate individuals
        for i in range(n_create):
            _,pop = self.get_algorithm([], initial_type)
            #print(pop) # Kept commented as in original
            for p in pop:
                population.append(p)

        return population

#######################################
#######################################
### Inner layer! Seems ready? Time to prepare for evolution! ###
#######################################
#######################################

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts, **kwargs):

        # Set prompt interface
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()

        # Concatenate input and output names for prompt string
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
        self.api_endpoint = api_endpoint      # LLM API endpoint for external service interaction.
        self.api_key = api_key                # API key for authentication and authorization.
        self.model_LLM = model_LLM            # Name or identifier of the language model to use.
        self.debug_mode = debug_mode          # Debug mode flag

        # Set LLM using defined InterfaceLLM. Now its get_response(self, prompt_content) function can be used for communication.
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

    def get_prompt_initial(self):
        #######################################
        ### Generate prompt for initial strategy ###
        #######################################

        # First describe the task, then describe what the LLM needs to do:
        # 1. First, describe your new algorithm and main steps in one sentence.
        # 2. Next, implement it in Python as a function named self.prompt_func_name.
        # 3. Tell the LLM how many inputs and outputs this function has, and what they are (corresponding to processed self.prompt_func_inputs and self.prompt_func_outputs).
        # 4. Describe some properties of the input and output data.
        # 5. Describe some other supplementary properties.
        # 6. Finally, emphasize not to give additional explanations, just output as required.
        prompt_content = self.prompt_task + "\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
" + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): "\
+ self.joined_inputs + ". The function should return " + str(len(self.prompt_func_outputs)) + " output(s): "\
+ self.joined_outputs + ". " + self.prompt_inout_inf + " "\
+ self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_cross(self,indivs, prompt):
        ##################################################
        ### Generate prompt for new heuristics as different as possible from parent heuristics ###
        ##################################################

        # Combine the ideas and corresponding code of algorithms in indivs, forming statements like "The thought and code of the 1st algorithm are..., the 2nd algorithm are..."
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(i+1) + " algorithm's thought, objective function value, and the corresponding code are: \n" + indivs[i]['algorithm'] + "\n" + str(indivs[i]['objective']) + "\n" + indivs[i]['code'] + "\n"
        # 1. Describe the task.
        # 2. Tell the LLM how many algorithms we are providing, and what they are like (combined with prompt_indiv).
        # 3. Request that the LLM create a new algorithm that is totally different from the previously given algorithms.
        # 4. Ask the LLM to describe its new algorithm and main steps in one sentence, the description must be in parentheses.
        # 5. Next, implement it in Python as a function named self.prompt_func_name.
        # 6. Tell the LLM how many inputs and outputs this function has, and what they are (corresponding to processed self.prompt_func_inputs and self.prompt_func_outputs).
        # 7. Describe some properties of the input and output data.
        # 8. Describe some other supplementary properties.
        # 9. Finally, emphasize not to give additional explanations, just output as required.
        prompt_content = self.prompt_task + "\n"\
"I have " + str(len(indivs)) + " existing algorithm's thought, objective function value with their codes as follows: \n"\
+ prompt_indiv + prompt + "\n" +\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
" + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): "\
+ self.joined_inputs + ". The function should return " + str(len(self.prompt_func_outputs)) + " output(s): "\
+ self.joined_outputs + ". " + self.prompt_inout_inf + " "\
+ self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content


    def get_prompt_variation(self,indiv1, prompt):
        ##################################################
        ### Prompt to modify a heuristic to improve performance ###
        ##################################################

        # 1. Describe the task.
        # 2. Tell the LLM about 1 algorithm, what it is like, provide both its idea and code.
        # 3. Request that the LLM create a new algorithm whose form is different but can be a modified version of the provided algorithm.
        # 4. Ask the LLM to describe its new algorithm and main steps in one sentence, the description must be in parentheses.
        # 5. Next, implement it in Python as a function named self.prompt_func_name.
        # 6. Tell the LLM how many inputs and outputs this function has, and what they are (corresponding to processed self.prompt_func_inputs and self.prompt_func_outputs).
        # 7. Describe some properties of the input and output data.
        # 8. Describe some other supplementary properties.
        # 9. Finally, emphasize not to give additional explanations, just output as required.
        prompt_content = self.prompt_task + "\n"\
"I have one algorithm with its code as follows. \
Algorithm description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n" + \
prompt + "\n" + \
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
" + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): "\
+ self.joined_inputs + ". The function should return " + str(len(self.prompt_func_outputs)) + " output(s): "\
+ self.joined_outputs + ". " + self.prompt_inout_inf + " "\
+ self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def _get_alg(self,prompt_content):
        # Get response from LLM interface for the given prompt_content.
        response = self.interface_llm.get_response(prompt_content)
        # Use regex re.findall(r"\{(.*)\}", response, re.DOTALL) to try to extract the algorithm description enclosed in curly braces {}.
        # The re.DOTALL option allows the regex to match newline characters.
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        # If no algorithm description found within curly braces, use alternative patterns
        if len(algorithm) == 0:
            # If the response contains the word 'python', extract the part from the beginning until the 'python' keyword as the algorithm description.
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
            # If it contains 'import', extract the part from the beginning until 'import' as the algorithm description.
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
            # Otherwise, extract the part from the beginning until 'def' as the algorithm description.
            else:
                algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

        # Try to extract the code section using regex re.findall(r"import.*return", response, re.DOTALL), which starts from 'import' and ends with 'return'.
        code = re.findall(r"import.*return", response, re.DOTALL)
        # If no matching code block is found, try the code block starting from 'def' and ending with 'return'.
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        # If initial extraction of algorithm description or code fails, retry
        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 second and retrying ... ")

            # Call get_response method again to get a new response, and repeat attempts to extract algorithm description and code
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

            # If retry count exceeds 3 (n_retry > 3), exit loop
            if n_retry > 3:
                break
            n_retry +=1

        # Assuming algorithm description and code have been successfully extracted, extract them from the list (i.e., take only the first match result)
        if not algorithm or not code:
            return [None, None]

        algorithm = algorithm[0]
        code = code[0]

        return [code, algorithm]

    def initial(self):
        ##################################################
        ### Get algorithm description and code for creating initial population ###
        ##################################################

        # Get prompt to help LLM create initial population
        prompt_content = self.get_prompt_initial()

        # In debug mode, output prompt to help LLM create initial population
        if self.debug_mode:
            print("\n >>> Check prompt for creating algorithm using [ initial ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()

        # Call _get_alg to input prompt to LLM and split returned text into code and algorithm description
        [code_all, algorithm] = self._get_alg(prompt_content)

        # In debug mode, output returned text split into code and algorithm description
        if self.debug_mode:
            print("\n >>> Check designed algorithm: \n", algorithm)
            print("\n >>> Check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def cross(self, parents, prompt):
        ##################################################
        ### Get algorithm description and code to generate new heuristics as different as possible from parent heuristics ###
        ##################################################

        # Get prompt to help LLM create algorithms as different as possible from parent heuristics
        prompt_content = self.get_prompt_cross(parents, prompt)

        # In debug mode, output prompt to help LLM create algorithms as different as possible from parent heuristics
        if self.debug_mode:
            print("\n >>> Check prompt for creating algorithm using [ cross ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
        # Call _get_alg to input prompt to LLM and split returned text into code and algorithm description
        [code_all, algorithm] = self._get_alg(prompt_content)

        # In debug mode, output returned text split into code and algorithm description
        if self.debug_mode:
            print("\n >>> Check designed algorithm: \n", algorithm)
            print("\n >>> Check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def variation(self,parents, prompt):
        ###########################################################
        ### Get algorithm description and code for new heuristics that modify current heuristics to improve performance ###
        ###########################################################
        prompt_content = self.get_prompt_variation(parents, prompt)

        if self.debug_mode:
            print("\n >>> Check prompt for creating algorithm using [ variation ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> Check designed algorithm: \n", algorithm)
            print("\n >>> Check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

############################################################################################################
### Used to add import statements to given Python code.                                                  ###
### This function inserts an import package_name as as_name statement into the code, if the package is not already imported ###
############################################################################################################
def add_import_package_statement(program: str, package_name: str, as_name=None, *, check_imported=True) -> str:
    """Add 'import package_name as as_name' in the program code."""
    # Use ast.parse() method to parse the input Python code string into an Abstract Syntax Tree (AST)
    tree = ast.parse(program)

    # If check_imported is True, traverse each node in the AST to check if there is already a statement importing the specified package
    if check_imported:
        # Check if 'import package_name' code exists
        package_imported = False
        for node in tree.body:
            # The first part checks if the node is an import statement.
            # The second part checks if the package name in the import statement is the same as package_name.
            if isinstance(node, ast.Import) and any(alias.name == package_name for alias in node.names):
                package_imported = True
                break
        # If the imported package is found, the package_imported flag is set to True, and the unmodified code is returned directly.
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
### Used to add NumPy's @numba.jit(nopython=True) decorator to the given Python code.                    ###
### The decorator is added above a specified function to improve its execution efficiency.             ###
### Numba is a JIT (Just-In-Time) compiler used to accelerate numerical computations.                  ###
############################################################################################################
def _add_numba_decorator(
        program: str,
        function_name: str
) -> str:
    # Use ast.parse() method to parse the input Python code string into an Abstract Syntax Tree (AST)
    tree = ast.parse(program)

    # Traverse each node in the AST tree to check if there is already an 'import numba' statement
    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break

    # If numba is not yet imported, create an 'import numba' node and insert it at the very top of the AST tree
    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)

    for node in ast.walk(tree):
        # Use ast.walk(tree) to traverse all nodes of the AST tree and find function definitions matching the function_name
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Create a @numba.jit(nopython=True) decorator node
            # ast.Call creates a call node, ast.Attribute represents attribute access (i.e., numba.jit), ast.keyword creates a keyword node with named arguments
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='numba', ctx=ast.Load()),
                    attr='jit',
                    ctx=ast.Load()
                ),
                args=[],  # args do not have argument name
                keywords=[ast.keyword(arg='nopython', value=ast.Constant(value=True))] # Changed ast.NameConstant to ast.Constant
            )
            # Add it to the decorator_list attribute of the target function
            node.decorator_list.append(decorator)

    # Use ast.unparse(tree) to convert the modified AST tree back into a Python code string, and return that string
    modified_program = ast.unparse(tree)
    return modified_program

def add_numba_decorator(
        program: str,
        function_name: str | Sequence[str],
) -> str:
    # If function_name is a string, it means only one function needs a decorator.
    # In this case, call the helper function _add_numba_decorator and return its result as the final result.
    if isinstance(function_name, str):
        return _add_numba_decorator(program, function_name)
    # If function_name is a sequence (e.g., list or tuple), iterate through each function name.
    # For each function name, call _add_numba_decorator and update the program.
    for f_name in function_name:
        program = _add_numba_decorator(program, f_name)
    return program

############################################################################################################
### Add a fixed random seed, at the beginning, in the function                                           ###
############################################################################################################
# Used to insert a statement to set the random seed np.random.seed(...) into the specified Python code.
# If the numpy module is not yet imported in the code (i.e., no 'import numpy as np' statement), the function first adds this import statement.
def add_np_random_seed_below_numpy_import(program: str, seed: int = 2024) -> str:
    # This line calls the add_import_package_statement function to ensure 'import numpy as np' is included in the program.
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
            # Insert the 'np.random.seed(seed)' statement after the found 'import numpy as np' statement.
            # This is done by creating a new AST node representing a call to the np.random.seed function.
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
                    args=[ast.Constant(value=seed)],
                    keywords=[]
                )
            )
            tree.body.insert(node_idx + 1, seed_node)
            break
    # If the 'import numpy as np' statement is not found in the code, raise a ValueError exception.
    # This step ensures that numpy is imported before np.random.seed(seed) is inserted.
    if not found_numpy_import:
        raise ValueError("No 'import numpy as np' found in the code.")
    # Use ast.unparse method to convert the modified syntax tree back into a Python code string, and return that string.
    modified_code = ast.unparse(tree)
    return modified_code

# Used to add the np.random.seed(seed) statement within a specified Python function to set the seed for the random number generator.
# This operation is typically used to ensure reproducibility of random number generation across different runs.
def add_numpy_random_seed_to_func(program: str, func_name: str, seed: int = 2024) -> str:
    # This line parses the input code string into an Abstract Syntax Tree (AST).
    tree = ast.parse(program)

    # Insert the new np.random.seed(seed) statement at the beginning of the target function's body.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Create the AST node for np.random.seed(seed)
            seed_stmt = ast.Expr(
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
                    args=[ast.Constant(value=seed)],
                    keywords=[]
                )
            )
            node.body.insert(0, seed_stmt)
            break

    # Convert the entire syntax tree into a new code string containing the seed setting.
    modified_code = ast.unparse(tree)
    return modified_code

############################################################################################################
### Replace division operator (/) with custom protected division function _protected_div in Python code, ###
### and use numba library for acceleration as needed.                                                    ###
############################################################################################################
# First, define a _CustomDivisionTransformer class, which inherits from ast.NodeTransformer.
# Its purpose is to traverse the Abstract Syntax Tree (AST), find all division operations (/), and replace them with calls to a custom division function.
class _CustomDivisionTransformer(ast.NodeTransformer):
    # Accepts a parameter custom_divide_func_name, which is the name of the custom function to be used to replace the division operator. Here, the name is _protected_div.
    def __init__(self, custom_divide_func_name: str):
        super().__init__()
        self._custom_div_func = custom_divide_func_name

    # Used to visit all binary operator nodes. If a division operator (/) is detected, it is replaced with a custom function.
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
    # Define the protected division function _protected_div
    protected_div_str = f'''
def _protected_div(x, y, delta={delta}):
    return x / (y + delta)
    '''
    # Parse the input code string into an AST tree
    tree = ast.parse(code_str)

    # Create _CustomDivisionTransformer instance and traverse AST, find division operations and replace them
    transformer = _CustomDivisionTransformer('_protected_div')
    modified_tree = transformer.visit(tree)

    # Convert the modified AST tree back into a code string, here returning the replaced code and the protected division function definition together
    modified_code = ast.unparse(modified_tree)
    modified_code = '\n'.join([modified_code, '', '', protected_div_str])

    # If numba_accelerate is true, add @numba.jit() decorator to _protected_div function to accelerate computation
    if numba_accelerate:
        modified_code = add_numba_decorator(modified_code, '_protected_div')
    # Return the modified code string and the name of the custom division function
    return modified_code, '_protected_div'

#######################################
#######################################
### How to communicate, how to process algorithms, all ready, let's start! ###
#######################################
#######################################
class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # Set information needed by LLM
        self.pop_size = pop_size                    # Define population size
        self.interface_eval = interface_prob        # TSPGLS() type, can call evaluate function to evaluate algorithm code
        prompts = interface_prob.prompts            # Problem description, input/output prompts, can be used to generate subsequent prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode,prompts, **kwargs)  # Evolution type, including initialization, crossover, mutation, etc.
        self.m = m                                  # Number of parent algorithms for 'cross' operations
        self.debug = debug_mode                     # Enable debug mode

        # If not in debug mode, suppress warnings
        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select                        # Parent selection method
        self.n_p = n_p                              # Number of parallel processes/threads

        self.timeout = timeout                      # Timeout definition
        self.use_numba = use_numba                  # Whether to use numba library to accelerate generated functions

    # Add a new individual (offspring) to an existing population,
    # provided that this new individual's objective value is not duplicated with other individuals in the population.
    # If no duplicate objective function value, add it and return True, otherwise return False.
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("Duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    # Used to check if the given code snippet (code) already exists in any individual in the population.
    # By checking for duplicate code snippets, adding identical individuals multiple times to the population can be avoided.
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    # Used to generate offspring individuals based on specified evolutionary operators
    def _get_alg(self,pop,operator, prompt):
        # Initialize offspring: Create an offspring dictionary
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        # Get initial algorithm
        if operator == "initial":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.initial()
        # Generate algorithms dissimilar to parents
        elif operator == "cross":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.cross(parents, prompt)
        # Generate new algorithms by improving current algorithms
        elif operator == "variation":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.variation(parents[0], prompt)
        # No such operation!
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        # Return selected parent algorithms and generated offspring
        return parents, offspring

    # Used to generate offspring individuals and evaluate their fitness
    def get_offspring(self, pop, operator, prompt):
        try:
            # Call _get_alg method to generate offspring individuals offspring from pop based on operator (initial, cross, variation),
            # and return parent individuals p and offspring individuals offspring
            p, offspring = self._get_alg(pop, operator, prompt)

            # Check if offspring is None (e.g., if code/algorithm extraction failed in _get_alg)
            if offspring['code'] is None:
                raise ValueError("Failed to generate code from LLM.")

            # Use Numba?
            if self.use_numba:
                # Use regex r"def\s+(\w+)\s*\(.*\):" to match function definition
                pattern = r"def\s+(\w+)\s*\(.*\):"
                # Extract function name from offspring['code']
                match = re.search(pattern, offspring['code'])
                if match:
                    function_name = match.group(1)
                    # Call add_numba_decorator method to add Numba decorator to the function
                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    # If function name not found, proceed without Numba or raise error
                    if self.debug:
                        print("Warning: Function name not found for Numba decorator. Proceeding without Numba.")
                    code = offspring['code']
            else:
                code = offspring['code']

            # Handle duplicate code
            n_retry= 1
            while self.check_duplicate(pop, code):
                n_retry += 1
                if self.debug:
                    print("Duplicated code, wait 1 second and retrying ... ")

                # If generated code duplicates existing code in population, regenerate offspring
                p, offspring = self._get_alg(pop, operator, prompt)

                # Check if regeneration failed
                if offspring['code'] is None:
                    raise ValueError("Failed to regenerate code from LLM after duplication.")

                # Apply Numba again if enabled
                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring['code'])
                    if match:
                        function_name = match.group(1)
                        code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                    else:
                        code = offspring['code']
                else:
                    code = offspring['code']

                # Try at most twice
                if n_retry > 1:
                    break

            # If after retries, code is still duplicated or None, signal failure
            if offspring['code'] is None or self.check_duplicate(pop, code):
                 raise ValueError("Failed to generate unique or valid code after retries.")

            # Create thread pool: Use ThreadPoolExecutor to execute evaluation task
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit self.interface_eval.evaluate method for evaluation, passing the generated code code
                future = executor.submit(self.interface_eval.evaluate, code)
                # Get evaluation result fitness, round it to 5 decimal places, and store in offspring['objective']
                fitness = future.result(timeout=self.timeout)
                offspring['objective'] = np.round(fitness, 5)
                # Cancel task to release resources
                future.cancel()

        # If an exception occurs, set offspring to a dictionary containing all None values, and set p to None
        except Exception as e:
            # print(f"Error in get_offspring evaluation: {e}") # Debugging aid, kept commented as in original style
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            p = None

        # Return parent individuals p and generated offspring individuals offspring
        return p, offspring

    def get_algorithm(self, pop, operator, prompt):
        # results: Create an empty list results to store generated offspring individuals
        results = []
        try:
            # Generate self.pop_size offspring individuals. Results are stored in the results list, each element is a (p, off) tuple, where p is the parent individual and off is the generated offspring individual.
            results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator, prompt) for _ in range(self.pop_size))
        except Exception as e:
            if self.debug:
                print(f"Error in get_algorithm parallel execution: {e}")
            print("Parallel timeout.")

        time.sleep(2)

        out_p = []   # All parent individuals
        out_off = [] # All offspring individuals

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            # If in debug mode, output offspring individuals
            if self.debug:
                print(f">>> Check offsprings: \n {off}")
        return out_p, out_off

    def population_generation(self):
        # Set to 2, meaning 2 rounds of individuals will be generated
        n_create = 2
        # Create an empty list to store generated initial population individuals
        population = []
        # Loop to generate individuals
        for i in range(n_create):
            _, pop = self.get_algorithm([], 'initial', None) # Pass None for prompt as it's not used for 'initial'
            for p in pop:
                population.append(p)

        return population

    # Used to generate population based on seed (recorded algorithms), where fitness of each individual is obtained through parallel evaluation.
    def population_generation_seed(self,seeds,n_p):
        # Create an empty list to store generated population individuals
        population = []
        # Evaluate each seed's code using self.interface_eval.evaluate method and calculate its fitness.
        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)
        # Iterate through each seed and its corresponding fitness
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
                print(f"Error in seed algorithm: {e}")
                exit()

        print("Initialization finished! Got "+str(len(seeds))+" seed algorithms.")

        return population

class EOH:

    # Initialization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem      # Define problem
        self.select = select     # Define parent selection method
        self.manage = manage     # Define population management method

        # LLM settings
        self.api_endpoint = paras.llm_api_endpoint  # Define API endpoint URL for communication with language model
        self.api_key = paras.llm_api_key            # API private key
        self.llm_model = paras.llm_model            # Define large language model to use

        # Prompt
        self.pop_size_cross = 2
        self.pop_size_variation = 2
        self.problem_type = "minimization"

        # Experimental settings
        self.pop_size = paras.ec_pop_size  # Population size
        self.n_pop = paras.ec_n_pop        # Number of generations to run

        self.operators = paras.ec_operators   # Define evolution operators, e.g., ['e1', 'e2', 'm1', 'm2']

        self.operator_weights = paras.ec_operator_weights    # Define operator weights [0, 1], higher weight means higher probability of use
        if paras.ec_m > self.pop_size or paras.ec_m < 2:    # Number of parents required for cross operations, must be at least two but not exceed population size
            print("m should not be larger than pop size or smaller than 2, adjusting it to m=2.")
            paras.ec_m = 2
        self.m = paras.ec_m                                  # Set number of parents for cross operations

        self.debug_mode = paras.exp_debug_mode               # Debug mode enabled/disabled
        self.ndelay = 1  # default

        self.output_path = paras.exp_output_path             # Path to save population results

        self.exp_n_proc = paras.exp_n_proc                   # Number of processes set

        self.timeout = paras.eva_timeout                     # Timeout definition

        self.use_numba = paras.eva_numba_decorator           # Whether to use numba library for acceleration

        print("- EoH parameters loaded -")

        # Set random seed
        random.seed(2024)

    # Add new offspring to the population. If debug mode is enabled, compare with existing individuals in the population for redundancy.
    def add2pop(self, population, offspring):
        for off in offspring:
            is_duplicated = False
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("Duplicated result, retrying ... ")
                    is_duplicated = True
                    break
            if not is_duplicated:
                population.append(off)

    def add2pop_prompt(self, population, offspring):
        for off in offspring:
            is_duplicated = False
            for ind in population:
                if ind['prompt'] == off['prompt']:
                    if (self.debug_mode):
                        print("Duplicated result, retrying ... ")
                    is_duplicated = True
                    break
            if not is_duplicated:
                population.append(off)

    # Run EOH
    def run(self):

        print("- Evolution Start -")
        # Record start time
        time_start = time.time()

        # Set problem evaluation interface
        interface_prob = self.prob

        # Set prompt evolution interfaces
        interface_promt_cross = InterfaceEC_Prompt(self.pop_size_cross, self.m, self.api_endpoint, self.api_key, self.llm_model, self.debug_mode, self.select, self.exp_n_proc, self.timeout, self.problem_type)
        interface_promt_variation = InterfaceEC_Prompt(self.pop_size_variation, self.m, self.api_endpoint, self.api_key, self.llm_model, self.debug_mode, self.select, self.exp_n_proc, self.timeout, self.problem_type)
        # Set evolution mode, including initialization, evolution, and management
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba
                                   )

        # Initialize prompt population
        print("Creating initial prompt population:")
        cross_operators = interface_promt_cross.population_generation("initial_cross")
        #cross_operators = self.manage.population_management(cross_operators, self.pop_size_cross) # Kept commented as in original
        variation_operators = interface_promt_variation.population_generation("initial_variation")
        #variation_operators = self.manage.population_management(variation_operators, self.pop_size_variation) # Kept commented as in original
        print("Prompt initial:")

        for prompt in cross_operators:
            print("Cross Prompt:", prompt['prompt'])
        for prompt in variation_operators:
            print("Variation Prompt:", prompt['prompt'])
        print("Initial prompt population has been created!")

        print("=======================================")
        population = []
        print("Creating initial population:")
        population = interface_ec.population_generation()
        population = self.manage.population_management(population, self.pop_size)

        print("Initial population objectives:")
        for off in population:
            print(" Obj:", off['objective'], end="|")
        print()
        print("Initial population has been created!")
        # Save generated population to file
        filename = os.path.join(self.output_path, "results", "pops", "population_generation_0.json")
        with open(filename, 'w') as f:
            json.dump(population, f, indent=5)
        n_start = 0

        print("=======================================")

        worst = []
        delay_turn = 3
        change_flag = 0
        last = -1
        max_k = 4
        # n_pop represents the number of generations to run
        for pop_idx in range(n_start, self.n_pop):
            #print(f" [{na + 1} / {self.pop_size}] ", end="|") # Kept commented as in original
            if change_flag:
                change_flag -= 1
                if change_flag == 0:
                    cross_operators = self.manage.population_management(cross_operators, self.pop_size_cross)
                    for prompt in cross_operators:
                        print("Cross Prompt:", prompt['prompt'])
                    variation_operators = self.manage.population_management(variation_operators, self.pop_size_variation)
                    for prompt in variation_operators:
                        print("Variation Prompt:", prompt['prompt'])

            if len(worst) >= delay_turn and worst[-1] == worst[-delay_turn] and pop_idx - last > delay_turn:
                parents, offsprings = interface_promt_cross.get_algorithm(cross_operators, 'cross')
                #print(offsprings) # Kept commented as in original
                self.add2pop_prompt(cross_operators, offsprings)
                parents, offsprings = interface_promt_cross.get_algorithm(cross_operators, 'variation')
                self.add2pop_prompt(cross_operators, offsprings)
                for prompt in cross_operators:
                    print("Cross Prompt:", prompt['prompt'])
                    prompt["objective"] = 1e9
                    prompt["number"] = []

                parents, offsprings = interface_promt_variation.get_algorithm(variation_operators, 'cross')
                self.add2pop_prompt(variation_operators, offsprings)
                parents, offsprings = interface_promt_variation.get_algorithm(variation_operators, 'variation')
                self.add2pop_prompt(variation_operators, offsprings)
                for prompt in variation_operators:
                    print("Variation Prompt:", prompt['prompt'])
                    prompt["objective"] = 1e9
                    prompt["number"] = []

                change_flag = 2
                last = pop_idx

            # First, look at crossover operations
            for i in range(len(cross_operators)):
                promot = cross_operators[i]["prompt"]
                print(f" OP: cross, [{i + 1} / {len(cross_operators)}] ", end="|")
                parents, offsprings = interface_ec.get_algorithm(population, "cross", promot)
                # Add newly generated offspring to the population. If debug mode is enabled, compare with existing individuals in the population for redundancy.
                self.add2pop(population, offsprings)
                for off in offsprings:
                    print(" Obj:", off['objective'], end="|")
                    if off['objective'] is None:
                        continue

                    if len(cross_operators[i]["number"]) < max_k:
                        heapq.heappush(cross_operators[i]["number"], -off['objective'])
                    else:
                        # If heap is full, and current element is smaller than heap top element, replace heap top element
                        if off['objective'] < -cross_operators[i]["number"][0]:
                            heapq.heapreplace(cross_operators[i]["number"], -off['objective'])  # Replace heap top element

                    cross_operators[i]["objective"] = -sum(cross_operators[i]["number"]) / len(cross_operators[i]["number"])
                # Original commented out block:
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)

                # Add new generation. If population size exceeds, manage population to keep size at most pop_size.
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f"Cross {i + 1}, objective: {cross_operators[i]['objective']}", end = "|")
                print()

            # Next, look at variation operations
            for i in range(len(variation_operators)):
                promot = variation_operators[i]["prompt"]
                print(f" OP: variation, [{i + 1} / {len(variation_operators)}] ", end="|")
                parents, offsprings = interface_ec.get_algorithm(population, "variation", promot)
                # Add newly generated offspring to the population. If debug mode is enabled, compare with existing individuals in the population for redundancy.
                self.add2pop(population, offsprings)
                for off in offsprings:
                    print(" Obj:", off['objective'], end="|")
                    if off['objective'] is None:
                        continue
                    if len(variation_operators[i]["number"]) < max_k:
                        heapq.heappush(variation_operators[i]["number"], -off['objective'])
                    else:
                        # If heap is full, and current element is smaller than heap top element, replace heap top element
                        if off['objective'] < -variation_operators[i]["number"][0]:
                            heapq.heapreplace(variation_operators[i]["number"], -off['objective'])  # Replace heap top element

                    variation_operators[i]["objective"] = -sum(variation_operators[i]["number"]) / len(variation_operators[i]["number"])
                # Original commented out block:
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)

                # Add new generation. If population size exceeds, manage population to keep size at most pop_size.
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f"Variation {i + 1}, objective: {variation_operators[i]['objective']}", end = "|")
                print()

            # Save population to file, each generation has its own file
            filename = os.path.join(self.output_path, "results", "pops", f"population_generation_{pop_idx + 1}.json")
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best individual of the population to file, each generation has its own file
            filename = os.path.join(self.output_path, "results", "pops_best", f"population_generation_{pop_idx + 1}.json")
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            # Output time in minutes
            print(f"--- {pop_idx + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objectives:", end=" ")
            # Output the objective function values of the managed population
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            worst.append(population[-1]['objective'])
            print()

class Methods():
    # Set parent selection methods and population management methods, mapping strings to function methods.
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
            print("Selection method "+paras.selection+" has not been implemented !")
            exit()

        if paras.management == "pop_greedy":
            self.manage = pop_greedy
        elif paras.management == 'ls_greedy':
            self.manage = ls_greedy
        elif paras.management == 'ls_sa':
            self.manage = ls_sa
        else:
            print("Management method "+paras.management+" has not been implemented !")
            exit()

    def get_method(self):
        # Must run EOH
        if self.paras.method == "eoh":
            return EOH(self.paras,self.problem,self.select,self.manage)
        else:
            print("Method "+self.paras.method+" has not been implemented!")
            exit()

class EVOL:
    # Initialization
    def __init__(self, paras, prob=None, **kwargs):

        print("----------------------------------------- ")
        print("---              Start EoH            ---")
        print("-----------------------------------------")
        # Create folder #
        create_folders(paras.exp_output_path)
        print("- Output folder created -")

        self.paras = paras

        print("- Parameters loaded -")

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

        print("> End of Evolution!")
        print("----------------------------------------- ")
        print("---     EoH successfully finished!    ---")
        print("-----------------------------------------")

# Parameter initialization
paras = Paras()

# Set parameters
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = "tsp_construct", #['tsp_construct','bp_online']
                llm_api_endpoint = "your_llm_endpoint", # Set your LLM endpoint
                llm_api_key = "your_api_key",   # Set your key
                llm_model = "gpt-4o-mini",
                ec_pop_size = 4, # Number of samples in each population
                ec_n_pop = 20,  # Number of populations
                exp_n_proc = 8,  # Multi-core parallel
                exp_debug_mode = False)

# Initialization
evolution = EVOL(paras)

# Run
evolution.run()