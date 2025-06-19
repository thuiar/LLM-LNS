import os
import shutil
import copy
import numpy as np
import json
import random
import time
import numpy as np
import pickle
import sys
import types
import re
import time
import warnings
import http.client
import json
from typing import Collection
import requests
import ast
from gurobipy import GRB, read, Model
import concurrent.futures
import heapq
from typing import Sequence, Tuple
from joblib import Parallel, delayed
from pathlib import Path
import traceback
import concurrent.futures

from selection import prob_rank,equal,roulette_wheel,tournament
from management import pop_greedy,ls_greedy,ls_sa
#定义一些必要的类
class Paras():
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'eoh'                #选定使用的方法
        self.problem = 'milp_construct'     #选定解决的问题
        self.selection = None              #选定个体选择方法（种群中如何选出个体进行演化）
        self.management = None             #选定种群的管理方法

        #####################
        ###  EC settings  ###
        #####################
        self.ec_pop_size = 5  # number of algorithms in each population, default = 10
        self.ec_n_pop = 5 # number of populations, default = 10
        self.ec_operators = None # evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
        self.ec_m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = None  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]
        
        #####################
        ### LLM settings  ###
        ##################### 
        self.llm_api_endpoint = None # endpoint for remote LLM, e.g., api.deepseek.com
        self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = None  # model type for remote LLM, e.g., deepseek-chat

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False  # if debug
        self.exp_output_path = "./ACP_MVC/"  # default folder for ael outputs
        self.exp_n_proc = 1
        
        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 5 * 300
        self.prompt_eva_timeout = 30
        self.eva_numba_decorator = False


    def set_parallel(self):
        #################################
        ###  设置线程数量为机器最大线程数  ###
        #################################
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")
    
    def set_ec(self):    
        ###########################################################
        ###  设置种群管理策略，父代选择策略以及演化的策略与对应权重
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
            print("> single-point-based, set pop size to 1. ")
            
    def set_evaluation(self):
        #################################
        ###  设置种群评价的相关参数（基于问题）
        #################################
        if self.problem == 'bp_online':
            self.eva_timeout = 20
            self.eva_numba_decorator  = True
        elif self.problem == 'milp_construct':
            self.eva_timeout = 350 * 5
                
    def set_paras(self, *args, **kwargs):
        #################################
        ###  设置多线程、种群策略和评价
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
###  好了！基本设置好了，开始提示词设置吧
#######################################
#######################################

def create_folders(results_path):
    #####################################################
    ###  创建结果文件夹，并在里面创建历史、种群、种群最佳文件夹
    #####################################################
    folder_path = os.path.join(results_path, "results")

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Remove the existing folder and its contents
        #shutil.rmtree(folder_path)

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
    ###  提示词类，定义了各种提示词和提示词的相关返回
    #####################################################
    def __init__(self):
        #任务描述的提示词，
        
        self.prompt_task = "Given an initial feasible solution and a current solution to a Mixed-Integer Linear Programming (MILP) problem, with a complete description of the constraints and objective function.\
        We want to improve the current solution using Large Neighborhood Search (LNS). \
        The task can be solved step-by-step by starting from the current solution and iteratively selecting a subset of decision variables to relax and re-optimize. \
        In each step, most decision variables are fixed to their values in the current solution, and only a small subset is allowed to change. \
        You need to score all the decision variables based on the information I give you, and I will choose the decision variables with high scores as neighborhood selection.\
        To avoid getting stuck in local optima, the choice of the subset can incorporate a degree of randomness.\
        You can also consider the correlation between decision variables, for example, assigning similar scores to variables involved in the same constraint, which often exhibit high correlation. This will help me select decision variables from the same constraint.\
        Of course, I also welcome other interesting strategies that you might suggest."
        #提示词函数的名字：选择下一个点
        self.prompt_func_name = "select_neighborhood"
        #提示词函数的输入-当前点，终点，没有访问过的点，距离矩阵
        self.prompt_func_inputs = ["n", "m", "k", "site", "value", "constraint", "initial_solution", "current_solution", "objective_coefficient"]
        #提示词函数的输出-下一个点
        self.prompt_func_outputs = ["neighbor_score"]
        #提示词函数的输入输出的描述
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
        This corresponds to the Minimum Vertex Cover MILP problem, where all decision variables are binary (0-1 variables), and all constraints are in the form of LHS ≤ RHS.\
        'neighbor_score' is also a numpy array that you need to create manually. The i-th element of the arrays corresponds to the i-th decision variable."
        #提示词的其他描述
        self.prompt_other_inf = "All are Numpy arrays. I don't give you 'neighbor_score' so that you need to create it manually. The length of the 'neighbor_score' array is also 'n'."

    def get_task(self):
        #################################
        ###  获得任务描述
        #################################
        return self.prompt_task
    
    def get_func_name(self):
        #################################
        ###  获得提示词函数的名字
        #################################
        return self.prompt_func_name
    
    def get_func_inputs(self):
        #################################
        ###  获得提示词函数的输入
        #################################
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        #################################
        ###  获得提示词函数的输出
        #################################
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        #################################
        ###  获得提示词函数的输入输出的描述
        #################################
        return self.prompt_inout_inf

    def get_other_inf(self):
        #################################
        ###  获得提示词函数的其它描述
        #################################
        return self.prompt_other_inf

#######################################
#######################################
###  该开始创建问题示例了！
#######################################
#######################################

class GetData():
    ########################################################
    ###  传入实例数量和城市数，获得每个实例种每个点的坐标以及距离矩阵
    ########################################################
    def generate_instances(self, lp_path): #'./test'
        sample_files = [str(path) for path in Path(lp_path).glob("*.lp")]
        instance_data = []
        for f in sample_files: #对每一个实例都随机生成一下
            model = read(f)
            value_to_num = {}
            value_to_type = {}
            value_num = 0
            #n表示决策变量个数
            #m表示约束数量
            #k[i]表示第i条约束中决策变量数量
            #site[i][j]表示第i个约束的第j个决策变量是哪个决策变量
            #value[i][j]表示第i个约束的第j个决策变量的系数
            #constraint[i]表示第i个约束右侧的数
            #constraint_type[i]表示第i个约束的类型，1表示<，2表示>，3表示=
            #coefficient[i]表示第i个决策变量在目标函数中的系数
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

            #1最小化，-1最大化
            obj_type = model.ModelSense
            model.setObjective(0, GRB.MAXIMIZE)
            model.optimize()
            new_sol = {}
            for val in model.getVars():
                if(val.VarName not in value_to_num.keys()):
                    value_to_num[val.VarName] = value_num
                    value_num += 1
                new_sol[value_to_num[val.VarName]] = val.x
            
            #后处理一下
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
    ###  创建全新的 TSP 问题实例
    ###########################################
    def __init__(self) -> None:
        self.path = "./test"
        self.set_time = 100
        self.n_p = 5 #测试时间
        self.epsilon = 1e-3

        self.prompts = GetPrompts()
        #调用定义好的GetData(），获得随机生成的问题实例
        getData = GetData()
        self.instance_data = getData.generate_instances(self.path)
        #print(self.instance_data[0])
    
    def Gurobi_solver(self, n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, value_type, now_sol, now_col):
        '''
        函数说明：
        根据传入的问题实例，使用Gurobi求解器进行求解。

        参数说明：
        - n: 问题实例的决策变量数量。
        - m: 问题实例的约束数量。
        - k: k[i]表示第i条约束的决策变量数量。
        - site: site[i][j]表示第i个约束的第j个决策变量是哪个决策变量。
        - value: value[i][j]表示第i个约束的第j个决策变量的系数。
        - constraint: constraint[i]表示第i个约束右侧的数。
        - constraint_type: constraint_type[i]表示第i个约束的类型，1表示<=，2表示>=
        - coefficient: coefficient[i]表示第i个决策变量在目标函数中的系数。
        - time_limit: 最大求解时间。
        - obj_type: 问题是最大化问题还是最小化问题。
        '''
        #获得起始时间
        begin_time = time.time()
        #定义求解模型
        model = Model("Gurobi")
        #设定变量映射
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
        
        #设定目标函数和优化目标（最大化/最小化）
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
        #添加m条约束
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
        #设定最大求解时间
        model.setParam('OutputFlag', 0)
        if(time_limit - (time.time() - begin_time) <= 0):
            return -1, -1, -1
        model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
        #优化求解
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
        """在 time_limit 秒内运行函数 func，超时则返回 None"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                # 等待结果，最多 time_limit 秒
                return future.result(timeout=time_limit)
            except concurrent.futures.TimeoutError:
                # 超时，返回 None
                print(f"Function {func.__name__} timed out after {time_limit} seconds.")
                return None

    def greedy(self, eva):
        ###############################################################################
        ###  使用类似于贪心的方法，每一步都通过eva中的select_next_node选择下一个点
        ###  跑多个实例，以多个实例结果的均值作为最后返回的结果
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
            #生成 self.pop_size 个后代个体。结果存储在 results 列表中，每个元素是一个 (p, off) 元组，其中 p 是父代个体，off 是生成的后代个体
            #results = Parallel(n_jobs=self.n_p,timeout=self.set_time+30)(delayed(self.greedy_one)(now_instance_data, eva) for now_instance_data in self.instance_data)
        except Exception as e:
            print(f"Parallel MILP Error: {e}")
            traceback.print_exc()
            results = [1e9]
        
        return sum(results) / len(results)


    def evaluate(self, code_string):
        ###############################################################################
        ###  调用 greedy 评测当前策略的适应度
        ###  关键问题：当前策略是字符串，怎么变成能跑的代码块呢？
        ###############################################################################
        try:
            # 使用 warnings.catch_warnings() 以捕获和控制在代码执行过程中产生的警告
            with warnings.catch_warnings():
                #将捕获的警告设置为忽略模式。这意味着在此代码块中，任何产生的警告都会被忽略，不会显示给用户
                warnings.simplefilter("ignore")
                
                #创建一个新的模块对象，命名为 "heuristic_module"。types.ModuleType 创建了一个新的空模块，类似于一个容器，用于存放将要执行的代码
                heuristic_module = types.ModuleType("heuristic_module")
                
                #使用 exec 函数在 heuristic_module 模块的命名空间中执行 code_string
                #exec 可以动态地执行字符串形式的代码，并将执行结果存放在指定的命名空间中
                exec(code_string, heuristic_module.__dict__)

                #将新创建的模块添加到 sys.modules 中，使其在程序中可以像普通模块一样被导入。sys.modules 是一个字典，保存了所有已导入的模块
                #通过将 heuristic_module 添加到这个字典中，其他部分的代码可以使用 import heuristic_module 来访问它
                sys.modules[heuristic_module.__name__] = heuristic_module

                #调用类中的一个方法 self.greedy，并将 heuristic_module 作为参数传入
                #这一行代码将返回一个 fitness 值，基于传入的 heuristic_module 中定义的逻辑。
                fitness = self.greedy(heuristic_module)
                #如果没有异常发生，返回计算出的 fitness 值
                return fitness
        except Exception as e:
            #在发生异常时返回 None，表示代码执行失败或未能成功计算出 fitness 值
            print(f"Greedy MILP Error: {e}")
            return None


class Probs():
    ###########################################
    ###  读入问题实例或调用PROBLEMCONST()创建问题实例
    ###########################################
    def __init__(self,paras):
        if not isinstance(paras.problem, str):
            #读入已有的问题实例
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "milp_construct":
            #创建新的问题实例
            self.prob = PROBLEMCONST()
            print("- Prob "+paras.problem+" loaded ")
        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):
        #返回问题实例
        return self.prob


#######################################
#######################################
###  终于来到和大模型交流的板块了！
#######################################
#######################################
class InterfaceAPI:
    #######################################
    ###  调用 API 与大模型通讯
    #######################################
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint #API 的终端地址
        self.api_key = api_key           #API 密钥
        self.model_LLM = model_LLM       #使用的大语言模型名称
        self.debug_mode = debug_mode     #是否启用调试模式
        self.n_trial = 5                 #表示在尝试获取响应时最多进行 5 次尝试

    def get_response(self, prompt_content):
        #创建一个 JSON 格式的字符串 payload_explanation，包含了模型名称和消息内容。这个 JSON 将被用作请求的负载
        payload_explanation = json.dumps(
            {
                #指定要使用的模型。
                "model": self.model_LLM,  
                #消息内容，表示用户的输入。
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
            }
        )
        # 定义请求的头部信息（headers）
        headers = {
            "Authorization": "Bearer " + self.api_key,           #包含 API 密钥，用于认证请求。
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",   #标识请求的客户端信息。
            "Content-Type": "application/json",                  #指定请求的内容类型为 JSON。
            "x-api2d-no-cache": 1,                               #用于控制缓存行为的自定义标头。
        }
        
        response = None   #初始化变量 response 为 None，用于存储 API 返回的响应内容
        n_trial = 1       #初始化尝试次数 n_trial 为 1，表示将开始第一次尝试
        
        #开始一个无限循环，用于反复尝试获取 API 响应，直到成功或达到最大尝试次数
        while True:
            n_trial += 1
            #检查当前尝试次数是否超过了最大允许次数 self.n_trial（5 次）。如果超过，返回当前的 response（可能为 None），并退出函数
            if n_trial > self.n_trial:
                return response
            try:
                #创建一个到 API 终端的 HTTPS 连接
                conn = http.client.HTTPSConnection(self.api_endpoint)
                #通过 POST 请求方法向 /v1/chat/completions 端点发送请求，并传递请求负载和头部信息
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                #获取请求的响应
                res = conn.getresponse()
                #读取响应的内容
                data = res.read()
                #将响应内容从 JSON 格式转换为 Python 字典
                json_data = json.loads(data)
                #从 JSON 响应中提取出实际的模型回复内容，并将其存储在 response 变量中
                response = json_data["choices"][0]["message"]["content"]

                #server_b_url = "http://43.134.189.32:5000/openai"
                #response = requests.post(server_b_url, json={"prompt": prompt_content}).json()['response']
                break
            except:
                if self.debug_mode:  #如果启用了调试模式，输出调试信息。
                    print("Error in API. Restarting the process...")
                continue
            
        return response


class InterfaceLLM:
    #######################################
    ###  调用 InterfaceAPI 类与大模型通讯
    #######################################
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint #API的端点URL，用于与语言模型通信
        self.api_key = api_key           #API密钥，用于身份验证
        self.model_LLM = model_LLM       #使用的语言模型名称
        self.debug_mode = debug_mode     #是否启用调试模式

        print("- check LLM API")

        print('remote llm api is used ...')
        #如果没有更改默认设置，提醒一下
        if self.api_key == None or self.api_endpoint ==None or self.api_key == 'xxx' or self.api_endpoint == 'xxx':
            print(">> Stop with wrong API setting: Set api_endpoint (e.g., api.chat...) and api_key (e.g., kx-...) !")
            exit()
        #创建一个 InterfaceAPI 类的实例，并将其赋值给 self.interface_llm。InterfaceAPI 是一个上面定义的用于实际处理API请求的类。
        self.interface_llm = InterfaceAPI(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            self.debug_mode,
        )

        #调用 InterfaceAPI 实例的 get_response 方法，发送一个简单的请求 "1+1=?" 以测试API连接和配置是否正确   
        res = self.interface_llm.get_response("1+1=?")

        #检查响应是否为 None，这意味着API请求失败或配置错误
        if res == None:
            print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
            exit()

    def get_response(self, prompt_content):
        ##############################################################################
        #定义一个方法 get_response，用于获取LLM对给定内容的响应。
        #它接受一个参数 prompt_content，表示用户提供的提示内容。
        ##############################################################################

        #调用 InterfaceAPI 实例的 get_response 方法，发送提示内容并获取响应。
        response = self.interface_llm.get_response(prompt_content)

        #返回从 InterfaceAPI 获取的响应
        return response


#######################################
#######################################
###  外层！似乎都准备好了？可以开始准备进化了！
#######################################
#######################################
class Evolution_Prompt():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, problem_type, **kwargs):
        #problem_type:minimization/maximization

        self.prompt_task = "We are working on solving a " + problem_type + " problem." + \
        " Our objective is to leverage the capabilities of the Language Model (LLM) to generate heuristic algorithms that can efficiently tackle this problem." + \
        " We have already developed a set of initial prompts and observed the corresponding outputs." + \
        " However, to improve the effectiveness of these algorithms, we need your assistance in carefully analyzing the existing prompts and their results." + \
        " Based on this analysis, we ask you to generate new prompts that will help us achieve better outcomes in solving the " + problem_type + " problem."

        #设置大模型参数
        self.api_endpoint = api_endpoint      #LLM API的端点，用于与外部服务交互。
        self.api_key = api_key                #API密钥，用于认证和授权。
        self.model_LLM = model_LLM            #使用的语言模型的名称或标识符。
        self.debug_mode = debug_mode          #调试模式标志

        #使用定义好的InterfaceLLM设置 LLM，接下来就可以使用其get_response(self, prompt_content)函数来通讯
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)
    
        
    def get_prompt_cross(self,prompts_indivs):
        ##################################################
        ###  生成 prompt 的交叉方式
        ##################################################
        
        #将indivs中的算法的思想和对应的代码组合起来，形成第 1 个算法和对应代码是……，第 2 个算法和对应代码是……这样的语句
        prompt_indiv = ""
        for i in range(len(prompts_indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" prompt's tasks assigned to LLM, and objective function value are: \n" + prompts_indivs[i]['prompt']+"\n" + str(prompts_indivs[i]['objective']) +"\n"
        #1.描述任务
        #2.告诉 LLM 我们告诉它几个算法，分别怎么样（结合prompt_indiv）
        #3.提出请求，希望 LLM 创造一个完全不同于之前给出算法的算法
        #4.告诉LLM回答先用一句话描述你的新算法和主要步骤，描述必须在括号内。
        #5.接下来，在 Python 中将其实现为一个名为 self.prompt_func_name的函数
        #6.告诉 LLM 这个函数有多少个输入输出，以及什么输入和什么输出（对应处理好的self.prompt_func_inputs和self.prompt_func_outputs）
        #7.描述一下输入输出数据的一些性质。
        #8.描述一下一些其它补充的性质。
        #9.最后强调不要做其他解释，老老实实按要求输出就行。
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
        ###  修改一个启发式以提高性能的 prompt
        ##################################################
        
        #1.描述任务
        #2.告诉 LLM 我们告诉它1个算法， 是怎么样的，把想法和代码都告诉它
        #3.提出请求，希望 LLM 创建一种新算法，它的形式不同，但可以是所提供算法的修改版
        #4.告诉LLM回答先用一句话描述你的新算法和主要步骤，描述必须在括号内。
        #5.接下来，在 Python 中将其实现为一个名为 self.prompt_func_name的函数
        #6.告诉 LLM 这个函数有多少个输入输出，以及什么输入和什么输出（对应处理好的self.prompt_func_inputs和self.prompt_func_outputs）
        #7.描述一下输入输出数据的一些性质。
        #8.描述一下一些其它补充的性质。
        #9.最后强调不要做其他解释，老老实实按要求输出就行。
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
        ###  获得生成尽可能不同于父启发式的新启发式的算法描述和代码
        ##################################################

        #获得给 LLM 帮忙创建尽可能不同于父启发式算法的 prompt
        prompt_content = self.get_prompt_cross(parents)
        #print("???", prompt_content)
        response = self.interface_llm.get_response(prompt_content)

        #debug 模式下输出给 LLM 帮忙创建尽可能不同于父启发式算法的 prompt
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ cross ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            

        return response
    
    def variation(self,parents):
        ###########################################################
        ###  获得修改当前启发式以提高性能的新启发式算法描述和代码
        ###########################################################
        prompt_content = self.get_prompt_variation(parents)
        response = self.interface_llm.get_response(prompt_content)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ variation ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            
    
        return response

#######################################
#######################################
###  prompt:如何通信如何处理算法都准备好了，开始吧！
#######################################
#######################################
class InterfaceEC_Prompt():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, select,n_p,timeout, problem_type, **kwargs):

        # 设置LLM 需要的一些信息
        self.pop_size = pop_size                   #定义种群大小

        self.evol = Evolution_Prompt(api_endpoint, api_key, llm_model, debug_mode, problem_type , **kwargs)  #Evolution类型，有 i1，e1,e2,m1,m2，可以用于算法的演化
        self.m = m                                  #prompt 的 cross操作父代的算法数量
        self.debug = debug_mode                     #debug 模式是否开启

        #如果不开 debug 模式，warning 都不显示
        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select                        #父代的选择方式
        self.n_p = n_p                              #种群大小
        
        self.timeout = timeout                      #超时时间定义
    
    #将文本代码写到文件./ael_alg.py中
    def prompt2file(self,prompt):
        with open("./prompt.txt", "w") as file:
        # Write the code to the file
            file.write(prompt)
        return 
    
    #将一个新的个体（offspring）添加到已有的种群（population）中
    #前提是这个新的个体在目标值（objective）上与种群中的其他个体没有重复
    #如果没有重复的目标函数值，加进去并返回 True，反之返回 False
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['prompt'] == offspring['prompt']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def extract_first_quoted_string(self, text):
        # 使用正则表达式匹配第一个双引号中的内容\
        match = re.search(r'"(.*?)"', text)
        if match:
            text =  match.group(1)  # 提取出第一个匹配的内容
        prefix = "Prompt: "
        if text.startswith(prefix):
            return text[len(prefix):].strip()  # 移除前缀并去除可能的前后空格
        return text  # 如果没有匹配到，返回原始字符串
    
    
    #用于根据指定的进化操作符生成后代个体
    def _get_alg(self,pop,operator):
        #print("Begin: 3")
        #初始化后代: 创建一个字典 offspring
        offspring = {
            'prompt': None,
            'objective': None,
            'number': None
        }
        off_set = []
        #获得初始prompt
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
        #通过交叉生成新的 prompt      
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
        # 通过变异生成新的 prompt
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
        #没有这样的操作！
        else:
            print(f"Prompt operator [{operator}] has not been implemented ! \n") 

        #同时返回选出来的父代算法和生成的子代
        return parents, offspring, off_set

    #用于生成后代个体并评估其适应度
    def get_offspring(self, pop, operator):
        try:
            #print("Begin: 2")
            #调用 _get_alg 方法，根据 operator（i1，m1……） 从 pop 中生成后代个体 offspring，并返回父代个体 p 和后代个体 offspring
            #print(operator)
            p, offspring, off_set = self._get_alg(pop, operator)
            
        #如果发生异常，设置 offspring 为包含所有 None 值的字典，并将 p 设置为 None
        except Exception as e:
            print("get_offspring", e)
            offspring = {
                'prompt': None,
                'objective': None,
                'number': None
            }
            p = None
            off_set = None

        #返回父代个体 p 和生成的后代个体 offspring
        return p, offspring, off_set
    
    def get_algorithm(self, pop, operator):
        #results: 创建一个空列表 results 用于存储生成的后代个体
        results = []
        try:
            #生成 self.pop_size 个后代个体。结果存储在 results 列表中，每个元素是一个 (p, off) 元组，其中 p 是父代个体，off 是生成的后代个体
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


        out_p = []   #所有父代个体
        out_off = [] #所有后代个体

        for p, off, off_set in results:
            out_p.append(p)
            if(operator == 'cross' or operator == 'variation'):
                out_off.append(off)
            else:
                for now_off in off_set:
                    out_off.append(now_off)
            #如果是 debug 模式输出后代个体
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off

    def population_generation(self, initial_type):
        #设定为 2，表示要生成 2 轮的个体
        n_create = 1
        #创建一个空列表，用于存储生成的初始种群个体
        population = []
        #循环生成个体
        for i in range(n_create):
            _,pop = self.get_algorithm([], initial_type)
            #print(pop)
            for p in pop:
                population.append(p)
             
        return population

    
#######################################
#######################################
###  内层！似乎都准备好了？可以开始准备进化了！
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
        
        #["current_node","destination_node","univisited_nodes","distance_matrix"] -> "'current_node','destination_node','univisited_nodes','distance_matrix'"
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        #["next_node"] -> "'next_node'"
        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        #设置大模型参数
        self.api_endpoint = api_endpoint      #LLM API的端点，用于与外部服务交互。
        self.api_key = api_key                #API密钥，用于认证和授权。
        self.model_LLM = model_LLM            #使用的语言模型的名称或标识符。
        self.debug_mode = debug_mode          #调试模式标志

        #使用定义好的InterfaceLLM设置 LLM，接下来就可以使用其get_response(self, prompt_content)函数来通讯
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

    def get_prompt_initial(self):
        #######################################
        ###  生成初始策略的 prompt
        #######################################

        #首先描述任务，然后描述需要 LLM 做的事情：
        #1.首先，用一句话描述你的新算法和主要步骤，描述必须在括号内。
        #2.接下来，在 Python 中将其实现为一个名为 self.prompt_func_name的函数
        #3.告诉 LLM 这个函数有多少个输入输出，以及什么输入和什么输出（对应处理好的self.prompt_func_inputs和self.prompt_func_outputs）
        #4.描述一下输入输出数据的一些性质。
        #5.描述一下一些其它补充的性质。
        #6.最后强调不要做其他解释，老老实实按要求输出就行。
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
        ###  生成尽可能不同于父启发式的新启发式的 prompt
        ##################################################
        
        #将indivs中的算法的思想和对应的代码组合起来，形成第 1 个算法和对应代码是……，第 2 个算法和对应代码是……这样的语句
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm's thought, objective function value, and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" + str(indivs[i]['objective']) +"\n" +indivs[i]['code']+"\n"
        #1.描述任务
        #2.告诉 LLM 我们告诉它几个算法，分别怎么样（结合prompt_indiv）
        #3.提出请求，希望 LLM 创造一个完全不同于之前给出算法的算法
        #4.告诉LLM回答先用一句话描述你的新算法和主要步骤，描述必须在括号内。
        #5.接下来，在 Python 中将其实现为一个名为 self.prompt_func_name的函数
        #6.告诉 LLM 这个函数有多少个输入输出，以及什么输入和什么输出（对应处理好的self.prompt_func_inputs和self.prompt_func_outputs）
        #7.描述一下输入输出数据的一些性质。
        #8.描述一下一些其它补充的性质。
        #9.最后强调不要做其他解释，老老实实按要求输出就行。
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
        ###  修改一个启发式以提高性能的 prompt
        ##################################################
        
        #1.描述任务
        #2.告诉 LLM 我们告诉它1个算法， 是怎么样的，把想法和代码都告诉它
        #3.提出请求，希望 LLM 创建一种新算法，它的形式不同，但可以是所提供算法的修改版
        #4.告诉LLM回答先用一句话描述你的新算法和主要步骤，描述必须在括号内。
        #5.接下来，在 Python 中将其实现为一个名为 self.prompt_func_name的函数
        #6.告诉 LLM 这个函数有多少个输入输出，以及什么输入和什么输出（对应处理好的self.prompt_func_inputs和self.prompt_func_outputs）
        #7.描述一下输入输出数据的一些性质。
        #8.描述一下一些其它补充的性质。
        #9.最后强调不要做其他解释，老老实实按要求输出就行。
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
        #通过 LLM 接口获取给定提示词（prompt_content）的响应。
        #print("QwQ~!")
        response = self.interface_llm.get_response(prompt_content)

        if self.debug_mode:
            print("\n >>> check response for creating algorithm using [ i1 ] : \n", response )
            print(">>> Press 'Enter' to continue")
            
        #使用正则表达式 re.findall(r"\{(.*)\}", response, re.DOTALL) 尝试从响应中提取包含在花括号 {} 内的算法描述。
        #re.DOTALL 选项允许正则表达式匹配换行符。
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        #如果没有找到花括号内的算法描述，使用替代模式
        if len(algorithm) == 0:
            #如果响应中包含单词 'python'，提取从开头到 'python' 关键字之前的部分，为算法描述。
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
            #如果包含 'import'，提取从开头到 'import' 之前的部分，为算法描述。
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
            #否则，提取从开头到 'def' 之前的部分，为算法描述。
            else:
                algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

        #尝试使用正则表达式 re.findall(r"import.*return", response, re.DOTALL) 提取代码部分，该代码部分从 import 开始到 return 结束。
        code = re.findall(r"import.*return", response, re.DOTALL)
        #如果没有找到符合条件的代码块，则尝试从 def 开始到 return 结束的代码块。
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        #如果初始提取算法描述或代码失败，重试
        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            #再次调用 get_response 方法获取新的响应，并重复尝试提取算法描述和代码
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
            
            #如果重试次数超过 3 次（n_retry > 3），则退出循环
            if n_retry > 3:
                break
            n_retry += 1

        #假设算法描述和代码已经提取成功，将它们从列表中提取出来（即只取第一个匹配结果）
        algorithm = algorithm[0]
        code = code[0] 

        #提取的代码直道 return，我们补上后面的。将提取的代码与输出变量（存储在 self.prompt_func_outputs 中）连接起来形成一个完整的代码字符串。
        code_all = code+" "+", ".join(s for s in self.prompt_func_outputs) 


        return [code_all, algorithm]


    def initial(self):
        ##################################################
        ###  获得创建初始种群的算法描述和代码
        ##################################################
        
        #获得给 LLM 帮忙创建初始种群的 prompt
        prompt_content = self.get_prompt_initial()

        #debug 模式下输出给 LLM 帮忙创建初始种群的 prompt
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            
        #print("QAQ~!")
        #调用_get_alg，将 prompt 输入给 llm，并将返回的文本拆分成代码和算法描述
        [code_all, algorithm] = self._get_alg(prompt_content)

        #debug 模式下返回的文本拆分成代码和算法描述
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            

        return [code_all, algorithm]
    
    def cross(self, parents, prompt):
        ##################################################
        ###  获得生成尽可能不同于父启发式的新启发式的算法描述和代码
        ##################################################

        #获得给 LLM 帮忙创建尽可能不同于父启发式算法的 prompt
        prompt_content = self.get_prompt_cross(parents, prompt)

        #debug 模式下输出给 LLM 帮忙创建尽可能不同于父启发式算法的 prompt
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            
        #调用_get_alg，将 prompt 输入给 llm，并将返回的文本拆分成代码和算法描述
        [code_all, algorithm] = self._get_alg(prompt_content)

        #debug 模式下返回的文本拆分成代码和算法描述
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            

        return [code_all, algorithm]
    
    def variation(self,parents, prompt):
        ###########################################################
        ###  获得修改当前启发式以提高性能的新启发式算法描述和代码
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
###  用于向给定的 Python 代码中添加导入语句。
###  该函数的功能是向代码中插入一个 import package_name as as_name 形式的导入语句，如果该包尚未导入的话
############################################################################################################
def add_import_package_statement(program: str, package_name: str, as_name=None, *, check_imported=True) -> str:
    """Add 'import package_name as as_name' in the program code.
    """
    #使用 ast.parse() 方法将输入的 Python 代码字符串解析为抽象语法树（AST）
    tree = ast.parse(program)

    #如果 check_imported 为 True，则遍历 AST 中的每个节点以检查是否已经有导入指定包的语句
    if check_imported:
        # check if 'import package_name' code exists
        package_imported = False
        for node in tree.body:
            #前半句检查节点是否是 import 语句
            #后半句检查 import 语句的包名是否与 package_name 相同
            if isinstance(node, ast.Import) and any(alias.name == package_name for alias in node.names):
                package_imported = True
                break
        #如果找到已经导入的包，package_imported 标志会被设为 True，并且直接返回未经修改的代码
        if package_imported:
            return ast.unparse(tree)

    #创建一个新的 import 节点。使用 ast.Import 创建一个新的导入语句节点，包名为 package_name，别名为 as_name（如果有的话
    import_node = ast.Import(names=[ast.alias(name=package_name, asname=as_name)])
    #将新的导入节点插入到 AST 的最顶部
    tree.body.insert(0, import_node)
    #使用 ast.unparse(tree) 将修改后的 AST 转换回 Python 代码字符串并返回
    program = ast.unparse(tree)
    return program


############################################################################################################
###  用于向给定的 Python 代码中添加 NumPy 的 @numba.jit(nopython=True) 装饰器
###  装饰器被添加到一个指定的函数之上，以提高该函数的执行效率
###  Numba 是一个用于加速数值计算的 JIT (Just-In-Time) 编译器
############################################################################################################
def _add_numba_decorator(
        program: str,
        function_name: str
) -> str:
    #使用 ast.parse() 方法将输入的 Python 代码字符串解析为抽象语法树（AST）
    tree = ast.parse(program)

    #遍历 AST 树的每个节点，检查是否已经存在 import numba 的语句
    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break

    #如果 numba 尚未导入，则创建一个 import numba 节点，并将其插入 AST 树的最顶部
    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)

    
    for node in ast.walk(tree):
        #使用 ast.walk(tree) 遍历 AST 树的所有节点，找到与 function_name 名称匹配的函数定义
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            #创建一个 @numba.jit(nopython=True) 装饰器节点
            #ast.Call 创建一个调用节点，ast.Attribute 表示属性访问（即 numba.jit），ast.keyword 创建带有命名参数的关键字节点
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
            #将其添加到目标函数的 decorator_list 属性中
            node.decorator_list.append(decorator)

    #使用 ast.unparse(tree) 将修改后的 AST 树转换回 Python 代码字符串，并返回该字符串
    modified_program = ast.unparse(tree)
    return modified_program


def add_numba_decorator(
        program: str,
        function_name: str | Sequence[str],
) -> str:
    #如果 function_name 是一个字符串，表示只需要给一个函数添加装饰器。此时，调用辅助函数 _add_numba_decorator 并将其返回的结果作为最终结果
    if isinstance(function_name, str):
        return _add_numba_decorator(program, function_name)
    #如果 function_name 是一个序列（如列表或元组），则遍历每个函数名。对于每个函数名，调用 _add_numba_decorator 并更新 program
    for f_name in function_name:
        program = _add_numba_decorator(program, f_name)
    return program


############################################################################################################
##   添加一下固定的随机种子，在开头，在函数
############################################################################################################
#用于在指定的 Python 代码中插入一个设定随机种子的语句  np.random.seed(...)
#如果代码中还没有导入 numpy 模块（即 import numpy as np 语句），该函数首先添加这条导入语句
def add_np_random_seed_below_numpy_import(program: str, seed: int = 2024) -> str:
    #该行调用 add_import_package_statement 函数，确保程序中包含 import numpy as np
    program = add_import_package_statement(program, 'numpy', 'np')
    #使用 Python 的 ast（抽象语法树）模块将代码解析为一个语法树
    tree = ast.parse(program)

    # find 'import numpy as np'
    found_numpy_import = False

    # find 'import numpy as np' statement
    for node in tree.body:
        #循环遍历语法树的节点，查找 import numpy as np 语句
        if isinstance(node, ast.Import) and any(alias.name == 'numpy' and alias.asname == 'np' for alias in node.names):
            found_numpy_import = True
            #在找到的 import numpy as np 语句之后插入 np.random.seed(seed) 语句。这通过创建一个新的 AST 节点来实现，表示对 np.random.seed 函数的调用
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
    #如果在代码中找不到 import numpy as np 语句，抛出一个 ValueError 异常。这一步确保在插入 np.random.seed(seed) 之前，numpy 已被导入
    if not found_numpy_import:
        raise ValueError("No 'import numpy as np' found in the code.")
    #使用 ast.unparse 方法将修改后的语法树转换回 Python 代码字符串，并返回该字符串。
    modified_code = ast.unparse(tree)
    return modified_code

#用于在指定的 Python 函数内添加 np.random.seed(seed) 语句，以设置随机数生成器的种子
#这个操作通常用于确保随机数生成在不同运行中具有可重复性。下面是对这段代码的详细解释。
def add_numpy_random_seed_to_func(program: str, func_name: str, seed: int = 2024) -> str:
    #这一行将输入的代码字符串解析为一个抽象语法树 (AST)
    tree = ast.parse(program)

    #将新的 np.random.seed(seed) 语句插入到目标函数体的开头
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            node.body = [ast.parse(f'np.random.seed({seed})').body[0]] + node.body
    
    #将整个语法树转换成新的、包含种子设置的代码字符串
    modified_code = ast.unparse(tree)
    return modified_code

############################################################################################################
###  将 Python 代码中的除法操作符 (/) 替换为自定义的保护性除法函数 _protected_div，并根据需求使用 numba 库加速计算
############################################################################################################
#首先定义了一个 _CustomDivisionTransformer 类，它继承自 ast.NodeTransformer
#它的作用是遍历抽象语法树 (AST)，找到所有的除法操作 (/)，并将其替换为自定义的除法函数调用
class _CustomDivisionTransformer(ast.NodeTransformer):
    #接受一个参数 custom_divide_func_name，表示要用来替换除法操作符的自定义函数名。在这里，该名字是 _protected_div
    def __init__(self, custom_divide_func_name: str):
        super().__init__()
        self._custom_div_func = custom_divide_func_name

    #用于访问所有的二元操作符节点。如果检测到除法操作符 (/)，就用自定义函数替换。
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

#它的目的是将输入的代码字符串中的所有除法操作符替换为一个名为 _protected_div 的自定义保护性除法函数，该函数会避免除以零的问题。
def replace_div_with_protected_div(code_str: str, delta=1e-5, numba_accelerate=False) -> Tuple[str, str]:
    #定义保护性除法函数 _protected_div
    protected_div_str = f'''
def _protected_div(x, y, delta={delta}):
    return x / (y + delta)
    '''
    #将输入的代码字符串解析为一个 AST 树
    tree = ast.parse(code_str)

    #创建 _CustomDivisionTransformer 实例并遍历 AST，找到除法操作并替换
    transformer = _CustomDivisionTransformer('_protected_div')
    modified_tree = transformer.visit(tree)

    #将修改后的 AST 树转回代码字符串，这里将替换过的代码和保护性除法函数定义一起返回
    modified_code = ast.unparse(modified_tree)
    modified_code = '\n'.join([modified_code, '', '', protected_div_str])

    #如果 numba_accelerate 为真，则为 _protected_div 函数添加 @numba.jit() 装饰器来加速计算
    if numba_accelerate:
        modified_code = add_numba_decorator(modified_code, '_protected_div')
    #返回修改后的代码字符串和自定义除法函数的名字
    return modified_code, '_protected_div'
    
#######################################
#######################################
###  如何通信如何处理算法都准备好了，开始吧！
#######################################
#######################################
class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # 设置LLM 需要的一些信息
        self.pop_size = pop_size                    #定义种群大小
        self.interface_eval = interface_prob        #PROBLEMCONST()类型，可以调用evaluate函数对算法代码进行评估
        prompts = interface_prob.prompts            #问题描述、输入输出信息的 prompt，可以用于生成后面的 prompt
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode,prompts, **kwargs)  #Evolution类型，有 i1，e1,e2,m1,m2，可以用于算法的演化
        self.m = m                                  #'e1' 和 'e2' 操作父代的算法数量
        self.debug = debug_mode                     #debug 模式是否开启

        #如果不开 debug 模式，warning 都不显示
        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select                        #父代的选择方式
        self.n_p = n_p                              #种群大小
        
        self.timeout = timeout                      #超时时间定义
        #self.timeout = 400
        self.use_numba = use_numba                  #是否使用numba库给生成的函数加速
    
    #将文本代码写到文件./ael_alg.py中
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    #将一个新的个体（offspring）添加到已有的种群（population）中
    #前提是这个新的个体在目标值（objective）上与种群中的其他个体没有重复
    #如果没有重复的目标函数值，加进去并返回 True，反之返回 False
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    #用于检查给定的代码片段（code）是否已经存在于种群中的任何个体中
    #通过检查代码片段是否重复，可以避免将相同的个体多次添加到种群中
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False
    
    #用于根据指定的进化操作符生成后代个体
    def _get_alg(self,pop,operator, prompt):
        #初始化后代: 创建一个字典 offspring
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        #获得初始算法
        if operator == "initial":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.initial()    
        #生成和父代不相似的算法        
        elif operator == "cross":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.cross(parents, prompt)
        #生成改进当前算法生成新算法
        elif operator == "variation":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.variation(parents[0], prompt)    
        #没有这样的操作！
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        #同时返回选出来的父代算法和生成的子代
        return parents, offspring

    #用于生成后代个体并评估其适应度
    def get_offspring(self, pop, operator, prompt):

        try:
            #调用 _get_alg 方法，根据 operator（i1，m1……） 从 pop 中生成后代个体 offspring，并返回父代个体 p 和后代个体 offspring
            p, offspring = self._get_alg(pop, operator, prompt)
            
            #是否使用 Numb
            if self.use_numba:
                #使用正则表达式 r"def\s+(\w+)\s*\(.*\):" 匹配函数定义
                pattern = r"def\s+(\w+)\s*\(.*\):"
                #从 offspring['code'] 中提取函数名
                match = re.search(pattern, offspring['code'])
                function_name = match.group(1)
                #调用 add_numba_decorator 方法为函数添加 Numba 装饰器
                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            #处理重复代码
            n_retry= 1
            while self.check_duplicate(pop, offspring['code']):
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                
                #如果生成的代码与当前种群中的代码重复，则重新生成后代
                p, offspring = self._get_alg(pop, operator)

                #是否使用 Numb
                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring['code'])
                    function_name = match.group(1)
                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']
                
                #最多尝试一次
                if n_retry > 1:
                    break
                
            #创建线程池: 使用 ThreadPoolExecutor 执行评估任务
            with concurrent.futures.ThreadPoolExecutor() as executor:
                #提交 self.interface_eval.evaluate 方法进行评估，传入生成的代码 code
                future = executor.submit(self.interface_eval.evaluate, code)
                #获取评估结果 fitness，并将其四舍五入至小数点后 5 位，存储在 offspring['objective'] 中
                fitness = future.result(timeout=self.timeout)
                offspring['objective'] = np.round(fitness, 5)
                #结束任务以释放资源
                future.cancel()              

        #如果发生异常，设置 offspring 为包含所有 None 值的字典，并将 p 设置为 None
        except Exception as e:
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            p = None

        #返回父代个体 p 和生成的后代个体 offspring
        return p, offspring
    
    def get_algorithm(self, pop, operator, prompt):
        #results: 创建一个空列表 results 用于存储生成的后代个体
        results = []
        try:
            #生成 self.pop_size 个后代个体。结果存储在 results 列表中，每个元素是一个 (p, off) 元组，其中 p 是父代个体，off 是生成的后代个体
            results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator, prompt) for _ in range(self.pop_size))
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel time out .")
            
        time.sleep(2)


        out_p = []   #所有父代个体
        out_off = [] #所有后代个体

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            #如果是 debug 模式输出后代个体
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off

    def population_generation(self):
        #设定为 2，表示要生成 2 轮的个体
        n_create = 2
        #创建一个空列表，用于存储生成的初始种群个体
        population = []
        #循环生成个体
        for i in range(n_create):
            _,pop = self.get_algorithm([],'initial', [])
            for p in pop:
                population.append(p)
             
        return population
    
    #用于基于 seed（记录的算法）生成种群，其中每个个体的适应度是通过并行评估得到的
    def population_generation_seed(self,seeds,n_p):
        #创建一个空列表，用于存储生成的种群个体
        population = []
        #对每个种子的 code 使用 self.interface_eval.evaluate 方法进行评估，并计算其适应度。
        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)
        #遍历每个种子及其对应的适应度
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

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem      #定义问题
        self.select = select     #定义父代选择方式
        self.manage = manage     #定义种群的管理方式
        
        # LLM settings
        self.api_endpoint = paras.llm_api_endpoint  #定义API 的端点URL，用于与语言模型通信
        self.api_key = paras.llm_api_key            #API的私钥
        self.llm_model = paras.llm_model            #定义使用的大语言模型

        # prompt
        self.pop_size_cross = 2
        self.pop_size_variation = 2
        self.problem_type = "minimization"

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # 种群的大小
        self.n_pop = paras.ec_n_pop        # 跑多少轮

        self.operators = paras.ec_operators   #定义操作的数量，默认是 e1，e2，m1，m2 四个
        
        self.operator_weights = paras.ec_operator_weights    #定义操作的权重[0, 1]，权重越大越有可能使用这个操作
        if paras.ec_m > self.pop_size or paras.ec_m == 1:    #e1 和 e2 操作需要多少个父代，数量至少是两个但不能超过种群的大小
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m                                  #设定e1 和 e2 操作需要多少个父代

        self.debug_mode = paras.exp_debug_mode               # debug 模式是否开启
        self.ndelay = 1  # default

        self.output_path = paras.exp_output_path             #种群的结果保存路径

        self.exp_n_proc = paras.exp_n_proc                   #设置的进程数
        
        self.timeout = paras.eva_timeout                     ##超时时间定义

        self.prompt_timeout = paras.prompt_eva_timeout

        self.use_numba = paras.eva_numba_decorator           #是否使用 numba 库进行加速

        print("- EoH parameters loaded -")

        #设置随机种子
        random.seed(2024)

    #将新生成子代加入到种群当中，如果开启了调试模式就和当前的种群中个体逐个对比一下，看看有没有冗余
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
    

    #跑跑 EOH 吧
    def run(self):

        print("- Evolution Start -")
        #记录一下开始时间
        time_start = time.time()

        #设定问题的评估窗口
        interface_prob = self.prob
        
        #设定一下 prompt 演化
        interface_promt_cross = InterfaceEC_Prompt(self.pop_size_cross, self.m, self.api_endpoint, self.api_key, self.llm_model, self.debug_mode, self.select, self.exp_n_proc, self.prompt_timeout, self.problem_type)
        interface_promt_variation = InterfaceEC_Prompt(self.pop_size_variation, self.m, self.api_endpoint, self.api_key, self.llm_model, self.debug_mode, self.select, self.exp_n_proc, self.prompt_timeout, self.problem_type)
        #设定演化模式，包含初始化包含演化包含管理
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba
                                   )

        #初始化一下种群
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
        #将生成的种群保存为文件
        filename = self.output_path + "/results/pops/population_generation_0.json"
        with open(filename, 'w') as f:
            json.dump(population, f, indent=5)
        n_start = 0

        print("=======================================")

        #n_op表示有多少种演化操作
        n_op = len(self.operators)
        worst = []
        delay_turn = 3
        change_flag = 0
        last = -1
        max_k = 4
        #n_pop表示要跑多少轮
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

            #先看看交叉操作吧  
            for i in range(len(cross_operators)):
                promot = cross_operators[i]["prompt"]
                print(f" OP: cross, [{i + 1} / {len(cross_operators)}] ", end="|") 
                parents, offsprings = interface_ec.get_algorithm(population, "cross", promot)
                #将新生成子代加入到种群当中，如果开启了调试模式就和当前的种群中个体逐个对比一下，看看有没有冗余
                self.add2pop(population, offsprings)  
                for off in offsprings:
                    print(" Obj: ", off['objective'], end="|")
                    if(off['objective'] is None):
                        continue

                    if len(cross_operators[i]["number"]) < max_k:
                        heapq.heappush(cross_operators[i]["number"], -off['objective'])
                    else:
                        # 如果堆已满，且当前元素比堆顶元素小，替换堆顶元素
                        if off['objective'] < -cross_operators[i]["number"][0]:
                            heapq.heapreplace(cross_operators[i]["number"], -off['objective'])  # 替换堆顶元素
                        
                    cross_operators[i]["objective"] = -sum(cross_operators[i]["number"]) / len(cross_operators[i]["number"])
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)
                
                #加上新生代，种群大小超了吧，管理一下种群让数量变成最多pop_size个
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f"Cross {i + 1}, objective: {cross_operators[i]['objective']}", end = "|")
                print()
            
            #再看看变异操作吧 
            for i in range(len(cross_operators)):
                promot = variation_operators[i]["prompt"]
                print(f" OP: variation, [{i + 1} / {len(variation_operators)}] ", end="|") 
                parents, offsprings = interface_ec.get_algorithm(population, "variation", promot)
                #将新生成子代加入到种群当中，如果开启了调试模式就和当前的种群中个体逐个对比一下，看看有没有冗余
                self.add2pop(population, offsprings)  
                for off in offsprings:
                    print(" Obj: ", off['objective'], end="|")
                    if(off['objective'] is None):
                        continue
                    if len(variation_operators[i]["number"]) < max_k:
                        heapq.heappush(variation_operators[i]["number"], -off['objective'])
                    else:
                        # 如果堆已满，且当前元素比堆顶元素小，替换堆顶元素
                        if off['objective'] < -variation_operators[i]["number"][0]:
                            heapq.heapreplace(variation_operators[i]["number"], -off['objective'])  # 替换堆顶元素
                        
                    variation_operators[i]["objective"] = -sum(variation_operators[i]["number"]) / len(variation_operators[i]["number"])
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)
                
                #加上新生代，种群大小超了吧，管理一下种群让数量变成最多pop_size个
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f"variation {i + 1}, objective: {variation_operators[i]['objective']}", end = "|")
                print()

            ''' 
            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|") 
                op_w = self.operator_weights[i]
                #如果随机数小于权重（权重的范围在 0 到 1 之前），那么才跑
                #换句话说对于每个操作，其operator_weights就是跑这个操作的概率
                if (np.random.rand() < op_w):
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                #将新生成子代加入到种群当中，如果开启了调试模式就和当前的种群中个体逐个对比一下，看看有没有冗余
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
                
                #加上新生代，种群大小超了吧，管理一下种群让数量变成最多pop_size个
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()
            '''  

            #将种群保存在文件中，每一代都有自己的文件
            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            #将种群的最佳个体保存在文件中，每一代都有自己的文件
            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            #输出一下时间，以分钟为单位
            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            #将管理过后剩下的种群的目标函数值输出一下
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            worst.append(population[-1]['objective'])
            print()


class Methods():
    #设定一下父代的选择方法和种群的管理方法，就是将字符串映射位函数方法，还挺神奇的
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
        #必须得跑 eoh 吧
        if self.paras.method == "eoh":   
            return EOH(self.paras,self.problem,self.select,self.manage)
        else:
            print("method "+self.method+" has not been implemented!")
            exit()

class EVOL:
    # initilization
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

        
    # run methods
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



# Parameter initilization #
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

# initilization
evolution = EVOL(paras)

# run 
evolution.run()