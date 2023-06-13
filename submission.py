from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import numpy as np
from enum import Enum

time_offset = 0.1 # an offset from the time_limit given to our algorithms to ensure we don't overuse resources.

class AgentTurn(Enum):
    MAX = 0
    MIN = 1

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    # it's important to remember that there are only 2 packages throughout the game
    agent = env.get_robot(robot_id)
    adversary = env.get_robot((robot_id+1)%2)
    
    agent_value = 0
    adv_value = 0
    if agent.package == None:  # Agent doesn't carry a package 
        
        if adversary.package != None:  # Adversary Carries A package
            adv_package = adversary.package
            pack_ind = 0 if (env.packages[1] == adv_package) else 1
            agent_pack_dist = manhattan_distance(agent.position, env.packages[pack_ind].position)
            agent_add_credit = manhattan_distance(env.packages[pack_ind].destination, env.packages[pack_ind].position)
            adv_pack_dist = manhattan_distance(adversary.position, adv_package.destination)
            adv_add_credit = manhattan_distance(adv_package.destination, adv_package.position)
            agent_feasable = (env.num_steps//2) >= (agent_pack_dist + agent_add_credit)
            adv_feasable = (env.num_steps//2) >= manhattan_distance(adversary.position, adversary.package.destination)
            
            if not agent_feasable:
                agent_value = agent.credit
            else: # agent is feasable
                agent_value = 0.75* 2 * agent_add_credit -0.25* agent_pack_dist
            
            if not adv_feasable:
                adv_value = adversary.credit
            else:
                adv_value =  0.75* 2 * adv_add_credit -0.25* adv_pack_dist
        
        else:  # BOTH DONT HAVE ANY PACKAGE
            
            agent_pack_dist_0 = manhattan_distance(agent.position, env.packages[0].position)
            agent_pack_dist_1 = manhattan_distance(agent.position, env.packages[1].position)
            adv_pack_dist_0 = manhattan_distance(adversary.position, env.packages[0].position)
            adv_pack_dist_1 = manhattan_distance(adversary.position, env.packages[1].position)
            add_credit_pack_0 = manhattan_distance(env.packages[0].destination, env.packages[0].position)
            add_credit_pack_1 = manhattan_distance(env.packages[1].destination, env.packages[1].position)
            min_agent_pack = min( (agent_pack_dist_0 + add_credit_pack_0), (agent_pack_dist_1 + add_credit_pack_1) )
            agent_feasable = env.num_steps//2 >=  min_agent_pack
            min_adv_pack = min( (adv_pack_dist_0 + add_credit_pack_0), (adv_pack_dist_1 + add_credit_pack_1) )
            adv_feasable = env.num_steps//2 >=  min_adv_pack
            
            if not agent_feasable:
                agent_value = agent.credit
            else:
                agent_pack_0_val = 0.75*2*add_credit_pack_0 - 0.25*agent_pack_dist_0
                agent_pack_1_val = 0.75*2*add_credit_pack_1 - 0.25*agent_pack_dist_1
                agent_value = max(agent_pack_0_val, agent_pack_1_val)
                
            if not adv_feasable:
                adv_value = adversary.credit
            else:
                adv_pack_0_val = 0.75*2*add_credit_pack_0 - 0.25*adv_pack_dist_0
                adv_pack_1_val = 0.75*2*add_credit_pack_1 - 0.25*adv_pack_dist_1
                adv_value = max(adv_pack_0_val, adv_pack_1_val)
    else: # agent carries a package
        agent_feasable = env.num_steps//2 >= manhattan_distance(agent.package.destination, agent.position)
        
        if not agent_feasable:
            agent_value = agent.credit
        else:
            agent_add_credit = manhattan_distance(agent.package.destination, agent.package.position)
            agent_pack_dist = manhattan_distance(agent.position, agent.package.destination)
            agent_value = 0.75*2*agent_add_credit - 0.25*agent_pack_dist
        
        if adversary.package != None: # adverasy carries a package
            adv_pack_dist = manhattan_distance(adversary.position, adversary.package.destination)
            adv_add_credit = manhattan_distance(adversary.package.destination, adversary.package.position)
            adv_feasable = env.num_steps//2 >= adv_pack_dist
            
            if not adv_feasable:
                adv_value = adversary.credit
            else:
                adv_value = 0.75*2*adv_add_credit - 0.25*adv_pack_dist
        else: # adversary doesn't carry a package
            adv_pack_ind = 0 if agent.package == env.packages[1] else 1
            adv_pack_dist = manhattan_distance(adversary.position, env.packages[adv_pack_ind].position)
            adv_add_credit = manhattan_distance(env.packages[adv_pack_ind].destination, env.packages[adv_pack_ind].position)
            adv_feasable = env.num_steps//2 >= (adv_pack_dist + adv_add_credit)
            
            if not adv_feasable:
                adv_value = adversary.credit
            else:
                adv_value = 0.75*2*adv_add_credit - 0.25*adv_pack_dist
    return (agent_value - adv_value)


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def compute_next_operation(self, env: WarehouseEnv, agent_id, time_limit, turn: AgentTurn, depth, operation = None):
        start_time = time.time()
        agent = env.get_robot(agent_id)
        
        if time_limit <= time_offset:
            return (None,None)
        
        if (env.num_steps == 0) or (depth == 0): # game ends when we run out of steps
            dest_value = self.heuristic(env,agent_id)
            return (operation,dest_value) 
        
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            
        if turn == AgentTurn.MAX: 
            curr_max = -(np.inf)
            max_op = None
            for environment, op in zip(children, operators):
                time_left = time_limit - (start_time - time.time())
                result = self.compute_next_operation(environment,(agent_id+1)%2,time_left, AgentTurn.MIN, depth-1, op )
                if result[0] > curr_max:
                    curr_max = result[0]
                    max_op = result[1]
            return (curr_max, max_op)
        else:
            curr_min = np.inf
            min_op = None
            for environment, op in zip(children, operators):
                time_left = time_limit - (start_time - time.time())
                result = self.compute_next_operation(environment,(agent_id+1)%2,time_left, AgentTurn.MAX, depth-1, op )
                if result[0] < curr_min:
                    curr_min = result[0]
                    min_op = result[1]
            return (curr_min, min_op)
    
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        max_depth = env.num_steps
        depth = 1
        start_time = time.time()
        operation = None
        while ( (time.time() - start_time) > time_offset ) and (depth <= max_depth):
            result = self.compute_next_operation(env,agent_id, (time_limit - (time.time() - start_time)), AgentTurn.MAX, depth, operation )
            if result != (None,None):
                operation = result[1]
            depth += 1
        return operation

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
    
    
    
    
    
    
    
    
    
    
    # agent_dist_from_pack_1 = manhattan_distance(agent.position, env.packages[0].position)
    # agent_dist_from_pack_2 = manhattan_distance(agent.position, env.packages[1].position)
    # adv_dist_from_pack_1 = manhattan_distance(adversary.position, env.packages[0].position)
    # adv_dist_from_pack_2 = manhattan_distance(adversary.position, env.packages[1].position)
    # agent_nearest_pack_dist = min(agent_dist_from_pack_1,agent_dist_from_pack_2)
    # drop_off_cost = manhattan_distance(agent.position, agent_nearest_pack_dist)
    # feasibility = max(0,(env.num_steps/2) - drop_off_cost)
    
    # agent_delivery_cost = 0
    # if agent.package != None: 
    #     agent_delivery_cost = 2 * manhattan_distance(agent.package.position, agent.package.destination) 
    
    # agent_pick_up_cost = 0
    # if agent.package == None:  # Agent doesn't carry a package: 
    #     agent_value = 0
    #     agent_credit = 2 * manhattan_distance(env.packages[pack_ind].position, env.packages[pack_ind].destination) 
    #     agent_dist = manhattan_distance(agent.position,env.packages[pack_ind].position)
        
    #     if adversary.package == None: 
    #         if (env.num_steps//2) < ():
    #     else: # adversary has a package and agent doesn't have a package 
    #         adv_package = adversary.package
    #         pack_ind = 1 if (adv_package == env.packages[0]) else 0
           
    #         agent_value = 0.75*agent_credit - 0.25*agent_dist
    #         adv_credit = 2 * manhattan_distance(adv_package.position, adv_package.destination)
    #         adv_dist = manhattan_distance(adversary.position, adv_package.destination)
    #         adv_value = 0.75*adv_credit - 0.25*adv_dist
            
            
             
        
    # return (feasibility + agent_pick_up_cost + agent_delivery_cost)