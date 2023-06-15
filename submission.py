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
    PROB = 2

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    # it's important to remember that there are only 2 packages throughout the game
    agent = env.get_robot(robot_id)
    adv = env.get_robot((robot_id+1)%2)
    magic_num_pick_up = 100
    magic_num_drop_off = 10
    
    charge_st_dists = [manhattan_distance(agent.position, station.position) for station in env.charge_stations]
    nearest_charge_st = min(charge_st_dists) # cost to reach nearest charge station
    
    value = 0
    if agent.package != None:
        drop_off_dist = manhattan_distance(agent.position, agent.package.destination)
        drop_off_gain = manhattan_distance(agent.package.position, agent.package.destination)
        if agent.battery >= drop_off_dist: # has enough battery to deliver package
            if drop_off_dist == 0 :
                value = ((1 + agent.credit) * drop_off_gain)
            else:
                value = (((1 + agent.credit) * drop_off_gain) / magic_num_drop_off)
        else: # doesn't have enough battery to drop off package --> should go and recharge
            if nearest_charge_st == 0:
                value = magic_num_drop_off * 2 
            else:
                value = (magic_num_drop_off / nearest_charge_st)
    else: # agent doesn't have a package
        packages_dists = [manhattan_distance(agent.position, package.position) for package in env.packages if (package.on_board == True)]
        pick_up_dist = min(packages_dists) # cost tp pick up nearest package
        package = None
        for pack in env.packages:
            if (pack.on_board == True) and (manhattan_distance(pack.position, agent.position) == pick_up_dist):
                package = pack
                
        potential_gain = 2 * manhattan_distance(package.position, package.destination)
        if agent.battery >= pick_up_dist:
            if pick_up_dist == 0:
                value = (((1 + agent.credit) * potential_gain) / magic_num_pick_up)
            else:
                value = (( ( 1 + agent.credit ) / pick_up_dist) / magic_num_pick_up)
            
        else: # doesn't have enough battery to pick up a package.
            if nearest_charge_st == 0:
                value = magic_num_pick_up * 2
            else:
                value = (magic_num_pick_up / nearest_charge_st)
                
    return value

# def smart_heuristic(env: WarehouseEnv, robot_id: int):
#     # it's important to remember that there are only 2 packages throughout the game
#     agent = env.get_robot(robot_id)
#     adv = env.get_robot((robot_id+1)%2)
#     magic_num_pick_up = 10
#     magic_num_drop_off = 100
    
#     charge_st_dists = [manhattan_distance(agent.position, station.position) for station in env.charge_stations]
#     nearest_charge_st = min(charge_st_dists) # cost to reach nearest charge station
    
#     value = 0
#     if agent.package != None:
#         drop_off_dist = manhattan_distance(agent.package.position, agent.package.destination)
#         return (2*drop_off_dist)
#     else: # agent doesn't have a package
#         packages_dists = [manhattan_distance(agent.position, package.position) for package in env.packages if (package.on_board == True)]
#         nearest_package = min(packages_dists)
#         new_package_credit = 0
#         for package in env.packages:
#             if package.on_board:
#                 new_package_credit = 2 * manhattan_distance(package.position, package.destination)

#         score = - nearest_package - nearest_charge_st - agent.battery + new_package_credit
#         return score
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        agent = env.get_robot(robot_id)
        adv = env.get_robot((robot_id + 1)%2)
        return ( ( agent.credit - adv.credit ) + 0.5 * (agent.battery - adv.battery) )

    def compute_next_operation(self, env: WarehouseEnv, agent_id, time_left, turn: AgentTurn, depth):
        start_time = time.time()
        agent = env.get_robot(agent_id)
        
        if time_left <= time_offset:
            return (None,None)
        
        if (env.num_steps == 0) or (depth == 0):  # game ends when we run out of steps
            if turn == AgentTurn.MIN:
                agent_id = (agent_id + 1)%2
            dest_value = self.heuristic(env, agent_id)
            return dest_value, None
        
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            
        if turn == AgentTurn.MAX: 
            curr_max = -(np.inf)
            max_op = None
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment,(agent_id+1)%2,time_left, AgentTurn.MIN, depth-1)
                if (result[0] != None) and (result[0] > curr_max):
                    curr_max = result[0]
                    max_op = op 
            return (curr_max, max_op)
        else: # turn == AgentTurn.MIN
            curr_min = np.inf
            min_op = None
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment,(agent_id+1)%2,time_left, AgentTurn.MAX, depth-1)
                if (result[0] != None) and (result[0] < curr_min):
                    curr_min = result[0]
                    min_op = op
            return (curr_min, min_op)
    
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        
        max_depth = env.num_steps
        depth = 1
        start_time = time.time()
        operation = None
        while ( ( (time.time() - start_time) < time_offset ) and (depth <= max_depth) ):
            result = self.compute_next_operation(env,agent_id, (time_limit - (time.time() - start_time)), AgentTurn.MAX, depth )
            if result[1] is not None:
                operation = result[1]
            else: # ran out of time
                break 
            depth += 1
        return operation

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        agent = env.get_robot(robot_id)
        adv = env.get_robot((robot_id + 1)%2)
        return ( ( agent.credit - adv.credit ) + 0.5 * (agent.battery - adv.battery) )
    
    def compute_next_operation( self, env:WarehouseEnv, agent_id, time_left, turn:AgentTurn, depth, alpha, beta):
        start_time = time.time()
        
        if time_left <= time_offset:
            return (None,None)
        
        if (env.num_steps == 0) or (depth == 0):  # game ends when we run out of steps
            if turn == AgentTurn.MIN:
                agent_id = (agent_id + 1)%2
            dest_value = self.heuristic(env, agent_id)
            return dest_value, None
        
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            
        if turn == AgentTurn.MAX: 
            max_op = None
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment,(agent_id+1)%2,time_left, AgentTurn.MIN, depth-1, alpha, beta)
                if (result[0] != None):
                    if result[0] >= beta:
                        return (beta, None) # cut off
                    elif (result[0] > alpha):
                        alpha = result[0]
                        max_op = op 
            return (alpha, max_op)
        else: # turn == AgentTurn.MIN
            min_op = None
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment,(agent_id+1)%2,time_left, AgentTurn.MAX, depth-1, alpha, beta)
                if (result[0] != None):
                    if result[0] <= alpha:
                        return (alpha, None) # cut off
                    elif (result[0] < beta):
                        beta = result[0]
                        min_op = op
            return (beta, min_op)
    
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        max_depth = env.num_steps
        depth = 1
        start_time = time.time()
        operation = None
        alpha = -(np.inf)
        beta = np.inf
        while ( ( (time.time() - start_time) < time_offset ) and (depth <= max_depth) ):
            result = self.compute_next_operation(env,agent_id, (time_limit - (time.time() - start_time)), AgentTurn.MAX, depth, alpha, beta )
            if result[1] is not None:
                operation = result[1]
            else: # ran out of time
                break 
            depth += 1
        return operation


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        agent = env.get_robot(robot_id)
        adv = env.get_robot((robot_id + 1)%2)
        return ( ( agent.credit - adv.credit ) + 0.5 * (agent.battery - adv.battery) )
    
    def compute_next_operation( self, env:WarehouseEnv, agent_id, time_left, turn:AgentTurn, depth, last_caller:AgentTurn = None):
        start_time = time.time()
        
        if time_left <= time_offset:
            return (None,None)
        
        if (env.num_steps == 0) or (depth == 0):  # game ends when we run out of steps
            if turn == AgentTurn.MIN:
                agent_id = (agent_id + 1)%2
            dest_value = self.heuristic(env, agent_id)
            return dest_value, None
        
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        
        if turn == AgentTurn.PROB:
            expected = 0
            probability = (1 / len(operators))
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                if last_caller == AgentTurn.MAX:
                    result = self.compute_next_operation(environment,(agent_id+1)%2,time_left, AgentTurn.MIN, depth-1, AgentTurn.PROB)
                elif last_caller == AgentTurn.MIN:
                    result = self.compute_next_operation(environment,(agent_id+1)%2,time_left, AgentTurn.MAX, depth-1, AgentTurn.PROB)
                if result[0] != None:
                    expected += (probability * result[0])
            return (expected, None)
        
        elif turn == AgentTurn.MAX: 
            max_op = None
            curr_max = -(np.inf)
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment,(agent_id+1)%2,time_left, AgentTurn.PROB, depth-1, AgentTurn.MAX)
                if (result[0] != None) and ((result[0] > curr_max)):
                        curr_max = result[0]
                        max_op = op 
            return (curr_max, max_op)
        elif turn == AgentTurn.MIN: 
            min_op = None
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment,(agent_id+1)%2,time_left, AgentTurn.PROB, depth-1, AgentTurn.MIN)
                if (result[0] != None):
                        curr_min = result[0]
                        min_op = op
            return (curr_min, min_op)
        
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        max_depth = env.num_steps
        depth = 1
        start_time = time.time()
        operation = None
        while ( ( (time.time() - start_time) < time_offset ) and (depth <= max_depth) ):
            result = self.compute_next_operation(env,agent_id, (time_limit - (time.time() - start_time)), AgentTurn.MAX, depth, None )
            if result[1] is not None:
                operation = result[1]
            else: # ran out of time
                break 
            depth += 1
        return operation


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
    
    
