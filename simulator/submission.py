from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance, Robot
import random
import time
import numpy as np
from enum import Enum

time_offset = 0.1 # an offset from the time_limit given to our algorithms to ensure we don't overuse resources.

class AgentTurn(Enum):
    MAX = 0
    MIN = 1
    EXP = 2

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    agent: Robot = env.get_robot(robot_id)
    station_dists = [manhattan_distance(agent.position, station.position) for station in env.charge_stations]
    closest_station = min(station_dists)
    dist = 0
    
    if agent.package is None:
        dists_from_packs = [manhattan_distance(agent.position, p.position) for p in env.packages if (p.on_board == True)]
        dist = min(dists_from_packs)
    else:  # agent carries a package
        dist = manhattan_distance(agent.position, agent.package.destination)

    if agent.battery <= dist:        # once all you can do is get to a charging station, go charge..
        return (-1) * (agent.credit + closest_station)

    return (100 * (agent.package is not None)) + (100 * agent.credit) - dist

# this utility function is only called when the game is done!
    # return value is positive when we win
    # return value is 0 when it's a draw
    # return value is negative when we lose
def game_done_utility(env: WarehouseEnv, robot_id: int):
        agent = env.get_robot(robot_id)
        adversary = env.get_robot((robot_id+1)%2)
        return agent.credit - adversary.credit
    
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

class AgentMinimax(Agent):
    # TODO: section b : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        if env.done():
            return game_done_utility(env,robot_id)
        else:
            return smart_heuristic(env,robot_id)

    def compute_next_operation(self, env: WarehouseEnv, agent_id, time_left, turn: AgentTurn, depth):
        start_time = time.time()
        
        if time_left <= time_offset:
            raise TimeoutError
        
        if ( env.done() ) or ( depth == 0 ):  # game ends when we run out of steps
            return self.heuristic(env, agent_id)
        
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            
        if turn == AgentTurn.MAX: 
            curr_max = -(np.inf)
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MIN, depth-1)
                curr_max = max(curr_max, result)
            return curr_max
        else: # turn == AgentTurn.MIN
            curr_min = np.inf
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MAX, depth-1)
                curr_min = min(curr_min, result)
            return curr_min
    
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        # max_depth = env.num_steps
        depth = 1
        operation = 'park'
        while (True) :
            try:
                # we perform the MAX action on the children heuristics here therefore, we call the function with MIN argument
                operators = env.get_legal_operators(agent_id)
                children = [env.clone() for _ in operators]
                for child, op in zip(children, operators):
                    child.apply_operator(agent_id, op)
                heuristics = [  self.compute_next_operation(child,agent_id, (time_limit - (time.time() - start_time)), AgentTurn.MIN, depth)
                                for child in children   ]
                operation = operators[ heuristics.index( max(heuristics) ) ]
                depth += 1
            except TimeoutError: # ran out of time
                return operation
        return operation

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        if env.done():
            return game_done_utility(env,robot_id)
        else:
            return smart_heuristic(env,robot_id)
    
    def compute_next_operation( self, env:WarehouseEnv, agent_id, time_left, turn:AgentTurn, depth, alpha, beta):
        start_time = time.time()
        
        if time_left <= time_offset:
            raise TimeoutError
        
        if ( env.done() ) or ( depth == 0 ):  # game ends when we run out of steps
            return self.heuristic(env, agent_id)
        
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            
        if turn == AgentTurn.MAX: 
            curr_max = -(np.inf)
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MIN, depth-1, alpha, beta)
                curr_max = max(result, curr_max)
                alpha = max(curr_max, alpha)
                if (result >= beta): # prune this sub-tree
                    # we return infinity because we want to prune this sub-tree which will happen
                    # because this value is returned to MIN player and he picks min(result=np.inf, curr_min)==curr_min
                    return (np.inf) 
            return curr_max
        else: # turn == AgentTurn.MIN
            curr_min = np.inf
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MAX, depth-1, alpha, beta)
                curr_min = min(result, curr_min)
                beta = min(curr_min, beta)
                if result <= alpha: # prune this sub-tree
                    # we return -infinity because we want to prune this sub-tree which will happen
                    # because this value is returned to MAX player and he picks max(result=-np.inf, curr_max)==curr_max
                    return (-(np.inf))
            return curr_min
    
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # max_depth = env.num_steps
        depth = 1
        start_time = time.time()
        operation = 'park'
        while  (True):
            try:
                # we perform the MAX action on the children heuristics here therefore, we call the function with MIN argument
                operators = env.get_legal_operators(agent_id)
                children = [env.clone() for _ in operators]
                for child, op in zip(children, operators):
                    child.apply_operator(agent_id, op)
                heuristics = [  self.compute_next_operation(child, agent_id, (time_limit - (time.time() - start_time)), AgentTurn.MIN, depth, alpha=(-(np.inf)), beta=(np.inf) )
                                for child in children   ]
                operation = operators[ heuristics.index( max(heuristics) ) ]
                depth += 1
            except TimeoutError: # ran out of time
                return operation
        return operation


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        if env.done():
            return game_done_utility(env, robot_id)
        else:
            return smart_heuristic(env,robot_id)
    
    def compute_next_operation( self, env:WarehouseEnv, agent_id, time_left, turn:AgentTurn, depth):
        start_time = time.time()
        
        if time_left <= time_offset:
            raise TimeoutError
        
        if ( env.done() ) or ( depth == 0 ):  # game ends when we run out of steps
            return self.heuristic(env, agent_id)
        
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
    
        if turn == AgentTurn.MAX: 
            curr_max = -(np.inf)
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.EXP, depth-1)
                curr_max = max(result, curr_max)
            return curr_max
        
        else: # turn == AgentTurn.EXP: <--> probabilistic player( consider implementing one or use random)
            expectancy = 0
            probabilities = [1] * len(operators) 
            for i, (environment, op) in enumerate(zip(children, operators)):
                if 'charge' in environment.get_legal_operators(agent_id): 
                    probabilities[i] *= 2 # is this the correct implementation that was described in piazza?
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MAX, depth-1)
                weight_sum = sum(probabilities)
                probability = ( probabilities[i] / weight_sum )
                expectancy += (probability * result)
            return expectancy
        
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # max_depth = env.num_steps
        depth = 1
        start_time = time.time()
        operation = 'park'
        while  (True):
            try:
                # we perform the MAX action on the children heuristics here therefore, we call the function with EXP argument
                operators = env.get_legal_operators(agent_id)
                children = [env.clone() for _ in operators]
                for child, op in zip(children, operators):
                    child.apply_operator(agent_id, op)
                heuristics = [  self.compute_next_operation(child, agent_id, (time_limit - (time.time() - start_time)), AgentTurn.EXP, depth )
                                for child in children   ]
                operation = operators[ heuristics.index( max(heuristics) ) ]
                depth += 1
            except TimeoutError: # ran out of time
                return operation
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
