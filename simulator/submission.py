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
    PROB = 2

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

def game_done_utility(env: WarehouseEnv, robot_id: int):
    # this utility function is only called when the game is done!
    # return value is positive when we win
    # return value is 0 when it's a draw
    # return value is negative when we lose
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
            return smart_heuristic(env,robot_id) - smart_heuristic(env, (robot_id+1)%2)

    def compute_next_operation(self, env: WarehouseEnv, agent_id, time_left, turn: AgentTurn, depth):
        start_time = time.time()
        
        if time_left <= time_offset:
            return (None,None)
        
        if ( env.done() ) or ( depth == 0 ):  # game ends when we run out of steps
            dest_value = self.heuristic(env, agent_id) ## changed here!!
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
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MIN, depth-1)
                if (result[0] != None) and (result[0] > curr_max):
                    curr_max = result[0]
                    max_op = op 
            return (curr_max, max_op)
        else: # turn == AgentTurn.MIN
            curr_min = np.inf
            min_op = None
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MAX, depth-1)
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
        if env.done():
            return game_done_utility(env,robot_id)
        else:
            return smart_heuristic(env,robot_id) - smart_heuristic(env, (robot_id+1)%2)
    
    def compute_next_operation( self, env:WarehouseEnv, agent_id, time_left, turn:AgentTurn, depth, alpha, beta):
        start_time = time.time()
        
        if time_left <= time_offset:
            return (None,None)
        
        if ( env.done() ) or ( depth == 0 ):  # game ends when we run out of steps
            dest_value = self.heuristic(env, agent_id) ## changed here!!
            return dest_value, None
        
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            
        if turn == AgentTurn.MAX: 
            max_op = None
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MIN, depth-1, alpha, beta)
                if (result[0] != None):
                    if result[0] >= beta:
                        return (beta, op) # cut off
                    elif (result[0] > alpha):
                        alpha = result[0]
                        max_op = op 
            return (alpha, max_op)
        else: # turn == AgentTurn.MIN
            min_op = None
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MAX, depth-1, alpha, beta)
                if (result[0] != None):
                    if result[0] <= alpha:
                        return (alpha, op) # cut off
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
        if env.done():
            return game_done_utility(env, robot_id)
        else:
            return smart_heuristic(env,robot_id) - smart_heuristic(env, (robot_id+1)%2)
    
    def compute_next_operation( self, env:WarehouseEnv, agent_id, time_left, turn:AgentTurn, depth, last_caller:AgentTurn = None):
        start_time = time.time()
        
        if time_left <= time_offset:
            return (None,None)
        
        if ( env.done() ) or ( depth == 0 ):  # game ends when we run out of steps
            dest_value = self.heuristic(env, agent_id) ## changed here!!
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
                    result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MIN, depth-1, AgentTurn.PROB)
                elif last_caller == AgentTurn.MIN:
                    result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.MAX, depth-1, AgentTurn.PROB)
                if result[0] != None:
                    expected += (probability * result[0])
            return (expected, None)
        
        elif turn == AgentTurn.MAX: 
            max_op = None
            curr_max = -(np.inf)
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.PROB, depth-1, AgentTurn.MAX)
                if (result[0] != None) and ((result[0] > curr_max)):
                        curr_max = result[0]
                        max_op = op 
            return (curr_max, max_op)
        elif turn == AgentTurn.MIN: 
            min_op = None
            for environment, op in zip(children, operators):
                time_left = time_left - (time.time() - start_time)
                result = self.compute_next_operation(environment, agent_id, time_left, AgentTurn.PROB, depth-1, AgentTurn.MIN)
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
