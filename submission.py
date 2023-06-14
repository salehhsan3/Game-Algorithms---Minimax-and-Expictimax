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
    adv = env.get_robot((robot_id+1)%2)
    magic_num = 25
    is_agent_winning = agent.credit - adv.credit
    value = 0
    if agent.package != None:
        drop_off_dist = manhattan_distance(agent.position, agent.package.destination)
        dist_charge_st_0 = manhattan_distance(agent.position, env.charge_stations[0].position)
        dist_charge_st_1 = manhattan_distance(agent.position, env.charge_stations[1].position)
        nearest_charge_st = min(dist_charge_st_1, dist_charge_st_0)
        
        if agent.battery >= drop_off_dist: # has enough battery to deliver package
            if drop_off_dist == 0:
                value = magic_num * 2 * (10 + agent.credit)
            else: # drop_off_dist > 0
                value = (magic_num * (10 + agent.credit) / drop_off_dist)
        else: # doesn't have enough battery to drop off package --> should go and recharge
            if nearest_charge_st == 0:
                value = magic_num * 2 
            else:
                value = (magic_num / nearest_charge_st)
    else: # agent doesn't have a package
        dist_agent_pack_0 = manhattan_distance(agent.position, env.packages[0].position)
        dist_agent_pack_1 = manhattan_distance(agent.position, env.packages[1].position)
        pick_up_dist = min(dist_agent_pack_0, dist_agent_pack_1) # cost tp pick up nearest package
        
        dist_charge_st_0 = manhattan_distance(agent.position, env.charge_stations[0].position)
        dist_charge_st_1 = manhattan_distance(agent.position, env.charge_stations[1].position)
        nearest_charge_st = min(dist_charge_st_1, dist_charge_st_0) # cost to reach nearest charge station
        
        if agent.battery >= pick_up_dist:
            if pick_up_dist == 0:
                value = magic_num * 2 * (10 + agent.credit)
            else:
                value = (magic_num * (10 + agent.credit) / pick_up_dist)
        else: # doesn't have enough battery to pick up a package.
            if nearest_charge_st == 0:
                value = magic_num * 2
            else: # nearest_charge_st > 0
                value = (magic_num / nearest_charge_st)
                
    return value
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def compute_next_operation(self, env: WarehouseEnv, agent_id, time_left, turn: AgentTurn, depth):
        start_time = time.time()
        agent = env.get_robot(agent_id)
        
        if time_left <= time_offset:
            return (None,None)
        
        if (env.num_steps == 0) or (depth == 0):  # game ends when we run out of steps
            if turn == AgentTurn.MIN:
                agent_id = 1 - agent_id
            dest_value = smart_heuristic(env, agent_id)
            # if turn == AgentTurn.MIN:
            #    dest_value *= (-1)
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
    
    
