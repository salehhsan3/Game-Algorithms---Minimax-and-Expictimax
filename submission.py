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
    
    dist_from_pack_1 = manhattan_distance(agent.position, env.packages[0].position)
    dist_from_pack_2 = manhattan_distance(agent.position, env.packages[1].position)
    nearest_pack_dist = min(dist_from_pack_1,dist_from_pack_2)
    drop_off_cost = manhattan_distance(agent.position,nearest_pack_dist)
    feasibility = max(0,(env.num_steps/2) - drop_off_cost)
    
    agent_delivery_cost = 0
    if agent.package != None: 
        if agent.package == env.packages[0]:
            agent_delivery_cost = manhattan_distance(agent.position, env.packages[0].destination)
        else:
            agent_delivery_cost = manhattan_distance(agent.position, env.packages[1].destination)
    
    agent_pick_up_cost = 0
    if agent.package == None: 
        if agent.package == env.packages[0]:
            agent_pick_up_cost = dist_from_pack_1
        else:
            agent_pick_up_cost = dist_from_pack_2
    
    return (feasibility + agent_pick_up_cost + agent_delivery_cost)

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def compute_next_operation(self, env: WarehouseEnv, agent_id, time_limit, turn: AgentTurn, depth, operation):
        start_time = time.time()
        agent = env.get_robot(agent_id)
        agent_pos = agent.position
        agent_package = agent.package
        
        if time_limit <= time_offset:
            return (None,None)
        
        if (agent_package != None and agent_pos == agent_package.destination) or (depth == 0):
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
                result = self.compute_next_operation(environment,agent_id,time_left, AgentTurn.MIN, depth-1, op )
            if result[0] > curr_max:
                curr_max = result[0]
                max_op = result[1]
            return (max_op, curr_max)
        else:
            curr_min = np.inf
            min_op = None
            for environment, op in zip(children, operators):
                time_left = time_limit - (start_time - time.time())
                result = self.compute_next_operation(environment,agent_id,time_left, AgentTurn.MAX, depth-1, op )
            if result[0] < curr_min:
                curr_min = result[0]
                min_op = result[1]
            return (min_op, curr_min)
    
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        depth = 1
        start_time = time.time()
        while (time.time() - start_time) > time_offset:
            operation, value = self.compute_next_operation(env,agent_id,(time.time() - start_time), AgentTurn.MAX, depth, None )
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