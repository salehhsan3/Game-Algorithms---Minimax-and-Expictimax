import os
import shutil
import csv
import random
# format: ((agent_0, agent_1), simulations_num)
# agents = [] # for every possible agent
# for agent_0 in ["random", "greedy", "greedyImproved", "minimax", "alphabeta", "expectimax"]:
#     for agent_1 in ["random", "greedy", "greedyImproved", "minimax", "alphabeta", "expectimax"]:
#         agents.append(((agent_0, agent_1), 10))
# for the agents we implemented
agents = [ 
        # (('random', 'random'), 6),
        # (('random', 'greedy'), 6),
        # (('random', 'greedyImproved'), 6),
        # (('random', 'minimax'), 6),
        # (('random', 'alphabeta'), 6),
        # (('random', 'expectimax'), 6),
        # (('greedy', 'random'), 6),
        # (('greedy', 'greedy'), 6),
        # (('greedy', 'greedyImproved'), 6),
        # (('greedy', 'minimax'), 6),
        # (('greedy', 'alphabeta'), 6),
        # (('greedy', 'expectimax'), 6),
        # (('greedyImproved', 'random'), 6),
        # (('greedyImproved', 'greedy'), 6),
        (('greedyImproved', 'greedyImproved'), 6),
        (('greedyImproved', 'minimax'), 6),
        (('greedyImproved', 'alphabeta'), 6),
        (('greedyImproved', 'expectimax'), 6),
        # (('minimax', 'random'), 6),
        # (('minimax', 'greedy'), 6),
        (('minimax', 'greedyImproved'), 6),
        (('minimax', 'minimax'), 6),
        (('minimax', 'alphabeta'), 6),
        (('minimax', 'expectimax'), 6),
        # (('alphabeta', 'random'), 6),
        # (('alphabeta', 'greedy'), 6),
        (('alphabeta', 'greedyImproved'), 6),
        (('alphabeta', 'minimax'), 6),
        (('alphabeta', 'alphabeta'), 6),
        (('alphabeta', 'expectimax'), 6),
        # (('expectimax', 'random'), 6),
        # (('expectimax', 'greedy'), 6),
        (('expectimax', 'greedyImproved'), 6),
        (('expectimax', 'minimax'), 6),
        (('expectimax', 'alphabeta'), 6),
        (('expectimax', 'expectimax'), 6),   
    ]


def simulations():
    student = 'saleh' # add credentials if u want
    header = ['Agent_0:', 'Agent_1:', 'Number Of Simulations:', 'Agent_0 wins:', 'Agent_1 wins:', 'Draws:', 'Errors', 'Student:']
    # we assume that you're in the: "simulator" directory
    shutil.copy('../submission.py', './submission.py')
    if not os.path.isdir('./output'):
        os.mkdir("./output")
    if not os.path.isdir('./errors'):
        os.mkdir("./errors")
    for couple,simulations_num in agents:
        agent_0 = couple[0]
        agent_1 = couple[1]
        match_out = './output/' + agent_0 + '_vs_' + agent_1 + '.txt'
        match_err = './errors/' + agent_0 + '_vs_' + agent_1 + '.txt'
        if os.path.isfile(match_out):
            os.remove(match_out) # erase previous results and make new ones later on
        if os.path.isfile(match_err):
            os.remove(match_err) # erase previous results and make new ones later on
        open(match_out, 'w+')
        open(match_err, 'w+')
            
    with open('./output/excel_file.csv', 'w') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        for couple,simulations_num in agents:
            agent_0 = couple[0]
            agent_1 = couple[1]
            agent_0_wins = 0
            agent_1_wins = 0
            draws = 0
            errors = 0
            match_out = './output/' + agent_0 + '_vs_' + agent_1 + '.txt'
            match_err = './errors/' + agent_0 + '_vs_' + agent_1 + '.txt'
            for i in range(simulations_num):
                # exit_val = os.system("python main.py " + agent_0 + " " + agent_1 + " -t 1 -s 1234 -c 200 --console_print --screen_print") # to watch the match
                seed = random.randint(0,1234)
                steps_num = random.randint(50,500)
                # seed = 1234
                # steps_num = 200
                seed_arg = " -s " + str(seed)
                step_num_arg = " -c " + str(steps_num)
                cmd = "python ./main.py " + agent_0 + " " + agent_1 + " -t 1" + seed_arg + step_num_arg
                exit_val = os.system( cmd + ' >> ' + f'{match_out}' + ' 2>> ' + f'{match_err}' )
                if exit_val == 20:
                    agent_0_wins += 1
                elif exit_val == 21:
                    agent_1_wins += 1
                elif exit_val == 22:
                    draws += 1
                else:
                    errors += 1
            row = [agent_0, agent_1, simulations_num, agent_0_wins, agent_1_wins, draws, errors, student]
            csvwriter.writerow(row)
           # takes too long!
simulations()