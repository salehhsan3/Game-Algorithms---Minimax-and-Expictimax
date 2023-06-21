import os
import contextlib
import csv
# format: ((agent_0, agent_1), simulations_num)
agents = [
        (("greedyImproved",	"greedyImproved"), 10),
        (("minimax",	"minimax"), 10),
        (("alphabeta",	"alphabeta"), 10),
        (("expectimax",	"expectimax"), 10),
        (("greedyImproved",	"expectimax"), 10),
        (("minimax",	"greedyImproved"), 10),
        (("alphabeta",	"minimax"), 10),
        (("expectimax",	"alphabeta"), 10),
        (("greedyImproved",	"alphabeta"), 10),
        (("minimax",	"expectimax"), 10),
        (("alphabeta",	"greedyImproved"), 10),
        (("expectimax",	"minimax"), 10),
        (("greedyImproved",	"minimax"), 10),
        (("minimax",	"alphabeta"), 10),
        (("alphabeta",	"expectimax"), 10),
        (("expectimax",	"greedyImproved"), 10),        
    ]


def simulations():
    student = 'saleh' # add credentials if u want
    header = ['Agent_0:', 'Agent_1:', 'Number Of Simulations:', 'Agent_0 wins:', 'Agent_1 wins:', 'Draws:', 'Errors', 'Student:']
    data = []
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
            os.remove(match_out)
        if os.path.isfile(match_err):
            os.remove(match_err)
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
                cmd = "python main.py " + agent_0 + " " + agent_1 + " -t 1 -s 1234 -c 200"
                exit_val = os.system( cmd + ' >> ' + f'{match_out}' + ' 2>> ' + f'{match_err}' )
                if exit_val == 256:
                    agent_0_wins += 1
                elif exit_val == 257:
                    agent_1_wins += 1
                elif exit_val == 258:
                    draws += 1
                else:
                    errors += 1
            row = [agent_0, agent_1, simulations_num, agent_0_wins, agent_1_wins, draws, errors, student]
            csvwriter.writerow(row)
           # started: 11:51
           # finished: 12:14
           # takes too long!
simulations()