import os
import shutil
import csv
# format: ((agent_0, agent_1), simulations_num)
agents = [
        (("greedyImproved",	"greedyImproved"), 4),
        (("minimax",	"minimax"), 4),
        (("alphabeta",	"alphabeta"), 4),
        (("expectimax",	"expectimax"), 4),
        (("greedyImproved",	"expectimax"), 4),
        (("minimax",	"greedyImproved"), 4),
        (("alphabeta",	"minimax"), 4),
        (("expectimax",	"alphabeta"), 4),
        (("greedyImproved",	"alphabeta"), 4),
        (("minimax",	"expectimax"), 4),
        (("alphabeta",	"greedyImproved"), 4),
        (("expectimax",	"minimax"), 4),
        (("greedyImproved",	"minimax"), 4),
        (("minimax",	"alphabeta"), 4),
        (("alphabeta",	"expectimax"), 4),
        (("expectimax",	"greedyImproved"), 4),        
    ]


def simulations():
    student = 'saleh' # add credentials if u want
    header = ['Agent_0:', 'Agent_1:', 'Number Of Simulations:', 'Agent_0 wins:', 'Agent_1 wins:', 'Draws:', 'Errors', 'Student:']
    data = []
    if not os.path.isdir('./simulator'):
        os.mkdir("./simulator")
    shutil.copy('./submission.py', './simulator/submission.py')
    if not os.path.isdir('./simulator/output'):
        os.mkdir("./simulator/output")
    if not os.path.isdir('./simulator/errors'):
        os.mkdir("./simulator/errors")
    for couple,simulations_num in agents:
        agent_0 = couple[0]
        agent_1 = couple[1]
        match_out = './simulator/output/' + agent_0 + '_vs_' + agent_1 + '.txt'
        match_err = './simulator/errors/' + agent_0 + '_vs_' + agent_1 + '.txt'
        if os.path.isfile(match_out):
            os.remove(match_out)
        if os.path.isfile(match_err):
            os.remove(match_err)
        open(match_out, 'w+')
        open(match_err, 'w+')
            
    with open('./simulator/output/excel_file.csv', 'w') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        for couple,simulations_num in agents:
            agent_0 = couple[0]
            agent_1 = couple[1]
            agent_0_wins = 0
            agent_1_wins = 0
            draws = 0
            errors = 0
            match_out = './simulator/output/' + agent_0 + '_vs_' + agent_1 + '.txt'
            match_err = './simulator/errors/' + agent_0 + '_vs_' + agent_1 + '.txt'
            for i in range(simulations_num):
                # exit_val = os.system("python main.py " + agent_0 + " " + agent_1 + " -t 1 -s 1234 -c 200 --console_print --screen_print") # to watch the match
                cmd = "python ./simulator/main.py " + agent_0 + " " + agent_1 + " -t 1 -s 1234 -c 200"
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
           # started: 11:51
           # finished: 12:14
           # takes too long!
simulations()