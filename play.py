import numpy as np
import argparse
import os
import json
import time
import random
import tqdm

from rl_gomoku.envs.array_gomoku import ArrayGomoku
from rl_gomoku.utils import start_play, collect_data, start_self_play, PolicyValueNet, collect_self_play_data
from rl_gomoku.agents.mcts_alpha import MCTSZeroAgent
from rl_gomoku.agents import GreedyAgent
from rl_gomoku.envs import GomokuEnv
from collections import deque


cache_data = json.load(open("./cache_data/quick_list_set.json", "r"))
parser = argparse.ArgumentParser(description='Example Argparse Program')

parser.add_argument('--checkpoint', type=str, default=None, help='Input file')


args = parser.parse_args()


class configs:

    training_steps = 3000
    save_dir = "./model"
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    n_playout = 1200
    board_width = 15
    board_height = 15
    
    device = "cpu"
    

class context:
    env = ArrayGomoku(board_size=configs.board_width)
    buffer = deque(maxlen=9999)

    net = PolicyValueNet(configs.board_width, configs.board_height, device=configs.device)
    if args.checkpoint:
        net.load_checkpoint(args.checkpoint)
    else:
        raise ValueError("model not found!")
    
    mcts_player = MCTSZeroAgent(net.policy_value_fn, player_id=1, n_playout=configs.n_playout)
    player_a = GreedyAgent(configs.board_height, player_id=1, cache_data=cache_data)
    player_b = GreedyAgent(configs.board_height, player_id=2, cache_data=cache_data)
    

for i in tqdm.tqdm(range(configs.training_steps)):
    buffer = context.buffer
    agent = context.mcts_player
    env = context.env
    data_shape = (configs.board_width, configs.board_height)
    
    seed = int(time.time())
    context.player_b.rng = np.random.default_rng(seed)
    context.player_b.rng = np.random.default_rng(seed)

    winner, *_ = start_play(env, context.mcts_player, context.player_b, render=True)
    time.sleep(5)
