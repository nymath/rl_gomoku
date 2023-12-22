import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .envs import ArrayGomoku
from .agents import MCTSZeroAgent, GreedyAgent
from .common import NaiveNet, AlphaZeroValueNet
from typing import Optional, Union


def ensure_tuple(x, size=15):
    if isinstance(x, tuple):
        return x
    else:
        return x // size, x % size
    

def start_play(
    env: ArrayGomoku, 
    player_1: Union[MCTSZeroAgent, GreedyAgent], 
    player_2: Union[MCTSZeroAgent, GreedyAgent], 
    render=False, 
    temp=1e-3
):
    
    env.reset()

    states, mcts_probs, current_players = [], [], []
    while True:
        if env.current_player == 1:
            move, move_probs = player_1.get_action(env, temp, 1)

        else:
            move, move_probs = player_2.get_action(env, temp, 1)

        states.append(env.current_state())
        mcts_probs.append(move_probs)
        current_players.append(env.current_player)
        # perform a move
        env.step(move)
        if render:
            env.render()
        terminated, winner = env.terminated()

        if terminated:
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0

            try:
                player_1.reset_player()
                player_2.reset_player()
            except Exception:
                pass
            finally:
                break
    return winner, zip(states, mcts_probs, winners_z)


def replay_data_from_list(env: ArrayGomoku, position_list, render=False):
    env.reset()

    states, mcts_probs, current_players = [], [], []
    for i in range(len(position_list)):
        move = position_list[i]
        move_probs = np.zeros(shape=(225, ))
        move_probs[move] = 1
        states.append(env.current_state())
        mcts_probs.append(move_probs)
        current_players.append(env.current_player)
 
        env.step(move)
        if render:
            env.render()
        terminated, winner = env.terminated()

        if terminated:
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            break
        
    return winner, zip(states, mcts_probs, winners_z) 
    

def start_self_play(env: ArrayGomoku, player: MCTSZeroAgent, render=False, temp=1e-3):
    """ start a self-play game using a MCTS player, reuse the search tree,
    and store the self-play data: (state, mcts_probs, z) for training
    """
    env.reset()
    states, mcts_probs, current_players = [], [], []

    while True:
        move, move_probs = player.get_action(env,
                                             temp=temp,
                                             return_prob=1,
                                             is_selfplay=True)
        # store the data

        states.append(env.current_state())
        mcts_probs.append(move_probs)
        current_players.append(env.current_player)
        # perform a move
        env.step(move)
        if render:
            env.render()

        terminated, winner = env.terminated()
        if terminated:
            # winner from the perspective of the current player of each state
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            # reset MCTS root node
            player.reset_player()
            if render:
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
            break

    return winner, zip(states, mcts_probs, winners_z)


def get_equi_data(play_data, data_shape=(15, 15)):
    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    # Data Augmentation
    extend_data = []
    for state, mcts_porb, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(
                mcts_porb.reshape(*data_shape)), i)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
    return extend_data


def collect_data(buffer, env, player_1, player_2, n_games=1, data_shape=(15, 15), render=False):
    """collect self-play data for training"""
    episode_len = []
    for i in range(n_games):
        winner, play_data = start_play(env, player_1, player_2, render=render)  # TODO: control temp
        play_data = list(play_data)[:]
        episode_len.append(len(play_data))
        play_data = get_equi_data(play_data, data_shape)
        buffer.extend(play_data)

    return sum(episode_len) / n_games


def collect_self_play_data(buffer, env, player_1, n_games=1, data_shape=(15, 15), render=False):
    """collect self-play data for training"""
    episode_len = []
    winners = np.array([0, 0, 0])
    for i in range(n_games):
        winner, play_data = start_self_play(env, player_1, render=render)  # TODO: control temp
        play_data = list(play_data)[:]
        if winner == -1:
            winners[0] += 1
        else:
            winners[winner] += 1
            
        episode_len.append(len(play_data))
        play_data = get_equi_data(play_data, data_shape)
        buffer.extend(play_data)

    return sum(episode_len) / n_games, winners


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class PolicyValueNet:
    """policy-value network """

    def __init__(
            self,
            board_width=15,
            board_height=15,
            device="cuda:0",
            in_channels=7,
            drop_out=0.1,
            lr=0.0002,
            weight_decay=0.0001,
            use_attention=False,
    ):

        self.device = torch.device(device)
        self.factory_kwargs = {"dtype": torch.float32, "device": self.device}

        self.board_width = board_width
        self.board_height = board_height
        self.weight_decay = weight_decay
        self.net = AlphaZeroValueNet(
            board_size=board_height, 
            in_channels=in_channels, 
            dropout=drop_out,
            use_attention=use_attention,
            )
 
        self.net = self.net.to(**self.factory_kwargs)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=self.weight_decay)

    @torch.no_grad()
    def policy_value(self, state_batch):
        self.eval()
        state_batch_tensor = torch.tensor(np.array(state_batch), **self.factory_kwargs)
        log_act_probs, value = self.net(state_batch_tensor)
        act_probs = np.exp(log_act_probs.cpu().numpy())
        return act_probs, value.cpu().numpy()

    @torch.no_grad()
    def policy_value_fn(self, env):
        self.eval()
        legal_positions = env.availables
        current_state = env.current_state()

        current_state = torch.tensor(np.array(current_state)).to(**self.factory_kwargs)
        with torch.no_grad():
            log_act_probs, value = self.net(current_state.unsqueeze(0))

        act_probs = np.exp(log_act_probs.squeeze().cpu().numpy())
        act_probs = zip(legal_positions, act_probs[legal_positions])

        return act_probs, value.item()


    def train_step(self, state_batch, mcts_probs, winner_batch):
        self.train()

        state_batch = torch.tensor(np.array(state_batch)).to(**self.factory_kwargs)
        mcts_probs = torch.tensor(np.array(mcts_probs)).to(**self.factory_kwargs)
        winner_batch = torch.tensor(np.array(winner_batch)).to(**self.factory_kwargs)

        log_act_probs, value = self.net(state_batch)

        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = - torch.mean(torch.sum(mcts_probs * log_act_probs, 1)) # kl divergence
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))

        return loss.item(), entropy.item()

    def load_checkpoint(self, path):
        check_point = torch.load(path)
        self.net.load_state_dict(check_point["model_state_dict"])
        self.optimizer.load_state_dict(check_point["optim_state_dict"])

    def save_checkpoint(self, path):
        """ save model params to file """
        model_state_dict = self.net.state_dict()
        optim_state_dict = self.optimizer.state_dict()
        torch.save({
            "model_state_dict": model_state_dict,
            "optim_state_dict": optim_state_dict,
        }, path)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()