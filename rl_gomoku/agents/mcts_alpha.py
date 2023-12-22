# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy
from typing import Callable, Tuple
from ..envs.array_gomoku import ArrayGomoku


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


PolicyValue = Callable[[ArrayGomoku], Tuple[np.ndarray, float]]


class TreeNode(object):
    def __init__(self, parent, prior_p, position_idx=None):
        self.parent = parent
        self.children = {}  # a map from action to TreeNode
        self.n_visits = 0
        self.position_idx = position_idx
        self.Q = 0
        self.P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob, action)

    def select(self, c_param) -> tuple[int, "TreeNode"]:
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_param))

    def update(self, leaf_value):
        self.n_visits += 1
        # (n+1)s_{n+1} = n * s_n + x_{n+1}
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        # If it is not root, this node's parent should be updated first.
        # 感觉更新顺序似乎没有影响
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_param):
        confident_bound = self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + c_param * confident_bound

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTSZero(object):
    def __init__(
            self,
            policy_value_fn: PolicyValue,
            c_param=5,
            n_playout=10000
    ):
        self.root = TreeNode(None, 1.0)
        self.policy = policy_value_fn
        self.c_param = c_param
        self.n_playout = n_playout
        self.checkpoint = None

    def _playout(self, env: ArrayGomoku):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_param)
            env.step(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        # TODO:
        action_probs, leaf_value = self.policy(env)
        # Check for end of game.
        end, winner = env.terminated()  # end 一定是由node造成的, 但是current_layer 在 state.do_move时已经进行切换了

        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = -1
        # we do not rollout, insteat we just step one time and then use nn to summarize the reward afterwards.
        # Update value and visit count of nodes in this traversal.
        # here we input -reward, since the reward is obtained by the child of node;
        node.update_recursive(-leaf_value)

    def get_move_probs(self, env: ArrayGomoku, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self.n_playout):
            # do plannings
            env.save_checkpoint()
            # 在playout时, node没有变化
            self._playout(env)
            env.load_checkpoint()
            
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.n_visits)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        # 看看temp的
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)


class MCTSZeroAgent(object):
    def __init__(
            self,
            policy_value_function,
            player_id=1,
            c_param=5,
            n_playout=2000,
    ):
        self.mcts = MCTSZero(policy_value_function, c_param, n_playout)
        self.player_id = player_id

    def reset_player(self):
        self.mcts.update_with_move(-1)
        self.mcts.checkpoint = {}

    def get_action(self, env, temp=1e-3, return_prob=0, is_selfplay=False):
        legal_moves = env.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(env.width * env.height)
        if len(legal_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp)
            move_probs[list(acts)] = probs
            if is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                # TODO: modify this
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it iss almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                # self.mcts.update_with_move(move)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
