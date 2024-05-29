import chess
import numpy as np
import random
import torch

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.policy = 0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.state.legal_moves))

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for move, child in self.children.items()
        ]
        best_move = max(self.children.items(), key=lambda item: (item[1].value / item[1].visits) + c_param * np.sqrt(
            (2 * np.log(self.visits) / item[1].visits)))[0]
        return self.children[best_move], best_move

    def add_child(self, move, child_state):
        child = Node(child_state, self)
        self.children[move] = child
        return child

def tree_policy(node, model):
    while not node.state.is_game_over():
        if not node.is_fully_expanded():
            return expand(node)
        else:
            node, move = node.best_child()
    return node

def expand(node):
    tried_moves = node.children.keys()
    legal_moves = list(node.state.legal_moves)
    for move in legal_moves:
        if move not in tried_moves:
            new_state = node.state.copy()
            new_state.push(move)
            return node.add_child(move, new_state)
    raise Exception("Should not reach here")

def simulate_game(state):
    board = state.copy()
    while not board.is_game_over():
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        board.push(move)
    result = board.result()
    if result == '1-0':
        return 1
    elif result == '0-1':
        return -1
    else:
        return 0

def best_action(root, model, num_episodes):
    for i in range(num_episodes):
        if i % 10 == 0:
            node = tree_policy(root, model)
            reward = simulate_game(node.state)
            backpropagate(node, reward)

    # Calculate policy based on visit counts
    visit_counts = np.array([child.visits for move, child in root.children.items()])
    policy = visit_counts / visit_counts.sum()
    best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return policy, best_move

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.value += (reward - node.value) / node.visits
        node = node.parent

def mcts_chess_move(board, model, num_episodes):
    root = Node(board)
    policy, best_move = best_action(root, model, num_episodes)
    return policy, best_move

def loss_fn(pred_p, pred_v, target_p, target_v, model, c=1e-4):
    value_loss = (target_v - pred_v) ** 2
    policy_loss = -torch.sum(target_p * torch.log(pred_p))
    l2_loss = c * sum(p.pow(2).sum() for p in model.parameters())
    return value_loss + policy_loss + l2_loss
