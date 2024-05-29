import torch
import torch.optim as optim
import chess
import random
from model import ChessNN, board_to_tensor, save_model
from mcts import mcts_chess_move, loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train(model, optimizer, num_games=10, num_episodes=10):
    model.train()
    for game in range(num_games):
        print(f"Training game {game + 1}/{num_games}")
        board = chess.Board()
        states, mcts_policies, rewards = [], [], []
        while not board.is_game_over():
            state_tensor = board_to_tensor(board, device)
            states.append(state_tensor)
            mcts_policy, move = mcts_chess_move(board, model, num_episodes)
            mcts_policies.append(mcts_policy)
            board.push(move)
            if board.is_game_over():
                break
            # Opponent move (random move for simplicity)
            legal_moves = list(board.legal_moves)
            board.push(random.choice(legal_moves))

        # Determine the game result
        result = board.result()
        reward = 1 if result == '1-0' else -1 if result == '0-1' else 0
        rewards.extend([reward] * len(states))

        # Update the model
        optimizer.zero_grad()
        losses = []
        for state, mcts_policy, reward in zip(states, mcts_policies, rewards):
            state = state.unsqueeze(0)
            pred_p, pred_v = model(state)
            target_p = torch.tensor(mcts_policy, dtype=torch.float32, device=device).unsqueeze(0)
            target_v = torch.tensor([reward], dtype=torch.float32, device=device).unsqueeze(0)
            loss = loss_fn(pred_p, pred_v, target_p, target_v, model)
            losses.append(loss.item())
            loss.backward()
        optimizer.step()
        print(f"Game {game + 1} loss: {sum(losses) / len(losses)}")


if __name__ == "__main__":
    model = ChessNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, num_games=10, num_episodes=10)
    save_model(model)
