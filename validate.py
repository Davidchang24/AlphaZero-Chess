import torch
import chess
import random
from model import ChessNN, board_to_tensor, load_model
from mcts import mcts_chess_move

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(model, num_games=100, num_episodes=10):
    model.eval()
    results = []
    for game in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            state_tensor = board_to_tensor(board, device)
            _, move = mcts_chess_move(board, model, num_episodes)
            board.push(move)
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            board.push(random.choice(legal_moves))

        result = board.result()
        results.append(result)

    wins = results.count('1-0')
    losses = results.count('0-1')
    draws = results.count('1/2-1/2')
    print(f"Validation Results: {wins} wins, {losses} losses, {draws} draws")


if __name__ == "__main__":
    model = ChessNN().to(device)
    load_model(model, "chess_model.pth", device)
    validate(model, num_games=100, num_episodes=10)
