from model import ChessNN, save_model, load_model
from train import train
from validate import validate
from game import play_against_model
import torch.optim as optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = ChessNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_iterations = 10  # Number of training-validation iterations
    games_per_iteration = 100  # Number of games per training iteration
    episodes_per_game = 10  # Number of MCTS episodes per game
    validation_games = 50  # Number of games for validation

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")

        # Train the model
        train(model, optimizer, num_games=games_per_iteration, num_episodes=episodes_per_game)

        # Save the model
        save_model(model, path=f"chess_model_iter_{iteration + 1}.pth")

        # Validate the model
        validate(model, num_games=validation_games, num_episodes=episodes_per_game)

    # Final save
    save_model(model, path="chess_model_final.pth")

    # Play against the trained model
    load_model(model, "chess_model_final.pth", device)
    play_against_model(model)

if __name__ == "__main__":
    main()
