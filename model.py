import torch
import torch.nn as nn
import chess

# Define the neural network model
class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.policy_head = nn.Linear(256, 4672)  # 4672 is an example size for all possible moves
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        p = torch.softmax(self.policy_head(x), dim=1)
        v = torch.tanh(self.value_head(x))
        return p, v

# Function to convert board state to neural network input
def board_to_tensor(board, device):
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32, device=device)
    piece_to_index = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square, piece in board.piece_map().items():
        index = piece_to_index[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
        row, col = divmod(square, 8)
        tensor[index, row, col] = 1
    return tensor

# Save the model
def save_model(model, path="chess_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load the model
def load_model(model, path="chess_model.pth", device=torch.device('cpu')):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
