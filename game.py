import pygame
import chess
from model import board_to_tensor, load_model
from mcts import mcts_chess_move
import torch
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((480, 480))
pygame.display.set_caption('Chess')

# Load images
def load_images():
    pieces = ['white-pawn', 'white-rook', 'white-knight', 'white-bishop', 'white-queen', 'white-king',
              'black-pawn', 'black-rook', 'black-knight', 'black-bishop', 'black-queen', 'black-king']
    images = {}
    for piece in pieces:
        path = os.path.join('images', f'{piece}.png')
        if os.path.exists(path):
            image = pygame.image.load(path).convert_alpha()
            image = pygame.transform.scale(image, (60, 60))  # Scale image to 60x60 pixels to fit the squares
            images[piece] = image
            print(f"Loaded and scaled image: {path}")
        else:
            print(f"Warning: Image {path} not found. Using placeholder.")
            images[piece] = pygame.Surface((60, 60))  # Placeholder
    return images

images = load_images()

def draw_board(screen, board, selected_square, legal_moves):
    colors = [pygame.Color(235, 235, 208), pygame.Color(119, 148, 85)]
    for r in range(8):
        for c in range(8):
            color = colors[(r + c) % 2]
            square = chess.square(c, 7 - r)
            if square == selected_square:
                color = pygame.Color(186, 202, 43)
            pygame.draw.rect(screen, color, pygame.Rect(c * 60, r * 60, 60, 60))
            if square in legal_moves:
                pygame.draw.circle(screen, pygame.Color(246, 246, 105), (c * 60 + 30, r * 60 + 30), 10)
            piece = board.piece_at(square)
            if piece:
                draw_piece(screen, piece, c * 60, r * 60)

def draw_piece(screen, piece, x, y):
    piece_names = {
        chess.PAWN: 'pawn',
        chess.ROOK: 'rook',
        chess.KNIGHT: 'knight',
        chess.BISHOP: 'bishop',
        chess.QUEEN: 'queen',
        chess.KING: 'king'
    }
    color = 'white' if piece.color == chess.WHITE else 'black'
    piece_name = f"{color}-{piece_names[piece.piece_type]}"
    if piece_name in images:
        screen.blit(images[piece_name], (x, y))
    else:
        print(f"Warning: {piece_name} not found in images.")

def play_against_model(model):
    model.eval()
    board = chess.Board()
    running = True
    selected_square = None
    legal_moves = []
    player_turn = True

    while running:
        draw_board(screen, board, selected_square, legal_moves)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if player_turn:
                    x, y = event.pos
                    col, row = x // 60, 7 - (y // 60)
                    square = chess.square(col, row)
                    if selected_square is None:
                        if board.piece_at(square) and board.piece_at(square).color == chess.WHITE:
                            selected_square = square
                            legal_moves = [move.to_square for move in board.legal_moves if move.from_square == selected_square]
                    else:
                        move = chess.Move(selected_square, square)
                        if move in board.legal_moves:
                            board.push(move)
                            player_turn = False
                            selected_square = None
                            legal_moves = []
                        else:
                            selected_square = None
                            legal_moves = []

        if not player_turn and not board.is_game_over():
            _, model_move = mcts_chess_move(board, model, num_episodes=100)
            board.push(model_move)
            player_turn = True

    pygame.quit()
