import numpy as np
import chess

def board_to_tensor(board):

    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    piece_map = board.piece_map()

    # Chess pieces 
    # True for white, False for black
    piece_to_channel = {
        (chess.PAWN, True): 0,
        (chess.KNIGHT, True): 1,
        (chess.BISHOP, True): 2,
        (chess.ROOK, True): 3,
        (chess.QUEEN, True): 4,
        (chess.KING, True): 5,
        (chess.PAWN, False): 6,
        (chess.KNIGHT, False): 7,
        (chess.BISHOP, False): 8,
        (chess.ROOK, False): 9,
        (chess.QUEEN, False): 10,
        (chess.KING, False): 11
    }

    for square, piece in piece_map.items():

        row = 7 - (square // 8)
        col = square % 8

        channel = piece_to_channel[(piece.piece_type, piece.color)]

        tensor[channel][row][col] = 1

    # Side to move
    tensor[12][:][:] = int(board.turn)

    # Castling rights
    tensor[13][:][:] = int(board.has_kingside_castling_rights(chess.WHITE))
    tensor[14][:][:] = int(board.has_queenside_castling_rights(chess.WHITE))
    tensor[15][:][:] = int(board.has_kingside_castling_rights(chess.BLACK))
    tensor[16][:][:] = int(board.has_queenside_castling_rights(chess.BLACK))

    # En passant
    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        tensor[17][row][col] = 1

    return tensor