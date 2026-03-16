import chess

# Map UCI move → class index (0–4095)
def uci_to_class(uci: str) -> int:
    from_square = chess.parse_square(uci[:2])
    to_square = chess.parse_square(uci[2:4])
    return from_square * 64 + to_square

# Map class index → UCI move
def class_to_uci(cls: int) -> str:
    from_square = cls // 64
    to_square = cls % 64
    return chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]