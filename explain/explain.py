import chess
from explain.move_analysis import analyze_move

def generate_explanation(features):
    # Priority order

    if features["is_checkmate"]:
        return "This move delivers checkmate."
    
    if features["captures_hanging_piece"]:
        return "This move captures a hanging piece, winning material for free."
    
    if features["creates_skewer"]:
        return "This move creates a skewer, forcing a high-value piece to move and exposing another."

    if features["creates_pin"]:
        return "This move creates a pin, restricting the opponent's piece from moving."
    
    if features["creates_hanging_piece"]:
        return "This move creates a threat against an undefended piece."

    if features["is_check"] and features["creates_fork"]:
        return "This move delivers check while creating a fork, attacking multiple pieces."

    if features["creates_fork"]:
        return "This move creates a fork, attacking multiple valuable pieces at once."

    if features["material_gain"] > 0 and features["is_capture"]:
        return "This move captures material and gains an advantage."

    if features["is_favorable_trade"]:
        return "This move forces a favorable trade, improving material balance."

    if features["is_check"]:
        return "This move puts the opponent in check."

    if features["controls_center"] and features["improves_mobility"]:
        return "This move improves piece activity and increases control of the center."

    if features["improves_mobility"]:
        return "This move improves piece mobility and overall activity."

    if features["develops_piece"]:
        return "This move develops a piece to a more active position."

    return "This move improves the position."

def explain_move(board: chess.Board, move: chess.Move):
    features = analyze_move(board, move)
    explanation = generate_explanation(features)
    return explanation