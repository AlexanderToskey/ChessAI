def elo_to_bucket(elo: int) -> int:
    """Convert raw ELO to bucket 0–4"""
    if elo < 800:
        return 0
    elif elo < 1200:
        return 1
    elif elo < 1600:
        return 2
    elif elo < 2000:
        return 3
    else:
        return 4