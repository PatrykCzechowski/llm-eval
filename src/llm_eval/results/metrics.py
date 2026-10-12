def agreement(judge: list[str], humans: list[str]) -> float:
    if len(judge) != len(humans):
        raise ValueError("The length of judge and humans lists must be the same.")
    if len(judge) == 0:
        raise ValueError("The lists must not be empty.")
    agreement_count = sum(1 for j, h in zip(judge, humans) if j == h)
    return agreement_count / len(judge)
