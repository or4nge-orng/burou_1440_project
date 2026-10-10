

def taylor_sin(x: float, n: int) -> float:
    if n < 0:
        raise ValueError("n must be non-negative")
    res = 0
    term = x
    for i in range(n):
        res += term
        term *= -(x**2) / ((2 * i + 2) * (2 * i + 3))
        
    return res

def taylor_exp(x: float, n: int) -> float:
    if n < 0:
        raise ValueError("n must be non-negative")
    res = 0
    term = 1
    for i in range(n):
        res += term
        term *= x / (i + 1)
    
    return res

def taylor_log1p(x: float, n: int) -> float:
    if n < 0:
        raise ValueError("n must be non-negative")
    
    res = 0.0
    for i in range(n):
        res += x * (-x)**i / (i + 1)

    return res
    