class EnvPara:
    height = 11
    width = 11
    I_contr = (3, 4)
    I_ad = (9, 3)
    O = []
    A = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    A_full = {"North": (-1, 0), "South": (1, 0), "West": (0, -1), "East": (0, 1)}
    contr_rand = 0.0
    ad_rand = 0.05
    gamma = 0.95
