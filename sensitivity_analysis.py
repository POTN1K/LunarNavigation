"""Script to perform unit test on trade off table.
It runs the viable weights for different criteria.
By Nikolaus Ricker & Lennart van der Peet"""


# External Libraries
import numpy as np
from itertools import product

def sensitivity_analysis(score, weight):
    if score.shape[0] != 2:
        raise ValueError("The score matrix should have two rows")
    if score.shape[1] != weight.shape[0]:
        raise ValueError("The score matrix and weight matrix should have the same number of columns")

    win = np.zeros(3)
    for comb in weight:
        score1 = score[0,0]*comb[0] + score[0,1]*comb[1] + score[0,2]*comb[2] + score[0,3]*comb[3] + score[0,4]*comb[4] + score[0,5]*comb[5]
        score2 = score[1,0]*comb[0] + score[1,1]*comb[1] + score[1,2]*comb[2] + score[1,3]*comb[3] + score[1,4]*comb[4] + score[1,5]*comb[5]
        if score1 > score2:
            win[0] += 1
        elif score1 < score2:
            win[1] += 1
        else:
            win[2] += 1
    return win

if __name__ == "__main__":

    score_gs = np.array([[3,2,3,2,3,4], 
                        [4,1,2,2,2,2]])

    weight_gs = product(range(3,6), range(3,5), range(3,6), range(2,5), (3,5), (1,3))
    win_gs = sensitivity_analysis(score_gs, weight_gs)
    print(f"Points for ODTS: {win_gs[0]}\nPoints for Data Relay: {win_gs[1]}\nDraws: {win_gs[2]}")


    score_orbit = np.array([[4,3,3,3,4,3], 
                            [3,2,2,4,3,2]])
    weight_orbit = product(range(3,5), range(3,6), range(1,4), range(3,5), (1,5), (3,5))
    win_orbit = sensitivity_analysis(score_orbit, weight_orbit)
    print(f"Points for Lunar: {win_orbit[0]}\nPoints for Lagrange: {win_orbit[1]}\nDraws: {win_orbit[2]}")