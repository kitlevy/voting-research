from helpers import *
from tests import *

P1 = [2/3, 1/3, 0]
P2 = [1/3, 2/3, 0]
P3 = [0, 2/3, 1/3]
P4 = [0, 1/3, 2/3]
P5 = [1/3, 0, 2/3]
P = [P1, P2, P3, P4, P5]

Q1 = [1, 0, 0]
Q2 = [2/3, 0, 1/3]
Q3 = [1/2, 0, 1/2]
Q = [Q1, Q2, Q3]

print("Solving region cab")
for i in range(len(P)):
    p = P[i]
    for q in Q:
        print("P{}".format(i + 1), "({}, {}, {})".format(q[0],q[1],q[2]), solve_linear_threshold(p, q, i=2, j=0, k=1), '\n')
