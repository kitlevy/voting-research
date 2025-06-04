import numpy as np
import math
from helpers import *
from preflibtools.properties import has_condorcet

def first_past_the_post(instance):
    rankings, weights = orders_to_matrix(instance.flatten_strict(), instance.num_alternatives)
    
    return get_modes_by_col(rankings,weights)
    
def borda(instance):
    n = instance.num_alternatives
    rankings, weights = orders_to_matrix(instance.flatten_strict(), n)
    counts = {}
    for alt, alt_name in instance.alternatives_name.items():
        counts[alt] = 0

    for row in range(len(rankings)):
        w = weights[row][0]
        value = n-1 
        for i in range(n-1):
            alt = rankings[row][i]
            counts[alt] += value * w
            value -= 1

    return get_column_winner(counts)

def IRV(instance):
    remaining = list(instance.alternatives_name.keys())
    rankings, weights = orders_to_matrix(instance.flatten_strict(), instance.num_alternatives)
    
    while len(remaining) > 1:
        counts = restricted_count(rankings, weights, remaining)
        losers = get_column_loser(counts)
        for loser in losers:
            remaining.remove(loser)

    return remaining

def copeland(instance, tie_weight=0.5):
    n = instance.num_alternatives
    rankings, weights = orders_to_matrix(instance.flatten_strict(), instance.num_alternatives)
    points = n * [0]
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if a == b:
                continue
            x = net_compare_a_over_b(rankings, weights, a, b)
            if x > 0:
                points[a - 1] += win_weight
            elif x == 0:
                points[a - 1] += tie_weight
    
    max_alt = []
    max_seen = -math.inf
    for alt in range(1, n + 1):
        if points[alt - 1] > max_seen:
            max_seen = points[alt - 1]
            max_alt = [alt]
        elif points[alt - 1] == max_seen:
            max_alt.append(alt)

    return max_alt

def find_smith_set(instance):
    n = instance.num_alternatives
    rankings, weights = orders_to_matrix(instance.flatten_strict(), n)
    net_counts = np.zeros((n, n), dtype=int)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            x = net_compare_a_over_b(rankings, weights, a, b)
            net_counts[a - 1][b - 1] = x
            net_counts[b - 1][a - 1] = -x
    sccs = kosaraju_scc(net_counts)
    return check_smith_set(net_counts, sccs)

def net_condorcet(instance, tiebreak=True):
    if not instance.has_condorcet and not tiebreak:
        return []
    n = instance.num_alternatives
    rankings, weights = orders_to_matrix(instance.flatten_strict(), instance.num_alternatives)
    net_counts = np.zeros((n,n))
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            x = net_compare_a_over_b(rankings, weights, a, b)
            net_counts[a - 1][b - 1] = x
            net_counts[a - 1][b - 1] = -x
        #case when outright winner is found
        if all(net_counts[a - 1][i] > 0 for i in range(n) if i != a - 1):
            return [a]

    #case when no outright winner found
    #get smith set of finalists
    finalists = kosaraju_scc(net_counts)
    finalists = check_smith_set(net_counts, finalists)

    #tiebreak: margin of worst loss
    best_seen = math.inf
    best_alt = []
    for alt in finalists:
        worst = min(net_counts[alt - 1])
        if worst > best_seen:
            best_seen = worst
            best_alt = [alt]
        elif worst == best_seen:
            best_alt.append(alt)
    return best_alt







#testing zone



def test_cond(arr):
    finalists = kosaraju_scc(arr)
    finalists = check_smith_set(arr, finalists)
    best_seen = -math.inf
    best_alt = []
    for alt in finalists:
        worst = min(net_counts[alt - 1])
        if worst > best_seen:
            best_seen = worst
            best_alt = [alt]
        elif worst == best_seen:
            best_alt.append(alt)
    return best_alt

prefs = np.array([
    [0, -6, 5, -12, -24, 0],
    [6, 0, 2, -3, -12, -3],
    [-5, -2, 0, -4, -2, -4],
    [12, 3, 4, 0, 16, -27],
    [24, 12, 2, -16, 0, 13],
    [0, 3, 4, 27, -13, 0]
    ])

print(test_cond(prefs))


def test_IRV(rankings, weights, remaining):    
    while len(remaining) > 1:
        counts = restricted_count(rankings, weights, remaining)
        losers = get_column_loser(counts)
        for loser in losers:
            remaining.remove(loser)

    return remaining

def test_net_cond_matrix(rankings, weights, n):
    net_counts = np.zeros((n,n))
    winners = []
    #don't have to loop through all of them
    for a in range(1, n + 1 // 2 + 1):
        cond_winner = True
        for b in range(a + 1, n + 1):
            x = net_compare_a_over_b(rankings, weights, a, b)
            if cond_winner and x < 0:
                cond_winner = False
            net_counts[a - 1][b - 1] = x
            net_counts[b - 1][a - 1] = -x
        #if cond_winner:
            #return [a]
    return net_counts

def test_cond_matrix(rankings, weights, n):
    pairwise_counts = np.zeros((n,n))
    winners = []
    #creating pairwise matrix, dealing with scenario where there is an outright winner
    for a in range(1, n + 1):
        cond_winner = True
        not_all_zero = False
        for b in range(1, n + 1):
            x = compare_a_over_b(rankings, weights, a, b)
            if not_all_zero == False and x:
                not_all_zero = True
            if cond_winner and x < 0:
                cond_winner = False
            pairwise_counts[a - 1][b - 1] = x
        if cond_winner and not_all_zero:
            #return [a]
            continue
    return pairwise_counts

def test_copeland(rankings, weights, n, win_weight=1, tie_weight=0.5):
    points = n * [0]
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if a == b:
                continue
            x = net_compare_a_over_b(rankings, weights, a, b)
            if x > 0:
                points[a - 1] += win_weight
            elif x == 0:
                points[a - 1] += tie_weight
    
    max_alt = []
    max_seen = -math.inf
    for alt in range(1, n + 1):
        if points[alt - 1] > max_seen:
            max_seen = points[alt - 1]
            max_alt = [alt]
        elif points[alt - 1] == max_seen:
            max_alt.append(alt)

    return max_alt



a = ((1,2,3,4,5),8)
b = ((2,1,3,4,5),8)
c = ((3,1,4,5,2),9)
d = ((1,4,5,2,3),2)
e = ((4,2,3,5,1),9)
f = ((1,5,4,3,2),2)


#should give 4 with IRV
a = ((4,3,1,5,2),9)
b = ((2,5,1,3,4),5)
c = ((5,1,4,2,3),2)
d = ((2,3,1,4,5),5)
e = ((3,1,4,2,5),8)
f = ((2,4,3,1,5),6)

#should give 4 with copeland
a = ((2,3,1,4,5),3)
b = ((3,1,4,2,5),4)
c = ((2,4,3,1,5),4)
d = ((4,3,1,5,2),6)
e = ((2,5,1,3,4),2)
f = ((5,1,4,2,3),1)


arr = [a,b,c,d,e,f]
r,w = orders_to_matrix(arr,5)

#rem = [1,2,3,4,5]
#print(test_IRV(r,w,rem)) #should give 4

#print(test_cond_matrix(r, w, 5))
#print(test_net_cond_matrix(r, w, 5))

#print(test_copeland(r, w, 5))




