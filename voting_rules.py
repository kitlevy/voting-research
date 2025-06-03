import numpy as np
from helpers import *

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

#not using this one unless net counts prove better somehow
def trial_net_condorcet(instance):
    n = instance.num_alternatives
    rankings, weights = orders_to_matrix(instance.flatten_strict(), instance.num_alternatives)
    net_counts = np.zeros((n,n))
    winners = []
    for a in range(1, n + 1 // 2 + 1):
        cond_winner = True
        for b in range(a + 1, n + 1):
            x = net_compare_a_over_b(rankings, weights, a, b)
            if cond_winner and x < 0:
                cond_winner = False
            net_counts[a - 1][b - 1] = x
            net_counts[b - 1][a - 1] = -1 * x
        #if cond_winner:
            #return [a]
    return net_counts

def condorcet(instance):
    n = instance.num_alternatives
    rankings, weights = orders_to_matrix(instance.flatten_strict(), instance.num_alternatives)
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
            return [a]
    return pairwise_counts
    







#testing zone

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
            net_counts[b - 1][a - 1] = -1 * x
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



a = ((4,3,1,5,2),9)
b = ((2,5,1,3,4),5)
c = ((5,1,4,2,3),2)
d = ((2,3,1,4,5),5)
e = ((3,1,4,2,5),8)
f = ((2,4,3,1,5),6)

'''
a = ((1,2,3,4,5),8)
b = ((2,1,3,4,5),8)
c = ((3,1,4,5,2),9)
d = ((1,4,5,2,3),2)
e = ((4,2,3,5,1),9)
f = ((1,5,4,3,2),2)
'''

arr = [a,b,c,d,e,f]
r,w = orders_to_matrix(arr,5)

#rem = [1,2,3,4,5]
#print(test_IRV(r,w,rem)) #should give 4

print(test_cond_matrix(r, w, 5))
print(test_net_cond_matrix(r, w, 5))


