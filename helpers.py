import numpy as np
import math

def orders_to_matrix(flat_profile, alt_count):
    insts = len(flat_profile)
    mults = []
    rankings = np.zeros((insts, alt_count))
    for i in range(insts):
        rankings[i] = list(flat_profile[i][0])
        mults.append([flat_profile[i][1]])
    return rankings, mults

#SciPy mode function
def mode(a, axis=0):
    scores = np.unique(np.ravel(a))
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

#adapted mode function
def get_modes_by_col(rankings, weights):
    n_positions = len(rankings[0])
    mode_result = []
    count_result = []
    weights = np.array(weights)

    for i in range(n_positions):
        column = np.array([ranking[i] for ranking in rankings])
        unique_vals = np.unique(column)
        counts = np.zeros_like(unique_vals, dtype=int)

        for j, val in enumerate(unique_vals):
            counts[j] = np.sum(weights[column == val])

        #tied candidates
        max_count = np.max(counts)
        tied_candidates = unique_vals[counts == max_count]

        mode_result.append(tied_candidates.tolist())
        count_result.append(int(max_count))

    return mode_result, count_result

#getting candidate count by column
def get_column_counts(rankings, weights, col):
    column = np.array(rankings[:,col])
    unique_vals = np.unique(column)
    counts = np.zeros_like(unique_vals, dtype=int)

    for i, val in enumerate(unique_vals):
        counts[i] = np.sum(np.array(weights)[column == val])  #sum of weights for each unique value
    
    return {candidate: count for candidate, count in zip(unique_vals, counts)}

#get loser of a column
def get_column_loser(counts):
    min_alt = []
    min_seen = math.inf
    for alt in counts.keys():
        if counts[alt] < min_seen:
            min_seen = counts[alt]
            min_alt = [alt]
        elif counts[alt] == min_seen:
            min_alt.append(alt)

    return min_alt

#get winner of a column
def get_column_winner(counts):
    max_alt = []
    max_seen = -math.inf
    for alt in counts.keys():
        if counts[alt] > max_seen:
            max_seen = counts[alt]
            max_alt = [alt]
        elif counts[alt] == max_seen:
            max_alt.append(alt)

    return max_alt

#get candidate count for only included candidates, IRV-style
def restricted_count(rankings, weights, remaining):
    vote_count = len(rankings)
    alt_count = len(weights)
    counts = {}
    for val in remaining:
        counts[val] = 0
    
    for i in range(vote_count):
        trial = 0
        while trial < alt_count:
            alt_here = rankings[i][trial]
            if alt_here in remaining:
                counts[alt_here] += weights[i][0]
                break
            else:
                trial += 1

    return counts

def net_compare_a_over_b(rankings, weights, alt_a, alt_b):
    net_a_over_b = 0
    for row in range(len(rankings)):
        alt_count = len(rankings[row])
        a_rank = alt_count + 1
        b_rank = alt_count + 1
        for i in range(alt_count):
            if rankings[row][i] == alt_a:
                a_rank = i
            elif rankings[row][i] == alt_b:
                b_rank = i
        if a_rank < b_rank:
            net_a_over_b += weights[row][0]
        elif b_rank < a_rank:
            net_a_over_b -= weights[row][0]
    return net_a_over_b

def compare_a_over_b(rankings, weights, alt_a, alt_b):
    if alt_a == alt_b:
        return 0
    a_over_b = 0
    for row in range(len(rankings)):
        alt_count = len(rankings[row])
        a_rank = alt_count + 1
        b_rank = alt_count + 1
        for i in range(alt_count):
            if rankings[row][i] == alt_a:
                a_rank = i
            elif rankings[row][i] == alt_b:
                b_rank = i
        if a_rank < b_rank:
            a_over_b += weights[row][0]
    return a_over_b


#testing functions
a = ((1,2,3,4,5),8)
b = ((2,1,3,4,5),8)
c = ((3,1,4,5,2),9)
d = ((1,4,5,2,3),2)
e = ((4,2,3,5,1),9)
f = ((1,5,4,3,2),2)

arr = [a,b,c,d,e,f]

ranks, weights = orders_to_matrix(arr,5)

#print(get_modes_by_col(ranks,weights))
#print(get_column_counts(ranks,weights,0))
#print(get_column_counts(ranks,weights,1))
print(compare_a_over_b(ranks,weights,2,1))


