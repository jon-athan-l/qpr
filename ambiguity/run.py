from ambiguity_solving import DataSet
import numpy as np

d = DataSet('event_data/track_data_tqscore.csv',1)
good_tid = [t.tid for t in d.tracks if t.is_valid]

good_tracks = []
bad_tracks = []

for t in d.tracks:
    if t.tid in good_tid:
        good_tracks.append(t)
    else:
        bad_tracks.append(t)

good_quality = [t.quality for t in good_tracks if t.quality]
bad_quality = [t.quality for t in bad_tracks if t.quality]


#use this to generate plots
import matplotlib.pyplot as plt
bins = np.linspace(min(bad_quality + good_quality), max(bad_quality + good_quality), 10)
x, bins, p = plt.hist(good_quality,bins, color='blue', label='Good Tracks')
y, bins, p1 = plt.hist(bad_quality, bins, color='red', label='Bad Tracks')
plt.legend()
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Scores for Good and Bad Tracks')
plt.show()


good_weights = np.ones_like(good_quality)/float(len(good_quality))
bad_weights = np.ones_like(bad_quality)/float(len(bad_quality))
plt.hist(good_quality, bins, weights = good_weights, color='blue', label='Good Tracks')
plt.hist(bad_quality, bins, weights = bad_weights, color='red', label='Bad Tracks')
plt.legend()
plt.title('Scores for Good and Bad Tracks')
plt.xlabel('Score')
plt.ylabel('Count')
plt.show()


#good_inner = [t.inner_hit for t in good_tracks]
#bad_inner = [t.inner_hit for t in bad_tracks]
#bins = np.linspace(min(bad_inner + good_inner), max(bad_inner + good_inner), 10)
##plt.hist(bad_inner, bins)
##plt.hist(good_inner,bins)
##plt.show()
#
#
def compare(good_tracks, bad_tracks, var_string):
    good = [eval('t.' + var_string) for t in good_tracks]
    bad = [eval('t.' + var_string) for t in bad_tracks]
    bins = np.linspace(min(bad + good), max(bad + good), 10)
    plt.hist(bad, bins)
    plt.hist(good,bins)
    plt.show()

def compute_common_elements(a,b):
    s = 0
    for e in a:
        if e in b:
            s += 1
    return s
#
#
##QUBO portion
#
taken, not_taken = d.solve_qubo()
#
correct_taken = compute_common_elements(taken, good_tid)
incorrect_taken = len(taken) - correct_taken

correct_not_taken = compute_common_elements(not_taken, good_tid)
incorrect_not_taken = len(not_taken) - correct_not_taken

print('Correct Taken ' + str(correct_taken))
print('Incorrect Taken ' + str(incorrect_taken))
print('Correct Not Taken ' + str(correct_not_taken))
print('Incorrect Not Taken ' + str(incorrect_not_taken))

precision = correct_taken/len(taken)
recall = correct_taken/len(good_tid)
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
