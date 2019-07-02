from ambiguity_solving import DataSet
import pandas as pd
import numpy as np

dataset = DataSet('track_data_tqscore.csv', 1)
trackSet = dataset.tracks

truth = pd.read_csv('event000001000-truth.csv')


# # creates a new property for each track--all the
# # other tracks that share a hit with the track
# checked = []
# for track in tracks:
#     hits = tracks.hits
#     checked.append(track)
#     for other_track in tracks:
#         if not checked.contains(other_track):
#             other_hits = other_track.hits
#             if not hits.isdisjoint(other_hits):
#                 track.shared_hit_tracks.append(other_track.tid)
#     track.num_shared_hit_tracks = len(track.shared_hit_tracks)

#hits is the number of hits per track.
hits = []

# *efficiency* and *fake_rate*:
# *efficiency* is the number of objectively (based on the "truth" csv file) true hits
# reconstructed over the total number of true hits supposed to be in the track.
# *fake_rate* is the number of objectively true hits reconstructed over the total
# number of hits reconstructed. A better name would be 'true rate'?
efficiency = []
fake_rate = []

# partitions the tracks into "good" and "bad" tracks, pre-input (a broad cut), based on the trackML score.
# mostly arbitrary and up to the user--QPR by convention has taken all tracks that are at least 50% complete.
good_track_id = [track.tid for track in trackSet if track.is_valid]
good_tracks, good_track_hits = [], []
bad_tracks, bad_track_hits = [], []

for track in trackSet:
    if track.tid in good_track_id:
        good_tracks.append(track)
    else:
        bad_tracks.append(track)


###############################################################
###############################################################


for track in trackSet:
    hit_truth = track.particle_id
    hits.append(track.num_hits)
    if track in good_tracks:
        good_track_hits.append(track.num_hits)
    else:
        bad_track_hits.append(track.num_hits)

    # true_hits = []
    # fake_hits = []
    # for hit in track.hits:
    #     if truth[hit] == hit_truth:
    #         true_hits.append(hit)
    #     else:
    #         fake_hits.append(hit)
    # this_efficiency = true_hits / num_hits
    # efficiency.append(this_effiency)

# Plots the number of hits per track.
import matplotlib.pyplot as plt
bins = np.arange(0, 20, 1)
x, bins, p = plt.hist(hits, bins, color='blue', label='Number of Hits')
plt.legend()
plt.xlabel('Hits')
plt.ylabel('Number of tracks')
plt.title('Number of hits on each track')
plt.show()

# Plots the number of hits per good track.
bins = np.arange(0, 20, 1)
x, bins, p = plt.hist(good_track_hits, bins, color='blue', label='Number of Hits')
plt.legend()
plt.xlabel('Hits')
plt.ylabel('Number of *good* tracks')
plt.title('Number of hits on each *good* track')
plt.show()

# Plots the number of hits per bad track.
bins = np.arange(0, 20, 1)
x, bins, p = plt.hist(bad_track_hits, bins, color="red", label='Number of Hits')
plt.legend()
plt.xlabel('Hits')
plt.ylabel('Number of *bad* tracks')
plt.title('Number of hits on each *bad* track')
plt.show()

# Plots number of hits per track, partitioned into good and bad according to the scheme mentioned in the commments above (trackML completeness).
bins = np.arange(0, 20, 1)
x, bins, p = plt.hist(good_track_hits, bins, color='blue', label='*Good* tracks')
y, bins, p1 = plt.hist(bad_track_hits, bins, color='red', label='*Bad* tracks')
plt.legend()
plt.xlabel('Hits')
plt.ylabel('Number of tracks')
plt.title('Number of hits on each track, good and bad')
plt.show()

# Plots the efficiency.
bins = np.arange(0, 16, 1)
x, bins, p = plt.hist(efficiency, bins, color='blue', label='Number of Hits')
plt.legend()
plt.xlabel('Efficiency')
plt.ylabel('Number of tracks')
plt.title('Efficiency of each track')
plt.show()
