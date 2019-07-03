from ambiguity_solving import DataSet
import pandas as pd
import numpy as np

# IMPORTS
################################################################################
dataset = DataSet('event_data/track_data_tqscore.csv')
trackSet = dataset.tracks

truth = pd.read_csv('event_data/event000001000-truth.csv')
truth_hits_id = truth["hit_id"]
truth_track_id = truth["particle_id"]

truth_tracks = pd.read_csv('event_data/event000001000-particles.csv')
truth_tracks_nhits = truth_tracks["nhits"]


# DECLARATIONS
################################################################################
# hits is the number of hits per track.
hits = []

# *efficiency* and *fake_rate*:
# *efficiency* is the number of objectively (based on the "truth" csv file) true hits
# reconstructed over the total number of true hits supposed to be in the track.
# *fake_rate* is the number of objectively true hits reconstructed over the total
# number of hits reconstructed. A better name would be 'true rate'?
efficiency = []
fake_rate = []
bad_efficiency = []
bad_fake_rate = []

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



# LOOPS AND ALGORITHMS >:)
################################################################################

for track in trackSet:
    # adds to hits
    hits.append(track.num_hits)

    # adds to *good and bad* hits
    if track in good_tracks:
        good_track_hits.append(track.num_hits)
    else:
        bad_track_hits.append(track.num_hits)

    # efficiency & fake rate calcuations
    # although not strictly necessary, hits are stored in a list rather than simply counted,
    # in preparation for anything in the future that might find a list of truth hits useful.
    track_truth = track.particle_id
    true_hits = []
    fake_hits = []
    for hit in track.hits:
        # hit is indexed at 1 ...
        if truth_track_id[hit - 1] == track_truth:
            true_hits.append(hit)
        else:
            fake_hits.append(hit)
    # if the track is not reconstructable (for example, if the particle was actually detector
    # noise), the track ID, track_truth, is set to 0 by QPR convention.
    # It then doesn't make sense to include this efficiency value--we will disregard it.
    if (track_truth != 0):
        this_efficiency = len(true_hits) / int(truth_tracks[truth_tracks.particle_id == track_truth]["nhits"])
    this_fake_rate = len(true_hits) / track.num_hits
    if track in good_tracks:
        efficiency.append(this_efficiency)
        fake_rate.append(this_fake_rate)
    else:
        bad_efficiency.append(this_efficiency)
        bad_fake_rate.append(this_fake_rate)

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


# PLOTTING
################################################################################
import matplotlib.pyplot as plt

# # Plots the number of hits per track.
# bins = np.arange(0, 20, 1)
# x, bins, p = plt.hist(hits, bins, color='blue', label='Number of Hits')
# plt.legend()
# plt.xlabel('Hits')
# plt.ylabel('Number of tracks')
# plt.title('Number of hits on each track')
# # plt.show()

# # Plots the number of hits per good track.
# bins = np.arange(0, 20, 1)
# x, bins, p = plt.hist(good_track_hits, bins, color='blue', label='Number of Hits')
# plt.legend()
# plt.xlabel('Hits')
# plt.ylabel('Number of *good* tracks')
# plt.title('Number of hits on each *good* track')
# # plt.show()

# # Plots the number of hits per bad track.
# bins = np.arange(0, 20, 1)
# x, bins, p = plt.hist(bad_track_hits, bins, color="red", label='Number of Hits')
# plt.legend()
# plt.xlabel('Hits')
# plt.ylabel('Number of *bad* tracks')
# plt.title('Number of hits on each *bad* track')
# # plt.show()

# # Plots number of hits per track, partitioned into good and bad according to the scheme mentioned in the commments above (trackML completeness).
# bins = np.arange(0, 20, 1)
# x, bins, p = plt.hist(good_track_hits, bins, color='blue', label='*Good* tracks')
# y, bins, p1 = plt.hist(bad_track_hits, bins, color='red', label='*Bad* tracks')
# plt.legend()
# plt.xlabel('Hits')
# plt.ylabel('Number of tracks')
# plt.title('Number of hits on each track, good and bad')
# # plt.show()

# Plots the efficiency of good tracks.
bins = np.linspace(0, 1, 20)
x, bins, p = plt.hist(efficiency, bins, color='blue', label='Track efficiency')
plt.legend()
plt.xlabel('Efficiency')
plt.ylabel('Number of tracks')
plt.title('Efficiency of each good track')
plt.show()

# Plots the fake rate of good tracks.
bins = np.linspace(0, 1, 20)
x, bins, p = plt.hist(fake_rate, bins, color='blue', label='Fake rate')
plt.legend()
plt.xlabel('Fake rate')
plt.ylabel('Number of tracks')
plt.title('Fake rate of each good track')
plt.show()

# Plots the efficiency of bad tracks.
bins = np.linspace(0, 1, 20)
x, bins, p = plt.hist(bad_efficiency, bins, color='red', label='Track efficiency')
plt.legend()
plt.xlabel('Efficiency')
plt.ylabel('Number of tracks')
plt.title('Efficiency of each bad track')
plt.show()

# Plots the fake rate of bad tracks.
bins = np.linspace(0, 1, 20)
x, bins, p = plt.hist(bad_fake_rate, bins, color='red', label='Fake rate')
plt.legend()
plt.xlabel('Fake rate')
plt.ylabel('Number of tracks')
plt.title('Fake rate of each bad track')
plt.show()

# Plots the efficiency of good and bad tracks.
bins = np.linspace(0, 1, 20)
x, bins, p = plt.hist(efficiency, bins, color='blue', label='Good track efficiency')
y, bins, p = plt.hist(bad_efficiency, bins, color='red', label='Bad track efficiency')
plt.legend()
plt.xlabel('Efficiency')
plt.ylabel('Number of tracks')
plt.title('Efficiency of each track')
plt.show()

# Plots the fake rate of good and bad tracks.
bins = np.linspace(0, 1, 20)
x, bins, p = plt.hist(fake_rate, bins, color='blue', label='Good track fake rate')
y, bins, p = plt.hist(bad_fake_rate, bins, color='red', label='Bad track fake rate')
plt.legend()
plt.xlabel('Fake rate')
plt.ylabel('Number of tracks')
plt.title('Fake rate of each track')
plt.show()
