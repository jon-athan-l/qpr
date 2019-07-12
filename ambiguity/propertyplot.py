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
    track.true_hits = []
    track.fake_hits = []
    for hit in track.hits:
        # hit is indexed at 1 ...
        if truth_track_id[hit - 1] == track_truth:
            track.true_hits.append(hit)
        else:
            track.fake_hits.append(hit)
    # if the track is not reconstructable (for example, if the particle was actually detector
    # noise), the track ID, track_truth, is set to 0 by QPR convention.
    # It then doesn't make sense to include this efficiency value--we will disregard it.
    if (track_truth != 0):
        this_efficiency = len(track.true_hits) / int(truth_tracks[truth_tracks.particle_id == track_truth]["nhits"])
    this_fake_rate = len(true_hits) / track.num_hits
    if track in good_tracks:
        efficiency.append(this_efficiency)
        fake_rate.append(this_fake_rate)
    else:
        bad_efficiency.append(this_efficiency)
        bad_fake_rate.append(this_fake_rate)

# creates a new property for each track--all the
# other tracks that share a hit with the track
checked = {}
for track in trackSet:
    checked[track] = []
    track.shared_hit_tracks = set()
    track.shared_hit_list = set()

for track in trackSet:
    hits = frozenset(track.hits)
    for other_track in good_tracks:
        if track not in checked[other_track]:
            print(track.tid)
            print(other_track.tid)
            checked[other_track].append(track)
            other_hits = frozenset(other_track.hits)
            if not hits.isdisjoint(other_hits):
                track.shared_hit_tracks.add(other_track.tid)
                track.shared_hit_list.add(hits & other_hits)
    print()

num_shared_track_list = []
num_shared_hit_list = []
good_num_shared_track_list = []
good_num_shared_hit_list = []
bad_num_shared_track_list = []
bad_num_shared_hit_list = []

for track in trackSet:
    num_shared_track_list.append(len(track.shared_hit_tracks))
    num_shared_hit_list.append(len(track.shared_hit_list))
    if track in good_tracks:
        good_num_shared_track_list.append(len(track.shared_hit_tracks))
        good_num_shared_hit_list.append(len(track.shared_hit_list))
    else:
        bad_num_shared_track_list.append(len(track.shared_hit_tracks))
        bad_num_shared_hit_list.append(len(track.shared_hit_list))



import math
reconstructed_pts = []
reconstructed_thetas = []
reconstructed_etas = []
reconstructed_phis = []
print(truth_track_id)
for track in good_tracks:
    print(track.particle_id)
    if track.tid in truth_track_id:
        reconstructed_pts.append(track.pt)
        reconstructed_thetas.append(track.theta)
        eta = 0 - math.log(abs(math.tan(track.theta) / 2))
        reconstructed_etas.append(eta)
        reconstructed_phis.append(track.phi)
        print(track.pt, track.theta, eta, track.phi)

px = truth_tracks["px"]
py = truth_tracks["py"]
pz = truth_tracks["pz"]
true_pts = [math.sqrt(a**2 + b**2) for a, b in zip(px, py)]
true_thetas = [math.atan(y / z) / 2 for y, z in zip(py, pz)]
true_etas = [0 - (math.log(abs(math.tan(math.atan(y / z) / 2)))) for y, z in zip(py, pz)]
true_phis = [math.atan(y / x) for y, x in zip(py, px)]

# PLOTTING
################################################################################
import matplotlib.pyplot as plt

# Plots the number of hits per track.
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

# Plots the number of tracks that share hits for all tracks.
plt.figure()
plt.subplot(321)
bins = np.arange(0, max(good_num_shared_track_list) + 2, 1)
x, bins, p = plt.hist(bad_num_shared_track_list, bins, color='blue')
plt.legend()
plt.xlabel('Number of tracks that share hits')
plt.ylabel('Number of tracks')
plt.title('Number of shared tracks on all tracks')

# Plots the number of hits that are shared for all tracks.
plt.subplot(322)
bins = np.arange(0, max(num_shared_hit_list) + 2, 1)
x, bins, p = plt.hist(good_num_shared_hit_list, bins, color='blue')
y, bins, p = plt.hist(bad_num_shared_hit_list, bins, color='red', label='Number of shared hits')
plt.legend()
plt.xlabel('Number of hits shared with other tracks')
plt.ylabel('Number of tracks')
plt.title('Number of shared hits on all tracks')

# Plots the number of tracks that share hits per good track.
plt.subplot(323)
bins = np.arange(0, max(good_num_shared_track_list) + 2, 1)
x, bins, p = plt.hist(good_num_shared_track_list, bins, color='blue')
plt.legend()
plt.xlabel('Number of tracks that share hits')
plt.ylabel('Number of tracks')
plt.title('Number of shared tracks on good tracks')

# Plots the number of hits that are shared per good track.
plt.subplot(324)
bins = np.arange(0, max(good_num_shared_hit_list) + 2, 1)
x, bins, p = plt.hist(good_num_shared_hit_list, bins, color='blue')
plt.legend()
plt.xlabel('Number of hits shared with other tracks')
plt.ylabel('Number of tracks')
plt.title('Number of shared hits on good tracks')

# Plots the number of tracks that share hits per bad track.
plt.subplot(325)
bins = np.arange(0, max(bad_num_shared_track_list) + 2, 1)
x, bins, p = plt.hist(bad_num_shared_track_list, bins, color='red')
plt.legend()
plt.xlabel('Number of tracks that share hits')
plt.ylabel('Number of tracks')
plt.title('Number of shared tracks on bad tracks')

# Plots the number of hits that are shared per bad track.
plt.subplot(326)
bins = np.arange(0, max(bad_num_shared_hit_list) + 2, 1)
x, bins, p = plt.hist(bad_num_shared_hit_list, bins, color='red')
plt.legend()
plt.xlabel('Number of hits shared with other tracks')
plt.ylabel('Number of tracks')
plt.title('Number of shared hits on bad tracks')

plt.show()


bins = np.linspace(min(reconstructed_thetas) - 2, max(reconstructed_thetas) + 2)
x, bins, p = plt.hist(reconstructed_thetas, bins, color='blue', label='Reconstructed thetas')
plt.legend()
plt.xlabel('Reconstructed thetas')
plt.ylabel('Number of tracks')
plt.title('Reconstructed theta distribution')
plt.show()

bins = np.linspace(min(reconstructed_etas) - 2, max(reconstructed_etas) + 2)
x, bins, p = plt.hist(reconstructed_etas, bins, color='blue', label='Reconstructed etas')
plt.legend()
plt.xlabel('Reconstructed etas')
plt.ylabel('Number of tracks')
plt.title('Reconstructed eta distribution')
plt.show()

bins = np.linspace(min(reconstructed_phis) - 2, max(reconstructed_phis) + 2)
x, bins, p = plt.hist(reconstructed_phis, bins, color='blue', label='Reconstructed phis')
plt.legend()
plt.xlabel('Reconstructed phis')
plt.ylabel('Number of tracks')
plt.title('Reconstructed phi distribution')
plt.show()

bins = np.linspace(min(reconstructed_pts) - 2, max(reconstructed_pts) + 2)
x, bins, p = plt.hist(reconstructed_pts, bins, color='blue', label='Reconstructed Pts')
plt.legend()
plt.xlabel('Reconstructed Pts')
plt.ylabel('Number of tracks')
plt.title('Reconstructed Pt distribution')
plt.show()

bins = np.linspace(min(true_thetas) - 2, max(true_thetas) + 2)
x, bins, p = plt.hist(true_thetas, bins, color='blue', label='True thetas')
plt.legend()
plt.xlabel('True thetas')
plt.ylabel('Number of tracks')
plt.title('True theta distribution')
plt.show()

bins = np.linspace(min(true_etas) - 2, max(true_etas) + 2)
x, bins, p = plt.hist(true_etas, bins, color='blue', label='True etas')
plt.legend()
plt.xlabel('True etas')
plt.ylabel('Number of tracks')
plt.title('True eta distribution')
plt.show()

bins = np.linspace(min(true_phis) - 2, max(true_phis) + 2)
x, bins, p = plt.hist(true_phis, bins, color='blue', label='True phis')
plt.legend()
plt.xlabel('true phis')
plt.ylabel('Number of tracks')
plt.title('true phi distribution')
plt.show()

bins = np.linspace(min(true_pts) - 2, max(true_pts) + 2)
x, bins, p = plt.hist(true_pts, bins, color='blue', label='True Pts')
plt.legend()
plt.xlabel('True Pts')
plt.ylabel('Number of tracks')
plt.title('True Pt distribution')
plt.show()

bins = np.linspace(min(true_thetas) - 2, max(true_thetas) + 2)
x, bins, p = plt.hist(true_thetas, bins, color='blue', label='True thetas')
plt.legend()
plt.xlabel('True thetas')
plt.ylabel('Number of tracks')
plt.title('True theta distribution')
plt.show()

bins = np.linspace(min(true_etas) - 2, max(true_etas) + 2)
x, bins, p = plt.hist(true_etas, bins, color='blue', label='True etas')
plt.legend()
plt.xlabel('True etas')
plt.ylabel('Number of tracks')
plt.title('True eta distribution')
plt.show()

bins = np.linspace(min(true_phis) - 2, max(true_phis) + 2)
x, bins, p = plt.hist(true_phis, bins, color='blue', label='True phis')
plt.legend()
plt.xlabel('true phis')
plt.ylabel('Number of tracks')
plt.title('true phi distribution')
plt.show()

bins = np.linspace(min(true_pts) - 2, max(true_pts) + 2)
y, bins, p = plt.hist(true_pts, bins, color='green', label='Reconstructed Pts')
x, bins, p = plt.hist(true_pts, bins, color='blue', label='True Pts')
plt.legend()
plt.xlabel('True Pts')
plt.ylabel('Number of tracks')
plt.title('Reconstructed/True Pt distribution')
plt.show()

# Plots the efficiency v Pt
bins = np.arange(0, 20, 1)
t, bins, a = plt.hist(true_pts, bins, color='red')
r, bins, a = plt.hist(reconstructed_pts, bins, color='blue')
print(r, t)
plt.show()
x, bins, p = plt.hist(np.divide(r, t), color='steelblue', label='Pt')
plt.legend()
plt.xlabel('Number of tracks')
plt.ylabel('Pt efficiency')
plt.title('Efficiency v Pt')
plt.show()

# Plots the efficiency v eta
bins = np.linspace(0, 1, 20)
t, bins, a = plt.hist(true_etas, bins, color='red')
r, bins, a = plt.hist(reconstructed_etas, bins, color='blue')
print(r, t)
plt.show()
x, bins, p = plt.hist(np.divide(r, t), color='steelblue', label='Eta')
plt.legend()
plt.xlabel('Number of tracks')
plt.ylabel('Eta efficiency')
plt.title('Efficiency v eta')
plt.show()

# Plots the efficiency v phi
bins = np.linspace(0, 1, 20)
t, bins, p = plt.hist(true_phis, bins, color='red')
r, bins, p = plt.hist(reconstructed_phis, bins, color='blue')
print(r, t)
plt.show()
x, bins, p = plt.hist(np.divide(r, t), color='steelblue', label='Phi')
plt.legend()
plt.xlabel('Number of tracks')
plt.ylabel('Phi efficiency')
plt.title('Efficiency v phi')
plt.show()
