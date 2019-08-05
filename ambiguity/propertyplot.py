from ambiguity_solving import DataSet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


class Ground:
    def __init__(self, dataset_path='event_data/track_data_tqscore.csv', truth_path='event_data/event000001000-truth.csv', particles_path='event_data/event000001000-particles.csv'):
        dataset = DataSet(dataset_path)
        truth = pd.read_csv(truth_path)
        particles = pd.read_csv(particles_path)
        self.tracks = dataset.tracks
        self.true_tracks = truth["particle_id"]
        self.true_hits = truth["hit_id"]
        self.track_info = particles
        self.valid_tracks, self.invalid_tracks = self.verify()

    # partitions the tracks into "good" and "bad" tracks, pre-input (a broad cut), based on the trackML score.
    # mostly arbitrary and up to the user--QPR by convention has taken all tracks that are at least 50% complete.
    def verify(self):
        valid_ids = [track.tid for track in self.tracks if track.is_valid]
        valid_tracks = [track for track in self.tracks if track.tid in valid_ids]
        invalid_tracks = [track for track in self.tracks if track.tid not in valid_ids]

        return valid_tracks, invalid_tracks


class Engine:
    def __init__(self):
        self.ground = Ground()


    def find_reconstructed_nhits(self, plot=True):
        def plot_reconstructed_nhits(hits):
            valid_hits = hits[0]
            invalid_hits = hits[1]
            all_hits = valid_hits + invalid_hits

            plt.figure()
            # Plots the number of hits per track.
            plt.subplot(311)
            bins = np.arange(0, 20, 1)
            x, bins, p = plt.hist(all_hits, bins, color='blue', label='Number of Hits')
            plt.legend()
            plt.xlabel('Hits')
            plt.ylabel('Number of tracks')
            plt.title('Number of hits on each track')

            # Plots number of hits per track, partitioned into good and bad according to the scheme mentioned in the commments above (trackML completeness).
            plt.subplot(312)
            bins = np.arange(0, 20, 1)
            x, bins, p = plt.hist(valid_hits, bins, color='blue', label='Valid tracks')
            y, bins, p1 = plt.hist(invalid_hits, bins, color='red', label='*Invalid tracks')
            plt.legend()
            plt.xlabel('Hits')
            plt.ylabel('Number of tracks')
            plt.title('Number of hits on each track, valid and invalid')

            # Plots the number of hits per valid track.
            plt.subplot(325)
            bins = np.arange(0, 20, 1)
            x, bins, p = plt.hist(valid_hits, bins, color='blue', label='Number of Hits')
            plt.legend()
            plt.xlabel('Hits')
            plt.ylabel('Number of valid tracks')
            plt.title('Number of hits on each valid track')

            # Plots the number of hits per invalid track.
            plt.subplot(326)
            bins = np.arange(0, 20, 1)
            x, bins, p = plt.hist(invalid_hits, bins, color="red", label='Number of Hits')
            plt.legend()
            plt.xlabel('Hits')
            plt.ylabel('Number of invalid tracks')
            plt.title('Number of hits on each invalid track')

            plt.show()


        valid_hits = [track.num_hits for track in self.ground.tracks if track in self.ground.valid_tracks]
        invalid_hits = [track.num_hits for track in self.ground.tracks if track in self.ground.invalid_tracks]

        hits = [valid_hits, invalid_hits]

        if (plot):
            plot_reconstructed_nhits(hits)

        return hits

    """recall & purity calcuations

    *purity* is the number of objectively true hits reconstructed over the total
    number of hits reconstructed.
    *recall* is the number of objectively (based on the "truth" csv file) true hits
    reconstructed over the total number of true hits supposed to be in the track.
    """
    def find_purity_recall(plot=True):
        def plot_purity_recall(purities, recalls):
            valid_purities, invalid_recalls = purities[0], purities[1]
            valid_recalls, invalid_recalls = recalls[0], recalls[1]

            plt.figure()
            # Plots the purity of valid tracks.
            plt.subplot(423)
            bins = np.linspace(0, 1)
            x, bins, p = plt.hist(valid_purities, bins, color='blue', label='Purity')
            plt.legend()
            plt.xlabel('Track purity')
            plt.ylabel('Number of tracks')
            plt.title('Purity of each valid track')

            # Plots the recall of valid tracks.
            plt.subplot(427)
            bins = np.linspace(0, 1)
            x, bins, p = plt.hist(valid_recalls, bins, color='blue', label='Recall')
            plt.legend()
            plt.xlabel('Track recall')
            plt.ylabel('Number of tracks')
            plt.title('Recall of each valid track')

            # Plots the purity of invalid tracks.
            plt.subplot(424)
            bins = np.linspace(0, 1)
            x, bins, p = plt.hist(invalid_purities, bins, color='red', label='Purity')
            plt.legend()
            plt.xlabel('Track purity')
            plt.ylabel('Number of tracks')
            plt.title('Purity of each invalid track')

            # Plots the recall of invalid tracks.
            plt.subplot(428)
            bins = np.linspace(0, 1)
            x, bins, p = plt.hist(invalid_recalls, bins, color='red', label='Recall')
            plt.legend()
            plt.xlabel('Track recall')
            plt.ylabel('Number of tracks')
            plt.title('Recall of each invalid track')

            # Plots the purity of valid and invalid tracks.
            plt.subplot(411)
            bins = np.linspace(0, 1)
            x, bins, p = plt.hist(valid_purities, bins, color='blue', label='Valid track purity')
            y, bins, p = plt.hist(invalid_purities, bins, color='red', label='Invalid track purity')
            plt.legend()
            plt.xlabel('Track purity')
            plt.ylabel('Number of tracks')
            plt.title('Purity of each track')

            # Plots the recall of valid and valid tracks.
            plt.subplot(413)
            bins = np.linspace(0, 1)
            x, bins, p = plt.hist(valid_recalls, bins, color='blue', label='Valid track recall')
            y, bins, p = plt.hist(invalid_recalls, bins, color='red', label='Invalid recall')
            plt.legend()
            plt.xlabel('Track recall')
            plt.ylabel('Number of tracks')
            plt.title('Recall of each track')

            plt.show()


        valid_purities = []
        valid_recalls = []
        invalid_purities = []
        invalid_recalls = []

        for track in ground.tracks:
            track_id = track.particle_id
            # although not strictly necessary, hits are stored in a list rather than simply counted,
            # in preparation for anything in the future that might find a list of true/fake hits useful.
            true_hits = [hit for hit in track.hits if ground.true_tracks[hit - 1] == track_id]
            fake_hits = [hit for hit in track.hits if ground.true_tracks[hit - 1] != track_id]
            # if the track is not reconstructable (for example, if the particle was actually detector
            # noise), the track ID, track_id, is set to 0 by QPR convention.
            # It then doesn't make sense to include this recall value--we will disregard it.
            # Although this seems like a TrackML error, depending on how the track ID is determined.
            if (track_id == 0):
                continue
            # clunky pandas syntax... what it means is it gets the reconstructed track's actual number of hits
            purity = len(true_hits) / track.num_hits
            recall = len(true_hits) / int(ground.track_info[ground.track_info.particle_id == track_id]["nhits"])
            if track in ground.valid_tracks:
                valid_purities.append(purity)
                valid_recalls.append(recall)
            elif track in ground.invalid_tracks:
                invalid_purities.append(purity)
                invalid_recalls.append(recall)
        purities = [valid_purities, invalid_purities]
        recalls = [valid_recalls, invalid_recalls]

        if (plot):
            plot_purity_recall(purities, recalls)

        return purities, recalls

    def find_shared_hits(self, plot=True, remember=False, verbose=False):
        # checked is a dictionary that stores all the tracks another track has been compared with already.
        checked = {}
        for track in self.ground.valid_tracks:
            checked[track] = []
            hits = frozenset(track.hits)

            track.shared_tracks = set()
            track.nonshared_tracks = set()
            track.shared_hits = set()
            track.nonshared_hits = set()

            for other_track in self.ground.valid_tracks:
                if (other_track not in checked[track]):
                    checked[track].append(other_track)
                    other_hits = frozenset(other_track.hits)
                    if not hits.isdisjoint(other_hits):
                        track.shared_tracks.add(other_track.tid)
                        track.shared_hits.add(hits & other_hits)
                    else:
                        track.nonshared_tracks.add(other_track.tid)
                        track.nonshared_hits.add(other_track.tid)

        for track in self.ground.invalid_tracks:
            checked[track] = []
            hits = frozenset(track.hits)

            track.shared_tracks = set()
            track.nonshared_tracks = set()
            track.shared_hits = set()
            track.nonshared_hits = set()

            # phi slicing: (abs(track.phi - other_track.phi) <= math.pi / 192)
            for other_track in self.ground.invalid_tracks:
                if (other_track not in checked[track]):
                    checked[track].append(other_track)
                    other_hits = frozenset(other_track.hits)
                    if not hits.isdisjoint(other_hits):
                        track.shared_tracks.add(other_track.tid)
                        track.shared_hits.add(hits & other_hits)
                    else:
                        track.nonshared_tracks.add(other_track.tid)
                        track.nonshared_hits.add(other_track.tid)

        num_shared_tracks = []
        num_shared_hits = []
        good_num_shared_tracks = []
        good_num_shared_hits = []
        bad_num_shared_tracks = []
        bad_num_shared_hits = []

        for track in self.ground.tracks:
            num_shared_tracks.append(len(track.shared_tracks))
            num_shared_hits.append(len(track.shared_hits))
            for i in range(len(track.nonshared_hits)):
                num_shared_tracks.append(0)
                num_shared_hits.append(0)
            if track in self.ground.valid_tracks:
                good_num_shared_tracks.append(len(track.shared_tracks))
                good_num_shared_hits.append(len(track.shared_hits))
            else:
                bad_num_shared_tracks.append(len(track.shared_tracks))
                bad_num_shared_hits.append(len(track.shared_hits))

        plt.figure()

        # Plots the number of tracks that share hits for all tracks.
        plt.subplot(411)
        bins = np.arange(-1, max(num_shared_hits) + 2, 1)
        x, bins, p = plt.hist([good_num_shared_tracks, bad_num_shared_tracks], bins, stacked=True, color=['blue','red'])
        plt.legend()
        plt.xlabel('Number of tracks that share hits')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared tracks on all tracks')

        # Plots the number of hits that are shared for all tracks.
        plt.subplot(412)
        bins = np.arange(-1, max(num_shared_hits) + 2, 1)
        x, bins, p = plt.hist([good_num_shared_hits, bad_num_shared_hits], bins, stacked=True, color=['blue','red'])
        plt.legend()
        plt.xlabel('Number of hits shared with other tracks')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared hits on all tracks')

        # Plots the number of tracks that share hits per good track.
        plt.subplot(425)
        bins = np.arange(-1, max(good_num_shared_tracks) + 2, 1)
        x, bins, p = plt.hist(good_num_shared_tracks, bins, color='blue', label='Shared tracks')
        plt.legend()
        plt.xlabel('Number of tracks that share hits')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared tracks on valid tracks')

        # Plots the number of hits that are shared per good track.
        plt.subplot(427)
        bins = np.arange(-1, max(good_num_shared_hits) + 2, 1)
        x, bins, p = plt.hist(good_num_shared_hits, bins, color='blue', label='Shared hits')
        plt.legend()
        plt.xlabel('Number of hits shared with other tracks')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared hits on valid tracks')

        # Plots the number of tracks that share hits per bad track.
        plt.subplot(426)
        bins = np.arange(-1, max(bad_num_shared_tracks) + 2, 1)
        x, bins, p = plt.hist(bad_num_shared_tracks, bins, color='red')
        plt.legend()
        plt.xlabel('Number of tracks that share hits')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared tracks on bad tracks')

        # Plots the number of hits that are shared per bad track.
        plt.subplot(428)
        bins = np.arange(-1, max(bad_num_shared_hits) + 2, 1)
        x, bins, p = plt.hist(bad_num_shared_hits, bins, color='red')
        plt.legend()
        plt.xlabel('Number of hits shared with other tracks')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared hits on bad tracks')

        plt.show()



    def find_spatials(self, plot=True):

        def find_eta(px, py, pz):
            euclidian = math.sqrt(px**2 + py**2 + pz**2)
            return math.atanh(pz / euclidian)

        def plot_spatials(reconstructed, true):
            reconstructed_pts = reconstructed[0]
            reconstructed_thetas = reconstructed[1]
            reconstructed_etas = reconstructed[2]
            reconstructed_phis = reconstructed[3]

            true_pts = true[0]
            true_thetas = true[1]
            true_etas = true[2]
            true_phis = true[3]

            print(min(reconstructed_thetas), " min rec theta")
            print(min(true_etas), " min true eta")

            plt.figure()
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


        reconstructed_pts = [math.log(track.pt) for track in self.ground.valid_tracks]
        reconstructed_thetas = [track.theta for track in self.ground.valid_tracks if track.theta >= 0] + [math.pi + track.theta for track in self.ground.valid_tracks if track.theta < 0]
        reconstructed_etas = [0 - math.log(abs(math.tan(theta) / 2)) for theta in reconstructed_thetas]
        reconstructed_phis = [track.phi for track in self.ground.valid_tracks]

        reconstructed_pts = [math.log(track.pt) for track in self.ground.valid_tracks]
        reconstructed_thetas = [track.theta for track in self.ground.valid_tracks if track.theta >= 0] + [math.pi + track.theta for track in self.ground.valid_tracks if track.theta < 0]
        reconstructed_etas = [0 - math.log(abs(math.tan(theta) / 2)) for theta in reconstructed_thetas]
        reconstructed_phis = [track.phi for track in self.ground.valid_tracks]

        px = self.ground.track_info["px"]
        py = self.ground.track_info["py"]
        pz = self.ground.track_info["pz"]
        true_pts = [math.log(1000 * math.sqrt(a**2 + b**2), 10) for a, b in zip(px, py)]
        almost_true_thetas = [math.atan(x / z) for x, z in zip(px, pz)]
        true_thetas = [theta for theta in almost_true_thetas if theta >= 0] + [math.pi + theta for theta in almost_true_thetas if theta < 0]
        true_etas = [find_eta(x, y, z) for x, y, z in zip(px, py, pz)]
        true_phis = [math.atan(y / x) for y, x in zip(py, px)]

        reconstructed = [reconstructed_pts, reconstructed_thetas, reconstructed_etas, reconstructed_phis]
        true = [true_pts, true_thetas, true_etas, true_phis]

        if (plot):
            plot_spatials(reconstructed, true)

        return reconstructed, true


""" RUN HERE """
e = Engine()
e.find_shared_hits()
