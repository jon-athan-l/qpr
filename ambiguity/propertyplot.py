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

    def find_shared_hits(self, plot=True, phi_slice=False, remember=False, verbose=False):
        def phi_slicing(phi_slice):
            if not phi_slice:
                return True
            return (abs(track.phi - other_track.phi) <= math.pi / phi_slice)

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
                if (other_track not in checked[track] and phi_slicing(phi_slice)):
                    print('WORKING!')
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

            for other_track in self.ground.invalid_tracks:
                if (other_track not in checked[track] and phi_slicing(phi_slice)):
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
                for i in range(len(track.nonshared_hits)):
                    good_num_shared_tracks.append(0)
                    good_num_shared_hits.append(0)
            else:
                bad_num_shared_tracks.append(len(track.shared_tracks))
                bad_num_shared_hits.append(len(track.shared_hits))
                for i in range(len(track.nonshared_hits)):
                    bad_num_shared_tracks.append(0)
                    bad_num_shared_hits.append(0)

        plt.figure()
        # Plots the number of tracks that share hits for all tracks.
        plt.subplot(121)
        bins = np.arange(-1, max(num_shared_hits) + 2, 1)
        x, bins, p = plt.hist(good_num_shared_tracks, bins, color='darkcyan',  alpha=0.7, label='Valid tracks', edgecolor='darkslategray')
        y, bins, p = plt.hist(bad_num_shared_tracks, bins, color='powderblue',  alpha=0.7, label='Invalid tracks', edgecolor='darkslategray')
        plt.legend()
        plt.xlabel('Number of tracks that share hits')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared tracks on all tracks')

        # Plots the number of tracks that share hits per good track.
        plt.subplot(222)
        bins = np.arange(-1, max(good_num_shared_tracks) + 2, 1)
        x, bins, p = plt.hist(good_num_shared_tracks, bins, color='darkcyan', label='Valid tracks', edgecolor='darkslategray')
        plt.legend()
        plt.xlabel('Number of tracks that share hits')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared tracks on valid tracks')

        # Plots the number of tracks that share hits per bad track.
        plt.subplot(224)
        bins = np.arange(-1, max(bad_num_shared_tracks) + 2, 1)
        x, bins, p = plt.hist(bad_num_shared_tracks, bins, color='powderblue', label='Invalid tracks', edgecolor='darkslategray')
        plt.legend()
        plt.xlabel('Number of tracks that share hits')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared tracks on invalid tracks')
        plt.show()

        # Plots the number of hits that are shared for all tracks.
        plt.subplot(121)
        bins = np.arange(-1, max(num_shared_hits) + 2, 1)
        x, bins, p = plt.hist(good_num_shared_hits, bins, color='darkcyan', alpha=0.7, label='Valid tracks', edgecolor='darkslategray')
        y, bins, p = plt.hist(bad_num_shared_hits, bins, color='powderblue', alpha=0.7, label = 'Invalid tracks', edgecolor='darkslategray')
        plt.legend()
        plt.xlabel('Number of hits shared with other tracks')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared hits on all tracks')

        # Plots the number of hits that are shared per good track.
        plt.subplot(222)
        bins = np.arange(-1, max(good_num_shared_hits) + 2, 1)
        x, bins, p = plt.hist(good_num_shared_hits, bins, color='darkcyan', label='Valid tracks', edgecolor='darkslategray')
        plt.legend()
        plt.xlabel('Number of hits shared with other tracks')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared hits on valid tracks')

        # Plots the number of hits that are shared per bad track.
        plt.subplot(224)
        bins = np.arange(-1, max(bad_num_shared_hits) + 2, 1)
        x, bins, p = plt.hist(bad_num_shared_hits, bins, color='powderblue', label='Invalid tracks', edgecolor='darkslategray')
        plt.legend()
        plt.xlabel('Number of hits shared with other tracks')
        plt.ylabel('Number of tracks')
        plt.title('Number of shared hits on invalid tracks')
        plt.show()



    def find_spatials(self, plot=True):

        def find_eta(px, py, pz):
            euclidian = math.sqrt(px**2 + py**2 + pz**2)
            return math.atanh(pz / euclidian)

        def plot_spatials(valid, invalid, truth):
            valid_reconstructed_pts = valid[0]
            valid_reconstructed_thetas = valid[1]
            valid_reconstructed_etas = valid[2]
            valid_reconstructed_phis = valid[3]

            invalid_reconstructed_pts = invalid[0]
            invalid_reconstructed_thetas = invalid[1]
            invalid_reconstructed_etas = invalid[2]
            invalid_reconstructed_phis = invalid[3]

            true_pts = truth[0]
            true_thetas = truth[1]
            true_etas = truth[2]
            true_phis = truth[3]

            print(min(valid_reconstructed_thetas), " min rec theta")
            print(min(true_etas), " min true eta")

            plt.figure()
            plt.subplot(2, 1, 1)
            bins = np.linspace(min(valid_reconstructed_thetas), max(valid_reconstructed_thetas))
            x, bins, p = plt.hist(valid_reconstructed_thetas, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed thetas', edgecolor='darkslategray')
            y, bins, p = plt.hist(invalid_reconstructed_thetas, bins, color='powderblue', alpha=0.7, label='Invalid reconstructed thetas', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Reconstructed thetas')
            plt.ylabel('Number of tracks')
            plt.title('Reconstructed theta distribution')

            plt.subplot(2, 2, 3)
            bins = np.linspace(min(valid_reconstructed_thetas), max(valid_reconstructed_thetas))
            x, bins, p = plt.hist(valid_reconstructed_thetas, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed thetas', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Valid reconstructed thetas')
            plt.ylabel('Number of tracks')
            plt.title('Valid reconstructed theta distribution')

            plt.subplot(2, 2, 4)
            bins = np.linspace(min(invalid_reconstructed_thetas), max(invalid_reconstructed_thetas))
            x, bins, p = plt.hist(invalid_reconstructed_thetas, bins, color='powderblue', alpha=0.7, label='Invalid reconstructed thetas', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Invalid reconstructed thetas')
            plt.ylabel('Number of tracks')
            plt.title('Invalid reconstructed theta distribution')
            plt.show()

            plt.subplot(1, 1, 1)
            bins = np.linspace(min(true_thetas), max(true_thetas))
            x, bins, p = plt.hist(true_thetas, bins, color='steelblue', alpha=0.7, label='True thetas', edgecolor='darkslategray')
            y, bins, p = plt.hist(valid_reconstructed_thetas, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed thetas', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('True and valid reconstructed thetas')
            plt.ylabel('Number of tracks')
            plt.title('True theta and valid reconstructed theta distribution')
            plt.show()

            plt.subplot(2, 1, 1)
            bins = np.linspace(min(valid_reconstructed_etas), max(valid_reconstructed_etas))
            x, bins, p = plt.hist(valid_reconstructed_etas, bins, color='darkcyan', alpha = 0.7, label='Valid reconstructed etas', edgecolor='darkslategray')
            y, bins, p = plt.hist(invalid_reconstructed_etas, bins, color='powderblue', alpha = 0.7, label='Invalid reconstructed etas', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Reconstructed etas')
            plt.ylabel('Number of tracks')
            plt.title('Reconstructed eta distribution')

            plt.subplot(2, 2, 3)
            bins = np.linspace(min(valid_reconstructed_etas), max(valid_reconstructed_etas))
            x, bins, p = plt.hist(valid_reconstructed_etas, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed etas', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Valid reconstructed etas')
            plt.ylabel('Number of tracks')
            plt.title('Valid reconstructed eta distribution')

            plt.subplot(2, 2, 4)
            bins = np.linspace(min(invalid_reconstructed_etas), max(invalid_reconstructed_etas))
            x, bins, p = plt.hist(invalid_reconstructed_etas, bins, color='powderblue', alpha=0.7, label='Invalid reconstructed etas', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Invalid reconstructed etas')
            plt.ylabel('Number of tracks')
            plt.title('Invalid reconstructed eta distribution')
            plt.show()

            plt.subplot(1, 1, 1)
            bins = np.linspace(min(true_etas), max(true_etas))
            x, bins, p = plt.hist(true_etas, bins, color='steelblue', alpha=0.7, label='True etas', edgecolor='darkslategray')
            y, bins, p = plt.hist(valid_reconstructed_etas, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed etas', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('True, valid, and invalid reconstructed etas')
            plt.ylabel('Number of tracks')
            plt.title('True eta and valid reconstructed eta distribution')
            plt.show()


            plt.subplot(2, 1, 1)
            bins = np.linspace(min(valid_reconstructed_phis), max(valid_reconstructed_phis))
            x, bins, p = plt.hist(valid_reconstructed_phis, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed phis', edgecolor='darkslategray')
            y, bins, p = plt.hist(invalid_reconstructed_phis, bins, color='powderblue', alpha=0.7, label='Invalid reconstructed phis', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Reconstructed phis')
            plt.ylabel('Number of tracks')
            plt.title('Reconstructed phi distribution')

            plt.subplot(2, 2, 3)
            bins = np.linspace(min(valid_reconstructed_phis), max(valid_reconstructed_phis))
            x, bins, p = plt.hist(valid_reconstructed_phis, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed phis', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Valid reconstructed phis')
            plt.ylabel('Number of tracks')
            plt.title('Valid reconstructed phi distribution')

            plt.subplot(2, 2, 4)
            bins = np.linspace(min(invalid_reconstructed_phis), max(invalid_reconstructed_phis))
            x, bins, p = plt.hist(invalid_reconstructed_phis, bins, color='powderblue', alpha=0.7, label='Invalid reconstructed phis', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Invalid reconstructed phis')
            plt.ylabel('Number of tracks')
            plt.title('Invalid reconstructed phi distribution')
            plt.show()

            plt.subplot(1, 1, 1)
            bins = np.linspace(min(true_phis), max(true_phis))
            x, bins, p = plt.hist(true_phis, bins, color='steelblue', alpha=0.7, label='True phis', edgecolor='darkslategray')
            y, bins, p = plt.hist(valid_reconstructed_phis, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed phis', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('True and valid reconstructed phis')
            plt.ylabel('Number of tracks')
            plt.title('True phi and valid reconstructed phi distribution')
            plt.show()


            plt.subplot(2, 1, 1)
            bins = np.linspace(min(valid_reconstructed_pts), max(valid_reconstructed_pts))
            x, bins, p = plt.hist(valid_reconstructed_pts, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed pTs', edgecolor='darkslategray')
            y, bins, p = plt.hist(invalid_reconstructed_pts, bins, color='powderblue', alpha=0.7, label='Invalid reconstructed pTs', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Reconstructed pTs')
            plt.ylabel('Number of tracks')
            plt.title('Reconstructed and log10 relativized pT distribution')

            plt.subplot(2, 2, 3)
            bins = np.linspace(min(valid_reconstructed_pts), max(valid_reconstructed_pts))
            x, bins, p = plt.hist(valid_reconstructed_pts, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed pTs', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Valid reconstructed pTs')
            plt.ylabel('Number of tracks')
            plt.title('Valid reconstructed and log10 relativized pT distribution')

            plt.subplot(2, 2, 4)
            bins = np.linspace(min(invalid_reconstructed_pts), max(invalid_reconstructed_pts))
            x, bins, p = plt.hist(invalid_reconstructed_pts, bins, color='powderblue', alpha=0.7, label='Invalid reconstructed pTs', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('Invalid reconstructed pTs')
            plt.ylabel('Number of tracks')
            plt.title('Invalid reconstructed and log10 relativized pT distribution')
            plt.show()

            plt.subplot(1, 1, 1)
            bins = np.linspace(min(true_pts), max(true_pts))
            x, bins, p = plt.hist(true_pts, bins, color='steelblue', alpha=0.7, label='True pts', edgecolor='darkslategray')
            y, bins, p = plt.hist(valid_reconstructed_pts, bins, color='darkcyan', alpha=0.7, label='Valid reconstructed pTs', edgecolor='darkslategray')
            plt.legend()
            plt.xlabel('True and valid reconstructed pTs')
            plt.ylabel('Number of tracks')
            plt.title('True pt and valid reconstructed and log10 relativized pT distribution')
            plt.show()


        valid_reconstructed_pts = [math.log(track.pt, 10) for track in self.ground.valid_tracks]
        valid_reconstructed_thetas = [track.theta for track in self.ground.valid_tracks if track.theta >= 0] + [math.pi + track.theta for track in self.ground.valid_tracks if track.theta < 0]
        valid_reconstructed_etas = [0 - math.log(abs(math.tan(theta) / 2)) for theta in valid_reconstructed_thetas]
        valid_reconstructed_phis = [track.phi for track in self.ground.valid_tracks]

        invalid_reconstructed_pts = [math.log(track.pt, 10) for track in self.ground.invalid_tracks]
        invalid_reconstructed_thetas = [track.theta for track in self.ground.invalid_tracks if track.theta >= 0] + [math.pi + track.theta for track in self.ground.invalid_tracks if track.theta < 0]
        invalid_reconstructed_etas = [0 - math.log(abs(math.tan(theta) / 2)) for theta in invalid_reconstructed_thetas]
        invalid_reconstructed_phis = [track.phi for track in self.ground.invalid_tracks]

        px = self.ground.track_info["px"]
        py = self.ground.track_info["py"]
        pz = self.ground.track_info["pz"]
        true_pts = [math.log(1000 * math.sqrt(a**2 + b**2), 10) for a, b in zip(px, py)]
        almost_true_thetas = [math.atan(x / z) for x, z in zip(px, pz)]
        true_thetas = [theta for theta in almost_true_thetas if theta >= 0] + [math.pi + theta for theta in almost_true_thetas if theta < 0]
        true_etas = [find_eta(x, y, z) for x, y, z in zip(px, py, pz)]
        true_phis = [math.atan(y / x) for y, x in zip(py, px)]

        valid = [valid_reconstructed_pts, valid_reconstructed_thetas, valid_reconstructed_etas, valid_reconstructed_phis]
        invalid = [invalid_reconstructed_pts, invalid_reconstructed_thetas, invalid_reconstructed_etas, invalid_reconstructed_phis]
        truth = [true_pts, true_thetas, true_etas, true_phis]

        if (plot):
            plot_spatials(valid, invalid, truth)

        return valid, invalid, truth


""" RUN HERE """
e = Engine()
e.find_spatials()
