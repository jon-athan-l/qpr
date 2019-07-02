import numpy as np
import pandas as pd
from dwave_qbsolv import QBSolv

class Track:
    def __init__(self, tid : str, hits : [[float]], metrics : [float]):
        self.tid = tid
        self.hits = hits
        self.metrics = metrics
        self.conflicting_tracks = []
        self.pt = 1000 * self.metrics[3] #Convert from GeV/c to MeV/c (since the ATLAS code used MeV)
        self.num_hits = self.metrics[1]
        assert(self.num_hits == len(self.hits))
        self.holes = self.metrics[2]
        #particle id is 0 if it isn't a valid track
        self.particle_id = self.metrics[4]
        #we just define a good track to be one with score >= 0.5
        #this is what Nick has been doing
        self.inner_hit = metrics[5]
        self.d0 = metrics[6]
        self.z0 = metrics[7]
        #new
        self.quality = self.compute_score()
        #self.quality = metrics[8]
        self.is_valid = metrics[0] >= 0.5
        #self.is_valid = self.compute_score() >= 0.5

    def compute_score(self):
        prob = 1
        prob = prob * np.log10(self.pt) - 1 #The ATLAS code gave p_t = 100 MeV/c a score of 1
        if prob < 0:
            #print(self.pt)
            prob = 0
        #assert(prob > 0)
        prob = prob/(self.holes+1)
        #prob = prob*(self.num_hits + 1)
        prob = prob*(self.num_hits + 1)**2
        prob = prob*(-np.log10(self.d0) + 3)
        prob = prob*(1+8*self.inner_hit)**2
        #prob = prob*(1+self.inner_hit)
        if prob < 0:
            print(prob)
            exit()
        return prob


        #This just follows from the ambiguity solving score in the ATLAS codebase

    def __str__(self):
        return "Track " + str(self.tid)

class DataSet:
    def __init__(self, track_path : str, conflict_value):
        #load data from tracks
        self.tracks = self.load_data(track_path)
        self.conflict_value = conflict_value
        self.find_track_conflicts()
        self.Q = self.generate_qubo()

    def load_data(self, track_path):
        #this determines the minimum number of hits we will allow tracks to have
        hit_cut = 5
        #not sure what will go in here yet
        data = pd.read_csv(track_path)
        ids = list(data['Unnamed: 0'])
        scores = list(data['score'])
        hits = list(data['hits'])
        holes = list(data['holes'])
        pt = list(data['pt'])
        phi = list(data['phi'])
        theta = list(data['theta'])
        d0 = list(data['d0'])
        z0 = list(data['z0'])
        pid = list(data['particle ID'])
        tq = list(data['tq_score'])
        inner_hits = list(data['inner hit'])
        hit_list = []
        #what we did when it was a list of hit ids
        #for hit in data['hit ID list']:
        #    hit_list.append(eval(hit))
        #for the hit id dict thing, do this:
        for track in data['hit_dict']:
            t_hit = []
            for hit in eval(track).keys():
                t_hit.append(eval(hit))
            hit_list.append(t_hit)


        tracks = []
        for i in range(len(scores)):
            if len(hit_list[i]) >= hit_cut:
                t = Track(ids[i], hit_list[i], [scores[i], hits[i], holes[i], pt[i],pid[i], inner_hits[i], d0[i], z0[i], tq[i]])
                tracks.append(t)
        return tracks

    def generate_qubo(self):
        Q = {}
        maxval = max([t.quality for t in self.tracks])
        for track in self.tracks:
            Q[(track.tid, track.tid)] = -1*track.quality/maxval
            for other_track in track.conflicting_tracks:
                Q[(track.tid, other_track.tid)] = self.conflict_value
        return Q

    def find_track_conflicts(self):
        #here, we just assume that each track is a list of hits
        for (i, t) in enumerate(self.tracks):
            for j in range(i+1, len(self.tracks)):
                #just compare all of the hits in the two tracks:
                for hit1 in t.hits:
                    for hit2 in self.tracks[j].hits:
                        if hit1 ==  hit2:
                            self.tracks[i].conflicting_tracks.append(self.tracks[j])
                            self.tracks[j].conflicting_tracks.append(self.tracks[i])
                            break

    def make_cuts(self, hit_cut = 5, hole_cut = 3):
        self.tracks = [t for t in self.tracks if t.num_hits >= hit_cut]

        self.tracks = [t for t in self.tracks if t.holes <= hole_cut]


    def hit_equal(self,hit1, hit2):
        #both are arrays of floats
        #define a threshold for floating point equality
        e = 1e-3
        distance = np.sqrt(sum([(a-b)**2 for (a,b) in zip(hit1,hit2)]))
        if distance < e:
            return True
        else:
            return False
    def solve_qubo(self):
        response = QBSolv().sample_qubo(self.Q)
        solution = next(response.samples())
        taken = [a for a in solution if solution[a] == 1]
        not_taken = [a for a in solution if solution[a] != 1]
        #taken and not taken have the ids of tracks the qubo identified as good
        #do something with the chosen tracks
        return taken,not_taken
