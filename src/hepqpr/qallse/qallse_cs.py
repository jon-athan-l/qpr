import itertools
import time
from typing import Union

from .data_structures import *
from .qallse import Qallse, Config1GeV


class CsConfig(Config1GeV):
    pass


class QallseCs(Qallse):
    config: CsConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_base_config(self):
        return CsConfig()

    def _register_qubo_quadruplet(self, qplet: Quadruplet):
        super()._register_qubo_quadruplet(qplet)

        for e in [qplet.t1, qplet.t2]:
            # for excl couplers of type 2
            e.hits[0].outer_tplets.append(e)
            e.hits[-1].inner_tplets.append(e)

    def to_qubo(self, return_stats=False) -> Union[TQubo, Tuple[TQubo, Tuple[int, int, int]]]:
        """
        Generate the QUBO. Attention: ensure that :py:meth:~`build_model` has been called previously.
        :param return_stats: if set, also return the number of variables and coulpers.
        :return: either the QUBO, or a tuple (QUBO, (n_vars, n_incl_couplers, n_excl_couplers))
        """

        Q = {}
        hits, doublets, triplets = self.qubo_hits, self.qubo_doublets, self.qubo_triplets
        quadruplets = self.quadruplets

        start_time = time.process_time()
        # 1: qbits with their weight (doublets with a common weight)
        for q in triplets:
            q.weight = self._compute_weight(q)
            Q[(str(q), str(q))] = q.weight
        n_vars = len(Q)

        if not hasattr(self, 'se_lookup'):
            self.se_lookup = set((str(q) for q in quadruplets))

        n_excl_couplers_2 = 0
        a = 0
        # 2a: exclusion couplers (no two triplets can share the same doublet)
        for hit_id, hit in hits.items():
            for conflicts in [hit.inner_kept, hit.outer_kept]:
                for (d1, d2) in itertools.combinations(conflicts, 2):
                    for t1 in d1.inner_kept | d1.outer_kept:
                        for t2 in d2.inner_kept | d2.outer_kept:
                            if t1 == t2:
                                self.logger.warning(f'tplet_1 == tplet_2 == {t1}')
                                continue
                            key = (str(t1), str(t2))
                            if key not in Q and tuple(reversed(key)) not in Q:
                                Q[key] = self._compute_conflict_strength(t1, t2)

            for t1 in hit.inner_tplets:
                for t2 in hit.outer_tplets:
                    q1 = Xplet.hit_ids_to_name(t1.hits[-3:] + [t2.hits[1]])
                    q2 = Xplet.hit_ids_to_name([t1.hits[-2]] + t2.hits[:3])
                    if q1 not in self.se_lookup and q2 not in self.se_lookup: # TODO using OR ??
                        key = (str(t1), str(t2))
                        Q[key] = self.config.qubo_conflict_strength
                        n_excl_couplers_2 += 1

        n_excl_couplers = len(Q) - n_vars
        # 2b: inclusion couplers (consecutive doublets with a good triplet)
        for q in quadruplets:
            key = (str(q.t1), str(q.t2))
            Q[key] = q.strength

        n_incl_couplers = len(Q) - (n_vars + n_excl_couplers)
        exec_time = time.process_time() - start_time

        self.logger.info(f'Qubo generated in {exec_time:.2f}s. Size: {len(Q)}. Vars: {n_vars}, '
                         f'excl. couplers: {n_excl_couplers} (2:{n_excl_couplers_2}), incl. couplers: {n_incl_couplers}')
        if return_stats:
            return Q, (n_vars, n_incl_couplers, n_excl_couplers)
        else:
            return Q