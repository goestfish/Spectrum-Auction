from agt_server.agents.base_agents.lsvm_agent import MyLSVMAgent
from agt_server.local_games.lsvm_arena import LSVMArena
from agt_server.agents.test_agents.lsvm.min_bidder.my_agent import MinBidAgent
from agt_server.agents.test_agents.lsvm.jump_bidder.jump_bidder import JumpBidder
from agt_server.agents.test_agents.lsvm.truthful_bidder.my_agent import TruthfulBidder
import time
import os
import random
import gzip
import json
from collections import defaultdict
from path_utils import path_from_local_root
from scpp import Scpp


NAME = "......"


class MyAgent2(MyLSVMAgent):

    PRICE_HISTORY = defaultdict(list)
    GAMES_PLAYED = 0

    EPSILON = 0.1

    def setup(self):
        self.round = 0
        self.is_national = self.is_national_bidder()
        self.goods = sorted(self.get_goods())
        self.shape = self.get_shape()
        self.goods_to_idx = self.get_goods_to_index()
        self.idx_to_good = {tuple(rc): g for g, rc in self.goods_to_idx.items()}

        self.single_vals = self.get_valuations()
        self._build_static_price_model()
        self.aggressiveness = 1.0
        self.current_target_bundle = set()

        if self.is_national:
            self.candidate_bundles = self._generate_national_bundles()
        else:
            self.candidate_bundles = self._generate_regional_bundles()

    def _build_static_price_model(self):
        self.static_ref_price = {}

        for g in self.goods:
            hist = MyAgent2.PRICE_HISTORY[g]
            base_v = max(float(self.single_vals.get(g, 0.0)), 0.0)

            if hist:
                h = sorted(hist)
                n = len(h)

                def q(alpha):
                    if n == 1:
                        return h[0]
                    idx = int(alpha * (n - 1))
                    return h[idx]

                q50 = q(0.5)
                q70 = q(0.7)
                q90 = q(0.9)
                ema = 0.7 * h[-1] + 0.3 * q50

                ref = 0.4 * q50 + 0.4 * q70 + 0.2 * ema

                if base_v > 0:
                    cap_mult = 1.6 if self.is_national else 1.4
                    ref = min(ref, cap_mult * base_v)

                self.static_ref_price[g] = float(max(ref, 0.0))
            else:

                self.static_ref_price[g] = 0.8 * base_v

    def _estimate_final_price(self, g, p_now):

        base_v = max(float(self.single_vals.get(g, 0.0)), 0.0)
        ref = float(self.static_ref_price.get(g, base_v))

        price_hist = self.get_price_history_map()
        trend_boost = 0.0
        if len(price_hist) >= 3:
            last_k = price_hist[-3:]
            deltas = []
            for t in range(1, len(last_k)):
                prev_p = last_k[t - 1].get(g, 0.0)
                cur_p = last_k[t].get(g, 0.0)
                dp = cur_p - prev_p
                if dp > 0:
                    deltas.append(dp)
            if deltas:
                avg_step = sum(deltas) / len(deltas)
                trend_boost = min(5.0 * avg_step, 0.3 * ref)

        est = ref + trend_boost

        est = max(est, float(p_now))
        if base_v > 0:
            cap_mult = 1.8 if self.is_national else 1.5
            est = min(est, cap_mult * base_v)

        return float(max(est, 0.0))

    def _neighbors(self, g):
        r, c = self.goods_to_idx[g]
        R, C = self.shape
        res = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                res.append(self.idx_to_good[(nr, nc)])
        return res

    def _bundle_value(self, bundle):
        if not bundle:
            return 0.0
        return float(self.calc_total_valuation(bundle))

    def _bundle_value_without(self, bundle, g):
        if g not in bundle:
            return self._bundle_value(bundle)
        smaller = set(bundle)
        smaller.remove(g)
        return self._bundle_value(smaller)

    def _generate_national_bundles(self):
        vals = self.single_vals
        sorted_goods = sorted(self.goods, key=lambda x: vals[x], reverse=True)
        seeds = sorted_goods[:6]

        bundles = []
        for seed in seeds:
            bundle = {seed}
            current_value = self._bundle_value(bundle)

            while len(bundle) < 10:
                candidates = set()
                for g in bundle:
                    for nb in self._neighbors(g):
                        if nb not in bundle:
                            candidates.add(nb)

                if not candidates:
                    break

                best_nb = None
                best_gain = 0.0
                for nb in candidates:
                    new_bundle = set(bundle)
                    new_bundle.add(nb)
                    new_val = self._bundle_value(new_bundle)
                    gain = new_val - current_value
                    if gain > best_gain:
                        best_gain = gain
                        best_nb = nb

                if best_nb is None:
                    break

                bundle.add(best_nb)
                current_value += best_gain

                if len(bundle) >= 4:
                    bundles.append(set(bundle))

        unique = []
        seen = set()
        for b in bundles:
            f = frozenset(b)
            if f not in seen:
                seen.add(f)
                unique.append(b)
        for g in sorted_goods[:6]:
            unique.append({g})

        return unique

    def _generate_regional_bundles(self):
        vals = self.single_vals
        goods_in_prox = set(self.get_goods_in_proximity())
        if not goods_in_prox:
            goods_in_prox = set(self.goods)

        positive_goods = [g for g in goods_in_prox if vals[g] > 0]
        if not positive_goods:
            positive_goods = list(goods_in_prox)

        positive_goods.sort(key=lambda g: vals[g], reverse=True)
        seeds = positive_goods[:5]

        bundles = []
        for seed in seeds:
            bundle = {seed}
            current_value = self._bundle_value(bundle)

            while len(bundle) < 6:
                candidates = set()
                for g in bundle:
                    for nb in self._neighbors(g):
                        if nb in goods_in_prox and nb not in bundle:
                            candidates.add(nb)

                if not candidates:
                    break

                best_nb = None
                best_gain = 0.0
                for nb in candidates:
                    new_bundle = set(bundle)
                    new_bundle.add(nb)
                    new_val = self._bundle_value(new_bundle)
                    gain = new_val - current_value
                    if gain > best_gain:
                        best_gain = gain
                        best_nb = nb

                if best_nb is None:
                    break

                bundle.add(best_nb)
                current_value += best_gain

                if 3 <= len(bundle) <= 6:
                    bundles.append(set(bundle))

        for g in seeds:
            bundles.append({g})

        unique = []
        seen = set()
        for b in bundles:
            f = frozenset(b)
            if f not in seen:
                seen.add(f)
                unique.append(b)

        return unique

    def current_prices_map(self):
        price_history = self.get_price_history_map()
        if price_history:
            return price_history[-1]
        else:
            return {g: 0.0 for g in self.get_goods()}

    def _update_aggressiveness(self):

        current_prices = self.current_prices_map()
        ratios = []
        for g, p in current_prices.items():
            ref = float(self.static_ref_price.get(g, 0.0))
            if ref > 0:
                ratios.append(p / ref)

        if not ratios:
            self.aggressiveness = 1.0
            return

        avg = sum(ratios) / len(ratios)

        if self.is_national:
            low, high = 0.8, 1.4
        else:
            low, high = 0.7, 1.3

        self.aggressiveness = max(low, min(high, avg))

    def _choose_target_bundle(self):

        current_prices = self.current_prices_map()

        risk_mult = 1.3 if self.is_national_bidder() else 1.15

        scored = []
        for C in self.candidate_bundles:
            if not C:
                continue

            if self.is_national_bidder() and len(C) > 7:
                continue

            vC = self._bundle_value(C)
            if vC <= 0:
                continue

            cost_est = 0.0
            for g in C:
                p_now = current_prices.get(g, 0.0)
                est_final = self._estimate_final_price(g, p_now)
                cost_est += est_final * self.aggressiveness

            base_score = vC - risk_mult * cost_est
            scored.append((base_score, C))

        if not scored:
            return set(), 0.0

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = scored[:min(5, len(scored))]

        seed = self.get_current_round() * 10007 + int(sum(current_prices.values()) * 10)
        rnd = random.Random(seed)

        best_bundle = set()
        best_score = float("-inf")

        for base_score, C in top_k:
            noisy_score = base_score + rnd.uniform(-0.5, 0.5)
            if noisy_score > best_score:
                best_score = noisy_score
                best_bundle = C

        if self.current_target_bundle:
            inter = set(best_bundle) & set(self.current_target_bundle)
            if inter:
                best_bundle = inter

        return best_bundle, best_score

    def _bids_for_target_bundle(self, target_bundle):

        bids = {}
        if not target_bundle:
            return bids

        min_bids = self.get_min_bids()
        current_prices = self.current_prices_map()

        total_v = self._bundle_value(target_bundle)
        if total_v <= 0:
            return {}

        marginals = {}
        for g in target_bundle:
            without = self._bundle_value_without(target_bundle, g)
            marginals[g] = max(total_v - without, 0.0)

        if self.is_national_bidder():
            risk_factor = 0.75
        else:
            risk_factor = 0.8

        early_round = (self.get_current_round() <= 60)

        for g in target_bundle:
            mb = float(min_bids[g])
            cur_p = float(current_prices.get(g, 0.0))
            m = float(marginals[g])

            if m <= 0 or m <= mb:
                continue

            est_final = self._estimate_final_price(g, cur_p)

            per_good_cap = risk_factor * m

            if early_round:
                raw_target = min(est_final, per_good_cap)
                bid_g = max(mb, raw_target)
            else:
                bid_g = mb
            bid_g = min(bid_g, m)

            if bid_g >= mb:
                bids[g] = bid_g

        bids = self.clip_bids(bids)

        est_cost = sum(bids.get(g, 0.0) for g in target_bundle)
        U_est = total_v - est_cost

        if self.is_national_bidder():
            stop_loss = -3.0
        else:
            stop_loss = -1.5

        if U_est < stop_loss:
            return {}

        return bids

    def national_bidder_strategy(self):
        target, score = self._choose_target_bundle()
        self.current_target_bundle = set(target)
        bids = self._bids_for_target_bundle(self.current_target_bundle)
        return bids

    def regional_bidder_strategy(self):
        target, score = self._choose_target_bundle()
        self.current_target_bundle = set(target)
        bids = self._bids_for_target_bundle(self.current_target_bundle)
        return bids

    def get_bids(self):

        self._update_aggressiveness()

        if self.is_national_bidder():
            bids = self.national_bidder_strategy()
        else:
            bids = self.regional_bidder_strategy()

        if self.is_valid_bid_bundle(bids):
            return bids

        safe_bids = {}
        tentative = self.get_tentative_allocation()
        if tentative:
            min_bids = self.get_min_bids(tentative)
            for g, mb in min_bids.items():
                safe_bids[g] = mb

        if not safe_bids and self.get_current_round() == 0:
            vals = self.single_vals
            best_g = max(self.goods, key=lambda g: vals[g])
            mb = self.get_min_bids({best_g})[best_g]
            safe_bids[best_g] = mb

        safe_bids = self.clip_bids(safe_bids)
        if self.is_valid_bid_bundle(safe_bids):
            return safe_bids

        return {}

    def update(self):
        pass

    def teardown(self):

        final_prices = self.current_prices_map()
        for g, p in final_prices.items():
            MyAgent2.PRICE_HISTORY[g].append(float(p))
        MyAgent2.GAMES_PLAYED += 1


################### SUBMISSION #####################
my_agent_submission2 = MyAgent2(NAME)
####################################################


def process_saved_game(filepath): 
    """ 
    Here is some example code to load in a saved game in the format of a json.gz and to work with it
    """
    print(f"Processing: {filepath}")
    
    # NOTE: Data is a dictionary mapping 
    with gzip.open(filepath, 'rt', encoding='UTF-8') as f:
        game_data = json.load(f)
        for agent, agent_data in game_data.items(): 
            if agent_data['valuations'] is not None: 
                # agent is the name of the agent whose data is being processed 
                agent = agent 
                
                # bid_history is the bidding history of the agent as a list of maps from good to bid
                bid_history = agent_data['bid_history']
                
                # price_history is the price history of the agent as a list of maps from good to price
                price_history = agent_data['price_history']
                
                # util_history is the history of the agent's previous utilities 
                util_history = agent_data['util_history']
                
                # util_history is the history of the previous tentative winners of all goods as a list of maps from good to winner
                winner_history = agent_data['winner_history']
                
                # elo is the agent's elo as a string
                elo = agent_data['elo']
                
                # is_national_bidder is a boolean indicating whether or not the agent is a national bidder in this game 
                is_national_bidder = agent_data['is_national_bidder']
                
                # valuations is the valuations the agent recieved for each good as a map from good to valuation
                valuations = agent_data['valuations']
                
                # regional_good is the regional good assigned to the agent 
                # This is None in the case that the bidder is a national bidder 
                regional_good = agent_data['regional_good']
            
            # TODO: If you are planning on learning from previously saved games enter your code below. 
            
            
        
def process_saved_dir(dirpath): 
    """ 
     Here is some example code to load in all saved game in the format of a json.gz in a directory and to work with it
    """
    for filename in os.listdir(dirpath):
        if filename.endswith('.json.gz'):
            filepath = os.path.join(dirpath, filename)
            process_saved_game(filepath)
            

if __name__ == "__main__":
    
    # Heres an example of how to process a singular file 
    # process_saved_game(path_from_local_root("saved_games/2024-04-08_17-36-34.json.gz"))
    # or every file in a directory
    # process_saved_dir(path_from_local_root("saved_games"))
    
    ### DO NOT TOUCH THIS #####
    agent = MyAgent2(NAME)
    arena = LSVMArena(
        num_cycles_per_player = 3,
        timeout=1,
        local_save_path="saved_games",
        players=[ 
            agent,
            # MyAgent("CP - MyAgent"),
            # MyAgent("CP2 - MyAgent"),
            # MyAgent("CP3 - MyAgent"),
            MinBidAgent("CP - MyAgent"), 
            JumpBidder("CP2 - MyAgent"), 
            TruthfulBidder("CP3 - MyAgent"), 
            MinBidAgent("Min Bidder"), 
            JumpBidder("Jump Bidder"), 
            TruthfulBidder("Truthful Bidder"), 
        ]
    )
    
    start = time.time()
    arena.run()
    end = time.time()
    print(f"{end - start} Seconds Elapsed")
