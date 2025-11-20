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
from my_agent import MyAgent


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

        self.expected_final_price = {}
        for g in self.goods:
            hist = MyAgent2.PRICE_HISTORY[g]
            if hist:
                sorted_hist = sorted(hist)
                q = 0.7
                idx = int(q * (len(sorted_hist) - 1))
                self.expected_final_price[g] = sorted_hist[idx]
            else:
                self.expected_final_price[g] = 0.7 * self.single_vals[g]

        self.aggressiveness = 1.0

        self.current_target_bundle = set()

        if self.is_national:
            self.candidate_bundles = self._generate_national_bundles()
        else:
            self.candidate_bundles = self._generate_regional_bundles()


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
        return self.calc_total_valuation(bundle)

    def _bundle_value_without(self, bundle, g):

        if g not in bundle:
            return self._bundle_value(bundle)
        smaller = set(bundle)
        smaller.remove(g)
        return self._bundle_value(smaller)


    def _generate_national_bundles(self):

        vals = self.single_vals
        sorted_goods = sorted(self.goods, key=lambda g: vals[g], reverse=True)
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

    def _choose_target_bundle(self):

        current_prices = self.current_prices_map()

        risk_mult = 1.4 if self.is_national_bidder() else 1.2

        best_bundle = set()
        best_score = 0.0

        for C in self.candidate_bundles:
            if not C:
                continue

            if self.is_national_bidder() and len(C) > 6:
                continue

            v = self._bundle_value(C)
            if v <= 0:
                continue

            cost_est = 0.0
            for g in C:
                p_now = current_prices.get(g, 0.0)
                exp = self.expected_final_price.get(g, p_now)
                eff_p = max(p_now, exp)
                cost_est += eff_p

            score = v - risk_mult * cost_est
            if score > best_score:
                best_score = score
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
        n = len(target_bundle)
        if n == 0 or total_v <= 0:
            return {}

        risk_factor = 0.7 if self.is_national_bidder() else 0.8
        per_cap = (total_v / n) * risk_factor

        early_round = (self.get_current_round() <= 50)

        for g in target_bundle:
            min_bid = min_bids[g]
            cur_p = current_prices.get(g, 0.0)

            if per_cap <= min_bid:
                continue

            if early_round:

                jump_target = min(min_bid + 2.0, per_cap)
                bid_g = max(min_bid, jump_target)
            else:

                bid_g = min_bid

            bid_g = min(bid_g, per_cap)

            if bid_g >= min_bid:
                bids[g] = bid_g

        bids = self.clip_bids(bids)

        est_cost = sum(bids.get(g, 0.0) for g in target_bundle)
        U_est = total_v - est_cost

        if self.is_national_bidder():
            stop_loss = -6.0
        else:
            stop_loss = -3.0

        if U_est < stop_loss:
            return {}

        return bids



    def national_bidder_strategy(self):
        target, U = self._choose_target_bundle()
        self.current_target_bundle = set(target)
        bids = self._bids_for_target_bundle(self.current_target_bundle)
        return bids

    def regional_bidder_strategy(self):
        target, U = self._choose_target_bundle()
        self.current_target_bundle = set(target)
        bids = self._bids_for_target_bundle(self.current_target_bundle)
        return bids


    def get_bids(self):

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
            MyAgent2.PRICE_HISTORY[g].append(p)
        MyAgent2.GAMES_PLAYED += 1

################### SUBMISSION #####################
my_agent_submission = MyAgent2(NAME)
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
            MyAgent2("CP - MyAgent"),
            Scpp("..."),
            MyAgent2("CP2 - MyAgent"),
            #MyAgent2("CP3 - MyAgent"),
            # MinBidAgent("CP - MyAgent"), 
            # JumpBidder("CP2 - MyAgent"), 
            # TruthfulBidder("CP3 - MyAgent"), 
            MinBidAgent("Min Bidder"), 
            JumpBidder("Jump Bidder"), 
            TruthfulBidder("Truthful Bidder"), 
        ]
    )
    
    start = time.time()
    arena.run()
    end = time.time()
    print(f"{end - start} Seconds Elapsed")