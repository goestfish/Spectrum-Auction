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
from path_utils import path_from_local_root

from independent_histogram import IndependentHistogram


NAME = "..."


class Scpp(MyLSVMAgent):

    PRICE_HIST = None

    def setup(self):

        goods = sorted(list(self.get_goods()))
        self.BUCKET_SIZE = 1
        self.BID_UPPER_BOUND = 30

        if Scpp.PRICE_HIST is None:
            bucket_sizes = [self.BUCKET_SIZE] * len(goods)
            max_bids = [self.BID_UPPER_BOUND] * len(goods)
            Scpp.PRICE_HIST = IndependentHistogram(goods, bucket_sizes, max_bids)

        self.price_distribution = Scpp.PRICE_HIST

        self.NUM_LOCALBID_ITER = 1
        self.NUM_SAMPLES = 5

        self.LOCALBID_RISK_FACTOR = 0.5
        if self.is_national_bidder():
            self.TOTAL_BID_FRACTION = 0.2
        else:
            self.TOTAL_BID_FRACTION = 0.3 

        self.last_bids = {}
        self.last_util = None

    def _valuation_function(self, bundle):

        return float(self.calc_total_valuation(bundle=bundle))

    def _calculate_marginal_value(self, goods, selected_good, valuation_function, bids, prices):

        if not isinstance(goods, set):
            goods = set(goods)

        won_goods = {
            g for g in goods
            if g != selected_good and bids.get(g, 0.0) >= prices.get(g, 0.0)
        }

        valuation_without = valuation_function(won_goods)
        price_without = sum(prices.get(g, 0.0) for g in won_goods)
        util_without = valuation_without - price_without

        won_goods_with = set(won_goods)
        won_goods_with.add(selected_good)
        valuation_with = valuation_function(won_goods_with)
        price_with = price_without + prices.get(selected_good, 0.0)
        util_with = valuation_with - price_with

        marginal_utility = util_with - util_without
        return float(marginal_utility)

    def _calculate_expected_marginal_value(self,
                                           goods,
                                           selected_good,
                                           valuation_function,
                                           bids,
                                           num_samples):

        if num_samples <= 0:
            return 0.0

        if not isinstance(goods, set):
            goods = set(goods)

        total_mv = 0.0
        for _ in range(num_samples):
            sampled_prices = self.price_distribution.sample()

            mv = self._calculate_marginal_value(
                goods=goods,
                selected_good=selected_good,
                valuation_function=valuation_function,
                bids=bids,
                prices=sampled_prices,
            )
            total_mv += mv

        avg_mv = total_mv / float(num_samples)
        return float(avg_mv)

    def _hist_total_samples(self):

        total = 0.0
        for hist in self.price_distribution.histograms.values():
            total += hist.total
        return total

    def _per_good_cap(self):
        return 27.0 if self.is_national_bidder() else 21.0

    def _safe_truthful_bids(self, goods):
        if not isinstance(goods, set):
            goods = set(goods)

        valuations = self.get_valuations()
        min_bids = self.get_min_bids()

        ranked_goods = sorted(
            [g for g in goods if g in valuations],
            key=lambda g: valuations[g],
            reverse=True,
        )

        cap = self._per_good_cap()

        if self.is_national_bidder():
            MAX_GOODS = 6
            BID_FACTOR = 0.6
        else:
            MAX_GOODS = 4
            BID_FACTOR = 0.6

        chosen = set(ranked_goods[:MAX_GOODS])

        bids = {}
        for g in chosen:
            v = float(valuations[g])
            mb = float(min_bids[g])

            if v <= 0 or v <= mb:
                continue

            bid = BID_FACTOR * v
            bid = max(bid, mb)
            bid = min(bid, cap)

            if bid > mb:
                bids[g] = bid

        return bids

    def _expected_local_bid(self, goods):

        if not isinstance(goods, set):
            goods = set(goods)

        v = self._valuation_function

        bold = {}
        for g in goods:
            bold[g] = float(v({g}))

        for _ in range(self.NUM_LOCALBID_ITER):
            bnew = bold.copy()
            for gk in goods:
                amv = self._calculate_expected_marginal_value(
                    goods=goods,
                    selected_good=gk,
                    valuation_function=v,
                    bids=bold,
                    num_samples=self.NUM_SAMPLES,
                )
                single_v = v({gk})

                cap = self._per_good_cap()
                amv = amv - 3.0
                if amv < 0.0:
                    amv = 0.0

                amv = max(0.0, min(amv, single_v))
                amv *= self.LOCALBID_RISK_FACTOR
                amv = min(amv, cap)

                bnew[gk] = float(amv)

            bold = bnew

        return bold

    def _cap_total_bids(self, bids, goods):

        if not bids:
            return bids

        valuations = self.get_valuations()
        total_val = 0.0
        for g in goods:
            if g in valuations:
                total_val += float(valuations[g])

        total_bid = sum(float(b) for b in bids.values())

        if total_val <= 0 or total_bid <= 0:
            return bids

        max_total_bid = self.TOTAL_BID_FRACTION * total_val
        if total_bid > max_total_bid:
            scale = max_total_bid / total_bid
            return {g: float(b * scale) for g, b in bids.items()}

        return bids


    def _make_valid(self, bids):

        cap = self._per_good_cap()
        current_min_bids = self.get_min_bids()

        bids = self.clip_bids(bids)
        cleaned_bids = {}
        for g, b in bids.items():
            b_cap = float(min(b, cap))
            mb_now = float(current_min_bids.get(g, 0.0))
            if b_cap >= mb_now:
                cleaned_bids[g] = b_cap
        bids = cleaned_bids

        if bids and self.is_valid_bid_bundle(bids):
            self.last_bids = bids
            return bids
        try:
            prev = self.get_previous_bid_map()
        except Exception:
            prev = self.last_bids or {}
        prev = self.clip_bids(prev)

        cleaned_prev = {}
        for g, b in prev.items():
            b_cap = float(min(b, cap))
            mb_now = float(current_min_bids.get(g, 0.0))
            if b_cap >= mb_now:
                cleaned_prev[g] = b_cap
        prev = cleaned_prev

        if prev and self.is_valid_bid_bundle(prev):
            self.last_bids = prev
            return prev

        alloc = set(self.get_tentative_allocation())
        if alloc:
            mb = self.get_min_bids(bundle=alloc)
            mb = self.clip_bids(mb)

            cleaned_mb = {}
            for g, b in mb.items():
                b_cap = float(min(b, cap))
                mb_now = float(current_min_bids.get(g, 0.0))
                if b_cap >= mb_now:
                    cleaned_mb[g] = b_cap

            if cleaned_mb and self.is_valid_bid_bundle(cleaned_mb):
                self.last_bids = cleaned_mb
                return cleaned_mb

        self.last_bids = {}
        return {}

    def _use_localbid(self):
        return self._hist_total_samples() >= 300 

    def national_bidder_strategy(self):
        goods = self.get_goods()

        if not self._use_localbid():
            raw_bids = self._safe_truthful_bids(goods)
        else:
            raw_bids = self._expected_local_bid(goods)

        raw_bids = self._cap_total_bids(raw_bids, goods)
        return self._make_valid(raw_bids)

    def regional_bidder_strategy(self):
        goods = self.get_goods_in_proximity()

        if not self._use_localbid():
            raw_bids = self._safe_truthful_bids(goods)
        else:
            raw_bids = self._expected_local_bid(goods)

        raw_bids = self._cap_total_bids(raw_bids, goods)
        return self._make_valid(raw_bids)

    def get_bids(self):
        if self.is_national_bidder():
            return self.national_bidder_strategy()
        else:
            return self.regional_bidder_strategy()

    def update(self):

        try:
            self.last_util = self.get_previous_util()
        except Exception:
            self.last_util = None

    def teardown(self):
        try:
            final_prices = self.get_price_history_map()[-1]
            self.price_distribution.add_record(final_prices)
        except Exception:
            pass


################### SUBMISSION #####################
my_agent_submission = Scpp(NAME)
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
    agent = Scpp(NAME)
    arena = LSVMArena(
        num_cycles_per_player = 3,
        timeout=1,
        local_save_path="saved_games",
        players=[
            agent,
            Scpp("CP - MyAgent"),
            Scpp("CP2 - MyAgent"),
            Scpp("CP3 - MyAgent"),
            # MyAgent("CP4 - MyAgent"),
            # MyAgent("CP5 - MyAgent"),
            # MyAgent("CP6 - MyAgent"),
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