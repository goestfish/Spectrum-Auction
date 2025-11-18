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

NAME = "???"  # TODO: Please give your agent a NAME


class MyAgent(MyLSVMAgent):
    def setup(self):
        self.epsilon = 0.1
        self.national_aggressiveness = 1.5
        self.regional_aggressiveness = 1.5
        self.max_bid_factor = 1.0
        self.last_bids = {}

    def _estimate_surplus(self, good, valuations, min_bids):
        v = valuations.get(good, 0.0)
        mb = min_bids.get(good, 0.0)
        return max(0.0, v - mb)

    def _dynamic_aggressiveness(self):
        try:
            prices = self.get_current_prices()
            prev_prices = self.get_previous_price_map()

            increases = [prices[g] - prev_prices.get(g, prices[g]) for g in prices]
            avg_inc = sum(increases) / len(increases)

            if avg_inc < self.epsilon * 2:
                return 2.0
            else:
                return 0.7
        except:
            return 1.5

    def _choose_bundle(self, candidate_goods):

        valuations = self.get_valuations()
        min_bids = self.get_min_bids()
        current_alloc = set(self.get_tentative_allocation())

        surplus_list = [(self._estimate_surplus(g, valuations, min_bids), g) for g in candidate_goods]
        surplus_list = [x for x in surplus_list if x[0] > 0]
        surplus_list.sort(reverse=True)

        core_candidates = [g for _, g in surplus_list[:6]]

        best_bundle = set(current_alloc)
        best_utility = self.calc_total_utility(best_bundle)

        from itertools import combinations
        for r in range(1, len(core_candidates) + 1):
            for comb in combinations(core_candidates, r):
                bundle = current_alloc | set(comb)
                utility = self.calc_total_utility(bundle)
                if utility > best_utility + 1e-9:
                    best_bundle = set(bundle)
                    best_utility = utility

        return best_bundle, valuations, min_bids

    def _bundle_to_bids(self, bundle, valuations, min_bids, aggressiveness):
        bids = {}
        for g in bundle:
            v = float(valuations.get(g, 0.0))
            mb = float(min_bids.get(g, 0.0))

            surplus = max(0.0, v - mb)
            if surplus <= 0:
                continue

            jump = aggressiveness * surplus
            raw_bid = mb + jump

            cap = self.max_bid_factor * max(v, 1.0)

            bid = max(mb + self.epsilon, min(raw_bid, cap))
            bids[g] = bid

        return bids

    def _make_valid_bids(self, proposed_bids):
        if proposed_bids is None:
            proposed_bids = {}

        bids = self.clip_bids(proposed_bids)
        if self.is_valid_bid_bundle(bids):
            self.last_bids = bids
            return bids

        prev_bids = {}
        try:
            prev_bids = self.get_previous_bid_map()
        except Exception:
            prev_bids = self.last_bids or {}

        if prev_bids:
            prev_bids = self.clip_bids(prev_bids)
            if self.is_valid_bid_bundle(prev_bids):
                self.last_bids = prev_bids
                return prev_bids

        current_alloc = set(self.get_tentative_allocation())
        if current_alloc:
            min_on_alloc = self.get_min_bids(bundle=current_alloc)
            min_on_alloc = self.clip_bids(min_on_alloc)
            if self.is_valid_bid_bundle(min_on_alloc):
                self.last_bids = min_on_alloc
                return min_on_alloc

        self.last_bids = {}
        return {}

    def national_bidder_strategy(self):
        goods = self.get_goods()
        bundle, valuations, min_bids = self._choose_bundle_greedy(goods)
        proposed_bids = self._bundle_to_bids(
            bundle, valuations, min_bids, self.national_aggressiveness
        )
        return self._make_valid_bids(proposed_bids)

    def regional_bidder_strategy(self):
        candidate_goods = set(self.get_goods_in_proximity())
        bundle, valuations, min_bids = self._choose_bundle(candidate_goods)
        proposed_bids = self._bundle_to_bids(
            bundle, valuations, min_bids, self.regional_aggressiveness
        )
        return self._make_valid_bids(proposed_bids)

    def get_bids(self):
        if self.is_national_bidder():
            bundle, valuations, min_bids = self._choose_bundle(self.get_goods())
        else:
            bundle, valuations, min_bids = self._choose_bundle(set(self.get_goods_in_proximity()))

        dynamic_aggr = self._dynamic_aggressiveness()

        proposed = self._bundle_to_bids(bundle, valuations, min_bids, dynamic_aggr)
        return self._make_valid_bids(proposed)

    def update(self):
        try:
            self.last_util = self.get_previous_util()
        except Exception:
            self.last_util = None

    def teardown(self):
        pass


    ################### SUBMISSION #####################


my_agent_submission = MyAgent(NAME)


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
    agent = MyAgent(NAME)
    arena = LSVMArena(
        num_cycles_per_player=3,
        timeout=1,
        local_save_path="saved_games",
        players=[
            agent,
            MyAgent("CP - MyAgent"),
            MyAgent("CP2 - MyAgent"),
            MyAgent("CP3 - MyAgent"),
            MinBidAgent("Min Bidder"),
            JumpBidder("Jump Bidder"),
            TruthfulBidder("Truthful Bidder"),
        ]
    )

    start = time.time()
    arena.run()
    end = time.time()
    print(f"{end - start} Seconds Elapsed")
