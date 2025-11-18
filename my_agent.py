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
        # TODO: Fill out with anything you want to initialize each auction
        self.epsilon = 0.1
        self.national_aggressiveness = 0.8
        self.regional_aggressiveness = 1.0
        self.max_bid_factor = 1.0
        self.last_bids = {}

    def _estimate_surplus(self, good, valuations, min_bids):

        v = valuations.get(good, 0.0)
        mb = min_bids.get(good, 0.0)
        return max(0.0, v - mb)

    def _choose_bundle_greedy(self, candidate_goods):

        current_alloc = set(self.get_tentative_allocation())
        candidate_goods = set(candidate_goods)

        valuations = self.get_valuations()
        min_bids = self.get_min_bids()

        best_bundle = set(current_alloc)
        best_utility = self.calc_total_utility(bundle=best_bundle)

        scored_goods = []
        for g in (candidate_goods - best_bundle):
            surplus = self._estimate_surplus(g, valuations, min_bids)
            scored_goods.append((surplus, g))

        scored_goods.sort(reverse=True)

        for surplus, g in scored_goods:
            if surplus <= 0:
                continue
            new_bundle = best_bundle | {g}
            new_utility = self.calc_total_utility(bundle=new_bundle)
            if new_utility > best_utility + 1e-9:
                best_bundle = new_bundle
                best_utility = new_utility

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
        # TODO: Fill out with your national bidder strategy
        goods = self.get_goods()
        bundle, valuations, min_bids = self._choose_bundle_greedy(goods)
        proposed_bids = self._bundle_to_bids(
            bundle, valuations, min_bids, self.national_aggressiveness
        )
        return self._make_valid_bids(proposed_bids)

    def regional_bidder_strategy(self):
        # TODO: Fill out with your regional bidder strategy
        candidate_goods = set(self.get_goods_in_proximity())

        bundle, valuations, min_bids = self._choose_bundle_greedy(candidate_goods)
        proposed_bids = self._bundle_to_bids(
            bundle, valuations, min_bids, self.regional_aggressiveness
        )
        return self._make_valid_bids(proposed_bids)

    def get_bids(self):
        if self.is_national_bidder():
            return self.national_bidder_strategy()
        else:
            return self.regional_bidder_strategy()

    def update(self):
        # TODO: Fill out with anything you want to update each round
        try:
            self.last_util = self.get_previous_util()
        except Exception:
            self.last_util = None

    def teardown(self):
        # TODO: Fill out with anything you want to run at the end of each auction
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
