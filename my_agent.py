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
import pickle
from path_utils import path_from_local_root
from collections import defaultdict
from independent_histogram import IndependentHistogram
from single_good_histogram import SingleGoodHistogram

NAME = "???"  # TODO: Please give your agent a NAME


class MyAgent(MyLSVMAgent):
    PRETRAINED_HIST = None
    HIST_FILENAME = "lsvm_hist.pkl"

    def setup(self):
        # TODO: Fill out with anything you want to initialize each auction
        goods = sorted(list(self.get_goods()))
        if not hasattr(self, "price_distribution"):
            self.BUCKET_SIZE = 1
            self.BID_UPPER_BOUND = 200
            self.has_pretrained = False

        if MyAgent.PRETRAINED_HIST is None:
            hist_path = path_from_local_root(MyAgent.HIST_FILENAME)
            if os.path.exists(hist_path):
                with open(hist_path, "rb") as f:
                    MyAgent.PRETRAINED_HIST = pickle.load(f)
                print(f"[MyAgent] Loaded pretrained histogram from {hist_path}")
            else:
                bucket_sizes = [self.BUCKET_SIZE] * len(goods)
                max_bids = [self.BID_UPPER_BOUND] * len(goods)
                MyAgent.PRETRAINED_HIST = IndependentHistogram(goods, bucket_sizes, max_bids)
                print(f"[MyAgent] No pretrained histogram found, using fresh histogram.")

        self.price_distribution = MyAgent.PRETRAINED_HIST
        self.NUM_LOCALBID_ITER = 3
        self.NUM_SAMPLES = 30
        self.ALPHA = 0.2

        self.last_bids = {}

    # scpp
    def _valuation_function(self, bundle: set[str]) -> float:
        return float(self.calc_total_valuation(bundle=bundle))

    def _calculate_marginal_value(self,
                                  goods: set[str],
                                  selected_good: str,
                                  valuation_function,
                                  bids: dict,
                                  prices: dict) -> float:

        won_goods = {
            g for g in goods
            if g != selected_good and bids.get(g, 0.0) >= prices.get(g, 0.0)
        }

        valuation_without = valuation_function(won_goods)
        won_goods_with = set(won_goods)
        won_goods_with.add(selected_good)
        valuation_with = valuation_function(won_goods_with)

        return float(valuation_with - valuation_without)

    def _calculate_expected_marginal_value(self,
                                           goods,
                                           selected_good: str,
                                           valuation_function,
                                           bids: dict,
                                           num_samples: int) -> float:
        if num_samples <= 0:
            return 0.0

        if not isinstance(goods, set):
            goods = set(goods)

        total_mv = 0.0
        for _ in range(num_samples):
            sampled_prices = self.price_distribution.sample()  # dict[str, float]

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

    def _expected_local_bid(self, goods) -> dict:

        if not isinstance(goods, set):
            goods = set(goods)

        v = self._valuation_function

        bold: dict[str, float] = {}
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
                bnew[gk] = float(amv)

            bold = bnew

        return bold

    def _make_valid(self, bids: dict) -> dict:

        bids = self.clip_bids(bids)
        if self.is_valid_bid_bundle(bids):
            self.last_bids = bids
            return bids

        try:
            prev = self.get_previous_bid_map()
        except Exception:
            prev = self.last_bids or {}
        prev = self.clip_bids(prev)

        if prev and self.is_valid_bid_bundle(prev):
            self.last_bids = prev
            return prev

        alloc = set(self.get_tentative_allocation())
        if alloc:
            mb = self.get_min_bids(bundle=alloc)
            mb = self.clip_bids(mb)
            if self.is_valid_bid_bundle(mb):
                self.last_bids = mb
                return mb

        self.last_bids = {}
        return {}

    def national_bidder_strategy(self):
        # TODO: Fill out with your national bidder strategy
        goods = self.get_goods()
        raw_bids = self._expected_local_bid(goods)
        return self._make_valid(raw_bids)

    def regional_bidder_strategy(self):
        # TODO: Fill out with your regional bidder strategy
        goods = self.get_goods_in_proximity()
        raw_bids = self._expected_local_bid(goods)
        return self._make_valid(raw_bids)

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

def build_histogram_from_logs(saved_dir: str, output_filename: str = "lsvm_hist.pkl"):
    saved_path = path_from_local_root(saved_dir)
    if not os.path.isdir(saved_path):
        print(f"[OfflineTrain] Directory not found: {saved_path}")
        return

    files = [f for f in os.listdir(saved_path) if f.endswith(".json.gz")]
    if not files:
        print(f"[OfflineTrain] No .json.gz files found in {saved_path}")
        return
    first_file = os.path.join(saved_path, files[0])
    with gzip.open(first_file, "rt", encoding="UTF-8") as f:
        game_data = json.load(f)
    any_agent_data = next(iter(game_data.values()))
    final_prices = any_agent_data["price_history"][-1]
    goods = sorted(list(final_prices.keys()))

    BUCKET_SIZE = 1
    BID_UPPER_BOUND = 50
    bucket_sizes = [BUCKET_SIZE] * len(goods)
    max_bids = [BID_UPPER_BOUND] * len(goods)

    hist = IndependentHistogram(goods, bucket_sizes, max_bids)
    for fname in files:
        fpath = os.path.join(saved_path, fname)
        with gzip.open(fpath, "rt", encoding="UTF-8") as f:
            game_data = json.load(f)
        any_agent_data = next(iter(game_data.values()))
        final_prices = any_agent_data["price_history"][-1]
        hist.add_record(final_prices)

    out_path = path_from_local_root(output_filename)
    with open(out_path, "wb") as f:
        pickle.dump(hist, f)
    print(f"[OfflineTrain] Saved histogram to {out_path}, from {len(files)} games.")


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
    MODE = "test"

    if MODE == "train":
        build_histogram_from_logs(saved_dir="saved_games", output_filename=MyAgent.HIST_FILENAME)

    elif MODE == "test":
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

    else:
        print(f"Unknown MODE = {MODE}, nothing to do.")
