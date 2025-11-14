import pickle
import os
from agt_server.agents.base_agents.sa_agent import SimultaneousAuctionAgent
from agt_server.agents.test_agents.sa.truth_bidder.my_agent import TruthfulAgent
from independent_histogram import IndependentHistogram
from localbid import expected_local_bid
import argparse
import time
from agt_server.local_games.sa_arena import SAArena


class SCPPAgent(SimultaneousAuctionAgent):
    def setup(self):
        # NOTE: Many internal methods (e.g. self.get_valuations) aren't available during setup.
        # So we delay any setup that requires those until get_action() is called.

        self.mode = 'TRAIN'

        self.simulation_count = 0
        self.NUM_ITERATIONS = 100
        self.NUM_SIMULATIONS_PER_ITERATION = 10
        self.ALPHA = 0.1
        self.NUM_ITERATIONS_LOCALBID = 100
        self.NUM_SAMPLES = 50
        self.BUCKET_SIZE = 5
        self.distribution_file = f"learned_distribution_{self.name}.pkl"

        self.valuation_function = None
        self.learned_distribution = None
        self.curr_distribution = None

    def load_distribution(self):
        """
        Load the learned distribution from disk, if it exists.
        """
        if os.path.exists(self.distribution_file):
            with open(self.distribution_file, "rb") as f:
                self.learned_distribution = pickle.load(f)
            self.curr_distribution = self.create_independent_histogram()
        else:
            self.initialize_distribution()

    def save_distribution(self):
        """
        Save the learned distribution to disk.
        """
        with open(self.distribution_file, "wb") as f:
            pickle.dump(self.learned_distribution, f)

    def create_independent_histogram(self):
        return IndependentHistogram(
            self.goods,
            bucket_sizes=[self.BUCKET_SIZE for _ in range(len(self.goods))],
            max_bids=[self.bid_upper_bound for _ in range(len(self.goods))]
        )

    def initialize_distribution(self):
        """
        Initialize the learned distribution using the goods and default parameters.
        We assume bucket sizes of 5 and max values of 100 per good.
        """
        self.learned_distribution = self.create_independent_histogram()
        self.curr_distribution = self.learned_distribution.copy()

    def get_action(self):
        """
        Compute and return a bid vector by running the LocalBid routine with expected marginal values.
        In RUN mode, load the distribution from disk.
        In TRAIN mode, initialize a new distribution if needed.
        """
        self.valuation_function = self.calculate_valuation

        self.load_distribution()

        return self.get_bids()

    def get_bids(self):
        bids = expected_local_bid(
            goods=self.goods,
            valuation_function=self.valuation_function,
            price_distribution=self.learned_distribution
        )
        return bids

    def update(self):
        other_bids_raw = self.game_report.game_history['opp_bid_history'][-1]

        predicted_prices = {}

        k = 1
        for good in self.goods:
            bids = []
            for opp in other_bids_raw:
                bids.append(opp.get(good, 0.0))
            bids.sort(reverse=True)
            if k == 2:
                predicted_prices[good] = bids[1]
            elif k == 1:
                predicted_prices[good] = bids[0]
            else:
                predicted_prices[good] = 0.0

        if predicted_prices:
            # TODO: insert prices into self.curr_distibution
            # TODO: update simulation_count
            if self.curr_distribution is None:
                self.curr_distribution = self.create_independent_histogram()
            self.curr_distribution.add_record(predicted_prices)
            self.simulation_count += 1

            if self.simulation_count % self.NUM_SIMULATIONS_PER_ITERATION == 0:
                # TODO: Update the learned distribution with the newly gathered data
                # TODO: Reset the current distribution
                # TODO: Save the learned distribution to disk (for use in live auction mode).
                self.learned_distribution.update(self.curr_distribution, self.ALPHA)
                self.curr_distribution = self.create_independent_histogram()
                self.save_distribution()


################### SUBMISSION #####################
agent_submission = SCPPAgent("SCPP Agent")
####################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCPP Agent')
    parser.add_argument('--join_server', action='store_true',
                        help='Connects the agent to the server')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port number (default: 8080)')
    parser.add_argument('--mode', type=str, default='TRAIN',
                        help='Mode: TRAIN or RUN (default: TRAIN)')

    parser.add_argument('--num_rounds', type=int, default=100,
                        help='Number of rounds (default: 100)')

    args = parser.parse_args()
    agent_submission.mode = args.mode
    print(agent_submission.mode)

    if args.join_server:
        agent_submission.connect(ip=args.ip, port=args.port)
    elif args.mode == "TRAIN":
        # TODO: Check for bias
        VALUE_UPPER_BOUND = 100
        VALUE_LOWER_BOUND = 0
        arena = SAArena(
            timeout=1,
            num_goods=3,
            num_rounds=args.num_rounds,
            kth_price=2,
            valuation_type="complement",
            players=[
                agent_submission,
                SCPPAgent("Agent_1"),
                SCPPAgent("Agent_2"),
                SCPPAgent("Agent_3"),
            ],
            value_range=(VALUE_LOWER_BOUND, VALUE_UPPER_BOUND)
        )
        start = time.time()
        arena.run()
        end = time.time()
        print(f"{end - start} Seconds Elapsed")
    else:
        arena = SAArena(
            timeout=1,
            num_goods=3,
            num_rounds=500,
            kth_price=2,
            valuation_type="complement",
            players=[
                agent_submission,
                TruthfulAgent("Agent_1"),
                TruthfulAgent("Agent_2"),
            ]
        )
        start = time.time()
        arena.run()
        end = time.time()
        print(f"{end - start} Seconds Elapsed")
