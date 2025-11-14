import random

import random


class SingleGoodHistogram:
    def __init__(self, bucket_size, bid_upper_bound):
        self.bucket_size = bucket_size
        self.bid_upper_bound = bid_upper_bound
        self.buckets = {}
        for b in range(0, bid_upper_bound, bucket_size):
            self.buckets[b] = 0.0
        self.total = 0.0

    def get_bucket(self, price):
        bucket = int(price // self.bucket_size) * self.bucket_size
        if bucket > self.bid_upper_bound:
            bucket = self.bid_upper_bound
        return bucket

    def add_record(self, price):
        """
        Add a price to the histogram.
        Increment the frequency of the bucket that contains the price.
        """
        bucket = self.get_bucket(price)
        # Update the bucket frequencies
        self.buckets[bucket] += 1
        self.total += 1

    def smooth(self, alpha):
        """
        Smooth the histogram using the technique described in the handout.
        """
        factor = (1.0 - alpha)
        for b in self.buckets:
            self.buckets[b] *= factor
        self.total = sum(self.buckets.values())

    def update(self, new_hist, alpha):
        """
        Actually updating the histogram with new information:
        1. Smooth the current histogram.
        2. Add the new histogram to the current histogram.
        """
        self.smooth(alpha)
        for b in self.buckets:
           self.buckets[b] += alpha * new_hist.buckets.get(b, 0.0)
        self.total = sum(self.buckets.values())

    def sample(self):
        """
        Return a random sample from the histogram.
        """
        # We need to handle the edge case when self.total == 0.
        # One solution is returning a uniform sample from 0 to self.bid_upper_bound.
        use_uniform_weights = (self.total <= 0.0)
        total_weight = 0.0
        for v in self.buckets.values():
            if use_uniform_weights:
                total_weight += 1.0
            else:
                total_weight += v

        if total_weight <= 0.0:
            return 0.0

        z = random.random() * total_weight
        cum = 0.0

        for b in sorted(self.buckets.keys()):
            w = 1.0 if use_uniform_weights else self.buckets[b]
            cum += w
            if z <= cum:
                low = b
                high = min(b + self.bucket_size, self.bid_upper_bound)
                if high <= low:
                    return float(low)
                return low + random.random() * (high - low)

        last_b = max(self.buckets.keys())
        low = last_b
        high = min(last_b + self.bucket_size, self.bid_upper_bound)
        if high <= low:
            return float(low)
        return low + random.random() * (high - low)
    def __repr__(self):
        return str(self.buckets)