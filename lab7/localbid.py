from marginal_value import calculate_expected_marginal_value
from independent_histogram import IndependentHistogram


def expected_local_bid(goods, valuation_function, price_distribution, num_iterations=100, num_samples=50):
    """
    Iteratively computes a bid vector by updating bids to be the expected marginal value for each good.
    """
    bold: dict[str, float] = {}
    for g in goods:
        bold[g] = float(valuation_function({g}))

    for t in range(num_iterations):
        bnew = bold.copy()

        for gk in goods:
            amv = calculate_expected_marginal_value(
                goods=set(goods) if not isinstance(goods, set) else goods,
                selected_good=gk,
                valuation_function=valuation_function,
                bids=bnew,
                price_distribution=price_distribution,
                num_samples=num_samples
            )
            bnew[gk] = float(amv)

        bold = bnew

    return bold


if __name__ == "__main__":
    def valuation(bundle): 
        if len(bundle) == 1: 
            return 10 
        elif len(bundle) == 2:
            return 80 
        elif len(bundle) == 3: 
            return 50 
        else: 
            return 0
    
    print(expected_local_bid(
        goods=["a", "b", "c"],
        valuation_function=valuation,
        price_distribution=IndependentHistogram(["a", "b", "c"], 
                                                [5, 5, 5], 
                                                [100, 100, 100]),
        num_iterations=10,
        num_samples=1000
    ))
