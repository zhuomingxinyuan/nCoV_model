from scipy.stats import norm
import random


class patient:

    def __init__(self, state, time, better_mean, better_std, worse_mean, worse_std):
        # state:
        # 0:cured
        # 1:mild case
        # 2:severe case
        self.state = state
        self.time = time
        self.better_mean = better_mean
        self.better_std = better_std

        self.worse_mean = worse_mean
        self.worse_std = worse_std
        self.prob_better = 0
        self.prob_worse = 0

    def get_prob(cls):

        cls.prob_better = 0.7 * (norm.cdf(cls.time + 1, loc=cls.better_mean, scale=cls.better_std)
                                 - norm.cdf(cls.time, loc=cls.better_mean, scale=cls.better_std)) \
                          / (0.7 * (1 - norm.cdf(cls.time, loc=cls.better_mean, scale=cls.better_std)) + 0.3 * (
                    1 - norm.cdf(cls.time, loc=cls.worse_mean, scale=cls.worse_std)))

        cls.prob_worse = 0.3 * (norm.cdf(cls.time + 1, loc=cls.worse_mean, scale=cls.worse_std)
                                - norm.cdf(cls.time, loc=cls.worse_mean, scale=cls.worse_std)) \
                         / (0.7 * (1 - norm.cdf(cls.time, loc=cls.better_mean, scale=cls.better_std)) + 0.3 * (
                    1 - norm.cdf(cls.time, loc=cls.worse_mean, scale=cls.worse_std)))

    def evolve(self):

        # the setting is such that if it is in state 1 or 2, it does not evolve anymore.
        if self.state == 1:
            self.get_prob()
            rnd = random.random()
            if rnd < self.prob_better:
                self.state = 0
                self.time = 0
            if rnd > (1 - self.prob_worse):
                self.state = 2
                self.time = 0

            else:
                self.time = self.time + 1
        else:
            self.time = self.time + 1