
import numpy as np
import conf
import simpy
import sys

class Factory():
    def __init__(self, output):
        self.output = output
        self.stock = {}
        self.sold = {}
        for item in output:
            self.stock[item] = output[item]['initial']
            self.sold[item] = 0

    def process_order(self, item, num_sold):
        self.stock[item] -= num_sold
        self.sold[item] += num_sold

    def order_input(self, item, num):
        pass

def generate_orders(env, factory, interval):
    """
    Generates new orders and processes them if the expected
    wait time for an item is less than the given threshold
    """
    items = list(factory.output.keys())
    print('here')
    while True:
        next_order = np.random.exponential(1/interval)
        yield env.timeout(next_order)

        item = np.random.choice(items)
        num_ordered = int(np.random.uniform(1,6))
        if factory.stock[item] > 0:
            factory.process_order(item, num_ordered)

def run_simulation(seed = None):
    if seed != None:
        np.random.seed = seed
    env = simpy.Environment()
    factory = Factory(output = conf.output)
    env.process(generate_orders(env, factory, conf.interval))
    env.run(until = conf.end_time)
