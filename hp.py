import pickle

class HP:
    def __init__(self, grid_size, max_iter, discount):
        self.grid_size = grid_size
        self.max_iter = max_iter
        self.discount = discount


class RlHp:
    def __init__(self, rl_type, radius, filter_actions, lambda_, q_type):
        self.rl_type = rl_type
        self.radius = radius
        self.filter_actions = filter_actions
        self.lambda_ = lambda_
        self.q_type = q_type
        self.model = None

    def save_model(self, model, filename):
        self.model = model

        with open("data/" + filename, "wb") as fout:
            pickle.dump(self, fout)

class EsHp:
    def __init__(self, radius):
        self.radius = radius

    def save_model(self, model, filename):
        self.model = model

        with open("data/" + filename, "wb") as fout:
            pickle.dump(self, fout)

def load_from(filename):
    with open("data/" + filename, "rb") as fin:
        out = pickle.load(fin)
    return out
   