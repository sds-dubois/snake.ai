import interface
class TDLearningAlgorithm:
    def __init__(self, n_opponents,discount, featureExtractor, explorationProb=0.2, weights = None):
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.numIters = 0
        self.n_opponents = n_opponents
        self.TDagent_id = n_opponents + 1
        
        if weights:
            with open(weights, "rb") as fin:
                weights_ = pickle.load(fin)
                self.weights = weights_
        else:
            self.weights = defaultdict(float)


    def evalV(self, state):
        """
        Evaluate V for a given ('state')
        """
        score = 0
        for f, v in self.featureExtractor(state,self.TDagent_id):
            score += self.weights[f] * v
        return score

    def getAction(self, state, player):

        """
        The strategy implemented by this algorithm.
        With probability `explorationProb` take a random action.
        """
        #self.numIters += 1
        actions = state.simple_actions(player)
        if len(actions) == 0:
            return None
        
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            if player == self.TDagent_id
                return max((self.evalV(state.generateSuccessor(player,action)),  action) for action in actions)[1]
            else:
                return min((self.evalV(state.generateSuccessor(player,action)),  action) for action in actions)[1]


    def getStepSize(self):
        """
        Get the step size to update the weights.
        """
        return 1.0 / math.sqrt(self.numIters)

    def incorporateFeedback(self, state, reward, newState):
        if newState is None:
            return
        
        phi = self.featureExtractor(state)
        pred = sum(self.weights[k] * v for k,v in phi)
        try:
            newV = self.evalV(newState)
        except:
            newV = 0.
        target = reward + self.discount * newV
        for k,v in phi:     
            self.weights[k] = self.weights[k] - self.getStepSize() * (pred - target) * v

    def train(self, grid_size, num_trials=100, max_iter=1000, verbose=False):
        print "RL training"
        totalRewards = []  # The rewards we get on each trial
        for trial in xrange(num_trials):
            progressBar(trial, num_trials)
            game = interface.Game(grid_size, n_snakes= self.n_opponents + 1, candy_ratio = 1., max_iter = max_iter)
            state = game.startState()
            totalDiscount = 1
            totalReward = 0
            points = state.snakes[self.TDagent_id].points
            while not game.isEnd(state) and self.TDagent_id in state.snakes:
                # Compute the actions for each player following its strategy
                actions = {i: self.getAction(state,i)}
                newState = game.succ(state, actions)
                if self.TDagent_id in newState.snakes:
                    if len(newState.snakes) == 1: # it won
                        reward = 2.0 * newState.snakes[self.TDagent_id].points
                    else:
                        reward = newState.snakes[self.TDagent_id].points - points

                    points = newState.snakes[self.TDagent_id].points
                    self.incorporateFeedback(state, action, reward, newState)
                else: # it died
                    reward = - points
                    self.incorporateFeedback(state, action, reward, newState)

                totalReward += totalDiscount * reward
                totalDiscount *= self.discount
                state = newState

            if verbose:
                print "Trial %d (totalReward = %s)" % (trial, totalReward)
            totalRewards.append(totalReward)

        progressBar(num_trials, num_trials)
        print "Average reward:", sum(totalRewards)/num_trials
        return totalRewards

############################################################
############################################################