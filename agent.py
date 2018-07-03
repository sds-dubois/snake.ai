"""
Agent interface
"""

class Agent:
    """
    An agent defined by a strategy.
    """

    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy
        self.player_id = None

    def __str__(self):
        return self.name

    def setPlayerId(self, i):
        self.player_id = i

    def getPlayerId(self):
        return self.player_id

    def nextAction(self, state):
        if self.player_id is None:
            raise("Player ID missing")
        return self.strategy(self.player_id, state)

    def lastReward(self, game):
        if self.player_id is None:
            raise("Player ID missing")
        return game.agentLastReward(self.player_id)

    def isAlive(self, game):
        if self.player_id is None:
            raise("Player ID missing")
        return game.isAlive(self.player_id)