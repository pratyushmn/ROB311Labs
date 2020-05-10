from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score

class StudentAgent(IteratedGamePlayer):
    """
    YOUR DOCUMENTATION GOES HERE!
    My agent uses a combination of 3 different strategies - Bayesian predictions, Q learning, and random choice.

    The Bayesian predictions work by creating a matrix that tracks what move the opponent
    plays given the previous state (which is a number from 0 to 8 that encodes both mine and
    the opponent's last moves). Based on those counts (and the total counts of how often the game
    was in that state before) the probabilities of what move the opponent is likely to play next given 
    the past state of the game can be calculated. Using that information, the agent randomly
    chooses it's own move by picking the move that would beat whatever move it believes the 
    opponent will play next. The prediction is randomly picked based on the calculated probabilities
    of each possible move from the Bayesian matrix, and isn't just picking the move with maximum
    probability.

    The Q-Learning works using the Bellman equation for stochastic environments (ie. Markov Decision Processes). 
    The Q function essentially evaluates the "value" of picking a certain action given the current state.
    Since there are only 9 possible states and 3 actions, the function is implemented as a matrix
    where values can be looked up with the current state and proposed action. After every turn, the relevant Q
    values are updated using Q_t (s,a) = Q_(t−1) (s,a) + αlpha*(R (s,a)+ γ*max(Q (s′,a′)) − Q_(t−1) (s,a))
    (where s, a are the past state and past action, and s', a' are the current state and possible next actions; R is the reward evaluated at that state and action)
    which was derived from the Bellman equation as seen in the following link:
    # https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/
    Then, when it comes time to make a move, the action with highest Q value given the
    last state is picked.

    The random choice is implemented by uniformly choosing a possible move. This was added to help
    make the agent more unpredictable against opposing agents.

    Overall, the agent makes a move using the Q-values 60% of the time, using the Bayesian matrix 35%
    of the time, and completely randomly 5% of the time. These values were chosen by trial and error.
    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        # YOUR CODE GOES HERE
        self.actions = 3
        self.states = 9
        self.responses = {0: 1, 1: 2, 2: 0}
        self.probs = np.ones((self.states, 4)) # rows -> past states (9 possibilities), columns -> next opponent move observation (3 possibilities) + total observations of the past state
        for i in range(self.states):
            self.probs[i, 3] = 3 # since matrix was initialized with Laplace Smoothing, we started with one observation for each element in the matrix - thus there are three total observations in each state
        
        self.Q_func = np.random.randn(self.states, self.actions) # states, actions
        self.last_state = 0
        self.gamma = 0.01 # weighing future rewards
        self.alpha = 0.99 # LR
        self.epsilon = 0.4 # degree of exploration (using Bayesian Probabilities)
        self.r_cutoff = 0.05 # probability of completely random choice
        self.times_played = 0        

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        # YOUR CODE GOES HERE
        rng = np.random.randint(0, 101)/100
        move = np.random.randint(0, self.n_moves)
                
        if self.times_played <= 1 or rng < self.r_cutoff:
            # if self.times_played <= 2:
            #     move = self.times_played
            pass
        elif rng < self.epsilon:
            options = np.copy(self.probs[self.last_state, 0:3])
            total = float(self.probs[self.last_state, 3])
            options /= total
            prediction = np.random.choice(3, p=options)
            move = self.responses[prediction]
        else:
            move = np.argmax(self.Q_func[self.last_state, :])
            
        self.times_played += 1
        return move

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        # YOUR CODE GOES HERE
        if self.times_played > 1:
            # reward = self.game_matrix[my_move][other_move]
            self.probs[self.last_state, other_move] += 1
            self.probs[self.last_state, 3] += 1
            # self.rewards[self.last_state, my_move] = reward

            max_q = self.Q_func[3*my_move + other_move, 0]
            for i in range(self.actions):
                max_q = max(max_q, self.Q_func[3*my_move + other_move, i])

            self.Q_func[self.last_state, my_move] = self.Q_func[self.last_state, my_move] + self.alpha*(self.game_matrix[my_move][other_move] + self.gamma*np.max(self.Q_func[3*my_move + other_move])* - self.Q_func[self.last_state, my_move])

        self.last_state = 3*my_move + other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        # YOUR CODE GOES HERE
        self.Q_func = np.random.randn(3*3, 3) # states, actions
        self.last_state = 0
        self.gamma = 0.01 # weighing future rewards
        self.alpha = 0.99 # LR
        self.epsilon = 0.4 # degree of exploration
        self.times_played = 0   
        self.probs = np.ones((self.states, 4))
        for i in range(self.states):
            self.probs[i, 3] = 3

if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """

    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    copy_cat_player = CopycatPlayer(game_matrix)
    uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix)

    print("Uniform player's score: {:}".format(uniform_score))
    print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score, first_move_score = play_game(student_player, copy_cat_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("First-move player's score: {:}".format(first_move_score))