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

class CEMNetwork():
    # Cross Entropy Method for RL
    def __init__(self, num_inputs, HLUs, num_outputs):
        self.inputs = num_inputs
        self.hidden_layer = HLUs
        self.outputs = num_outputs
        
        self.fc1_weights = np.zeros((self.hidden_layer, self.inputs))
        self.fc1_bias = np.zeros((self.hidden_layer, 1))
        self.fc2_weights = np.zeros((self.outputs, self.hidden_layer))
        self.fc2_bias = np.zeros((self.outputs, 1))

    def forward(self, input):
        # print("predict")
        pred = np.add(np.dot(self.fc1_weights, input), self.fc1_bias[:, np.newaxis])
        pred[pred < 0] = 0 # ReLU
        pred = np.add(np.dot(self.fc2_weights, pred), self.fc2_bias[:, np.newaxis])
        return self.softmax(pred)

    def get_num_weights(self):
        return (self.inputs+1)*self.hidden_layer + (self.hidden_layer+1)*self.outputs

    def softmax(self, x):
        z = x - max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator/denominator
        return softmax

    def update_weights(self, weights):
        self.fc1_weights = weights[: self.inputs*self.hidden_layer].reshape(self.hidden_layer, self.inputs)
        self.fc1_bias = weights[self.inputs*self.hidden_layer : (self.inputs*self.hidden_layer) + self.hidden_layer]
        self.fc2_weights = weights[(self.inputs*self.hidden_layer) + self.hidden_layer : (self.inputs*self.hidden_layer) + self.hidden_layer + self.hidden_layer*self.outputs].reshape(self.outputs, self.hidden_layer)
        self.fc2_bias = weights[(self.inputs*self.hidden_layer) + self.hidden_layer + self.hidden_layer*self.outputs :]

class StudentAgent(IteratedGamePlayer):
    """
    YOUR DOCUMENTATION GOES HERE!
    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        self.responses = {0: 1, 1: 2, 2: 0}
        self.network = CEMNetwork(10, 8, 3)

        self.sigma_CEM = 0.5
        self.mu_CEM = 0
        self.pop_size = 5
        self.frac_elite = 0.4
        self.max_timesteps = 5

        self.scores = []
        self.curr_t_score = 0
        self.best_weights = self.sigma_CEM*np.random.randn(self.network.get_num_weights())
        self.all_weights = np.array([self.best_weights + (self.sigma_CEM*np.random.randn(self.network.get_num_weights())) for i in range(self.pop_size)])
        self.network.update_weights(self.all_weights[0])
        self.times_played = 0
        self.current_timesteps = 0
        self.iteration = 0
        self.opponent_past = []
        self.my_past = []
        self.past_states = []
        # YOUR CODE GOES HERE
        pass

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        # YOUR CODE GOES HERE

        if (self.times_played < 10):
            move = np.random.randint(3)

        else:
            predictions = self.network.forward(np.array(self.past_states).reshape(10, 1))
            prediction = np.random.choice(3, p=predictions[:, 0])
            move = self.responses[prediction]


        self.times_played += 1
        self.current_timesteps += 1
        return move

        
        #return 1

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        # YOUR CODE GOES HERE
        self.curr_t_score += self.game_matrix[my_move][other_move]
        self.past_states.append(other_move)
        self.past_states.append(my_move)
        self.opponent_past.append(other_move)
        self.my_past.append(my_move)

        if self.times_played < 10:
            self.current_timesteps = 0
            return

        if len(self.past_states) > 10:
            self.past_states = self.past_states[-10 :]

        if self.current_timesteps == self.max_timesteps:
            # try next set of generated weights
            # record score
            self.iteration += 1
            self.current_timesteps = 0
            self.scores.append(self.curr_t_score)
            self.curr_t_score = 0
            if self.iteration < self.pop_size:
                self.network.update_weights(np.array(self.all_weights[self.iteration]))            
            
        if len(self.scores) == self.pop_size:
            # choose elite weights
            # generate new weights
            self.iteration = 0
            # print(sum(self.scores))
            elite_idx = np.array(self.scores).argsort()[int(-self.frac_elite*self.pop_size):]
            elite_weights = [self.all_weights[i] for i in elite_idx]
            self.best_weights = np.array(elite_weights).mean(axis=0)
            self.all_weights = [self.best_weights + (self.sigma_CEM*np.random.randn(self.network.get_num_weights())) for i in range(self.pop_size)]
            self.network.update_weights(np.array(self.all_weights[self.iteration]))
            self.scores.clear()
        

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        # YOUR CODE GOES HERE
        self.opponent_past.clear()
        self.my_past.clear()
        self.scores.clear()
        self.past_states.clear()
        self.curr_t_score = 0
        self.best_weights = self.sigma_CEM*np.random.randn(self.network.get_num_weights())
        self.all_weights = np.array([self.best_weights + (self.sigma_CEM*np.random.randn(self.network.get_num_weights())) for i in range(self.pop_size)])
        self.network.update_weights(self.all_weights[0])
        self.times_played = 0
        self.current_timesteps = 0
        self.iteration = 0



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
    student_score, first_move_score = play_game(student_player, copy_cat_player, game_matrix,N=10000)

    print("Your player's score: {:}".format(student_score))
    print("First-move player's score: {:}".format(first_move_score))