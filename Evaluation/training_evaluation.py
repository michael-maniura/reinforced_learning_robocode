import numpy as np

class TrainingEvaluation:

    def __init__(self, epochs_to_train, batch_size):
        self.epochs_trained = epochs_to_train
        self.batch_size_in_training = batch_size

        # lists based on epoch count
        self.epochs = []
        self.game_counter_per_epoch = []
        self.win_counter_per_epoch = []
        self.reward_per_epoch = []

        # lists based on game count
        self.games = []
        self.wins_per_game = []

        # lists based on model version
        self.model_versions = []
        self.games_per_model_version = []
        self.wins_per_model_version = []

    def add_epoch(self, epoch: int, reward: float, game_count: int, win_count: int):
        '''
        Add a given epoch number with associated game information to the epoch evalution lists

                Parameters:
                        epoch: Number of the epoch at which the other parameters have been observed
                        reward: Reward in this epoch
                        game_count: Number of played games in this epoch
                        win_count: Number of won games in this epoch
        '''
        self.epochs.append(epoch)
        self.reward_per_epoch.append(reward)
        self.game_counter_per_epoch.append(game_count)
        self.win_counter_per_epoch.append(win_count)

    def add_game(self, game_count: int, win_count: int):
        '''
        Add a given game number with associated win count to the game evalution lists

                Parameters:
                        game_count: Number of played games
                        win_count: Number of won games at this game count
        '''
        self.games.append(game_count)
        self.wins_per_game.append(win_count)

    def add_model(self, model_version: int, game_count: int, win_count: int):
        '''
        Add a given model number with associated game information to the model evalution lists

                Parameters:
                        model_version: Number of model version with which the other parameters have been observed
                        game_count: Number of played games up until this model version
                        win_count: Number of won games up until this model version
        '''
        self.model_versions.append(model_version)
        self.games_per_model_version.append(game_count)
        self.wins_per_model_version.append(win_count)

    def get_epoch_results(self) -> tuple:
        '''
        Get a tuple of epoch based evaluation data

                Returns:
                        A tuple containing lists with the epoch numbers,
                        game counts per epoch, wins per epoch and the reward per epoch
        '''
        return (
            self.epochs,
            self.game_counter_per_epoch,
            self.win_counter_per_epoch,
            self.reward_per_epoch
        )

    def get_game_results(self) -> tuple:
        '''
        Get a tuple of game based evaluation data

                Returns:
                        A tuple containing a list of game numbers and a list of won games associated
                        to each number from the first list.
        '''
        return (
            self.games,
            self.wins_per_game
        )

    def get_model_version_results(self) -> tuple:
        '''
        Get a tuple of model version based evaluation data

                Returns:
                        A tuple containing lists with the model version numbers,
                        game counts per version and wins per version
        '''
        return (
            self.model_versions,
            self.games_per_model_version,
            self.wins_per_model_version
        )