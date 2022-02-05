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

    def add_epoch(self, epoch, reward, game_count, win_count):
        self.epochs.append(epoch)
        self.reward_per_epoch.append(reward)
        self.game_counter_per_epoch.append(game_count)
        self.win_counter_per_epoch.append(win_count)

    def add_game(self, game_count, win_count):
        self.games.append(game_count)
        self.wins_per_game.append(win_count)

    def add_model(self, model_version, game_count, win_count):
        self.model_versions.append(model_version)
        self.games_per_model_version.append(game_count)
        self.wins_per_model_version.append(win_count)

    def get_epoch_results(self):
        return (
            self.epochs,
            self.game_counter_per_epoch,
            self.win_counter_per_epoch,
            self.reward_per_epoch
        )

    def get_game_results(self):
        return (
            self.games,
            self.wins_per_game
        )

    def get_model_version_results(self):
        return (
            self.model_versions,
            self.games_per_model_version,
            self.wins_per_model_version
        )