import os
import datetime as dt
import matplotlib.pyplot as plt
from Evaluation.training_evaluation import TrainingEvaluation

class EvaluationVisualization:
    
    def __init__(self, training_evaluation):
        self.training_evaluation = training_evaluation
        self.figure_size = (10, 6)

        self.base_directory = os.path.join(os.getcwd(), "evaluation_plots")
        self.current_time = dt.datetime.today().strftime("%Y-%m-%d_%H_%M")
        self.directory = os.path.join(self.base_directory, self.current_time)
        self.prepare_directory()

        self.plot_cumulated_win_ratio_per_epoch()
        self.plot_reward_per_epoch()
        self.plot_cumulated_win_ratio_per_game()
        self.plot_cumulated_win_ratio_per_model()

    def prepare_directory(self):
        if not os.path.exists(self.base_directory):
            os.mkdir(self.base_directory)
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def plot_cumulated_win_ratio_per_epoch(self):
        figure = plt.figure(figsize=self.figure_size, dpi=300)
        ax = plt.subplot(1,1,1)
        
        epochs, game_counter, win_counter, reward = self.training_evaluation.get_epoch_results()
        cumulated_win_ratio_per_epoch = []

        for game_count, win_count in zip(game_counter, win_counter):
            cumulated_win_ratio_per_epoch.append(win_count/game_count)
        
        plt.plot(epochs, cumulated_win_ratio_per_epoch, 'k')
        plt.xlabel("Epoch")
        plt.ylabel("Cumulated win ratio [%]")
        plt.title("Cumulated win ratio per epoch")

        filename = "{0}-cumulated_win_ratio_per_epoch.png".format(self.current_time)
        filepath = os.path.join(self.directory, filename)
        plt.savefig(filepath)

    def plot_reward_per_epoch(self):
        figure = plt.figure(figsize=self.figure_size, dpi=300)
        ax = plt.subplot(1,1,1)
        
        epochs, game_counter, win_counter, reward = self.training_evaluation.get_epoch_results()
        
        plt.plot(epochs, reward, 'k')
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.title("Reward per epoch")

        filename = "{0}-reward_per_epoch.png".format(self.current_time)
        filepath = os.path.join(self.directory, filename)
        plt.savefig(filepath)

    def plot_cumulated_win_ratio_per_game(self):
        figure = plt.figure(figsize=self.figure_size, dpi=300)
        ax = plt.subplot(1,1,1)
        
        games, wins_per_game = self.training_evaluation.get_game_results()
        cumulated_win_ratio_per_game = []

        for game_count, win_count in zip(games, wins_per_game):
            cumulated_win_ratio_per_game.append(win_count/game_count)
        
        plt.plot(games, cumulated_win_ratio_per_game, 'k')
        plt.xlabel("Game number")
        plt.ylabel("Cumulated win ratio [%]")
        plt.title("Cumulated win ratio per game played")

        filename = "{0}-cumulated_win_ratio_per_game.png".format(self.current_time)
        filepath = os.path.join(self.directory, filename)
        plt.savefig(filepath)

    def plot_cumulated_win_ratio_per_model(self):
        figure = plt.figure(figsize=self.figure_size, dpi=300)
        ax = plt.subplot(1,1,1)
        
        model_versions, games_per_model, wins_per_model = self.training_evaluation.get_model_version_results()
        cumulated_win_ratio_per_model = []

        for game_count, win_count in zip(games_per_model, wins_per_model):
            cumulated_win_ratio_per_model.append(win_count/game_count)
        
        plt.plot(model_versions, cumulated_win_ratio_per_model, 'k')
        plt.xlabel("Model number")
        plt.ylabel("Cumulated win ratio [%]")
        plt.title("Cumulated win ratio per model version")

        filename = "{0}-cumulated_win_ratio_per_model.png".format(self.current_time)
        filepath = os.path.join(self.directory, filename)
        plt.savefig(filepath)
