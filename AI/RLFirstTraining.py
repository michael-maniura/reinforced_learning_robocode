import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

from AI.game_environment import GameEnvironment
from AI.Experience_Replay import ExperienceReplay
from AI.actions import Action

from AI.DoubleQLearning import reward_induction
from Evaluation.training_evaluation import TrainingEvaluation
from Evaluation.evaluation_visualization import EvaluationVisualization

class RLFirstTaining:

    def __init__(self, x, y, botList, no_graphics = True, verbose=1):
        #  initialize training related variables
        self.epochs_to_train = 3000
        self.current_epoch = 1
        self.batch_size = 20
        self.model_version = 1
        self.count_of_available_actions = len(Action.get_actions())
        
        # game environment related variables
        self.width = x
        self.height = y
        self.botList = botList

        # intialize training and evaluation related class instances and variables
        self.training_evaluation = TrainingEvaluation(self.epochs_to_train, self.batch_size)
        self.experience_replay = ExperienceReplay()
        self.game_count = 0
        self.win_count = 0
        self.batch_training_epsilon = 0.2

        # console output related variables
        self.padding_size = len(str(self.epochs_to_train))
        self.verbose = verbose # 0 = none, 1 = epoch counter, 2 = all
        
        # create models for training and acting
        self.model_train = self.create_model()
        if self.verbose > 1:
            print("Created and assigned new training model")

        self.model_act = self.load_model()
        if self.model_act == None:
            self.model_act = self.create_model()
            if self.verbose > 1:
                print("Created new acting model")
        else:
            if self.verbose > 1:
                print("Loaded existing acting model")

        # finalize initialization by creating game enviroment and starting the "game loop" 
        self.env = GameEnvironment(self.width, self.height, no_grafics = no_graphics)
        self.reset_game(True)

    def save_model(self, model):
        '''
        Save a given model as a serialized JSON file.
        The serialized model weights are saved to a .h5 file.

                Parameters:
                        model (keras.model): Model to save

                Returns:
                        None
        '''
        #model.save("ReinforcedLearningFirst_model")
        model_json = model.to_json()
        with open("ReinforcedLearningFirst_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("ReinforcedLearningFirst_model_weights.h5")

    def load_model(self):
        '''
        Load a model from a serialized JSON file.
        The serialized model weights are loaded from a .h5 file.

                Parameters:
                        None

                Returns:
                        model (keras.model): The loaded model with the loaded weights
        '''
        try:
            json_file = open("ReinforcedLearningFirst_model.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.compile(optimizer = "adam", loss="mse")
            # load weights into new model
            model.load_weights("ReinforcedLearningFirst_model_weights.h5")
            return model
        except BaseException as e:
            print("Could not load model: {0}".format(e))
        return None

    def training_done(self):
        # add the latest model data
        self.training_evaluation.add_model(self.model_version+1, self.game_count, self.win_count)
        if self.verbose  > 1:
            print("Training done")
            print("Won {wins} games out of {games}".format(wins=self.win_count, games=self.game_count))
        EvaluationVisualization(self.training_evaluation)
        self.save_model(self.model_train)
        sys.exit()

    def train(self, training_data, game_over = False):
        if self.current_epoch > self.epochs_to_train:
            self.training_done()
            return

        reward = reward_induction.get_reward(training_data, game_over)

        input = training_data['input']
        last_input = training_data['last_input']
        last_action = training_data['last_action']

        #if input_t is None or last_input is None or last_action is None or reward is None:
        #    return

        self.experience_replay.remember(
            [
                last_input,
                last_action,
                reward,
                input
                ],
            game_over)



        batch_loss = None
        if self.current_epoch > self.batch_size:# and np.random.rand() > self.batch_training_epsilon:
            inputs, targets = self.experience_replay.get_batch(self.model_train, self.model_act, self.batch_size)
            batch_loss = self.model_train.train_on_batch(inputs, targets)

        # update act model after a certain number of epochs
        if self.current_epoch % self.batch_size == 0:
            self.update_acting_model()

        self.training_evaluation.add_epoch(
            self.current_epoch,
            reward,
            self.game_count,
            self.win_count
            )

        if self.verbose > 0:
            self.print_epoch_values(reward, batch_loss)

        self.current_epoch += 1
        if game_over:
            self.reset_game(False)
            return

    def print_epoch_values(self, reward, batch_loss):
        if self.verbose > 1:
            print("\nReward: {0}".format(reward))
            print("Epoch {epoch:>{padding_size}} of {epochs} (Batch loss: {batch_loss})".format(
                epoch=self.current_epoch,
                epochs=self.epochs_to_train,
                padding_size=self.padding_size,
                batch_loss=batch_loss)
            )
        else:
            print("Epoch {epoch:>{padding_size}} of {epochs})".format(
                epoch=self.current_epoch,
                epochs=self.epochs_to_train,
                padding_size=self.padding_size
                )
            )
    
    def update_acting_model(self):
        if self.verbose > 1:
            print("Updating acting model...")
        self.save_model(self.model_train)
        #self.model_act = self.load_model()
        self.model_act.load_weights("ReinforcedLearningFirst_model_weights.h5")
        if self.verbose > 1:
            print("Successfully updated acting model")
        self.training_evaluation.add_model(self.model_version, self.game_count, self.win_count)
        self.model_version += 1

    def on_own_death(self):
        self.training_evaluation.add_game(self.game_count, self.win_count)
        if self.verbose > 1:
            print("Own robot dead")

    def on_enemy_death(self):
        self.win_count += 1
        self.training_evaluation.add_game(self.game_count, self.win_count)
        if self.verbose > 1:
            print("Enemy robot dead")

    def reset_game(self, first_time = False):
        if self.verbose > 1:
            print("Resetting game after a game over")

        botList = []
        models = []
        trainings = []
        for bot in self.botList:
            botList.append(bot[0])
            if bot[1]:
                models.append(self.model_act)
                trainings.append(self)
            else:
                models.append(None)
                trainings.append(None)

        self.game_count += 1
        if first_time:
            self.env.start(botList, models, trainings)
        else:
            self.env.restart(botList, models, trainings)

    def print_keras_device_list(self):
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

    
    """
    def create_model(self):
        if self.verbose > 1:
            self.print_keras_device_list()
            
        model = tf.keras.models.Sequential()
        model.add(keras.layers.Dense(units = 20, activation = 'relu', input_shape = (6, )))
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = 80, activation = 'relu'))
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = 40, activation = 'relu'))
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = len(Action.get_actions())))
        
        model.summary()
        model.compile(
            optimizer = 'adam',
            loss='mse'
        )
        return model
    """
    """
    def create_model(self):
        if self.verbose > 1:
            self.print_keras_device_list()

        model = tf.keras.models.Sequential()
        model.add(keras.layers.Dense(units = 100, activation = 'relu', input_shape = (6, )))
        #model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(units = 100, activation = 'relu'))
        #model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(units = len(Action.get_actions())))
        
        model.summary()
        model.compile(
            optimizer = 'adam',
            loss='mse'
            )
        return model

    """
    def create_model(self):
        if self.verbose > 1:
            self.print_keras_device_list()

        model = tf.keras.models.Sequential()
        model.add(keras.layers.Dense(units = 50, activation = 'relu', input_shape = (6, )))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = 150, activation = 'relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = 250, activation = 'relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = 50, activation = 'relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = len(Action.get_actions())))
        
        model.summary()

        model.compile(
            optimizer = 'adam',
            loss='mse'
            )
        return model

    """
    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(keras.layers.Conv1D(32, 3, activation='relu', input_shape=(10, 1)))
        print(model.output_shape)
        model.add(keras.layers.MaxPooling1D((2)))
        print(model.output_shape)
        model.add(keras.layers.Conv1D(64, 2, activation='relu'))
        print(model.output_shape)
        model.add(keras.layers.MaxPooling1D((2)))
        print(model.output_shape)
        model.add(keras.layers.Flatten())
        print(model.output_shape)
        model.add(keras.layers.Dense(units = len(Action.get_actions()), activation = 'linear'))
        
        model.summary()
        model.compile(
            optimizer = 'adam',
            loss='mse'
            )
    """