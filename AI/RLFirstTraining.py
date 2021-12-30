from AI.game_environment import GameEnvironment
from AI.Experience_Replay import ExperienceReplay
from keras.models import Sequential
import keras.layers
from AI.actions import Action

import sys
from matplotlib import pyplot

class RLFirstTaining:

    def __init__(self, x, y, botList, no_graphics = True):
        # parameters
        self.epochs_to_train = 100#0
        self.current_epoch = 1
        self.batch_size = 20

        self.padding_size = len(str(self.epochs_to_train))
        self.count_of_available_actions = len(Action.get_actions())

        self.games_played = 0
        self.wins = 0

        self.experience_replay = ExperienceReplay()

        self.width = x
        self.height = y

        self.botList = botList
        
        self.model_train = self.create_model()
        print("Created and assigned new training model")

        self.model_act = self.load_model()
        if self.model_act == None:
            self.model_act = self.create_model()
            print("Created new acting model")
        else:
            print("Loaded existing acting model")

        self.env = GameEnvironment(self.width, self.height, no_grafics = no_graphics)
        self.reset_game(True)

    def save_model(self, model):
        # serialize model to JSON
        model_json = model.to_json()
        with open("ReinforcedLearningFirst_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("ReinforcedLearningFirst_model_weights.h5")

    def load_model(self):
        try:
            json_file = open("ReinforcedLearningFirst_model.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("ReinforcedLearningFirst_model_weights.h5")
            return model
        except:
            print("Could not load model")
        return None

    def training_done(self):
        print("Training done")
        print("Won {wins} games out of {games}".format(wins=self.wins, games=self.games_played))
        self.save_model(self.model_train)
        sys.exit()

    def train(self, training_data, game_over = False):
        if self.current_epoch > self.epochs_to_train:
            self.training_done()
            return

        input_t = training_data['input_t']
        last_input = training_data['last_input']
        last_action = training_data['last_action']

        reward = self.get_reward(training_data, game_over)

        if input_t is None or last_input is None or last_action is None or reward is None:
            return

        self.experience_replay.remember(
            [
                last_input,
                last_action,
                reward,
                input_t
                ],
            game_over)

        inputs, targets = self.experience_replay.get_batch(self.model_train, self.model_act, self.batch_size)

        batch_loss = self.model_train.train_on_batch(inputs, targets)

        # update act model after a certain number of epochs
        if self.current_epoch % 30 == 0:
            print("Updating acting model...")
            self.save_model(self.model_train)
            self.model_act.load_weights("ReinforcedLearningFirst_model_weights.h5")
            print("Successfully updated acting model")

        print("\nReward: {0}".format(reward))
        print("Angle: {angle:>20}\t|\tOwn health: {own_health:>3}\t|\tEnemy health: {enemy_health:>3}\t|\tEnemy in sight: {enemy_in_sight}\t|\tIn enemy sight: {in_enemy_sight}".format(
            angle=input_t[0][0],
            own_health=input_t[0][1],
            enemy_health=input_t[0][2],
            enemy_in_sight=input_t[0][3],
            in_enemy_sight=input_t[0][4]
        ))
        #print(input_t)

        print("Epoch {epoch:>{padding_size}} of {epochs} (Batch loss: {batch_loss})".format(
            epoch=self.current_epoch,
            epochs=self.epochs_to_train,
            padding_size=self.padding_size,
            batch_loss=batch_loss)
        )

        self.current_epoch += 1
        
        if game_over:
            print("Resetting game after a game over")
            self.reset_game(False)
            return

    def get_reward(self, training_data, game_over):
        reward = 0

        input_t = training_data["input_t"][0]
        last_input = training_data["last_input"][0]

        angle_to_enemy = input_t[0]        
        own_energy = input_t[1]
        enemy_energy = input_t[2]
        shot_possible_by_enemy = input_t[3]
        shot_possible_at_enemy = input_t[4]

        previous_angle_to_enemy = last_input[0]

        if game_over:
            if enemy_energy == 0:
                return 1
            else:
                return -1
        
        if shot_possible_at_enemy:
            if training_data['last_action'] == 4:
                return 0.2
                #reward = reward + 0.2
        """
        if shot_possible_by_enemy:
            if training_data['last_action'] != 4:
                reward = reward + 0.1
        """
        
        reward = self.normalize_angle(angle_to_enemy) - self.normalize_angle(previous_angle_to_enemy)
        if reward < 0:
            reward = -0.1

        return reward

    def normalize_angle(self, angle):
        angle_normalized = (1-abs(2*angle))/5-0.1
        return angle_normalized

    def on_own_death(self):
        print("Own robot dead")

    def on_enemy_death(self):
        self.wins += 1
        print("Enemy robot dead")

    def reset_game(self, first_time = False):
        self.games_played += 1
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

        if first_time:
            self.env.start(botList, models, trainings)
        else:
            self.env.restart(botList, models, trainings)

    def create_model(self):
        # You have to set up the model with keras here
        model = Sequential()
        model.add(keras.layers.Dense(units = 20, activation = 'relu', input_shape = (5, )))
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = 40, activation = 'relu'))
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = len(Action.get_actions())))
        
        model.summary()
        model.compile(optimizer = 'adam', loss='mse')
        return model