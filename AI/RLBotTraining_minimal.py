from AI.game_environment import GameEnvironment
from keras.models import Sequential
import keras.layers
from AI.actions import Action

class Training:

    def __init__(self, x, y, botList, no_graphics = True):
        # parameters for training
        num_actions = len(Action.get_actions())
        self.width = x
        self.height = y
        self.botList = botList
        self.model = self.create_model()
        self.env = GameEnvironment(self.width, self.height, no_grafics = no_graphics)
        self.reset_game(True)

    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model_YourBotName.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model_YourBotName.h5")

    def training_done(self):
        print("Training done")
        self.save_model()

    def train(self, game_over = False):
        # Implement the Training here

        self.model.train_on_batch(input_observations, best_actions)
        if game_over:
            self.reset_game(False)
            return
            
    def get_current_input_vector(self):
        #self.env.
        own_bot = self.botList[0][0]
        enemy_bot = self.botList[1][0]

    def reset_game(self, first_time = False):
        botList = []
        models = []
        trainings = []
        for bot in self.botList:
            botList.append(bot[0])
            if bot[1]:
                models.append(self.model)
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
        model.add(keras.layers.Dense(units = 20, activation = 'relu', input_shape = (1, )))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = 40, activation = 'relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units = len(Action.get_actions()), activation = 'softmax'))
        
        model.summary()
        model.compile(optimizer = 'adam')
        return model