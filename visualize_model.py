import keras
import tensorflow

json_file = open("ReinforcedLearningFirst_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)
model.compile(optimizer = "adam", loss="mse")
# load weights into new model
model.load_weights("ReinforcedLearningFirst_model_weights.h5")


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)