from Robots.charlier import Charlier
from Robots.RL_First import ReinforcedLearningFirst
from AI.RLFirstTraining import RLFirstTaining

botList = []
#your bot
bot = ReinforcedLearningFirst
botList.append((bot, True))
#enemy
bot = Charlier
botList.append((bot, False))
x = 500
y = 700
rlbot_training = RLFirstTaining(x, y, botList, no_graphics = True)