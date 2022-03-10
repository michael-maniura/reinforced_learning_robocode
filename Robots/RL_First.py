#! /usr/bin/python
# -*- coding: utf-8 -*-
from AI.actions import Action
from Objects.robot import Robot  # Import a base Robot
from keras.models import model_from_json
import numpy as np


class ReinforcedLearningFirst(Robot):
    # Create a Robot

    def init(self):
        # To initialise your robot
        # Feel free to customize: Set the bot color in RGB
        self.setColor(0, 0, 100)
        self.setGunColor(0, 0, 100)
        self.setRadarColor(0, 60, 0)
        self.setBulletsColor(255, 150, 150)
        self.maxDepth = 5

        #Don't Change
        self.setRadarField("thin")
        self.radarVisible(True)  # if True the radar field is visible
        self.gun_to_side()
        self.lockRadar("gun")
        self.map_size = self.getMapSize()
        
        self.run_count = 0
        self.batch_size = 50

        self.last_action = None
        self.last_input = None
        self.previous_enemy_position = None

        # Training variables
        self.epsilon = 0.1
        self.num_actions = len(Action.get_actions())
        self.model_file_name = 'ReinforcedLearningFirst_model.json'
        self.model_weights_file_name = 'ReinforcedLearningFirst_model_weights.h5'

        if self.model is None:
            self.load_model()

    def load_model(self):
        json_file = open(self.model_file_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.model_weights_file_name)

    def randmax(self, values):
        max_values = []
        current_max = values[0]
        index = 0
        for v in values:
            if v > current_max:
                max_values = [index]
                current_max = v
            elif v == current_max:
                max_values.append(index)
            index += 1
        if len(max_values) == 0:
            return np.random.randint(0,len(values)-1)
        else:
            return np.random.choice(max_values)

    def get_training_data(self):
        input = self.observe()
        training_data = {}
        training_data["input"] = input
        training_data['last_input'] = self.last_input
        training_data['last_action'] = self.last_action
        return training_data

    def run(self):
        input = self.observe() # current state

        if self.training and self.last_action is not None:
            game_over = False
            # use the current state to evaluate the last action in the last state
            self.training.train(self.get_training_data(), game_over)
        
        # GLIE
        if self.training and np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.num_actions, size=1)[0]
        else: # Select the action with the highest expected reward
            q = self.model.predict(input)
            action = self.randmax(q[0])

        # save current action and state for evaluation with the next, resulting state
        self.last_action = action
        self.last_input = input

        if action == 0:
            print("turning right")
            self.turn_right()
        elif action == 1:
            print("turning left")
            self.turn_left()
        elif action == 2:
            print("going forward")
            self.forward()
        elif action == 3:
            print("going backwards")
            self.backwards()
        elif action == 4:
            print("shooting")
            self.shoot()

        self.run_count += 1
        if self.run_count % self.batch_size == 0:
            self.model.load_weights(self.model_weights_file_name)

    def normalize_position(self, x, y):
        return (x/self.map_size[0], y/self.map_size[1])
    
    def get_own_normalized_position(self):
        x = self.getPosition().x()
        y = self.getPosition().y()
        return self.normalize_position(x, y)
    
    def get_enemy_normalized_position(self):
        enemy_position = self.getPosition_enemy()
        if enemy_position is None:
            enemy_position = self.previous_enemy_position
        else:
            enemy_position = (self.getPosition_enemy().x(), self.getPosition_enemy().y())
        return self.normalize_position(enemy_position[0], enemy_position[1])

    def get_normalized_positions(self):
        own_position_normalized = self.get_own_normalized_position()
        enemy_position_normalized = self.get_enemy_normalized_position()
        return (own_position_normalized, enemy_position_normalized)

    def get_normalized_energy_levels(self):
        own_energy = self.energy_left_self()
        if own_energy is None:
            if enemy_energy <= 0:
                own_energy = 100
            else:
                own_energy = 0
        own_energy_normalized = own_energy/100

        enemy_energy = self.energy_left_enemy()
        if enemy_energy is None:
            if own_energy <= 0:
                enemy_energy = 100
            else:
                enemy_energy = 0
        enemy_energy_normalized = enemy_energy/100

        return (own_energy_normalized, enemy_energy_normalized)
    
    def get_normalized_gun_heading(self):
        return ((self.getGunHeading() % 360) /360)
    
    def get_normalized_angle_to_enemy(self):
        own_position = (self.getPosition().x(), self.getPosition().y())

        enemy_position = self.getPosition_enemy()
        if enemy_position is None:
            enemy_position = self.previous_enemy_position
        else:
            enemy_position = (self.getPosition_enemy().x(), self.getPosition_enemy().y())
            self.previous_enemy_position = enemy_position
        angle_to_enemy = self.calculate_angle_to_enemy(
            own_position,
            enemy_position
            )
        return angle_to_enemy/360

    def calculate_angle_to_enemy(self, own_position, enemy_position):
        test_step_to_left = 10
        gun_heading_to_left = self.getGunHeading() - test_step_to_left
        angle_to_enemy = np.round(Action.angleTo(own_position, enemy_position, self.getGunHeading()), 2)
        angle_to_enemy_to_left = np.round(Action.angleTo(own_position, enemy_position, gun_heading_to_left), 2)

        if (abs(angle_to_enemy - angle_to_enemy_to_left) >= test_step_to_left \
            and (angle_to_enemy_to_left > angle_to_enemy)) \
                or (abs(angle_to_enemy - angle_to_enemy_to_left) < test_step_to_left \
                    and angle_to_enemy >= 90):
            angle_to_enemy *= -1
        return angle_to_enemy

    def get_shots_possible(self):
        shot_possible_at_enemy = self.shot_possible_at_enemy()
        if shot_possible_at_enemy is None:
            shot_possible_at_enemy = 0
        else:
            shot_possible_at_enemy = int(shot_possible_at_enemy)
        
        shot_possible_by_enemy = self.shot_possible_by_enemy()
        if shot_possible_by_enemy is None:
            shot_possible_by_enemy = 0
        else:
            shot_possible_by_enemy = int(shot_possible_by_enemy)
        return (shot_possible_at_enemy, shot_possible_by_enemy)

    def observe(self):
        own_energy, enemy_energy = self.get_normalized_energy_levels()
        own_position, enemy_position = self.get_normalized_positions()
        gun_heading = self.get_normalized_gun_heading()
        angle_to_enemy = self.get_normalized_angle_to_enemy()

        shot_possible_at_enemy, shot_possible_by_enemy = self.get_shots_possible()

        inputs = [
            own_energy,
            enemy_energy,
            #own_position[0],
            #own_position[1],
            #enemy_position[0],
            #enemy_position[1],
            gun_heading,
            angle_to_enemy,
            shot_possible_at_enemy,
            shot_possible_by_enemy
        ]

        return np.array(inputs).reshape(1, -1)

    def onHitWall(self):
        self.reset()  # To reset the run function to the begining (automatically called on hitWall, and robotHit event)
        self.rPrint('ouch! a wall !')

    def sensors(self):  # NECESARY FOR THE GAME
        pass

    def onRobotHit(self, robotId, robotName):  # when My bot hit another
        self.rPrint('collision with:' + str(robotId))

    def onHitByRobot(self, robotId, robotName):
        self.rPrint("damn a bot collided me!")

    def onHitByBullet(self, bulletBotId, bulletBotName, bulletPower):  # NECESARY FOR THE GAME
        """ When i'm hit by a bullet"""
        self.rPrint("hit by " + str(bulletBotId) + "with power:" + str(bulletPower))

    def onBulletHit(self, botId, bulletId):  # NECESARY FOR THE GAME
        """when my bullet hit a bot"""
        self.rPrint("fire done on " + str(botId))

    def onBulletMiss(self, bulletId):  # NECESARY FOR THE GAME
        """when my bullet hit a wall"""
        self.rPrint("the bullet " + str(bulletId) + " fail")

    def onRobotDeath(self):  # NECESARY FOR THE GAME
        """When my bot die"""
        self.rPrint("damn I'm Dead")
        if self.training and self.last_action is not None:
            # here you might want to add things
            game_over = True
            self.training.on_own_death()
            self.training.train(self.get_training_data(), game_over)

    def onTargetSpotted(self, botId, botName, botPos):  # NECESARY FOR THE GAME
        "when the bot see another one"
        self.rPrint("I see the bot:" + str(botId) + "on position: x:" + str(botPos.x()) + " , y:" + str(botPos.y()))

    def onEnemyDeath(self):
        """When an enemy dies"""
        if self.training and self.last_action is not None:
            #here you might want to add things
            game_over = True
            self.training.on_enemy_death()
            self.training.train(self.get_training_data(), game_over)