from AI.DoubleQLearning import heuristics

def get_reward(training_data, game_over):
    reward = 0

    input = training_data["input"][0]
    last_input = training_data["last_input"][0]
    last_action = training_data['last_action']

    current_own_energy = input[0]
    current_enemy_energy = input[1]
    #current_own_position_x = input[2]
    #current_own_position_y = input[3]
    #current_enemy_position_x = input[4]
    #current_enemy_position_y = input[5]
    current_gun_heading = input[2]
    current_angle_to_enemy = input[3]
    current_shot_possible_at_enemy = input[4]
    current_shot_possible_by_enemy = input[5]

    previous_own_energy = last_input[0]
    previous_enemy_energy = last_input[1]
    #previous_own_position_x = last_input[2]
    #previous_own_position_y = last_input[3]
    #previous_enemy_position_x = last_input[4]
    #previous_enemy_position_y = last_input[5]
    previous_gun_heading = last_input[2]
    previous_angle_to_enemy = last_input[3]
    previous_shot_possible_at_enemy = last_input[4]
    previous_shot_possible_by_enemy = last_input[5]
    
    if game_over:
        return heuristics.game_over_heuristic(previous_enemy_energy)
    
    reward += heuristics.shot_possible_at_enemy_heuristic(last_action, previous_shot_possible_at_enemy)
    reward += heuristics.shot_possible_by_enemy_heuristic(last_action, previous_shot_possible_by_enemy)
    reward += heuristics.angle_change_heuristic(current_angle_to_enemy, previous_angle_to_enemy)
    reward += heuristics.own_energy_change_heuristic(current_own_energy, previous_own_energy)
    reward += heuristics.enemy_energy_change_heuristic(current_enemy_energy, previous_enemy_energy)
    
    return reward