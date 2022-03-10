def game_over_heuristic(enemy_energy: float) -> int:
    if enemy_energy <= 0:
        return 1
    else:
        return -1

def shot_possible_at_enemy_heuristic(action: int, shot_possible: int):
    if shot_possible:
        if action == 4: # shoot
            return 0.5
        else: # something else than shooting
            return -0.5
    else:
        if action == 4: # shooting without being able to is not efficient
            return -0.3
        else: # something else than shooting is wanted in that case
            return 0.5

def shot_possible_by_enemy_heuristic(action, shot_possible_by_enemy):
    if shot_possible_by_enemy:
        if action == 2 or action == 3: # evade forwards or backwards 
            return 0.3
        else: # something else than evading
            return -0.3
    else:
        return 0

def normalize_angle(angle):
    angle_normalized = (1 - abs(2 * angle)) / 5-0.1
    return angle_normalized

def angle_change_heuristic(current_angle_to_enemy, previous_angle_to_enemy):
    reward = normalize_angle(current_angle_to_enemy) - normalize_angle(previous_angle_to_enemy)
    
    if reward < 0 and previous_angle_to_enemy <= 0.05: # punish a worse angle
        reward = -0.5

    if abs(reward) > 0.1:
        reward = 0
    
    return reward

def own_energy_change_heuristic(current_energy, previous_energy, action):
    if current_energy > previous_energy:
        return 0.1 # reward energy gain by hitting the enemy
    elif current_energy < previous_energy:
        if action == 2 or action == 3: #
            return -0.2 # penalize energy loss by hitting a wall
        else:
            return -0.1 # penalize energy loss by being shot
    elif current_energy == previous_energy:
        return 0 # no penalty as of now for keeping the same energy level
    
def enemy_energy_change_heuristic(current_energy, previous_energy):
    if current_energy > previous_energy:
        return -0.1 # penalize enemy energy gain by hitting the own bot
    elif current_energy < previous_energy:
        return 0.1 # reward enemy energy loss - maybe add check if the own bot has shot
    elif current_energy == previous_energy:
        return -0.01 # slightly penalize, since the own bot has do to something
