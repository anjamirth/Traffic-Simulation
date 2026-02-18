import random

class Cars:
    def __init__(self, v_max, current_edge, is_human, lane=0, pos=0, speed=0):
        self.speed = speed
        self.v_max = v_max
        self.edge = current_edge
        self.lane = lane
        self.pos = pos
        self.is_human = is_human   # True = human, False = autonomous

    def accelerate(self):
        if self.speed < self.v_max:
            self.speed += 1

    def brake(self, gap):
        if self.speed > gap:
            self.speed = gap

    def randomise(self, human_brake_prob, random_braking_reduction):
        # humans brake with probability human_brake_prob
        # autonomous brake with reduced probability human_brake_prob * (1 - random_braking_reduction)
        prob = human_brake_prob if self.is_human else human_brake_prob * (1 - random_braking_reduction)
        if self.speed > 0 and random.random() < prob:
            self.speed -= 1           

    def move(self):
        self.pos += self.speed


# Added random_braking_reduction 