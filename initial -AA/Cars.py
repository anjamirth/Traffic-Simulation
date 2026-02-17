import random
class Cars:
    def __init__(self, v_max, current_edge,human, lane=0, pos=0):
        self.speed = 0              
        self.v_max = v_max      
        self.edge = current_edge 
        self.lane = lane        
        self.pos = pos   
        self.isHuman = human
    def  accelarate(self):
        if self.speed < self.v_max:
            self.speed += 1
    def brake(self, gap):
        
        if self.speed > gap:
            self.speed = gap
    def randomise(self,p):
         if self.isHuman and self.speed > 0 and random.random() < p:
            self.speed -= 1
    def move(self):
        self.pos += self.speed
