import random

class QLearn():
    def __init__(self, actions, epsion, alpha, gamma) :
        self.q = {}
        self.epsion = epsion
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
    
    def get_q (self, state, action) :
        return self.q.get((state, action), 0.0)
    
    def learn_q (self, state, action, reward, delta_value) :
        '''
        Q(s, a) = (1 - alpha) * Q(s,a) + alpha * (reward(s,a) + gamma * max_Q(s'))   
        Q(s, a) = Q(s, a) + alpha * (reward(s, a) + gamma * max_Q(s') - Q(s,a))
        delta_value = reward(s, a) + gamma * max_Q(s') - Q(s,a)
        '''
        old_q = self.q.get((state, action), None)
        if old_q is None:
            self.q[(state, action)] = reward
        else :
            self.q[(state, action)] = old_q + self.alpha * (delta_value - old_q)
    
    def choose_action(self, state) :
        q = [self.get_q(state, i) for i in self.actions]
        max_q = max(q)
        
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            return action

        count_max_q = q.count(max_q)
        if count_max_q > 1 :
            best_action = [i for i in range(len(self.actions)) if q[i] == max_q]
            i = random.choice(best_action)
        else :
            i = q.index(max_q)
        
        action = self.actions[i]
        return action
    
    def learn(self, state1, action1, reward, state2) :
        max_next_q = max([self.get_q(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma * max_next_q)