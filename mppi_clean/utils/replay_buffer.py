import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, timestep, state, control_sequence):
        """Store (state, control sequence) in buffer."""
        experience = (timestep, state, control_sequence)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size):
        """Randomly sample a batch from buffer."""
        return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else []
