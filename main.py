import numpy as np
import tensorflow as tf

# Параметры поля и амеб
field_size = 10
initial_amoeba_energy = 50
energy_per_grass = 10


# Генерация случайной позиции на поле
def generate_random_position():
    return np.random.randint(0, field_size, size=2)


# Генерация начальной энергии для амебы
def generate_initial_energy():
    return np.random.randint(30, initial_amoeba_energy)


# Создание поля и распределение травы
field = np.zeros((field_size, field_size))
num_grass = np.random.randint(field_size, field_size * 2)
grass_positions = [generate_random_position() for _ in range(num_grass)]
for pos in grass_positions:
    field[pos[0], pos[1]] = 1


# Создание нейронной сети для выбора направления движения
def create_neural_network():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Инициализация амеб
class Amoeba:
    def __init__(self, position, energy):
        self.position = position
        self.energy = energy
        self.neural_network = create_neural_network()

    def get_state(self, grass_positions):
        distances = [np.linalg.norm(np.array(self.position) - np.array(grass_pos)) for grass_pos in grass_positions]
        return np.array([self.energy] + distances)

    def choose_movement_direction(self, grass_positions):
        state = self.get_state(grass_positions)
        action_probs = self.neural_network.predict(np.array([state]))[0]
        return np.random.choice(range(4), p=action_probs)


# Симуляция
num_amoebas = 5
amoebas = [Amoeba(generate_random_position(), generate_initial_energy()) for _ in range(num_amoebas)]

num_steps = 100
for step in range(num_steps):
    for amoeba in amoebas:
        grass_to_eat = []
        for pos in grass_positions:
            if np.array_equal(pos, amoeba.position):
                grass_to_eat.append(pos)

        if grass_to_eat:
            chosen_grass = grass_to_eat[
                np.argmax([np.linalg.norm(np.array(amoeba.position) - np.array(pos)) for pos in grass_to_eat])]
            amoeba.energy += energy_per_grass
            grass_positions.remove(chosen_grass)

        movement_direction = amoeba.choose_movement_direction(grass_positions)

        if movement_direction == 0 and amoeba.position[0] > 0:
            amoeba.position[0] -= 1
        elif movement_direction == 1 and amoeba.position[0] < field_size - 1:
            amoeba.position[0] += 1
        elif movement_direction == 2 and amoeba.position[1] > 0:
            amoeba.position[1] -= 1
        elif movement_direction == 3 and amoeba.position[1] < field_size - 1:
            amoeba.position[1] += 1

        amoeba.energy -= 1

        if amoeba.energy <= 0:
            amoebas.remove(amoeba)

    if step % 10 == 0:
        print(f"Step {step}: {len(amoebas)} amoebas remaining")

print("Simulation completed.")
