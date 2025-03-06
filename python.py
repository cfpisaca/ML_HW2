import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -------------------------------
# Data Preparation and NN Setup
# -------------------------------

# Load the dataset (adjust the file path as needed)
df = pd.read_csv("alzheimers_data.csv")

# Specify the 10 features and the target (adjust column names as needed)
features = ['Age', 'Gender', 'Education Level', 'BMI', 'Physical Activity Level',
            'Smoking Status', 'Diabetes', 'Hypertension', 'Cholesterol Level', 'Family History of Alzheimer’s']
target = 'Alzheimer’s Diagnosis'

# Encode categorical features if needed
categorical_cols = ['Gender', 'Physical Activity Level', 'Smoking Status', 'Diabetes', 
                    'Hypertension', 'Cholesterol Level', 'Family History of Alzheimer’s']
for col in categorical_cols:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# Separate features and target
X = df[features].values
y = df[target].apply(lambda x: 1 if x=='Yes' else 0).values

# Standardize numeric features (applied to all features for simplicity)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a simple 10-20-1 neural network using Keras
model = Sequential()
model.add(Dense(20, input_dim=10, activation='relu', name='hidden'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

print("Training baseline NN...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print("Baseline NN training accuracy:", train_acc)

# Extract the first layer weights (shape: (10,20)) and flatten them into a vector (200 weights)
W_input_hidden, b_hidden = model.get_layer('hidden').get_weights()
W_vec = W_input_hidden.flatten()

# -------------------------------
# Genetic Algorithm Setup
# -------------------------------

# GA parameters
bit_length = len(W_vec)  # 200
num_masked = bit_length // 2  # 100 bits must be 0
pop_size = 2 * bit_length  # population size of 400
crossover_prob = 0.9
mutation_rate = 0.05  # increased mutation rate from 0.01 to 0.05
num_generations = 50

# Helper: Create an individual mask with exactly num_masked zeros
def create_individual():
    individual = np.ones(bit_length, dtype=int)
    zero_indices = np.random.choice(bit_length, num_masked, replace=False)
    individual[zero_indices] = 0
    return individual

# Helper: Repair an individual to ensure exactly num_masked zeros
def repair(ind):
    ones = np.sum(ind)
    target_ones = bit_length - num_masked
    if ones > target_ones:  # too many ones: flip some ones to zeros
        diff = ones - target_ones
        one_indices = np.where(ind == 1)[0]
        flip = np.random.choice(one_indices, diff, replace=False)
        ind[flip] = 0
        print(f"Repair: Flipped {diff} ones to zeros.")
    elif ones < target_ones:  # too few ones: flip some zeros to ones
        diff = target_ones - ones
        zero_indices = np.where(ind == 0)[0]
        flip = np.random.choice(zero_indices, diff, replace=False)
        ind[flip] = 1
        print(f"Repair: Flipped {diff} zeros to ones.")
    return ind

# Fitness function: Apply mask to NN weights, evaluate accuracy on training set
def evaluate_fitness(mask):
    masked_W = (W_vec * mask).reshape(W_input_hidden.shape)
    orig_weights = model.get_layer('hidden').get_weights()[0].copy()
    model.get_layer('hidden').set_weights([masked_W, b_hidden])
    loss, acc = model.evaluate(X_train, y_train, verbose=0)
    # Restore original weights
    model.get_layer('hidden').set_weights([orig_weights, b_hidden])
    return acc

# Crossover: Two-point or uniform (equal chance)
def crossover(parent1, parent2):
    child1, child2 = parent1.copy(), parent2.copy()
    if random.random() < 0.5:
        # Two-point crossover
        points = sorted(random.sample(range(bit_length), 2))
        child1[points[0]:points[1]] = parent2[points[0]:points[1]]
        child2[points[0]:points[1]] = parent1[points[0]:points[1]]
        print(f"Crossover: Two-point between indices {points[0]} and {points[1]}.")
    else:
        # Uniform crossover
        for i in range(bit_length):
            if random.random() < 0.5:
                child1[i], child2[i] = parent2[i], parent1[i]
        print("Crossover: Uniform crossover executed.")
    return repair(child1), repair(child2)

# Mutation: Flip each bit with probability mutation_rate
def mutate(ind):
    for i in range(bit_length):
        if random.random() < mutation_rate:
            ind[i] = 1 - ind[i]
    print("Mutation: Mutation applied.")
    return repair(ind)

# Selection Methods
def fitness_proportionate(population, fitnesses):
    total_fit = sum(fitnesses)
    pick = random.uniform(0, total_fit)
    current = 0
    for ind, fit in zip(population, fitnesses):
        current += fit
        if current > pick:
            return ind
    return population[-1]

def tournament_selection(population, fitnesses, tournament_size=10):
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

# Helper: Compute average Hamming distance for a sample of individuals to assess diversity
def compute_diversity(population, sample_size=10):
    sample = random.sample(population, sample_size)
    distances = []
    for i in range(len(sample)):
        for j in range(i+1, len(sample)):
            distances.append(np.sum(sample[i] != sample[j]))
    avg_distance = np.mean(distances) if distances else 0
    return avg_distance

# Initialize population
population = [create_individual() for _ in range(pop_size)]
print("Initial population created.")

# -------------------------------
# Run the GA
# -------------------------------

def run_ga(selection_method):
    best_individual = None
    best_fitness = -np.inf
    current_pop = population.copy()
    history = []
    stagnation_counter = 0

    for gen in range(num_generations):
        print("\n----------------------------------")
        print(f"Generation {gen+1} starting...")
        # Evaluate fitness for each individual with detailed prints
        fitnesses = []
        for idx, ind in enumerate(current_pop):
            fit = evaluate_fitness(ind)
            fitnesses.append(fit)
            if idx < 3:  # Print fitness for first few individuals for debug
                print(f"Individual {idx} fitness: {fit:.4f}")

        gen_best_index = np.argmax(fitnesses)
        gen_best = current_pop[gen_best_index].copy()
        current_best_fitness = fitnesses[gen_best_index]
        print(f"Generation {gen+1}: Best Fitness = {current_best_fitness:.4f}")

        avg_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        print(f"Generation {gen+1}: Average Fitness = {avg_fitness:.4f}, Std Dev = {std_fitness:.4f}")
        
        diversity = compute_diversity(current_pop)
        print(f"Generation {gen+1}: Population Diversity (avg Hamming distance) = {diversity:.2f}")

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = gen_best.copy()
            stagnation_counter = 0
            print(f"New overall best found: {best_fitness:.4f}")
        else:
            stagnation_counter += 1
            print(f"No improvement in this generation. Stagnation counter: {stagnation_counter}")

        # Stagnation handling: reinitialize 10% of population if stagnation for 5 generations
        if stagnation_counter >= 5:
            num_reinit = int(0.1 * pop_size)
            print(f"Stagnation detected. Reinitializing {num_reinit} individuals to boost diversity.")
            for _ in range(num_reinit):
                idx = random.randint(0, pop_size-1)
                current_pop[idx] = create_individual()
            stagnation_counter = 0

        new_pop = [gen_best]  # weak elitism: carry the best individual

        # Create new individuals until population is replenished
        while len(new_pop) < pop_size:
            # Parent selection using the given method
            if selection_method == 'fitness':
                parent1 = fitness_proportionate(current_pop, fitnesses)
                parent2 = fitness_proportionate(current_pop, fitnesses)
                print("Selected two parents using fitness proportionate selection.")
            elif selection_method == 'tournament':
                parent1 = tournament_selection(current_pop, fitnesses, tournament_size=10)
                parent2 = tournament_selection(current_pop, fitnesses, tournament_size=10)
                print("Selected two parents using tournament selection.")
            else:
                raise ValueError("Unknown selection method")

            # Crossover
            if random.random() < crossover_prob:
                child1, child2 = crossover(parent1, parent2)
                print("Crossover performed.")
            else:
                child1, child2 = parent1.copy(), parent2.copy()
                print("No crossover; children copied directly from parents.")

            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            print("Children mutated.")

            new_pop.extend([child1, child2])
        
        current_pop = new_pop[:pop_size]
        history.append(best_fitness)
        print(f"Generation {gen+1} complete. Best Fitness so far: {best_fitness:.4f}")

    return best_individual, best_fitness, history

# -------------------------------
# Run GA with two selection methods
# -------------------------------

print("\n=== Running GA with Fitness Proportionate Selection ===")
best_mask_fp, best_fitness_fp, history_fp = run_ga('fitness')

print("\n=== Running GA with Tournament Selection ===")
best_mask_tourn, best_fitness_tourn, history_tourn = run_ga('tournament')

# -------------------------------
# Evaluate the Best GA Individual
# -------------------------------

def evaluate_mask_on_data(mask, X, y, dataset_name="Data"):
    masked_W = (W_vec * mask).reshape(W_input_hidden.shape)
    orig_weights = model.get_layer('hidden').get_weights()[0].copy()
    model.get_layer('hidden').set_weights([masked_W, b_hidden])
    # Get predictions
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix on {dataset_name}:\n{cm}")
    # Restore original weights
    model.get_layer('hidden').set_weights([orig_weights, b_hidden])
    return cm

print("\n=== Evaluating Best GA Individual (Tournament Selection) on Training and Test Sets ===")
cm_train = evaluate_mask_on_data(best_mask_tourn, X_train, y_train, dataset_name="Training Data")
cm_test  = evaluate_mask_on_data(best_mask_tourn, X_test, y_test, dataset_name="Test Data")

# -------------------------------
# Baseline: Highest-Magnitude Weights Unmasked
# -------------------------------

print("\n=== Evaluating Baseline (Highest-Magnitude Weights) ===")
sorted_indices = np.argsort(np.abs(W_vec))[::-1]
baseline_mask = np.zeros(bit_length, dtype=int)
# Unmask top (bit_length - num_masked) weights (i.e. 100 ones)
baseline_mask[sorted_indices[:bit_length - num_masked]] = 1

cm_train_baseline = evaluate_mask_on_data(baseline_mask, X_train, y_train, dataset_name="Training Data (Baseline)")
cm_test_baseline  = evaluate_mask_on_data(baseline_mask, X_test, y_test, dataset_name="Test Data (Baseline)")

print("\n=== Summary ===")
print(f"Best GA Fitness (Fitness Prop.) = {best_fitness_fp:.4f}")
print(f"Best GA Fitness (Tournament) = {best_fitness_tourn:.4f}")
