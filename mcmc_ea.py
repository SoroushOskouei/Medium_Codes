import numpy as np
import random
import matplotlib.pyplot as plt

# Problem Setup
N = 70
start = (0, 0)
goal = (N-1, N-1)

# Generate random terrain costs:
terrain = np.random.uniform(1.0, 5.0, (N, N))

# Introduce local diagonal streaks with lower costs
num_streaks = 90
for _ in range(num_streaks):
    start_i = random.randint(0, N-5)
    start_j = random.randint(0, N-5)
    length = random.randint(5, 10)
    for k in range(length):
        if start_i+k < N and start_j+k < N:
            terrain[start_i+k, start_j+k] = np.random.uniform(0.5, 1.5)

# EA Parameters
population_size = 20
max_generations = 1500
mutation_rate = 0.3
random.seed(42)

def random_path():
    """Generate a path from start to goal."""
    path = [start]
    x, y = start
    while (x, y) != goal:
        moves = []
        if x < N-1: moves.append((x+1, y))
        if y < N-1: moves.append((x, y+1))
        if x < N-1 and y < N-1: moves.append((x+1, y+1))
        (x_next, y_next) = random.choice(moves)
        path.append((x_next, y_next))
        x, y = x_next, y_next
    return path

def path_cost(path):
    """Sum of the terrain costs for the given path."""
    return sum(terrain[x, y] for x, y in path)

def path_fitness(path):
    """
    Compute the fitness of a path using both cost and directionality.
    Penalize paths that are longer than the minimal possible steps.
    """
    cost = path_cost(path)
    length = len(path)
    minimal_steps = (N - 1)
    length_penalty = (length - minimal_steps)
    alpha = 0.5
    fitness_value = cost + alpha * length_penalty
    return fitness_value

def mutate(path):
    """Mutate by replacing a segment and regrowing the path to ensure it still reaches the goal."""
    if random.random() < mutation_rate:
        idx = random.randint(1, len(path)-2)
        x, y = path[idx]
        moves = []
        if x < N-1: moves.append((x+1, y))
        if y < N-1: moves.append((x, y+1))
        if x < N-1 and y < N-1: moves.append((x+1, y+1))
        if moves:
            path = path[:idx] + [random.choice(moves)]
            # Rebuild path to the goal from here
            x_current, y_current = path[-1]
            while (x_current, y_current) != goal:
                moves2 = []
                if x_current < N-1: moves2.append((x_current+1, y_current))
                if y_current < N-1: moves2.append((x, y_current+1))
                if x_current < N-1 and y_current < N-1: moves2.append((x_current+1, y_current+1))
                (x_current, y_current) = random.choice(moves2)
                path.append((x_current, y_current))
    return path

def crossover(p1, p2):
    """Simple crossover taking half from p1 and then rebuilding toward the goal."""
    cut = len(p1)//2
    prefix = p1[:cut]
    last = prefix[-1]
    x, y = last
    visited = set(prefix)
    remainder = []
    while (x, y) != goal:
        moves = []
        if x < N-1: moves.append((x+1, y))
        if y < N-1: moves.append((x, y+1))
        if x < N-1 and y < N-1: moves.append((x+1, y+1))

        # Try to follow p2 if possible
        possible_next = [m for m in moves if m in p2 and m not in visited]
        if not possible_next:
            possible_next = moves
        nxt = random.choice(possible_next)
        remainder.append(nxt)
        visited.add(nxt)
        x, y = nxt
    return prefix + remainder

def initialize_population():
    return [random_path() for _ in range(population_size)]

def select_parents(pop):
    """Tournament selection based on fitness."""
    selected = []
    for _ in range(population_size):
        a, b = random.sample(pop, 2)
        if path_fitness(a) < path_fitness(b):
            selected.append(a)
        else:
            selected.append(b)
    return selected

def prior_probability(path):
    """
    Prior probability: Favor paths that consistently reduce Manhattan distance.
    """
    current = start
    total_reduction = 0
    for nxt in path[1:]:
        old_distance = abs(current[0] - goal[0]) + abs(current[1] - goal[1])
        new_distance = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
        reduction = old_distance - new_distance
        if reduction > 0:
            total_reduction += reduction
        current = nxt
    return np.exp(total_reduction * 0.1)

def mcmc_selection(parents):
    """Metropolis-Hastings style MCMC to accept or reject offspring."""
    offspring = []
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[i+1] if i+1 < len(parents) else parents[i]
        
        child = mutate(crossover(p1, p2))
        
        f_child = path_fitness(child)
        f_p1 = path_fitness(p1)
        
        post_child = np.exp(-f_child) * prior_probability(child)
        post_parent = np.exp(-f_p1) * prior_probability(p1)
        
        alpha = min(1, post_child / post_parent)
        
        if random.random() < alpha:
            offspring.append(child)
        else:
            offspring.append(p1)
    return offspring

def run_ea(use_mcmc=False):
    pop = initialize_population()
    best_raw_costs = []
    for gen in range(max_generations):
        parents = select_parents(pop)
        
        if use_mcmc:
            pop = mcmc_selection(parents)
        else:
            offspring = []
            for i in range(0, len(parents), 2):
                p1 = parents[i]
                p2 = parents[i+1] if i+1 < len(parents) else parents[i]
                child = mutate(crossover(p1, p2))
                offspring.append(child)
            pop = offspring
        
        # Track best (lowest) raw cost in this generation
        costs = [path_cost(ind) for ind in pop]
        best_raw_costs.append(min(costs))
    
    best_ind = pop[np.argmin([path_cost(ind) for ind in pop])]
    return best_ind, best_raw_costs

# Run experiments and compare
best_no_mcmc, costs_no_mcmc = run_ea(use_mcmc=False)
best_mcmc, costs_mcmc = run_ea(use_mcmc=True)

# Visualization of the actual cumulative cost of the best path per generation
plt.figure(figsize=(10,5))
plt.plot(costs_no_mcmc, label='Standard EA')
plt.plot(costs_mcmc, label='EA with MCMC')
plt.xlabel('Generation')
plt.ylabel('Lowest Path Cost')
plt.title('Comparison of Standard EA vs. EA with MCMC (Lowest Cumulative Path Cost)')
plt.legend()
plt.show()

def plot_path(path, title):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(terrain, cmap='viridis', origin='lower')
    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    ax.plot(xs, ys, color='red', linewidth=2)
    ax.scatter([start[1], goal[1]], [start[0], goal[0]], color='white')
    ax.set_title(title)
    plt.show()

plot_path(best_no_mcmc, "Best Path (Standard EA)")
plot_path(best_mcmc, "Best Path (EA with MCMC)")
