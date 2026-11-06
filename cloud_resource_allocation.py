# ==============================================================================
# AI310 & CS361 Artificial Intelligence - Project 24
# Cloud Resource Allocation using Genetic Algorithm (GA) and Memetic Algorithm (MA)
# Faculty of Computing & Artificial Intelligence, Capital University
# ==============================================================================

import random
import time
import copy

# Import matplotlib for visualization
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Set random seed for reproducibility
random.seed(42)


# ==============================================================================
# 1. Cloud Environment Modeling (Tasks & Resources)
# ==============================================================================

class Task:
    def __init__(self, task_id, cpu_required, memory_required, execution_length):
        self.task_id = task_id
        self.cpu_required = cpu_required  # CPU units required
        self.memory_required = memory_required  # RAM required in GB
        self.execution_length = execution_length  # Task length in Million Instructions (MI)


class Resource:
    def __init__(self, res_id, mips, cost_per_mips, memory_capacity):
        self.res_id = res_id
        self.mips = mips  # Processing speed (MIPS)
        self.cost_per_mips = cost_per_mips  # Cost rate per MIPS
        self.memory_capacity = memory_capacity  # Total RAM capacity in GB


def generate_simulation_data(num_tasks=30, num_resources=5):
    tasks = []
    for i in range(num_tasks):
        cpu = random.randint(1, 4)
        mem = random.randint(1, 8)
        length = random.randint(100, 1000)
        tasks.append(Task(i, cpu, mem, length))

    resources = []
    for j in range(num_resources):
        mips = random.choice([50, 100, 150, 200, 250])
        cost = round(random.uniform(0.5, 2.5), 2)
        mem_cap = random.choice([16, 32, 64])
        resources.append(Resource(j, mips, cost, mem_cap))

    return tasks, resources


# ==============================================================================
# 2. Fitness Evaluation Function
# ==============================================================================

def evaluate_chromosome(chromosome, tasks, resources):
    num_resources = len(resources)
    resource_loads = [0] * num_resources
    total_cost = 0.0

    for task_id, res_id in enumerate(chromosome):
        task = tasks[task_id]
        res = resources[res_id]

        exec_time = task.execution_length / res.mips
        resource_loads[res_id] += exec_time
        total_cost += exec_time * res.cost_per_mips

    makespan = max(resource_loads)  # Total response time for all tasks

    w1, w2 = 0.6, 0.4
    fitness_score = (w1 * makespan) + (w2 * total_cost)

    return fitness_score, makespan, total_cost


# ==============================================================================
# 3. Baseline Strategy (Greedy Allocation)
# ==============================================================================

def baseline_greedy_allocation(tasks, resources):
    chromosome = [i % len(resources) for i in range(len(tasks))]
    fitness, makespan, cost = evaluate_chromosome(chromosome, tasks, resources)
    return chromosome, fitness, makespan, cost


# ==============================================================================
# 4. Genetic Algorithm (GA)
# ==============================================================================

class GeneticAlgorithm:
    def __init__(self, tasks, resources, pop_size=40, generations=50, crossover_rate=0.8, mutation_rate=0.1):
        self.tasks = tasks
        self.resources = resources
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_tasks = len(tasks)
        self.num_resources = len(resources)

    def initialize_population(self):
        return [[random.randint(0, self.num_resources - 1) for _ in range(self.num_tasks)] for _ in
                range(self.pop_size)]

    def selection(self, population, fitnesses):
        selected = []
        for _ in range(self.pop_size):
            i1, i2 = random.sample(range(self.pop_size), 2)
            selected.append(population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2])
        return selected

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.num_tasks - 1)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def mutate(self, individual):
        for i in range(self.num_tasks):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, self.num_resources - 1)
        return individual

    def run(self):
        population = self.initialize_population()
        history = []

        for gen in range(self.generations):
            fitnesses = [evaluate_chromosome(ind, self.tasks, self.resources)[0] for ind in population]
            history.append(min(fitnesses))

            selected_pop = self.selection(population, fitnesses)
            next_generation = []

            for i in range(0, self.pop_size, 2):
                p1, p2 = selected_pop[i], selected_pop[(i + 1) % self.pop_size]
                c1, c2 = self.crossover(p1, p2)
                next_generation.extend([self.mutate(c1), self.mutate(c2)])

            population = next_generation

        fitnesses = [evaluate_chromosome(ind, self.tasks, self.resources)[0] for ind in population]
        best_idx = fitnesses.index(min(fitnesses))
        best_solution = population[best_idx]
        best_fitness, best_makespan, best_cost = evaluate_chromosome(best_solution, self.tasks, self.resources)

        return best_solution, best_fitness, best_makespan, best_cost, history


# ==============================================================================
# 5. Memetic Algorithm (MA)
# ==============================================================================

class MemeticAlgorithm(GeneticAlgorithm):
    def local_search(self, individual):
        best_ind = copy.deepcopy(individual)
        best_fit, _, _ = evaluate_chromosome(best_ind, self.tasks, self.resources)

        for _ in range(10):
            task_to_shift = random.randint(0, self.num_tasks - 1)
            new_res = random.randint(0, self.num_resources - 1)

            candidate = copy.deepcopy(best_ind)
            candidate[task_to_shift] = new_res
            cand_fit, _, _ = evaluate_chromosome(candidate, self.tasks, self.resources)

            if cand_fit < best_fit:
                best_fit = cand_fit
                best_ind = candidate

        return best_ind

    def run(self):
        population = self.initialize_population()
        history = []

        for gen in range(self.generations):
            population = [self.local_search(ind) for ind in population]
            fitnesses = [evaluate_chromosome(ind, self.tasks, self.resources)[0] for ind in population]
            history.append(min(fitnesses))

            selected_pop = self.selection(population, fitnesses)
            next_generation = []

            for i in range(0, self.pop_size, 2):
                p1, p2 = selected_pop[i], selected_pop[(i + 1) % self.pop_size]
                c1, c2 = self.crossover(p1, p2)
                next_generation.extend([self.mutate(c1), self.mutate(c2)])

            population = next_generation

        fitnesses = [evaluate_chromosome(ind, self.tasks, self.resources)[0] for ind in population]
        best_idx = fitnesses.index(min(fitnesses))
        best_solution = population[best_idx]
        best_fitness, best_makespan, best_cost = evaluate_chromosome(best_solution, self.tasks, self.resources)

        return best_solution, best_fitness, best_makespan, best_cost, history


# ==============================================================================
# 6. Execution, Comparison & Plotting
# ==============================================================================

if __name__ == "__main__":
    print("=========================================================")
    print("--- Starting Cloud Resource Allocation Simulation ---")
    print("=========================================================")
    tasks, resources = generate_simulation_data(num_tasks=40, num_resources=6)

    # 1. Baseline Model
    _, base_fit, base_makespan, base_cost = baseline_greedy_allocation(tasks, resources)
    print(f"\n[Baseline Model]")
    print(f" -> Fitness Score: {base_fit:.2f}")
    print(f" -> Makespan (Time): {base_makespan:.2f}s")
    print(f" -> Total Cost: ${base_cost:.2f}")

    # 2. Genetic Algorithm
    ga = GeneticAlgorithm(tasks, resources, pop_size=40, generations=50)
    start_time = time.time()
    _, ga_fit, ga_makespan, ga_cost, ga_history = ga.run()
    ga_time = time.time() - start_time
    print(f"\n[Genetic Algorithm (GA)]")
    print(f" -> Fitness Score: {ga_fit:.2f}")
    print(f" -> Makespan (Time): {ga_makespan:.2f}s")
    print(f" -> Total Cost: ${ga_cost:.2f}")
    print(f" -> Execution Time: {ga_time:.2f}s")

    # 3. Memetic Algorithm
    ma = MemeticAlgorithm(tasks, resources, pop_size=40, generations=50)
    start_time = time.time()
    _, ma_fit, ma_makespan, ma_cost, ma_history = ma.run()
    ma_time = time.time() - start_time
    print(f"\n[Memetic Algorithm (MA)]")
    print(f" -> Fitness Score: {ma_fit:.2f}")
    print(f" -> Makespan (Time): {ma_makespan:.2f}s")
    print(f" -> Total Cost: ${ma_cost:.2f}")
    print(f" -> Execution Time: {ma_time:.2f}s")

    print("\n=========================================================")
    print("--- Summary Performance Comparison ---")
    print("=========================================================")
    improvement = ((base_fit - ma_fit) / base_fit) * 100
    print(f"MA Improvement over Baseline: {improvement:.2f}%\n")

    # 4. Generate & Save Plots for the Report
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 5))
        plt.plot(ga_history, label='Genetic Algorithm (GA)', color='blue', linewidth=2)
        plt.plot(ma_history, label='Memetic Algorithm (MA)', color='green', linewidth=2, linestyle='--')
        plt.axhline(y=base_fit, color='red', linestyle=':', label='Baseline Model')
        plt.xlabel('Generations', fontsize=12)
        plt.ylabel('Fitness Score (Cost & Makespan)', fontsize=12)
        plt.title('Performance Comparison: GA vs MA vs Baseline', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Save plot as image for the report PDF
        plt.savefig('cloud_allocation_comparison.png', dpi=300, bbox_inches='tight')
        print("[Success] Performance plot saved as 'cloud_allocation_comparison.png' ready for your report!")
        plt.show()
    else:
        print("[Notice] Matplotlib plotting skipped as library is not active.")