"""Printable lab-practical code, pseudocode, and procedure steps.

Each public function prints the corresponding material from the supplied
Foundations of AI and Machine Learning lab manual.
"""

def _show(title, source):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(source.strip())


_LINEAR = r"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Create dataset
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([35, 40, 50, 55, 65, 70, 78, 85])

# Step 3: Visualize the data
plt.scatter(x, y)
plt.xlabel("Hours of Study")
plt.ylabel("Test Scores")
plt.title("Hours of study vs test scores")
plt.show()

# Step 4: Create the model
model = LinearRegression()

# Step 5: Train the model
model.fit(x, y)

# Step 6: Display what model learned
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 7: Make a prediction
prediction = model.predict([[9]])
print("Predicted marks for 9 hours:", prediction[0])

# Step 8: Plot Regression line
plt.scatter(x, y, label="Actual Data")
plt.plot(x, model.predict(x), label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.legend()
plt.show()
"""

_LOGISTIC = r"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Step 2: Create dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Step 3: Visualize the data
plt.scatter(X, y)
plt.xlabel("Hours Studied")
plt.ylabel("Result (0 = Fail, 1 = Pass)")
plt.title("Study Hours vs Result")
plt.show()

# Step 4: Create the model
model = LogisticRegression()

# Step 5: Train the model
model.fit(X, y)

# Step 6: Display what model learned
print("Coefficient:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])

# Step 7: Make a prediction
prediction = model.predict([[4.5]])
print("Predicted result for 4.5 hours:", prediction[0])
if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")

# Step 8: Get prediction probability
probability = model.predict_proba([[4.5]])
print("Probability of Fail:", probability[0][0])
print("Probability of Pass:", probability[0][1])

# Step 9: Plot classification curve
X_test = np.linspace(1, 8, 100).reshape(-1, 1)
probabilities = model.predict_proba(X_test)[:, 1]
plt.scatter(X, y, label="Actual Data")
plt.plot(X_test, probabilities, label="Classification Curve")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression")
plt.legend()
plt.show()
"""

_CLUSTERS = r"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Step 2: Load the dataset
iris = load_iris()
X = iris.data

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Step 4: Get cluster labels
labels = kmeans.labels_
print("Cluster Labels:")
print(labels)

# Step 5: Display cluster centers
print("\\nCluster Centers:")
print(kmeans.cluster_centers_)

# Step 6: Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 7: Visualize the clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="viridis",
    s=50
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering on Iris Dataset")
plt.show()

# Step 8: Display inertia
print("\\nInertia:", kmeans.inertia_)
"""

_ANN = r"""
import numpy as np

# Step 1: Create the XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# Step 2: Define the activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Step 3: Initialize weights and biases
np.random.seed(42)
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))

learning_rate = 0.5
epochs = 10000

# Step 4: Train the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(final_input)

    # Calculate error
    error = y - predicted_output

    # Backpropagation
    output_delta = error * sigmoid_derivative(predicted_output)
    hidden_delta = np.dot(output_delta, W2.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += learning_rate * np.dot(hidden_output.T, output_delta)
    b2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    W1 += learning_rate * np.dot(X.T, hidden_delta)
    b1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

# Step 5: Display predictions
print("Predictions after training:")
print(np.round(predicted_output, 3))
print("\\nActual outputs:")
print(y)
print("\\nPredicted classes:")
print((predicted_output >= 0.5).astype(int))
"""

_ACTIVATION = r"""
import numpy as np
import matplotlib.pyplot as plt

# Input values
x = np.linspace(-10, 10, 100)

# 1. Sigmoid Function: f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 2. ReLU Function: f(x) = max(0, x)
def relu(x):
    return np.maximum(0, x)

# 3. Tanh Function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Calculate outputs
y1 = sigmoid(x)
y2 = relu(x)
y3 = tanh(x)

# Plot the functions
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, y1)
plt.title("Sigmoid: 1 / (1 + e^(-x))")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(x, y2)
plt.title("ReLU: max(0, x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(x, y3)
plt.title("Tanh: (e^x - e^(-x)) / (e^x + e^(-x))")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()

plt.tight_layout()
plt.show()

# Analysis
print("ANALYSIS")
print("Sigmoid: Output range is 0 to 1.")
print("ReLU: Output is 0 for x < 0 and x for x >= 0.")
print("Tanh: Output range is -1 to 1.")
"""

_VACUUM = r"""
# Step 1: Define the environment
environment = {
    "A": "Dirty",
    "B": "Dirty"
}

# Step 2: Define the initial state
current_location = "A"

# Step 3: Define the goal state
goal_state = {
    "A": "Clean",
    "B": "Clean"
}

# Step 4: Define the vacuum cleaner agent
def vacuum_agent(location, environment):
    print("\\nInitial Environment:", environment)
    print("Initial Location:", location)

    # Continue until the goal state is achieved
    while environment != goal_state:
        # Perceive the current room
        percept = environment[location]
        print("\\nCurrent Location:", location)
        print("Percept:", percept)

        # Choose action based on percept
        if percept == "Dirty":
            action = "Suck"
            environment[location] = "Clean"
            print("Action:", action)
        else:
            if location == "A":
                action = "Right"
                location = "B"
            else:
                action = "Left"
                location = "A"
            print("Action:", action)

        print("Updated Environment:", environment)

    print("\\nGoal State Achieved!")
    print("Final Environment:", environment)
    print("Both rooms are clean.")
    print("Problem Solved Successfully.")

# Step 5: Execute the agent
vacuum_agent(current_location, environment)
"""

_TRAFFIC = r"""
# Step 1: Define the environment
traffic_environment = {
    "North": "High",
    "South": "Low",
    "East": "Medium",
    "West": "Low"
}

# Step 2: Define the rational agent
def traffic_signal_agent(environment):
    print("Traffic Signal Agent Started")
    print("\\nTraffic Environment:")
    for direction, density in environment.items():
        print(direction, ":", density)

    print("\\nAgent Decisions:")

    # Step 3: Select the best action based on traffic density
    for direction, density in environment.items():
        if density == "High":
            green_time = 60
        elif density == "Medium":
            green_time = 40
        else:
            green_time = 20

        print("\\nDirection:", direction)
        print("Traffic Density:", density)
        print("Action: Green Signal")
        print("Green Signal Duration:", green_time, "seconds")

    print("\\nTraffic management completed.")
    print("Rational Agent Stopped.")

# Step 4: Execute the agent
traffic_signal_agent(traffic_environment)
"""

_GENETIC = r"""
import random
import matplotlib.pyplot as plt

# 1. Parameters
BITS = 5
POPULATION_SIZE = 10
GENERATIONS = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05

# 2. Create a random chromosome
def create_chromosome():
    return [random.randint(0, 1) for _ in range(BITS)]

# 3. Convert a binary chromosome to decimal
def decode(chromosome):
    binary = "".join(map(str, chromosome))
    return int(binary, 2)

# 4. Fitness function: maximize f(x) = x^2
def fitness(chromosome):
    x = decode(chromosome)
    return x ** 2

# 5. Tournament selection
def select(population):
    candidate1 = random.choice(population)
    candidate2 = random.choice(population)
    if fitness(candidate1) > fitness(candidate2):
        return candidate1.copy()
    return candidate2.copy()

# 6. One-point crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, BITS - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1.copy(), parent2.copy()

# 7. Mutation
def mutate(chromosome):
    for i in range(BITS):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# 8. Initialize the population
population = [create_chromosome() for _ in range(POPULATION_SIZE)]
best_fitness_history = []

# 9. Genetic algorithm main loop
for generation in range(GENERATIONS):
    best_chromosome = max(population, key=fitness)
    best_x = decode(best_chromosome)
    best_score = fitness(best_chromosome)
    best_fitness_history.append(best_score)

    print(
        f"Generation {generation + 1}: "
        f"Best x = {best_x}, Fitness = {best_score}"
    )

    new_population = []
    while len(new_population) < POPULATION_SIZE:
        parent1 = select(population)
        parent2 = select(population)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.append(child1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    population = new_population

# 10. Find the best solution in the final population
best_chromosome = max(population, key=fitness)
best_x = decode(best_chromosome)
best_score = fitness(best_chromosome)

# 11. Display the final result
print("\\n========== FINAL RESULT ==========")
print("Best Chromosome:", "".join(map(str, best_chromosome)))
print("Best Value of x:", best_x)
print("Maximum Fitness:", best_score)
print("Theoretical Maximum Fitness:", 31 ** 2)

# 12. Plot the convergence graph
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, GENERATIONS + 1),
    best_fitness_history,
    marker="o"
)
plt.title("Genetic Algorithm Convergence")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()
"""

_PSEUDO = {
"linearRegression": """
START
  Import NumPy, Matplotlib, and LinearRegression
  Create study-hours and test-score data
  Plot the data points
  Create and train the linear regression model
  Display coefficient and intercept
  Predict marks for 9 study hours
  Plot the regression line
STOP
""",
"logisticRegression": """
START
  Import NumPy, Matplotlib, and LogisticRegression
  Create study-hours data and Pass/Fail labels
  Plot the data
  Create and train the logistic regression model
  Predict the result for 4.5 study hours
  Display Pass/Fail and both probabilities
  Plot the probability curve
STOP
""",
"Clusters": """
START
  Load the Iris dataset
  Create K-Means with 3 clusters
  Fit the model to the data
  Get cluster labels and cluster centers
  Apply PCA to reduce data to 2 dimensions
  Plot the clusters
  Display inertia
STOP
""",
"ANN": """
START
  Create the XOR input and output data
  Define sigmoid and its derivative
  Initialize weights, biases, learning rate, and epochs
  Repeat for 10,000 epochs:
    Perform forward propagation
    Calculate prediction error
    Perform backpropagation
    Update weights and biases
  Display predictions, actual outputs, and predicted classes
STOP
""",
"Activation": """
START
  Generate input values from -10 to 10
  Define Sigmoid, ReLU, and Tanh
  Calculate each function's output
  Plot the three functions in separate subplots
  Print the output-range analysis
STOP
""",
"Vaccum": """
START
  Create rooms A and B as Dirty
  Set the vacuum's starting location to A
  Set the goal: both rooms Clean
  WHILE the goal is not reached:
    Check the current room
    IF it is Dirty, Suck and clean it
    ELSE move to the other room
    Display the action and environment
  Display that the goal is achieved
STOP
""",
"Traffic": """
START
  Create traffic density for North, South, East, and West
  For each direction:
    IF density is High, set green time to 60 seconds
    ELSE IF density is Medium, set green time to 40 seconds
    ELSE set green time to 20 seconds
    Display direction, density, and green time
  Display that traffic management is completed
STOP
""",
"Genetic": """
START
  Set chromosome length, population, generations,
  crossover rate, and mutation rate
  Create random 5-bit chromosomes
  Decode each chromosome into a decimal value
  Calculate fitness as x squared
  Repeat for 30 generations:
    Select parents using tournament selection
    Apply one-point crossover
    Apply mutation
    Build the next population
    Display the best solution
  Display the final best chromosome, x, and fitness
  Plot the fitness convergence graph
STOP
"""
}

_STEPS = {
"linearRegression": """
1. Import NumPy, Matplotlib, and Scikit-learn.
2. Create study-hours and test-scores dataset.
3. Visualize the dataset using a scatter plot.
4. Create a LinearRegression model.
5. Train the model using fit().
6. Display the coefficient and intercept.
7. Predict marks for 9 hours of study.
8. Plot the regression line with the actual data.
""",
"logisticRegression": """
1. Import NumPy, Matplotlib, and Scikit-learn.
2. Create the classification dataset (0 = Fail, 1 = Pass).
3. Visualize the dataset.
4. Create and train a LogisticRegression model.
5. Predict the result for 4.5 hours of study.
6. Calculate the probability of failing and passing.
7. Visualize the classification curve.
""",
"Clusters": """
1. Import NumPy, Matplotlib, and Scikit-learn.
2. Load the Iris dataset using load_iris().
3. Extract the dataset features.
4. Create a K-Means model with 3 clusters.
5. Train the model using fit().
6. Obtain cluster labels and cluster centers.
7. Apply PCA to reduce dimensions to 2.
8. Visualize the clusters using a scatter plot.
9. Display the inertia value.
""",
"ANN": """
1. Import NumPy.
2. Create the XOR dataset.
3. Define sigmoid and its derivative.
4. Initialize weights and biases.
5. Set the learning rate and number of epochs.
6. Perform forward propagation.
7. Calculate the error.
8. Apply backpropagation.
9. Update weights and biases.
10. Repeat training for 10,000 epochs.
11. Display predicted outputs, actual outputs, and classes.
""",
"Activation": """
1. Import NumPy and Matplotlib.
2. Generate input values from -10 to 10.
3. Define Sigmoid, ReLU, and Tanh.
4. Calculate outputs for all three functions.
5. Create separate subplots for each function.
6. Add titles, labels, legends, and grid lines.
7. Display all three graphs.
8. Analyze and print their output ranges and behavior.
""",
"Vaccum": """
1. Define rooms A and B as Dirty.
2. Set the vacuum cleaner's initial location to Room A.
3. Define the goal state with both rooms Clean.
4. Create the vacuum cleaner agent function.
5. Perceive the current room.
6. If Dirty, perform Suck; if Clean, move to the other room.
7. Repeat until both rooms are Clean.
8. Display actions and the updated environment.
9. Display the final environment and goal confirmation.
""",
"Traffic": """
1. Define the traffic environment with North, South, East, and West.
2. Assign High, Medium, or Low density to each direction.
3. Define the rational traffic signal agent.
4. Display the traffic environment and densities.
5. Apply decision logic based on traffic density.
6. Assign 60 seconds for High, 40 for Medium, and 20 for Low.
7. Display each direction, density, and green duration.
8. Execute the agent and display completion status.
""",
"Genetic": """
1. Import Random and Matplotlib.
2. Set population size, generations, crossover rate, and mutation rate.
3. Create random binary chromosomes of 5 bits.
4. Convert chromosomes to decimal values.
5. Define fitness to maximize f(x) = x^2.
6. Implement tournament selection.
7. Apply one-point crossover.
8. Apply mutation.
9. Initialize the population.
10. Run the algorithm for 30 generations.
11. Display the best chromosome, x, and fitness per generation.
12. Identify the final best solution.
13. Plot the convergence graph.
"""
}

def linearRegression():
    """Print the Practical 1 linear-regression code."""
    _show("PRACTICAL 1A - LINEAR REGRESSION", _LINEAR)

def logisticRegression():
    """Print the Practical 1 logistic-regression code."""
    _show("PRACTICAL 1B - LOGISTIC REGRESSION", _LOGISTIC)

def Clusters():
    """Print the Practical 2 K-Means clustering code."""
    _show("PRACTICAL 2 - K-MEANS CLUSTERING", _CLUSTERS)

def ANN():
    """Print the Practical 3 ANN-from-scratch code."""
    _show("PRACTICAL 3 - ARTIFICIAL NEURAL NETWORK", _ANN)

def Activation():
    """Print the Practical 4 activation-functions code."""
    _show("PRACTICAL 4 - ACTIVATION FUNCTIONS", _ACTIVATION)

def Vaccum():
    """Print the Practical 5 vacuum-cleaner-agent code."""
    _show("PRACTICAL 5 - VACUUM CLEANER AGENT", _VACUUM)

def Traffic():
    """Print the Practical 6 traffic rational-agent code."""
    _show("PRACTICAL 6 - TRAFFIC SIGNAL AGENT", _TRAFFIC)

def Genetic():
    """Print the Practical 7 genetic-algorithm code."""
    _show("PRACTICAL 7 - GENETIC ALGORITHM", _GENETIC)

def _pseudo(name):
    _show(f"{name} - PSEUDOCODE", _PSEUDO[name])

def _steps(name):
    _show(f"{name} - PROCEDURE STEPS", _STEPS[name])

def linearRegression_pseudo():
    """Print simple pseudocode for linear regression."""
    _pseudo("linearRegression")

def linearRegression_steps():
    """Print lab-manual steps for linear regression."""
    _steps("linearRegression")

def logisticRegression_pseudo():
    """Print simple pseudocode for logistic regression."""
    _pseudo("logisticRegression")

def logisticRegression_steps():
    """Print lab-manual steps for logistic regression."""
    _steps("logisticRegression")

def Clusters_pseudo():
    """Print simple pseudocode for K-Means clustering."""
    _pseudo("Clusters")

def Clusters_steps():
    """Print lab-manual steps for K-Means clustering."""
    _steps("Clusters")

def ANN_pseudo():
    """Print simple pseudocode for the ANN practical."""
    _pseudo("ANN")

def ANN_steps():
    """Print lab-manual steps for the ANN practical."""
    _steps("ANN")

def Activation_pseudo():
    """Print simple pseudocode for activation functions."""
    _pseudo("Activation")

def Activation_steps():
    """Print lab-manual steps for activation functions."""
    _steps("Activation")

def Vaccum_pseudo():
    """Print simple pseudocode for the vacuum cleaner practical."""
    _pseudo("Vaccum")

def Vaccum_steps():
    """Print lab-manual steps for the vacuum cleaner practical."""
    _steps("Vaccum")

def Traffic_pseudo():
    """Print simple pseudocode for the traffic-agent practical."""
    _pseudo("Traffic")

def Traffic_steps():
    """Print lab-manual steps for the traffic-agent practical."""
    _steps("Traffic")

def Genetic_pseudo():
    """Print simple pseudocode for the genetic-algorithm practical."""
    _pseudo("Genetic")

def Genetic_steps():
    """Print lab-manual steps for the genetic-algorithm practical."""
    _steps("Genetic")


# ============================================================
# PRACTICAL 8 - CONSTRAINT-BASED OPTIMIZATION
# ============================================================

_OPTIMIZATION = r"""
import numpy as np
from scipy.optimize import linprog

# Profit per unit of each product (A, B, C)
profit = [40, 50, 60]

# Objective function (negative for maximization)
c = [-p for p in profit]

# Resource constraints
# Each row represents Labour, Raw Material and Machine Hours
A = [
    [2, 3, 4],  # Labour
    [3, 2, 1],  # Raw Material
    [1, 2, 2]   # Machine Hours
]

# Available resources
b = [100, 80, 60]

# Non-negative production quantities
bounds = [(0, None), (0, None), (0, None)]

# Solve the optimization problem
result = linprog(
    c,
    A_ub=A,
    b_ub=b,
    bounds=bounds,
    method='highs'
)

# Resource names
resource_names = ["Labour", "Raw Material", "Machine Hours"]

# Display results
if result.success:
    print("Optimal Solution Found!\n")

    print("Production Quantity:")
    print("Product A:", round(result.x[0], 2))
    print("Product B:", round(result.x[1], 2))
    print("Product C:", round(result.x[2], 2))

    print("\nMaximum Total Profit:", round(-result.fun, 2))

    print("\nResource Utilization:")
    resources_used = np.dot(A, result.x)

    for i in range(len(b)):
        print(
            resource_names[i],
            "Used:", round(resources_used[i], 2),
            "| Available:", b[i]
        )
else:
    print("No optimal solution found.")
"""


def Optimization():
    """Print Practical 8 constraint-based optimization code."""
    _show(
        "PRACTICAL 8 - CONSTRAINT-BASED OPTIMIZATION",
        _OPTIMIZATION
    )


_OPTIMIZATION_PSEUDO = """
START
  Define profit for products A, B, and C
  Define the objective function to maximize profit
  Define labour, raw material, and machine-hour constraints
  Set available resource quantities
  Set non-negative production bounds
  Solve the problem using Linear Programming
  IF an optimal solution is found:
    Display production quantities
    Display maximum total profit
    Calculate resource utilization
    Display resource usage and availability
  ELSE:
    Display no optimal solution found
STOP
"""


_OPTIMIZATION_STEPS = """
1. Open Google Colab and create a new Python notebook.
2. Import NumPy and linprog from the SciPy library.
3. Define the profit per unit for products A, B, and C.
4. Define the objective function to maximize total profit.
5. Specify resource constraints for labour, raw materials, and machine hours.
6. Define the available quantity of each resource.
7. Set non-negative bounds for the production quantities.
8. Use the linprog() function to solve the optimization problem.
9. Check whether an optimal solution is found.
10. Display the optimal production quantities and maximum total profit.
11. Calculate and display resource utilization for each resource.
"""


def Optimization_pseudo():
    """Print simple pseudocode for constraint-based optimization."""
    _show("PRACTICAL 8 - PSEUDOCODE", _OPTIMIZATION_PSEUDO)


def Optimization_steps():
    """Print procedure steps for constraint-based optimization."""
    _show("PRACTICAL 8 - PROCEDURE STEPS", _OPTIMIZATION_STEPS)


def getpract():
    """Print all available practical, pseudocode, and steps function names."""
    names = [
        "linearRegression", "linearRegression_pseudo", "linearRegression_steps",
        "logisticRegression", "logisticRegression_pseudo", "logisticRegression_steps",
        "Clusters", "Clusters_pseudo", "Clusters_steps",
        "ANN", "ANN_pseudo", "ANN_steps",
        "Activation", "Activation_pseudo", "Activation_steps",
        "Vaccum", "Vaccum_pseudo", "Vaccum_steps",
        "Traffic", "Traffic_pseudo", "Traffic_steps",
        "Genetic", "Genetic_pseudo", "Genetic_steps",
        "getpract",
    ]
    print("Available mytools functions:")
    for name in names:
        print(name)
