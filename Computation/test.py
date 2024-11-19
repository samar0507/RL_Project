import traci
import numpy as np
import random
import matplotlib.pyplot as plt

# SUMO Configuration
sumo_binary = "sumo-gui"  # Use "sumo" for non-GUI
sumo_config_file = "C:/Users/souso/Documents/GitHub/RL_Project/Simulation/network.sumocfg"

# Q-Learning Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_phases = 3  # Adjust based on your traffic light setup
q_table = {}  # Q-table
total_rewards = []  # List to track total rewards for evaluation
steps_progress = []  # List to track progress of each simulation step
rewards_at_steps = []  # List to track rewards at each step

# Constants for the reward function
k1 = 2.0  # Increase the reward for throughput
k2 = 0.5  # Reduce the congestion penalty
k3 = 0.2  # Reduce the long phase penalty
Q_threshold = 5  # Adjust threshold to a smaller value
P_threshold = 10  # Ensure phase duration isn't excessively penalized

# Global Traffic Light ID (Assuming a single traffic light in this case)
tls_id = None


def get_state(tls_id):
    """Extract state representation."""
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    state = []
    for lane in lanes:
        vehicles = traci.lane.getLastStepVehicleNumber(lane)
        state.append(vehicles)
    return tuple(state)


def choose_action(state):
    """Choose an action using epsilon-greedy policy."""
    if state not in q_table:
        q_table[state] = np.zeros(num_phases)  # Adjust this to the correct number of phases
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_phases - 1)  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit


def get_reward(tls_id):
    """Calculate the reward (e.g., negative cumulative waiting time, throughput, penalties)."""
    lanes = traci.trafficlight.getControlledLanes(tls_id)

    # Calculate total waiting time
    total_waiting_time = 0
    for lane in lanes:
        total_waiting_time += traci.lane.getWaitingTime(lane)

    # Calculate vehicle throughput (number of vehicles passing)
    throughput = 0
    for lane in lanes:
        throughput += traci.lane.getLastStepVehicleNumber(lane)

    # Calculate queue length
    queue_length = sum([traci.lane.getLastStepOccupancy(lane) for lane in lanes])

    # Calculate phase duration
    phase_duration = traci.trafficlight.getPhaseDuration(tls_id)

    # Calculate rewards and penalties
    reward_waiting = -total_waiting_time  # Negative reward for waiting time
    reward_throughput = k1 * throughput  # Reward for vehicle throughput
    penalty_congestion = -k2 * max(0, queue_length - Q_threshold)  # Penalty for congestion
    penalty_long_phase = -k3 * max(0, phase_duration - P_threshold)  # Penalty for long phase

    # Total reward
    total_reward = reward_waiting + reward_throughput + penalty_congestion + penalty_long_phase
    return total_reward


def evaluate_model(tls_id):
    total_reward = 0
    total_waiting_time = 0
    total_vehicles = 0

    for step in range(500):  # Define the evaluation duration
        traci.simulationStep()  # Advance the simulation

        state = get_state(tls_id)
        action = choose_action(state)
        traci.trafficlight.setPhase(tls_id, action)

        for _ in range(10):
            traci.simulationStep()

        # Calculate waiting time and number of vehicles
        for lane in traci.trafficlight.getControlledLanes(tls_id):
            total_waiting_time += traci.lane.getWaitingTime(lane)
            total_vehicles += traci.lane.getLastStepVehicleNumber(lane)

        reward = get_reward(tls_id)
        total_reward += reward

    avg_waiting_time = total_waiting_time / total_vehicles if total_vehicles > 0 else 0
    avg_reward = total_reward / 500  # Average reward per step
    print(f"Average Waiting Time: {avg_waiting_time:.4f} seconds per vehicle")
    return total_reward, avg_waiting_time


# Start SUMO
traci.start([sumo_binary, "-c", sumo_config_file])

try:
    step = 0
    tls_id = traci.trafficlight.getIDList()[0]  # Assume a single traffic light

    while step < 1000:  # Define the simulation duration for training
        traci.simulationStep()  # Advance the simulation

        # Get current state
        state = get_state(tls_id)

        # Choose action (traffic light phase)
        action = choose_action(state)

        # Set the action (change traffic light phase)
        traci.trafficlight.setPhase(tls_id, action)

        # Wait for a few steps to allow the action to take effect
        for _ in range(10):
            traci.simulationStep()

        # Get reward and next state
        reward = get_reward(tls_id)
        next_state = get_state(tls_id)

        # Q-Learning Update
        if state not in q_table:
            q_table[state] = np.zeros(num_phases)
        if next_state not in q_table:
            q_table[next_state] = np.zeros(num_phases)
        q_table[state][action] = q_table[state][action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
        )

        # Track step progress and reward
        steps_progress.append(step)
        rewards_at_steps.append(reward)

        total_rewards.append(reward)
        step += 10

    # Evaluate the model after training
    avg_reward = evaluate_model(tls_id)
    print(f"Average reward during evaluation: {avg_reward}")

    # Plot the total rewards during training
    plt.plot(steps_progress, rewards_at_steps)
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Reward Progress During Training')
    plt.show()

finally:
    traci.close()
