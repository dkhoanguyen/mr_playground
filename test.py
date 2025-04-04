# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal

# # Environment and target setup
# env_size = 10  # Size of environment
# true_target_position = np.array([7, 3])  # Actual position of target

# # Robot parameters
# robot_position = np.array([2, 2])  # Initial robot position
# robot_step_size = 0.1  # Step size for random movement

# # # Belief parameters (Kalman Filter)
# # belief_mean = np.array([0.0, 0.0])  # Initial belief mean
# # # Initial covariance (uncertainty)
# # belief_cov = np.array([[10.0, 0.0], [0.0, 10.0]])

# belief_mean = np.array([2, 2])  # Randomized initial belief
# belief_cov = np.array([[env_size**2, 0.0], [0.0, env_size**2]])

# # Process noise (how much we trust our motion model)
# process_noise = np.array([[0.001, 0.0], [0.0, 0.001]])

# # Simulation parameters
# num_steps = 100

# # Gaussian measurement model where noise increases with distance


# def sensor_model(robot_pos, true_pos):
#     distance = np.linalg.norm(robot_pos - true_pos)
#     # Quadratic increase in noise level with distance
#     noise_level = np.abs(distance)
#     # Measurement noise covariance
#     measurement_noise = np.array([[noise_level, 0.0], [0.0, noise_level]])
#     measurement = true_pos + \
#         np.random.multivariate_normal([0, 0], measurement_noise)
#     return measurement, measurement_noise

# # Kalman Filter Update Step


# def update_belief_kalman(belief_mean, belief_cov, measurement, measurement_noise):
#     # Compute Kalman Gain
#     kalman_gain = belief_cov @ np.linalg.inv(belief_cov + measurement_noise)

#     # Update Belief
#     updated_mean = belief_mean + kalman_gain @ (measurement - belief_mean)
#     updated_cov = (np.eye(2) - kalman_gain) @ belief_cov + \
#         process_noise  # Include process noise

#     return updated_mean, updated_cov

# # Random walk for the robot


# def random_walk(robot_pos, step_size):
#     delta = np.random.uniform(-step_size, step_size, size=2)
#     new_pos = np.clip(robot_pos + delta, 0, env_size)  # Keep within bounds
#     return new_pos

# # Simple proportional controller to move robot towards the target
# def move_towards_target(robot_pos, target_pos, step_size):
#     direction = target_pos - robot_pos
#     if np.linalg.norm(direction) > 0:
#         direction = direction / np.linalg.norm(direction)  # Normalize direction
#     new_pos = robot_pos + step_size * direction  # Move in direction of target
#     return np.clip(new_pos, 0, env_size)  # Keep within bounds

# # Proportional controller to move robot towards the target
# def move_towards_target(robot_pos, target_pos, Kp, step_size):
#     direction = target_pos - robot_pos
#     velocity = Kp * direction  # Proportional control
#     velocity = np.clip(velocity, -step_size, step_size)  # Limit step size
#     new_pos = robot_pos + velocity  # Apply control
#     return np.clip(new_pos, 0, env_size)  # Keep within bounds

# # Ergodic controller for exploratory motion
# def ergodic_control(robot_pos, belief_mean, belief_cov, step_size):
#     control_gain = 0.5  # Gain factor for ergodic control
#     direction = belief_mean - robot_pos  # Move toward belief mean
#     uncertainty = np.trace(belief_cov)  # Use belief uncertainty to modulate motion
#     velocity = control_gain * direction * np.exp(-uncertainty)  # Reduce motion as belief becomes more certain
#     velocity = np.clip(velocity, -step_size, step_size)  # Limit step size
#     new_pos = robot_pos + velocity  # Apply control
#     return np.clip(new_pos, 0, env_size)  # Keep within bounds


# # Run the simulation
# robot_path = []
# belief_path = []

# plt.ion()
# fig, ax = plt.subplots()
# ax.set_xlim(0, env_size)
# ax.set_ylim(0, env_size)
# ax.set_xlabel("X Position")
# ax.set_ylabel("Y Position")
# plt.title("Robot Movement and Belief Distribution")

# for _ in range(num_steps):
#     # Visualization
#     ax.clear()
#     ax.set_xlim(0, env_size)
#     ax.set_ylim(0, env_size)

#     # Plot the robot movement
#     ax.plot(*zip(*robot_path), label='Robot Path',
#             linestyle='dashed', color='blue')
#     ax.scatter(*robot_position, color='blue', label='Robot', marker='o', s=80)

#     # Plot the true target location
#     ax.scatter(*true_target_position, color='red',
#                label='True Target', marker='x', s=100)

#     # Plot the estimated belief of target location
#     ax.scatter(*belief_mean, color='green',
#                label='Estimated Target', marker='s', s=80)

#     # Plot the belief distribution
#     grids = np.linspace(0, env_size, 100)
#     x, y = np.meshgrid(grids, grids)
#     pos = np.dstack((x, y))

#     try:
#         rv = multivariate_normal(belief_mean, belief_cov)
#         ax.contourf(x, y, rv.pdf(pos), cmap='Blues', alpha=0.6)
#     except np.linalg.LinAlgError:
#         pass  # Skip plotting if covariance matrix is singular

#     ax.legend()
#     # plt.axis("equal")
#     plt.pause(0.075)
#     # Move the robot randomly
#     # robot_position = random_walk(robot_position, robot_step_size)
#     # Move the robot towards the true target
#     # robot_position = move_towards_target(robot_position, true_target_position, robot_step_size)

#     # Get a sensor measurement
#     measurement, measurement_noise = sensor_model(
#         robot_position, true_target_position)

#     # Update belief using the Kalman Filter
#     belief_mean, belief_cov = update_belief_kalman(
#         belief_mean, belief_cov, measurement, measurement_noise)

    
#     # Implement the controller
#     # Ergodic controller goes here
#     robot_position = move_towards_target(robot_position, true_target_position, 0.5, robot_step_size)
#     # Move the robot using ergodic control
#     # robot_position = ergodic_control(robot_position, belief_mean, belief_cov, robot_step_size)


#     # Store paths
#     robot_path.append(robot_position.copy())
#     belief_path.append(belief_mean.copy())

# plt.ioff()
# plt.show()


