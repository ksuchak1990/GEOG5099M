"""
car_model.py
A python script to test out the Ensemble Kalman Filter with a very basic model.
@author: ksuchak1990
data_created: 19/04/08
last_modified: 19/04/08
"""
# Imports
import matplotlib.pyplot as plt
import numpy as np
from EnsembleKalmanFilter import EnsembleKalmanFilter as EnKF
np.random.seed(42)

class Car_Model():
    def __init__(self, model_params):
        """
        Initialise model.
        """
        # Set params to default values if not provided in model_params
        self.required_model_params = {'x_speed': 5,
                                      'y_speed': 5,
                                      'noise_mean': 0,
                                      'noise_std': 0}
        for k, v in self.required_model_params.items():
            if k in model_params:
                setattr(self, k, model_params[k])
            else:
                setattr(self, k, v)

        # Initial state
        self.time_step = 1
        self.time = 0
        self.state = np.array([0, 0])

        # Collecting states
        self.times = [self.time]
        self.states = [self.state.copy()]

    def step(self):
        """
        Step model forward one step in time.
        """
        # Update state
        x_noise = np.random.normal(self.noise_mean, self.noise_std)
        y_noise = np.random.normal(self.noise_mean, self.noise_std)
        x_update = self.x_speed * self.time_step + x_noise
        y_update = self.y_speed * self.time_step + y_noise

        # Update state and time
        self.time += self.time_step
        self.state[0] += x_update
        self.state[1] += y_update

        # Add to list of states
        self.times.append(self.time)
        self.states.append(self.state.copy())

# Constants
CAR_X_SPEED = 5
CAR_Y_SPEED = 5
N_STEPS = 50
OBS_NOISE_MEAN = 0
OBS_NOISE_STD = 5
MODEL_NOISE_MEAN = 0
MODEL_NOISE_STD = 10

def make_data(params, n, vis=True):
    """
    Function to create data from model
    """

    cm = Car_Model(params)
    for i in range(n):
        cm.step()

    # Get state as lists
    x = [x[0] for x in cm.states]
    y = [x[1] for x in cm.states]
    times = cm.times

    # Check with plot
    if vis:
        plt.figure()
        plt.scatter(x, y)
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.show()

    return x, y, times

# Make true data
track_params = {'x_speed': CAR_X_SPEED,
                'y_speed': CAR_Y_SPEED,
                'noise_mean': 0,
                'noise_std': 0}

true_x, true_y, true_times = make_data(track_params, N_STEPS, vis=False)

# Make observation data
obs_params = {'x_speed': CAR_X_SPEED,
              'y_speed': CAR_Y_SPEED,
              'noise_mean': OBS_NOISE_MEAN,
              'noise_std': OBS_NOISE_STD}

obs_x, obs_y, obs_times = make_data(obs_params, N_STEPS, vis=False)

# Set up EnKF
model_params = {'x_speed': CAR_X_SPEED,
                'y_speed': CAR_Y_SPEED,
                'noise_mean': MODEL_NOISE_MEAN,
                'noise_std': MODEL_NOISE_STD}

filter_params = {'max_iterations': 50,
                 'ensemble_size': 10,
                 'state_vector_length': 2,
                 'data_vector_length': 2,
                 'H': np.identity(2),
                 'R_vector': np.array([OBS_NOISE_STD, OBS_NOISE_STD])}

# Initialise filter with car model
e = EnKF(Car_Model, filter_params, model_params)

# Step filter
for i in range(N_STEPS):
    if i % 5 == 0:
        observation = np.array([obs_x[i], obs_y[i]])
        e.step(observation)
    else:
        e.step()

model_x = [x[0] for x in e.results]
model_y = [x[1] for x in e.results]

# Some plotting
def do_plots():
    plt.figure()
    plt.plot(true_x, true_y, '--b', label='truth')
    plt.scatter(obs_x, obs_y, color='black', alpha=0.5, label='obs')
    plt.scatter(model_x, model_y, color='red', alpha=0.5, label='model')
    plt.title('$\sigma_o={0}$, $\sigma_m={1}$'.format(OBS_NOISE_STD,
                                                                MODEL_NOISE_STD))
    plt.legend()
    plt.show()

def do_more_plots():
    plt.figure()
    plt.plot(obs_times, model_x, label='model x')
    plt.plot(obs_times, model_y, label='model y')
    plt.plot(obs_times, obs_x, label='obs x')
    plt.plot(obs_times, obs_y, label='obs y')
    plt.title('$\sigma_o={0}$, $\sigma_m={1}$'.format(OBS_NOISE_STD,
                                                                MODEL_NOISE_STD))
    plt.legend()
    plt.show()

do_plots()
do_more_plots()
