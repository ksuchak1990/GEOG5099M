"""
car_model.py
A python script to test out the Ensemble Kalman Filter with a very basic model.
@author: ksuchak1990
data_created: 19/04/08
"""
# Imports
import matplotlib.pyplot as plt
import numpy as np
from EnsembleKalmanFilter import EnsembleKalmanFilter as EnKF
np.random.seed(666)

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
N_STEPS = 100
OBS_NOISE_MEAN = 0
OBS_NOISE_STD = 20
MODEL_NOISE_MEAN = 0
MODEL_NOISE_STD = 20
ASSIMILATION_PERIOD = 5

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
              'noise_mean': 0,
              'noise_std': 0}

obs_x, obs_y, obs_times = make_data(obs_params, N_STEPS, vis=False)
obs_x = [x + np.random.normal(OBS_NOISE_MEAN, OBS_NOISE_STD) for x in obs_x]
obs_y = [y + np.random.normal(OBS_NOISE_MEAN, OBS_NOISE_STD) for y in obs_y]

# Set up EnKF
model_params = {'x_speed': CAR_X_SPEED,
                'y_speed': CAR_Y_SPEED,
                'noise_mean': MODEL_NOISE_MEAN,
                'noise_std': MODEL_NOISE_STD}

filter_params = {'max_iterations': 50,
                 'ensemble_size': 100,
                 'state_vector_length': 2,
                 'data_vector_length': 2,
                 'H': np.identity(2),
                 'R_vector': np.array([OBS_NOISE_STD, OBS_NOISE_STD])}

# Initialise filter with car model
e = EnKF(Car_Model, filter_params, model_params)

# Step filter
for i in range(N_STEPS):
    if i % ASSIMILATION_PERIOD == 0:
        observation = np.array([obs_x[i], obs_y[i]])
        e.step(observation)
    else:
        e.step()

model_x = [x[0] for x in e.results]
model_y = [x[1] for x in e.results]

# Some plotting
def do_plots(x, y):
    plt.figure()
    plt.plot(true_x, true_y, '--b', label='truth')
    plt.scatter(obs_x, obs_y, color='black', alpha=0.5, label='obs')
    plt.scatter(x, y, color='red', alpha=0.5, label='model')
    plt.title('$\sigma_o={0}$, $\sigma_m={1}$'.format(OBS_NOISE_STD,
                                                                MODEL_NOISE_STD))
    plt.legend()
    plt.show()

def do_more_plots(x, y):
    plt.figure()
    plt.plot(obs_times, x, label='model x')
    plt.plot(obs_times, y, label='model y')
    plt.plot(obs_times, obs_x, label='obs x')
    plt.plot(obs_times, obs_y, label='obs y')
    plt.title('$\sigma_o={0}$, $\sigma_m={1}$'.format(OBS_NOISE_STD,
                                                                MODEL_NOISE_STD))
    plt.legend()
    plt.show()

def do_error_plots(x, y):
    x_model_error = [np.abs(x[i] - true_x[i]) for i in range(len(true_x))]
    y_model_error = [np.abs(y[i] - true_y[i]) for i in range(len(true_y))]

    plt.figure()
    data = [x_model_error, y_model_error]
    plt.boxplot(data)
    plt.show()

#do_plots(model_x, model_y)
#do_more_plots(model_x, model_y)
#do_error_plots(model_x, model_y)

def test_ensemble_size(n, t=ASSIMILATION_PERIOD):
    # Set up params
    model_params = {'x_speed': CAR_X_SPEED,
                'y_speed': CAR_Y_SPEED,
                'noise_mean': MODEL_NOISE_MEAN,
                'noise_std': MODEL_NOISE_STD}

    filter_params = {'max_iterations': 50,
                 'ensemble_size': n,
                 'state_vector_length': 2,
                 'data_vector_length': 2,
                 'H': np.identity(2),
                 'R_vector': np.array([OBS_NOISE_STD, OBS_NOISE_STD])}

    # Set up filter
    e = EnKF(Car_Model, filter_params, model_params)

    # Step filter
    for i in range(N_STEPS):
        if i % t == 0:
            observation = np.array([obs_x[i], obs_y[i]])
            e.step(observation)
        else:
            e.step()

    # Get model outputs
    model_x = [x[0] for x in e.results]
    model_y = [x[1] for x in e.results]

    # Plotting
    do_plots(model_x, model_y)

def wrap_runner():
    sizes = [2, 5, 10, 20, 50, 100]
    periods = [2, 5, 10, 20, 50]
    for t in periods:
        for n in sizes:
            print(n, t)
            test_ensemble_size(n, t)
wrap_runner()
