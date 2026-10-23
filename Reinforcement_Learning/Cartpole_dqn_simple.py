# %%
import random
import time
# %pip install "gymnasium[classic-control]"
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

# %%
# settings
env_id = "CartPole-v1"

# number of environments to run in parallel
num_envs = 1

# total number of timesteps for training over multiple episodes
total_timesteps = 100_000

# adam learning rate
learning_rate = 1e-3

# discount factor
gamma = 0.99

# epsilon value for epsilon-greedy
eps = 0.1

# seed for reproducibility
seed = None

video_path = "qlearning_cartpole_videos"

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# %%
def make_env(env_id, capture_video, seed=None):
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda episode: True)
    else:
        env = gym.make(env_id)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    if seed is not None:
        env.action_space.seed(seed)

    return env

# %%
# NN
class QNetwork(nn.Module):

    def __init__(self, envs, n_hidden=128):
        super().__init__()

        self.network = nn.Sequential(
                        nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden, envs.single_action_space.n)
        )

    def forward(self, state_x):
        return self.network(state_x)

def np2torch(a):
    return torch.Tensor(a).float().to(device)

# %%
# make environaments
envs = gym.vector.SyncVectorEnv(
    [
        lambda i=i: make_env(
            env_id,
            False,
            seed=None if seed is None else seed + i,
        )
        for i in range(num_envs)
    ]
)

# %%
episode_returns = []
losses = []
current_state_obs, _ = envs.reset(seed=seed) # States [B, 4] [B=1, [Cart position, Cart velocity, Pole angle, Pole angular velocity]]
auto_reset = np.zeros(num_envs, dtype = bool)

# NN to GPU
q_network = QNetwork(envs).to(device) # state --> NN --> Q(s,a), where predicted_value = Q(s,a).max() and predicted_action = arg_max(Q(s,a))
optimizer = opt.Adam(q_network.parameters(), lr=learning_rate)

start_time = time.time()
for global_step in range(total_timesteps):
    
    #select action(s) by epsilon-greedy method
    if random.random() < eps:
        actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
    else:
        # select best actions by arg_max from NN (Q(s,a))
        q_values = q_network(np2torch(current_state_obs)) # to be optimized by NN
        actions = torch.argmax(q_values, dim=1).cpu().numpy() # predicted_actions = arg_max(Q(s,a))

    # get s', R, done - take a step in the environment
    next_state_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

    # record returns for plotting and update episode_returns
    for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
        if terminated or truncated:
            ret = infos['episode']['r'][i] # accumulated rewards
            episode_returns.append(ret)
            print(f"global_step = {global_step}, episode = {len(episode_returns)}, episode_return = {ret}")

    # training

    # use only envs that haven't been autoreset
    auto_reset_mask = np.logical_not(auto_reset)
    # calculate the target = R + γ * (max(Q(s', a')))
    if(np.any(auto_reset_mask)):

        # masked with the not resetted
        next_state_obs_masked = next_state_obs[auto_reset_mask]
        current_state_obs_masked = current_state_obs[auto_reset_mask]
        actions_masked = actions[auto_reset_mask]
        rewards_masked = rewards[auto_reset_mask]
        dones_masked = terminateds[auto_reset_mask] | truncateds[auto_reset_mask]

        with torch.no_grad():
            # get the next max(Q(s', a'))
            next_max_q_values = torch.max(q_network(np2torch(next_state_obs_masked)), dim=1).values
            # Bellman: Q(s,a) = Q(s,a) + α * (R + γ * (max(Q(s', a'))) - Q(s,a))  --> ( Q(s,a) = Q(s,a) + α * (target - Q(s,a)))
            target = np2torch(rewards_masked) + gamma * (1 - np2torch(dones_masked)) * next_max_q_values

        # prediction: forward by masked Q(s) and get the values Q(s,a) for the actions performed (gather)
        q_values_masked = q_network(np2torch(current_state_obs_masked)) # [B, 2 (left, right)]
        # gather() selects the Q-value of the action actually performed, not the action with the maximum Q-value. 
        # This is exactly what we want when computing the prediction term (Q(s,a) in the loss function
        q_values_performed = torch.gather(
            input=q_values_masked, 
            dim=1, 
            index = torch.from_numpy(actions_masked.reshape(-1,1)).to(device)
            ) # [B, 1]

        #prepare the dimensions for the loss function
        q_values_predictions = q_values_performed.squeeze(1)  # [B]

        loss = F.mse_loss(q_values_predictions, target)
        losses.append(loss.detach().cpu().numpy())

        # optimize
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()

    # go to the next state
    current_state_obs = next_state_obs

    # update auto reset
    auto_reset = np.logical_or(terminateds, truncateds)

    if global_step % 1000 == 0:
        print("steps per second:", int(global_step / (time.time() - start_time)))

    # let's brek if the last 10 episodes had max reward
    if(len(episode_returns)) > 10 and np.all(np.equal(episode_returns[-10:], 500)):
        print("max reward achieved")
        break

envs.close()   


# %%
def smooth(x, a=0.1):
    y = [x[0]]
    for xi in x[1:]:
        yi = a * xi + (1-a) * y[-1]
        y.append(yi)
    return y

plt.plot(episode_returns, alpha=0.2)
plt.plot(smooth(episode_returns))
plt.title("episode_returns")
plt.show()

# %%
plt.plot(losses)
plt.title("losses")
plt.show()

# %%
# save model
model_path = "/workspace/repos/jupyter_notebooks/Reinforcement_Learning/q_network.pth"
torch.save(q_network.state_dict(), model_path)

# %%
# load the model for eval
# %pip install "gymnasium[other]"
envs_eval = gym.vector.SyncVectorEnv([lambda: make_env(env_id, True)]) # open with recording
model = QNetwork(envs_eval).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# %%
# evaluate the model
n_episodes_eval = 10
eval_returns = np.zeros(n_episodes_eval)
state_obs, _ = envs_eval.reset()

for i in range(n_episodes_eval):
    episode_done = False

    while not episode_done:
        q_values = model(np2torch(state_obs))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        state_obs, rewards, terminateds, truncateds, infos = envs_eval.step(actions)

        if terminateds[0] or truncateds[0]:
            G = infos['episode']['r'][0]
            print(f"episode: {i}, return= {G}")
            eval_returns[i] = G
            episode_done = True

# %%
plt.hist(eval_returns)
plt.title("Eval Returns")
plt.show();

# %%
# prompt display video in notebook
from IPython.display import HTML
from base64 import b64encode

mp4 = open(f'{video_path}/rl-video-episode-8.mp4', 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML("""
<video width=400 controls>
    <source src="%s" type="video/mp4">
</video>
""" % data_url)


