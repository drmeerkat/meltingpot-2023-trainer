import copy
import dmlab2d
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from meltingpot import substrate
from ml_collections import config_dict
from ray.rllib.env import multi_agent_env
from .constants import PLAYER_STR_FORMAT
from .env_utils import (DownSamplingSubstrateWrapper, 
                        remove_unrequired_observations_from_space, 
                        spec_to_space, 
                        timestep_to_observations,
					    downsample_observation)

RIPE_BERRY_COLORS = [
        [82, 43, 43],
        [43, 82, 43],
        [43, 43, 82]
    ]

UNRIPE_NONRED_COLORS = [
        [46, 62, 46],
        [46, 46, 62],
    ]

ACTIONS = {
  'noop': 0,
  'up': 1,
  'down': 2,
  'left': 3,
  'right': 4,
  'turn_left': 5,
  'turn_right': 6,
  'zap': 7,
  'plant_red': 8,
  'plant_two': 9,
  'plant_three': 10
}

# Map from the rotation angle when matching local view to global map
# to the corresponding agent ego-centric coordinate in global view
AGENT_POS_MAPPING = {
    0:(9,5),
    1:(5,9),
    2:(1,5),
    3:(5,1),
}


class AlHarvestMeltingPotEnv(multi_agent_env.MultiAgentEnv):
	"""Interfacing Melting Pot substrates and RLLib MultiAgentEnv."""

	def __init__(self, env: dmlab2d.Environment, custom_reward: bool = True):
		"""Initializes the instance.

		Args:
			env: dmlab2d environment to wrap. Will be closed when this wrapper closes.
		"""
		self._image_only = True
		self._env = env
		self._num_players = len(self._env.observation_spec())
		self._ordered_agent_ids = [
			PLAYER_STR_FORMAT.format(index=index)
			for index in range(self._num_players)
		]
		# RLLib requires environments to have the following member variables:
		# observation_space, action_space, and _agent_ids
		self._agent_ids = set(self._ordered_agent_ids)

		# Use self-defined reward function by default
		self._use_custom_reward = custom_reward
		
		# RLLib expects a dictionary of agent_id to observation or action,
		# Melting Pot uses a tuple, so we convert them here
		self.observation_space = self._convert_spaces_tuple_to_dict(
			spec_to_space(self._env.observation_spec()),
			remove_world_observations=True, image_only=self._image_only)
		self.action_space = spaces.Dict({
            agent_id: Discrete(9) for agent_id in self._ordered_agent_ids
        })

		super().__init__()

	def reset(self, *args, **kwargs):
		"""See base class."""
		timestep = self._env.reset()
		obs = timestep_to_observations(timestep, image_only=self._image_only)
		return obs, {}

	def step(self, action_dict):
		"""See base class."""
		actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
		obs = copy.deepcopy(self._env.observation())
		
		# custom_rewards = [get_custom_rewards(obs[index], actions[index], index) for index in range(len(self._ordered_agent_ids))]
		custom_rewards = [get_custom_rewards_global(obs[index], actions[index], index) for index in range(len(self._ordered_agent_ids))]
		timestep = self._env.step(actions)
		if self._use_custom_reward:
			rewards = {
				agent_id: custom_rewards[index]
				for index, agent_id in enumerate(self._ordered_agent_ids)
			}
		else:
			rewards = {
				agent_id: timestep.reward[index]
				for index, agent_id in enumerate(self._ordered_agent_ids)
			}
		done = {'__all__': timestep.last()}
		info = {}

		observations = timestep_to_observations(timestep, image_only=self._image_only)
		return observations, rewards, done, done, info

	def close(self):
		"""See base class."""

		self._env.close()

	def get_dmlab2d_env(self):
		"""Returns the underlying DM Lab2D environment."""

		return self._env

	# Metadata is required by the gym `Env` class that we are extending, to show
	# which modes the `render` method supports.
	metadata = {'render.modes': ['rgb_array']}

	def render(self) -> np.ndarray:
		"""Render the environment.

		This allows you to set `record_env` in your training config, to record
		videos of gameplay.

		Returns:
			np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
			representing RGB values for an x-by-y pixel image, suitable for turning
			into a video.
		"""

		observation = self._env.observation()
		world_rgb = observation[0]['WORLD.RGB']

		# RGB mode is used for recording videos
		return world_rgb

	def _convert_spaces_tuple_to_dict(
		self,
		input_tuple: spaces.Tuple,
		remove_world_observations: bool = False,
		image_only: bool = False) -> spaces.Dict:
		"""Returns spaces tuple converted to a dictionary.

		Args:
			input_tuple: tuple to convert.
			remove_world_observations: If True will remove non-player observations.
		"""

		if image_only:
		
			return spaces.Dict({
				agent_id: Box(low=-1.0, high=1.0, shape=(363,), dtype=np.float32)
				for agent_id in self._ordered_agent_ids
			})

		return spaces.Dict({
			agent_id: (remove_unrequired_observations_from_space(input_tuple[i])
					   if remove_world_observations else input_tuple[i])
			for i, agent_id in enumerate(self._ordered_agent_ids)
		})

def get_custom_rewards(obs, action, index):
	"""Reward players for moving closer to ripe berry"""
	rgb_data = downsample_observation(obs['RGB'], 8)
	total_reward = 0.0
	
	nearest_ripe_berry, _ = find_nearest_berry(rgb_data, RIPE_BERRY_COLORS)
	if nearest_ripe_berry:
		# berry in front and player moves forward
		if nearest_ripe_berry == (8, 5) and action == ACTIONS['up']:
			total_reward += 2.0
		# berry behind and player moves backward
		elif nearest_ripe_berry == (10, 5) and action == ACTIONS['down']:
			total_reward += 2.0
		# berry to left and player moves left
		elif nearest_ripe_berry == (9, 4) and action == ACTIONS['left']:
			total_reward += 2.0
		# berry to right and player moves right
		elif nearest_ripe_berry == (9, 6) and action == ACTIONS['right']:
			total_reward += 2.0
		# else:
		#    direction = direction_from_reference(nearest_ripe_berry)
		#    expected_action = direction_to_number(direction)
		#    total_reward += 0.03 if action == expected_action else 0

	nearest_unripe, _ = find_nearest_berry(rgb_data, UNRIPE_NONRED_COLORS)
	if nearest_unripe:
		# if in front, plant red
		if nearest_unripe == (8, 5):
			total_reward += 1.0 if action == ACTIONS['plant_red'] else 0
		# if on left side, turn left
		elif nearest_unripe == (9, 4):
			total_reward += 0.025 if action == ACTIONS['turn_left'] else 0
		# if on right side, turn right
		elif nearest_unripe == (9, 6):
			total_reward += 0.025 if action == ACTIONS['turn_right'] else 0
		# if behind, turn around
		elif nearest_unripe == (10, 5):
			total_reward += 0.025 if action == ACTIONS['turn_right'] else 0
		else:
			# Find closest berry to the position in front of player (where the zapper is)
			# at this point, the nearest unripe is definitely not the cross positions

			# In our set up, we use global nearest instead of agent local nearest (the one can be seen)
			nearest_unripe, _ = find_nearest_berry(rgb_data, UNRIPE_NONRED_COLORS, offset_in_front_of_player=True)
			# nearest_unripe, _ = find_nearest_berry_global(rgb_data, UNRIPE_NONRED_COLORS, offset_in_front_of_player=True)
			# ensure player navigates in front of berry
			nearest_unripe = (nearest_unripe[0]+1, nearest_unripe[1])
			direction = direction_from_reference(nearest_unripe)
			expected_action = direction_to_number(direction)
			total_reward += 0.025 if action == expected_action else 0

	# ensure the player does not plant blue or green- these should be disabled in the action space
	if action in [ACTIONS['plant_two'], ACTIONS['plant_three']]:
		assert 1==2
		total_reward -= 1.0

	return total_reward

def get_custom_rewards_global(obs, action, index):
	"""
	Reward players for moving closer to ripe berry; for unripe ones, 
	reward agent for moving towards those even they are out of sight
	"""
	padded_obs = np.pad(obs['WORLD.RGB'], [[0, 24], [0, 32], [0, 0]], 'constant', constant_values=56)
	down_world_obs = np.zeros([padded_obs.shape[0]//8, padded_obs.shape[1]//8, 3], dtype=int)

	# pad the global obs image to 264x264 with grey color
	for i in range(0, 240, 88):
		for j in range(0, 232, 88):
			# print((i,j), (i+88, j+88))
			down_world_obs[i//8:i//8+11, j//8:j//8+11, :] = downsample_observation(padded_obs[i:i+88, j:j+88, :], 8)
	total_reward = 0.0

	# Using the same local obs to keep other rewards unchanged
	rgb_data = downsample_observation(obs['RGB'], 8)
	
	nearest_ripe_berry, _ = find_nearest_berry(rgb_data, RIPE_BERRY_COLORS)
	if nearest_ripe_berry:
		# berry in front and player moves forward
		if nearest_ripe_berry == (8, 5) and action == ACTIONS['up']:
			total_reward += 2.0
		# berry behind and player moves backward
		elif nearest_ripe_berry == (10, 5) and action == ACTIONS['down']:
			total_reward += 2.0
		# berry to left and player moves left
		elif nearest_ripe_berry == (9, 4) and action == ACTIONS['left']:
			total_reward += 2.0
		# berry to right and player moves right
		elif nearest_ripe_berry == (9, 6) and action == ACTIONS['right']:
			total_reward += 2.0
		# else:
		#    direction = direction_from_reference(nearest_ripe_berry)
		#    expected_action = direction_to_number(direction)
		#    total_reward += 0.03 if action == expected_action else 0

	nearest_unripe, _ = find_nearest_berry(rgb_data, UNRIPE_NONRED_COLORS)
	if nearest_unripe:
		# if in front, plant red
		if nearest_unripe == (8, 5):
			total_reward += 1.0 if action == ACTIONS['plant_red'] else 0
		# if on left side, turn left
		elif nearest_unripe == (9, 4):
			total_reward += 0.025 if action == ACTIONS['turn_left'] else 0
		# if on right side, turn right
		elif nearest_unripe == (9, 6):
			total_reward += 0.025 if action == ACTIONS['turn_right'] else 0
		# if behind, turn around
		elif nearest_unripe == (10, 5):
			total_reward += 0.025 if action == ACTIONS['turn_right'] else 0
		else:
			# Find closest berry to the position in front of player (where the zapper is)
			# at this point, the nearest unripe is definitely not the cross positions
			# In our set up, we use global nearest instead of agent local nearest (the one can be seen)
			# TODO: we might still need to navigate towards the front of unripe berries!
			nearest_unripe, _ = find_nearest_berry_global(down_world_obs, rgb_data, UNRIPE_NONRED_COLORS)
			direction = direction_from_reference(nearest_unripe)
			expected_action = direction_to_number(direction)
			total_reward += 0.025 if action == expected_action else 0

	# ensure the player does not plant blue or green- these should be disabled in the action space
	if action in [ACTIONS['plant_two'], ACTIONS['plant_three']]:
		assert 1==2
		total_reward -= 1.0

	return total_reward

def find_nearest_berry(rgb_data, target_colors, offset_in_front_of_player=False):
	# Define the target RGB values
	
	combined_mask = np.zeros((11, 11), dtype=bool)
	
	# Create a mask for each target color and combine them
	for color in target_colors:
		mask = np.all(rgb_data == color, axis=-1)
		combined_mask = np.logical_or(combined_mask, mask)
	
	# Get row and column indices of pixels matching any of the target colors
	rows, cols = np.where(combined_mask)
	
	# If there's no matching pixel, return None
	if len(rows) == 0:
		return None, None
	
	# TODO: we don't really need this offset??? same as outside, can remove "navigate to front" thing
	origin_pos = (9, 5)
	if offset_in_front_of_player:
		origin_pos = (8, 5)

	# Calculate the Euclidean distance for each matching pixel to the point (10, 6)
	distances = np.sqrt((rows - origin_pos[0])**2 + (cols - origin_pos[1])**2)  # Subtracting 1 because of 0-based indexing
	
	# Find the pixel with the minimum distance
	min_index = np.argmin(distances)
	nearest_pixel = (rows[min_index], cols[min_index])
	shortest_distance = distances[min_index]
	
	return nearest_pixel, shortest_distance

def find_nearest_berry_global(down_world_obs, down_obs, target_colors):
	# Input rgb data is the downsampled world map

	combined_mask = np.zeros((33, 33), dtype=bool)
	
	# Create a mask for each target color and combine them
	for color in target_colors:
		mask = np.all(down_world_obs == color, axis=-1)
		combined_mask = np.logical_or(combined_mask, mask)
	
	# Get row and column indices of pixels matching any of the target colors
	rows, cols = np.where(combined_mask)
	
	# If there's no matching pixel, return None
	if len(rows) == 0:
		return None, None
	
	# Find each agent's glocal coordination
	row, col = down_world_obs.shape[:2]
	score = 9999
	pos = (-1, -1, -1)
	for i in range(row-10):
		for j in range(col-10):
			# even though the agent color might not be matching, we can still use the most 
			# matching patch as the true location!
			if len(np.where(down_world_obs[i:i+11, j:j+11, :] - np.rot90(down_obs, 0, axes=[0,1]))[0])//3 < score:
				score = len(np.where(down_world_obs[i:i+11, j:j+11, :] - np.rot90(down_obs, 0, axes=[0,1]))[0])//3
				pos = (i, j, 0)
			elif len(np.where(down_world_obs[i:i+11, j:j+11, :] - np.rot90(down_obs, 1, axes=[0,1]))[0])//3 < score:
				score = len(np.where(down_world_obs[i:i+11, j:j+11, :] - np.rot90(down_obs, 1, axes=[0,1]))[0])//3
				pos = (i, j, 1)
			elif len(np.where(down_world_obs[i:i+11, j:j+11, :] - np.rot90(down_obs, 2, axes=[0,1]))[0])//3 < score:
				score = len(np.where(down_world_obs[i:i+11, j:j+11, :] - np.rot90(down_obs, 2, axes=[0,1]))[0])//3
				pos = (i, j, 2)
			elif len(np.where(down_world_obs[i:i+11, j:j+11, :] - np.rot90(down_obs, 3, axes=[0,1]))[0])//3 < score:
				score = len(np.where(down_world_obs[i:i+11, j:j+11, :] - np.rot90(down_obs, 3, axes=[0,1]))[0])//3
				pos = (i, j, 3)
	
	# The pos tuple has three entries, pos[:2] is the global coordinate of the origin in agent's 
	# ego centric view (w.r.t that (9,5) coord), pos[2] is the rotation angle needed to map the agent's ego view
	# with the world obs
	global_agent_pos = np.array(pos[:2]) + np.array(AGENT_POS_MAPPING[pos[2]])

	# Calculate the Euclidean distance for each matching pixel to the point (10, 6)
	# Subtracting 1 because of 0-based indexing
	distances = np.sqrt((rows - global_agent_pos[0])**2 + (cols - global_agent_pos[1])**2)  
	
	# Find the pixel with the minimum distance
	min_index = np.argmin(distances)
	nearest_pixel = (rows[min_index], cols[min_index])
	shortest_distance = distances[min_index]

	# Find the local coordinate of nearest unripe
	nearest_unripe = nearest_pixel
	relative_nearest_unripe = rot90point(nearest_unripe, angle=pos[2]) + np.array([9, 5]) - rot90point(global_agent_pos, angle=pos[2])
	
	return relative_nearest_unripe, shortest_distance

def rot90point(point, angle=0, img_shape=(33,33)):
    # given a point in a 2D image, rotate the coordinate system 
    # counterclock wise by 90 x (angle % 4), return the coordinate of the point
    # under the new coordinate system

    # assume the input point's coordinate system's origin is at top left
    if angle % 4 == 0:
        new_point = copy.deepcopy(point)
    elif angle % 4 == 1:
        new_point = (point[1], img_shape[0] - point[0])
    elif angle % 4 == 2:
        new_point = (img_shape[0] - point[0], img_shape[1] - point[1])
    else:
        new_point = (img_shape[1] - point[1], point[0])
    return new_point

# TODO: This can be improved, use a set, if the action is within the set, its ok
# consider the example of moving in squres
def direction_from_reference(nearest_pixel):
    reference_pixel = (9, 5)
    
    # Determine the direction
    if nearest_pixel[0] < reference_pixel[0]:  # Checking row value
        return "up"
    elif nearest_pixel[0] > reference_pixel[0]:
        return "down"
    elif nearest_pixel[1] < reference_pixel[1]:  # Checking column value
        return "left"
    elif nearest_pixel[1] > reference_pixel[1]:
        return "right"
    else:
        return "same_location"

def direction_to_number(direction):
    mapping = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "same_location": 0
    }
    return mapping.get(direction, 0)

def al_harvest_env_creator(env_config):
    """
    Build the substrate, interface with RLLib and apply Downsampling to observations.
    """
    env_config = config_dict.ConfigDict(env_config)
    env = substrate.build(env_config['substrate'], roles=env_config['roles'])
    env = DownSamplingSubstrateWrapper(env, env_config['scaled'])
    env = AlHarvestMeltingPotEnv(env, custom_reward=env_config.get('use_custom_reward', True))
    return env
