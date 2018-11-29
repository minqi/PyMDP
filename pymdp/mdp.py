"""
Discrete action-space MDP environments, 
which play nice with OpenAI Gym 
"""

import numpy as np

from .mdp_solver import MDPSolver


class State():
	def __init__(self, key, terminal=False):
		self.key = key
		self.actions = set()
		self.action_next_states = {}
		self.action_next_state_p = {}
		self.action_next_state_rewards = {}
		self.terminal = False

	def observation(self):
		return self.key

	def info_for_action(self, action):
		info = {
			'action_next_states': self.action_next_states,
			'action_next_states_p': self.action_next_state_p,
			'action_next_state_rewards': self.action_next_state_rewards
		}
		return info


class MDP():
	def __init__(self):
		self.key_to_state = {} # str key to state object
		self.key_to_action = {} # str key to action index
		self.state = None

	def _state_from_key(self, key):
		if key not in self.key_to_state.keys():
			self.key_to_state[key] = State(key)
		state = self.key_to_state[key]
		return state

	def _normalize_next_state_params(self, next_state_keys, rewards, p):
		if isinstance(next_state_keys, str):
			next_state_keys = [next_state_keys]
		if isinstance(rewards, int) or isinstance(rewards, float):
			rewards = [rewards]
		if isinstance(p, int) or isinstance(p, float):
			p = [p]
		len_rewards = len(rewards)
		len_next_state_keys = len(next_state_keys)
		len_p = len(p)

		verror = ValueError('Length mismatch among rewards, next_state_keys, and p')

		if len_next_state_keys != len_rewards \
			or len_rewards != len_p \
			or len_p != len_next_state_keys:
			if len_next_state_keys > 1:
				if len_rewards == 1:
					rewards *= len_next_state_keys
				else:
					raise verror
				if len_p == 1:
					p *= len_next_state_keys
				else:
					raise verror
			else:
				raise verror

		return next_state_keys, rewards, p

	def add_transition(self, state_key, action_key, next_state_keys, rewards=0, p=1):
		next_state_keys, rewards, p = self._normalize_next_state_params(next_state_keys, rewards, p)

		state = self._state_from_key(state_key)
		next_states = [self._state_from_key(key) for key in next_state_keys]

		action_keys = self.key_to_action.keys()
		if action_key not in action_keys:
			action_keys[action_key] = len(action_keys)
		action = self.key_to_action[action_key]

		state.actions.add(action)
		state.action_next_states[action] = next_states
		state.action_next_state_p[action] = p
		state.action_next_state_rewards[action] = rewards

	def add_action(self, key=None):
		action_keys = self.key_to_action.keys()
		if not key:
			key = str(len(action_keys))
		
		if key in action_keys:
			return key
		else:
			self.key_to_action[key] = len(action_keys)

		return key

	def make_terminal(self, state_key):
		state = self.key_to_state[state_key]
		state.terminal = True
		return state

	def set_state(self, state_key):
		self.state = self.key_to_state[state_key]

	def state_space(self):
		return [self.key_to_state[k] for k in self.key_to_state.keys()]

	def action_space(self):
		return [self.key_to_action[k] for k in self.key_to_action.keys()]

	def _canonicalize_action(self, action):
		if isinstance(action, str):
			if action in self.key_to_action.keys():
				action = self.key_to_action[action]
			else:
				raise ValueError('Unknown action {} for state'.format(action))
		elif isinstance(action, int):
			return action
		else:
			raise ValueError('Invalid type for action {}'.format(action))

	def step(self, action):
		state = self.state
		action = self._canonicalize_action(action)
		if action not in self.state.actions:
			raise ValueError('Invalid action {} for state {}'.format(action, self.state.key))

		idx = np.random.choice(
			len(state.action_next_state_p[action]), 1, state.action_next_state_p[action]).item()
		next_state = state.action_next_states[action][idx]
		reward = state.action_next_state_rewards[action][idx]
		self.state = next_state

		observation = state.observation()
		info = state.info_for_action(action)

		return observation, reward, self.state.terminal, info

	def solve(self, discount):
		return MDPSolver(self).solve(discount=discount, solver='value_iteration')


if __name__ == '__main__':
	m = MDP()

	a_forward = m.add_action('forward')
	m.add_transition('s0', a_forward, 's1', 1)
	m.add_transition('s1', a_forward, 's2', 1)
	m.add_transition('s2', a_forward, 's3', 1)
	m.add_transition('s3', a_forward, 't', 1)
	m.make_terminal('t')
	m.set_state('s0')

	total_r = 0

	done = False
	while not done:
		a = int(np.random.choice(list(m.state.actions)))
		import pdb; pdb.set_trace()
		o, r, done, _ =  m.step(a)
		total_r += r
	print('total reward is {}'.format(total_r))
	print(m.solve(discount=1))