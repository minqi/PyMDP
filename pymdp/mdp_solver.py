import numpy as np

SOLVER_TYPE_VALUE_ITERATION = 'value_iteration'

class MDPSolver():
	def __init__(self, mdp):
		self.mdp = mdp

	def _value_iteration(self, discount=1.0):
		states = self.mdp.state_space()
		actions = self.mdp.action_space()
		num_states = len(states)
		num_actions = len(actions)
		state_key_to_idx = dict(zip([s.key for s in states], range(num_states)))

		v = np.zeros(num_states)
		sas_p = np.zeros((num_states, num_actions, num_states))
		sas_r = np.zeros(sas_p.shape)

		# initialize transition matrix (sas_p) and reward matrix (sas_r)
		for i, s in enumerate(states):
			if s.terminal:
				sas_p[i, :, i] = 1
			else:
				for a in s.actions:
					for j, s_next in enumerate(s.action_next_states[a]):
						s_next_idx = state_key_to_idx[s_next.key]
						sas_p[i, a, s_next_idx] = s.action_next_state_p[a][j]
						sas_r[i, a, s_next_idx] = s.action_next_state_rewards[a][j]
		
		v_old = [np.inf] * len(v)
		while not np.allclose(v_old, v, rtol=0):
			v_old = v
			v_next_state = (discount * v).reshape(1, 1, v.shape[-1])
			v = (sas_p  * (sas_r + v_next_state)).sum(-1).max(-1)

		return dict(zip([s.key for s in states], v))

	def solve(self, discount=1.0, solver=SOLVER_TYPE_VALUE_ITERATION):
		if solver == SOLVER_TYPE_VALUE_ITERATION:
			return self._value_iteration(discount=discount)
		else:
			raise ValueError('Solver of type {} is not supported'.format(solver))
