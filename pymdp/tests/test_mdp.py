from nose.tools import assert_equals, assert_true, assert_false

from ..mdp import MDP

class TestMDP:

	def test_mdp_solver_value_iteration_linear_mdp(self):
		"""
		Value iteration should find optimal V(s) for a simple linear MDP
		"""
		m = MDP()
		a_forward = m.add_action('forward')
		m.add_transition('s0', a_forward, 's1', 1)
		m.add_transition('s1', a_forward, 's2', 1)
		m.add_transition('s2', a_forward, 's3', 1)
		m.add_transition('s3', a_forward, 't', 1)
		m.make_terminal('t')

		v = m.solve(discount=1.0)

		assert_equals(v['s0'], 4)
		assert_equals(v['s1'], 3)
		assert_equals(v['s2'], 2)
		assert_equals(v['s3'], 1)
		assert_equals(v['t'], 0)

	def test_mdp_solver_value_iteration_toy_mdp(self):
		"""
		Value iteration should find optimal V(s) for toy MDP from Slide 38 of
		David Silver's MDP lecture:
		http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf
		"""
		m = MDP()
		a1 = m.add_action('a1')
		a2 = m.add_action('a2')
		a3 = m.add_action('a3')
		m.add_transition('facebook', a1, 'facebook', -1)
		m.add_transition('facebook', a2, 'class1', 0)
		m.add_transition('class1', a1, 'facebook', -1)
		m.add_transition('class1', a2, 'class2', -2)
		m.add_transition('class2', a1, 'sleep', 0)
		m.add_transition('class2', a2, 'class3', -2)
		m.add_transition('class3', a1, 'sleep', 10)
		m.add_transition('class3', a2, 'pub', 1)
		m.add_transition('pub', a1, ['class1', 'class2', 'class3'], [0, 0, 0], [0.2, 0.4, 0.4])
		m.make_terminal('sleep')

		v = m.solve(discount=1.0)

		assert_equals(v['facebook'], 6.0)
		assert_equals(v['class1'], 6.0)
		assert_equals(v['class2'], 8.0)
		assert_equals(v['class3'], 10)
		assert_equals(v['sleep'], 0)
		assert_equals(v['pub'], 8.4)