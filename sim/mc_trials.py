# test
from numpy import random 
from math import floor, ceil
import matplotlib.pyplot as plt

# Trials are a Directed Acyclic Graph (DAG) -> for simplicity it's a singly linked list
# we can represent this by using a list [0, 0, 0, 0]
# This would be a drug that hasn't move though any trials

class State(object):

	# outcomes
	SUCCESS = 1
	FAIL = 0
	NOT_STARTED = -1

	def __init__(self, val=-1, next=None):
		self.val = val
		self.next = next

	def attempt_trial(self, p, distribution=None):
		if distribution is None:
			num = random.random()
			if num <= p:
				return self.SUCCESS
			return self.FAIL
		else:
			dist_funcion, dist_args = distribution
			num = dist_funcion(*dist_args)
			if num <= p:
				return self.SUCCESS
			return self.FAIL

class PreTrial(State):
	
	def transition(self, p):
		if self.val == self.NOT_STARTED:
			trial_outcome = self.attempt_trial(p)
			self.val = trial_outcome
			if trial_outcome == self.SUCCESS:
				return P1()
		return self


class P1(State):

	def transition(self, p):
		if self.val == self.NOT_STARTED:
			trial_outcome = self.attempt_trial(p)
			self.val = trial_outcome
			if trial_outcome == self.SUCCESS:
				return P2()
		return self

class P2(State):

	def transition(self, p):
		if self.val == self.NOT_STARTED:
			trial_outcome = self.attempt_trial(p)
			self.val = trial_outcome
			if trial_outcome == self.SUCCESS:
				return P3()
		return self

class P3(State):

	def transition(self, p):
		if self.val == self.NOT_STARTED:
			trial_outcome = self.attempt_trial(p)
			self.val = trial_outcome
			if trial_outcome == self.SUCCESS:
				return Complete()
		return self

class Complete(State):

	def __init__(self):
		super(Complete, self).__init__()
		self.val = self.SUCCESS

class ClinicalStages(object):

	def __init__(self, start):
		self.state = start

	def trial_runs(self, probs):
		idx = 0
		while True:
			if self.state.val == State.FAIL:
				break
			if isinstance(self.state, Complete):
				yield self.state
				break
			new_state = self.state.transition(probs[idx])
			yield self.state
			self.state = new_state
			idx += 1

class Simulation(object):

	def __init__(self, drug_id, num_trials, npv=None, start_stage=PreTrial, probs=[], distributions=[]):

		self.drug_id = drug_id
		self.num_trials = num_trials
		self.npv = npv
		self.start_stage = start_stage
		self.probs = probs
		self.distributions = distributions

	def run_simulation(self):
		stop_point_counts = [0]*self.get_stop_point_len()
		for run in range(self.num_trials):
			new_start = self.start_stage()
			stages = ClinicalStages(new_start)
			outcomes =[trial.val for trial in stages.trial_runs(self.probs)]
			stop_point_counts[len(outcomes)-1] += 1
		return stop_point_counts


	def get_stop_point_len(self):
		"""
		I really can't think of a good way to do this
		"""
		if self.start_stage == PreTrial:
			return 5
		if self.start_stage == P1:
			return 4
		if self.start_stage == P2:
			return 3
		if self.start_stage == P3:
			return 2
		if self.start_stage == Complete:
			return 1

if __name__ == '__main__':
	sim = Simulation(1234, 100000, probs=[.7, .3, .5, .40])
	results = sim.run_simulation()
	plt.bar(range(len(results)), results)
	plt.show()
