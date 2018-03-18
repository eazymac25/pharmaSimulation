# test
from numpy import random 
from math import floor, ceil
import matplotlib.pyplot as plt

# Trials are a Directed Acyclic Graph (DAG) -> for simplicity it's a singly linked list
# we can represent this by using a list [0, 0, 0, 0]
# This would be a drug that hasn't move though any trials

class NPV(object):

	def __init__(self, fcf, period_len, cost, prob, r=0.1, start_year=0):
		# argument validation
		if not isinstance(prob, list) and not isinstance(prob, float):
			raise ValueError('please insert correct prob as a list of floats or a float')
		if not isinstance(fcf, float) and not isinstance(fcf, list):
			raise ValueError('please include a free cash flow as a float or list of floats')
		if isinstance(fcf, list) and len(fcf) != period_len:
			raise ValueError('please include a list of cash flows equal to the length of the period or use a static cash flow')
		if isinstance(prob, list) and len(prob) != period_len:
			raise ValueError('please include a list of probabilities equal to the length of the period or use a static probability')

		self.fcf = fcf
		self.start_year = start_year
		self.period_len = period_len
		self.cost = cost
		self.prob = prob
		self.r = r

	def calc(self):
		npv = -cost
		for t in range(self.period_len):
			prob = self.prob
			fcf = self.fcf
			if isinstance(self.prob, list):
				prob = self.prob[t]
			if isinstance(self.fcf, list):
				fcf = self.fcf[t]
			npv += (prob*fcf)/((1.+self.r)^(start_year+1))
		return npv

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

	def transition(self, p, success_state):
		if self.val == self.NOT_STARTED:
			trial_outcome = self.attempt_trial(p)
			self.val = trial_outcome
			if trial_outcome == self.SUCCESS:
				return success_state()
			return Fail()
		return self

class PreTrial(State):
	pass

class P1(State):
	pass

class P2(State):
	pass

class P3(State):
	pass

class NDA(State):
	pass

class Complete(State):

	def __init__(self):
		super(Complete, self).__init__()
		self.val = self.SUCCESS

class Fail(State):

	def __init__(self):
		super(Complete, self).__init__()
		self.val = self.FAIL

class ClinicalStages(object):

	def __init__(self, stages=[]):
		FST_STAGE = 0
		self.state = stages[FST_STAGE]
		self.stages = stages

	def trial_runs(self, probs):
		idx = 0
		while True:
			if isinstance(self.state, Fail):
				break
			if isinstance(self.state, Complete):
				yield self.state
				break
			new_state = self.state.transition(probs[idx], self.stages[idx])
			yield self.state
			self.state = new_state
			idx += 1

class Simulation(object):

	state_map = {
		'PRE': PreTrial
		'P1': P1,
		'P2': P2,
		'P3': P3,
		'NDA': NDA,
		'DONE': Complete
	}

	def __init__(self, drug_id, num_trials, npv=None,
		stage_list=('PRE','P1','P2','P3','NDA','DONE'), probs=[], distributions=[]):

		self.drug_id = drug_id
		self.num_trials = num_trials
		self.npv = npv
		self.stage_list = stage_list
		self.probs = probs
		self.distributions = distributions

	def run_simulation(self):
		stages = self.get_stages()
		stop_point_counts = [0]*len(stages)
		for run in range(self.num_trials):
			stages = ClinicalStages(stages=stages)
			outcomes =[trial.val for trial in stages.trial_runs(self.probs)]
			stop_point_counts[len(outcomes)-1] += 1
		return stop_point_counts

	def get_stages(self):
		return [self.state_map.get(stage) for stage in self.stage_list]

if __name__ == '__main__':
	sim = Simulation(1234, 100000, probs=[.7, .3, .5, .40])
	results = sim.run_simulation()
	plt.bar(range(len(results)), results)
	plt.show()
