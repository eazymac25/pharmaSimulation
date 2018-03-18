# test
from numpy import random 
from math import floor, ceil
import matplotlib.pyplot as plt

# Trials are a Directed Acyclic Graph (DAG) -> for simplicity it's a singly linked list
# we can represent this by using a list [0, 0, 0, 0]
# This would be a drug that hasn't move though any trials

class NPV(object):

	def __init__(self, fcf, initial_cost=0, end_year=11, start_year=0, prob=1., r=0.1):
		# argument validation
		if not isinstance(prob, list) and not isinstance(prob, float):
			raise ValueError('please insert correct prob as a list of floats or a float')
		if not isinstance(fcf, float) and not isinstance(fcf, list):
			raise ValueError('please include a free cash flow as a float or list of floats')

		self.fcf = fcf
		self.start_year = start_year
		self.end_year = end_year
		self.initial_cost = initial_cost
		self.prob = prob
		self.r = r

	def calc(self):
		npv = -self.initial_cost
		for t in range(self.start_year, self.end_year):
			prob = self.prob
			fcf = self.fcf
			if isinstance(self.prob, list):
				prob = self.prob[t]
			if isinstance(self.fcf, list):
				fcf = self.fcf[t]
			npv += (prob*fcf)/((1.+self.r)**t)
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
		super(Fail, self).__init__()
		self.val = self.FAIL

class ClinicalStages(object):

	def __init__(self, stages=[], stage_len=[], start_year=0, initial_cost=0, yearly_cost=[], yearly_rev=[], cost_dist=(), rev_dist=()):
		FIRST_STAGE = 0
		self.state = stages[FIRST_STAGE]()
		self.stages = stages
		self.stage_len = stage_len
		self.start_year = start_year
		self.initial_cost = initial_cost
		self.yearly_cost = yearly_cost
		self.yearly_rev = yearly_rev
		self.cost_dist = cost_dist
		self.rev_dist = rev_dist

	def trial_runs(self, probs):
		idx = 0
		last_state = self.state
		while True:
			if isinstance(self.state, Fail):
				end_year = self.get_end_year(idx)
				fcf = self.get_fcf()
				npv = NPV(fcf, initial_cost=self.initial_cost, start_year=self.start_year, end_year=end_year)
				npv_val = npv.calc()
				return ('FAILED', last_state, npv_val)
				break
			if isinstance(self.state, Complete):
				fcf = self.get_fcf()
				npv = NPV(fcf, initial_cost=self.initial_cost, start_year=self.start_year)
				npv_val = npv.calc()
				return ('DONE', self.state, npv_val)
				break
			new_state = self.state.transition(probs[idx], self.stages[idx])
			last_state = self.state
			self.state = new_state
			idx += 1

	def get_end_year(self, step):
		year = 0
		for i in range(step):
			year+=self.stage_len[i]
		return year

	def get_fcf(self):
		if self.cost_dist and self.rev_dist:
			costs, revs = self._randomize_cash_flows()
			return [revs[i] - costs[i] 
				for i in range(len(costs))]

		return [self.yearly_rev[i] - self.yearly_cost[i]
			for i in range(len(self.yearly_cost))]

	def _randomize_cash_flows(self):
		cost_dist_funcion, cost_dist_args = self.cost_dist
		rev_dist_funcion, rev_dist_args = self.rev_dist
		costs = []
		revs = []
		for idx in range(len(self.yearly_cost)):
			costs.append(self.yearly_cost[idx] + cost_dist_funcion(*cost_dist_args))
			revs.append(self.yearly_rev[idx] + rev_dist_funcion(*rev_dist_args))
		return costs, revs

class Simulation(object):

	state_map = {
		'PRE': PreTrial,
		'P1': P1,
		'P2': P2,
		'P3': P3,
		'NDA': NDA,
		'DONE': Complete
	}

	def __init__(self, drug_id, num_trials,
		stage_list=('P2','P3','NDA','DONE'), start_year=0, stage_len=[2,2,1,4], initial_cost=50,
		yearly_cost=[], yearly_rev=[], cost_dist=(), rev_dist=(), probs=[], distributions=[]):

		self.drug_id = drug_id
		self.num_trials = num_trials
		self.stage_list = stage_list
		self.stage_len = stage_len
		self.start_year = start_year
		self.initial_cost = initial_cost
		self.yearly_cost = yearly_cost
		self.yearly_rev = yearly_rev
		self.cost_dist = cost_dist
		self.rev_dist = rev_dist
		self.probs = probs
		self.distributions = distributions

	def run_simulation(self):
		stages = self.get_stages()
		outcomes = []
		for run in range(self.num_trials):
			clinical_trials = ClinicalStages(
				stages=stages,
				stage_len=self.stage_len,
				initial_cost=self.initial_cost,
				yearly_cost=self.yearly_cost,
				yearly_rev=self.yearly_rev,
				cost_dist=self.cost_dist,
				rev_dist=self.rev_dist,
				start_year=self.start_year
			)
			outcomes.append(clinical_trials.trial_runs(self.probs))
		return outcomes

	def get_stages(self):
		return [self.state_map.get(stage) for stage in self.stage_list]

if __name__ == '__main__':

	sim = Simulation(
		1234,
		10000,
		start_year=2,
		yearly_cost=[1., 1., 4., 4., 4., 1., 1., 2., 2., 1., 1.],
		yearly_rev = [0., 0., 0., 0., 0., 10., 50., 80., 70., 70., 50.],
		cost_dist = (random.normal, (0, 1)),
		rev_dist = (random.normal, (0, 10)),
		probs=[1, 1, 1, 1]
	)
	results = sim.run_simulation()
	print len(results)
	buckets = []
	for result in results:
		if result[0]=='DONE':
			buckets.append(result[2])
	plt.hist([result[2] for result in results])
	plt.title('NPV')
	plt.show()
