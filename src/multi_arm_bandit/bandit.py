from __future__ import division, print_function

import numpy as np
import random

class MultiArmBandit(object):
	"""docstring for MultiArmBandit."""
	def __init__(self, arms, est = None, mode=1, epsilon=0.1):
		"""
		mode = 1 epsilon greedy
		mode = 2 epsilon decreasing
		"""
		super(MultiArmBandit, self).__init__()
		if len(set(arms)) != len(arms):
			print('error')
			return False
		self.arms = arms
		if est is None or len(est) == 0:
			self.est = np.ones(len(self.arms))
			self.est_hist = [[] for _ in range(0,len(self.arms))]
		else:
			self.est = np.array(est)
			self.est_hist = [[0] for e in est]
		self.mode = mode
		if self.mode == 1:
			self.epsilon = epsilon
		elif self.mode == 2:
			self.epsilon = 1.0
			self.t = 0
		self.removed = set([])
		self._window_size = 10  # n for the moving average

		self._vdbe_window = 10
		self._vdbe_hist = []
		self._discount = 0.8
		self._d = {}


	def getDiscount(self, s):
		if s not in self._d:
			self._d[s] = self._discount ** s
		return self._d[s]


	def nextArm(self, arms=None, rprob=0.0):
		if len(self.arms) == len(self.removed):
			return False
		if self.mode == 1:
			return self.epsilonGreedy(arms=arms, rprob=rprob)
		elif self.mode == 2:
			return self.epsilonDecreasing(arms=arms, rprob=rprob)
		elif self.mode == 3:
			return self.vdbe(arms=arms, rprob=rprob)
		elif self.mode == 4:
			return self.ucb(arms=arms)

	def getEst(self, arm=None):
		#print(self.est)
		if arm is None:
			return self.est
		return self.est[arm]

	def decreasingFunction(self, t):
		# Logistic function
		k = 0.1     # slope
		l = 0.1     # min prob value
		m = 50      # mid point
		return 1 - (1 - l) / (1 + np.exp(-k * (t - m)))

	def epsilonDecreasing(self, arms=None, rprob=0.0):
		if arms is None or len(arms) == 0:
			arms = self.arms
		self.t += 1

		# The decreasing function
		self.epsilon = self.decreasingFunction(self.t)
		r = np.random.random()

		if r > self.epsilon:
			est = np.array([self.est[i] if self.arms[i] in arms
			 else np.NINF for i in range(0, len(self.arms))])
			
			if np.random.random() > rprob:
				# Select highest
				i = np.random.choice(np.where(est == est.max())[0])
			else:
				# Select min
				i = np.random.choice(np.where(est == est.min())[0])
			return self.arms[i], self.epsilon
		#else:
		#	x = [a for a in self.arms if a in arms\
		#	 and a not in self.removed]
		i = np.random.choice(range(0, len(arms)))
		return self.arms[i], self.epsilon

		

	def epsilonGreedy(self, arms=None, rprob=0.0):
		"""
		With probability epsilon, select the best arm.
		With probability (1-epsilon), select a random arm.
		"""
		if arms is None or len(arms) == 0:
			arms = self.arms
		
		r = np.random.random()
		if r > self.epsilon:
			# Select the best arm
			# If multiple best arms, select randomly
			est = np.array([self.est[i] if self.arms[i] in arms\
			 else np.NINF for i in range(0, len(self.arms))])	
			if np.random.random() > rprob:
				# Select highest
				i = np.random.choice(np.where(est == est.max())[0])
			else:
				# Select min
				i = np.random.choice(np.where(est == est.min())[0])

			return self.arms[i], self.epsilon
			
		i = np.random.choice(range(0, len(arms)))
		return arms[i], self.epsilon


	def vdbe(self, arms=None, rprob=0.0):
		"""
		epsilon high if high variance
		"""
		if arms is None or len(arms) == 0:
			arms = self.arms

		# Max variance of bounded variable (0,1) is 0.25
		#var = np.var(self._vdbe_hist[-self._vdbe_window:])
		#prob = var / 0.25

		prob = 1.0

		if len(self._vdbe_hist) > 0:
			#prob = 1 - np.mean(self._vdbe_hist[-self._vdbe_window:]) / max(self._vdbe_hist)
			#prob = max(prob, 0.0)

			var = np.var(self._vdbe_hist[-self._vdbe_window:])
			#mvar = max(self._vdbe_hist)**2 / 4
			#mvar = max(self._vdbe_hist)**2 / 4
			mvar = max(self._vdbe_hist)**2

			if var == 0 or mvar == 0:
				prob = 0.0
			else:
				prob = var / mvar

		#print('Prob',prob)
		r = np.random.random()
		if r > prob:
			# Select the best arm
			# If multiple best arms, select randomly
			est = np.array([self.est[i] if self.arms[i] in arms\
			 else np.NINF for i in range(0, len(self.arms))])	
			if np.random.random() > rprob:
				# Select highest
				i = np.random.choice(np.where(est == est.max())[0])
			else:
				# Select min
				i = np.random.choice(np.where(est == est.min())[0])

			return self.arms[i], prob
			
		i = np.random.choice(range(0, len(arms)))
		return arms[i], prob


	def ucb(self, arms=None, rprob=0.0):
		if arms is None or len(arms) == 0:
			arms = self.arms

		mval = -1
		marm = 0
		n = np.sum([len(r) for r in self.est_hist])

		if n > 10:
			t = len(self._vdbe_hist)
			n0 = np.sum([self.getDiscount(t-s) for s in range(0, t)])
			for a in arms:
				i = self.arms.index(a)
				t = len(self.est_hist[i])
				if len(self.est_hist[i]) > 0:
					n1 = [self.getDiscount(t-s) for s in range(0, t)]
					#v = np.mean(self.est_hist[i]) + np.sqrt(2 * np.log(len(self._vdbe_hist))/ len(self.est_hist[i]))
					v_0 = np.sum([self.est_hist[i][s] * n1[s] for s in range(0, t)]) / np.sum(n1)
					#v_1 = np.sqrt( 2 * np.log10(n) / max(1,len(self.est_hist[i])))
					v_1 = 0.1 * np.sqrt( np.log10(n0) / max(1,np.sum(n1)))
					v = v_0 + v_1
					#print(a,v, v_0, v_1)
					if v > mval:
						mval = v
						marm = a

		if mval < 0:
			i = np.random.choice(range(0,len(arms)))
			return arms[i], 0

		return marm, mval


	def removeArm(self, arm):
		self.removed.update([arm])
		self.est[arm] = np.NINF

	def updateEst(self, reward, arm):
		i = self.arms.index(arm)
		self.est_hist[i].append(reward)
		self.est[i] = np.mean(self.est_hist[i][-self._window_size :])
		#print(self.est[i])
		self._vdbe_hist.append(reward)
		return self.est[i]


if __name__ == '__main__':
	print('Nope')
