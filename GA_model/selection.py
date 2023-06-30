from numpy.random import randint

# defining tournament selection
# More the selection pressure (probabilistic measure of a candidate’s likelihood of participation in a tournament) more will be the Convergence rate.
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	# generate 2 random inx and compare them with third random index, lowest among three is the winner
	for ix in randint(0, len(pop), k-1): # k-1 numbers between 0 and 99
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]




