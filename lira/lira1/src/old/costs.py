import numpy as np
import theano
import theano.tensor as T

class log_likelihood(object):
    "Return the log-likelihood cost."
    #-mean( yln(a) )
    @staticmethod
    def get_cost(a, y, n):

        #When we do [T.arange(y.shape[0]), y]
        #It converts our vector of outputs for y to something like
        #[[0, 1, 2, 3, 4], [7, 2, 5, 6]]
        #Where y[0] is our index of the value
        #Where y[1] is the y value
        return -T.mean(T.log(a)[T.arange(y.shape[0]), y])

class cross_entropy(object):
    "Return Cross-Entropy Cost"
    #Best when combined with sigmoid output layer
    #Consider rewriting this to make sense in terms of matrices
    @staticmethod
    def get_cost(a, y, n):
        #-mean( y*ln(a) + (1-y)*ln(1-a) )

        #Inverted log a is (1-y)ln(1-a)
        #This is basically taking ln(1-a) and assigning all the ones that are equal to the index of y to 0,
        #Since we'd have y as [0, 0, 1, 0] then 1-y = [1, 1, 0, 1] so we keep everything the same
        #Except for those matching the index, which we make = 0
        inverted_log_a = T.set_subtensor(T.log(1-a)[T.arange(y.shape[0]), y], 0.0)
        f = theano.function(inputs=[a, y], outputs=inverted_log_a)
        return -T.mean(
                    T.log(a)[T.arange(y.shape[0]), y]
                    +
                    T.sum(inverted_log_a, axis=1)
                )

class quadratic(object):
    "Return Quadratic Cost"
    @staticmethod
    def get_cost(a, y, n):
        #     ( ||y-a||^2 )
        #mean ( --------- )
        #     (    2      )

        #a_update = (a, T.set_subtensor(a[T.arange(y.shape[0]), y], 1 - a[T.arange(y.shape[0]), y]))
        #f = function([y], updates=[a_update])

        #We have each y as a one hot vector like [0, 0, 1, 0] but from everything i've worked out so far it's better to
        #do 1 - a for the a values that are the same as y, so if we had 
        #y = [0, 0, 1, 0]
        #a = [.1, .1, .8, 0]
        #It's better to get result of 
        #y-a = [.1, .1, .2, 0], rather than getting massive values that make the cost way bigger if we were to make the .1 = 0-.1 = .9
        #since we'd then be squaring it elementwise right after, and since the closer it gets to 0, which we want(since it's not the right output)
        #the higher the value would get, i.e. 0.00001 = .99999^2, that doesn't make any sense at all. I hope i'm not doing something completely wrong,
        #But what I did was set the new a to be a with our 1-a indices of y.
        #Then we square, sum on axis 1, get mean, and then * 1/2 in accordance with the earlier equation.
        new_a = T.set_subtensor(a[T.arange(y.shape[0]), y], 1 - a[T.arange(y.shape[0]), y])
        f = theano.function(inputs=[a, y], outputs=new_a)

        return T.dot((1.0/2.0), T.mean(T.sum(T.sqr(new_a), axis=1)))
