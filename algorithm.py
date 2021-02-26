from math import exp, sqrt, pi, cos, e
import numpy as np
from random import randint
from copy import deepcopy


class Individual:


    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.f = None
        self.minus = False 

    def evaluation(self, function, mu_1 = None, mu_2 = None, mu_3 = None, sig_1 = None, sig_2 = None, sig_3 = None):
        if function == function_2:
            self.f = function(np.array([[self.x1], [self.x2]]))
            self.minus = True
        else:
            self.f = function(np.array([[self.x1], [self.x2]]), mu_1, mu_2, mu_3, sig_1, sig_2, sig_3)

    def __repr__(self):
        return self.f

    def __str__(self):
        if self.minus:
            return f"[{self.x1}, {self.x2}]\t f(x) = {-self.f}"
        return f"[{self.x1}, {self.x2}]\t f(x) = {self.f}"


def function_phi(x, mu, sigma):
    return (exp(np.dot(np.dot(-0.5*np.transpose(x - mu),np.linalg.inv(sigma)),(x-mu))))/sqrt(pow(2*pi, x.ndim)*np.linalg.det(sigma))

def function_1(x, mu_1, mu_2, mu_3, sig_1, sig_2, sig_3):
    return function_phi(x, mu_1, sig_1) + function_phi(x, mu_2, sig_2) + function_phi(x, mu_3, sig_3)

def function_2(x):
    return 20*exp(-0.2*sqrt(np.dot(0.5*np.transpose(x), x))) + exp(0.5*(cos(2*pi*x[0]) + cos(2*pi*x[1]))) - e - 20

def first_population(size, seed, avrg, sdev):
    population = []
    np.random.seed(seed)
    for i in range(0, size):
        population.append(Individual(np.random.normal(avrg, sdev), np.random.normal(avrg, sdev)))
    return population

def mutation(x, sigma):
    x.x1 += np.random.normal(0, sigma)
    x.x2 += np.random.normal(0, sigma)
    return x

def algorithm(function, pop_size, mut, tournament_size, max_ev, k, seed, mu_1 = None, mu_2 = None, mu_3 = None, sig_1 = None, sig_2 = None, sig_3 = None):
    if function == function_1:
        p0 = first_population(pop_size, seed, 0, 1)
    else:
        p0 = first_population(pop_size, seed, 3, 1)
    evaluations_counter = 0
    best = [] #contains best individuals from every population
    avg = [] #contains average of every population
    evaluations = [] #contains value of evaluations_counter at the end of every population 

    while evaluations_counter <= max_ev:
        next_p = []
        temp = []

        """Evaluations of starting population"""
        for i in p0:
            if function == function_1:
                i.evaluation(function, mu_1, mu_2, mu_3, sig_1, sig_2, sig_3)
            else:
                i.evaluation(function)
            evaluations_counter += 1

        """Choosing the elite"""
        p0 = sorted(p0, key = lambda Individual: Individual.f, reverse = True)
        for i in range (0, k):
            next_p.append(p0[i])
        
        """Tournaments"""
        for i in range(0, pop_size - k):
            tournament = []
            for a in range(0, tournament_size):
                tournament.append(p0[randint(0, pop_size -k - 1)])
            temp.append(sorted(tournament, key = lambda Individual: Individual.f, reverse = True)[0])
        
        """Mutations"""
        for i in range(0, pop_size - k):
            a = deepcopy(temp[i])
            a = mutation(a, mut)
            if function == function_1:
                a.evaluation(function, mu_1, mu_2, mu_3, sig_1, sig_2, sig_3)
            else:
                a.evaluation(function)
            evaluations_counter += 1
            next_p.append(a)
        
        """Saving average, best individualand and number of evaluations for this population"""
        evaluations.append(evaluations_counter)
        if (function == function_1):
            best.append(next_p[0].f)
        else: 
            best.append(-next_p[0].f)
        avg.append(average(next_p))

        p0 = next_p
        #print(f"{evaluations_counter}. {next_p[0]}")


    return p0[0], average(p0), evaluations, best, avg

def average(tab):
    sum = 0
    for i in tab:
        if i.minus:
            sum -= i.f
        else:
            sum += i.f
    return sum/len(tab)



