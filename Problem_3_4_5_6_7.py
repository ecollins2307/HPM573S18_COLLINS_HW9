# HW 9, Problems 3, 4, 5, 6, and 7

# imports, requires newest version of SupportLib to be loaded in content root and  Anaconda to be installed locally and chosen as the interpreter
from enum import Enum
import numpy as np
import scipy.stats as stat
import math as math
import scr.ProbDistParEst as Est
import scr.SamplePathClasses as PathCls
import scr.StatisticalClasses as StatCls
import scr.RandomVariantGenerators as rndClasses
import scr.FormatFunctions as F
import scr.FigureSupport as Figs


# Parameter classes, modified from ParameterClasses.py
class HealthStats(Enum):
    """ health states of patients """
    WELL = 0
    STROKE = 1
    POSTSTROKE = 2
    DEATH = 3


class Therapies(Enum):
    """ none vs. anticoag therapy """
    NONE = 0
    ANTICOAG = 1

class ParametersFixed():
    def __init__(self, therapy):

        # selected therapy
        self._therapy = therapy

        # simulation time step
        self._delta_t = DELTA_T

        # initial health state
        self._initialHealthState = HealthStats.WELL

        # transition probability matrix of the selected therapy
        if therapy == Therapies.NONE:
            self._prob_matrix = TRANS_MATRIX
        elif therapy == Therapies.ANTICOAG:
            self._prob_matrix = TRANS_MATRIX_ANTICOAG

    def get_initial_health_state(self):
        return self._initialHealthState

    def get_delta_t(self):
        return self._delta_t

    def get_transition_prob(self, state):
        return self._prob_matrix[state.value]

# Markov classes, modified from MarkovModelClasses.py
class Patient:
    def __init__(self, id, parameters):
        """ initiates a patient
        :param id: ID of the patient
        :param parameters: parameter object
        """

        self._id = id
        # random number generator for this patient
        self._rng = None
        # parameters
        self._param = parameters
        # state monitor
        self._stateMonitor = PatientStateMonitor(parameters)
        # simulation time step
        self._delta_t = parameters.get_delta_t()

    def simulate(self, sim_length):
        """ simulate the patient over the specified simulation length """

        # random number generator for this patient
        self._rng = rndClasses.RNG(self._id)

        k = 0  # current time step

        # while the patient is alive and simulation length is not yet reached
        while self._stateMonitor.get_if_alive() and k*self._delta_t < sim_length:

            # find the transition probabilities of the future states
            trans_probs = self._param.get_transition_prob(self._stateMonitor.get_current_state())
            # create an empirical distribution
            empirical_dist = rndClasses.Empirical(trans_probs)
            # sample from the empirical distribution to get a new state
            # (returns an integer from {0, 1, 2, ...})
            new_state_index = empirical_dist.sample(self._rng)

            # update health state
            self._stateMonitor.update(k, HealthStats(new_state_index))

            # increment time step
            k += 1

    def get_survival_time(self):
        """ returns the patient's survival time"""
        return self._stateMonitor.get_survival_time()

    def get_time_to_STROKE(self):
        """ returns the patient's time to STROKE """
        return self._stateMonitor.get_time_to_STROKE()

class PatientStateMonitor:
    """ to update patient outcomes (years survived, cost, etc.) throughout the simulation """
    def __init__(self, parameters):
        """
        :param parameters: patient parameters
        """
        self._currentState = parameters.get_initial_health_state() # current health state
        self._delta_t = parameters.get_delta_t()    # simulation time step
        self._survivalTime = 0          # survival time
        self._numberofstrokes = 0        # number of STROKE

    def update(self, k, next_state):
        """
        :param k: current time step
        :param next_state: next state
        """

        # if the patient has died, do nothing
        if not self.get_if_alive():
            return

        # update survival time
        if next_state == HealthStats.DEATH:
            self._survivalTime = (k+0.5)*self._delta_t  # corrected for the half-cycle effect

        # update time until STROKE
        if self._currentState != HealthStats.POSTSTROKE and next_state == HealthStats.POSTSTROKE:
            self._numberofstrokes = self._numberofstrokes + 1  # had a stroke

        # update current health state
        self._currentState = next_state

    def get_if_alive(self):
        result = True
        if self._currentState == HealthStats.DEATH:
            result = False
        return result

    def get_current_state(self):
        return self._currentState

    def get_survival_time(self):
        """ returns the patient survival time """
        # return survival time only if the patient has died
        if not self.get_if_alive():
            return self._survivalTime
        else:
            return None

    def get_time_to_STROKE(self):
        """ returns the number of strokes """
        # return number of strokes
        return self._numberofstrokes

class Cohort:
    def __init__(self, id, therapy):
        """ create a cohort of patients
        :param id: an integer to specify the seed of the random number generator
        """
        self._initial_pop_size = POP_SIZE
        self._patients = []      # list of patients

        # populate the cohort
        for i in range(self._initial_pop_size):
            # create a new patient (use id * pop_size + i as patient id)
            patient = Patient(id * self._initial_pop_size + i, ParametersFixed(therapy))
            # add the patient to the cohort
            self._patients.append(patient)


    def simulate(self):
        """ simulate the cohort of patients over the specified number of time-steps
        :returns outputs from simulating this cohort
        """
        # simulate all patients
        for patient in self._patients:
            patient.simulate(SIM_LENGTH)

        # return the cohort outputs
        return CohortOutputs(self)

    def get_initial_pop_size(self):
        return self._initial_pop_size

    def get_patients(self):
        return self._patients

class CohortOutputs:
    def __init__(self, simulated_cohort):
        """ extracts outputs from a simulated cohort
        :param simulated_cohort: a cohort after being simulated
        """

        self._survivalTimes = []        # patients' survival times
        self._times_to_STROKE = []        # patients' times to STROKE
         # survival curve
        self._survivalCurve = \
            PathCls.SamplePathBatchUpdate('Population size over time', id, simulated_cohort.get_initial_pop_size())

        # find patients' survival times
        for patient in simulated_cohort.get_patients():

            # get the patient survival time
            survival_time = patient.get_survival_time()
            if not (survival_time is None):
                self._survivalTimes.append(survival_time)           # store the survival time of this patient
                self._survivalCurve.record(survival_time, -1)       # update the survival curve

            # get the patient's time to STROKE
            time_to_STROKE = patient.get_time_to_STROKE()
            self._times_to_STROKE.append(time_to_STROKE)


        # summary statistics
        self._sumStat_survivalTime = StatCls.SummaryStat('Patient survival time', self._survivalTimes)
        self._sumState_timeToSTROKE = StatCls.SummaryStat('Average number of strokes', self._times_to_STROKE)

    def get_survival_times(self):
        return self._survivalTimes

    def get_times_to_STROKE(self):
        return self._times_to_STROKE

    def get_sumStat_survival_times(self):
        return self._sumStat_survivalTime

    def get_sumStat_time_to_STROKE(self):
        return self._sumState_timeToSTROKE

    def get_survival_curve(self):
        return self._survivalCurve

# Markov support functions, based on SupportMarkovModel.py
def print_outcomes(simOutput, therapy_name, Problem7):
    """ prints the outcomes of a simulated cohort
    :param simOutput: output of a simulated cohort
    :param therapy_name: the name of the selected therapy
    :param Problem7: is this for problem 7?
    """
    # mean and confidence interval text of patient survival time
    survival_mean_CI_text = F.format_estimate_interval(
        estimate=simOutput.get_sumStat_survival_times().get_mean(),
        interval=simOutput.get_sumStat_survival_times().get_t_CI(alpha=ALPHA),
        deci=2)

    # mean and confidence interval text of time to STROKE
    time_to_STROKE_death_CI_text = F.format_estimate_interval(
        estimate=simOutput.get_sumStat_time_to_STROKE().get_mean(),
        interval=simOutput.get_sumStat_time_to_STROKE().get_t_CI(alpha=ALPHA),
        deci=2)

    # print outcomes
    print(therapy_name)
    if Problem7 == "No":
        print("  Estimate of mean survival time and {:.{prec}%} confidence interval:".format(1 - ALPHA, prec=0),
          survival_mean_CI_text)

    elif Problem7 == "Yes":
        print("  Estimate of mean number of strokes and {:.{prec}%} confidence interval:".format(1 - ALPHA, prec=0),
          time_to_STROKE_death_CI_text)


# Problem 3
# Input data
# simulation settings
POP_SIZE = 2000     # cohort population size
SIM_LENGTH = 50    # length of simulation (years)
ALPHA = 0.05        # significance level for calculating confidence intervals
DELTA_T = 1       # years

# transition matrix, made up numbers to fit probabilities
TRANS_MATRIX = [
    [0.75,  0.15,    0,    0.1],   # WELL
    [0,     0,    1,    0],   # STROKE
    [0,     0.25,      0.55,   0.2],   # POSTSTROKE
    [0,     0,      0,   1],   # DEATH
    ]

# create a cohort
cohort = Cohort(
    id=0,
    therapy=Therapies.NONE)

# simulate the cohort
simOutputs = cohort.simulate()

# print the outcomes of this simulated cohort
print("Problem 3:")
print_outcomes(simOutputs, 'WITHOUT anticoagulation', "No")
print("")

# Problem 4
TRANS_MATRIX_ANTICOAG = [
    [0.75,  0.15,    0,    0.1],   # WELL
    [0,     0,    1,    0],   # STROKE
    [0,     0.1625,      0.701,   0.1365],   # POSTSTROKE
    [0,     0,      0,   1],   # DEATH
    ]

# print the transmatrix
print("Problem 4:",'\n',TRANS_MATRIX_ANTICOAG, '\n')

# Problem 5
# create a cohort
cohort2 = Cohort(
    id=0,
    therapy=Therapies.ANTICOAG)

# simulate the cohort
simOutputs2 = cohort2.simulate()

# print the outcomes of this simulated cohort
print("Problem 5:")
print_outcomes(simOutputs2, 'WITH anticoagulation:', "No")
print("")

# Problem 6
# graph survival curves
PathCls.graph_sample_path(
    sample_path=simOutputs.get_survival_curve(),
    title='Problem 6: Survival curve for WITHOUT anticoagulation',
    x_label='Simulation time step',
    y_label='Number of alive patients'
    )

PathCls.graph_sample_path(
    sample_path=simOutputs2.get_survival_curve(),
    title='Problem 6: Survival curve for WITH anticoagulation',
    x_label='Simulation time step',
    y_label='Number of alive patients'
    )

# Problem 7
# means without
print("Problem 7:")
print_outcomes(simOutputs, 'WITHOUT anticoagulation', "Yes")
print("")
# graph histogram
Figs.graph_histogram(
    data=simOutputs.get_times_to_STROKE(), # not really TIMES to stroke, but I was lazy and didn't want to change it
    title='Number of strokes in patients WITHOUT anticoagulation',
    x_label='Number of strokes',
    y_label='Counts',
    bin_width=1
)

# means with
print_outcomes(simOutputs2, 'WITH anticoagulation', "Yes")
print("")
# graph histogram
Figs.graph_histogram(
    data=simOutputs2.get_times_to_STROKE(), # not really TIMES to stroke, but I was lazy and didn't want to change it
    title='Number of strokes in patients WITH anticoagulation',
    x_label='Number of strokes',
    y_label='Counts',
    bin_width=1
)
