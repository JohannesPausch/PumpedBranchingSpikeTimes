import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special as sc

# SpikeTimeTheory.py is a program for calculating the first five moments of inter-spike intervals of a pumped branching process
# created by Johannes Pausch
# copyright (2020) Johannes Pausch
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>. 




# used formulas are only exact for binary branching
# careful with how many bits longdouble precision corresponds to your machine


def creationProbability(r,gamma,s,p2,state):
    if state >1 :
        return np.longdouble(2*(r/(r+s))*(s*(state-1)/gamma+1)*((s*p2/(r+s*p2))**(state-1))*((r/(r+s*p2))**(gamma/(s*p2)))*(s*p2*(state-1)+gamma)/((state-1)*sc.beta(gamma/(s*p2),state-1)*(s*(state-1)+gamma)))
    else:
        return np.longdouble(2*(r/(r+s))*(r/(r+s*p2))**(gamma/(s*p2)))

def firstSpikeMoment(r,gamma,s,p2,precision):
    state = 1
    firstmoment = np.longdouble(0.0)
    previousSum = np.longdouble(1.0/(gamma*sc.beta(gamma/s,1)))
    powerp2 = np.longdouble(1.0)
    while state <= precision:
        powerp2 = powerp2*np.longdouble(1.0-p2)
        previousSum += np.longdouble(1.0)/(np.longdouble((state*s+gamma))*np.longdouble(sc.beta(gamma/s,state+1))*powerp2)
        firstmoment += previousSum*creationProbability(r,gamma,s,p2,state)*powerp2*np.longdouble(sc.beta(gamma/s,state+1))
        state += 1
    return firstmoment

def secondSpikeMoment(r,gamma,s,p2,precision):
    state = 1
    secondmoment = np.longdouble(0.0)
    previousSum2 = np.longdouble(1.0/(gamma*sc.beta(gamma/s,1)))
    previousSum1 = np.longdouble(2.0/(gamma**2*sc.beta(gamma/s,1)))
    powerp2 = np.longdouble(1.0)
    while state <= precision:
        powerp2 = powerp2*np.longdouble(1.0-p2)
        previousSum2 += np.longdouble(1.0)/(np.longdouble((state*s+gamma))*np.longdouble(sc.beta(gamma/s,state+1))*powerp2)
        previousSum1 += np.longdouble(2.0)*previousSum2/(np.longdouble(state*s+gamma))
        secondmoment += previousSum1*creationProbability(r,gamma,s,p2,state)*powerp2*np.longdouble(sc.beta(gamma/s,state+1))
        state += 1
    return secondmoment

def thirdSpikeMoment(r,gamma,s,p2,precision):
    state = 1
    thirdmoment = np.longdouble(0.0)
    previousSum2 = np.longdouble(1.0/(gamma*sc.beta(gamma/s,1)))
    previousSum3 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**2))
    previousSum1 = np.longdouble(6.0/(sc.beta(gamma/s,1)*gamma**3))
    powerp2 = np.longdouble(1.0)
    while state <= precision:
        powerp2 = powerp2*np.longdouble(1.0-p2)
        previousSum2 += np.longdouble(1.0)/(np.longdouble(state*s+gamma)*np.longdouble(sc.beta(gamma/s,state+1))*powerp2)
        previousSum3 += previousSum2/np.longdouble(state*s+gamma)
        previousSum1 += np.longdouble(6.0)*previousSum3/(np.longdouble(state*s+gamma))
        thirdmoment += previousSum1*creationProbability(r,gamma,s,p2,state)*powerp2*np.longdouble(sc.beta(gamma/s,state+1))
        state += 1
    return thirdmoment

def fourthSpikeMoment(r,gamma,s,p2,precision):
    state = 1
    fourthmoment = np.longdouble(0.0)
    previousSum2 = np.longdouble(1.0/(gamma*sc.beta(gamma/s,1)))
    previousSum3 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**2))
    previousSum4 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**3))
    previousSum1 = np.longdouble(24.0/(sc.beta(gamma/s,1)*gamma**4))
    powerp2 = np.longdouble(1.0)
    while state <= precision:
        powerp2 = powerp2*np.longdouble(1.0-p2)
        previousSum2 += np.longdouble(1.0)/(np.longdouble(state*s+gamma)*np.longdouble(sc.beta(gamma/s,state+1))*powerp2)
        previousSum3 += previousSum2/np.longdouble(state*s+gamma)
        previousSum4 += previousSum3/np.longdouble(state*s+gamma)
        previousSum1 += np.longdouble(24.0)*previousSum4/(np.longdouble(state*s+gamma))
        fourthmoment += previousSum1*creationProbability(r,gamma,s,p2,state)*powerp2*np.longdouble(sc.beta(gamma/s,state+1))
        state += 1
    return fourthmoment

def fifthSpikeMoment(r,gamma,s,p2,precision):
    state = 1
    fifthmoment = np.longdouble(0.0)
    previousSum2 = np.longdouble(1.0/(gamma*sc.beta(gamma/s,1)))
    previousSum3 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**2))
    previousSum4 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**3))
    previousSum5 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**4))
    previousSum1 = np.longdouble(120.0/(sc.beta(gamma/s,1)*gamma**5))
    powerp2 = np.longdouble(1.0)
    while state <= precision:
        powerp2 = powerp2*np.longdouble(1.0-p2)
        previousSum2 += np.longdouble(1.0)/(np.longdouble(state*s+gamma)*np.longdouble(sc.beta(gamma/s,state+1))*powerp2)
        previousSum3 += previousSum2/np.longdouble(state*s+gamma)
        previousSum4 += previousSum3/np.longdouble(state*s+gamma)
        previousSum5 += previousSum4/np.longdouble(state*s+gamma)
        previousSum1 += np.longdouble(120.0)*previousSum5/(np.longdouble(state*s+gamma))
        fifthmoment += previousSum1*creationProbability(r,gamma,s,p2,state)*powerp2*np.longdouble(sc.beta(gamma/s,state+1))
        state += 1
    return fifthmoment


s = 1.0 # rate of binary branching
r = 0.1 # effectice extinction rete / effective mass in field theory
g = 1.0 # relative spontaneous creation (g=gamma/s, where gamma is the spontaneous creation rate)
precision = 16000 # maximum particle number that is included in calculation, requires at least 128bit double precision


print('# pumped branching\n# binary branching\n# first five moments\n# 1st column rate s\n# 2nd column effective mass r\n#  3rd column relative spontaneous creation gamma/s \n# column 4 to 8: 1st to 5th moment')

# (1-r/s)/2 = p0 = probability for a single particle to go extinct at a branching/extinction event
firstmoment = firstSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
secondmoment = secondSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
thirdmoment = thirdSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
fourthmoment = fourthSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
fifthmoment = fifthSpikeMoment(r,g*s,s,(1-r/s)/2,precision)

print(str(s)+'\t'+str(r)+'\t'+str(g)+'\t'+str(firstmoment)+'\t'+str(secondmoment)+'\t'+str(thirdmoment)+'\t'+str(fourthmoment)+'\t'+str(fifthmoment))
