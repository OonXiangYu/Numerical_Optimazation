import unittest
import math
import csv
import random
import copy
import numpy as np
import pyswarms as ps
from simanneal import Annealer
from deap import base, creator, tools

# the unit tests to check that the simulation has been implemented correctly
class UnitTests (unittest.TestCase):
    # this will read in the track locations file and will pick out 5 fields to see if the file has been read correctly
    def testReadCSV(self):
        # read in the locations file
        rows = readCSVFile('track-locations.csv')

        # test that the corners and a middle value are read in correctly
        self.assertEqual('GP', rows[0][0])
        self.assertEqual('Abu Dhabi', rows[0][24])
        self.assertEqual('temp week 52', rows[55][0])
        self.assertEqual('24.4', rows[55][24])
        self.assertEqual('14.5', rows[11][8])
    
    # this will test to see if the row conversion works. here we will convert the latitude rwo and will test 5 values
    # as we are dealing with floating point we will use almost equals rather than a direct equality
    def testRowToFloat(self): 
        # read in the locations file and convert the latitude column to floats
        rows = readCSVFile('track-locations.csv')
        convertRowToFloat(rows, 2)

        # check that 5 of the values have converted correctly
        self.assertAlmostEqual(26.0325, rows[2][1], delta=0.0001)
        self.assertAlmostEqual(24.4672, rows[2][24], delta=0.0001)
        self.assertAlmostEqual(40.3725, rows[2][17], delta=0.0001)
        self.assertAlmostEqual(30.1327, rows[2][19], delta=0.0001)
        self.assertAlmostEqual(25.49, rows[2][23], delta=0.0001)

        # check that the conversion of a temperature row to floating point is also correct
        convertRowToFloat(rows, 5)

        # check that 5 of the values have converted correctly
        self.assertAlmostEqual(20.5, rows[5][1], delta=0.0001)
        self.assertAlmostEqual(24.5, rows[5][24], delta=0.0001)
        self.assertAlmostEqual(7.25, rows[5][17], delta=0.0001)
        self.assertAlmostEqual(16.75, rows[5][19], delta=0.0001)
        self.assertAlmostEqual(22.5, rows[5][23], delta=0.0001)
    
    # this will test to see if the file conversion overall is successful for the track locations
    # it will read in the file and will test a string, float, and int from 2 rows to verify it worked correctly
    def testReadTrackLocations(self): 
        # read in the locations file
        rows = readTrackLocations()

        # check the name, latitude, and final temp of the first race
        self.assertEqual(rows[0][0], 'Bahrain')
        self.assertAlmostEqual(rows[2][0], 26.0325, delta=0.0001)
        self.assertAlmostEqual(rows[55][0], 20.6, delta=0.0001)

        # check the name, longitude, and initial temp of the last race        
        self.assertEqual(rows[0][23], 'Abu Dhabi')
        self.assertAlmostEqual(rows[2][23], 24.4672, delta=0.0001)
        self.assertAlmostEqual(rows[4][23], 24, delta=0.0001)
    
    # tests to see if the race weekends file is read in correctly
    def testReadRaceWeekends(self): 
        # read in the race weekends file
        weekends = readRaceWeekends()

        # check that bahrain is weekend 9 and abu dhabi is weekend 49
        self.assertEqual(weekends[0], 9)
        self.assertEqual(weekends[23], 49)

        # check that hungaroring is weekend 29
        self.assertEqual(weekends[12], 29)

    # this will test to see if the haversine function will work correctly we will test 4 sets of locations
    def testHaversine(self):
        # read in the locations file with conversion
        rows = readTrackLocations()

        # check the distance of Bahrain against itself this should be zero
        self.assertAlmostEqual(haversine(rows, 0, 0), 0.0, delta=0.01)
        
        # check the distance of Bahrain against Silverstone this should be 5158.08 km
        self.assertAlmostEqual(haversine(rows, 0, 11), 5158.08, delta=0.01)

        # check the distance of silverstone against monza this should be 1039.49 Km
        self.assertAlmostEqual(haversine(rows, 11, 15), 1039.49, delta=0.01)

        # check the distance of monza to the red bull ring this should be 455.69 Km
        self.assertAlmostEqual(haversine(rows, 10, 15), 455.69, delta=0.01)
    
    # will test to see if the season distance calculation is correct using the 2023 calendar
    def testDistanceCalculation(self):
        # read in the locations & race weekends, generate the weekends, and calculate the season distance
        tracks = readTrackLocations()
        weekends = readRaceWeekends()
        
        # calculate the season distance using silverstone as the home track as this will be the case for 8 of the teams we will use monza
        # for the other two teams.
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 11), 193645.3219, delta=0.0001)
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 15), 191408.7483, delta=0.0001)
    
    # will test that the temperature constraint is working this should fail as azerbijan should fail the test
    def testTempConstraint(self):
        # load in the tracks, race weekends, and the sundays
        tracks = readTrackLocations()
        weekends1 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 34, 35, 37, 38, 42, 43, 44, 47, 48, 49]
        weekends2 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 31, 35, 37, 38, 42, 43, 44, 40, 48, 49]

        # the test with the default calender should be false because of zaandvort
        self.assertEqual(checkTemperatureConstraint(tracks, weekends1, 20, 35), False)
        self.assertEqual(checkTemperatureConstraint(tracks, weekends2, 20, 35), True)
    
    # will test that we can detect four race weekends in a row.
    def testFourRaceInRow(self):
        # weekend patterns the first does not have four in a row the second does
        weekends1 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 34, 35, 37, 38, 42, 43, 44, 47, 48, 49]
        weekends2 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 35, 36, 37, 38, 42, 43, 44, 47, 48, 49]

        # the first should pass and the second should fail
        self.assertEqual(checkFourRaceInRow(weekends1), False)
        self.assertEqual(checkFourRaceInRow(weekends2), True)
    
    # will test that we can detect a period for a summer shutdown in the prescribed weeks
    def testSummerShutdown(self):
        # weekend patterns the first has a summer shutdown the second doesn't
        weekends1 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 34, 35, 37, 38, 42, 43, 44, 47, 48, 49]
        weekends2 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 33, 35, 37, 38, 42, 43, 44, 47, 48, 49]

        # the first should pass and the second should fail
        self.assertEqual(checkSummerShutdown(weekends1), True)
        self.assertEqual(checkSummerShutdown(weekends2), False)
        
# function that will calculate the total distance for the season assuming a given racetrack as the home racetrack
# the following will be assumed:
# - on a weekend where there is no race the team will return home
# - on a weekend in a double or triple header a team will travel straight to the next race and won't go back home
# - the preseason test will always take place in Bahrain
# - for the summer shutdown and off season the team will return home
def calculateSeasonDistance(tracks, weekends, home):

    race = [9,10,12,14,16,18,20,21,23,25,26,27,29,30,34,35,37,38,42,43,44,47,48,49]

    current = home
    total = 0
    j = 0
    

    for week in race:
        idx = weekends.index(week)

        total  += haversine(tracks, current, idx)
        current = idx

        if j + 1 < len(race) and race[j + 1] == week + 1: # if double/triper header
            j += 1
            continue
        else:
            total += haversine(tracks, current, home) 
            current = home

        j += 1


    if current != home:
        total += haversine(tracks, current, home)
        current = home

    return total


# function that will check to see if there is anywhere in our weekends where four races appear in a row. True indicates that we have four in a row
def checkFourRaceInRow(weekends):
    for startWeek in range(1, 48): 
        if all(week in weekends for week in range(startWeek, startWeek + 4)):  # Four-week gap
            return True
    return False

# function that will check to see if the temperature constraint for all races is satisfied. The temperature
# constraint is that a minimum temperature of min degrees for the month is required for a race to run
def checkTemperatureConstraint(tracks, weekends, min, max):
    j = 0

    for i in range(23):
        if tracks[weekends[j] + 4][i] < min or tracks[weekends[j] + 4][i] > max:
            return False
        else:
            j += 1        

    return True

# function that will check to see if there is a four week gap anywhere in july and august. we will need this for the summer shutdown.
# the way this is defined is that we have a gap of three weekends between successive races. this will be weeks 31, 32, and 33, they are not
# permitted to have a race during these weekends
def checkSummerShutdown(weekends):
    summer = range(27,36) # start of July until end of August

    for startWeek in range(27,33): # need four week to summer shutdown
        if all(week not in weekends for week in range(startWeek, startWeek + 3)): # four week gap
            return True

    return False

# function that will take in the set of rows and will convert the given row index into floating point values
# this assumes the header in the CSV file is still present so it will skip the first column
def convertRowToFloat(rows, row_index):
    for i in range(len(rows[row_index])):
        rows[row_index][i] = try_convert(rows[row_index][i]) # convert str to float

def try_convert(value): # try to convert it to float 
        try:
            return float(value)
        except ValueError:
            return value

# function that will use the haversine formula to calculate the distance in Km given two latitude/longitude pairs
# it will take in an index to two rows, and extract the latitude and longitude before the calculation.
def haversine(rows, location1, location2):
    earthRadius = 6371.0 # Earth Radius

    # Calculate latitude and longtitude
    lat1 = math.radians(rows[2][location1])
    long1 = math.radians(rows[3][location1])
    lat2 = math.radians(rows[2][location2])
    long2 = math.radians(rows[3][location2])

    delta_lat = lat2 - lat1
    delta_long = long2 - long1

    a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_long / 2)**2
    c = 2 *  math.asin(math.sqrt(a))
    hvs = earthRadius * c

    return hvs

# function that will give us the index of the highest temp above max. will return -1 if none found
def indexHighestTemp(tracks, weekends, max):
    hot = max
    idx = -1

    for i in range(23):
        if tracks[weekends[i] +4][i] > max and tracks[weekends[i] +4][i] > hot:
            idx = i
    
    return idx

# function that will give us the index of the lowest temp below min. will return -1 if none found
def indexLowestTemp(tracks, weekends, min):
    cold = min
    idx = -1

    for i in range(23):
        if tracks[weekends[i] +4][i] < min and tracks[weekends[i] +4][i]  < cold: 
            idx = i

    return idx


# prints out the itinerary that was generated on a weekend by weekend basis starting from the preaseason test
def printItinerary(tracks, weekends, home):

    i = -1
    current = home

    allWeek = [None] * 52

    for week in weekends:
        if len(allWeek) > week:
            allWeek[week - 1] = week

    for week in allWeek:
        if week == None:
            if current == home:
                print("Staying at home thus no travel this weekend")
            else:
                print(f"Travelling home from {tracks[0][current]}")
                current = home
        else:
            if current == home:
                i+=1
                current = i
                print(f"Travelling from home to {tracks[0][current]}. Race temperature is expected to be {tracks[int(week + 4)][current]} degrees")
            else:
                i+=1
                current = i
                print(f"Travelling directly from {tracks[0][current-1]} to {tracks[0][current]}. Race temperature is expected to be {tracks[int(week + 4)][current]} degrees")

# function that will take in the given CSV file and will read in its entire contents
# and return a list of lists
def readCSVFile(file):
    # the rows to return
    rows = []

    # open the file for reading and give it to the CSV reader
    csv_file = open(file)
    csv_reader = csv.reader(csv_file, delimiter=',')

    # read in each row and append it to the list of rows.
    for row in csv_reader:
        rows.append(row)

    # close the file when reading is finished
    csv_file.close()

    # return the rows at the end of the function
    return rows

# function that will read in the race weekends file and will perform all necessary conversions on it
def readRaceWeekends():
    raceWeekends = []

    # open the file for reading and give it to the CSV reader
    csv_file = open('race-weekends.csv')
    csv_reader = csv.reader(csv_file, delimiter=',')

    next(csv_reader)

    for row in csv_reader:
        raceWeekends.append(int(row[1]))

    # close the file when reading is finished
    csv_file.close()

    return raceWeekends

# function that will read the track locations file and will perform all necessary conversions on it.
# this should also strip out the first column on the left which is the header information for all the rows
def readTrackLocations():
    rows = []

    # open the file for reading and give it to the CSV reader
    csv_file = open('track-locations.csv')
    csv_reader = csv.reader(csv_file, delimiter=',')

    # read in each row and append it to the list of rows except first column.
    for row in csv_reader:
        rows.append(row[1:])

    numOfRow = len(rows)
    numOfCol = len(rows[0])

    for i in range(0,numOfRow):
        convertRowToFloat(rows, i)
              
    # close the file when reading is finished
    csv_file.close()

    return rows

# function that will run the simulated annealing case for shortening the distance seperately for both silverstone and monza
class SAcases(Annealer):
    def __init__(self, tracks, weekend, home):
        self.tracks = tracks
        self.home = home
        super(SAcases,self).__init__(weekend)

    def move(self):

        if indexLowestTemp(tracks,self.state,15) != -1:
            a = indexLowestTemp(tracks,self.state,15)
        elif indexHighestTemp(tracks,self.state,35) != -1:
            a = indexHighestTemp(tracks,self.state,35)
        else:
            a = random.randint(1, len(self.state) - 2) 

            while a == 7:
                a = random.randint(1, len(self.state) - 2) 

        b = random.randint(1, len(self.state) - 2)# Bahrain start, Abu Dhabi Close

        while b == 7: # don't move Monaco
            b = random.randint(1, len(self.state) - 2) 

        temp = self.state[a]
        self.state[a] = self.state[b]
        self.state[b] = temp

    def energy(self):

        total = calculateSeasonDistance(tracks, self.state, self.home)

        if checkTemperatureConstraint(tracks, self.state,15,35) == False:
            total += 100000

        return total

# function that will run the genetic algorithms cases for all four situations
# function that will generate a shuffled itinerary. However, this will make sure that the bahrain, abu dhabi, and monaco
# will retain their fixed weeks in the calendar
def generateShuffledItinerary(weekends, numParticle):

    fixed_indices = [0, 7, 23]  # Bahrain, Monaco, Abu Dhabi

    itineraries = []

    for _ in range(numParticle):
        temp_itinerary = copy.deepcopy(weekends)

        shuffle_weeks = [temp_itinerary[i] for i in range(len(temp_itinerary)) if i not in fixed_indices]

        random.shuffle(shuffle_weeks)

        shuffled_itinerary = []
        shuffle_iter = iter(shuffle_weeks)

        for index in range(len(temp_itinerary)):
            if index in fixed_indices:
                shuffled_itinerary.append(temp_itinerary[index])
            else:
                shuffled_itinerary.append(next(shuffle_iter))

        # Store shuffled results
        itineraries.append(shuffled_itinerary)

    return itineraries

def countGreaterEqual(array, value):
    count = 0
    for i in array:
        if i >= value:
            count += 1

    return count

def swapIndexes(particle):
    swapList = []

    for i in range(len(particle)):
        if particle[i] >= 0.5:
            swapList.append(i)

    return swapList

def swapElement(itinerary, particle):
    #toSwap = countGreaterEqual(particle, 0.5)

    #if toSwap == 0:
    #    return

    swapIndex = swapIndexes(particle)
    
    fixed_indices = {0, 7, len(itinerary) - 1} # don't swap Bahrain,Monaco and Abu Dhabi
    swapIndex = [idx for idx in swapIndex if idx not in fixed_indices]

    if indexLowestTemp(tracks,itinerary,15) != -1:
        other = indexLowestTemp(tracks,itinerary,15)
        temp = itinerary[other]
        itinerary[other] = itinerary[swapIndex[0]]
        itinerary[swapIndex[0]] = temp
    elif indexHighestTemp(tracks,itinerary,35) != -1:
        other = indexHighestTemp(tracks,itinerary,35)
        temp = itinerary[other]
        itinerary[other] = itinerary[swapIndex[0]]
        itinerary[swapIndex[0]] = temp
    else:
        other = random.choice([i for i in range(1, len(itinerary) - 1) if i != 7]) # no Bahrain,Monaco and Abu Dhabi
        temp = itinerary[other]
        itinerary[other] = itinerary[swapIndex[0]]
        itinerary[swapIndex[0]] = temp



def PSOcases(particles):
    global bestCost

    cost = []

    for i in range(len(particles)):
        swapElement(itineraries[i], particles[i])
        total = calculateSeasonDistance(tracks,itineraries[i],11)
        if not checkTemperatureConstraint(tracks,itineraries[i],15,35):
            total += 100000
        cost.append(total)

    if total < bestCost:
        bestCost = total

    return cost

# function that will run particle swarm optimisation in an attempt to find a solution
CXPB = 0.5
MUTPB = 0.2

# will go through the genetic code of this child and will make sure that all the required weekends are in it.
# it's highly likely that with crossover that there will be weekends missing and others duplicated. we will
# randomly replace the duplicated ones with the missing ones
def childGeneticCodeFix(child, race):
        
    new_arr = [element for element in race if element not in child.weekend] # those week not in child
    empty = []

    for i in range(len(child.weekend)):
        if child.weekend[i] not in empty:
            empty.append(child.weekend[i])
        else:
            child.weekend[i] = new_arr[0]
            new_arr.pop(0)
                
    return child

class GAcases():

    def __init__(self):
        self.weekend = [9,10,12,14,16,18,20,21,23,25,26,27,29,30,34,35,37,38,42,43,44,47,48,49]

    def randomise(self):
        subarray_to_shuffle = self.weekend[1:7] + self.weekend[8:23]

        # Shuffle the subarray
        random.shuffle(subarray_to_shuffle)

        self.weekend[1:7] = subarray_to_shuffle[:6]
        self.weekend[8:-1] = subarray_to_shuffle[6:]

def initF1Individual(ind_class):
    ind = ind_class()
    ind.randomise()
    return ind

def crossoverF1(ind1, ind2):

    child1 = GAcases()
    child1.weekend.clear()

    fixed_indices = {0, 7, 23} # dont move Bahrain, Monaco, Abu Dhabi

    for i in range(len(ind1.weekend)):
        if i in fixed_indices: 
            child1.weekend.append(ind1.weekend[i])
        else:
            child1.weekend.append(rouletteWheel(ind1.weekend[i], ind2.weekend[i]))

    child1 = childGeneticCodeFix(child1, race)

    return (child1)

# function that performs a roulette wheel randomisation on the two given values and returns the chosen on
def rouletteWheel(a, b):

    if random.random() < 0.5:
        return a
    else: 
        return b

def mutateF1(individual, indpb):

    if indexLowestTemp(tracks,individual.weekend,15) != -1:
        other = indexLowestTemp(tracks,individual.weekend,15)
        individual.weekend[other] = individual.weekend[random.choice([i for i in range(1, len(race) - 1) if i != 7])]
    elif indexHighestTemp(tracks,individual.weekend,35) != -1:
        other = indexHighestTemp(tracks,individual.weekend,35)
        individual.weekend[other] = individual.weekend[random.choice([i for i in range(1, len(race) - 1) if i != 7])]
    else:
        other = random.choice([i for i in range(1, len(individual.weekend) - 1) if i != 7]) # no Bahrain,Monaco and Abu Dhabi
        individual.weekend[other] = individual.weekend[random.choice([i for i in range(1, len(race) - 1) if i != 7])]

def evaluateDistance(individual):

    if len(individual.weekend) != 24:
        return 1000000,

    for i in race:
        if i not in individual.weekend:
            return 1000000,

    total = 0
    total += calculateSeasonDistance(tracks,individual.weekend,11)

    if not checkTemperatureConstraint(tracks,individual.weekend,15,35): #penalty
        total += 100000
        
    return total,

if __name__ == '__main__':
    # uncomment this run all the unit tests. when you have satisfied all the unit tests you will have a working simulation
    # you can then comment this out and move onto your SA and GA solutions
    #unittest.main()
    
    # just to check that the itinerary printing mechanism works. we will assume that silverstone is the home track for this
    race = readRaceWeekends()
    #race = generateShuffledItinerary(weekends)
    #print(race)
    tracks = readTrackLocations()
    #printItinerary(tracks, race, 11)

    # run the cases for simulated annealing
    '''
    SA = SAcases(tracks, race, 11)
    SA.steps = 100000

    state, e =  SA.anneal()

    sorted_indices = sorted(range(len(state)), key=lambda i: state[i])
    sorted_tracks = [[row[i] for i in sorted_indices] for row in tracks]
    sorted_weekend = [state[i] for i in sorted_indices]

    print("\nBest route is :")
    for index, i in enumerate(sorted_indices):
        print(sorted_weekend[index] , " : ", tracks[0][i], " : ", tracks[sorted_weekend[index] + 4][i]) 

    print("total : ", e)
    #printItinerary(sorted_tracks, sorted_weekend, 11)
    '''

    # run the cases for genetic algorithms
    
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", GAcases, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("individual", initF1Individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluateDistance)
    toolbox.register("mate", crossoverF1)
    toolbox.register("mutate", mutateF1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=300)
    fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    generation = 0
    while generation < 1000:
        generation += 1
        print("====Generation %i ====" % generation)

        parents = toolbox.select(pop, len(pop))

        offspring = list(map(toolbox.clone, parents))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                #print(child1.weekend, " : ", child2.weekend)

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                #print(mutant.weekend)

        for individual in offspring:
            if not individual.fitness.valid:
                #print(individual.weekend)
                individual.fitness.values = toolbox.evaluate(individual)

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print(' Min: ', min(fits))
        print(' Max: ', max(fits))
        print(' Avg:', mean)
        print(' Std: ', std)
    
    
    # run the cases for particle swarm optimisation
    '''
    bestCost = 1000000
    options = {'w' : 0.9, 'c1' : 0.5, 'c2' : 0.3}

    numParticles = 100

    constraints24D = (np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                      np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
    
    itineraries = generateShuffledItinerary(race, numParticles)
    
    optimiser = ps.single.GlobalBestPSO(n_particles=numParticles, dimensions=24, options=options, bounds=constraints24D)
    bestCost, bestPosition = optimiser.optimize(PSOcases, iters = 1000)

    print("total distance : ",bestCost)
    '''
    
