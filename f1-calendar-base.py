import unittest
from math import radians, sin, cos, sqrt, asin
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
    def testReadCSV(self): #done
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
    def testRowToFloat(self): # done
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
    def testReadTrackLocations(self): #done
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
    def testReadRaceWeekends(self): #done
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
    totalDistance = 0.0
    currentLocation = home # start point
    index = 0

    fixedRaces = {9: "Bahrain", 21: "Monaco", 49: "Abu Dhabi"}

    for week in range(1,53): # F1 races weeks
        if week in weekends:
            if week in fixedRaces:
                raceTrack = fixedRaces[week] # Fixed locations
            else:
                raceIndex = weekends.index(week)
                raceTrack = tracks[week]

                for i in range(1,23):
                        if tracks['GP'][i] == raceTrack: # find the index of the location then we can get then number to apply on track_data
                            index = i

                totalDistance += haversine(tracks, int(currentLocation), int(index)) 
                currentLocation = index # now we at race location

            # if double/triper header
            if week < 52 and weekends[raceIndex + 1] == week + 1:
                currentLocation = index # remain location
            else:
                totalDistance = haversine(tracks, int(currentLocation), int(home))
                currentLocation = home # back home

    if currentLocation != home:
        totalDistance += haversine(tracks, currentLocation, home) # team should back home at off season

    return totalDistance


# function that will check to see if there is anywhere in our weekends where four races appear in a row. True indicates that we have four in a row
def checkFourRaceInRow(weekends):
    pass

# function that will check to see if the temperature constraint for all races is satisfied. The temperature
# constraint is that a minimum temperature of min degrees for the month is required for a race to run
def checkTemperatureConstraint(tracks, weekends, min, max):
    pass

# function that will check to see if there is a four week gap anywhere in july and august. we will need this for the summer shutdown.
# the way this is defined is that we have a gap of three weekends between successive races. this will be weeks 31, 32, and 33, they are not
# permitted to have a race during these weekends
def checkSummerShutdown(weekends):
    summer = range(27,36) # start of July until end of August

    for startWeek in range(27,33): # need four week to summer shutdown
        if all(week not in weekends for week in range(startWeek, startWeek + 3)): # four week gap
            return True

    return False

# will go through the genetic code of this child and will make sure that all the required weekends are in it.
# it's highly likely that with crossover that there will be weekends missing and others duplicated. we will
# randomly replace the duplicated ones with the missing ones
def childGeneticCodeFix(child):
    pass

# function that will take in the set of rows and will convert the given row index into floating point values
# this assumes the header in the CSV file is still present so it will skip the first column
def convertRowToFloat(rows, row_index):
    for i in range(len(rows[row_index])):
        rows[row_index][i] = try_convert(rows[row_index][i]) # convert str to float

# function that will generate a shuffled itinerary. However, this will make sure that the bahrain, abu dhabi, and monaco
# will retain their fixed weeks in the calendar
def generateShuffledItinerary(weekends):
    pass

# function that will use the haversine formula to calculate the distance in Km given two latitude/longitude pairs
# it will take in an index to two rows, and extract the latitude and longitude before the calculation.
def haversine(rows, location1, location2):
    earthRadius = 6371.0 # Earth Radius

    # Calculate latitude and longtitude
    lat1, long1 = radians(rows[2][location1]), radians(rows[3][location1])
    lat2, long2 = radians(rows[2][location2]), radians(rows[3][location2])

    part1 =  sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((long2 - long1) / 2) ** 2
    hvs = 2 * asin(sqrt(part1)) * earthRadius

    return hvs

# function that will give us the index of the lowest temp below min. will return -1 if none found
def indexHighestTemp(tracks, weekends, max):
    pass

# function that will give us the index of the lowest temp below min. will return -1 if none found
def indexLowestTemp(tracks, weekends, min):
    pass

# prints out the itinerary that was generated on a weekend by weekend basis starting from the preaseason test
def printItinerary(tracks, weekends, home):
    pass

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
         for j in range(0,numOfCol):
              value = try_convert(rows[i][j]) # convert str to float
              rows[i][j] = value
              

    # close the file when reading is finished
    csv_file.close()

    return rows

def try_convert(value): # try to convert it to float 
        try:
            return float(value)
        except ValueError:
            return value

# function that performs a roulette wheel randomisation on the two given values and returns the chosen on
def rouletteWheel(a, b):
    pass

# function that will run the simulated annealing case for shortening the distance seperately for both silverstone and monza
def SAcases():
    pass

# function that will run the genetic algorithms cases for all four situations
def GAcases():
    pass

# function that will run particle swarm optimisation in an attempt to find a solution
def PSOcases():
    pass

if __name__ == '__main__':
    # uncomment this run all the unit tests. when you have satisfied all the unit tests you will have a working simulation
    # you can then comment this out and move onto your SA and GA solutions
    unittest.main()

    # just to check that the itinerary printing mechanism works. we will assume that silverstone is the home track for this
    #weekends = readRaceWeekends()
    #print(generateShuffledItinerary(weekends))
    #tracks = readTrackLocations()
    #printItinerary(tracks, weekends, 11)

    # run the cases for simulated annealing
    #SAcases()

    # run the cases for genetic algorithms
    #GAcases()

    # run the cases for particle swarm optimisation
    #PSOcases()