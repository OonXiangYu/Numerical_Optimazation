import unittest
from math import radians, sin, cos, sqrt, asin
import csv

class TestSummerShutdown(unittest.TestCase):
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


def haversine(rows, location1, location2):
    earthRadius = 6371.0 # Earth Radius

    # Calculate latitude and longtitude
    lat1 = rows[2][location1]
    long1 = rows[3][location1]
    lat2 = rows[2][location2] 
    long2 = rows[3][location2]

    print(lat1, long1, lat2, long2)

    part1 =  sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((long2 - long1) / 2) ** 2
    hvs = 2 * asin(sqrt(part1)) * earthRadius

    return hvs

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

def readTrackLocations():
    rows = []

    # open the file for reading and give it to the CSV reader
    csv_file = open('track-locations.csv')
    csv_reader = csv.reader(csv_file, delimiter=',')

    # read in each row and append it to the list of rows except first column.
    for row in csv_reader:
        rows.append(row[1:])

    for i in range(len(rows)):
         for j in range(len(rows[0])):
              rows[i][j] = try_convert(rows[i][j]) # convert str to float

    # close the file when reading is finished
    csv_file.close()

    return rows

def try_convert(value): # try to convert it to float 
        try:
            return float(value)
        except ValueError:
            return value

if __name__ == "__main__":
    unittest.main()