import random
import math
import csv

def convertRowToFloat(rows, row_index):
    for i in range(len(rows[row_index])):
        rows[row_index][i] = try_convert(rows[row_index][i]) # convert str to float

def try_convert(value): # try to convert it to float 
        try:
            return float(value)
        except ValueError:
            return value
        
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

def calculateSeasonDistance(tracks, weekends, home):

    i = 0
    current = home
    total = 0
    j = 0
    
    for week in weekends:
        total  += haversine(tracks, current, i)
        current = i

        if j + 1 < len(weekends) and weekends[j + 1] == week + 1: # if double/triper header
            i += 1
            j += 1
            continue
        else:
            total += haversine(tracks, current, home) 
            current = home

        i += 1
        j += 1


    if current != home:
        total += haversine(tracks, current, home)
        current = home

    return total

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

# Simulated Annealing function
def simulated_annealing(rows):
    """
    Perform Simulated Annealing to optimize the race schedule.
    Args:
        rows: Data extracted from readTrackLocations(), including locations and weekly temperatures.
    Returns:
        Optimized solution with minimal total distance and valid temperature constraints.
    """
    # Extract relevant data
    locations = rows[0][1:]  # List of locations
    weekly_temperatures = rows[4:]  # Weekly temperatures for each location from week 1 to week 52
    num_weeks = 24  # Total number of race weeks to schedule
    weekends = [9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 49]  # Defined race weeks

    # Fixed Assignments
    fixed_assignments = {9: "Bahrain", 21: "Abu Dhabi", 49: "Monaco"}

    # Random initial solution
    def create_random_solution():
        """Generate an initial random solution respecting fixed constraints."""
        solution = []
        for week in weekends:
            if week in fixed_assignments:
                solution.append(fixed_assignments[week])
            else:
                # Randomly select a location within valid temperature range
                valid_options = [loc for loc in locations if 15 <= weekly_temperatures[week - 9][locations.index(loc)] <= 35]
                solution.append(random.choice(valid_options))
        return solution

    # Define the cost function
    def objective_function(solution):
        """
        Calculates the distance cost of the current solution based on `calculateSeasonDistance`.
        Args:
            solution: List of locations assigned to each weekend.
        Returns:
            Total distance calculated between consecutive race locations.
        """
        # Map solution into the track data indices
        track_indices = [locations.index(loc) for loc in solution]
        # Calculate total distance
        return calculateSeasonDistance(tracks=weekly_temperatures, weekends=weekends, home=0)

    # Simulated Annealing Loop
    current_solution = create_random_solution()
    current_cost = objective_function(current_solution)

    # Parameters
    initial_temperature = 1000
    cooling_rate = 0.99
    temperature = initial_temperature
    iterations_per_temperature = 100
    best_solution = current_solution
    best_cost = current_cost

    while temperature > 1:
        for _ in range(iterations_per_temperature):
            # Generate neighbor solution by swapping two random assignments
            neighbor_solution = current_solution[:]
            idx1, idx2 = random.sample(range(len(neighbor_solution)), 2)
            neighbor_solution[idx1], neighbor_solution[idx2] = neighbor_solution[idx2], neighbor_solution[idx1]

            # Calculate cost of neighbor
            neighbor_cost = objective_function(neighbor_solution)

            # Accept neighbor if it's better or with some probability if worse
            if neighbor_cost < current_cost or random.random() < math.exp((current_cost - neighbor_cost) / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost

            # Update best solution found
            if neighbor_cost < best_cost:
                best_solution = neighbor_solution
                best_cost = neighbor_cost

        # Cool down temperature
        temperature *= cooling_rate

    return best_solution, best_cost

# Read track location data from CSV
rows = readTrackLocations()

# Trigger the simulated annealing optimization
best_solution, best_cost = simulated_annealing(rows)

# Print the optimized race schedule
print("Optimized Race Schedule:")
for week, location in zip([9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 49], best_solution):
    print(f"Week {week}: {location}")

# Print the calculated total distance cost
print(f"Total Distance Cost: {best_cost}")