import math
import random
from collections import defaultdict
from simanneal import Annealer


# Function to calculate distance between two points
def distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    R = 3963  # Radius of Earth in miles
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    return math.acos(math.sin(lat1) * math.sin(lat2) +
                     math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)) * R


class TravellingSalesmanProblem(Annealer):
    """Simulated Annealing solver for the Travelling Salesman Problem with constraints."""

    def __init__(self, state, distance_matrix, race_weeks):
        self.distance_matrix = distance_matrix
        self.race_weeks = race_weeks  # A list of weeks during which the races occur
        super(TravellingSalesmanProblem, self).__init__(state)

    def move(self):
        """Randomly swaps two cities in the route, except fixed endpoints."""
        initial_energy = self.energy()
        
        # Randomly choose two indices to swap
        a = random.randint(1, len(self.state) - 2)  # Avoid endpoints
        b = random.randint(1, len(self.state) - 2)
        self.state[a], self.state[b] = self.state[b], self.state[a]
        
        return self.energy() - initial_energy

    def energy(self):
        """Calculate the total route distance."""
        e = 0
        for i in range(len(self.state) - 1):
            e += self.distance_matrix[self.state[i]][self.state[i + 1]]
        return e


# Helper function to determine movements (Silverstone to Bahrain, gaps logic)
def create_postman_journey(race_weeks, cities_data):
    """
    Create the postman's journey based on race weeks and constraints.
    Rules:
    - Start from Silverstone to Bahrain on week 9
    - If weeks are continuous, stay at location.
    - If there are skips, return to Silverstone and move back.
    - End at Abu Dhabi and return to Silverstone
    """
    journey = []
    current_location = "Silverstone"
    
    # Map weeks to their corresponding locations
    week_to_location = {
        9: 'Bahrain', 10: 'Saudi Arabia', 11: 'Australia', 12: 'Japan', 13: 'China',
        14: 'Miami', 15: 'Emilia Romagna', 16: 'Monaco', 17: 'Canada',
        18: 'Spain', 19: 'Austria', 20: 'Britain', 21: 'Hungary',
        22: 'Belgium', 23: 'Netherlands', 24: 'Italy', 25: 'Azerbaijan',
        26: 'Singapore', 27: 'USA', 28: 'Mexico', 29: 'Brazil', 30: 'Las Vegas',
        31: 'Qatar', 32: 'Abu Dhabi'
    }

    for idx, week in enumerate(race_weeks):
        if idx == 0:  # First race - go to Bahrain
            journey.append(('Silverstone', week_to_location[week]))
            current_location = week_to_location[week]
        else:
            if week - race_weeks[idx - 1] == 1:  # Continuous weeks
                journey.append((current_location, week_to_location[week]))
                current_location = week_to_location[week]
            else:  # Skip weeks - return home then to next race location
                journey.append((current_location, 'Silverstone'))
                current_location = 'Silverstone'
                journey.append(('Silverstone', week_to_location[week]))
                current_location = week_to_location[week]

    # After the final race, return home to Silverstone
    journey.append((current_location, 'Silverstone'))
    
    return journey


if __name__ == '__main__':
    # Provided data: GP race locations and their circuits
    cities_data = {
        'GP': [
            'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China', 'Miami', 'Emilia Romagna',
            'Monaco', 'Canada', 'Spain', 'Austria', 'Britain', 'Hungary', 'Belgium', 'Netherlands',
            'Italy', 'Azerbaijan', 'Singapore', 'USA', 'Mexico', 'Brazil', 'Las Vegas', 'Qatar',
            'Abu Dhabi'
        ],
        'latitude': [
            26.0325, 21.631944, -37.849722, 34.843056, 31.338889, 25.958056, 44.341111, 43.734722,
            45.500556, 41.57, 47.219722, 52.078611, 47.582222, 50.437222, 52.388819, 45.620556,
            40.3725, 1.291531, 30.132778, 19.406111, -23.701111, 36.081944, 25.49, 24.467222
        ],
        'longitude': [
            50.510556, 39.104444, 144.968333, 136.540556, 121.219722, -80.238889, 11.713333,
            7.420556, -73.5225, 2.261111, 14.764722, -1.016944, 19.251111, 5.971389, 4.540922,
            9.289444, 49.853333, 103.86385, -97.641111, -99.0925, -46.697222, -115.124722,
            51.454167, 54.603056
        ]
    }

    # Randomly select 24 unique weeks for the races between week 9 and week 49
    available_weeks = list(range(9, 50))  # Weeks from 9 to 49
    race_weeks = random.sample(available_weeks, 24)  # Randomly pick 24 unique weeks
    race_weeks.sort()

    # Create postman's journey based on the constraints provided
    journey = create_postman_journey(race_weeks, cities_data)

    print("Journey Path with Constraints:")
    print(journey)
