import json
# Create data/raw directory
import os
import random
from datetime import datetime, timedelta

import numpy as np

os.makedirs("data/raw", exist_ok=True)


def generate_realistic_trip(trip_id, device_id, scenario="normal"):
    """Generate realistic sensor data for a trip based on scenario."""

    # Trip parameters based on scenario
    scenarios = {
        "normal": {
            "duration": random.randint(15, 45),  # minutes
            "base_speed": 35,
            "speed_variation": 15,
            "harsh_events_prob": 0.02,
            "phone_use_prob": 0.05,
            "time_of_day": random.choice([7, 8, 17, 18, 12, 13]),
        },
        "aggressive": {
            "duration": random.randint(20, 40),
            "base_speed": 45,
            "speed_variation": 25,
            "harsh_events_prob": 0.15,
            "phone_use_prob": 0.20,
            "time_of_day": random.choice([22, 23, 0, 1]),
        },
        "cautious": {
            "duration": random.randint(25, 60),
            "base_speed": 28,
            "speed_variation": 8,
            "harsh_events_prob": 0.005,
            "phone_use_prob": 0.01,
            "time_of_day": random.choice([9, 10, 14, 15]),
        },
        "highway": {
            "duration": random.randint(45, 120),
            "base_speed": 65,
            "speed_variation": 10,
            "harsh_events_prob": 0.01,
            "phone_use_prob": 0.03,
            "time_of_day": random.choice([6, 7, 16, 17]),
        },
        "urban_short": {
            "duration": random.randint(5, 15),
            "base_speed": 20,
            "speed_variation": 15,
            "harsh_events_prob": 0.08,
            "phone_use_prob": 0.15,
            "time_of_day": random.choice([12, 13, 18, 19]),
        },
    }

    params = scenarios.get(scenario, scenarios["normal"])

    # Generate timestamps
    start_time = datetime.now() - timedelta(days=random.randint(1, 30))
    start_time = start_time.replace(hour=params["time_of_day"], minute=random.randint(0, 59))

    # Generate GPS trajectory
    start_lat = 37.7749 + random.uniform(-0.1, 0.1)  # San Francisco area
    start_lon = -122.4194 + random.uniform(-0.1, 0.1)

    # Calculate approximate end position based on duration and speed
    distance_miles = (params["base_speed"] * params["duration"]) / 60
    # Rough approximation: 1 degree ≈ 69 miles
    (distance_miles / 69) * random.uniform(-1, 1)
    (distance_miles / 69) * random.uniform(-1, 1)

    sensor_readings = []
    current_lat = start_lat
    current_lon = start_lon
    current_speed = 0
    phone_active = False
    phone_activation_countdown = 0

    # Generate readings every second
    for second in range(params["duration"] * 60):
        timestamp = start_time + timedelta(seconds=second)

        # Speed dynamics
        if second < 30:  # Acceleration phase
            target_speed = min(params["base_speed"], current_speed + random.uniform(0, 3))
        elif second > (params["duration"] * 60 - 60):  # Deceleration phase
            target_speed = max(0, current_speed - random.uniform(0, 2))
        else:  # Cruising with variations
            target_speed = params["base_speed"] + random.uniform(-params["speed_variation"], params["speed_variation"])
            target_speed = max(0, target_speed)

        # Smooth speed transitions
        current_speed = current_speed * 0.9 + target_speed * 0.1

        # GPS position update
        speed_ms = current_speed * 0.44704  # mph to m/s
        heading = random.uniform(0, 360)
        lat_delta = (speed_ms / 111111) * np.cos(np.radians(heading))
        lon_delta = (speed_ms / (111111 * np.cos(np.radians(current_lat)))) * np.sin(np.radians(heading))

        current_lat += lat_delta
        current_lon += lon_delta

        # Accelerometer data (in m/s²)
        base_accel_x = (current_speed - sensor_readings[-1]["speed_mph"] if sensor_readings else 0) * 0.44704
        accel_x = base_accel_x + random.gauss(0, 0.5)
        accel_y = random.gauss(0, 0.3)  # Lateral acceleration
        accel_z = 9.81 + random.gauss(0, 0.2)  # Gravity + vibration

        # Harsh events
        if random.random() < params["harsh_events_prob"]:
            event_type = random.choice(["brake", "accel", "turn"])
            if event_type == "brake":
                accel_x = -random.uniform(3, 5)  # -0.3 to -0.5g
                current_speed *= 0.85
            elif event_type == "accel":
                accel_x = random.uniform(3, 4.5)  # 0.3 to 0.45g
            else:  # turn
                accel_y = random.choice([-1, 1]) * random.uniform(3, 4)

        # Phone usage
        if phone_activation_countdown > 0:
            phone_activation_countdown -= 1
            phone_active = True
        elif phone_active and random.random() < 0.1:  # 10% chance to stop using phone
            phone_active = False
        elif not phone_active and random.random() < params["phone_use_prob"]:
            phone_active = True
            phone_activation_countdown = random.randint(5, 30)  # Use phone for 5-30 seconds

        # GPS accuracy (better on highways, worse in urban areas)
        gps_accuracy = random.uniform(5, 15) if scenario == "highway" else random.uniform(10, 30)

        reading = {
            "timestamp": timestamp.isoformat(),
            "trip_id": trip_id,
            "device_id": device_id,
            "gps_latitude": round(current_lat, 6),
            "gps_longitude": round(current_lon, 6),
            "gps_accuracy_meters": round(gps_accuracy, 1),
            "speed_mph": round(max(0, current_speed), 1),
            "accelerometer_x": round(accel_x, 2),
            "accelerometer_y": round(accel_y, 2),
            "accelerometer_z": round(accel_z, 2),
            "gyroscope_x": round(random.gauss(0, 0.1), 3),
            "gyroscope_y": round(random.gauss(0, 0.1), 3),
            "gyroscope_z": round(random.gauss(0, 0.1), 3),
            "screen_on": 1 if phone_active else 0,
            "app_foreground": (
                random.choice(["com.android.systemui", "com.whatsapp", "com.spotify", "com.google.maps"])
                if phone_active
                else "com.android.systemui"
            ),
            "battery_level": random.randint(20, 95),
            "network_type": random.choice(["4G", "5G", "WiFi"]),
        }

        sensor_readings.append(reading)

    return {
        "trip_metadata": {
            "trip_id": trip_id,
            "device_id": device_id,
            "start_time": start_time.isoformat(),
            "end_time": (start_time + timedelta(minutes=params["duration"])).isoformat(),
            "scenario": scenario,
        },
        "sensor_readings": sensor_readings,
    }


# Generate multiple trips with different scenarios
trips_data = []

# Trip 1: Normal commute
trip1 = generate_realistic_trip("trip_001", "device_A1B2C3", "normal")
with open("data/raw/trip_001_normal_commute.json", "w") as f:
    json.dump(trip1, f, indent=2)
print(f"Generated trip_001: Normal commute with {len(trip1['sensor_readings'])} readings")

# Trip 2: Aggressive night driver
trip2 = generate_realistic_trip("trip_002", "device_D4E5F6", "aggressive")
with open("data/raw/trip_002_aggressive_night.json", "w") as f:
    json.dump(trip2, f, indent=2)
print(f"Generated trip_002: Aggressive night driving with {len(trip2['sensor_readings'])} readings")

# Trip 3: Cautious elderly driver
trip3 = generate_realistic_trip("trip_003", "device_G7H8I9", "cautious")
with open("data/raw/trip_003_cautious_driver.json", "w") as f:
    json.dump(trip3, f, indent=2)
print(f"Generated trip_003: Cautious driver with {len(trip3['sensor_readings'])} readings")

# Trip 4: Highway commute
trip4 = generate_realistic_trip("trip_004", "device_A1B2C3", "highway")
with open("data/raw/trip_004_highway_commute.json", "w") as f:
    json.dump(trip4, f, indent=2)
print(f"Generated trip_004: Highway commute with {len(trip4['sensor_readings'])} readings")

# Trip 5: Urban short trip with frequent stops
trip5 = generate_realistic_trip("trip_005", "device_J1K2L3", "urban_short")
with open("data/raw/trip_005_urban_short.json", "w") as f:
    json.dump(trip5, f, indent=2)
print(f"Generated trip_005: Urban short trip with {len(trip5['sensor_readings'])} readings")

# Generate a batch file with multiple short trips from same driver
batch_trips = []
for i in range(3):
    trip = generate_realistic_trip(f"trip_00{6+i}", "device_M4N5O6", random.choice(["normal", "urban_short"]))
    batch_trips.append(trip)

with open("data/raw/batch_trips_same_driver.json", "w") as f:
    json.dump(batch_trips, f, indent=2)
print(f"Generated batch file with {len(batch_trips)} trips from same driver")

# Create a sample of raw readings (first 10 seconds) for documentation
sample_readings = trip1["sensor_readings"][:10]
with open("data/raw/sample_sensor_readings.json", "w") as f:
    json.dump(
        {
            "description": "Sample of first 10 seconds of sensor readings",
            "trip_id": "trip_001",
            "readings": sample_readings,
        },
        f,
        indent=2,
    )

print("\nAll JSON files created in data/raw/ directory")
print("Files are ready to be processed by the pipeline using:")
print("python main.py --input-dir data/raw --output-dir data/processed --aggregate-drivers")
