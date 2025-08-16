"""
Airbrakes Lookup Table Generator

Usage: python main.py
Output: lookup_table.csv
"""

""" MAIN TODO s
- Improve performance
- Work on refining input parameters
"""

import numpy as np
import csv
from datetime import datetime
import time
import threading
from queue import Queue, Empty
import os
import warnings
warnings.filterwarnings('ignore')
from rocketpy import Rocket, EmptyMotor, Environment, Flight


TARGET_APOGEE_M = 3048              # 10,000 ft TODO update

# ==================== HYPERION CONFIGURATION ====================

# Rocket parameters
ROCKET_DRY_MASS_KG = 20.5           # kg, without motor installed # TODO update with final mass once assembled
ROCKET_DIAMETER_M = 0.143           # 14.3cm diameter # TODO verify
ROCKET_RADIUS = ROCKET_DIAMETER_M / 2
ROCKET_REFERENCE_AREA = np.pi * ROCKET_RADIUS**2  # m²

def hyperion_drag_coefficient(mach): # TODO update with back-computed drag curve from last year's flight data
    """Hyperion Cd function from RASAero II"""
    # Simplified version - full implementation would include all points
    if mach <= 0.5:
        return 0.40
    elif mach <= 0.8:
        return 0.38
    elif mach <= 0.95:
        return 0.40 + (mach - 0.8) * (0.45 - 0.40) / (0.95 - 0.8)
    elif mach <= 1.05:
        return 0.45 + (mach - 0.95) * (0.60 - 0.45) / (1.05 - 0.95)
    else:
        return 0.60

# Cesaroni M2505 motor parameters NOTE will keep the values about before burnout here even though not needed for lookup table generation. Good to have everything in one place, might use to better bound the lookup table domain
MOTOR_DRY_MASS_KG = 2.310           # kg # TODO mass our casing, update
MOTOR_PROPELLANT_MASS_KG = 3.423    # kg # TODO mass our casing after fuel inserted, subtract empty casing mass, update
MOTOR_BURN_TIME = 3.0               # seconds
MOTOR_THRUST_CURVE = [              # Time (s), Thrust (N)
    [0, 0], [0.12, 2600], [0.21, 2482], [0.6, 2715],
    [0.9, 2876], [1.2, 2938], [1.5, 2889], [1.8, 2785],
    [2.1, 2573], [2.4, 2349], [2.7, 2182], [2.99, 85], [3, 0]
    ] # from https://www.thrustcurve.org/motors/Cesaroni/7450M2505-P/

# Airbrakes parameters
NUM_FLAPS = 3
PER_FLAP_AREA_M2 = 0.004215        # m² per flap TODO measure new flaps, update
TOTAL_FLAP_AREA_M2 = NUM_FLAPS * PER_FLAP_AREA_M2  # m² total
FLAP_CD = 0.95                     # Drag coefficient of flaps TODO update
MAX_DEPLOYMENT_ANGLE = 45          # degrees TODO update
DEPLOYMENT_RATE = 5.5              # deg/s under load TODO update after deployment rate under load testing
RETRACTION_RATE = 10.0             # deg/s unloaded TODO update after unloaded retraction rate testing

# Launch conditions # TODO add more environmental properties: temp, pressure, wind
LAUNCH_ALTITUDE_MSL = 364           # m, from https://earth.google.com/web/search/Launch+Canada+Launch+Pad/@47.9869503,-81.8485488,363.96383335a,679.10907018d,35y
LAUNCH_LATITUDE = 47.9870           # from https://maps.app.goo.gl/n76cD331j7LiQiTB6
LAUNCH_LONGITUDE = -81.8486         # from https://maps.app.goo.gl/n76cD331j7LiQiTB6
LAUNCH_RAIL_LENGTH = 5.18           # m TODO verify
LAUNCH_RAIL_ANGLE = 86.8            # degrees TODO update

# Simulation parameters
TOLERANCE_BINARY_SEARCH = 0.5       # degrees TODO review

# Lookup table parameters
TESTING_MODE = True  # Set to False for full resolution 

if TESTING_MODE:
    # 10x10 grid for testing
    HEIGHT_POINTS = 10
    VELOCITY_POINTS = 10
else:
    # Full resolution TODO test speed of flight computer in accessing different size lookup tables, update this
    HEIGHT_POINTS = 100
    VELOCITY_POINTS = 100

# Burnout state ranges TODO update based on sensitivity analysis
BURNOUT_HEIGHT_MIN, BURNOUT_HEIGHT_MAX = 240, 560      # m
BURNOUT_VELOCITY_MIN, BURNOUT_VELOCITY_MAX = 140, 340  # m/s

# ========================= SIMULATOR =========================
SIN_THETA_MAX = np.sin(np.deg2rad(MAX_DEPLOYMENT_ANGLE))
CLOSING_MARGIN = 2 # TODO review
CONTROLLER_SAMPLING_RATE = 4 # TODO confirm this is high enough
TOTAL_DRY_MASS = ROCKET_DRY_MASS_KG + MOTOR_DRY_MASS_KG

def airbrakes_sim(environment, rocket, initial_solution, angle_this_run):
    """
    For a flight where the airbrakes begin to extend at burnout up to a given angle, and begin to retract in time to be closed for apogee plus some margin, this function returns the apogee (AGL, in m) and time at which the airbrakes begin to retract (in s after burnout)
    """
    retraction_time = None
    def controller_function(
        time, sampling_rate, state, state_history, observed_variables, air_brakes
    ):
        # state = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]
        nonlocal retraction_time
        vz = state[5]
        previous_vz = observed_variables[-1][0]

        az = (vz - previous_vz) * CONTROLLER_SAMPLING_RATE
        if vz == previous_vz:
            az=-0.000001

        deployment_angle = np.rad2deg(np.arcsin(air_brakes.deployment_level/SIN_THETA_MAX))
        
        t_to_close = deployment_angle / RETRACTION_RATE
        
        if t_to_close >= - vz / az - CLOSING_MARGIN or retraction_time: # retract if you need to retract to be closed at apogee, or have already started retracting
            new_deployment_angle = max(0, deployment_angle - RETRACTION_RATE / sampling_rate) # TODO update from this worst case one which assumes drag doesn't decrease if it makes much difference/is worth it
            if not retraction_time:
                retraction_time = time
        else: # extend to/maintain the desired angle if not retracting yet
            new_deployment_angle = min(deployment_angle + DEPLOYMENT_RATE * time, angle_this_run)

        new_deployment_level = np.sin(np.deg2rad(new_deployment_angle)) * SIN_THETA_MAX

        air_brakes.deployment_level = new_deployment_level
        return vz, deployment_angle, new_deployment_level

    def drag_coefficient_curve(deployment_level, free_stream_mach):
        return FLAP_CD * deployment_level

    airbrakes = rocket.add_air_brakes(
        drag_coefficient_curve=drag_coefficient_curve,
        controller_function=controller_function,
        sampling_rate=CONTROLLER_SAMPLING_RATE,
        clamp = True,
        reference_area=TOTAL_FLAP_AREA_M2*SIN_THETA_MAX,
        initial_observed_variables=(initial_solution[6], 0, 0, False)
    )

    flight_airbrakes = Flight(
        rocket=rocket,
        environment=environment,
        rail_length=LAUNCH_RAIL_LENGTH,
        initial_solution=initial_solution,
        time_overshoot=False,
        terminate_on_apogee=True
    )
    return (flight_airbrakes.apogee - LAUNCH_ALTITUDE_MSL, retraction_time)

# ================== Find Optimal Deployment ==================

def find_optimal_deployment(h_burnout, vz_burnout):
    """
    Returns the optimal angle and the time after burnout at which to retract for a given burnout state.
    """
    # TODO have all parameters that will be changed at top of file
    # TODO defining these once outside the function instead of for every burnout state could give a speedup
    environment = Environment(
        latitude=LAUNCH_LATITUDE,
        longitude=LAUNCH_LONGITUDE,
        elevation=LAUNCH_ALTITUDE_MSL
    )
    motor = EmptyMotor()
    rocket = Rocket(
        radius=ROCKET_RADIUS,
        mass=TOTAL_DRY_MASS,
        inertia=(
            4.87,
            4.87,
            0.05,
        ), # TODO update
        power_off_drag = hyperion_drag_coefficient,
        power_on_drag = hyperion_drag_coefficient,
        center_of_mass_without_motor=1.3, # TODO update
        coordinate_system_orientation="tail_to_nose"
    )
    rocket.set_rail_buttons(0.69, 0.21, 60) # TODO update
    rocket.add_nose(length=0.731, kind='von karman', position=2.073) # TODO update
    rocket.add_motor(motor, position=0)
    rocket.add_trapezoidal_fins( # TODO update
        3,
        span=0.135,
        root_chord=0.331,
        tip_chord=0.1395,
        position=0.314,
        sweep_length=0.0698
    )
    # TODO confirm assumptions about burnout conditions
    vx_burnout = 0.08*vz_burnout
    vy_burnout = 0.08*vz_burnout
    burnout_orientation = [0.965,-0.007,0.043,-0.257]
    initial_solution=[
        0,
        0, 0, h_burnout,
        vx_burnout, vy_burnout, vz_burnout,
        burnout_orientation[0], burnout_orientation[1], burnout_orientation[2], burnout_orientation[3],
        0,0,0 # angular velocity
    ]

    # First, check if no airbrake deployment needed
    flight_no_brakes = Flight(
        rocket=rocket,
        environment=environment,
        rail_length=LAUNCH_RAIL_LENGTH,
        terminate_on_apogee=True,
        initial_solution=initial_solution
    )
    if flight_no_brakes.apogee - LAUNCH_ALTITUDE_MSL < TARGET_APOGEE_M:
        print(f'No airbrakes needed for h={h_burnout:.0f}m, vz={vz_burnout:.0f}m/s')
        return 0, None

    # If airbrakes deployment needed, check if max deployment isn't overkill
    flight_max_brakes = airbrakes_sim(environment, rocket, initial_solution, MAX_DEPLOYMENT_ANGLE)
    print(f"Apogee for max airbrakes deployment: {flight_max_brakes[0]:.0f}m")

    if flight_max_brakes[0] > TARGET_APOGEE_M:
        print(f'Max airbrakes deployment needed for h={h_burnout:.0f}m, vz={vz_burnout:.0f}m/s')
        return MAX_DEPLOYMENT_ANGLE, flight_max_brakes[1]

    # If an intermediate amount of stopping power is needed, run binary search
    print("NEED AN INTERMEDIATE AMOUNT OF STOPPING POWERRRR")
    lower_bound = 0
    upper_bound = MAX_DEPLOYMENT_ANGLE
    while upper_bound - lower_bound > TOLERANCE_BINARY_SEARCH:
        deployment_angle = (upper_bound + lower_bound) / 2
        print(f"Simulating for deployment angle {deployment_angle:.1f}°")
        apogee, retract_time = airbrakes_sim(environment, rocket, initial_solution, deployment_angle)
        if apogee > TARGET_APOGEE_M:
            lower_bound = deployment_angle
        else:
            upper_bound = deployment_angle
        print(f"Apogee: {apogee:.0f}m AGL")
    return (upper_bound + lower_bound) / 2, retract_time


# ==================== LOOKUP TABLE GENERATION ====================

def generate_lookup_table():
    """Generate lookup table"""
    
    print("\n=== GENERATING LOOKUP TABLE ===")
        
    # Create burnout states grid
    heights = np.linspace(BURNOUT_HEIGHT_MIN, BURNOUT_HEIGHT_MAX, HEIGHT_POINTS)
    velocities = np.linspace(BURNOUT_VELOCITY_MIN, BURNOUT_VELOCITY_MAX, VELOCITY_POINTS)
    
    print(f"Height increment: {heights[1]-heights[0]:.1f}m")
    print(f"Velocity increment: {velocities[1]-velocities[0]:.1f}m/s")
    
    # Generate lookup table
    lookup_table = []
    total_points = len(heights) * len(velocities)
    point_count = 0
    
    start_time = time.time()
    
    for height in heights:
        for velocity in velocities:
            point_count += 1
            
            # Progress indicator
            if point_count % 10 == 0 or point_count == 1:
                elapsed = time.time() - start_time
                rate = point_count / elapsed if elapsed > 0 else 0
                eta = (total_points - point_count) / rate if rate > 0 else 0
                print(f"  Point {point_count}/{total_points} " +
                      f"({100*point_count/total_points:.1f}%) " +
                      f"ETA: {eta/60:.1f}min")
            
            # Find optimal deployment
            deployment_angle, retraction_time = find_optimal_deployment(
                height + LAUNCH_ALTITUDE_MSL, velocity
            )
            if retraction_time is None: retraction_time = 0

            lookup_table.append({
                'burnout_height_m': round(height, 2),
                'burnout_velocity_ms': round(velocity, 2),
                'deployment_angle_deg': round(deployment_angle, 4),
                'retraction_time_s': round(retraction_time, 2)
            })
    
    elapsed = time.time() - start_time
    print(f"\n✓ Generated {len(lookup_table)} entries in {elapsed:.1f}s")
    
    return lookup_table

def save_lookup_table(lookup_table, filename='lookup_table.csv'):
    """Save lookup table to csv"""
    
    print(f"\nSaving lookup table to {filename}...")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header comments
        writer.writerow(['# Airbrakes Lookup Table - Hyperion with Cesaroni M2505'])
        writer.writerow([f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([f'# Target Apogee: {TARGET_APOGEE_M}m'])
        writer.writerow([f'# Grid: {HEIGHT_POINTS}x{VELOCITY_POINTS}'])
        writer.writerow(['#'])
        
        # Column headers
        writer.writerow([
            'burnout_height_m',
            'burnout_velocity_ms', 
            'deployment_angle_deg',
            'retraction_time_s'
        ])
        
        # Data rows
        for row in lookup_table:
            writer.writerow([
                row['burnout_height_m'],
                row['burnout_velocity_ms'],
                row['deployment_angle_deg'],
                row['retraction_time_s']
            ])
    
    # Print statistics
    angles = [row['deployment_angle_deg'] for row in lookup_table]
    print(f"✓ Saved {len(lookup_table)} entries")
    print(f"  Deployment angles: {min(angles):.1f}° to {max(angles):.1f}°")
    print(f"  Zero deployments: {sum(1 for a in angles if a == 0)}")
    print(f"  Max deployments: {sum(1 for a in angles if a >= MAX_DEPLOYMENT_ANGLE-1)}")
    print(f"  Partial deployments: {sum(1 for a in angles if 0 < a < MAX_DEPLOYMENT_ANGLE-1)}")

# ==================== MAIN ====================

def main():
    """Main function"""

    print(f"\n=== CONFIGURATION ===")
    print(f"Mode: {'TESTING (10x10)' if TESTING_MODE else f'FULL ({HEIGHT_POINTS}x{VELOCITY_POINTS})'}")
    print(f"Target Apogee: {TARGET_APOGEE_M}m")
    print(f"Grid Size: {HEIGHT_POINTS}×{VELOCITY_POINTS} = {HEIGHT_POINTS*VELOCITY_POINTS} burnout states")
    print(f"Height Range: {BURNOUT_HEIGHT_MIN}-{BURNOUT_HEIGHT_MAX}m")
    print(f"Velocity Range: {BURNOUT_VELOCITY_MIN}-{BURNOUT_VELOCITY_MAX}m/s")

    print("=" * 60)
    print("AIRBRAKES LOOKUP TABLE GENERATOR")
    print("=" * 60)
    
    # Burnout States
    print("\n=== BURNOUT STATE RANGES ===")
    print(f"Height: {BURNOUT_HEIGHT_MIN}-{BURNOUT_HEIGHT_MAX}m")
    print(f"Velocity: {BURNOUT_VELOCITY_MIN}-{BURNOUT_VELOCITY_MAX}m/s")
    print(f"Grid: {HEIGHT_POINTS}×{VELOCITY_POINTS}")
    
    # Lookup Table Generation
    print("\n=== LOOKUP TABLE GENERATION ===")
    
    try:
        # Generate lookup table
        lookup_table = generate_lookup_table()
        
        # Save results
        save_lookup_table(lookup_table)
        
        print("\n" + "=" * 60)
        print("✓ LOOKUP TABLE GENERATION COMPLETE")
        print("=" * 60)
        
        if TESTING_MODE:
            print("\nNote: Generated testing table (10×10)")
            print("Set TESTING_MODE = False for full resolution (1000×1000)")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
    # TODO make it optionally plot the angles over the lookup table as a colourmap

if __name__ == "__main__":
    # main()
    print(find_optimal_deployment(1850, 200))