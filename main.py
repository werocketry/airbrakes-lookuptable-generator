"""
Airbrakes Lookup Table Generator

Usage: python main.py
Output: lookup_table.csv
"""

# MAIN TODO: Work on refining input parameters

import numpy as np
from scipy import integrate
import csv
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')
from rocketpy import Rocket, EmptyMotor, Environment, Flight
import copy
import multiprocessing
from multiprocessing import Pool


TARGET_APOGEE_M = 8500 * 0.3048     # TODO confirm final

# ====================== ENVIRONMENT SETUP ====================
LAUNCH_ALTITUDE_MSL = 364           # m, from https://earth.google.com/web/search/Launch+Canada+Launch+Pad/@47.9869503,-81.8485488,363.96383335a,679.10907018d,35y
LAUNCH_LATITUDE = 47.9870           # from https://maps.app.goo.gl/n76cD331j7LiQiTB6
LAUNCH_LONGITUDE = -81.8486         # from https://maps.app.goo.gl/n76cD331j7LiQiTB6
LAUNCH_RAIL_LENGTH = 5.64           # m
LAUNCH_RAIL_ANGLE = 84              # degrees

LAUNCHPAD_TEMP = 18                 # deg C TODO update
LAUNCHPAD_PRESSURE = 102800         # Pa TODO update https://www.timeanddate.com/weather/@5914408/historic
WIND_EAST = 2                       # m/s TODO update
WIND_NORTH = 2                      # m/s TODO update

TEMP_LAPSE_RATE = 6.5e-3 # deg C/m
TEMP_SEA_LEVEL = LAUNCHPAD_TEMP + TEMP_LAPSE_RATE*LAUNCH_ALTITUDE_MSL + 273.15
def temp_at_h_ASL(h):
    return TEMP_SEA_LEVEL - TEMP_LAPSE_RATE*h

def get_local_gravity(latitude, h = 0):
    """
    Calculate the acceleration due to gravity at a given latitude and altitude above sea level.
    Args
    ----
    latitude : float
        Latitude of launch site in degrees.
    h : float
        Ground level elevation above sea level in meters. Defaults to 0.

    Returns
    -------
    float
        Acceleration due to gravity in meters per second squared.

    References
    ----------
    Based on the International Gravity Formula 1980 (IGF80) model, as outlined in https://en.wikipedia.org/wiki/Theoretical_gravity#International_gravity_formula_1980
    """

    gamma_a = 9.780327  # m/s^2
    c1 = 0.0052790414
    c2 = 0.0000232718
    c3 = 0.0000001262
    c4 = 0.0000000007

    phi = np.deg2rad(latitude)

    gamma_0 = gamma_a * (1 + c1 * np.sin(phi)**2 + c2 * np.sin(phi)**4 + c3 * np.sin(phi)**6 + c4 * np.sin(phi)**8)


    k1 = 3.15704e-07  # 1/m
    k2 = 2.10269e-09  # 1/m
    k3 = 7.37452e-14  # 1/m^2

    g = gamma_0 * (1 - (k1 - k2 * np.sin(phi)**2) * h + k3 * h**2)

    return g

GRAVITY_MAGNITUDE = get_local_gravity(LAUNCH_LATITUDE, LAUNCH_ALTITUDE_MSL)
R_universal = 8.3144598
MM_air = 0.0289644
R_AIR = R_universal / MM_air

def pressure_at_h_ASL(h):
    h_agl = h - LAUNCH_ALTITUDE_MSL
    return LAUNCHPAD_PRESSURE * pow(1-h_agl*TEMP_LAPSE_RATE/(LAUNCHPAD_TEMP+273.15), GRAVITY_MAGNITUDE/(R_AIR*TEMP_LAPSE_RATE))

# ==================== HYPERION CONFIGURATION ====================

# Cesaroni M2505 motor parameters
MOTOR_DRY_MASS = 2.866              # kg
MOTOR_PROPELLANT_MASS = 3.713       # kg
MOTOR_BURN_TIME = 3.0               # seconds
MOTOR_THRUST_CURVE = [              # Time (s), Thrust (N)
    [0, 0], [0.12, 2600], [0.21, 2482], [0.6, 2715],
    [0.9, 2876], [1.2, 2938], [1.5, 2889], [1.8, 2785],
    [2.1, 2573], [2.4, 2349], [2.7, 2182], [2.99, 85], [3, 0]
    ] # from https://www.thrustcurve.org/motors/Cesaroni/7450M2505-P/
MOTOR_IMPULSE = integrate.trapezoid(
    np.array([point[1] for point in MOTOR_THRUST_CURVE]),
    np.array([point[0] for point in MOTOR_THRUST_CURVE])
) # both thrustcurve.org files list lower impulses than the 7450 N*s from the mfr, and which are much closer to the integral of the thrust curve

# Rocket parameters
ROCKET_DRY_MASS = 20.5           # kg, without motor installed # TODO update with final mass once assembled
TOTAL_DRY_MASS = ROCKET_DRY_MASS + MOTOR_DRY_MASS
ROCKET_DIAMETER = 0.1427            # m
ROCKET_RADIUS = ROCKET_DIAMETER / 2
ROCKET_REFERENCE_AREA = np.pi * ROCKET_RADIUS**2  # m²

def hyperion_drag_coefficient(mach): # Looked at constant back-computed drag curves from last year's flight data (see hyperion1_back_calc_drag.ipynb), lowered a bit to account for better finish on airframe this year
    return 0.41

# Airbrakes parameters
NUM_FLAPS = 3
PER_FLAP_AREA_M2 = 82*36.5*1e-6    # m² per flap
TOTAL_FLAP_AREA_M2 = NUM_FLAPS * PER_FLAP_AREA_M2  # m² total
FLAP_CD = 0.9                      # Drag coefficient of flaps
MAX_DEPLOYMENT_ANGLE = 45          # degrees TODO update
DEPLOYMENT_RATE = 5.5              # deg/s under load TODO update after deployment rate under load testing
RETRACTION_RATE = 10.0             # deg/s unloaded TODO update after unloaded retraction rate testing
CLOSING_MARGIN = 2 # s

# Simulation parameters
TOLERANCE_BINARY_SEARCH = 0.5       # degrees

# Use the same portion of horizontal/vertical velocities at burnout as ork projects, assume same velocity in both horizontal directions
BURNOUT_V_HORIZONTAL_PROPORTION_OF_V = 0.05
BURNOUT_VX_PROPORTION_OF_V = BURNOUT_V_HORIZONTAL_PROPORTION_OF_V / np.sqrt(2)
BURNOUT_VY_PROPORTION_OF_V = BURNOUT_VX_PROPORTION_OF_V
BURNOUT_V_Z_PROPORTION_OF_V = np.sqrt(1 - BURNOUT_V_HORIZONTAL_PROPORTION_OF_V**2)

BURNOUT_ORIENTATION = [0.965,-0.007,0.043,-0.257] # TODO confirm assumptions about this burnout condition
BURNOUT_ANGULAR_VELOCITY = [0, 0, 0] # TODO confirm assumptions about this burnout condition

# Lookup table parameters
# TODO test speed of flight computer in accessing different size lookup tables, update this
HEIGHT_POINTS = 10
VELOCITY_POINTS = 10

# Burnout state ranges
# NOTE THEORETICAL_MAX_VELOCITY and THEORETICAL_MAX_BURNOUT_HEIGHT are actually significantly higher than the theoretical maxima. They assumes no drag, a perfectly vertical launch, and no weathercocking on leaving the rail(/no wind)
THEORETICAL_MAX_VELOCITY = (MOTOR_IMPULSE / TOTAL_DRY_MASS - GRAVITY_MAGNITUDE * MOTOR_BURN_TIME) * 1.05
thrust_times = np.array([pt[0] for pt in MOTOR_THRUST_CURVE])
accels = np.array([pt[1] for pt in MOTOR_THRUST_CURVE]) / TOTAL_DRY_MASS
THEORETICAL_MAX_BURNOUT_HEIGHT = integrate.trapezoid((MOTOR_BURN_TIME - thrust_times) * accels, thrust_times) * 1.05
# TODO update minima based on sensitivity analysis

BURNOUT_HEIGHT_MIN, BURNOUT_HEIGHT_MAX = 240, THEORETICAL_MAX_BURNOUT_HEIGHT      # m
BURNOUT_VELOCITY_MIN, BURNOUT_VELOCITY_MAX = 200, THEORETICAL_MAX_VELOCITY        # m/s

# ========================= SIMULATOR =========================
SIN_THETA_MAX = np.sin(np.deg2rad(MAX_DEPLOYMENT_ANGLE))
CONTROLLER_SAMPLING_RATE = 8

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
            new_deployment_angle = max(0, deployment_angle - RETRACTION_RATE / sampling_rate) # worst case, assumes drag doesn't decrease
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

def build_simulation_base():
    """Construct shared Environment and a base Rocket instance.

    Returns (environment, rocket). The returned rocket should be treated as
    a template and not modified directly by simulations (deepcopy before use).
    """
    environment = Environment(
        latitude=LAUNCH_LATITUDE,
        longitude=LAUNCH_LONGITUDE,
        elevation=LAUNCH_ALTITUDE_MSL,
        max_expected_height=5000
    )
    environment.set_atmospheric_model(
        'custom_atmosphere',
        temperature = temp_at_h_ASL,
        pressure = pressure_at_h_ASL,
        wind_u = WIND_EAST,
        wind_v = WIND_NORTH
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
    rocket.set_rail_buttons(0.69, 0.21, 60)
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
    return environment, rocket

environment, base_rocket = build_simulation_base()

def find_optimal_deployment(h_burnout, vz_burnout):
    """Returns the optimal angle and the time after burnout at which to retract for a given burnout state.

    This function expects shared simulation objects to be created once and reused.
    """
    # If shared objects not created, create them now
    try:
        environment
        base_rocket
    except NameError:
        environment, base_rocket = build_simulation_base()
    # Create fresh rocket instance for this simulation by deep-copying base_rocket. This avoids re-running the heavier construction logic and prevents accumulating airbrakes or other components on the same rocket object.
    rocket = copy.deepcopy(base_rocket)
    vx_burnout = vz_burnout / BURNOUT_V_Z_PROPORTION_OF_V * BURNOUT_VX_PROPORTION_OF_V
    vy_burnout = vz_burnout / BURNOUT_V_Z_PROPORTION_OF_V * BURNOUT_VY_PROPORTION_OF_V
    initial_solution=[
        0,
        0, 0, h_burnout,
        vx_burnout, vy_burnout, vz_burnout,
        BURNOUT_ORIENTATION[0], BURNOUT_ORIENTATION[1], BURNOUT_ORIENTATION[2], BURNOUT_ORIENTATION[3],
        BURNOUT_ANGULAR_VELOCITY[0], BURNOUT_ANGULAR_VELOCITY[1], BURNOUT_ANGULAR_VELOCITY[2]
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
        print(f'No airbrakes needed for h={h_burnout-LAUNCH_ALTITUDE_MSL:.0f}m, vz={vz_burnout:.0f}m/s')
        return 0, None

    # If airbrakes deployment needed, check if max deployment isn't overkill
    flight_max_brakes = airbrakes_sim(environment, rocket, initial_solution, MAX_DEPLOYMENT_ANGLE)

    if flight_max_brakes[0] > TARGET_APOGEE_M:
        print(f'Max airbrakes deployment needed for h={h_burnout-LAUNCH_ALTITUDE_MSL:.0f}m, vz={vz_burnout:.0f}m/s')
        return MAX_DEPLOYMENT_ANGLE, flight_max_brakes[1]

    # If an intermediate amount of stopping power is needed, run binary search
    print("NEED AN INTERMEDIATE AMOUNT OF STOPPING POWERRRR")
    lower_bound = 0
    upper_bound = MAX_DEPLOYMENT_ANGLE
    while upper_bound - lower_bound > TOLERANCE_BINARY_SEARCH:
        deployment_angle = (upper_bound + lower_bound) / 2
        apogee, retract_time = airbrakes_sim(environment, rocket, initial_solution, deployment_angle)
        if apogee > TARGET_APOGEE_M:
            lower_bound = deployment_angle
        else:
            upper_bound = deployment_angle
    print(f"Optimal deployment angle: {(upper_bound + lower_bound) / 2:.1f}° for h={h_burnout-LAUNCH_ALTITUDE_MSL:.0f}m, vz={vz_burnout:.0f}m/s")
    return (upper_bound + lower_bound) / 2, retract_time


# ==================== LOOKUP TABLE GENERATION ====================

def _worker_init():
    """Worker initializer for multiprocessing: create per-process simulation templates."""
    global environment, base_rocket
    environment, base_rocket = build_simulation_base()

def _worker_task(task):
    """Top-level worker task. Expects (h, v) and returns (h, v, angle, retract_time)."""
    h, v = task
    deployment_angle, retraction_time = find_optimal_deployment(h + LAUNCH_ALTITUDE_MSL, v)
    if retraction_time is None:
        retraction_time = 0
    return (h, v, deployment_angle, retraction_time)

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
    start_time = time.time()

    # Build task list
    tasks = [(float(h), float(v)) for h in heights for v in velocities]

    cpu_count = multiprocessing.cpu_count()
    use_parallel = cpu_count > 1 and total_points > 1

    if use_parallel:
        workers = min(cpu_count, total_points)
        print(f"Running in parallel with {workers} workers (cpus={cpu_count})")
        with Pool(processes=workers, initializer=_worker_init) as p:
            completed = 0
            for result in p.imap_unordered(_worker_task, tasks):
                h, v, deployment_angle, retraction_time = result
                lookup_table.append({
                    'burnout_height_m': round(h, 2),
                    'burnout_velocity_ms': round(v, 2),
                    'deployment_angle_deg': round(deployment_angle, 4),
                    'retraction_time_s': round(retraction_time, 2)
                })
                completed += 1
                # Progress
                if completed % max(1, total_points // 20) == 0 or completed == 1:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_points - completed) / rate if rate > 0 else 0
                    print(f"  Point {completed}/{total_points} ({100*completed/total_points:.1f}%) ETA: {eta/60:.1f}min")
    else:
        point_count = 0
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
                if retraction_time is None:
                    retraction_time = 0

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
    
    # Build index sets and mapping
    heights = sorted({row['burnout_height_m'] for row in lookup_table})
    velocities = sorted({row['burnout_velocity_ms'] for row in lookup_table})
    cell_map = {
        (row['burnout_height_m'], row['burnout_velocity_ms']): (
            row['deployment_angle_deg'], row['retraction_time_s']
        )
        for row in lookup_table
    }

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header comments
        writer.writerow(['# Airbrakes Lookup Table - Hyperion with Cesaroni M2505'])
        writer.writerow([f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([f'# Target Apogee: {TARGET_APOGEE_M}m'])
        writer.writerow([f'# Grid: {HEIGHT_POINTS}x{VELOCITY_POINTS}'])
        writer.writerow(['#'])
        
        # Table header: empty first cell, then velocity column headers
        header_row = ['burnout_height_m \\ burnout_velocity_ms'] + [f"{v:.2f}" for v in velocities]
        writer.writerow(header_row)
        
        # Data rows: each row starts with the height, then cells "angle;retraction_time"
        for h in heights:
            row = [f"{h:.2f}"]
            for v in velocities:
                val = cell_map.get((h, v))
                if val is None:
                    cell = ''
                else:
                    angle, retract = val
                    cell = f"{angle};{retract}"
                row.append(cell)
            writer.writerow(row)
    
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
                
    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    # print(find_optimal_deployment(1600, 185))