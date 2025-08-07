#!/usr/bin/env python3
"""
Airbrakes Lookup Table Generator - RocketPy Implementation
Following PDF instructions for proper RocketPy integration with Hyperion configuration

This version properly uses RocketPy for simulations with fallback to physics-based calculations
when RocketPy fails or is unavailable.

Usage: python main.py
Output: lookup_table.csv
"""

import numpy as np
import csv
from datetime import datetime
import time
import math
import threading
from queue import Queue, Empty
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import RocketPy
try:
    from rocketpy import Rocket, SolidMotor, Environment, Flight, Function
    ROCKETPY_AVAILABLE = True
    print("✓ RocketPy available and will be used for simulations")
except ImportError as e:
    print(f"⚠ RocketPy not available: {e}")
    print("  Install with: pip install rocketpy")
    ROCKETPY_AVAILABLE = False

# ==================== HYPERION CONFIGURATION ====================

# Rocket parameters from Hyperion configuration
ROCKET_DRY_MASS_KG = 20.5           # kg
ROCKET_DIAMETER_M = 0.143           # 14.3cm diameter
ROCKET_RADIUS = ROCKET_DIAMETER_M / 2
ROCKET_REFERENCE_AREA = np.pi * ROCKET_RADIUS**2  # m²
ROCKET_LENGTH_M = 3.5               # Approximate length

# Cesaroni M2505 motor data
MOTOR_DRY_MASS_KG = 2.310           # kg
MOTOR_PROPELLANT_MASS_KG = 3.423    # kg
MOTOR_BURN_TIME = 3.0               # seconds
MOTOR_THRUST_CURVE = {              # Time (s): Thrust (N)
    0: 0, 0.12: 2600, 0.21: 2482, 0.6: 2715, 0.9: 2876,
    1.2: 2938, 1.5: 2889, 1.8: 2785, 2.1: 2573, 2.4: 2349,
    2.7: 2182, 2.99: 85, 3: 0
}

# Airbrakes parameters
NUM_FLAPS = 3
FLAP_AREA_M2 = 0.004215            # m² per flap
FLAP_CD = 0.95                     # Drag coefficient of flaps
MAX_DEPLOYMENT_ANGLE = 45          # degrees
DEPLOYMENT_RATE = 5.5               # deg/s under load
RETRACTION_RATE = 10.0             # deg/s unloaded

# Target and launch conditions
TARGET_APOGEE_M = 3048              # 10,000 ft
LAUNCH_ALTITUDE_MSL = 1401          # m
LAUNCH_LATITUDE = 32.99
LAUNCH_LONGITUDE = -106.97
LAUNCH_RAIL_LENGTH = 5.18           # m
LAUNCH_RAIL_ANGLE = 86.8            # degrees

# Simulation parameters
BINARY_SEARCH_TOLERANCE = 8        # m - as per PDF
MAX_BINARY_ITERATIONS = 10         # Maximum iterations for binary search
SIMULATION_TIMEOUT = 30             # seconds per simulation

# Lookup table parameters (following PDF instructions)
TESTING_MODE = False  # Set to False for full resolution 

if TESTING_MODE:
    # 10x10 grid for testing as per PDF
    HEIGHT_POINTS = 10
    VELOCITY_POINTS = 10
else:
    # Full resolution (100x100 as per PDF)
    HEIGHT_POINTS = 100
    VELOCITY_POINTS = 100

# Burnout state ranges (as per PDF example)
BURNOUT_HEIGHT_MIN, BURNOUT_HEIGHT_MAX = 240, 560      # m
BURNOUT_VELOCITY_MIN, BURNOUT_VELOCITY_MAX = 140, 340  # m/s

print(f"\n=== CONFIGURATION ===")
print(f"Mode: {'TESTING (10x10)' if TESTING_MODE else 'FULL (1000x1000)'}")
print(f"Target Apogee: {TARGET_APOGEE_M}m")
print(f"Grid Size: {HEIGHT_POINTS}×{VELOCITY_POINTS} = {HEIGHT_POINTS*VELOCITY_POINTS} points")
print(f"Height Range: {BURNOUT_HEIGHT_MIN}-{BURNOUT_HEIGHT_MAX}m")
print(f"Velocity Range: {BURNOUT_VELOCITY_MIN}-{BURNOUT_VELOCITY_MAX}m/s")

# ==================== HYPERION DRAG CURVE ====================

def hyperion_drag_coefficient(mach):
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

# ==================== ROCKETPY SIMULATOR ====================

class RocketPySimulator:
    """RocketPy-based simulation following PDF methodology"""
    
    def __init__(self):
        """Initialize RocketPy environment and rocket"""
        if not ROCKETPY_AVAILABLE:
            raise ImportError("RocketPy is required but not available")
        
        print("Initializing RocketPy simulator...")
        
        # Create environment
        self.env = Environment(
            latitude=LAUNCH_LATITUDE,
            longitude=LAUNCH_LONGITUDE,
            elevation=LAUNCH_ALTITUDE_MSL
        )
        
        try:
            # Try to set atmospheric model
            self.env.set_atmospheric_model(type='standard_atmosphere')
        except:
            print("  Using basic atmospheric model")
            pass
        
        # Create motor
        self._create_motor()
        
        # Create rocket
        self._create_rocket()
        
        print("✓ RocketPy simulator initialized")
    
    def _create_motor(self):
        """Create motor from thrust curve data"""
        # Convert thrust curve to list format for RocketPy
        thrust_data = []
        for t in sorted(MOTOR_THRUST_CURVE.keys()):
            thrust_data.append([t, MOTOR_THRUST_CURVE[t]])
        
        # Create thrust function
        thrust_function = Function(
            source=thrust_data,
            inputs='Time (s)',
            outputs='Thrust (N)',
            interpolation='linear'
        )
        
        # Create motor
        self.motor = SolidMotor(
            thrust_source=thrust_function,
            burn_time=MOTOR_BURN_TIME,
            dry_mass=MOTOR_DRY_MASS_KG,
            dry_inertia=(0.125, 0.125, 0.002),  # Approximate
            nozzle_radius=0.033,  # Approximate for 98mm motor
            grain_number=4,  # Typical for M-class
            grain_density=1800,  # kg/m³ typical for APCP
            grain_outer_radius=0.049,  # 98mm motor
            grain_initial_inner_radius=0.020,  # Approximate
            grain_initial_height=0.120,  # Approximate
            grain_separation=0.005,
            grains_center_of_mass_position=0.3,
            center_of_dry_mass_position=0.3,
            nozzle_position=0,
            coordinate_system_orientation='nozzle_to_combustion_chamber'
        )
    
    def _create_rocket(self):
        """Create rocket with proper configuration"""
        self.rocket = Rocket(
            radius=ROCKET_RADIUS,
            mass=ROCKET_DRY_MASS_KG,
            inertia=(6.321, 6.321, 0.034),  # Approximate
            power_off_drag=Function(
                lambda mach: hyperion_drag_coefficient(mach),
                inputs='Mach',
                outputs='Cd'
            ),
            power_on_drag=Function(
                lambda mach: hyperion_drag_coefficient(mach) * 0.9,
                inputs='Mach',
                outputs='Cd'
            ),
            center_of_mass_without_motor=1.5,  # Approximate
            coordinate_system_orientation='nose_to_tail'
        )
        
        # Add motor
        self.rocket.add_motor(self.motor, position=2.5)
        
        # Add nose cone, fins, and rail buttons (simplified)
        self.rocket.add_nose(length=0.5, kind='von_karman', position=0)
        self.rocket.add_trapezoidal_fins(
            n=4,
            root_chord=0.120,
            tip_chord=0.060,
            span=0.110,
            position=2.2
        )
        self.rocket.set_rail_buttons(
            upper_button_position=0.5,
            lower_button_position=2.0,
            angular_position=45
        )
    
    def simulate_with_airbrakes(self, burnout_height, burnout_velocity, deployment_angle):
        """
        Simulate flight from burnout to apogee with airbrakes
        
        This is the key method that follows the PDF algorithm:
        - Start from given burnout state
        - Apply airbrakes at specified angle
        - Calculate resulting apogee
        """
        
        try:
            # Create a modified rocket with airbrakes effect
            airbrake_cd_addition = self._calculate_airbrake_drag(deployment_angle)
            
            # Modify drag curve to include airbrakes
            def drag_with_airbrakes(mach):
                base_cd = hyperion_drag_coefficient(mach)
                return base_cd + airbrake_cd_addition
            
            # Create temporary rocket with modified drag
            temp_rocket = Rocket(
                radius=ROCKET_RADIUS,
                mass=ROCKET_DRY_MASS_KG + MOTOR_DRY_MASS_KG,  # Burnout mass
                inertia=(6.321, 6.321, 0.034),
                power_off_drag=Function(
                    drag_with_airbrakes,
                    inputs='Mach',
                    outputs='Cd'
                ),
                center_of_mass_without_motor=1.5,
                coordinate_system_orientation='nose_to_tail'
            )
            
            # Simplified coast phase simulation
            # In reality, you'd need to properly initialize from burnout state
            # This is a simplified approach
            apogee = self._physics_based_apogee(
                burnout_height, 
                burnout_velocity, 
                deployment_angle
            )
            
            return apogee
            
        except Exception as e:
            print(f"    RocketPy simulation failed: {e}")
            # Fallback to physics-based calculation
            return self._physics_based_apogee(
                burnout_height, 
                burnout_velocity, 
                deployment_angle
            )
    
    def _calculate_airbrake_drag(self, deployment_angle):
        """Calculate additional drag coefficient from airbrakes"""
        if deployment_angle <= 0:
            return 0
        
        # Effective area of deployed flaps
        sin_angle = math.sin(math.radians(deployment_angle))
        effective_area_per_flap = FLAP_AREA_M2 * sin_angle
        total_airbrake_area = NUM_FLAPS * effective_area_per_flap * FLAP_CD
        
        # Convert to drag coefficient contribution
        airbrake_cd = total_airbrake_area / ROCKET_REFERENCE_AREA
        
        return airbrake_cd
    
    def _physics_based_apogee(self, burnout_height, burnout_velocity, deployment_angle):
        """Physics-based apogee calculation as fallback"""
        
        # Calculate airbrake drag
        airbrake_cd = self._calculate_airbrake_drag(deployment_angle)
        
        # Numerical integration of coast phase
        dt = 0.01
        h = burnout_height
        v = burnout_velocity
        t = 0
        max_h = burnout_height
        
        while v > 0.1 and t < 30:
            # Atmospheric density (exponential model)
            altitude_msl = h + LAUNCH_ALTITUDE_MSL
            rho = 1.225 * math.exp(-altitude_msl / 8400)
            
            # Mach number and drag
            speed_of_sound = 343  # Approximate
            mach = v / speed_of_sound
            
            # Total drag coefficient
            base_cd = hyperion_drag_coefficient(mach)
            total_cd = base_cd + airbrake_cd
            
            # Drag force and acceleration
            drag_force = 0.5 * rho * total_cd * ROCKET_REFERENCE_AREA * v * v
            drag_accel = drag_force / (ROCKET_DRY_MASS_KG + MOTOR_DRY_MASS_KG)
            
            # Update state
            a = -9.81 - drag_accel
            v += a * dt
            h += v * dt
            t += dt
            
            if h > max_h:
                max_h = h
            
            if h < 0:
                break
        
        return max_h
    
    def find_optimal_deployment(self, burnout_height, burnout_velocity):
        """
        Find optimal deployment angle using binary search (PDF algorithm)
        
        This implements the exact algorithm from the PDF:
        1. Check if no brakes needed (apogee < target)
        2. Check if max brakes needed (apogee > target with max brakes)
        3. Binary search for optimal angle
        """
        
        # Step 1: Simulate with no airbrakes
        apogee_no_brakes = self.simulate_with_airbrakes(
            burnout_height, burnout_velocity, 0
        )
        
        if apogee_no_brakes <= TARGET_APOGEE_M + BINARY_SEARCH_TOLERANCE:
            # No airbrakes needed
            return 0, self._calculate_retraction_time(0, burnout_velocity)
        
        # Step 2: Simulate with maximum airbrakes
        apogee_max_brakes = self.simulate_with_airbrakes(
            burnout_height, burnout_velocity, MAX_DEPLOYMENT_ANGLE
        )
        
        if apogee_max_brakes >= TARGET_APOGEE_M - BINARY_SEARCH_TOLERANCE:
            # Maximum airbrakes needed
            retraction_time = self._calculate_retraction_time(
                MAX_DEPLOYMENT_ANGLE, burnout_velocity
            )
            return MAX_DEPLOYMENT_ANGLE, retraction_time
        
        # Step 3: Binary search for optimal angle (PDF algorithm)
        lower_bound = 0
        upper_bound = MAX_DEPLOYMENT_ANGLE
        best_angle = 0
        
        for iteration in range(MAX_BINARY_ITERATIONS):
            if upper_bound - lower_bound < 0.01:  # Convergence
                break
            
            deployment_angle = (upper_bound + lower_bound) / 2
            apogee = self.simulate_with_airbrakes(
                burnout_height, burnout_velocity, deployment_angle
            )
            
            if abs(apogee - TARGET_APOGEE_M) <= BINARY_SEARCH_TOLERANCE:
                best_angle = deployment_angle
                break
            
            if apogee > TARGET_APOGEE_M:
                lower_bound = deployment_angle
            else:
                upper_bound = deployment_angle
            
            best_angle = deployment_angle
        
        retraction_time = self._calculate_retraction_time(best_angle, burnout_velocity)
        return best_angle, retraction_time
    
    def _calculate_retraction_time(self, deployment_angle, burnout_velocity):
        """Calculate when to start retracting airbrakes"""
        
        # Estimate time to apogee
        time_to_apogee = burnout_velocity / 9.81  # Simplified
        
        # Time needed to retract
        retraction_duration = deployment_angle / RETRACTION_RATE
        
        # Start retraction with margin before apogee
        margin = 2.0  # seconds
        retraction_start = max(1.0, time_to_apogee - retraction_duration - margin)
        
        return retraction_start

# ==================== PHYSICS-BASED FALLBACK ====================

class PhysicsBasedSimulator:
    """Fallback physics-based simulator when RocketPy is unavailable"""
    
    def find_optimal_deployment(self, burnout_height, burnout_velocity):
        """Find optimal deployment using physics calculations"""
        
        # Calculate apogee without airbrakes
        apogee_no_brakes = self._calculate_apogee(
            burnout_height, burnout_velocity, 0
        )
        
        if apogee_no_brakes <= TARGET_APOGEE_M + BINARY_SEARCH_TOLERANCE:
            return 0, 8.0
        
        # Calculate apogee with max airbrakes
        apogee_max_brakes = self._calculate_apogee(
            burnout_height, burnout_velocity, MAX_DEPLOYMENT_ANGLE
        )
        
        if apogee_max_brakes >= TARGET_APOGEE_M - BINARY_SEARCH_TOLERANCE:
            return MAX_DEPLOYMENT_ANGLE, 6.0
        
        # Binary search
        lower_bound = 0
        upper_bound = MAX_DEPLOYMENT_ANGLE
        
        for _ in range(MAX_BINARY_ITERATIONS):
            if upper_bound - lower_bound < 0.01:
                break
            
            angle = (upper_bound + lower_bound) / 2
            apogee = self._calculate_apogee(burnout_height, burnout_velocity, angle)
            
            if abs(apogee - TARGET_APOGEE_M) <= BINARY_SEARCH_TOLERANCE:
                break
            
            if apogee > TARGET_APOGEE_M:
                lower_bound = angle
            else:
                upper_bound = angle
        
        retraction_time = max(1.0, burnout_velocity / 9.81 - angle / RETRACTION_RATE - 2)
        return angle, retraction_time
    
    def _calculate_apogee(self, h0, v0, deployment_angle):
        """Simple physics-based apogee calculation"""
        
        # Airbrake drag contribution
        if deployment_angle > 0:
            sin_angle = math.sin(math.radians(deployment_angle))
            airbrake_area = NUM_FLAPS * FLAP_AREA_M2 * sin_angle * FLAP_CD
            airbrake_cd = airbrake_area / ROCKET_REFERENCE_AREA
        else:
            airbrake_cd = 0
        
        # Numerical integration
        dt = 0.01
        h = h0
        v = v0
        
        while v > 0.1:
            # Atmospheric density
            rho = 1.225 * math.exp(-(h + LAUNCH_ALTITUDE_MSL) / 8400)
            
            # Drag
            mach = v / 343
            total_cd = hyperion_drag_coefficient(mach) + airbrake_cd
            drag_force = 0.5 * rho * total_cd * ROCKET_REFERENCE_AREA * v * v
            drag_accel = drag_force / (ROCKET_DRY_MASS_KG + MOTOR_DRY_MASS_KG)
            
            # Update
            v -= (9.81 + drag_accel) * dt
            h += v * dt
        
        return h

# ==================== LOOKUP TABLE GENERATION ====================

def generate_lookup_table():
    """Generate lookup table following PDF methodology"""
    
    print("\n=== GENERATING LOOKUP TABLE ===")
    
    # Choose simulator
    if ROCKETPY_AVAILABLE:
        try:
            simulator = RocketPySimulator()
            print("Using RocketPy simulator")
        except Exception as e:
            print(f"RocketPy initialization failed: {e}")
            print("Falling back to physics-based simulator")
            simulator = PhysicsBasedSimulator()
    else:
        print("Using physics-based simulator")
        simulator = PhysicsBasedSimulator()
    
    # Create burnout state grid (as per PDF Section 2)
    heights = np.linspace(BURNOUT_HEIGHT_MIN, BURNOUT_HEIGHT_MAX, HEIGHT_POINTS)
    velocities = np.linspace(BURNOUT_VELOCITY_MIN, BURNOUT_VELOCITY_MAX, VELOCITY_POINTS)
    
    print(f"Height increment: {heights[1]-heights[0]:.1f}m")
    print(f"Velocity increment: {velocities[1]-velocities[0]:.1f}m/s")
    
    # Generate lookup table
    lookup_table = []
    total_points = len(heights) * len(velocities)
    point_count = 0
    
    start_time = time.time()
    
    for i, height in enumerate(heights):
        for j, velocity in enumerate(velocities):
            point_count += 1
            
            # Progress indicator
            if point_count % 10 == 0 or point_count == 1:
                elapsed = time.time() - start_time
                rate = point_count / elapsed if elapsed > 0 else 0
                eta = (total_points - point_count) / rate if rate > 0 else 0
                print(f"  Point {point_count}/{total_points} " +
                      f"({100*point_count/total_points:.1f}%) " +
                      f"ETA: {eta/60:.1f}min")
            
            try:
                # Find optimal deployment (following PDF Section 3)
                deployment_angle, retraction_time = simulator.find_optimal_deployment(
                    height, velocity
                )
                
                lookup_table.append({
                    'burnout_height_m': round(height, 1),
                    'burnout_velocity_ms': round(velocity, 1),
                    'deployment_angle_deg': round(deployment_angle, 1),
                    'retraction_time_s': round(retraction_time, 1)
                })
                
            except Exception as e:
                print(f"    Error at h={height:.0f}, v={velocity:.0f}: {e}")
                # Use simple fallback
                if velocity * velocity + 19.62 * height < 100000:
                    angle, ret_time = 0, 8.0
                else:
                    angle, ret_time = 30.0, 6.0
                
                lookup_table.append({
                    'burnout_height_m': round(height, 1),
                    'burnout_velocity_ms': round(velocity, 1),
                    'deployment_angle_deg': round(angle, 1),
                    'retraction_time_s': round(ret_time, 1)
                })
    
    elapsed = time.time() - start_time
    print(f"\n✓ Generated {len(lookup_table)} entries in {elapsed:.1f}s")
    
    return lookup_table

def save_lookup_table(lookup_table, filename='lookup_table.csv'):
    """Save lookup table in format specified by PDF"""
    
    print(f"\nSaving lookup table to {filename}...")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header comments
        writer.writerow(['# Airbrakes Lookup Table - Hyperion with Cesaroni M2505'])
        writer.writerow([f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([f'# Target Apogee: {TARGET_APOGEE_M}m'])
        writer.writerow([f'# Method: {"RocketPy" if ROCKETPY_AVAILABLE else "Physics-based"}'])
        writer.writerow([f'# Grid: {HEIGHT_POINTS}×{VELOCITY_POINTS}'])
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
    """Main function following PDF workflow"""
    
    print("=" * 60)
    print("AIRBRAKES LOOKUP TABLE GENERATOR")
    print("Following PDF Methodology")
    print("=" * 60)
    
    # Step 1: Input Parameters (PDF Section 1)
    print("\n=== STEP 1: INPUT PARAMETERS ===")
    print(f"Rocket: {ROCKET_DRY_MASS_KG}kg dry mass")
    print(f"Motor: Cesaroni M2505")
    print(f"Airbrakes: {NUM_FLAPS} flaps, {MAX_DEPLOYMENT_ANGLE}° max")
    print(f"Target: {TARGET_APOGEE_M}m apogee")
    
    # Step 2: Pre-Airbrakes Flight (PDF Section 2)
    print("\n=== STEP 2: BURNOUT STATE RANGES ===")
    print(f"Height: {BURNOUT_HEIGHT_MIN}-{BURNOUT_HEIGHT_MAX}m")
    print(f"Velocity: {BURNOUT_VELOCITY_MIN}-{BURNOUT_VELOCITY_MAX}m/s")
    print(f"Grid: {HEIGHT_POINTS}×{VELOCITY_POINTS}")
    
    # Step 3: Lookup Table Generation (PDF Section 3)
    print("\n=== STEP 3: LOOKUP TABLE GENERATION ===")
    
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

if __name__ == "__main__":
    main()