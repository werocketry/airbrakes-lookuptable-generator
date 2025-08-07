#!/usr/bin/env python3
"""
Quick test script to debug RocketPy components
Run this first to identify where the issue is occurring
"""

import sys
import time

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import numpy as np
        print("  ✓ numpy")
        
        import pandas as pd
        print("  ✓ pandas")
        
        from rocketpy import Rocket, SolidMotor, Environment, Flight, Function
        print("  ✓ rocketpy")
        
        print("All imports successful!")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_motor_creation():
    """Test motor creation"""
    print("\nTesting motor creation...")
    try:
        from rocketpy import SolidMotor, Function
        
        # Simple thrust curve
        thrust_curve = Function(
            source=[(0, 0), (0.5, 1000), (2.5, 1000), (3, 0)],
            inputs="Time (s)",
            outputs="Thrust (N)",
            interpolation="linear",
            extrapolation="zero"
        )
        
        motor = SolidMotor(
            thrust_source=thrust_curve,
            dry_mass=2.866,
            dry_inertia=[0.001, 0.001, 0.0002],
            center_of_dry_mass_position=0.317,
            grain_number=4,
            grain_density=1815,
            grain_outer_radius=0.049,
            grain_initial_inner_radius=0.015,
            grain_initial_height=0.15,
            grain_separation=0.005,
            grains_center_of_mass_position=0.217,
            nozzle_radius=0.021,
            throat_radius=0.010,
            burn_time=3.0,
            nozzle_position=0,
            coordinate_system_orientation="nozzle_to_combustion_chamber"
        )
        
        print(f"  ✓ Motor created, burn time: {motor.burn_out_time:.2f}s")
        return motor
    except Exception as e:
        print(f"  ✗ Motor creation failed: {e}")
        return None

def test_rocket_creation(motor):
    """Test rocket creation"""
    print("\nTesting rocket creation...")
    try:
        from rocketpy import Rocket, Function
        import numpy as np
        
        # Simple drag curve
        drag_curve = Function(
            source=[(0, 0.4), (1, 0.4), (2, 0.6)],
            inputs="Mach",
            outputs="Cd",
            interpolation="linear"
        )
        
        rocket = Rocket(
            radius=0.0715,
            mass=20.5,
            inertia=(6.0, 6.0, 0.05),
            power_off_drag=drag_curve,
            power_on_drag=drag_curve,
            center_of_mass_without_motor=1.2,
            coordinate_system_orientation="nose_to_tail"
        )
        
        # Add motor
        if motor:
            rocket.add_motor(motor, position=0)
        
        print(f"  ✓ Rocket created, mass: {rocket.total_mass(0):.1f}kg")
        return rocket
    except Exception as e:
        print(f"  ✗ Rocket creation failed: {e}")
        return None

def test_environment_creation():
    """Test environment creation"""
    print("\nTesting environment creation...")
    try:
        from rocketpy import Environment
        
        env = Environment(
            latitude=32.99,
            longitude=-106.97,
            elevation=1401,
            datum="WGS84"
        )
        
        # Try simple atmosphere first
        print("  Testing simple atmosphere setup...")
        
        def temp_func(h):
            return 308.15 - 0.00817 * h  # 35°C at ground level
        
        def pressure_func(h):
            return 86400 * (1 - 0.00817 * h / 308.15) ** (9.81 / (287.058 * 0.00817))
        
        env.set_atmospheric_model(
            type="custom_atmosphere",
            temperature=temp_func,
            pressure=pressure_func
        )
        
        print(f"  ✓ Environment created at elevation {env.elevation}m")
        return env
    except Exception as e:
        print(f"  ✗ Environment creation failed: {e}")
        return None

def test_simple_flight(rocket, env):
    """Test a simple flight simulation"""
    print("\nTesting flight simulation...")
    try:
        from rocketpy import Flight
        
        print("  Creating flight object...")
        start_time = time.time()
        
        flight = Flight(
            rocket=rocket,
            environment=env,
            rail_length=5.18,
            inclination=86,
            heading=0,
            terminate_on_apogee=True,
            max_time=30,
            verbose=False
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Flight simulation completed in {elapsed:.1f}s")
        print(f"  ✓ Apogee: {flight.apogee - env.elevation:.0f}m AGL")
        print(f"  ✓ Max velocity: {flight.max_mach_number * 343:.0f}m/s")
        
        return flight
    except Exception as e:
        print(f"  ✗ Flight simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests"""
    print("RocketPy Component Test")
    print("=" * 30)
    
    # Test imports
    if not test_imports():
        print("Fix import issues before continuing")
        return
    
    # Test motor
    motor = test_motor_creation()
    if not motor:
        print("Fix motor creation before continuing")
        return
    
    # Test rocket
    rocket = test_rocket_creation(motor)
    if not rocket:
        print("Fix rocket creation before continuing")
        return
    
    # Test environment
    env = test_environment_creation()
    if not env:
        print("Fix environment creation before continuing")
        return
    
    # Test flight
    flight = test_simple_flight(rocket, env)
    if not flight:
        print("Fix flight simulation before continuing")
        return
    
    print("\n" + "=" * 30)
    print("✓ All tests passed! RocketPy setup is working.")
    print("You can now run main.py")

if __name__ == "__main__":
    main()