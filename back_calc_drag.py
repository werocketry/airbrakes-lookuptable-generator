import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('2024-06-21-serial-6583-flight-0004.csv')
# drop all rows after 23.7 s and all rows before -0.1 s, add 0.1 s to each time value
df = df.loc[df['time'] <= 23.7]
df = df.loc[df['time'] >= -0.1]
df['time'] = df['time'] + 0.1

# drop all but one row when rows have identical times
df = df.loc[df['time'].diff() != 0]

# drop all columns except for time, height, pressure, and acceleration
df = df[['time', 'height', 'pressure', 'acceleration']]

# reset the index
df = df.reset_index(drop=True)

# write to csv
# df.to_csv('processed_flight_data.csv', index=False)

times = df['time']
heights_raw = df['height']
pressures = df['pressure']
accelerations = df['acceleration']

def cumtrapz_np(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    return np.concatenate(([0.0], np.cumsum(avg * dx)))

velocity_from_accel = cumtrapz_np(accelerations, times)
displacement_from_accel = cumtrapz_np(velocity_from_accel, times)

# plot distance vs time, using both 
# plt.figure(figsize=(10, 6))
# plt.plot(times, heights_raw, label='Height (m)')
# plt.plot(times, displacement_from_accel, label='Displacement from Acceleration (m)')
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (m)')
# plt.title('Rocket Distance vs Time')
# plt.grid()
# plt.legend()
# plt.show()

# get v_z curve as the derivative of height_raw
v_z_ddt_h_raw = np.gradient(heights_raw, times)

# smooth v_z_ddt_h_raw
def moving_average_reflect(x, window):
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")
dt_med = np.median(np.diff(times))
if not np.isfinite(dt_med) or dt_med <= 0:
    dt_med = (times.iloc[-1] - times.iloc[0]) / max(10, len(times)-1)

win = int(round(0.5 / dt_med))
win = max(win, 5)
if win % 2 == 0:
    win += 1
if win >= len(times):
    win = len(times) - 1 if (len(times) - 1) % 2 == 1 else len(times) - 2
win = max(win, 5 if len(times) >= 5 else len(times))

v_smooth = moving_average_reflect(v_z_ddt_h_raw, win)

# fit quadratic
mask_ascent = (times >= 3.0) & (times <= 23.0)
t_since_burnout = (times[mask_ascent] - 3.0).values
v_target = v_smooth[mask_ascent.values]
coeffs = np.polyfit(t_since_burnout, v_target, 2)
a, b, c = coeffs
v_hat = np.polyval(coeffs, t_since_burnout)
ss_res = np.sum((v_target - v_hat)**2)
ss_tot = np.sum((v_target - np.mean(v_target))**2)
R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

print("Quadratic fit (ascent only, t=0 at 3.0 s):")
print(f"a = {a:.9f}")
print(f"b = {b:.9f}")
print(f"c = {c:.9f}")
print(f"R^2 = {R2:.6f}")

mask = times >= 3.0
t_shift = times[mask] - 3.0
vz_quadratic_fit_of_smoothed_ddt_h = a * t_shift**2 + b * t_shift + c

if __name__ == "__main__":
    # plot speed vs time, both v_z_ddt_h_raw, v_quadratic_fit_of_smoothed_ddt_h, and velocity_from_accel
    plt.figure(figsize=(10, 6))
    plt.plot(times, v_z_ddt_h_raw, label='v_z_ddt_h_raw')
    plt.plot(times, velocity_from_accel, label='velocity_from_accel')
    plt.plot(times[mask], vz_quadratic_fit_of_smoothed_ddt_h, label='vz_quadratic_fit_of_smoothed_ddt_h', linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Rocket Speed vs Time')
    plt.grid()
    plt.legend()
    plt.show()

# Back-calculation of drag
# grab area, etc from main

# Hyperion I launch conditions
launchpad_pressure_hyp1 = 86368 # Pa
launchpad_temp_hyp1 = 25.8 # deg C
lapse_rate_SAC = -0.00817
def temp_at_h_hyp1(h):
    return launchpad_temp_hyp1 + lapse_rate_SAC * h
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

grav_SAC = get_local_gravity(32.99, 1401)
R_universal = 8.3144598
MM_air = 0.0289644
R_AIR = R_universal / MM_air

def pressure_at_h_SAC(h):
    return launchpad_pressure_hyp1 * pow(1-h*lapse_rate_SAC/(launchpad_temp_hyp1+273.15), grav_SAC/(R_AIR*lapse_rate_SAC))

def air_density_fn(pressure, temp):
    return pressure / (R_AIR * temp)

