#!/usr/bin/env python3
"""
Side View of Minimum Radial Ramp (3.5m rise) with Porsche 997 Turbo ON the ramp
Shows car behavior during the path from street (0m) to garage (3.5m above)
Based on minimum_radial_3.5m_blueprint.png design
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Ramp parameters from minimum_radial_3.5m design
R_CENTERLINE = 7.50  # m
VERTICAL_RISE = 3.5  # m (garage is ABOVE street)
ARC_LENGTH = np.pi * R_CENTERLINE / 2  # ~11.78m
GARAGE_LENGTH = 5.0  # m
STREET_LENGTH = 2.0  # m to show before ramp

# Car dimensions (as specified by user)
CAR_LENGTH = 4.450  # m
CAR_WIDTH = 1.852  # m

# Cubic spline coefficients for vertical profile
# Profile goes UP: z = a*s³ + b*s² where z goes from 0 to +3.5m
L = ARC_LENGTH
H = VERTICAL_RISE
a = 2 * H / L**3
b = -3 * H / L**2

def cubic_spline(s):
    """Cubic spline profile: returns elevation at arc distance s"""
    return -(a * s**3 + b * s**2)

def get_slope_at(s):
    """Get slope (dz/ds) at arc position s"""
    return -(3 * a * s**2 + 2 * b * s)

def get_angle_at(s):
    """Get angle in degrees at arc position s"""
    slope = get_slope_at(s)
    return np.degrees(np.arctan(slope))

# Create figure
fig, ax = plt.subplots(figsize=(20, 12))

# Generate ramp profile along arc distance
s_ramp = np.linspace(0, ARC_LENGTH, 1000)
z_ramp = cubic_spline(s_ramp)

# Define zones
street_start = -STREET_LENGTH
garage_end = ARC_LENGTH + GARAGE_LENGTH

# Draw street (gray filled area)
ax.fill([street_start, 0, 0, street_start],
        [-0.3, -0.3, 0, 0],
        color='gray', alpha=0.5, label='Street (Level 0m)')
ax.plot([street_start, 0], [0, 0], 'k-', linewidth=2)

# Draw ramp surface (tan/brown fill under the line)
ax.fill_between(s_ramp, z_ramp - 0.15, z_ramp, color='#8B7355', alpha=0.7)
ax.plot(s_ramp, z_ramp, 'k-', linewidth=3, label='Ramp')

# Draw garage floor (blue)
ax.fill([ARC_LENGTH, garage_end, garage_end, ARC_LENGTH],
        [VERTICAL_RISE, VERTICAL_RISE, VERTICAL_RISE + 0.3, VERTICAL_RISE + 0.3],
        color='blue', alpha=0.5, label=f'Garage ({GARAGE_LENGTH}m parking)')
ax.plot([ARC_LENGTH, garage_end], [VERTICAL_RISE, VERTICAL_RISE], 'k-', linewidth=2)

# Draw garage back wall indication
ax.fill([garage_end - 0.2, garage_end, garage_end, garage_end - 0.2],
        [VERTICAL_RISE, VERTICAL_RISE, VERTICAL_RISE + 2, VERTICAL_RISE + 2],
        color='blue', alpha=0.3)

# Load and place the car image ON THE RAMP
try:
    # Load image with PIL for rotation (using cleaned version without ground line)
    car_pil = Image.open('/workspaces/RAMP/porsche_997_turbo_clean.png')

    # Position car at middle of ramp to show behavior during climb
    car_center_s = ARC_LENGTH * 0.5  # Middle of the ramp

    # Get elevation and angle at car position
    car_angle_deg = get_angle_at(car_center_s)

    # Rotate the image using PIL (negative angle because PIL rotates counter-clockwise)
    car_rotated = car_pil.rotate(-car_angle_deg, expand=True, resample=Image.BICUBIC)
    car_img_rotated = np.array(car_rotated)

    # Car image properties
    img_height, img_width = car_pil.size[1], car_pil.size[0]
    aspect_ratio = img_height / img_width
    car_height_display = CAR_LENGTH * aspect_ratio

    # The car image has 35.6% empty space at bottom and ~27.7% at top
    empty_bottom = 0.356
    actual_car_height = car_height_display * (1 - empty_bottom - 0.277)

    # Position car so wheels touch the ramp surface
    car_rear_s = car_center_s - CAR_LENGTH / 2
    car_front_s = car_center_s + CAR_LENGTH / 2

    # Get Z at front and rear wheel positions
    z_rear = cubic_spline(car_rear_s)
    z_front = cubic_spline(car_front_s)

    # Calculate the center position for the rotated image
    center_s = (car_rear_s + car_front_s) / 2
    # Position so wheels touch the ramp - account for empty space in image
    # Fine-tuned to place wheels exactly on the ramp surface
    center_z = (z_rear + z_front) / 2 + actual_car_height * 0.15

    # Calculate zoom factor to get correct car size
    # The car should be CAR_LENGTH meters long in the plot
    # OffsetImage zoom is in pixels, we need to calculate based on figure DPI and axes scale
    fig_width_inches = fig.get_figwidth()
    axes_width_data = ax.get_xlim()[1] - ax.get_xlim()[0] if ax.get_xlim()[1] != ax.get_xlim()[0] else 20

    # Approximate: for equal aspect, 1 data unit = some pixels
    # zoom = desired_size_in_data / original_size_in_data
    # Use a manual calculation
    zoom = 0.28  # Calibrated for CAR_LENGTH = 4.45m

    imagebox = OffsetImage(car_img_rotated, zoom=zoom)
    ab = AnnotationBbox(imagebox, (center_s, center_z),
                       frameon=False, pad=0)
    ax.add_artist(ab)

    # Wheel positions for reference (hidden - used for positioning verification)
    wheel_rear_s = car_rear_s + CAR_LENGTH * 0.12
    wheel_front_s = car_front_s - CAR_LENGTH * 0.12

except Exception as e:
    print(f"Could not load car image: {e}")
    import traceback
    traceback.print_exc()

# Add grid lines every 0.5m
for s in np.arange(-STREET_LENGTH, garage_end + 0.5, 0.5):
    if s % 1.0 == 0:
        ax.axvline(x=s, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    else:
        ax.axvline(x=s, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)

for z in np.arange(-0.5, VERTICAL_RISE + 1.0, 0.5):
    if z % 1.0 == 0:
        ax.axhline(y=z, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    else:
        ax.axhline(y=z, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)

# Set axis ticks every 0.5m
ax.set_xticks(np.arange(-2, garage_end + 1, 0.5), minor=True)
ax.set_xticks(np.arange(-2, garage_end + 1, 1))
ax.set_yticks(np.arange(-1, VERTICAL_RISE + 2, 0.5), minor=True)
ax.set_yticks(np.arange(-1, VERTICAL_RISE + 2, 1))

# Configure axes
ax.set_xlim(street_start - 0.5, garage_end + 0.5)
ax.set_ylim(-0.8, VERTICAL_RISE + 1.5)
ax.set_xlabel('Distance along path (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Elevation (m)', fontsize=12, fontweight='bold')
ax.set_title('Garage Ramp Side View - Porsche 997 Turbo on Ramp\n'
             f'Minimum Radial Design: R={R_CENTERLINE}m, Arc={ARC_LENGTH:.2f}m, Rise={VERTICAL_RISE}m\n'
             '(Street Level → Ramp → Garage)',
             fontsize=14, fontweight='bold')

# Add zone labels
ax.annotate('STREET\n(Level 0m)', xy=(-1, -0.15), fontsize=11, ha='center',
            fontweight='bold', color='dimgray')
ax.annotate(f'RAMP\n{ARC_LENGTH:.1f}m arc\n{VERTICAL_RISE}m rise\nMax slope: 24°',
            xy=(ARC_LENGTH/2, 1.0), fontsize=10, ha='center',
            fontweight='bold', color='black')
ax.annotate(f'GARAGE\n{GARAGE_LENGTH}m parking\n(+{VERTICAL_RISE}m)',
            xy=(ARC_LENGTH + GARAGE_LENGTH/2, VERTICAL_RISE + 0.8), fontsize=10, ha='center',
            fontweight='bold', color='blue')

# Add dimension arrows
ax.annotate('', xy=(ARC_LENGTH, -0.5), xytext=(0, -0.5),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(ARC_LENGTH/2, -0.65, f'{ARC_LENGTH:.2f} m (arc length)', ha='center',
        fontsize=10, fontweight='bold', color='green')

ax.annotate('', xy=(-1.5, 0), xytext=(-1.5, VERTICAL_RISE),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(-1.7, VERTICAL_RISE/2, f'{VERTICAL_RISE} m', ha='right', fontsize=10,
        fontweight='bold', color='red', rotation=90, va='center')

ax.annotate('', xy=(garage_end, VERTICAL_RISE + 1.2), xytext=(ARC_LENGTH, VERTICAL_RISE + 1.2),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
ax.text(ARC_LENGTH + GARAGE_LENGTH/2, VERTICAL_RISE + 1.35, f'{GARAGE_LENGTH} m',
        ha='center', fontsize=10, fontweight='bold', color='blue')

# Car dimensions
car_s = ARC_LENGTH * 0.5
ax.annotate('', xy=(car_s + CAR_LENGTH/2, -0.3), xytext=(car_s - CAR_LENGTH/2, -0.3),
            arrowprops=dict(arrowstyle='<->', color='orange', lw=1.5))
ax.text(car_s, -0.15, f'Car: {CAR_LENGTH}m', ha='center', fontsize=9,
        fontweight='bold', color='orange')

# Show slope at car position
car_angle = get_angle_at(car_s)
ax.text(car_s, cubic_spline(car_s) + 1.8, f'Slope here: {car_angle:.1f}°',
        ha='center', fontsize=10, fontweight='bold', color='purple',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.legend(loc='upper left', fontsize=10)
ax.set_aspect('equal')
ax.grid(True, which='minor', alpha=0.2)
ax.grid(True, which='major', alpha=0.4)

plt.tight_layout()
plt.savefig('/workspaces/RAMP/ramp_side_view_with_porsche.png', dpi=150, bbox_inches='tight')
print("Saved: ramp_side_view_with_porsche.png")
plt.show()
