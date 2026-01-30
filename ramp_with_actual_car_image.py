#!/usr/bin/env python3
"""
Ramp visualization using the actual Art-Line Porsche 911 997.1 Turbo image.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D
import matplotlib.transforms as mtransforms

# =============================================================================
# CAR SPECIFICATIONS - Porsche 911 997.1 Turbo (2008)
# =============================================================================
CAR_LENGTH = 4.450       # m
CAR_WIDTH = 1.852        # m
CAR_HEIGHT = 1.300       # m
WHEELBASE = 2.350        # m
GROUND_CLEARANCE = 0.106 # m (106mm)

FRONT_OVERHANG = 0.840   # m
REAR_OVERHANG = CAR_LENGTH - WHEELBASE - FRONT_OVERHANG  # 1.26m

# Wheel dimensions
WHEEL_DIAMETER_FRONT = 0.647  # m
WHEEL_DIAMETER_REAR = 0.656   # m

# =============================================================================
# RAMP SPECIFICATIONS
# =============================================================================
VERTICAL_DROP = 3.5
R_CENTERLINE = 7.50
ARC_LENGTH = np.pi * R_CENTERLINE / 2  # 11.78m


def get_ramp_profile(num_points=500):
    """Generate the cubic spline ramp profile."""
    s = np.linspace(0, ARC_LENGTH, num_points)
    L = ARC_LENGTH
    H = VERTICAL_DROP

    a = 2 * H / L**3
    b = -3 * H / L**2
    z = a * s**3 + b * s**2

    dz_ds = 3 * a * s**2 + 2 * b * s
    slope_angles = np.arctan(dz_ds)

    return s, z, slope_angles


def get_car_position_on_ramp(rear_axle_arc_pos, s_array, z_array):
    """Calculate correct car position with both wheels on the curved ramp surface."""
    rear_idx = np.argmin(np.abs(s_array - rear_axle_arc_pos))
    rear_x = s_array[rear_idx]
    rear_y = z_array[rear_idx]

    front_arc_pos = rear_axle_arc_pos + WHEELBASE
    if front_arc_pos > s_array[-1]:
        front_arc_pos = s_array[-1]

    front_idx = np.argmin(np.abs(s_array - front_arc_pos))
    front_x = s_array[front_idx]
    front_y = z_array[front_idx]

    dx = front_x - rear_x
    dy = front_y - rear_y
    car_angle = np.arctan2(dy, dx)

    return rear_x, rear_y, front_x, front_y, car_angle


def place_car_image(ax, img, rear_axle_x, rear_axle_y, angle_rad):
    """
    Place the car image at correct position with wheels on ramp surface.

    CORRECT SCALING based on wheelbase measurement in the image:
    - Rear wheel center at ~27% from left
    - Front wheel center at ~79% from left
    - Wheelbase in image = 52% of image width
    - Real wheelbase = 2.350m
    - Therefore: image_width = 2.350 / 0.52 = 4.52m
    """

    img_height, img_width = img.shape[0], img.shape[1]

    # Measured wheel positions in image (fractions from left)
    rear_wheel_x_frac = 0.27   # Rear wheel at 27% from left
    front_wheel_x_frac = 0.79  # Front wheel at 79% from left

    # Calculate correct scale based on wheelbase
    wheelbase_frac = front_wheel_x_frac - rear_wheel_x_frac  # 0.52
    image_width_meters = WHEELBASE / wheelbase_frac  # 2.350 / 0.52 = 4.52m

    # Height maintaining aspect ratio
    image_height_meters = image_width_meters * (img_height / img_width)

    # Position image so rear wheel contact point is at origin (0,0)
    left = -rear_wheel_x_frac * image_width_meters
    right = left + image_width_meters
    bottom = 0  # Wheels touch ground at y=0
    top = image_height_meters

    extent = [left, right, bottom, top]

    # Transform: rotate around (0,0), then translate to ramp position
    transform = (
        Affine2D()
        .rotate(angle_rad)
        .translate(rear_axle_x, rear_axle_y)
        + ax.transData
    )

    im = ax.imshow(img, extent=extent, transform=transform,
                   aspect='auto', zorder=5, interpolation='bilinear',
                   origin='upper')

    return im


def create_visualization():
    """Create the ramp visualization with actual car image."""

    # Load the car image
    car_img_path = '/workspaces/RAMP/Art-Line-2008-997.1-Porsche-turbo.png'
    car_img = mpimg.imread(car_img_path)

    print(f"Loaded car image: {car_img.shape}")

    fig, axes = plt.subplots(2, 1, figsize=(20, 14))

    s, z, slopes = get_ramp_profile(1000)

    # =================================================================
    # TOP PLOT: Full ramp with car at multiple positions
    # =================================================================
    ax1 = axes[0]

    # Ground reference lines
    ax1.axhline(y=0, color='#228B22', linewidth=3, label='Street level')
    ax1.axhline(y=-VERTICAL_DROP, color='#4169E1', linewidth=3, label='Garage level (-3.5m)')

    # Labels
    ax1.fill_between([-2, 0], -0.2, 0.2, color='#90EE90', alpha=0.3)
    ax1.fill_between([ARC_LENGTH, ARC_LENGTH + 2], -VERTICAL_DROP - 0.2, -VERTICAL_DROP + 0.2,
                     color='#87CEEB', alpha=0.3)
    ax1.text(-1, 0.12, 'STREET', fontsize=10, ha='center', fontweight='bold', color='darkgreen')
    ax1.text(ARC_LENGTH + 1, -VERTICAL_DROP + 0.12, 'GARAGE', fontsize=10, ha='center',
             fontweight='bold', color='darkblue')

    # Ramp surface
    ax1.fill_between(s, z, z - 0.12, color='#C4A484', alpha=0.7, edgecolor='#8B7355', linewidth=1.5)
    ax1.plot(s, z, color='#5C4033', linewidth=2.5)

    # Place cars at multiple positions
    car_positions = [0.2, 2.5, 5.0, 7.5, 9.5]

    for pos in car_positions:
        rear_x, rear_y, front_x, front_y, car_angle = get_car_position_on_ramp(pos, s, z)
        place_car_image(ax1, car_img, rear_x, rear_y, car_angle)

    # Set axis limits as requested: x from -2 to 14
    ax1.set_xlim(-2, 14)
    ax1.set_ylim(-5.5, 2.5)
    ax1.set_xlabel('Distance along ramp path (m)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.set_title('SIDE VIEW: Porsche 911 997.1 Turbo (2008) Descending Radial Ramp\n'
                  f'3.5m Drop | R={R_CENTERLINE}m Centerline | Arc={ARC_LENGTH:.2f}m',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')

    # =================================================================
    # BOTTOM PLOT: Zoomed view at critical section
    # =================================================================
    ax2 = axes[1]

    # Ramp surface
    ax2.fill_between(s, z, z - 0.12, color='#C4A484', alpha=0.7, edgecolor='#8B7355', linewidth=1.5)
    ax2.plot(s, z, color='#5C4033', linewidth=2.5)
    ax2.axhline(y=0, color='#228B22', linewidth=2, alpha=0.5)
    ax2.axhline(y=-VERTICAL_DROP, color='#4169E1', linewidth=2, alpha=0.5)

    # Place car at the steepest point (around 5.9m)
    critical_pos = 4.5
    rear_x, rear_y, front_x, front_y, car_angle = get_car_position_on_ramp(critical_pos, s, z)
    place_car_image(ax2, car_img, rear_x, rear_y, car_angle)

    # Annotation
    ax2.annotate(f'Car angle: {np.degrees(car_angle):.1f}°',
                xy=(rear_x, rear_y),
                xytext=(rear_x + 2, rear_y + 1),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.95))

    ax2.set_xlim(critical_pos - 3, critical_pos + WHEELBASE + 4)
    ax2.set_ylim(rear_y - 2, rear_y + 2.5)
    ax2.set_xlabel('Distance along ramp path (m)', fontsize=12)
    ax2.set_ylabel('Elevation (m)', fontsize=12)
    ax2.set_title('ZOOMED VIEW: Car at Maximum Slope Point',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/workspaces/RAMP/ramp_side_view_2d.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("\nSaved: ramp_side_view_2d.png")

    return fig


def main():
    print("="*70)
    print("RAMP VISUALIZATION WITH ACTUAL PORSCHE 911 IMAGE")
    print("="*70)

    create_visualization()

    print("\nDone!")


if __name__ == '__main__':
    main()
