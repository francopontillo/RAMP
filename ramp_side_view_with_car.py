#!/usr/bin/env python3
"""
Side View Visualization of Ramp with Porsche 911 997.1 Turbo (2008)
Accurate line-art based on reference image.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Wheel dimensions - 997 Turbo 19" wheels
WHEEL_DIAMETER_FRONT = 0.647  # m
WHEEL_DIAMETER_REAR = 0.656   # m

# =============================================================================
# RAMP SPECIFICATIONS (3.5m drop, minimum radius)
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
    """
    Calculate correct car position with both wheels on the curved ramp surface.
    """
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


def get_porsche_997_turbo_profile():
    """
    Porsche 911 997.1 Turbo side profile - clean line art.
    Based on reference: Art-Line-2008-997.1-Porsche-turbo.png

    Coordinates relative to REAR AXLE at (0, 0) on ground.
    Using actual proportions from the reference image.
    """

    GC = GROUND_CLEARANCE  # 0.106m
    WB = WHEELBASE         # 2.350m
    FO = FRONT_OVERHANG    # 0.840m
    RO = REAR_OVERHANG     # 1.260m

    # Wheel radii
    R_front = WHEEL_DIAMETER_FRONT / 2  # 0.324m
    R_rear = WHEEL_DIAMETER_REAR / 2    # 0.328m

    # Key vertical positions (from reference image proportions)
    # Total height 1.30m, ground clearance 0.106m

    wheel_center_height = R_rear  # Wheel centers

    # Front bumper is very low - almost at ground clearance level
    front_bumper_bottom = GC + 0.02      # ~0.126m - very low
    front_bumper_top = GC + 0.22         # ~0.326m

    # Hood is flat and low
    hood_height = GC + 0.38              # ~0.486m - flat hood line

    # Fender peaks slightly above hood
    front_fender = GC + 0.44             # ~0.546m

    # Windshield and roof
    windshield_base = GC + 0.54          # ~0.646m
    roof_front = GC + 1.12               # ~1.226m (top of windshield)
    roof_peak = GC + 1.19                # ~1.296m (roof peak)

    # Rear section
    rear_window_top = GC + 1.14          # ~1.246m
    rear_window_base = GC + 0.70         # ~0.806m
    engine_cover = GC + 0.64             # ~0.746m
    spoiler_top = GC + 0.76              # ~0.866m (Turbo spoiler)
    rear_deck = GC + 0.54                # ~0.646m
    rear_bumper_top = GC + 0.38          # ~0.486m
    rear_bumper_bottom = GC + 0.06       # ~0.166m

    # Sill height
    sill = GC + 0.12                     # ~0.226m

    # =================================================================
    # MAIN BODY OUTLINE - Single continuous smooth line
    # =================================================================
    body = [
        # FRONT BUMPER (low, aggressive)
        (WB + FO, front_bumper_bottom),
        (WB + FO + 0.01, front_bumper_top - 0.05),
        (WB + FO, front_bumper_top),

        # HOOD - characteristic flat 911 hood
        (WB + FO - 0.08, hood_height - 0.04),
        (WB + FO - 0.20, hood_height),
        (WB + 0.30, hood_height + 0.02),      # Hood is nearly flat
        (WB + 0.05, hood_height + 0.04),
        (WB - 0.10, front_fender),            # Front fender rise

        # A-PILLAR / WINDSHIELD
        (WB - 0.22, windshield_base),
        (WB - 0.60, roof_front),              # Steep windshield rake

        # ROOF - smooth flowing curve
        (WB - 0.85, roof_peak - 0.02),
        (WB - 1.15, roof_peak),               # Roof peak
        (-0.05, roof_peak - 0.01),
        (-0.30, rear_window_top),

        # REAR WINDOW - iconic 911 slope
        (-0.55, rear_window_base + 0.12),
        (-0.80, rear_window_base),

        # ENGINE COVER
        (-0.90, engine_cover + 0.04),
        (-RO + 0.45, engine_cover),

        # TURBO REAR SPOILER
        (-RO + 0.40, spoiler_top - 0.04),
        (-RO + 0.25, spoiler_top),
        (-RO + 0.12, spoiler_top - 0.02),
        (-RO + 0.10, rear_deck + 0.06),

        # REAR BUMPER
        (-RO + 0.06, rear_bumper_top),
        (-RO + 0.02, rear_bumper_top - 0.10),
        (-RO, rear_bumper_bottom + 0.04),
        (-RO, rear_bumper_bottom),
    ]

    # =================================================================
    # LOWER BODY LINE
    # =================================================================
    # Rear lower
    lower_rear = [
        (-RO, rear_bumper_bottom),
        (-RO + 0.12, sill),
        (-0.48, sill),
    ]

    # Sill between wheels
    sill_line = [
        (-0.42, sill),
        (WB - 0.42, sill),
    ]

    # Front lower
    lower_front = [
        (WB + 0.35, sill),
        (WB + FO - 0.10, front_bumper_bottom + 0.02),
        (WB + FO, front_bumper_bottom),
    ]

    # =================================================================
    # WHEEL ARCHES - smooth curves
    # =================================================================
    def wheel_arch(cx, r_wheel, arch_clearance=0.035):
        """Create wheel arch points."""
        r = r_wheel + arch_clearance
        angles = np.linspace(np.pi * 0.08, np.pi * 0.92, 20)
        return [(cx + r * np.cos(a), r * np.sin(a)) for a in angles]

    rear_arch = wheel_arch(0, R_rear)
    front_arch = wheel_arch(WB, R_front)

    # =================================================================
    # WHEELS - tire, rim, spokes
    # =================================================================
    def create_wheel(cx, r_wheel):
        """Create detailed wheel."""
        angles = np.linspace(0, 2*np.pi, 50)
        tire = [(cx + r_wheel * np.cos(a), r_wheel * np.sin(a)) for a in angles]

        r_rim = r_wheel * 0.58
        rim = [(cx + r_rim * np.cos(a), r_rim * np.sin(a)) for a in angles]

        r_hub = r_wheel * 0.18
        hub = [(cx + r_hub * np.cos(a), r_hub * np.sin(a)) for a in angles]

        # 5 double spokes (Turbo twist style)
        spokes = []
        for i in range(5):
            a1 = i * 2 * np.pi / 5 - np.pi/2
            a2 = a1 + 0.15
            spokes.append([(cx + r_hub * np.cos(a1), r_hub * np.sin(a1)),
                          (cx + r_rim * 0.92 * np.cos(a1 + 0.08), r_rim * 0.92 * np.sin(a1 + 0.08))])
            spokes.append([(cx + r_hub * np.cos(a2), r_hub * np.sin(a2)),
                          (cx + r_rim * 0.92 * np.cos(a2 - 0.08), r_rim * 0.92 * np.sin(a2 - 0.08))])

        return {'tire': tire, 'rim': rim, 'hub': hub, 'spokes': spokes}

    rear_wheel = create_wheel(0, R_rear)
    front_wheel = create_wheel(WB, R_front)

    # =================================================================
    # DETAILS
    # =================================================================
    # Side window
    window = [
        (WB - 0.58, roof_front - 0.03),
        (WB - 0.83, roof_peak - 0.04),
        (-0.07, roof_peak - 0.04),
        (-0.32, rear_window_top - 0.03),
        (-0.53, rear_window_base + 0.14),
    ]

    # Side air intake (Turbo)
    intake = [
        (-0.52, GC + 0.32),
        (-0.32, GC + 0.34),
        (-0.30, GC + 0.44),
        (-0.50, GC + 0.42),
    ]

    # Door line
    door = [
        (0.25, sill + 0.01),
        (0.27, GC + 0.62),
    ]

    # Headlight
    hl_cx = WB + FO - 0.22
    hl_cy = hood_height - 0.06
    headlight = [(hl_cx + 0.09 * np.cos(a), hl_cy + 0.05 * np.sin(a))
                 for a in np.linspace(0, 2*np.pi, 20)]

    # Taillight
    taillight = [
        (-RO + 0.04, GC + 0.36),
        (-RO + 0.22, GC + 0.38),
    ]

    # Mirror
    mirror = [
        (WB - 0.30, roof_front - 0.10),
        (WB - 0.22, roof_front - 0.08),
        (WB - 0.22, roof_front - 0.18),
        (WB - 0.30, roof_front - 0.16),
    ]

    return {
        'body': body,
        'lower_rear': lower_rear,
        'sill': sill_line,
        'lower_front': lower_front,
        'rear_arch': rear_arch,
        'front_arch': front_arch,
        'rear_wheel': rear_wheel,
        'front_wheel': front_wheel,
        'window': window,
        'intake': intake,
        'door': door,
        'headlight': headlight,
        'taillight': taillight,
        'mirror': mirror,
    }


def transform_point(px, py, x_offset, y_offset, angle):
    """Rotate and translate a single point."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rx = px * cos_a - py * sin_a
    ry = px * sin_a + py * cos_a
    return rx + x_offset, ry + y_offset


def transform_points(points, x_offset, y_offset, angle):
    """Rotate and translate a list of points."""
    return [transform_point(px, py, x_offset, y_offset, angle) for px, py in points]


def draw_car(ax, rear_x, rear_y, angle, color='#000000', lw=1.2, alpha=1.0):
    """Draw the Porsche 911 997.1 Turbo."""

    profile = get_porsche_997_turbo_profile()

    # Main body outline
    body = transform_points(profile['body'], rear_x, rear_y, angle)
    bx, by = zip(*body)
    ax.plot(bx, by, color=color, linewidth=lw*1.3, alpha=alpha, solid_capstyle='round', solid_joinstyle='round')

    # Lower body
    for section in ['lower_rear', 'sill', 'lower_front']:
        pts = transform_points(profile[section], rear_x, rear_y, angle)
        px, py = zip(*pts)
        ax.plot(px, py, color=color, linewidth=lw, alpha=alpha)

    # Wheel arches
    for arch in ['rear_arch', 'front_arch']:
        pts = transform_points(profile[arch], rear_x, rear_y, angle)
        px, py = zip(*pts)
        ax.plot(px, py, color=color, linewidth=lw, alpha=alpha)

    # Wheels
    for wheel_name in ['rear_wheel', 'front_wheel']:
        wheel = profile[wheel_name]

        # Tire (thick line)
        tire = transform_points(wheel['tire'], rear_x, rear_y, angle)
        tx, ty = zip(*tire)
        ax.plot(tx, ty, color=color, linewidth=lw*1.4, alpha=alpha)

        # Rim
        rim = transform_points(wheel['rim'], rear_x, rear_y, angle)
        rx, ry = zip(*rim)
        ax.plot(rx, ry, color=color, linewidth=lw*0.7, alpha=alpha*0.8)

        # Hub
        hub = transform_points(wheel['hub'], rear_x, rear_y, angle)
        hx, hy = zip(*hub)
        ax.plot(hx, hy, color=color, linewidth=lw*0.5, alpha=alpha*0.7)

        # Spokes
        for spoke in wheel['spokes']:
            sp = transform_points(spoke, rear_x, rear_y, angle)
            ax.plot([sp[0][0], sp[1][0]], [sp[0][1], sp[1][1]],
                   color=color, linewidth=lw*0.4, alpha=alpha*0.6)

    # Window
    win = transform_points(profile['window'], rear_x, rear_y, angle)
    wx, wy = zip(*win)
    ax.plot(wx, wy, color='#444444', linewidth=lw*0.7, alpha=alpha*0.8)

    # Details
    for detail in ['intake', 'door', 'headlight', 'taillight', 'mirror']:
        pts = transform_points(profile[detail], rear_x, rear_y, angle)
        if len(pts) > 1:
            px, py = zip(*pts)
            ax.plot(px, py, color=color, linewidth=lw*0.5, alpha=alpha*0.6)


def get_lowest_point_clearance(rear_x, rear_y, angle, s_array, z_array):
    """
    Calculate the minimum clearance between the car's lowest points and the ramp.
    Returns the minimum clearance and where it occurs.
    """
    profile = get_porsche_997_turbo_profile()

    # Get all body points that could potentially touch the ground
    # Focus on the bottom edge of the car
    critical_points = []

    # Add front bumper bottom
    critical_points.extend(profile['body'][:3])  # Front bumper area

    # Add lower body line
    critical_points.extend(profile['lower_rear'])
    critical_points.extend(profile['sill'])
    critical_points.extend(profile['lower_front'])

    # Add rear bumper bottom
    critical_points.extend(profile['body'][-4:])

    min_clearance = float('inf')
    min_point = None
    min_location = None

    for px, py in critical_points:
        # Transform point to world coordinates
        wx, wy = transform_point(px, py, rear_x, rear_y, angle)

        # Find the ramp height at this x position
        if s_array[0] <= wx <= s_array[-1]:
            idx = np.argmin(np.abs(s_array - wx))
            ramp_z = z_array[idx]
            clearance = wy - ramp_z

            if clearance < min_clearance:
                min_clearance = clearance
                min_point = (wx, wy)
                min_location = (px, py)

    return min_clearance, min_point, min_location


def create_visualization():
    """Create the complete visualization."""

    fig, axes = plt.subplots(2, 1, figsize=(20, 14))

    s, z, slopes = get_ramp_profile(1000)

    # =================================================================
    # TOP PLOT: Full ramp with multiple car positions
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

    # Draw cars at multiple positions
    car_positions = [0.2, 2.2, 4.5, 6.8, 9.2]

    print("\n" + "="*70)
    print("GROUND CLEARANCE ANALYSIS")
    print("="*70)

    for i, pos in enumerate(car_positions):
        rear_x, rear_y, front_x, front_y, car_angle = get_car_position_on_ramp(pos, s, z)

        # Check clearance
        clearance, point, location = get_lowest_point_clearance(rear_x, rear_y, car_angle, s, z)

        print(f"\nCar position {i+1}: rear axle at {pos:.1f}m")
        print(f"  Car angle: {np.degrees(car_angle):.1f}°")
        print(f"  Minimum clearance: {clearance*1000:.1f}mm")
        if point:
            print(f"  At world position: ({point[0]:.2f}m, {point[1]:.2f}m)")

        # Draw car
        draw_car(ax1, rear_x, rear_y, car_angle, color='#1a1a1a', lw=1.2, alpha=0.9)

        # Mark if clearance is critical
        if clearance < 0.05 and point:  # Less than 50mm
            ax1.plot(point[0], point[1], 'ro', markersize=6)
            ax1.annotate(f'{clearance*1000:.0f}mm', xy=point,
                        xytext=(point[0]+0.3, point[1]+0.3),
                        fontsize=8, color='red')

    ax1.set_xlim(-2.5, ARC_LENGTH + 2.5)
    ax1.set_ylim(-5.5, 2.5)
    ax1.set_xlabel('Distance along ramp path (m)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.set_title('SIDE VIEW: Porsche 911 997.1 Turbo (2008) Descending Radial Ramp\n'
                  f'3.5m Drop | R={R_CENTERLINE}m Centerline | Arc={ARC_LENGTH:.2f}m | Max Slope=24.0°',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')

    # =================================================================
    # BOTTOM PLOT: Critical point analysis
    # =================================================================
    ax2 = axes[1]

    # Find the position with minimum clearance
    min_clearance_overall = float('inf')
    critical_position = None

    for pos in np.arange(0.2, ARC_LENGTH - WHEELBASE - 0.5, 0.2):
        rear_x, rear_y, front_x, front_y, car_angle = get_car_position_on_ramp(pos, s, z)
        clearance, point, _ = get_lowest_point_clearance(rear_x, rear_y, car_angle, s, z)

        if clearance < min_clearance_overall:
            min_clearance_overall = clearance
            critical_position = pos

    print(f"\n{'='*70}")
    print(f"CRITICAL POINT: Minimum clearance = {min_clearance_overall*1000:.1f}mm")
    print(f"At rear axle position: {critical_position:.2f}m")
    print(f"{'='*70}")

    # Draw ramp
    ax2.fill_between(s, z, z - 0.12, color='#C4A484', alpha=0.7, edgecolor='#8B7355', linewidth=1.5)
    ax2.plot(s, z, color='#5C4033', linewidth=2.5)
    ax2.axhline(y=0, color='#228B22', linewidth=2, alpha=0.5)
    ax2.axhline(y=-VERTICAL_DROP, color='#4169E1', linewidth=2, alpha=0.5)

    # Draw car at critical position
    rear_x, rear_y, front_x, front_y, car_angle = get_car_position_on_ramp(critical_position, s, z)
    clearance, crit_point, _ = get_lowest_point_clearance(rear_x, rear_y, car_angle, s, z)

    draw_car(ax2, rear_x, rear_y, car_angle, color='#1a1a1a', lw=1.6, alpha=1.0)

    # Mark critical clearance point
    if crit_point:
        ax2.plot(crit_point[0], crit_point[1], 'ro', markersize=10, zorder=10)

        # Draw clearance line
        idx = np.argmin(np.abs(s - crit_point[0]))
        ramp_z = z[idx]
        ax2.plot([crit_point[0], crit_point[0]], [ramp_z, crit_point[1]],
                'r-', linewidth=2, zorder=10)

        ax2.annotate(f'Min clearance: {clearance*1000:.1f}mm\nat {crit_point[0]:.1f}m',
                    xy=crit_point,
                    xytext=(crit_point[0] + 1.5, crit_point[1] + 0.5),
                    fontsize=11, fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFCC',
                             alpha=0.95, edgecolor='red'))

    # Add car angle annotation
    ax2.annotate(f'Car angle: {np.degrees(car_angle):.1f}°',
                xy=(rear_x, rear_y),
                xytext=(rear_x - 2, rear_y + 1.2),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.95))

    ax2.set_xlim(critical_position - 3, critical_position + WHEELBASE + 3)
    ax2.set_ylim(rear_y - 2, rear_y + 2.5)
    ax2.set_xlabel('Distance along ramp path (m)', fontsize=12)
    ax2.set_ylabel('Elevation (m)', fontsize=12)
    ax2.set_title('CRITICAL POINT ANALYSIS: Minimum Ground Clearance Location',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/workspaces/RAMP/ramp_side_view_2d.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("\nSaved: ramp_side_view_2d.png")

    # =================================================================
    # CLEARANCE SUMMARY
    # =================================================================
    if min_clearance_overall > 0:
        print(f"\n{'='*70}")
        print("RESULT: RAMP DESIGN IS SAFE")
        print(f"{'='*70}")
        print(f"Minimum clearance throughout descent: {min_clearance_overall*1000:.1f}mm")
        print(f"Required ground clearance used: {GROUND_CLEARANCE*1000:.0f}mm")
        if min_clearance_overall >= GROUND_CLEARANCE:
            print("Status: Car clears the ramp with full ground clearance maintained")
        else:
            print(f"Warning: Clearance reduced to {min_clearance_overall*1000:.1f}mm at critical point")
    else:
        print(f"\n{'='*70}")
        print("WARNING: CAR TOUCHES THE RAMP!")
        print(f"{'='*70}")
        print(f"Interference: {-min_clearance_overall*1000:.1f}mm")

    return fig


def main():
    print("="*70)
    print("PORSCHE 911 997.1 TURBO - RAMP CLEARANCE ANALYSIS")
    print("="*70)
    print(f"\nCar: Porsche 911 997.1 Turbo (2008)")
    print(f"  Length: {CAR_LENGTH}m")
    print(f"  Wheelbase: {WHEELBASE}m")
    print(f"  Ground clearance: {GROUND_CLEARANCE*1000:.0f}mm")
    print(f"\nRamp: Radial design for {VERTICAL_DROP}m drop")
    print(f"  Centerline radius: {R_CENTERLINE}m")
    print(f"  Arc length: {ARC_LENGTH:.2f}m")

    create_visualization()

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
