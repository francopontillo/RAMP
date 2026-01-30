#!/usr/bin/env python3
"""
Minimum Radial Ramp Design for Garage Access

This simulation finds the MINIMUM centerline radius that:
1. Allows the car's swept path to fit on the ramp
2. Maintains safe vertical curvature for ground clearance
3. Provides reasonable driving margins

The goal is to minimize space usage while ensuring safe passage.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Car specifications (Porsche 911, 997.1, 2008)
WHEELBASE = 2.350  # m
GROUND_CLEARANCE = 0.106  # m (106mm)
CAR_LENGTH = 4.461  # m
CAR_WIDTH = 1.808  # m
TRACK_WIDTH = 1.516  # m (front track)

# Additional car geometry
FRONT_OVERHANG = 0.85  # m (approximate for 911)
REAR_OVERHANG = CAR_LENGTH - WHEELBASE - FRONT_OVERHANG  # ~1.26m

# Ramp requirements
VERTICAL_DROP = 2.8  # m

# Safety margins
MIN_EDGE_CLEARANCE = 0.15  # m (15cm minimum clearance from edge)
COMFORT_MARGIN = 0.25  # m (25cm comfort margin)


def calculate_minimum_vertical_radius():
    """Calculate minimum radius of curvature for vertical profile."""
    R_min = WHEELBASE**2 / (8 * GROUND_CLEARANCE)
    return R_min


def calculate_swept_path(R_centerline):
    """
    Calculate the swept path of the car when following a curved path.
    """
    W = CAR_WIDTH
    L = WHEELBASE
    f_front = FRONT_OVERHANG
    f_rear = REAR_OVERHANG

    R_front_axle = R_centerline
    R_rear_axle = np.sqrt(R_front_axle**2 - L**2)
    off_tracking = R_front_axle - R_rear_axle

    # Four corners of the car:
    R_outer_front = np.sqrt((R_front_axle + W/2)**2 + f_front**2)
    R_inner_front = np.sqrt((R_front_axle - W/2)**2 + f_front**2)
    R_outer_rear = np.sqrt((R_rear_axle + W/2)**2 + f_rear**2)
    R_inner_rear = R_rear_axle - W/2  # Critical - smallest radius

    swept_width = R_outer_front - R_inner_rear

    return {
        'R_centerline': R_centerline,
        'R_front_axle': R_front_axle,
        'R_rear_axle': R_rear_axle,
        'off_tracking': off_tracking,
        'R_outer_front': R_outer_front,
        'R_inner_front': R_inner_front,
        'R_outer_rear': R_outer_rear,
        'R_inner_rear': R_inner_rear,
        'swept_width': swept_width,
    }


def calculate_vertical_curvature(R_horizontal, vertical_drop=VERTICAL_DROP):
    """Calculate the minimum vertical radius of curvature for a given horizontal radius."""
    arc_length = np.pi * R_horizontal / 2

    # Generate profile
    s = np.linspace(0, arc_length, 1000)
    L = arc_length
    H = vertical_drop
    a = 2 * H / L**3
    b = -3 * H / L**2
    z = a * s**3 + b * s**2

    # Calculate curvature
    ds = s[1] - s[0]
    dz_ds = np.gradient(z, ds)
    d2z_ds2 = np.gradient(dz_ds, ds)

    vertical_curvature = np.abs(d2z_ds2) / (1 + dz_ds**2)**1.5
    min_vertical_radius = 1 / np.max(vertical_curvature)

    max_slope = np.max(np.degrees(np.arctan(-dz_ds)))

    return {
        'arc_length': arc_length,
        'min_vertical_radius': min_vertical_radius,
        'max_slope': max_slope,
        's': s,
        'z': z,
        'slope_angles': np.degrees(np.arctan(-dz_ds)),
    }


def find_minimum_radius():
    """
    Find the minimum centerline radius that satisfies all constraints.

    Constraints:
    1. Swept path must fit on ramp with clearance margins
    2. Vertical radius must exceed minimum for ground clearance
    3. Ramp width must be reasonable (not too wide)
    """
    R_min_vertical = calculate_minimum_vertical_radius()

    print("=" * 80)
    print("FINDING MINIMUM SAFE CENTERLINE RADIUS")
    print("=" * 80)

    print(f"\nConstraints:")
    print(f"  Minimum vertical radius required: {R_min_vertical:.2f}m")
    print(f"  Minimum edge clearance: {MIN_EDGE_CLEARANCE:.2f}m")
    print(f"  Comfort margin: {COMFORT_MARGIN:.2f}m")
    print(f"  Car swept path width: ~2.2m (varies with radius)")

    print(f"\n{'R_center':>10} {'Arc Len':>10} {'Swept W':>10} {'Ramp W':>10} {'V_Radius':>10} {'Max Slope':>10} {'Status':<20}")
    print("-" * 90)

    results = []

    # Search from small to large radius
    for R in np.arange(5.0, 12.0, 0.25):
        swept = calculate_swept_path(R)
        vert = calculate_vertical_curvature(R)

        # Calculate required ramp width
        # Inner edge at: R_inner_rear - MIN_EDGE_CLEARANCE
        # Outer edge at: R_outer_front + MIN_EDGE_CLEARANCE
        inner_edge = swept['R_inner_rear'] - MIN_EDGE_CLEARANCE
        outer_edge = swept['R_outer_front'] + MIN_EDGE_CLEARANCE
        ramp_width = outer_edge - inner_edge

        # Check constraints
        vertical_ok = vert['min_vertical_radius'] >= R_min_vertical
        width_reasonable = ramp_width <= 4.0  # Max 4m wide ramp

        status = []
        if vertical_ok:
            status.append("V✓")
        else:
            status.append("V✗")
        if width_reasonable:
            status.append("W✓")
        else:
            status.append("W✗")

        is_valid = vertical_ok and width_reasonable

        results.append({
            'R_centerline': R,
            'arc_length': vert['arc_length'],
            'swept_width': swept['swept_width'],
            'ramp_width': ramp_width,
            'inner_edge': inner_edge,
            'outer_edge': outer_edge,
            'min_vertical_radius': vert['min_vertical_radius'],
            'max_slope': vert['max_slope'],
            'vertical_ok': vertical_ok,
            'width_ok': width_reasonable,
            'is_valid': is_valid,
            'swept': swept,
            'vert': vert,
        })

        status_str = " ".join(status)
        if is_valid:
            status_str += " ✓ VALID"

        print(f"{R:>10.2f} {vert['arc_length']:>10.2f} {swept['swept_width']:>10.2f} "
              f"{ramp_width:>10.2f} {vert['min_vertical_radius']:>10.2f} {vert['max_slope']:>10.1f} {status_str:<20}")

    # Find minimum valid radius
    valid_results = [r for r in results if r['is_valid']]

    if valid_results:
        min_result = valid_results[0]  # First valid (smallest)
        print(f"\n{'=' * 80}")
        print(f"MINIMUM VALID CENTERLINE RADIUS: {min_result['R_centerline']:.2f}m")
        print(f"{'=' * 80}")
        return min_result
    else:
        print("\nNo valid solution found!")
        return None


def create_minimum_ramp_design(result):
    """Create the minimum ramp design based on the analysis."""

    R = result['R_centerline']
    inner_edge = result['inner_edge']
    outer_edge = result['outer_edge']
    ramp_width = result['ramp_width']
    arc_length = result['arc_length']
    swept = result['swept']
    vert = result['vert']

    print(f"\n{'=' * 80}")
    print("MINIMUM RADIAL RAMP DESIGN")
    print(f"{'=' * 80}")

    print(f"\nGEOMETRY:")
    print(f"  Centerline radius:     {R:.2f}m")
    print(f"  Inner edge radius:     {inner_edge:.2f}m")
    print(f"  Outer edge radius:     {outer_edge:.2f}m")
    print(f"  Ramp width:            {ramp_width:.2f}m")
    print(f"  Arc length:            {arc_length:.2f}m")
    print(f"  Vertical drop:         {VERTICAL_DROP:.2f}m")

    print(f"\nSWEPT PATH:")
    print(f"  Inner rear corner:     {swept['R_inner_rear']:.2f}m")
    print(f"  Outer front corner:    {swept['R_outer_front']:.2f}m")
    print(f"  Swept width:           {swept['swept_width']:.2f}m")
    print(f"  Off-tracking:          {swept['off_tracking']:.3f}m")

    print(f"\nVERTICAL PROFILE:")
    print(f"  Min vertical radius:   {vert['min_vertical_radius']:.2f}m")
    print(f"  Required minimum:      {calculate_minimum_vertical_radius():.2f}m")
    print(f"  Safety factor:         {vert['min_vertical_radius']/calculate_minimum_vertical_radius():.2f}x")
    print(f"  Maximum slope:         {vert['max_slope']:.1f}°")

    print(f"\nCLEARANCES:")
    print(f"  Inner edge clearance:  {swept['R_inner_rear'] - inner_edge:.2f}m")
    print(f"  Outer edge clearance:  {outer_edge - swept['R_outer_front']:.2f}m")

    return {
        'R_centerline': R,
        'inner_edge': inner_edge,
        'outer_edge': outer_edge,
        'ramp_width': ramp_width,
        'arc_length': arc_length,
        'swept': swept,
        'vert': vert,
    }


def generate_construction_data(design):
    """Generate detailed construction measurements."""

    R = design['R_centerline']
    inner_edge = design['inner_edge']
    outer_edge = design['outer_edge']
    arc_length = design['arc_length']
    vert = design['vert']

    print(f"\n{'=' * 80}")
    print("CONSTRUCTION MEASUREMENTS (every 50cm)")
    print(f"{'=' * 80}")

    print(f"\n{'Arc Dist':>10} {'Angle':>8} {'X':>8} {'Y':>8} {'Depth':>10} {'Depth':>10} {'Slope':>8}")
    print(f"{'(m)':>10} {'(deg)':>8} {'(m)':>8} {'(m)':>8} {'(m)':>10} {'(cm)':>10} {'(deg)':>8}")
    print("-" * 80)

    s = vert['s']
    z = vert['z']
    angles = vert['slope_angles']

    measurements = []

    for dist in np.arange(0, arc_length + 0.25, 0.5):
        if dist <= arc_length:
            idx = np.argmin(np.abs(s - dist))
            theta = dist / R
            theta_deg = np.degrees(theta)
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            depth = z[idx]
            slope = angles[idx]

            measurements.append({
                'arc_dist': dist,
                'angle_deg': theta_deg,
                'x': x,
                'y': y,
                'depth': depth,
                'slope': slope,
            })

            print(f"{dist:>10.1f} {theta_deg:>8.1f} {x:>8.2f} {y:>8.2f} {depth:>10.3f} {depth*100:>10.1f} {slope:>8.1f}")

    # Save to CSV
    import csv
    with open('/workspaces/RAMP/minimum_radial_measurements.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Arc_Distance_m', 'Angle_deg', 'X_m', 'Y_m', 'Depth_m', 'Depth_cm', 'Slope_deg',
                        'Inner_X_m', 'Inner_Y_m', 'Outer_X_m', 'Outer_Y_m'])

        for m in measurements:
            theta = np.radians(m['angle_deg'])
            inner_x = inner_edge * np.cos(theta)
            inner_y = inner_edge * np.sin(theta)
            outer_x = outer_edge * np.cos(theta)
            outer_y = outer_edge * np.sin(theta)

            writer.writerow([f"{m['arc_dist']:.3f}", f"{m['angle_deg']:.2f}",
                           f"{m['x']:.4f}", f"{m['y']:.4f}",
                           f"{m['depth']:.4f}", f"{m['depth']*100:.2f}", f"{m['slope']:.2f}",
                           f"{inner_x:.4f}", f"{inner_y:.4f}",
                           f"{outer_x:.4f}", f"{outer_y:.4f}"])

    print(f"\nMeasurements saved to: minimum_radial_measurements.csv")

    return measurements


def create_visualization(design):
    """Create comprehensive visualization."""

    R = design['R_centerline']
    inner_edge = design['inner_edge']
    outer_edge = design['outer_edge']
    ramp_width = design['ramp_width']
    arc_length = design['arc_length']
    swept = design['swept']
    vert = design['vert']

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.25)

    # 1. Top View with swept path
    ax1 = fig.add_subplot(gs[0, 0])

    theta = np.linspace(0, np.pi/2, 100)

    # Draw ramp surface
    inner_x = inner_edge * np.cos(theta)
    inner_y = inner_edge * np.sin(theta)
    outer_x = outer_edge * np.cos(theta)
    outer_y = outer_edge * np.sin(theta)

    ax1.fill(np.concatenate([inner_x, outer_x[::-1]]),
             np.concatenate([inner_y, outer_y[::-1]]),
             color='#d4a574', alpha=0.6, label='Ramp surface')

    ax1.plot(inner_x, inner_y, 'g-', linewidth=2, label=f'Inner edge (R={inner_edge:.2f}m)')
    ax1.plot(outer_x, outer_y, 'r-', linewidth=2, label=f'Outer edge (R={outer_edge:.2f}m)')

    # Centerline
    center_x = R * np.cos(theta)
    center_y = R * np.sin(theta)
    ax1.plot(center_x, center_y, 'b-', linewidth=2, label=f'Centerline (R={R:.2f}m)')

    # Swept path boundaries
    inner_swept_x = swept['R_inner_rear'] * np.cos(theta)
    inner_swept_y = swept['R_inner_rear'] * np.sin(theta)
    outer_swept_x = swept['R_outer_front'] * np.cos(theta)
    outer_swept_y = swept['R_outer_front'] * np.sin(theta)

    ax1.plot(inner_swept_x, inner_swept_y, 'm--', linewidth=1.5,
             label=f'Swept inner (R={swept["R_inner_rear"]:.2f}m)')
    ax1.plot(outer_swept_x, outer_swept_y, 'm--', linewidth=1.5,
             label=f'Swept outer (R={swept["R_outer_front"]:.2f}m)')

    # Draw cars at positions
    for car_theta in [0, np.pi/6, np.pi/3, np.pi/2.2]:
        draw_car_on_curve(ax1, R, car_theta, alpha=0.4)

    # Mark start and end
    ax1.plot(R, 0, 'go', markersize=12)
    ax1.plot(0, R, 'rs', markersize=12)
    ax1.text(R + 0.3, 0.3, 'START', fontsize=10, fontweight='bold')
    ax1.text(0.3, R + 0.3, 'END', fontsize=10, fontweight='bold')

    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_title('TOP VIEW (Plan)\nMinimum Radius Design with Swept Path', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, outer_edge + 1)
    ax1.set_ylim(-1, outer_edge + 1)

    # 2. Side Elevation with 50cm measurements
    ax2 = fig.add_subplot(gs[0, 1:])

    s = vert['s']
    z = vert['z']
    angles = vert['slope_angles']

    ax2.axhline(y=0, color='green', linewidth=3, label='Street level')
    ax2.axhline(y=-VERTICAL_DROP, color='blue', linewidth=3, label='Garage level')

    ax2.fill_between(s, z, -3.5, color='#d4a574', alpha=0.7)
    ax2.plot(s, z, 'k-', linewidth=3)

    # Measurements every 50cm
    for dist in np.arange(0, arc_length + 0.25, 0.5):
        if dist <= arc_length:
            idx = np.argmin(np.abs(s - dist))
            depth = z[idx]
            ax2.plot(dist, depth, 'ro', markersize=6)
            ax2.plot([dist, dist], [0, depth], 'r--', alpha=0.3, linewidth=0.5)
            if int(dist * 2) % 2 == 0:
                ax2.text(dist, depth - 0.12, f'{abs(depth)*100:.0f}', fontsize=7,
                        ha='center', va='top', rotation=45, color='darkred')
            else:
                ax2.text(dist, depth + 0.08, f'{abs(depth)*100:.0f}', fontsize=6,
                        ha='center', va='bottom', rotation=45, color='darkred', alpha=0.8)

    ax2.set_xlabel('Distance along arc (m)', fontsize=11)
    ax2.set_ylabel('Elevation (m)', fontsize=11)
    ax2.set_title('SIDE ELEVATION - Depth measurements every 50cm', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, arc_length + 0.5)
    ax2.set_ylim(-3.5, 0.5)

    # 3. Cross Section
    ax3 = fig.add_subplot(gs[1, 0])

    width_points = np.linspace(-ramp_width/2, ramp_width/2, 50)
    surface_height = 0.15
    crown = 0.02
    surface = crown * (1 - (width_points / (ramp_width/2))**2)

    ax3.fill_between(width_points, surface, surface - surface_height,
                    color='#d4a574', alpha=0.9, edgecolor='black', linewidth=2)
    ax3.fill_between(width_points, surface - surface_height, surface - surface_height - 0.1,
                    color='gray', alpha=0.5, label='Gravel base')
    ax3.fill_between(width_points, surface - surface_height - 0.1, -0.5,
                    color='#8B4513', alpha=0.3, label='Compacted soil')

    # Car width indicator
    car_half = CAR_WIDTH / 2
    ax3.plot([-car_half, car_half], [0.15, 0.15], 'orange', linewidth=4, label=f'Car width ({CAR_WIDTH}m)')

    ax3.annotate('', xy=(ramp_width/2, 0.25), xytext=(-ramp_width/2, 0.25),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax3.text(0, 0.30, f'{ramp_width:.2f}m', fontsize=11, ha='center', color='red', fontweight='bold')

    ax3.set_xlabel('Width (m)', fontsize=11)
    ax3.set_ylabel('Height (m)', fontsize=11)
    ax3.set_title('CROSS SECTION', fontsize=12, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-ramp_width/2 - 0.5, ramp_width/2 + 0.5)
    ax3.set_ylim(-0.6, 0.5)

    # 4. Specifications
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    R_min_vert = calculate_minimum_vertical_radius()

    specs_text = f"""
╔═════════════════════════════════════════════╗
║   MINIMUM RADIAL RAMP SPECIFICATIONS        ║
╠═════════════════════════════════════════════╣
║  GEOMETRY                                   ║
║  Configuration:     Quarter circle (90°)    ║
║  Centerline radius: {R:.2f} m                ║
║  Inner edge radius: {inner_edge:.2f} m              ║
║  Outer edge radius: {outer_edge:.2f} m              ║
║  Arc length:        {arc_length:.2f} m              ║
║  Ramp width:        {ramp_width:.2f} m               ║
║  Vertical drop:     {VERTICAL_DROP:.1f} m                  ║
╠═════════════════════════════════════════════╣
║  SLOPES & CURVATURE                         ║
║  Maximum slope:     {vert['max_slope']:.1f}°                  ║
║  Entry/Exit slope:  0° (level)              ║
║  Min vert. radius:  {vert['min_vertical_radius']:.2f} m               ║
║  Required minimum:  {R_min_vert:.2f} m               ║
║  Safety factor:     {vert['min_vertical_radius']/R_min_vert:.2f}x                ║
╠═════════════════════════════════════════════╣
║  SWEPT PATH                                 ║
║  Inner rear corner: {swept['R_inner_rear']:.2f} m              ║
║  Outer front corner:{swept['R_outer_front']:.2f} m              ║
║  Swept width:       {swept['swept_width']:.2f} m               ║
║  Off-tracking:      {swept['off_tracking']:.3f} m              ║
╠═════════════════════════════════════════════╣
║  CLEARANCES                                 ║
║  Inner edge:        {swept['R_inner_rear'] - inner_edge:.2f} m (min {MIN_EDGE_CLEARANCE:.2f}m)   ║
║  Outer edge:        {outer_edge - swept['R_outer_front']:.2f} m               ║
╚═════════════════════════════════════════════╝
    """

    ax4.text(0.02, 0.98, specs_text, transform=ax4.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # 5. Measurement table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    table_header = "ELEVATION PROFILE (every 50cm)\n"
    table_header += "═" * 36 + "\n"
    table_header += f"{'Arc(m)':<7} {'Depth(cm)':<10} {'Slope(°)':<8}\n"
    table_header += "─" * 36 + "\n"

    table_rows = ""
    for dist in np.arange(0, arc_length + 0.25, 0.5):
        if dist <= arc_length:
            idx = np.argmin(np.abs(s - dist))
            depth_cm = abs(z[idx]) * 100
            slope = angles[idx]
            table_rows += f"{dist:<7.1f} {depth_cm:<10.1f} {slope:<8.1f}\n"

    ax5.text(0.02, 0.98, table_header + table_rows, transform=ax5.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    # 6. Full measurement table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    full_table = "DETAILED CONSTRUCTION MEASUREMENTS (Centerline coordinates every 50cm)\n"
    full_table += "═" * 120 + "\n"
    full_table += f"{'Arc Dist':<10} {'Angle':<8} {'X':<10} {'Y':<10} {'Depth':<12} {'Depth':<10} {'Slope':<8} │ "
    full_table += f"{'Arc Dist':<10} {'Angle':<8} {'X':<10} {'Y':<10} {'Depth':<12} {'Depth':<10} {'Slope':<8}\n"
    full_table += f"{'(m)':<10} {'(deg)':<8} {'(m)':<10} {'(m)':<10} {'(m)':<12} {'(cm)':<10} {'(deg)':<8} │ "
    full_table += f"{'(m)':<10} {'(deg)':<8} {'(m)':<10} {'(m)':<10} {'(m)':<12} {'(cm)':<10} {'(deg)':<8}\n"
    full_table += "─" * 120 + "\n"

    measurements = []
    for dist in np.arange(0, arc_length + 0.25, 0.5):
        if dist <= arc_length:
            idx = np.argmin(np.abs(s - dist))
            theta_val = dist / R
            theta_deg = np.degrees(theta_val)
            x = R * np.cos(theta_val)
            y = R * np.sin(theta_val)
            depth = z[idx]
            slope = angles[idx]
            measurements.append((dist, theta_deg, x, y, depth, slope))

    mid = (len(measurements) + 1) // 2
    for i in range(mid):
        m1 = measurements[i]
        row = f"{m1[0]:<10.1f} {m1[1]:<8.1f} {m1[2]:<10.2f} {m1[3]:<10.2f} {m1[4]:<12.3f} {abs(m1[4])*100:<10.1f} {m1[5]:<8.1f} │ "
        if i + mid < len(measurements):
            m2 = measurements[i + mid]
            row += f"{m2[0]:<10.1f} {m2[1]:<8.1f} {m2[2]:<10.2f} {m2[3]:<10.2f} {m2[4]:<12.3f} {abs(m2[4])*100:<10.1f} {m2[5]:<8.1f}"
        full_table += row + "\n"

    ax6.text(0.01, 0.95, full_table, transform=ax6.transAxes,
             fontsize=8, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black'))

    plt.suptitle(f'MINIMUM RADIAL RAMP CONSTRUCTION BLUEPRINT\n'
                 f'R={R:.2f}m Centerline - Arc Length {arc_length:.2f}m - Width {ramp_width:.2f}m',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('/workspaces/RAMP/minimum_radial_blueprint.png', dpi=200, bbox_inches='tight')
    print(f"\nBlueprint saved: minimum_radial_blueprint.png")


def draw_car_on_curve(ax, R_center, theta, alpha=1.0):
    """Draw a car positioned on a curved path."""
    cx = R_center * np.cos(theta)
    cy = R_center * np.sin(theta)
    car_angle = theta + np.pi/2

    corners = np.array([
        [-CAR_LENGTH/2, -CAR_WIDTH/2],
        [CAR_LENGTH/2, -CAR_WIDTH/2],
        [CAR_LENGTH/2, CAR_WIDTH/2],
        [-CAR_LENGTH/2, CAR_WIDTH/2],
        [-CAR_LENGTH/2, -CAR_WIDTH/2]
    ])

    rot = np.array([[np.cos(car_angle), -np.sin(car_angle)],
                    [np.sin(car_angle), np.cos(car_angle)]])
    corners_rot = corners @ rot.T
    corners_rot[:, 0] += cx
    corners_rot[:, 1] += cy

    ax.fill(corners_rot[:-1, 0], corners_rot[:-1, 1],
            color='orange', alpha=alpha*0.5, edgecolor='darkorange', linewidth=1)


def create_comparison_chart():
    """Create a chart comparing different radius options."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    radii = np.arange(5.0, 12.0, 0.25)

    arc_lengths = []
    swept_widths = []
    ramp_widths = []
    vert_radii = []
    max_slopes = []

    R_min_vert = calculate_minimum_vertical_radius()

    for R in radii:
        swept = calculate_swept_path(R)
        vert = calculate_vertical_curvature(R)

        inner_edge = swept['R_inner_rear'] - MIN_EDGE_CLEARANCE
        outer_edge = swept['R_outer_front'] + MIN_EDGE_CLEARANCE

        arc_lengths.append(vert['arc_length'])
        swept_widths.append(swept['swept_width'])
        ramp_widths.append(outer_edge - inner_edge)
        vert_radii.append(vert['min_vertical_radius'])
        max_slopes.append(vert['max_slope'])

    # Plot 1: Arc length vs radius
    ax1 = axes[0, 0]
    ax1.plot(radii, arc_lengths, 'b-', linewidth=2)
    ax1.axhline(y=12, color='gray', linestyle='--', label='Straight ramp (12m)')
    ax1.set_xlabel('Centerline Radius (m)')
    ax1.set_ylabel('Arc Length (m)')
    ax1.set_title('Arc Length vs Radius')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Ramp width vs radius
    ax2 = axes[0, 1]
    ax2.plot(radii, ramp_widths, 'g-', linewidth=2, label='Required ramp width')
    ax2.plot(radii, swept_widths, 'm--', linewidth=2, label='Car swept width')
    ax2.axhline(y=4.0, color='r', linestyle='--', label='Max practical width (4m)')
    ax2.set_xlabel('Centerline Radius (m)')
    ax2.set_ylabel('Width (m)')
    ax2.set_title('Width Requirements vs Radius')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Vertical radius vs centerline radius
    ax3 = axes[1, 0]
    ax3.plot(radii, vert_radii, 'purple', linewidth=2)
    ax3.axhline(y=R_min_vert, color='r', linestyle='--', label=f'Min required ({R_min_vert:.2f}m)')
    ax3.fill_between(radii, R_min_vert, vert_radii,
                     where=np.array(vert_radii) >= R_min_vert,
                     alpha=0.3, color='green', label='Safe zone')
    ax3.set_xlabel('Centerline Radius (m)')
    ax3.set_ylabel('Min Vertical Radius (m)')
    ax3.set_title('Vertical Curvature vs Radius')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Max slope vs radius
    ax4 = axes[1, 1]
    ax4.plot(radii, max_slopes, 'orange', linewidth=2)
    ax4.set_xlabel('Centerline Radius (m)')
    ax4.set_ylabel('Maximum Slope (°)')
    ax4.set_title('Maximum Slope vs Radius')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Radial Ramp Design Trade-offs\nHow centerline radius affects key parameters',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspaces/RAMP/radius_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison chart saved: radius_comparison.png")


def main():
    print("\n" + "=" * 80)
    print("MINIMUM RADIAL RAMP DESIGN")
    print("Finding the smallest possible centerline radius")
    print("=" * 80)

    # Find minimum valid radius
    result = find_minimum_radius()

    if result is None:
        print("Cannot find valid design!")
        return

    # Create design based on minimum radius
    design = create_minimum_ramp_design(result)

    # Generate construction data
    generate_construction_data(design)

    # Create visualization
    create_visualization(design)

    # Create comparison chart
    create_comparison_chart()

    # Final summary
    print(f"\n{'=' * 80}")
    print("SUMMARY - MINIMUM RADIAL RAMP")
    print(f"{'=' * 80}")
    print(f"""
    MINIMUM CENTERLINE RADIUS: {design['R_centerline']:.2f}m

    This is the smallest radius that:
    ✓ Allows the Porsche 911 swept path to fit with {MIN_EDGE_CLEARANCE}m clearance
    ✓ Maintains safe vertical curvature (>{calculate_minimum_vertical_radius():.2f}m)
    ✓ Keeps ramp width under 4m

    DESIGN SPECIFICATIONS:
    ─────────────────────────────────────────
    Centerline radius:   {design['R_centerline']:.2f}m
    Inner edge radius:   {design['inner_edge']:.2f}m
    Outer edge radius:   {design['outer_edge']:.2f}m
    Ramp width:          {design['ramp_width']:.2f}m
    Arc length:          {design['arc_length']:.2f}m
    Maximum slope:       {design['vert']['max_slope']:.1f}°

    COMPARISON TO R=9m DESIGN:
    ─────────────────────────────────────────
    Centerline: {design['R_centerline']:.2f}m vs 9.00m ({design['R_centerline'] - 9:.2f}m)
    Arc length: {design['arc_length']:.2f}m vs 14.14m ({design['arc_length'] - 14.14:.2f}m)
    Max slope:  {design['vert']['max_slope']:.1f}° vs 16.5°

    OUTPUT FILES:
    ─────────────────────────────────────────
    • minimum_radial_blueprint.png      - Construction blueprint
    • minimum_radial_measurements.csv   - Detailed measurements
    • radius_comparison.png             - Trade-off analysis
    """)

    plt.show()


if __name__ == '__main__':
    main()
