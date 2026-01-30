#!/usr/bin/env python3
"""
Radial Ramp Design for Garage Access

The ramp follows a quarter-circle path in the horizontal plane,
curving from the street entrance to the garage. This provides:
- More horizontal distance for the vertical descent
- A natural turning motion into the garage
- Better use of available space

Geometry:
- Horizontal path: Quarter circle with radius R (max 9m)
- Arc length: S = π*R/2
- Vertical profile: Cubic spline along the arc length
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

# Ramp requirements
VERTICAL_DROP = 2.8  # m
MAX_RADIUS = 9.0  # m (maximum horizontal turning radius)

# Ramp width
RAMP_WIDTH = 3.5  # m (wider for curved path)

# Additional car geometry
FRONT_OVERHANG = 0.85  # m (approximate for 911)
REAR_OVERHANG = CAR_LENGTH - WHEELBASE - FRONT_OVERHANG  # ~1.26m


def calculate_swept_path(R_centerline):
    """
    Calculate the swept path of the car when following a curved path.

    When a car turns, the rear wheels track inside the front wheels.
    The swept path is the area covered by the car body.

    Key radii:
    - R_outer_front: Outer front corner (largest radius)
    - R_inner_rear: Inner rear corner (smallest radius)

    Parameters:
    - R_centerline: Radius that the car's centerline follows

    Returns dict with all swept path dimensions.
    """
    W = CAR_WIDTH
    L = WHEELBASE
    f_front = FRONT_OVERHANG
    f_rear = REAR_OVERHANG

    # Front axle center follows approximately R_centerline
    R_front_axle = R_centerline

    # Rear axle center (tracks inside due to turning geometry)
    # For a car turning, rear axle is at: sqrt(R_front² - L²)
    R_rear_axle = np.sqrt(R_front_axle**2 - L**2)

    # Off-tracking distance (how much rear tracks inside front)
    off_tracking = R_front_axle - R_rear_axle

    # Four corners of the car:
    # Outer front corner (largest radius)
    R_outer_front = np.sqrt((R_front_axle + W/2)**2 + f_front**2)

    # Inner front corner
    R_inner_front = np.sqrt((R_front_axle - W/2)**2 + f_front**2)

    # Outer rear corner
    R_outer_rear = np.sqrt((R_rear_axle + W/2)**2 + f_rear**2)

    # Inner rear corner (smallest radius - critical!)
    R_inner_rear = R_rear_axle - W/2

    # Swept path width
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


def find_safe_driving_line(inner_edge_radius, outer_edge_radius):
    """
    Find where the car center should be so the swept path fits on the ramp.

    The constraint is: R_inner_rear >= inner_edge_radius
    """
    W = CAR_WIDTH
    L = WHEELBASE

    # We need: R_rear_axle - W/2 >= inner_edge_radius
    # So: R_rear_axle >= inner_edge_radius + W/2
    # And: sqrt(R_front² - L²) >= inner_edge_radius + W/2
    # So: R_front² >= (inner_edge_radius + W/2)² + L²
    # R_front >= sqrt((inner_edge_radius + W/2)² + L²)

    min_R_front = np.sqrt((inner_edge_radius + W/2)**2 + L**2)

    # The car center (between front and rear axles) would be slightly inside R_front
    # For simplicity, use R_front as the driving line

    swept = calculate_swept_path(min_R_front)

    return {
        'min_centerline_radius': min_R_front,
        'swept_path': swept,
        'fits_on_ramp': swept['R_outer_front'] <= outer_edge_radius,
        'inner_clearance': swept['R_inner_rear'] - inner_edge_radius,
        'outer_clearance': outer_edge_radius - swept['R_outer_front'],
    }


def calculate_minimum_vertical_radius():
    """
    Calculate minimum radius of curvature for vertical profile.
    R_min = L^2 / (8 * h)
    """
    R_min = WHEELBASE**2 / (8 * GROUND_CLEARANCE)
    return R_min


def calculate_minimum_turning_radius():
    """
    Calculate minimum turning radius for the car.
    The Porsche 911 has a turning circle of about 10.9m (curb-to-curb),
    so minimum turning radius is about 5.45m at the outer wheel.

    For comfortable driving on a ramp, we want more margin.
    """
    # Porsche 911 turning circle is ~10.9m diameter
    min_turning_radius = 10.9 / 2  # ~5.45m
    # Add comfort margin for ramp driving
    comfortable_radius = min_turning_radius * 1.3  # ~7.1m
    return min_turning_radius, comfortable_radius


def radial_ramp_geometry(R_horizontal, vertical_drop=VERTICAL_DROP, n_points=1000):
    """
    Generate the 3D geometry of a radial (quarter-circle) ramp.

    Parameters:
    - R_horizontal: Radius of the horizontal quarter-circle (m)
    - vertical_drop: Total vertical descent (m)
    - n_points: Number of points along the path

    Returns:
    - Dictionary with all geometric data
    """
    # Arc length of quarter circle
    arc_length = np.pi * R_horizontal / 2

    # Parameter along the arc (0 to arc_length)
    s = np.linspace(0, arc_length, n_points)

    # Angle along the quarter circle (0 to π/2)
    theta = s / R_horizontal

    # Horizontal coordinates (quarter circle)
    # Starting at (R, 0), ending at (0, R)
    x = R_horizontal * np.cos(theta)
    y = R_horizontal * np.sin(theta)

    # Vertical profile using cubic spline along arc length
    # z = a*s^3 + b*s^2 where z goes from 0 to -vertical_drop
    L = arc_length
    H = vertical_drop
    a = 2 * H / L**3
    b = -3 * H / L**2
    z = a * s**3 + b * s**2

    # Calculate derivatives for slope and curvature
    ds = s[1] - s[0]

    # Vertical slope (dz/ds)
    dz_ds = np.gradient(z, ds)
    d2z_ds2 = np.gradient(dz_ds, ds)

    # Slope angle
    slope_angles = np.degrees(np.arctan(-dz_ds))

    # Vertical curvature (in the plane tangent to the path)
    # κ_v = |d²z/ds²| / (1 + (dz/ds)²)^(3/2)
    vertical_curvature = np.abs(d2z_ds2) / (1 + dz_ds**2)**1.5
    vertical_radius = 1 / (vertical_curvature + 1e-10)
    vertical_radius = np.minimum(vertical_radius, 1000)  # Cap for visualization

    # Horizontal curvature (constant for a circle)
    horizontal_curvature = 1 / R_horizontal
    horizontal_radius = R_horizontal

    # Calculate path tangent vectors
    dx_ds = np.gradient(x, ds)
    dy_ds = np.gradient(y, ds)

    return {
        'R_horizontal': R_horizontal,
        'arc_length': arc_length,
        's': s,
        'theta': theta,
        'x': x,
        'y': y,
        'z': z,
        'dx_ds': dx_ds,
        'dy_ds': dy_ds,
        'dz_ds': dz_ds,
        'slope_angles': slope_angles,
        'max_slope_angle': np.max(slope_angles),
        'vertical_curvature': vertical_curvature,
        'vertical_radius': vertical_radius,
        'min_vertical_radius': np.min(vertical_radius),
        'horizontal_radius': horizontal_radius,
    }


def analyze_radial_ramp(R_horizontal):
    """Analyze a radial ramp design and check safety."""
    geom = radial_ramp_geometry(R_horizontal)

    # Safety checks
    R_min_vertical = calculate_minimum_vertical_radius()
    min_turn, comfortable_turn = calculate_minimum_turning_radius()

    # Check vertical curvature (ground clearance)
    vertical_safe = geom['min_vertical_radius'] >= R_min_vertical

    # Check horizontal curvature (turning radius)
    # The centerline radius is R_horizontal
    # Inner edge radius is R_horizontal - RAMP_WIDTH/2
    # Car needs to fit on the inner edge
    inner_radius = R_horizontal - RAMP_WIDTH / 2
    outer_radius = R_horizontal + RAMP_WIDTH / 2

    turning_safe = inner_radius >= min_turn
    turning_comfortable = inner_radius >= comfortable_turn

    return {
        **geom,
        'R_min_vertical': R_min_vertical,
        'min_turning_radius': min_turn,
        'comfortable_turning_radius': comfortable_turn,
        'inner_radius': inner_radius,
        'outer_radius': outer_radius,
        'vertical_safe': vertical_safe,
        'turning_safe': turning_safe,
        'turning_comfortable': turning_comfortable,
        'is_safe': vertical_safe and turning_safe,
        'is_comfortable': vertical_safe and turning_comfortable,
    }


def compare_straight_vs_radial():
    """Compare straight ramp vs radial ramp designs."""
    print("=" * 80)
    print("COMPARISON: STRAIGHT vs RADIAL RAMP DESIGNS")
    print("=" * 80)

    R_min_vertical = calculate_minimum_vertical_radius()
    min_turn, comfortable_turn = calculate_minimum_turning_radius()

    print(f"\nVehicle: Porsche 911 (997.1, 2008)")
    print(f"  Wheelbase: {WHEELBASE}m")
    print(f"  Ground Clearance: {GROUND_CLEARANCE*1000}mm")
    print(f"  Min turning radius: {min_turn:.2f}m")

    print(f"\nRequired vertical radius of curvature: {R_min_vertical:.2f}m")
    print(f"Vertical drop: {VERTICAL_DROP}m")

    # Straight ramp analysis
    print("\n" + "-" * 40)
    print("STRAIGHT RAMP (12m length)")
    print("-" * 40)
    straight_length = 12.0
    x = np.linspace(0, straight_length, 1000)
    a = 2 * VERTICAL_DROP / straight_length**3
    b = -3 * VERTICAL_DROP / straight_length**2
    y = a * x**3 + b * x**2

    dx = x[1] - x[0]
    dy_dx = np.gradient(y, dx)
    d2y_dx2 = np.gradient(dy_dx, dx)
    curvature = np.abs(d2y_dx2) / (1 + dy_dx**2)**1.5
    min_radius_straight = 1 / np.max(curvature)
    max_slope_straight = np.max(np.degrees(np.arctan(-dy_dx)))

    print(f"  Path length: {straight_length:.2f}m")
    print(f"  Min vertical radius: {min_radius_straight:.2f}m")
    print(f"  Max slope angle: {max_slope_straight:.1f}°")
    print(f"  Status: {'SAFE' if min_radius_straight >= R_min_vertical else 'UNSAFE'}")

    # Radial ramp analysis
    print("\n" + "-" * 40)
    print("RADIAL RAMP (Quarter circle)")
    print("-" * 40)

    for R in [7.0, 8.0, 9.0]:
        result = analyze_radial_ramp(R)
        print(f"\n  Radius = {R}m:")
        print(f"    Arc length: {result['arc_length']:.2f}m")
        print(f"    Inner edge radius: {result['inner_radius']:.2f}m")
        print(f"    Min vertical radius: {result['min_vertical_radius']:.2f}m")
        print(f"    Max slope angle: {result['max_slope_angle']:.1f}°")

        status = []
        if result['vertical_safe']:
            status.append("Vertical OK")
        else:
            status.append("Vertical UNSAFE")
        if result['turning_comfortable']:
            status.append("Turning COMFORTABLE")
        elif result['turning_safe']:
            status.append("Turning OK")
        else:
            status.append("Turning TIGHT")

        print(f"    Status: {', '.join(status)}")

    return result


def create_visualization():
    """Create comprehensive visualization of the radial ramp."""

    print("\n" + "=" * 80)
    print("RADIAL RAMP DESIGN - QUARTER CIRCLE PATH")
    print("=" * 80)

    R_horizontal = MAX_RADIUS  # 9m radius
    result = analyze_radial_ramp(R_horizontal)

    print(f"\nDesign Parameters:")
    print(f"  Horizontal radius: {R_horizontal}m")
    print(f"  Arc length (path length): {result['arc_length']:.2f}m")
    print(f"  Vertical drop: {VERTICAL_DROP}m")
    print(f"  Ramp width: {RAMP_WIDTH}m")

    print(f"\nGeometry Analysis:")
    print(f"  Inner edge radius: {result['inner_radius']:.2f}m")
    print(f"  Outer edge radius: {result['outer_radius']:.2f}m")
    print(f"  Min vertical radius: {result['min_vertical_radius']:.2f}m (need {result['R_min_vertical']:.2f}m)")
    print(f"  Max slope angle: {result['max_slope_angle']:.1f}°")

    print(f"\nSafety Status:")
    print(f"  Vertical clearance: {'SAFE' if result['vertical_safe'] else 'UNSAFE'}")
    print(f"  Turning radius: {'COMFORTABLE' if result['turning_comfortable'] else 'OK' if result['turning_safe'] else 'TIGHT'}")
    print(f"  Overall: {'SAFE AND COMFORTABLE' if result['is_comfortable'] else 'SAFE' if result['is_safe'] else 'NEEDS REVIEW'}")

    # Create figure
    fig = plt.figure(figsize=(20, 16))

    # 1. 3D View of the ramp
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    # Plot centerline
    ax1.plot(result['x'], result['y'], result['z'], 'b-', linewidth=3, label='Centerline')

    # Plot inner and outer edges
    inner_r = result['inner_radius']
    outer_r = result['outer_radius']
    theta = result['theta']
    z = result['z']

    x_inner = inner_r * np.cos(theta)
    y_inner = inner_r * np.sin(theta)
    x_outer = outer_r * np.cos(theta)
    y_outer = outer_r * np.sin(theta)

    ax1.plot(x_inner, y_inner, z, 'g-', linewidth=2, alpha=0.7, label='Inner edge')
    ax1.plot(x_outer, y_outer, z, 'r-', linewidth=2, alpha=0.7, label='Outer edge')

    # Plot surface
    for i in range(0, len(theta), 50):
        ax1.plot([x_inner[i], x_outer[i]], [y_inner[i], y_outer[i]],
                [z[i], z[i]], 'gray', alpha=0.3)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D View of Radial Ramp', fontweight='bold')
    ax1.legend()

    # 2. Top view (plan)
    ax2 = fig.add_subplot(2, 3, 2)

    # Draw the ramp surface
    theta_fill = np.linspace(0, np.pi/2, 100)
    inner_x = inner_r * np.cos(theta_fill)
    inner_y = inner_r * np.sin(theta_fill)
    outer_x = outer_r * np.cos(theta_fill)
    outer_y = outer_r * np.sin(theta_fill)

    ax2.fill(np.concatenate([inner_x, outer_x[::-1]]),
             np.concatenate([inner_y, outer_y[::-1]]),
             color='lightgray', alpha=0.5, label='Ramp surface')

    ax2.plot(result['x'], result['y'], 'b-', linewidth=2, label='Centerline')
    ax2.plot(inner_x, inner_y, 'g-', linewidth=2, label=f'Inner edge (r={inner_r:.1f}m)')
    ax2.plot(outer_x, outer_y, 'r-', linewidth=2, label=f'Outer edge (r={outer_r:.1f}m)')

    # Mark start and end
    ax2.plot(result['x'][0], result['y'][0], 'go', markersize=15, label='START (street)')
    ax2.plot(result['x'][-1], result['y'][-1], 'rs', markersize=15, label='END (garage)')

    # Draw car at midpoint
    mid_idx = len(result['x']) // 2
    car_x = result['x'][mid_idx]
    car_y = result['y'][mid_idx]
    car_angle = result['theta'][mid_idx] + np.pi/2  # Tangent direction

    # Car rectangle
    car_corners = np.array([
        [-CAR_LENGTH/2, -CAR_WIDTH/2],
        [CAR_LENGTH/2, -CAR_WIDTH/2],
        [CAR_LENGTH/2, CAR_WIDTH/2],
        [-CAR_LENGTH/2, CAR_WIDTH/2],
        [-CAR_LENGTH/2, -CAR_WIDTH/2]
    ])

    # Rotate and translate
    rot = np.array([[np.cos(car_angle), -np.sin(car_angle)],
                    [np.sin(car_angle), np.cos(car_angle)]])
    car_corners_rot = car_corners @ rot.T
    car_corners_rot[:, 0] += car_x
    car_corners_rot[:, 1] += car_y

    ax2.plot(car_corners_rot[:, 0], car_corners_rot[:, 1], 'orange', linewidth=2)
    ax2.fill(car_corners_rot[:-1, 0], car_corners_rot[:-1, 1],
             color='orange', alpha=0.3, label='Car (at midpoint)')

    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title('Top View (Plan)', fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='upper left')

    # Draw radius lines
    ax2.plot([0, R_horizontal], [0, 0], 'k--', alpha=0.5)
    ax2.plot([0, 0], [0, R_horizontal], 'k--', alpha=0.5)
    ax2.plot(0, 0, 'ko', markersize=8)
    ax2.annotate(f'R={R_horizontal}m', xy=(R_horizontal/2, -0.5), fontsize=10)

    # 3. Side elevation (along arc)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(result['s'], result['z'], 'b-', linewidth=3)
    ax3.fill_between(result['s'], result['z'], -3, alpha=0.3, color='blue')

    # Mark key points
    ax3.axhline(y=0, color='g', linestyle='--', linewidth=2, label='Street level')
    ax3.axhline(y=-VERTICAL_DROP, color='r', linestyle='--', linewidth=2, label='Garage level')

    ax3.set_xlabel('Distance along arc (m)', fontsize=11)
    ax3.set_ylabel('Elevation (m)', fontsize=11)
    ax3.set_title('Side Elevation (Along Path)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Slope angle profile
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(result['s'], result['slope_angles'], 'g-', linewidth=2.5)
    ax4.fill_between(result['s'], 0, result['slope_angles'], alpha=0.3, color='green')
    ax4.axhline(y=result['max_slope_angle'], color='orange', linestyle='--',
               linewidth=2, label=f'Max: {result["max_slope_angle"]:.1f}°')

    ax4.set_xlabel('Distance along arc (m)', fontsize=11)
    ax4.set_ylabel('Slope Angle (°)', fontsize=11)
    ax4.set_title('Slope Angle Profile', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Vertical radius of curvature
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(result['s'], result['vertical_radius'], 'purple', linewidth=2.5)
    ax5.axhline(y=result['R_min_vertical'], color='r', linestyle='--',
               linewidth=2, label=f'Min required: {result["R_min_vertical"]:.1f}m')
    ax5.fill_between(result['s'], result['R_min_vertical'], result['vertical_radius'],
                    where=(result['vertical_radius'] >= result['R_min_vertical']),
                    alpha=0.3, color='green', label='Safe zone')

    ax5.set_xlabel('Distance along arc (m)', fontsize=11)
    ax5.set_ylabel('Vertical Radius (m)', fontsize=11)
    ax5.set_title('Vertical Radius of Curvature', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 50)

    # 6. Summary comparison
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Compare with straight ramp
    straight_length = 12.0
    arc_length = result['arc_length']

    summary_text = f"""
    RADIAL RAMP DESIGN SUMMARY
    ══════════════════════════════════════

    GEOMETRY
    ────────────────────────────────────
    Horizontal radius:     {R_horizontal:.1f} m
    Quarter circle arc:    {arc_length:.2f} m
    Vertical drop:         {VERTICAL_DROP:.1f} m
    Ramp width:            {RAMP_WIDTH:.1f} m

    COMPARISON TO STRAIGHT RAMP
    ────────────────────────────────────
    Straight ramp length:  {straight_length:.1f} m
    Radial arc length:     {arc_length:.2f} m
    Extra path length:     +{arc_length - straight_length:.2f} m ({(arc_length/straight_length - 1)*100:.1f}%)

    SAFETY ANALYSIS
    ────────────────────────────────────
    Min vertical radius:   {result['min_vertical_radius']:.2f} m
    Required minimum:      {result['R_min_vertical']:.2f} m
    Safety factor:         {result['min_vertical_radius']/result['R_min_vertical']:.2f}x

    Inner turn radius:     {result['inner_radius']:.2f} m
    Car min turn radius:   {result['min_turning_radius']:.2f} m
    Turn margin:           +{result['inner_radius'] - result['min_turning_radius']:.2f} m

    Max slope angle:       {result['max_slope_angle']:.1f}°

    VERDICT
    ────────────────────────────────────
    Vertical clearance:    {'✓ SAFE' if result['vertical_safe'] else '✗ UNSAFE'}
    Turning radius:        {'✓ COMFORTABLE' if result['turning_comfortable'] else '✓ OK' if result['turning_safe'] else '✗ TIGHT'}
    Overall:               {'✓ RECOMMENDED' if result['is_comfortable'] else '✓ ACCEPTABLE' if result['is_safe'] else '✗ NEEDS REVISION'}
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Radial Ramp Design: R={R_horizontal}m Quarter Circle\n'
                 f'Arc Length: {arc_length:.2f}m | Drop: {VERTICAL_DROP}m',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/workspaces/RAMP/radial_ramp_solution.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved: radial_ramp_solution.png")

    return result


def generate_construction_data(R_horizontal=MAX_RADIUS):
    """Generate detailed construction measurements for the radial ramp."""
    result = analyze_radial_ramp(R_horizontal)

    print("\n" + "=" * 80)
    print("RADIAL RAMP - CONSTRUCTION MEASUREMENTS")
    print("=" * 80)

    print(f"\nRamp Specifications:")
    print(f"  Centerline radius: {R_horizontal}m")
    print(f"  Inner edge radius: {result['inner_radius']:.2f}m")
    print(f"  Outer edge radius: {result['outer_radius']:.2f}m")
    print(f"  Arc length: {result['arc_length']:.2f}m")
    print(f"  Width: {RAMP_WIDTH}m")

    # Generate measurement points every 0.5m along arc
    print(f"\n{'Arc Dist':>10} {'Angle':>8} {'X':>8} {'Y':>8} {'Depth':>10} {'Depth':>10} {'Slope':>8}")
    print(f"{'(m)':>10} {'(deg)':>8} {'(m)':>8} {'(m)':>8} {'(m)':>10} {'(cm)':>10} {'(deg)':>8}")
    print("-" * 80)

    # Sample at regular intervals
    arc_length = result['arc_length']
    n_samples = int(arc_length / 0.5) + 1

    for i in range(n_samples):
        s_val = i * 0.5
        if s_val > arc_length:
            s_val = arc_length

        # Find closest index
        idx = np.argmin(np.abs(result['s'] - s_val))

        theta_deg = np.degrees(result['theta'][idx])
        x = result['x'][idx]
        y = result['y'][idx]
        z = result['z'][idx]
        slope = result['slope_angles'][idx]

        print(f"{s_val:>10.1f} {theta_deg:>8.1f} {x:>8.2f} {y:>8.2f} {z:>10.3f} {z*100:>10.1f} {slope:>8.1f}")

    # Save to CSV
    import csv
    with open('/workspaces/RAMP/radial_measurements.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Arc_Distance_m', 'Angle_deg', 'X_m', 'Y_m', 'Depth_m', 'Depth_cm', 'Slope_deg',
                        'Inner_X_m', 'Inner_Y_m', 'Outer_X_m', 'Outer_Y_m'])

        inner_r = result['inner_radius']
        outer_r = result['outer_radius']

        # Every 10cm for detailed construction
        for i in range(0, len(result['s']), max(1, len(result['s'])//150)):
            s_val = result['s'][i]
            theta = result['theta'][i]
            theta_deg = np.degrees(theta)
            x = result['x'][i]
            y = result['y'][i]
            z = result['z'][i]
            slope = result['slope_angles'][i]

            inner_x = inner_r * np.cos(theta)
            inner_y = inner_r * np.sin(theta)
            outer_x = outer_r * np.cos(theta)
            outer_y = outer_r * np.sin(theta)

            writer.writerow([f'{s_val:.3f}', f'{theta_deg:.2f}',
                           f'{x:.4f}', f'{y:.4f}',
                           f'{z:.4f}', f'{z*100:.2f}', f'{slope:.2f}',
                           f'{inner_x:.4f}', f'{inner_y:.4f}',
                           f'{outer_x:.4f}', f'{outer_y:.4f}'])

    print(f"\nDetailed measurements saved to: radial_measurements.csv")

    return result


def create_construction_blueprint():
    """Create a detailed construction blueprint for the radial ramp."""

    R = MAX_RADIUS
    result = analyze_radial_ramp(R)

    fig = plt.figure(figsize=(28, 20))

    # Use GridSpec for flexible layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.25)

    # 1. Top View (Plan) - Main drawing
    ax1 = fig.add_subplot(gs[0, 0])

    # Draw ground/surroundings
    ax1.fill([-2, 12, 12, -2], [-2, -2, 12, 12], color='#e8e8e8', alpha=0.3)

    # Draw the ramp surface
    theta = np.linspace(0, np.pi/2, 100)
    inner_r = result['inner_radius']
    outer_r = result['outer_radius']

    inner_x = inner_r * np.cos(theta)
    inner_y = inner_r * np.sin(theta)
    outer_x = outer_r * np.cos(theta)
    outer_y = outer_r * np.sin(theta)

    # Ramp surface
    ax1.fill(np.concatenate([inner_x, outer_x[::-1]]),
             np.concatenate([inner_y, outer_y[::-1]]),
             color='#d4a574', alpha=0.7, edgecolor='black', linewidth=2)

    # Centerline with dashes
    ax1.plot(result['x'], result['y'], 'w--', linewidth=2, label='Centerline')

    # Grid lines on ramp (every meter along arc)
    arc_length = result['arc_length']
    for dist in np.arange(0, arc_length, 1.0):
        idx = np.argmin(np.abs(result['s'] - dist))
        t = result['theta'][idx]
        ax1.plot([inner_r * np.cos(t), outer_r * np.cos(t)],
                [inner_r * np.sin(t), outer_r * np.sin(t)],
                'gray', linewidth=0.5, alpha=0.7)
        # Label
        mid_r = R
        ax1.text(mid_r * np.cos(t), mid_r * np.sin(t), f'{dist:.0f}m',
                fontsize=8, ha='center', va='center', rotation=np.degrees(t)-90)

    # Start and end areas
    ax1.fill([R-1, R+1, R+1, R-1], [-2, -2, 0, 0], color='green', alpha=0.3, label='Street (start)')
    ax1.fill([-2, 0, 0, -2], [R-1, R-1, R+1, R+1], color='blue', alpha=0.3, label='Garage (end)')

    # Dimensions
    ax1.annotate('', xy=(R, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(R/2, -0.8, f'R = {R}m', fontsize=12, ha='center', color='red', fontweight='bold')

    ax1.annotate('', xy=(outer_r, 0), xytext=(inner_r, 0),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax1.text(R, -1.5, f'Width = {RAMP_WIDTH}m', fontsize=10, ha='center', color='blue')

    # Mark center of arc
    ax1.plot(0, 0, 'ko', markersize=10)
    ax1.text(0.3, 0.3, 'Center', fontsize=10)

    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('TOP VIEW (PLAN)\nRadial Ramp Layout', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim(-3, R + 3)
    ax1.set_ylim(-3, R + 3)

    # 2. Unrolled Side Elevation (spans 2 columns)
    ax2 = fig.add_subplot(gs[0, 1:])

    s = result['s']
    z = result['z']

    # Draw ground levels
    ax2.axhline(y=0, color='green', linewidth=3, label='Street level')
    ax2.axhline(y=-VERTICAL_DROP, color='blue', linewidth=3, label='Garage level')

    # Ramp profile
    ax2.fill_between(s, z, -3.5, color='#d4a574', alpha=0.7)
    ax2.plot(s, z, 'k-', linewidth=3)

    # Measurement points every 50cm
    for dist in np.arange(0, arc_length + 0.25, 0.5):
        if dist <= arc_length:
            idx = np.argmin(np.abs(s - dist))
            depth = z[idx]
            ax2.plot(dist, depth, 'ro', markersize=6)
            ax2.plot([dist, dist], [0, depth], 'r--', alpha=0.3, linewidth=0.5)
            # Alternate label positions to avoid overlap
            if int(dist * 2) % 2 == 0:  # Every meter: label above
                ax2.text(dist, depth - 0.12, f'{abs(depth)*100:.0f}', fontsize=7,
                        ha='center', va='top', rotation=45, color='darkred')
            else:  # Every 0.5m offset: label below
                ax2.text(dist, depth + 0.08, f'{abs(depth)*100:.0f}', fontsize=6,
                        ha='center', va='bottom', rotation=45, color='darkred', alpha=0.8)

    ax2.set_xlabel('Distance along arc (m)', fontsize=12)
    ax2.set_ylabel('Elevation (m)', fontsize=12)
    ax2.set_title('SIDE ELEVATION (Unrolled) - Depth measurements every 50cm\nProfile along centerline', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, arc_length + 0.5)
    ax2.set_ylim(-3.5, 0.5)

    # 3. Cross Section
    ax3 = fig.add_subplot(gs[1, 0])

    # Draw cross section at midpoint
    width_points = np.linspace(-RAMP_WIDTH/2, RAMP_WIDTH/2, 50)
    surface_height = 0.15  # Concrete thickness

    # Ramp surface (slightly crowned for drainage)
    crown = 0.02  # 2cm crown at center
    surface = crown * (1 - (width_points / (RAMP_WIDTH/2))**2)

    ax3.fill_between(width_points, surface, surface - surface_height,
                    color='#d4a574', alpha=0.9, edgecolor='black', linewidth=2)

    # Subbase
    ax3.fill_between(width_points, surface - surface_height, surface - surface_height - 0.1,
                    color='gray', alpha=0.5, label='Gravel base (10cm)')

    # Ground
    ax3.fill_between(width_points, surface - surface_height - 0.1, -0.5,
                    color='#8B4513', alpha=0.3, label='Compacted soil')

    # Drainage channels
    ax3.fill([-RAMP_WIDTH/2 - 0.15, -RAMP_WIDTH/2, -RAMP_WIDTH/2, -RAMP_WIDTH/2 - 0.15],
            [0, 0, -0.1, -0.1], color='gray', alpha=0.7, label='Drainage')
    ax3.fill([RAMP_WIDTH/2, RAMP_WIDTH/2 + 0.15, RAMP_WIDTH/2 + 0.15, RAMP_WIDTH/2],
            [0, 0, -0.1, -0.1], color='gray', alpha=0.7)

    # Dimensions
    ax3.annotate('', xy=(RAMP_WIDTH/2, 0.2), xytext=(-RAMP_WIDTH/2, 0.2),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax3.text(0, 0.25, f'{RAMP_WIDTH}m', fontsize=12, ha='center', color='red', fontweight='bold')

    ax3.set_xlabel('Width (m)', fontsize=12)
    ax3.set_ylabel('Height (m)', fontsize=12)
    ax3.set_title('CROSS SECTION\nTypical section at midpoint', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-2.5, 2.5)
    ax3.set_ylim(-0.6, 0.4)

    # 4. Specifications Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    specs_text = f"""
╔═════════════════════════════════════════════╗
║    RADIAL RAMP SPECIFICATIONS               ║
╠═════════════════════════════════════════════╣
║  GEOMETRY                                   ║
║  Configuration:     Quarter circle (90°)    ║
║  Centerline radius: {R:.1f} m                 ║
║  Inner edge radius: {inner_r:.2f} m             ║
║  Outer edge radius: {outer_r:.2f} m            ║
║  Arc length:        {result['arc_length']:.2f} m            ║
║  Ramp width:        {RAMP_WIDTH:.1f} m                 ║
║  Vertical drop:     {VERTICAL_DROP:.1f} m                 ║
╠═════════════════════════════════════════════╣
║  SLOPES & CURVATURE                         ║
║  Maximum slope:     {result['max_slope_angle']:.1f}°                 ║
║  Entry/Exit slope:  0° (level)              ║
║  Min vert. radius:  {result['min_vertical_radius']:.2f} m (req: {result['R_min_vertical']:.2f} m) ║
╠═════════════════════════════════════════════╣
║  MATERIALS                                  ║
║  Concrete: 32 MPa, 150mm thick              ║
║  Rebar: 12mm @ 200mm grid                   ║
║  Base: 100mm compacted gravel               ║
╠═════════════════════════════════════════════╣
║  VEHICLE (Porsche 911)                      ║
║  Ground clearance:  {GROUND_CLEARANCE*1000:.0f} mm               ║
║  Min turn radius:   {result['min_turning_radius']:.2f} m             ║
║  Inner edge margin: +{result['inner_radius'] - result['min_turning_radius']:.2f} m (SAFE)       ║
╚═════════════════════════════════════════════╝
    """

    ax4.text(0.02, 0.98, specs_text, transform=ax4.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # 5. Measurement Table (every 50cm)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    # Generate measurement data every 50cm
    arc_length = result['arc_length']
    s = result['s']
    z = result['z']
    angles = result['slope_angles']

    table_header = "ELEVATION PROFILE (every 50cm)\n"
    table_header += "═" * 40 + "\n"
    table_header += f"{'Arc(m)':<7} {'Depth(cm)':<10} {'Slope(°)':<8}\n"
    table_header += "─" * 40 + "\n"

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

    # 6. Bottom panel: Full measurement table with X, Y coordinates
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    # Create detailed table with all coordinates
    full_table = "DETAILED CONSTRUCTION MEASUREMENTS (Centerline coordinates every 50cm)\n"
    full_table += "═" * 120 + "\n"
    full_table += f"{'Arc Dist':<10} {'Angle':<8} {'X':<10} {'Y':<10} {'Depth':<12} {'Depth':<10} {'Slope':<8} │ "
    full_table += f"{'Arc Dist':<10} {'Angle':<8} {'X':<10} {'Y':<10} {'Depth':<12} {'Depth':<10} {'Slope':<8}\n"
    full_table += f"{'(m)':<10} {'(deg)':<8} {'(m)':<10} {'(m)':<10} {'(m)':<12} {'(cm)':<10} {'(deg)':<8} │ "
    full_table += f"{'(m)':<10} {'(deg)':<8} {'(m)':<10} {'(m)':<10} {'(m)':<12} {'(cm)':<10} {'(deg)':<8}\n"
    full_table += "─" * 120 + "\n"

    # Collect all measurement points
    measurements = []
    for dist in np.arange(0, arc_length + 0.25, 0.5):
        if dist <= arc_length:
            idx = np.argmin(np.abs(s - dist))
            theta_deg = np.degrees(result['theta'][idx])
            x = result['x'][idx]
            y = result['y'][idx]
            depth = z[idx]
            slope = angles[idx]
            measurements.append((dist, theta_deg, x, y, depth, slope))

    # Display in two columns
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

    plt.suptitle('RADIAL RAMP CONSTRUCTION BLUEPRINT\n'
                 f'Quarter Circle Design - R={R}m - Arc Length {result["arc_length"]:.2f}m - For Porsche 911',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig('/workspaces/RAMP/radial_construction_blueprint.png', dpi=200, bbox_inches='tight')
    print("\nConstruction blueprint saved: radial_construction_blueprint.png")


def analyze_swept_path():
    """
    Analyze whether the car can physically fit on the curved ramp.

    This addresses the critical question: when turning, the rear of the car
    tracks INSIDE the front. Does the inner rear corner stay on the ramp?
    """
    R = MAX_RADIUS
    result = analyze_radial_ramp(R)

    inner_edge = result['inner_radius']
    outer_edge = result['outer_radius']

    print("\n" + "=" * 80)
    print("SWEPT PATH ANALYSIS - CAN THE CAR FIT ON THE RAMP?")
    print("=" * 80)

    print(f"\nRamp dimensions:")
    print(f"  Centerline radius: {R:.2f}m")
    print(f"  Inner edge radius: {inner_edge:.2f}m")
    print(f"  Outer edge radius: {outer_edge:.2f}m")
    print(f"  Ramp width: {RAMP_WIDTH:.2f}m")

    print(f"\nCar dimensions:")
    print(f"  Length: {CAR_LENGTH:.3f}m")
    print(f"  Width: {CAR_WIDTH:.3f}m")
    print(f"  Wheelbase: {WHEELBASE:.3f}m")
    print(f"  Front overhang: {FRONT_OVERHANG:.2f}m")
    print(f"  Rear overhang: {REAR_OVERHANG:.2f}m")

    # Analyze if car drives on centerline
    print(f"\n--- If car CENTER follows the ramp CENTERLINE (R={R}m) ---")
    swept_center = calculate_swept_path(R)

    print(f"\n  Off-tracking (rear tracks inside front): {swept_center['off_tracking']:.3f}m")
    print(f"\n  Car corner radii:")
    print(f"    Outer front corner: {swept_center['R_outer_front']:.2f}m")
    print(f"    Inner front corner: {swept_center['R_inner_front']:.2f}m")
    print(f"    Outer rear corner:  {swept_center['R_outer_rear']:.2f}m")
    print(f"    Inner rear corner:  {swept_center['R_inner_rear']:.2f}m  ← CRITICAL (smallest)")

    print(f"\n  Swept path width: {swept_center['swept_width']:.2f}m")

    inner_clearance = swept_center['R_inner_rear'] - inner_edge
    outer_clearance = outer_edge - swept_center['R_outer_front']

    print(f"\n  Clearances:")
    print(f"    Inner edge: {inner_clearance:.2f}m {'✓ OK' if inner_clearance >= 0 else '✗ COLLISION!'}")
    print(f"    Outer edge: {outer_clearance:.2f}m {'✓ OK' if outer_clearance >= 0 else '✗ COLLISION!'}")

    if inner_clearance < 0:
        print(f"\n  ⚠ WARNING: Car inner rear corner goes {abs(inner_clearance):.2f}m PAST the inner edge!")

    # Find the safe driving line
    print(f"\n--- Finding SAFE driving line ---")
    safe = find_safe_driving_line(inner_edge, outer_edge)

    print(f"\n  Minimum safe centerline radius: {safe['min_centerline_radius']:.2f}m")
    print(f"  (Car must drive {safe['min_centerline_radius'] - R:.2f}m outside the geometric centerline)")

    swept_safe = safe['swept_path']
    print(f"\n  When driving at R={safe['min_centerline_radius']:.2f}m:")
    print(f"    Inner rear corner at: {swept_safe['R_inner_rear']:.2f}m")
    print(f"    Outer front corner at: {swept_safe['R_outer_front']:.2f}m")
    print(f"    Inner clearance: {safe['inner_clearance']:.2f}m")
    print(f"    Outer clearance: {safe['outer_clearance']:.2f}m")

    if safe['fits_on_ramp']:
        print(f"\n  ✓ Car CAN fit on the ramp if driven on the correct line")
    else:
        print(f"\n  ✗ Car CANNOT fit - ramp is too narrow!")
        print(f"    Need wider ramp or larger radius")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Top view with swept path
    ax1 = axes[0]

    # Draw ramp
    theta = np.linspace(0, np.pi/2, 100)
    inner_x = inner_edge * np.cos(theta)
    inner_y = inner_edge * np.sin(theta)
    outer_x = outer_edge * np.cos(theta)
    outer_y = outer_edge * np.sin(theta)

    ax1.fill(np.concatenate([inner_x, outer_x[::-1]]),
             np.concatenate([inner_y, outer_y[::-1]]),
             color='#d4a574', alpha=0.5, label='Ramp surface')
    ax1.plot(inner_x, inner_y, 'g-', linewidth=2, label=f'Inner edge (R={inner_edge:.2f}m)')
    ax1.plot(outer_x, outer_y, 'r-', linewidth=2, label=f'Outer edge (R={outer_edge:.2f}m)')

    # Draw centerline
    center_x = R * np.cos(theta)
    center_y = R * np.sin(theta)
    ax1.plot(center_x, center_y, 'b--', linewidth=1, label=f'Geometric centerline (R={R}m)')

    # Draw safe driving line
    safe_x = safe['min_centerline_radius'] * np.cos(theta)
    safe_y = safe['min_centerline_radius'] * np.sin(theta)
    ax1.plot(safe_x, safe_y, 'b-', linewidth=2, label=f'Safe driving line (R={safe["min_centerline_radius"]:.2f}m)')

    # Draw car at several positions along the safe line
    for car_theta in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2.2]:
        draw_car_on_curve(ax1, safe['min_centerline_radius'], car_theta, alpha=0.4)

    # Draw swept path boundary
    inner_swept_x = swept_safe['R_inner_rear'] * np.cos(theta)
    inner_swept_y = swept_safe['R_inner_rear'] * np.sin(theta)
    outer_swept_x = swept_safe['R_outer_front'] * np.cos(theta)
    outer_swept_y = swept_safe['R_outer_front'] * np.sin(theta)

    ax1.plot(inner_swept_x, inner_swept_y, 'm--', linewidth=1.5, label=f'Inner swept path (R={swept_safe["R_inner_rear"]:.2f}m)')
    ax1.plot(outer_swept_x, outer_swept_y, 'm--', linewidth=1.5, label=f'Outer swept path (R={swept_safe["R_outer_front"]:.2f}m)')

    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Top View: Car Swept Path on Ramp', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, outer_edge + 1)
    ax1.set_ylim(-1, outer_edge + 1)

    # Right plot: Summary diagram
    ax2 = axes[1]
    ax2.axis('off')

    summary = f"""
    SWEPT PATH ANALYSIS SUMMARY
    ══════════════════════════════════════════════════

    RAMP GEOMETRY
    ─────────────────────────────────────────────
    Centerline radius:     {R:.2f} m
    Inner edge radius:     {inner_edge:.2f} m
    Outer edge radius:     {outer_edge:.2f} m
    Ramp width:            {RAMP_WIDTH:.2f} m

    CAR (Porsche 911)
    ─────────────────────────────────────────────
    Length × Width:        {CAR_LENGTH:.2f} × {CAR_WIDTH:.2f} m
    Wheelbase:             {WHEELBASE:.2f} m
    Off-tracking:          {swept_safe['off_tracking']:.3f} m

    SWEPT PATH (when driving safe line)
    ─────────────────────────────────────────────
    Safe driving radius:   {safe['min_centerline_radius']:.2f} m
    Swept path width:      {swept_safe['swept_width']:.2f} m

    Inner rear corner:     {swept_safe['R_inner_rear']:.2f} m
    Outer front corner:    {swept_safe['R_outer_front']:.2f} m

    CLEARANCES
    ─────────────────────────────────────────────
    Inner edge clearance:  {safe['inner_clearance']:.2f} m {'✓' if safe['inner_clearance'] >= 0 else '✗'}
    Outer edge clearance:  {safe['outer_clearance']:.2f} m {'✓' if safe['outer_clearance'] >= 0 else '✗'}

    VERDICT
    ─────────────────────────────────────────────
    {'✓ CAR FITS ON RAMP' if safe['fits_on_ramp'] else '✗ CAR DOES NOT FIT'}

    {'Driver must stay ' + f'{safe["min_centerline_radius"] - R:.2f}m outside centerline' if safe['fits_on_ramp'] else 'NEED WIDER RAMP OR LARGER RADIUS'}
    """

    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('Radial Ramp - Swept Path Analysis\nCan the Porsche 911 navigate the curved ramp?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspaces/RAMP/swept_path_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSwept path analysis saved: swept_path_analysis.png")

    return safe


def draw_car_on_curve(ax, R_center, theta, alpha=1.0):
    """Draw a car positioned on a curved path at angle theta."""
    # Car center position
    cx = R_center * np.cos(theta)
    cy = R_center * np.sin(theta)

    # Car orientation (tangent to the curve, pointing in direction of travel)
    car_angle = theta + np.pi/2

    # Calculate corner positions accounting for turning geometry
    # Front axle is ahead of center by wheelbase/2
    # Rear axle tracks inside

    # Simplified: draw car rectangle at the center position
    corners = np.array([
        [-CAR_LENGTH/2, -CAR_WIDTH/2],
        [CAR_LENGTH/2, -CAR_WIDTH/2],
        [CAR_LENGTH/2, CAR_WIDTH/2],
        [-CAR_LENGTH/2, CAR_WIDTH/2],
        [-CAR_LENGTH/2, -CAR_WIDTH/2]
    ])

    # Rotate
    rot = np.array([[np.cos(car_angle), -np.sin(car_angle)],
                    [np.sin(car_angle), np.cos(car_angle)]])
    corners_rot = corners @ rot.T
    corners_rot[:, 0] += cx
    corners_rot[:, 1] += cy

    ax.fill(corners_rot[:-1, 0], corners_rot[:-1, 1],
            color='orange', alpha=alpha*0.5, edgecolor='darkorange', linewidth=1)


def main():
    """Main function to run all analyses."""
    print("\n" + "=" * 80)
    print("RADIAL RAMP DESIGN FOR GARAGE ACCESS")
    print("Quarter Circle Path - Maximum Radius 9m")
    print("=" * 80)

    # Compare designs
    compare_straight_vs_radial()

    # Analyze swept path - can the car fit?
    safe_driving = analyze_swept_path()

    # Create main visualization
    result = create_visualization()

    # Generate construction data
    generate_construction_data()

    # Create blueprint
    create_construction_blueprint()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print(f"""
    The RADIAL RAMP design with R={MAX_RADIUS}m is RECOMMENDED.

    ADVANTAGES OVER STRAIGHT RAMP:
    ─────────────────────────────────────────
    1. Arc length of {result['arc_length']:.2f}m vs 12m straight
       (+{result['arc_length'] - 12:.2f}m extra distance for descent)

    2. Lower maximum slope: {result['max_slope_angle']:.1f}° vs 19.3°
       (gentler gradient throughout)

    3. Better vertical curvature: {result['min_vertical_radius']:.2f}m vs 8.6m
       (more clearance margin for low cars)

    4. Natural turning motion into garage
       (car arrives oriented correctly)

    SAFETY VERIFICATION:
    ─────────────────────────────────────────
    ✓ Vertical radius: {result['min_vertical_radius']:.2f}m > {result['R_min_vertical']:.2f}m required
    ✓ Inner turn radius: {result['inner_radius']:.2f}m > {result['min_turning_radius']:.2f}m car minimum
    ✓ Maximum slope: {result['max_slope_angle']:.1f}° (very manageable)

    OUTPUT FILES:
    ─────────────────────────────────────────
    • radial_ramp_solution.png      - Analysis visualization
    • radial_construction_blueprint.png - Construction drawings
    • radial_measurements.csv       - Detailed measurement data
    """)

    plt.show()


if __name__ == '__main__':
    main()
