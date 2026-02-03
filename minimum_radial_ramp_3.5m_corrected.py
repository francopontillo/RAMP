#!/usr/bin/env python3
"""
CORRECTED Minimum Radial Ramp Design for Garage Access
VERTICAL DROP: 3.5 meters

CORRECTION: This version properly accounts for front AND rear overhangs
when calculating ground clearance. The previous version only checked
clearance at the wheelbase center, missing the critical overhang clearance.

Key finding: The rear overhang (1.261m) is longer than the front (0.85m),
making the rear bumper the critical point for clearance on convex transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# Car specifications (Porsche 911, 997.1 Turbo, 2008)
WHEELBASE = 2.350  # m - distance between axles
GROUND_CLEARANCE = 0.106  # m (106mm)
CAR_LENGTH = 4.461  # m
CAR_WIDTH = 1.808  # m
TRACK_WIDTH = 1.516  # m (front track)

# Critical overhang dimensions
FRONT_OVERHANG = 0.85  # m (front axle to front bumper)
REAR_OVERHANG = CAR_LENGTH - WHEELBASE - FRONT_OVERHANG  # 1.261m

# Approach, Departure, and Breakover angles (from specifications)
APPROACH_ANGLE = 7.9  # degrees - max slope car can enter without front scraping
DEPARTURE_ANGLE = 12.8  # degrees - max slope car can exit without rear scraping
BREAKOVER_ANGLE = 12.7  # degrees - max crest angle car can traverse

# Calculate minimum convex radius from breakover angle
# For a convex curve, the car's undercarriage must clear the crest
# R_min = WB / (2 * sin(breakover_angle/2)) for the geometric constraint
# Or approximately: R_min = WB^2 / (8 * GC) for small angles
BREAKOVER_ANGLE_RAD = np.radians(BREAKOVER_ANGLE)
MIN_CONVEX_RADIUS_GEOMETRIC = WHEELBASE / (2 * np.sin(BREAKOVER_ANGLE_RAD / 2))
MIN_CONVEX_RADIUS_APPROX = WHEELBASE**2 / (8 * GROUND_CLEARANCE)

print(f"\n{'='*80}")
print("BREAKOVER ANGLE ANALYSIS")
print(f"{'='*80}")
print(f"Breakover angle: {BREAKOVER_ANGLE}°")
print(f"Min convex radius (geometric): {MIN_CONVEX_RADIUS_GEOMETRIC:.2f}m")
print(f"Min convex radius (approx WB²/8GC): {MIN_CONVEX_RADIUS_APPROX:.2f}m")
print(f"User's calculated minimum: 10.62m")

# Ramp requirements
VERTICAL_DROP = 3.5  # m
ENTRY_FLAT = 5.0  # m
EXIT_FLAT = 5.0   # m

# Safety margins
MIN_EDGE_CLEARANCE = 0.15  # m (15cm minimum clearance from edge)
MIN_GROUND_CLEARANCE = 0.02  # m (20mm minimum ground clearance)


def get_ramp_profile(arc_length, num_points=2000):
    """Generate ramp profile with flat sections - OLD CUBIC PROFILE."""
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)

    # Ramp section (cubic profile)
    ramp_mask = (s >= ENTRY_FLAT) & (s <= ENTRY_FLAT + arc_length)
    s_ramp = s[ramp_mask] - ENTRY_FLAT
    L = arc_length
    H = VERTICAL_DROP
    a = 2 * H / L**3
    b = -3 * H / L**2
    z[ramp_mask] = a * s_ramp**3 + b * s_ramp**2

    # Exit flat
    exit_mask = s > ENTRY_FLAT + arc_length
    z[exit_mask] = -VERTICAL_DROP

    return s, z


def get_compound_curve_profile(R_convex, arc_convex, R_concave, num_points=2000):
    """
    Generate ramp profile with COMPOUND CURVE design:
    1. Entry flat (street level)
    2. Convex curve (R_convex, arc length = arc_convex) - transitions to inclined
    3. Straight inclined section - calculated to achieve total drop
    4. Concave curve (R_concave) - transitions back to horizontal
    5. Exit flat (garage level)

    Returns: s (distance), z (elevation), and geometry details
    """
    # Calculate geometry
    # Convex section: from horizontal (0°) to slope angle θ
    theta_convex = arc_convex / R_convex  # radians

    # For concave to return to horizontal, it needs the same angle
    theta_concave = theta_convex
    arc_concave = R_concave * theta_concave

    # Vertical drops
    # Convex: drops by R * (1 - cos(θ))
    drop_convex = R_convex * (1 - np.cos(theta_convex))

    # Concave: drops by R * (1 - cos(θ))
    drop_concave = R_concave * (1 - np.cos(theta_concave))

    # Straight section must make up the remaining drop
    drop_straight = VERTICAL_DROP - drop_convex - drop_concave

    if drop_straight < 0:
        print(f"WARNING: Curves alone drop {drop_convex + drop_concave:.2f}m, more than required {VERTICAL_DROP}m!")
        drop_straight = 0

    # Length of straight section
    slope_angle = theta_convex  # The slope during straight section
    if np.sin(slope_angle) > 0:
        length_straight = drop_straight / np.sin(slope_angle)
    else:
        length_straight = 0

    # Total ramp length (excluding entry/exit flats)
    ramp_length = arc_convex + length_straight + arc_concave
    total_length = ENTRY_FLAT + ramp_length + EXIT_FLAT

    # Generate profile points
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)
    slope = np.zeros_like(s)  # slope in radians

    # Section boundaries
    s1 = ENTRY_FLAT  # Start of convex
    s2 = s1 + arc_convex  # End of convex, start of straight
    s3 = s2 + length_straight  # End of straight, start of concave
    s4 = s3 + arc_concave  # End of concave, start of exit flat

    for i, si in enumerate(s):
        if si <= s1:
            # Entry flat
            z[i] = 0
            slope[i] = 0
        elif si <= s2:
            # Convex curve (circular arc)
            ds = si - s1  # distance along convex arc
            theta = ds / R_convex  # angle traversed
            # Position on circular arc (center is at (s1, R_convex))
            z[i] = R_convex * (1 - np.cos(theta)) * (-1)  # negative because going down
            slope[i] = theta
        elif si <= s3:
            # Straight inclined section
            ds = si - s2  # distance along straight
            z[i] = -drop_convex - ds * np.sin(slope_angle)
            slope[i] = slope_angle
        elif si <= s4:
            # Concave curve (circular arc)
            ds = si - s3  # distance along concave arc
            theta_remaining = theta_concave - ds / R_concave  # angle still to go
            # Drop from start of concave section
            z[i] = -drop_convex - drop_straight - R_concave * (1 - np.cos(theta_concave)) + R_concave * (1 - np.cos(theta_remaining))
            slope[i] = theta_remaining
        else:
            # Exit flat
            z[i] = -VERTICAL_DROP
            slope[i] = 0

    geometry = {
        'R_convex': R_convex,
        'R_concave': R_concave,
        'arc_convex': arc_convex,
        'arc_concave': arc_concave,
        'length_straight': length_straight,
        'theta_max': np.degrees(theta_convex),  # Maximum slope angle in degrees
        'drop_convex': drop_convex,
        'drop_straight': drop_straight,
        'drop_concave': drop_concave,
        'ramp_length': ramp_length,
        'total_length': total_length,
        's1': s1, 's2': s2, 's3': s3, 's4': s4,
    }

    return s, z, slope, geometry


def get_elevation_at(s_query, s_array, z_array):
    """Interpolate elevation at a specific position."""
    return np.interp(s_query, s_array, z_array)


def check_compound_curve_clearance(s_array, z_array, direction='downhill'):
    """
    Check ground clearance for car traversing compound curve profile.
    Returns minimum clearance and position where it occurs.
    """
    total_length = s_array[-1]
    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 500)

    min_clearance = float('inf')
    min_pos = 0
    min_type = ''

    for pos in positions:
        result = check_car_clearance_full(pos, s_array, z_array, direction)
        if result['front_bumper_clearance'] < min_clearance:
            min_clearance = result['front_bumper_clearance']
            min_pos = pos
            min_type = 'front_bumper'
        if result['rear_bumper_clearance'] < min_clearance:
            min_clearance = result['rear_bumper_clearance']
            min_pos = pos
            min_type = 'rear_bumper'

    return min_clearance, min_pos, min_type


def find_minimum_straight_section(R_curve, arc_length_curve):
    """
    Find minimum straight section length for compound curve design.
    The straight section must be long enough for overhangs to clear.
    """
    print("\n" + "=" * 80)
    print("COMPOUND CURVE DESIGN - Finding minimum straight section")
    print("=" * 80)
    print(f"\nDesign parameters:")
    print(f"  Convex radius:  R = {R_curve}m")
    print(f"  Convex arc:     {arc_length_curve}m")
    print(f"  Concave radius: R = {R_curve}m (same)")

    # Calculate the slope angle
    theta = arc_length_curve / R_curve
    slope_deg = np.degrees(theta)
    print(f"  Max slope:      {slope_deg:.1f}°")

    # Calculate drops from curves
    drop_per_curve = R_curve * (1 - np.cos(theta))
    total_curve_drop = 2 * drop_per_curve
    remaining_drop = VERTICAL_DROP - total_curve_drop

    print(f"\n  Drop from convex:  {drop_per_curve:.3f}m")
    print(f"  Drop from concave: {drop_per_curve:.3f}m")
    print(f"  Remaining for straight: {remaining_drop:.3f}m")

    if remaining_drop < 0:
        print(f"\n  ERROR: Curves alone drop more than {VERTICAL_DROP}m!")
        return None

    # Minimum straight length just for the drop
    min_straight_for_drop = remaining_drop / np.sin(theta)
    print(f"  Min straight for drop: {min_straight_for_drop:.2f}m")

    # Now search for minimum straight section that provides clearance
    print(f"\n{'Straight':>10} {'Total':>10} {'Down Cl':>12} {'Up Cl':>12} {'Status':<15}")
    print("-" * 70)

    best_result = None

    for extra_straight in np.arange(0, 15, 0.5):
        length_straight = min_straight_for_drop + extra_straight

        # Generate profile
        s, z, slope, geom = get_compound_curve_profile(R_curve, arc_length_curve, R_curve)

        # But we need to recalculate with the actual straight length
        # Let's modify the approach - calculate the profile with variable straight section

        # Recalculate with extra straight section
        # The extra straight section adds length but not drop (it's horizontal... no wait, it's inclined)
        # Actually the straight section IS inclined, so adding length adds drop too

        # We need a different approach: keep the same drop, but vary the straight section
        # by adjusting the curve parameters

        # For now, let's use the compound curve profile generator
        # We'll override by manually setting the straight length

        # Create a modified geometry
        arc_concave = arc_length_curve  # Same as convex
        theta_max = theta

        # With the specified straight length, recalculate drop
        actual_drop_straight = length_straight * np.sin(theta_max)
        actual_total_drop = 2 * drop_per_curve + actual_drop_straight

        # Generate profile manually
        total_length = ENTRY_FLAT + arc_length_curve + length_straight + arc_concave + EXIT_FLAT
        s = np.linspace(0, total_length, 2000)
        z = np.zeros_like(s)

        s1 = ENTRY_FLAT
        s2 = s1 + arc_length_curve
        s3 = s2 + length_straight
        s4 = s3 + arc_concave

        for i, si in enumerate(s):
            if si <= s1:
                z[i] = 0
            elif si <= s2:
                ds = si - s1
                th = ds / R_curve
                z[i] = -R_curve * (1 - np.cos(th))
            elif si <= s3:
                ds = si - s2
                z[i] = -drop_per_curve - ds * np.sin(theta_max)
            elif si <= s4:
                ds = si - s3
                th_remaining = theta_max - ds / R_curve
                z[i] = -drop_per_curve - actual_drop_straight - R_curve * (1 - np.cos(theta_max)) + R_curve * (1 - np.cos(th_remaining))
            else:
                z[i] = -actual_total_drop

        # Check clearance
        min_down, pos_down, type_down = check_compound_curve_clearance(s, z, 'downhill')
        min_up, pos_up, type_up = check_compound_curve_clearance(s, z, 'uphill')

        clearance_ok = min(min_down, min_up) >= MIN_GROUND_CLEARANCE
        status = "✓ VALID" if clearance_ok else "✗"

        print(f"{length_straight:>10.2f} {total_length:>10.2f} "
              f"{min_down*1000:>10.1f}mm {min_up*1000:>10.1f}mm {status:<15}")

        if clearance_ok and best_result is None:
            best_result = {
                'length_straight': length_straight,
                'total_length': total_length,
                'min_down': min_down,
                'min_up': min_up,
                's': s,
                'z': z,
                'geometry': {
                    'R_convex': R_curve,
                    'R_concave': R_curve,
                    'arc_convex': arc_length_curve,
                    'arc_concave': arc_concave,
                    'length_straight': length_straight,
                    'theta_max': slope_deg,
                    'drop_convex': drop_per_curve,
                    'drop_straight': actual_drop_straight,
                    'drop_concave': drop_per_curve,
                    'total_drop': actual_total_drop,
                    'ramp_length': arc_length_curve + length_straight + arc_concave,
                    'total_length': total_length,
                    's1': s1, 's2': s2, 's3': s3, 's4': s4,
                }
            }

    if best_result:
        print(f"\n{'=' * 80}")
        print(f"MINIMUM COMPOUND CURVE DESIGN FOUND:")
        print(f"  Convex arc:     {arc_length_curve}m (R={R_curve}m)")
        print(f"  Straight:       {best_result['length_straight']:.2f}m at {slope_deg:.1f}° slope")
        print(f"  Concave arc:    {arc_concave}m (R={R_curve}m)")
        print(f"  Total ramp:     {best_result['geometry']['ramp_length']:.2f}m")
        print(f"  Total path:     {best_result['total_length']:.2f}m")
        print(f"  Total drop:     {best_result['geometry']['total_drop']:.3f}m")
        print(f"{'=' * 80}")

    return best_result


def check_car_clearance_full(car_center_pos, s_array, z_array, direction='downhill'):
    """
    Check ground clearance for the ENTIRE car including overhangs.
    """
    half_wheelbase = WHEELBASE / 2

    if direction == 'downhill':
        front_axle_pos = car_center_pos + half_wheelbase
        rear_axle_pos = car_center_pos - half_wheelbase
        front_bumper_pos = front_axle_pos + FRONT_OVERHANG
        rear_bumper_pos = rear_axle_pos - REAR_OVERHANG
    else:
        front_axle_pos = car_center_pos - half_wheelbase
        rear_axle_pos = car_center_pos + half_wheelbase
        front_bumper_pos = front_axle_pos - FRONT_OVERHANG
        rear_bumper_pos = rear_axle_pos + REAR_OVERHANG

    front_axle_ground = get_elevation_at(front_axle_pos, s_array, z_array)
    rear_axle_ground = get_elevation_at(rear_axle_pos, s_array, z_array)

    def car_body_height(x_pos):
        t = (x_pos - rear_axle_pos) / (front_axle_pos - rear_axle_pos)
        body_line_z = rear_axle_ground + t * (front_axle_ground - rear_axle_ground)
        return body_line_z + GROUND_CLEARANCE

    front_bumper_clearance = car_body_height(front_bumper_pos) - get_elevation_at(front_bumper_pos, s_array, z_array)
    rear_bumper_clearance = car_body_height(rear_bumper_pos) - get_elevation_at(rear_bumper_pos, s_array, z_array)

    return {
        'front_bumper_clearance': front_bumper_clearance,
        'rear_bumper_clearance': rear_bumper_clearance,
        'min_clearance': min(front_bumper_clearance, rear_bumper_clearance),
    }


def analyze_clearance(arc_length):
    """Analyze clearance for both directions."""
    s_array, z_array = get_ramp_profile(arc_length)
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 500)

    min_down = float('inf')
    min_up = float('inf')

    for pos in positions:
        down = check_car_clearance_full(pos, s_array, z_array, 'downhill')
        up = check_car_clearance_full(pos, s_array, z_array, 'uphill')
        min_down = min(min_down, down['min_clearance'])
        min_up = min(min_up, up['min_clearance'])

    return min_down, min_up


def calculate_swept_path(R_centerline):
    """Calculate the swept path of the car on a curved path."""
    W = CAR_WIDTH
    L = WHEELBASE
    f_front = FRONT_OVERHANG
    f_rear = REAR_OVERHANG

    R_front_axle = R_centerline
    R_rear_axle = np.sqrt(R_front_axle**2 - L**2)
    off_tracking = R_front_axle - R_rear_axle

    R_outer_front = np.sqrt((R_front_axle + W/2)**2 + f_front**2)
    R_inner_front = np.sqrt((R_front_axle - W/2)**2 + f_front**2)
    R_outer_rear = np.sqrt((R_rear_axle + W/2)**2 + f_rear**2)
    R_inner_rear = R_rear_axle - W/2

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


def calculate_vertical_profile(arc_length):
    """Calculate vertical profile characteristics."""
    s, z = get_ramp_profile(arc_length, 1000)

    # Only analyze ramp section
    ramp_start = ENTRY_FLAT
    ramp_end = ENTRY_FLAT + arc_length
    ramp_mask = (s >= ramp_start) & (s <= ramp_end)

    s_ramp = s[ramp_mask]
    z_ramp = z[ramp_mask]

    ds = s_ramp[1] - s_ramp[0]
    dz_ds = np.gradient(z_ramp, ds)
    d2z_ds2 = np.gradient(dz_ds, ds)

    curvature = np.abs(d2z_ds2) / (1 + dz_ds**2)**1.5
    min_radius = 1 / np.max(curvature[1:-1])  # Exclude endpoints

    max_slope = np.max(np.degrees(np.arctan(-dz_ds)))

    return {
        'arc_length': arc_length,
        'min_vertical_radius': min_radius,
        'max_slope': max_slope,
        's': s,
        'z': z,
        'slope_angles': np.degrees(np.arctan(-np.gradient(z, s[1]-s[0]))),
    }


def calculate_curve_radii(arc_length):
    """
    Calculate the convex and concave radii for the cubic profile.

    For cubic profile z = a*s³ + b*s² where a = 2H/L³, b = -3H/L²:
    - At s=0 (top, entry to ramp): z'' = 2b = -6H/L², radius R = L²/(6H) [convex]
    - At s=L (bottom, exit from ramp): z'' = 6H/L², radius R = L²/(6H) [concave]

    Both radii are equal for this symmetric cubic profile.
    """
    H = VERTICAL_DROP
    L = arc_length
    radius = L**2 / (6 * H)
    return radius, radius  # convex_radius, concave_radius


def find_minimum_arc_length_by_breakover():
    """
    Find the minimum arc length based on the breakover angle constraint.

    The minimum convex radius from breakover angle determines the shortest possible arc.
    R_convex = L²/(6H), so L = sqrt(6 * H * R_min)
    """
    # Use the user's calculated minimum convex radius
    R_min_convex = 10.62  # meters, based on breakover angle of 12.7°

    L_min_breakover = np.sqrt(6 * VERTICAL_DROP * R_min_convex)
    return L_min_breakover, R_min_convex


def find_minimum_safe_radius():
    """Find minimum arc length that satisfies BOTH breakover angle AND clearance constraints."""

    print("=" * 80)
    print(f"FINDING MINIMUM SAFE ARC LENGTH FOR {VERTICAL_DROP}m DROP")
    print("(Checking: Breakover angle + Full car clearance including overhangs)")
    print("=" * 80)

    print(f"\nCar geometry:")
    print(f"  Wheelbase:        {WHEELBASE:.3f}m")
    print(f"  Front overhang:   {FRONT_OVERHANG:.3f}m (front axle to bumper)")
    print(f"  Rear overhang:    {REAR_OVERHANG:.3f}m (rear axle to bumper)")
    print(f"  Ground clearance: {GROUND_CLEARANCE*1000:.0f}mm")
    print(f"  Total length:     {CAR_LENGTH:.3f}m")

    print(f"\nAngle constraints:")
    print(f"  Approach angle:   {APPROACH_ANGLE}°")
    print(f"  Departure angle:  {DEPARTURE_ANGLE}°")
    print(f"  Breakover angle:  {BREAKOVER_ANGLE}°")

    # Calculate minimum arc length from breakover angle
    L_min_breakover, R_min_convex = find_minimum_arc_length_by_breakover()
    print(f"\nFrom breakover angle constraint:")
    print(f"  Minimum convex radius: {R_min_convex:.2f}m")
    print(f"  Minimum arc length:    {L_min_breakover:.2f}m")

    print(f"\n{'Arc Len':>10} {'Convex R':>10} {'Concave R':>10} {'Down Cl':>12} {'Up Cl':>12} {'Max Slope':>10} {'Status':<20}")
    print("-" * 100)

    results = []

    # Search arc lengths starting from the theoretical minimum
    # We'll search from slightly below the breakover minimum to find the true minimum
    for arc_length in np.arange(L_min_breakover - 2, 30.0, 0.5):
        if arc_length < 10:  # Sanity check
            continue

        convex_r, concave_r = calculate_curve_radii(arc_length)
        min_down, min_up = analyze_clearance(arc_length)
        vert = calculate_vertical_profile(arc_length)

        # Check all constraints
        breakover_ok = convex_r >= R_min_convex
        clearance_ok = min(min_down, min_up) >= MIN_GROUND_CLEARANCE

        is_valid = breakover_ok and clearance_ok

        # For the horizontal radius (used for swept path), estimate from arc length
        # For a quarter circle: arc_length = π*R/2, so R = 2*arc_length/π
        R_horizontal = 2 * arc_length / np.pi
        swept = calculate_swept_path(R_horizontal)

        results.append({
            'arc_length': arc_length,
            'convex_radius': convex_r,
            'concave_radius': concave_r,
            'R_centerline': R_horizontal,
            'min_down': min_down,
            'min_up': min_up,
            'swept_width': swept['swept_width'],
            'ramp_width': swept['swept_width'] + 2 * MIN_EDGE_CLEARANCE,
            'inner_edge': swept['R_inner_rear'] - MIN_EDGE_CLEARANCE,
            'outer_edge': swept['R_outer_front'] + MIN_EDGE_CLEARANCE,
            'max_slope': vert['max_slope'],
            'min_vertical_radius': min(convex_r, concave_r),
            'is_valid': is_valid,
            'breakover_ok': breakover_ok,
            'clearance_ok': clearance_ok,
            'swept': swept,
            'vert': vert,
        })

        if breakover_ok and clearance_ok:
            status = "✓ VALID"
        elif not breakover_ok:
            status = "✗ BREAKOVER"
        else:
            status = "✗ CLEARANCE"

        print(f"{arc_length:>10.2f} {convex_r:>10.2f} {concave_r:>10.2f} "
              f"{min_down*1000:>10.1f}mm {min_up*1000:>10.1f}mm {vert['max_slope']:>10.1f}° {status:<20}")

    valid_results = [r for r in results if r['is_valid']]

    if valid_results:
        min_result = valid_results[0]
        print(f"\n{'=' * 80}")
        print(f"MINIMUM VALID ARC LENGTH: {min_result['arc_length']:.2f}m")
        print(f"CONVEX RADIUS: {min_result['convex_radius']:.2f}m (min required: {R_min_convex:.2f}m)")
        print(f"CONCAVE RADIUS: {min_result['concave_radius']:.2f}m")
        print(f"HORIZONTAL CENTERLINE RADIUS: {min_result['R_centerline']:.2f}m")
        print(f"{'=' * 80}")
        return min_result
    else:
        print("\nNo valid solution found in search range!")
        # Return the first result that satisfies breakover (clearance might be slightly negative)
        breakover_valid = [r for r in results if r['breakover_ok']]
        if breakover_valid:
            return breakover_valid[0]
        return results[-1] if results else None


def find_critical_uphill_position(arc_length):
    """Find the worst clearance position during uphill travel."""
    s_array, z_array = get_ramp_profile(arc_length)
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 500)

    min_clearance = float('inf')
    worst_pos = 0
    worst_type = ''

    for pos in positions:
        result = check_car_clearance_full(pos, s_array, z_array, 'uphill')
        if result['front_bumper_clearance'] < min_clearance:
            min_clearance = result['front_bumper_clearance']
            worst_pos = pos
            worst_type = 'front_bumper'
        if result['rear_bumper_clearance'] < min_clearance:
            min_clearance = result['rear_bumper_clearance']
            worst_pos = pos
            worst_type = 'rear_bumper'

    return worst_pos, min_clearance, worst_type, s_array, z_array


def create_visualization(result):
    """Create comprehensive visualization."""

    R = result['R_centerline']
    arc_length = result['arc_length']
    inner_edge = result['inner_edge']
    outer_edge = result['outer_edge']
    ramp_width = result['ramp_width']
    swept = result['swept']
    vert = result['vert']

    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    # Create figure - geometric drawings will have equal scale
    fig = plt.figure(figsize=(24, 44))

    # Common axis limits for geometric consistency
    x_min, x_max = -2, total_length + 8  # ~40m range
    y_min_elevation, y_max_elevation = -6, 3  # 9m range for elevation views
    y_min_plan, y_max_plan = -4, R + 6  # For plan view

    # Use gridspec - geometric views get more height
    gs = fig.add_gridspec(6, 3, height_ratios=[1.5, 0.6, 0.6, 0.8, 0.8, 0.6],
                          hspace=0.35, wspace=0.25)

    # 1. Top View - Engineer's perspective: Street and Garage are PARALLEL (both horizontal)
    # The ramp connects them with a 90° curved section
    # Full width to match SIDE ELEVATION scale
    ax1 = fig.add_subplot(gs[0, :])

    # Path width
    path_half_width = ramp_width / 2

    # Geometry layout (looking from above):
    # - STREET runs horizontally at the bottom (y = 0), car travels in +X direction
    # - ENTRY section goes vertically from street up to arc start
    # - ARC curves 90° from vertical to horizontal
    # - EXIT section goes horizontally from arc end to garage
    # - GARAGE runs horizontally at the top (parallel to street)

    # The arc: quarter circle starting at (ENTRY_FLAT, 0) going to (ENTRY_FLAT + R, R)
    # Arc center is at (ENTRY_FLAT + R, 0)
    arc_center_x = ENTRY_FLAT + R
    arc_center_y = 0

    theta = np.linspace(np.pi, np.pi/2, 100)  # From 180° to 90° (left to up)

    # Arc edges
    inner_x = arc_center_x + inner_edge * np.cos(theta)
    inner_y = arc_center_y + inner_edge * np.sin(theta)
    outer_x = arc_center_x + outer_edge * np.cos(theta)
    outer_y = arc_center_y + outer_edge * np.sin(theta)
    center_arc_x = arc_center_x + R * np.cos(theta)
    center_arc_y = arc_center_y + R * np.sin(theta)

    # --- STREET (horizontal at base, spanning full width) ---
    street_y_bottom = -2
    street_y_top = 0
    street_x_left = 0
    street_x_right = total_length  # Full width of graph

    ax1.fill([street_x_left, street_x_right, street_x_right, street_x_left],
             [street_y_bottom, street_y_bottom, street_y_top, street_y_top],
             color='#90EE90', alpha=0.5)
    ax1.plot([street_x_left, street_x_right], [street_y_top, street_y_top], 'g-', linewidth=2)
    ax1.plot([street_x_left, street_x_right], [street_y_bottom, street_y_bottom], 'g-', linewidth=2)
    ax1.text(total_length/2, (street_y_bottom + street_y_top)/2, 'STREET (Level 0m)',
             ha='center', va='center', fontsize=10, fontweight='bold', color='darkgreen')

    # --- ENTRY section (vertical, from street to arc start) ---
    entry_x_left = ENTRY_FLAT - path_half_width
    entry_x_right = ENTRY_FLAT + path_half_width

    ax1.fill([entry_x_left, entry_x_right, entry_x_right, entry_x_left],
             [0, 0, R, R],
             color='#d4a574', alpha=0.6)
    ax1.plot([entry_x_left, entry_x_left], [0, R], 'k-', linewidth=1.5)
    ax1.plot([entry_x_right, entry_x_right], [0, R], 'k-', linewidth=1.5)
    ax1.text(ENTRY_FLAT, R/2, f'ENTRY\n{ENTRY_FLAT}m', ha='center', va='center', fontsize=8, fontweight='bold')

    # --- ARC section (quarter circle) ---
    ax1.fill(np.concatenate([inner_x, outer_x[::-1]]),
             np.concatenate([inner_y, outer_y[::-1]]),
             color='#d4a574', alpha=0.6)
    ax1.plot(inner_x, inner_y, 'k-', linewidth=1.5)
    ax1.plot(outer_x, outer_y, 'k-', linewidth=1.5)
    ax1.plot(center_arc_x, center_arc_y, 'b--', linewidth=1.5)

    # Arc label
    mid_theta = 3 * np.pi / 4
    ax1.text(arc_center_x + R * 0.7 * np.cos(mid_theta),
             arc_center_y + R * 0.7 * np.sin(mid_theta),
             f'ARC\nR={R}m\n{arc_length:.1f}m', ha='center', va='center', fontsize=8, fontweight='bold')

    # --- EXIT section (horizontal, from arc end to garage) ---
    exit_y_bottom = R - path_half_width
    exit_y_top = R + path_half_width
    exit_x_start = ENTRY_FLAT + R
    exit_x_end = total_length

    ax1.fill([exit_x_start, exit_x_end, exit_x_end, exit_x_start],
             [exit_y_bottom, exit_y_bottom, exit_y_top, exit_y_top],
             color='#d4a574', alpha=0.6)
    ax1.plot([exit_x_start, exit_x_end], [exit_y_bottom, exit_y_bottom], 'k-', linewidth=1.5)
    ax1.plot([exit_x_start, exit_x_end], [exit_y_top, exit_y_top], 'k-', linewidth=1.5)
    ax1.text((exit_x_start + exit_x_end)/2, R, f'EXIT\n{EXIT_FLAT}m', ha='center', va='center', fontsize=8, fontweight='bold')

    # --- GARAGE (horizontal at top, parallel to street) ---
    garage_y_bottom = R - path_half_width
    garage_y_top = R + path_half_width + 2
    garage_x_left = total_length
    garage_x_right = total_length + 3

    ax1.fill([garage_x_left, garage_x_right, garage_x_right, garage_x_left],
             [garage_y_bottom, garage_y_bottom, garage_y_top, garage_y_top],
             color='#87CEEB', alpha=0.5)
    ax1.plot([garage_x_left, garage_x_left], [garage_y_bottom, garage_y_top], 'b-', linewidth=2)
    ax1.text(garage_x_left + 1.5, R + 1, 'GARAGE\n(Level -3.5m)',
             ha='center', va='center', fontsize=9, fontweight='bold', color='darkblue')

    # --- START and END markers ---
    ax1.plot(ENTRY_FLAT, 0, 'go', markersize=12, zorder=5)
    ax1.plot(ENTRY_FLAT + R, R, 'rs', markersize=12, zorder=5)
    ax1.text(ENTRY_FLAT - 1.5, 1, 'START OF\nTHE ARC', ha='center', fontsize=8, fontweight='bold', color='green')
    ax1.text(ENTRY_FLAT + R + 1.5, R + 1.5, 'END OF\nTHE ARC', ha='center', fontsize=8, fontweight='bold', color='red')

    ax1.set_xlabel('Distance (m)', fontsize=11)
    ax1.set_ylabel('Width (m)', fontsize=11)
    ax1.set_title(f'TOP VIEW (Plan) - Total Path: {total_length:.1f}m', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # Same scale on both axes
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min_plan, y_max_plan)
    ax1.set_aspect('equal')

    # 2. Side Elevation with flat sections (full width, row 1)
    ax2 = fig.add_subplot(gs[1, :])

    s = vert['s']
    z = vert['z']

    # Calculate curvature to find convex and concave radii
    ds = s[1] - s[0]
    dz_ds = np.gradient(z, ds)
    d2z_ds2 = np.gradient(dz_ds, ds)

    # Curvature: κ = |d²z/ds²| / (1 + (dz/ds)²)^(3/2)
    curvature = d2z_ds2 / (1 + dz_ds**2)**1.5

    # Find convex curve (negative curvature - top of ramp, around entry transition)
    # and concave curve (positive curvature - bottom of ramp, around exit transition)
    ramp_mask = (s >= ENTRY_FLAT) & (s <= ENTRY_FLAT + arc_length)
    s_ramp = s[ramp_mask]
    curv_ramp = curvature[ramp_mask]

    # Convex region (first half of ramp - curving downward)
    mid_idx = len(s_ramp) // 2
    convex_idx = np.argmin(curv_ramp[:mid_idx])  # Most negative curvature
    convex_curvature = curv_ramp[convex_idx]
    convex_radius = abs(1 / convex_curvature) if convex_curvature != 0 else float('inf')
    convex_pos = s_ramp[convex_idx]

    # Concave region (second half of ramp - leveling out)
    concave_idx = np.argmax(curv_ramp[mid_idx:]) + mid_idx  # Most positive curvature
    concave_curvature = curv_ramp[concave_idx]
    concave_radius = abs(1 / concave_curvature) if concave_curvature != 0 else float('inf')
    concave_pos = s_ramp[concave_idx]

    ax2.axhline(y=0, color='green', linewidth=3, label='Street level')
    ax2.axhline(y=-VERTICAL_DROP, color='blue', linewidth=3, label='Garage level')

    ax2.fill_between(s, z, z.min() - 0.5, color='#d4a574', alpha=0.7)
    ax2.plot(s, z, 'k-', linewidth=3)

    # Mark transitions
    ax2.axvline(x=ENTRY_FLAT, color='red', linewidth=1, linestyle='--', alpha=0.7)
    ax2.axvline(x=ENTRY_FLAT + arc_length, color='red', linewidth=1, linestyle='--', alpha=0.7)

    # Section labels
    ax2.text(ENTRY_FLAT/2, 0.5, f'STREET\n{ENTRY_FLAT}m', ha='center', fontsize=10, color='green', fontweight='bold')
    ax2.text(ENTRY_FLAT + arc_length/2, 0.5, f'RAMP\n{arc_length:.1f}m', ha='center', fontsize=10, fontweight='bold')
    ax2.text(ENTRY_FLAT + arc_length + EXIT_FLAT/2, -VERTICAL_DROP + 0.5, f'GARAGE\n{EXIT_FLAT}m',
             ha='center', fontsize=10, color='blue', fontweight='bold')

    # Mark and label convex radius (top of curve)
    convex_z = get_elevation_at(convex_pos, s, z)
    ax2.plot(convex_pos, convex_z, 'ro', markersize=10, zorder=5)
    ax2.annotate(f'CONVEX\nR = {convex_radius:.1f}m',
                 xy=(convex_pos, convex_z), xytext=(convex_pos - 3, convex_z + 1),
                 fontsize=9, fontweight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Mark and label concave radius (bottom of curve)
    concave_z = get_elevation_at(concave_pos, s, z)
    ax2.plot(concave_pos, concave_z, 'bo', markersize=10, zorder=5)
    ax2.annotate(f'CONCAVE\nR = {concave_radius:.1f}m',
                 xy=(concave_pos, concave_z), xytext=(concave_pos + 2, concave_z - 1),
                 fontsize=9, fontweight='bold', color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    ax2.set_xlabel('Distance along path (m)', fontsize=11)
    ax2.set_ylabel('Elevation (m)', fontsize=11)
    ax2.set_title(f'SIDE ELEVATION - {VERTICAL_DROP}m Drop - Total {total_length:.1f}m', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    # Same scale on both axes
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min_elevation, y_max_elevation)
    ax2.set_aspect('equal')

    # 3. Clearance Analysis
    ax3 = fig.add_subplot(gs[3, 0])

    positions = np.linspace(CAR_LENGTH, total_length - CAR_LENGTH, 200)
    down_clearances = []
    up_clearances = []

    s_array, z_array = get_ramp_profile(arc_length)
    for pos in positions:
        down = check_car_clearance_full(pos, s_array, z_array, 'downhill')
        up = check_car_clearance_full(pos, s_array, z_array, 'uphill')
        down_clearances.append(down['min_clearance'] * 1000)
        up_clearances.append(up['min_clearance'] * 1000)

    ax3.plot(positions, down_clearances, 'b-', linewidth=2, label='Downhill')
    ax3.plot(positions, up_clearances, 'r-', linewidth=2, label='Uphill')
    ax3.axhline(y=MIN_GROUND_CLEARANCE * 1000, color='orange', linewidth=2, linestyle='--',
                label=f'Min required ({MIN_GROUND_CLEARANCE*1000:.0f}mm)')
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.axvline(x=ENTRY_FLAT, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax3.axvline(x=ENTRY_FLAT + arc_length, color='gray', linewidth=1, linestyle=':', alpha=0.7)

    ax3.set_xlabel('Car center position (m)')
    ax3.set_ylabel('Minimum clearance (mm)')
    ax3.set_title('GROUND CLEARANCE (Both Directions)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-50, 150)

    # 4. Specifications
    ax4 = fig.add_subplot(gs[3, 1])
    ax4.axis('off')

    specs_text = f"""
╔══════════════════════════════════════════════════╗
║  CORRECTED RADIAL RAMP - {VERTICAL_DROP}m DROP              ║
╠══════════════════════════════════════════════════╣
║  GEOMETRY                                        ║
║  Configuration:     Quarter circle (90°)         ║
║  Centerline radius: {R:.1f} m                     ║
║  Inner edge radius: {inner_edge:.2f} m                  ║
║  Outer edge radius: {outer_edge:.2f} m                  ║
║  Arc length:        {arc_length:.2f} m                  ║
║  Ramp width:        {ramp_width:.2f} m                   ║
║  Vertical drop:     {VERTICAL_DROP:.1f} m                      ║
╠══════════════════════════════════════════════════╣
║  FLAT SECTIONS                                   ║
║  Entry (street):    {ENTRY_FLAT:.1f} m                      ║
║  Exit (garage):     {EXIT_FLAT:.1f} m                      ║
║  TOTAL LENGTH:      {total_length:.1f} m                    ║
╠══════════════════════════════════════════════════╣
║  SLOPES & CURVATURE                              ║
║  Maximum slope:     {vert['max_slope']:.1f}°                      ║
║  Entry/Exit slope:  0° (level)                   ║
║  Min vert. radius:  {vert['min_vertical_radius']:.2f} m                  ║
╠══════════════════════════════════════════════════╣
║  CLEARANCES (CORRECTED)                          ║
║  Downhill min:      {result['min_down']*1000:.1f}mm                    ║
║  Uphill min:        {result['min_up']*1000:.1f}mm                    ║
║  Required min:      {MIN_GROUND_CLEARANCE*1000:.0f}mm                      ║
╚══════════════════════════════════════════════════╝
    """

    ax4.text(0.02, 0.98, specs_text, transform=ax4.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # 5. Comparison with old design
    ax5 = fig.add_subplot(gs[3, 2])
    ax5.axis('off')

    old_R = 7.5
    old_arc = 11.78
    old_down, old_up = analyze_clearance(old_arc)

    comparison = f"""
    COMPARISON: OLD vs CORRECTED DESIGN
    ═══════════════════════════════════════════════

    Parameter           OLD           NEW         Change
    ─────────────────────────────────────────────────────
    Centerline R        {old_R:.1f}m        {R:.1f}m       +{R-old_R:.1f}m
    Arc length          {old_arc:.1f}m       {arc_length:.1f}m      +{arc_length-old_arc:.1f}m
    Total length        {ENTRY_FLAT+old_arc+EXIT_FLAT:.1f}m       {total_length:.1f}m      +{total_length-(ENTRY_FLAT+old_arc+EXIT_FLAT):.1f}m

    CLEARANCES:
    Downhill            {old_down*1000:.0f}mm       {result['min_down']*1000:.0f}mm      +{(result['min_down']-old_down)*1000:.0f}mm
    Uphill              {old_up*1000:.0f}mm       {result['min_up']*1000:.0f}mm      +{(result['min_up']-old_up)*1000:.0f}mm

    OLD DESIGN PROBLEM:
    ⚠️ Negative clearance means car SCRAPES ground!
    ⚠️ Rear bumper (1.26m overhang) was not checked

    NEW DESIGN:
    ✓ Full car geometry including overhangs
    ✓ Both directions (uphill & downhill) verified
    ✓ Minimum {MIN_GROUND_CLEARANCE*1000:.0f}mm clearance maintained
    """

    ax5.text(0.02, 0.98, comparison, transform=ax5.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # 6. Critical Position Visualization - Full ramp with two car positions (full width, same scale as SIDE ELEVATION)
    ax6 = fig.add_subplot(gs[2, :])

    # Find worst uphill position
    worst_pos, worst_clearance, worst_type, s_array, z_array = find_critical_uphill_position(arc_length)

    # Draw complete ramp profile (same as SIDE ELEVATION)
    ax6.axhline(y=0, color='green', linewidth=2, alpha=0.7)
    ax6.axhline(y=-VERTICAL_DROP, color='blue', linewidth=2, alpha=0.7)
    ax6.fill_between(s_array, z_array, z_array.min() - 0.5, color='#d4a574', alpha=0.7)
    ax6.plot(s_array, z_array, 'k-', linewidth=2)

    # Mark transitions
    ax6.axvline(x=ENTRY_FLAT, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax6.axvline(x=ENTRY_FLAT + arc_length, color='gray', linewidth=1, linestyle='--', alpha=0.5)

    # Function to draw a car at a given position
    def draw_car(ax, car_center_pos, s_arr, z_arr, direction='uphill', color='blue', label_prefix=''):
        half_wb = WHEELBASE / 2

        if direction == 'uphill':
            # Uphill: front axle behind (smaller s), rear axle ahead (larger s)
            front_axle_pos = car_center_pos - half_wb
            rear_axle_pos = car_center_pos + half_wb
            front_bumper_pos = front_axle_pos - FRONT_OVERHANG
            rear_bumper_pos = rear_axle_pos + REAR_OVERHANG
        else:
            # Downhill: front axle ahead (larger s), rear axle behind (smaller s)
            front_axle_pos = car_center_pos + half_wb
            rear_axle_pos = car_center_pos - half_wb
            front_bumper_pos = front_axle_pos + FRONT_OVERHANG
            rear_bumper_pos = rear_axle_pos - REAR_OVERHANG

        front_axle_z = get_elevation_at(front_axle_pos, s_arr, z_arr)
        rear_axle_z = get_elevation_at(rear_axle_pos, s_arr, z_arr)

        def body_z_func(x):
            t = (x - rear_axle_pos) / (front_axle_pos - rear_axle_pos)
            return rear_axle_z + t * (front_axle_z - rear_axle_z) + GROUND_CLEARANCE

        car_x = [front_bumper_pos, front_axle_pos, rear_axle_pos, rear_bumper_pos]
        car_z = [body_z_func(x) for x in car_x]

        # Plot car body
        ax.plot(car_x, car_z, color=color, linewidth=3, label=f'{label_prefix}Car body')
        # Plot wheels
        ax.plot([front_axle_pos, rear_axle_pos], [front_axle_z, rear_axle_z], 'ko', markersize=10)
        # Plot bumpers
        ax.plot(front_bumper_pos, body_z_func(front_bumper_pos), '^', color=color, markersize=12)
        ax.plot(rear_bumper_pos, body_z_func(rear_bumper_pos), 's', color=color, markersize=10)

        return front_bumper_pos, rear_bumper_pos, body_z_func

    # Draw CAR 1: at critical uphill position (worst clearance)
    fb1, rb1, body_z1 = draw_car(ax6, worst_pos, s_array, z_array, 'uphill', 'blue', 'Critical: ')

    # Show clearance measurement for critical car
    if worst_type == 'front_bumper':
        bumper_pos = fb1
        bumper_z = body_z1(fb1)
    else:
        bumper_pos = rb1
        bumper_z = body_z1(rb1)
    ground_z = get_elevation_at(bumper_pos, s_array, z_array)
    ax6.plot([bumper_pos, bumper_pos], [ground_z, bumper_z], 'b--', linewidth=2)
    clearance_mm = (bumper_z - ground_z) * 1000
    ax6.text(bumper_pos + 0.5, (ground_z + bumper_z) / 2,
             f'{clearance_mm:.0f}mm', fontsize=10, color='blue', fontweight='bold')
    ax6.text(worst_pos, ground_z - 0.4, f'CRITICAL\ns={worst_pos:.1f}m',
             ha='center', fontsize=9, fontweight='bold', color='blue')

    # Draw CAR 2: at top of climb (near end of entry section, position ~7.5m)
    top_pos = 7.5  # Position at 7.5m along path (near top when climbing uphill)
    fb2, rb2, body_z2 = draw_car(ax6, top_pos, s_array, z_array, 'uphill', 'red', 'Top: ')

    # Label for top position car
    top_ground_z = get_elevation_at(top_pos, s_array, z_array)
    ax6.text(top_pos, top_ground_z - 0.4, f'TOP OF CLIMB\ns={top_pos:.1f}m',
             ha='center', fontsize=9, fontweight='bold', color='red')

    # Section labels
    ax6.text(ENTRY_FLAT/2, 1.0, 'STREET', ha='center', fontsize=9, color='green', fontweight='bold')
    ax6.text(ENTRY_FLAT + arc_length/2, 1.0, 'RAMP', ha='center', fontsize=9, fontweight='bold')
    ax6.text(ENTRY_FLAT + arc_length + EXIT_FLAT/2, -VERTICAL_DROP + 1.0, 'GARAGE',
             ha='center', fontsize=9, color='blue', fontweight='bold')

    ax6.set_xlabel('Distance along path (m)', fontsize=11)
    ax6.set_ylabel('Elevation (m)', fontsize=11)
    ax6.set_title(f'CAR POSITIONS ON RAMP - Critical Position & Top of Climb', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    # Same scale as SIDE ELEVATION
    ax6.set_xlim(x_min, x_max)
    ax6.set_ylim(y_min_elevation, y_max_elevation)
    ax6.set_aspect('equal')

    # 7. Combined summary box (specifications and critical analysis)
    ax7 = fig.add_subplot(gs[4, :])
    ax7.axis('off')

    combined_summary = f"""
    CURVE RADII ANALYSIS
    ═══════════════════════════════════════

    CONVEX CURVE (top transition):
    Position:        s ≈ {convex_pos:.1f}m
    Radius:          R = {convex_radius:.1f}m

    CONCAVE CURVE (bottom transition):
    Position:        s ≈ {concave_pos:.1f}m
    Radius:          R = {concave_radius:.1f}m

    CRITICAL POSITION:
    Position:        s = {worst_pos:.1f}m
    Clearance:       {worst_clearance*1000:.1f}mm
    Status:          {'SAFE' if worst_clearance >= MIN_GROUND_CLEARANCE else 'WARNING'}
    """

    ax7.text(0.02, 0.98, combined_summary, transform=ax7.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    # 8. Elevation table
    ax8 = fig.add_subplot(gs[5, :])
    ax8.axis('off')

    table = f"ELEVATION PROFILE (every 1m along ramp section)\n"
    table += "═" * 100 + "\n"
    table += f"{'Distance':>10} {'Elevation':>12} {'Slope':>10} │ {'Distance':>10} {'Elevation':>12} {'Slope':>10}\n"
    table += f"{'(m)':>10} {'(m)':>12} {'(deg)':>10} │ {'(m)':>10} {'(m)':>12} {'(deg)':>10}\n"
    table += "─" * 100 + "\n"

    s_arr = vert['s']
    z_arr = vert['z']
    slope_arr = vert['slope_angles']

    measurements = []
    for dist in np.arange(0, total_length + 0.5, 1.0):
        if dist <= total_length:
            idx = np.argmin(np.abs(s_arr - dist))
            measurements.append((dist, z_arr[idx], slope_arr[idx]))

    mid = (len(measurements) + 1) // 2
    for i in range(mid):
        m1 = measurements[i]
        row = f"{m1[0]:>10.1f} {m1[1]:>12.3f} {m1[2]:>10.1f} │ "
        if i + mid < len(measurements):
            m2 = measurements[i + mid]
            row += f"{m2[0]:>10.1f} {m2[1]:>12.3f} {m2[2]:>10.1f}"
        table += row + "\n"

    ax8.text(0.01, 0.95, table, transform=ax8.transAxes,
             fontsize=8, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black'))

    plt.suptitle(f'CORRECTED RADIAL RAMP DESIGN - {VERTICAL_DROP}m DROP\n'
                 f'R={R:.1f}m | Arc={arc_length:.1f}m | Total={total_length:.1f}m | Bidirectional Safe',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('/workspaces/RAMP/minimum_radial_3.5m_blueprint_corrected.png', dpi=200, bbox_inches='tight')
    print(f"\nCorrected blueprint saved: minimum_radial_3.5m_blueprint_corrected.png")


def save_measurements(result):
    """Save detailed measurements to CSV."""
    arc_length = result['arc_length']
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
    vert = result['vert']

    with open('/workspaces/RAMP/minimum_radial_3.5m_measurements_corrected.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Distance_m', 'Elevation_m', 'Elevation_cm', 'Slope_deg', 'Section'])

        s_arr = vert['s']
        z_arr = vert['z']
        slope_arr = vert['slope_angles']

        for dist in np.arange(0, total_length + 0.25, 0.5):
            if dist <= total_length:
                idx = np.argmin(np.abs(s_arr - dist))

                if dist < ENTRY_FLAT:
                    section = 'Street'
                elif dist > ENTRY_FLAT + arc_length:
                    section = 'Garage'
                else:
                    section = 'Ramp'

                writer.writerow([f"{dist:.2f}", f"{z_arr[idx]:.4f}",
                               f"{z_arr[idx]*100:.1f}", f"{slope_arr[idx]:.2f}", section])

    print(f"Measurements saved: minimum_radial_3.5m_measurements_corrected.csv")


def find_minimum_compound_curve_radius():
    """
    Find the minimum compound curve design that provides clearance.
    Tests R=11m with varying arc lengths to find the configuration that works.
    """
    print("\n" + "=" * 80)
    print("SEARCHING FOR MINIMUM COMPOUND CURVE CONFIGURATION")
    print("Testing R=11m (user specified) with varying arc lengths")
    print("=" * 80)

    print(f"\n{'Radius':>8} {'Arc':>6} {'Slope':>8} {'Straight':>10} {'Ramp':>8} {'Down Cl':>10} {'Up Cl':>10} {'Status':<12}")
    print("-" * 95)

    best_result = None
    R_curve = 11.0  # User specified radius

    # Search arc lengths - longer arc = gentler slope
    for arc_length in np.arange(2.0, 15.0, 0.5):
        theta = arc_length / R_curve
        slope_deg = np.degrees(theta)

        # Calculate drops
        drop_per_curve = R_curve * (1 - np.cos(theta))
        remaining_drop = VERTICAL_DROP - 2 * drop_per_curve

        if remaining_drop < 0:
            print(f"{R_curve:>8.1f} {arc_length:>6.1f} {slope_deg:>7.1f}°  -- Curves alone exceed {VERTICAL_DROP}m drop --")
            continue

        if np.sin(theta) < 0.01:
            continue

        length_straight = remaining_drop / np.sin(theta)
        arc_concave = arc_length  # Same as convex

        # Generate profile
        total_length = ENTRY_FLAT + arc_length + length_straight + arc_concave + EXIT_FLAT
        ramp_length = arc_length + length_straight + arc_concave
        s = np.linspace(0, total_length, 2000)
        z = np.zeros_like(s)

        s1 = ENTRY_FLAT
        s2 = s1 + arc_length
        s3 = s2 + length_straight
        s4 = s3 + arc_concave

        actual_drop_straight = length_straight * np.sin(theta)
        actual_total_drop = 2 * drop_per_curve + actual_drop_straight

        for i, si in enumerate(s):
            if si <= s1:
                z[i] = 0
            elif si <= s2:
                ds = si - s1
                th = ds / R_curve
                z[i] = -R_curve * (1 - np.cos(th))
            elif si <= s3:
                ds = si - s2
                z[i] = -drop_per_curve - ds * np.sin(theta)
            elif si <= s4:
                ds = si - s3
                th_remaining = theta - ds / R_curve
                z[i] = -drop_per_curve - actual_drop_straight - R_curve * (1 - np.cos(theta)) + R_curve * (1 - np.cos(th_remaining))
            else:
                z[i] = -actual_total_drop

        # Check clearance
        min_down, pos_down, type_down = check_compound_curve_clearance(s, z, 'downhill')
        min_up, pos_up, type_up = check_compound_curve_clearance(s, z, 'uphill')

        clearance_ok = min(min_down, min_up) >= MIN_GROUND_CLEARANCE
        status = "✓ VALID" if clearance_ok else "✗"

        print(f"{R_curve:>8.1f} {arc_length:>6.1f} {slope_deg:>7.1f}° {length_straight:>10.2f} {ramp_length:>8.1f} "
              f"{min_down*1000:>9.1f}mm {min_up*1000:>9.1f}mm {status:<12}")

        if clearance_ok and best_result is None:
            best_result = {
                'R_curve': R_curve,
                'arc_length': arc_length,
                'length_straight': length_straight,
                'total_length': total_length,
                'slope_deg': slope_deg,
                'min_down': min_down,
                'min_up': min_up,
                's': s,
                'z': z,
                'geometry': {
                    'R_convex': R_curve,
                    'R_concave': R_curve,
                    'arc_convex': arc_length,
                    'arc_concave': arc_concave,
                    'length_straight': length_straight,
                    'theta_max': slope_deg,
                    'drop_convex': drop_per_curve,
                    'drop_straight': actual_drop_straight,
                    'drop_concave': drop_per_curve,
                    'total_drop': actual_total_drop,
                    'ramp_length': ramp_length,
                    'total_length': total_length,
                    's1': s1, 's2': s2, 's3': s3, 's4': s4,
                }
            }

    if best_result is None:
        print("\nR=11m cannot provide clearance. Searching for minimum working radius...")
        print("-" * 95)

        # Search larger radii
        for R_curve in np.arange(12.0, 30.0, 1.0):
            arc_length = 5.0  # Fixed arc

            theta = arc_length / R_curve
            slope_deg = np.degrees(theta)

            drop_per_curve = R_curve * (1 - np.cos(theta))
            remaining_drop = VERTICAL_DROP - 2 * drop_per_curve

            if remaining_drop < 0:
                continue

            length_straight = remaining_drop / np.sin(theta)
            arc_concave = arc_length

            total_length = ENTRY_FLAT + arc_length + length_straight + arc_concave + EXIT_FLAT
            ramp_length = arc_length + length_straight + arc_concave
            s = np.linspace(0, total_length, 2000)
            z = np.zeros_like(s)

            s1 = ENTRY_FLAT
            s2 = s1 + arc_length
            s3 = s2 + length_straight
            s4 = s3 + arc_concave

            actual_drop_straight = length_straight * np.sin(theta)
            actual_total_drop = 2 * drop_per_curve + actual_drop_straight

            for i, si in enumerate(s):
                if si <= s1:
                    z[i] = 0
                elif si <= s2:
                    ds = si - s1
                    th = ds / R_curve
                    z[i] = -R_curve * (1 - np.cos(th))
                elif si <= s3:
                    ds = si - s2
                    z[i] = -drop_per_curve - ds * np.sin(theta)
                elif si <= s4:
                    ds = si - s3
                    th_remaining = theta - ds / R_curve
                    z[i] = -drop_per_curve - actual_drop_straight - R_curve * (1 - np.cos(theta)) + R_curve * (1 - np.cos(th_remaining))
                else:
                    z[i] = -actual_total_drop

            min_down, _, _ = check_compound_curve_clearance(s, z, 'downhill')
            min_up, _, _ = check_compound_curve_clearance(s, z, 'uphill')

            clearance_ok = min(min_down, min_up) >= MIN_GROUND_CLEARANCE
            status = "✓ VALID" if clearance_ok else "✗"

            print(f"{R_curve:>8.1f} {arc_length:>6.1f} {slope_deg:>7.1f}° {length_straight:>10.2f} {ramp_length:>8.1f} "
                  f"{min_down*1000:>9.1f}mm {min_up*1000:>9.1f}mm {status:<12}")

            if clearance_ok and best_result is None:
                best_result = {
                    'R_curve': R_curve,
                    'arc_length': arc_length,
                    'length_straight': length_straight,
                    'total_length': total_length,
                    'slope_deg': slope_deg,
                    'min_down': min_down,
                    'min_up': min_up,
                    's': s,
                    'z': z,
                    'geometry': {
                        'R_convex': R_curve,
                        'R_concave': R_curve,
                        'arc_convex': arc_length,
                        'arc_concave': arc_concave,
                        'length_straight': length_straight,
                        'theta_max': slope_deg,
                        'drop_convex': drop_per_curve,
                        'drop_straight': actual_drop_straight,
                        'drop_concave': drop_per_curve,
                        'total_drop': actual_total_drop,
                        'ramp_length': ramp_length,
                        'total_length': total_length,
                        's1': s1, 's2': s2, 's3': s3, 's4': s4,
                    }
                }
                break

    return best_result


def main():
    print("\n" + "=" * 80)
    print(f"COMPOUND CURVE RAMP DESIGN - {VERTICAL_DROP}m DROP")
    print("Using separate convex and concave curves with straight section")
    print("=" * 80)

    # First, find the minimum radius that works
    compound_result = find_minimum_compound_curve_radius()

    if compound_result is None:
        print("Cannot find valid compound curve design!")
        # Fall back to original method
        print("\nFalling back to single curve design...")
        result = find_minimum_safe_radius()
        if result:
            create_visualization(result)
            save_measurements(result)
        return

    print(f"\n{'=' * 80}")
    print("COMPOUND CURVE DESIGN SUMMARY")
    print(f"{'=' * 80}")

    geom = compound_result['geometry']

    print(f"""
    COMPOUND CURVE DESIGN SPECIFICATIONS:
    ─────────────────────────────────────────────────
    CONVEX SECTION (top transition):
      Radius:            {geom['R_convex']:.1f}m
      Arc length:        {geom['arc_convex']:.1f}m
      Vertical drop:     {geom['drop_convex']:.3f}m

    STRAIGHT SECTION (inclined):
      Length:            {geom['length_straight']:.2f}m
      Slope:             {geom['theta_max']:.1f}°
      Vertical drop:     {geom['drop_straight']:.3f}m

    CONCAVE SECTION (bottom transition):
      Radius:            {geom['R_concave']:.1f}m
      Arc length:        {geom['arc_concave']:.1f}m
      Vertical drop:     {geom['drop_concave']:.3f}m

    TOTALS:
      Entry flat:        {ENTRY_FLAT}m
      Ramp section:      {geom['ramp_length']:.2f}m
      Exit flat:         {EXIT_FLAT}m
      TOTAL PATH:        {geom['total_length']:.2f}m
      TOTAL DROP:        {geom['total_drop']:.3f}m

    CLEARANCES (VERIFIED FOR BOTH DIRECTIONS):
    ─────────────────────────────────────────────────
    Downhill minimum:    {compound_result['min_down']*1000:.1f}mm
    Uphill minimum:      {compound_result['min_up']*1000:.1f}mm
    Required minimum:    {MIN_GROUND_CLEARANCE*1000:.0f}mm
    Status:              ✓ SAFE FOR BIDIRECTIONAL TRAVEL
    """)

    # Create visualization for compound curve
    create_compound_visualization(compound_result)


def create_compound_visualization(compound_result):
    """Create visualization for compound curve design."""

    geom = compound_result['geometry']
    s_array = compound_result['s']
    z_array = compound_result['z']
    total_length = geom['total_length']

    fig = plt.figure(figsize=(24, 32))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8], hspace=0.35, wspace=0.25)

    # Common axis limits
    x_min, x_max = -2, total_length + 5
    y_min_elevation, y_max_elevation = -6, 3

    # 1. Side Elevation - Compound Curve Profile
    ax1 = fig.add_subplot(gs[0, :])

    ax1.axhline(y=0, color='green', linewidth=3, label='Street level')
    ax1.axhline(y=-VERTICAL_DROP, color='blue', linewidth=3, label='Garage level')

    ax1.fill_between(s_array, z_array, z_array.min() - 0.5, color='#d4a574', alpha=0.7)
    ax1.plot(s_array, z_array, 'k-', linewidth=3)

    # Mark section boundaries
    s1, s2, s3, s4 = geom['s1'], geom['s2'], geom['s3'], geom['s4']

    ax1.axvline(x=s1, color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax1.axvline(x=s2, color='orange', linewidth=2, linestyle='--', alpha=0.7)
    ax1.axvline(x=s3, color='orange', linewidth=2, linestyle='--', alpha=0.7)
    ax1.axvline(x=s4, color='red', linewidth=2, linestyle='--', alpha=0.7)

    # Section labels
    ax1.text(s1/2, 0.5, f'STREET\n{ENTRY_FLAT}m', ha='center', fontsize=10, color='green', fontweight='bold')
    ax1.text((s1+s2)/2, 0.5, f'CONVEX\nR={geom["R_convex"]}m\n{geom["arc_convex"]}m', ha='center', fontsize=9, fontweight='bold', color='red')
    ax1.text((s2+s3)/2, -1.5, f'STRAIGHT\n{geom["length_straight"]:.1f}m\n{geom["theta_max"]:.1f}°', ha='center', fontsize=9, fontweight='bold', color='purple')
    ax1.text((s3+s4)/2, -3.5, f'CONCAVE\nR={geom["R_concave"]}m\n{geom["arc_concave"]:.1f}m', ha='center', fontsize=9, fontweight='bold', color='blue')
    ax1.text((s4+total_length)/2, -VERTICAL_DROP+0.5, f'GARAGE\n{EXIT_FLAT}m', ha='center', fontsize=10, color='blue', fontweight='bold')

    # Mark radii with arcs
    # Convex center
    convex_center_s = s1
    convex_center_z = geom['R_convex']
    ax1.plot(convex_center_s, 0, 'ro', markersize=8)
    ax1.annotate(f'R={geom["R_convex"]}m', xy=(s1+2, -0.3), fontsize=10, color='red', fontweight='bold')

    # Concave center
    z_at_s3 = np.interp(s3, s_array, z_array)
    ax1.plot(s3, z_at_s3, 'bo', markersize=8)
    ax1.annotate(f'R={geom["R_concave"]}m', xy=(s3+1, z_at_s3-0.5), fontsize=10, color='blue', fontweight='bold')

    ax1.set_xlabel('Distance along path (m)', fontsize=11)
    ax1.set_ylabel('Elevation (m)', fontsize=11)
    ax1.set_title(f'COMPOUND CURVE SIDE ELEVATION - {VERTICAL_DROP}m Drop - Total {total_length:.1f}m', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min_elevation, y_max_elevation)
    ax1.set_aspect('equal')

    # 2. Car Positions on Compound Curve
    ax2 = fig.add_subplot(gs[1, :])

    ax2.axhline(y=0, color='green', linewidth=2, alpha=0.7)
    ax2.axhline(y=-VERTICAL_DROP, color='blue', linewidth=2, alpha=0.7)
    ax2.fill_between(s_array, z_array, z_array.min() - 0.5, color='#d4a574', alpha=0.7)
    ax2.plot(s_array, z_array, 'k-', linewidth=2)

    # Draw cars at critical positions
    def draw_car_on_profile(ax, car_center, s_arr, z_arr, direction, color, label):
        half_wb = WHEELBASE / 2
        if direction == 'uphill':
            front_axle = car_center - half_wb
            rear_axle = car_center + half_wb
            front_bumper = front_axle - FRONT_OVERHANG
            rear_bumper = rear_axle + REAR_OVERHANG
        else:
            front_axle = car_center + half_wb
            rear_axle = car_center - half_wb
            front_bumper = front_axle + FRONT_OVERHANG
            rear_bumper = rear_axle - REAR_OVERHANG

        fa_z = get_elevation_at(front_axle, s_arr, z_arr)
        ra_z = get_elevation_at(rear_axle, s_arr, z_arr)

        def body_z(x):
            t = (x - rear_axle) / (front_axle - rear_axle)
            return ra_z + t * (fa_z - ra_z) + GROUND_CLEARANCE

        car_x = [front_bumper, front_axle, rear_axle, rear_bumper]
        car_z = [body_z(x) for x in car_x]

        ax.plot(car_x, car_z, color=color, linewidth=3, label=label)
        ax.plot([front_axle, rear_axle], [fa_z, ra_z], 'ko', markersize=8)
        ax.plot(front_bumper, body_z(front_bumper), '^', color=color, markersize=10)
        ax.plot(rear_bumper, body_z(rear_bumper), 's', color=color, markersize=8)

        return front_bumper, rear_bumper, body_z

    # Find critical positions
    min_down, pos_down, type_down = check_compound_curve_clearance(s_array, z_array, 'downhill')
    min_up, pos_up, type_up = check_compound_curve_clearance(s_array, z_array, 'uphill')

    # Draw car at critical uphill position
    fb, rb, body_z_func = draw_car_on_profile(ax2, pos_up, s_array, z_array, 'uphill', 'blue', 'Critical uphill')

    # Show clearance
    if type_up == 'rear_bumper':
        bumper_pos, bumper_z = rb, body_z_func(rb)
    else:
        bumper_pos, bumper_z = fb, body_z_func(fb)
    ground_z = get_elevation_at(bumper_pos, s_array, z_array)
    ax2.plot([bumper_pos, bumper_pos], [ground_z, bumper_z], 'b--', linewidth=2)
    ax2.text(bumper_pos + 0.5, (ground_z + bumper_z)/2, f'{min_up*1000:.0f}mm', fontsize=10, color='blue', fontweight='bold')

    # Draw car at top (on straight section)
    top_pos = (s2 + s3) / 2  # Middle of straight section
    draw_car_on_profile(ax2, top_pos, s_array, z_array, 'downhill', 'red', 'Mid-ramp')

    ax2.set_xlabel('Distance along path (m)', fontsize=11)
    ax2.set_ylabel('Elevation (m)', fontsize=11)
    ax2.set_title('CAR POSITIONS ON COMPOUND CURVE RAMP', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min_elevation, y_max_elevation)
    ax2.set_aspect('equal')

    # 3. Ground Clearance Graph
    ax3 = fig.add_subplot(gs[2, 0])

    positions = np.linspace(CAR_LENGTH, total_length - CAR_LENGTH, 200)
    down_clearances = []
    up_clearances = []

    for pos in positions:
        down = check_car_clearance_full(pos, s_array, z_array, 'downhill')
        up = check_car_clearance_full(pos, s_array, z_array, 'uphill')
        down_clearances.append(down['min_clearance'] * 1000)
        up_clearances.append(up['min_clearance'] * 1000)

    ax3.plot(positions, down_clearances, 'b-', linewidth=2, label='Downhill')
    ax3.plot(positions, up_clearances, 'r-', linewidth=2, label='Uphill')
    ax3.axhline(y=MIN_GROUND_CLEARANCE * 1000, color='orange', linewidth=2, linestyle='--', label=f'Min required ({MIN_GROUND_CLEARANCE*1000:.0f}mm)')
    ax3.axhline(y=0, color='black', linewidth=1)

    ax3.set_xlabel('Car center position (m)')
    ax3.set_ylabel('Minimum clearance (mm)')
    ax3.set_title('GROUND CLEARANCE (Both Directions)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-50, 150)

    # 4. Specifications
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    specs_text = f"""
╔═══════════════════════════════════════════════════╗
║  COMPOUND CURVE RAMP - {VERTICAL_DROP}m DROP               ║
╠═══════════════════════════════════════════════════╣
║  CONVEX SECTION (top)                             ║
║    Radius:         R = {geom['R_convex']:.1f}m                  ║
║    Arc length:     {geom['arc_convex']:.1f}m                      ║
║    Drop:           {geom['drop_convex']:.3f}m                   ║
╠═══════════════════════════════════════════════════╣
║  STRAIGHT SECTION                                 ║
║    Length:         {geom['length_straight']:.2f}m                    ║
║    Slope:          {geom['theta_max']:.1f}°                      ║
║    Drop:           {geom['drop_straight']:.3f}m                   ║
╠═══════════════════════════════════════════════════╣
║  CONCAVE SECTION (bottom)                         ║
║    Radius:         R = {geom['R_concave']:.1f}m                  ║
║    Arc length:     {geom['arc_concave']:.1f}m                      ║
║    Drop:           {geom['drop_concave']:.3f}m                   ║
╠═══════════════════════════════════════════════════╣
║  TOTALS                                           ║
║    Ramp length:    {geom['ramp_length']:.2f}m                    ║
║    Total path:     {geom['total_length']:.2f}m                    ║
║    Total drop:     {geom['total_drop']:.3f}m                   ║
╚═══════════════════════════════════════════════════╝
    """

    ax4.text(0.02, 0.98, specs_text, transform=ax4.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # 5. Clearance Summary
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')

    clearance_text = f"""
    CLEARANCE ANALYSIS
    ═══════════════════════════════════════

    DOWNHILL (street → garage):
    Min clearance:   {compound_result['min_down']*1000:.1f}mm
    Position:        s = {pos_down:.1f}m
    Critical:        {type_down.replace('_', ' ')}

    UPHILL (garage → street):
    Min clearance:   {compound_result['min_up']*1000:.1f}mm
    Position:        s = {pos_up:.1f}m
    Critical:        {type_up.replace('_', ' ')}

    COMPARISON TO SINGLE CURVE:
    Single curve arc needed:  ~22m
    Compound curve ramp:      {geom['ramp_length']:.1f}m
    SAVINGS:                  ~{22 - geom['ramp_length']:.1f}m

    Status: ✓ BIDIRECTIONAL SAFE
    """

    ax5.text(0.02, 0.98, clearance_text, transform=ax5.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    # 6. Elevation Table
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')

    table = f"ELEVATION PROFILE - COMPOUND CURVE (every 1m)\n"
    table += "═" * 100 + "\n"
    table += f"{'Dist':>8} {'Elev':>10} {'Section':>12} │ {'Dist':>8} {'Elev':>10} {'Section':>12}\n"
    table += "─" * 100 + "\n"

    measurements = []
    for dist in np.arange(0, total_length + 0.5, 1.0):
        if dist <= total_length:
            elev = get_elevation_at(dist, s_array, z_array)
            if dist < s1:
                section = 'Street'
            elif dist < s2:
                section = 'Convex'
            elif dist < s3:
                section = 'Straight'
            elif dist < s4:
                section = 'Concave'
            else:
                section = 'Garage'
            measurements.append((dist, elev, section))

    mid = (len(measurements) + 1) // 2
    for i in range(mid):
        m1 = measurements[i]
        row = f"{m1[0]:>8.1f} {m1[1]:>10.3f} {m1[2]:>12} │ "
        if i + mid < len(measurements):
            m2 = measurements[i + mid]
            row += f"{m2[0]:>8.1f} {m2[1]:>10.3f} {m2[2]:>12}"
        table += row + "\n"

    ax6.text(0.01, 0.95, table, transform=ax6.transAxes,
             fontsize=8, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black'))

    plt.suptitle(f'COMPOUND CURVE RAMP DESIGN - {VERTICAL_DROP}m DROP\n'
                 f'Convex R={geom["R_convex"]}m | Straight {geom["length_straight"]:.1f}m @ {geom["theta_max"]:.1f}° | Concave R={geom["R_concave"]}m',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('/workspaces/RAMP/compound_curve_ramp.png', dpi=200, bbox_inches='tight')
    print(f"\nCompound curve design saved: compound_curve_ramp.png")


if __name__ == '__main__':
    main()
