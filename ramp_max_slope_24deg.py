#!/usr/bin/env python3
"""
Maximum Slope Ramp Design - Target 24°

The cubic profile concentrates curvature at transition points, causing scraping.
This script explores different profile types to achieve steeper slopes:

1. Cubic profile (current) - smooth but tight curvature
2. Linear with parabolic transitions - constant slope in middle
3. Circular arc segments - distributed curvature
4. Quintic profile - more control over curvature distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# Car specifications
WHEELBASE = 2.350
GROUND_CLEARANCE = 0.106
CAR_LENGTH = 4.461
FRONT_OVERHANG = 0.85
REAR_OVERHANG = CAR_LENGTH - WHEELBASE - FRONT_OVERHANG  # 1.261m

# Ramp parameters
VERTICAL_DROP = 3.5
ENTRY_FLAT = 5.0
EXIT_FLAT = 5.0

# Target
TARGET_SLOPE = 24.0  # degrees
MIN_CLEARANCE = 0.005  # 5mm


def get_profile_linear_with_transitions(arc_length, transition_length, num_points=2000):
    """
    Profile with constant slope in middle and parabolic transitions at ends.

    Structure:
    - Entry transition: parabolic from 0° to max slope
    - Middle: constant slope
    - Exit transition: parabolic from max slope to 0°
    """
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)

    # Calculate max slope needed for given arc length and transition length
    # Middle section length
    middle_length = arc_length - 2 * transition_length
    if middle_length < 0:
        middle_length = 0
        transition_length = arc_length / 2

    # For parabolic transition: z = as², dz/ds = 2as
    # At end of transition, slope = 2a*transition_length = max_slope
    # And z = a * transition_length² = transition_drop

    # Total drop = 2 * transition_drop + middle_drop
    # transition_drop = (1/2) * max_slope * transition_length (area under slope curve)
    # middle_drop = max_slope * middle_length

    # VERTICAL_DROP = max_slope * transition_length + max_slope * middle_length + max_slope * transition_length
    # VERTICAL_DROP = max_slope * (transition_length + middle_length + transition_length)
    # VERTICAL_DROP = max_slope * arc_length
    # max_slope = VERTICAL_DROP / arc_length

    # But with parabolic transitions, it's different:
    # transition_drop = integral of slope from 0 to transition_length where slope goes from 0 to max_slope
    # For linear slope increase: transition_drop = (1/2) * max_slope * transition_length
    # Total: VERTICAL_DROP = 2 * (1/2) * max_slope * transition_length + max_slope * middle_length
    #      = max_slope * (transition_length + middle_length)
    #      = max_slope * (arc_length - transition_length)

    max_slope_tan = VERTICAL_DROP / (arc_length - transition_length) if arc_length > transition_length else VERTICAL_DROP / arc_length

    ramp_start = ENTRY_FLAT
    ramp_end = ENTRY_FLAT + arc_length
    trans1_end = ramp_start + transition_length
    trans2_start = ramp_end - transition_length

    for i, pos in enumerate(s):
        if pos < ramp_start:
            # Entry flat
            z[i] = 0
        elif pos < trans1_end:
            # Entry transition (parabolic)
            local_s = pos - ramp_start
            # Parabolic: z = -a*s², where slope at end = -2*a*transition_length = -max_slope_tan
            a = max_slope_tan / (2 * transition_length)
            z[i] = -a * local_s**2
        elif pos < trans2_start:
            # Middle section (constant slope)
            local_s = pos - trans1_end
            z_at_trans1_end = -max_slope_tan * transition_length / 2
            z[i] = z_at_trans1_end - max_slope_tan * local_s
        elif pos < ramp_end:
            # Exit transition (parabolic)
            local_s = pos - trans2_start
            remaining = transition_length - local_s
            z_at_trans2_start = -VERTICAL_DROP + max_slope_tan * transition_length / 2
            # Parabolic deceleration
            a = max_slope_tan / (2 * transition_length)
            z[i] = z_at_trans2_start - max_slope_tan * local_s + a * local_s**2
        else:
            # Exit flat
            z[i] = -VERTICAL_DROP

    return s, z, np.degrees(np.arctan(max_slope_tan))


def get_profile_quintic(arc_length, num_points=2000):
    """
    Quintic polynomial profile - allows control of both slope AND curvature at endpoints.
    z(s) = as⁵ + bs⁴ + cs³ + ds² + es + f

    Boundary conditions:
    - z(0) = 0, z(L) = -H
    - z'(0) = 0, z'(L) = 0  (zero slope at ends)
    - z''(0) = 0, z''(L) = 0  (zero curvature at ends - KEY!)
    """
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)

    L = arc_length
    H = VERTICAL_DROP

    # Quintic with zero slope AND zero curvature at endpoints
    # z = a5*s^5 + a4*s^4 + a3*s^3
    # Conditions: z(0)=0, z(L)=-H, z'(0)=0, z'(L)=0, z''(0)=0, z''(L)=0
    # This gives: a5 = 6H/L^5, a4 = -15H/L^4, a3 = 10H/L^3

    a5 = 6 * H / L**5
    a4 = -15 * H / L**4
    a3 = 10 * H / L**3

    ramp_mask = (s >= ENTRY_FLAT) & (s <= ENTRY_FLAT + arc_length)
    s_ramp = s[ramp_mask] - ENTRY_FLAT
    z[ramp_mask] = a5 * s_ramp**5 + a4 * s_ramp**4 + a3 * s_ramp**3

    exit_mask = s > ENTRY_FLAT + arc_length
    z[exit_mask] = -VERTICAL_DROP

    # Calculate max slope
    # z' = 5*a5*s^4 + 4*a4*s^3 + 3*a3*s^2
    # Max at s = L/2
    s_mid = L / 2
    max_slope_tan = abs(5*a5*s_mid**4 + 4*a4*s_mid**3 + 3*a3*s_mid**2)

    return s, z, np.degrees(np.arctan(max_slope_tan))


def get_profile_cubic(arc_length, num_points=2000):
    """Original cubic profile for comparison."""
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)

    L = arc_length
    H = VERTICAL_DROP
    a = 2 * H / L**3
    b = -3 * H / L**2

    ramp_mask = (s >= ENTRY_FLAT) & (s <= ENTRY_FLAT + arc_length)
    s_ramp = s[ramp_mask] - ENTRY_FLAT
    z[ramp_mask] = a * s_ramp**3 + b * s_ramp**2

    exit_mask = s > ENTRY_FLAT + arc_length
    z[exit_mask] = -VERTICAL_DROP

    # Max slope at s = L/2
    s_mid = L / 2
    max_slope_tan = abs(3*a*s_mid**2 + 2*b*s_mid)

    return s, z, np.degrees(np.arctan(max_slope_tan))


def get_elevation_at(s_query, s_array, z_array):
    return np.interp(s_query, s_array, z_array)


def check_car_clearance(car_center_pos, s_array, z_array, direction='downhill'):
    """Check clearance for entire car including overhangs."""
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

    front_clearance = car_body_height(front_bumper_pos) - get_elevation_at(front_bumper_pos, s_array, z_array)
    rear_clearance = car_body_height(rear_bumper_pos) - get_elevation_at(rear_bumper_pos, s_array, z_array)

    return min(front_clearance, rear_clearance)


def analyze_profile(s_array, z_array):
    """Analyze minimum clearance for a profile."""
    total_length = s_array[-1]
    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 500)

    min_down = float('inf')
    min_up = float('inf')

    for pos in positions:
        down_cl = check_car_clearance(pos, s_array, z_array, 'downhill')
        up_cl = check_car_clearance(pos, s_array, z_array, 'uphill')
        min_down = min(min_down, down_cl)
        min_up = min(min_up, up_cl)

    return min_down, min_up


def find_minimum_for_target_slope():
    """Find the minimum arc length that achieves target slope while maintaining clearance."""

    print("=" * 90)
    print(f"FINDING MINIMUM RAMP FOR {TARGET_SLOPE}° MAXIMUM SLOPE")
    print("=" * 90)

    # For cubic profile, arc_length for given slope:
    # max_slope = 3H/(2L) => L = 3H/(2*tan(slope))
    target_arc_cubic = 3 * VERTICAL_DROP / (2 * np.tan(np.radians(TARGET_SLOPE)))
    print(f"\nFor {TARGET_SLOPE}° slope with CUBIC profile: arc = {target_arc_cubic:.2f}m")

    # For quintic profile, the relationship is different
    # max_slope = 15H/(8L) => L = 15H/(8*tan(slope))
    target_arc_quintic = 15 * VERTICAL_DROP / (8 * np.tan(np.radians(TARGET_SLOPE)))
    print(f"For {TARGET_SLOPE}° slope with QUINTIC profile: arc = {target_arc_quintic:.2f}m")

    print("\n" + "=" * 90)
    print("TESTING DIFFERENT PROFILE TYPES")
    print("=" * 90)

    results = []

    # Test cubic profile at various arc lengths
    print(f"\n--- CUBIC Profile ---")
    print(f"{'Arc':>8} {'Slope':>8} {'Down Cl':>10} {'Up Cl':>10} {'Status':<10}")
    for arc in np.arange(10.0, 25.0, 0.5):
        s, z, slope = get_profile_cubic(arc)
        min_down, min_up = analyze_profile(s, z)
        overall = min(min_down, min_up)
        status = "OK" if overall >= MIN_CLEARANCE else "SCRAPE"
        results.append(('cubic', arc, slope, min_down, min_up, overall >= MIN_CLEARANCE))
        if 20 <= slope <= 28:  # Only show relevant range
            print(f"{arc:>8.1f} {slope:>8.1f}° {min_down*1000:>8.1f}mm {min_up*1000:>8.1f}mm {status:<10}")

    # Test quintic profile
    print(f"\n--- QUINTIC Profile (zero curvature at ends) ---")
    print(f"{'Arc':>8} {'Slope':>8} {'Down Cl':>10} {'Up Cl':>10} {'Status':<10}")
    for arc in np.arange(8.0, 20.0, 0.5):
        s, z, slope = get_profile_quintic(arc)
        min_down, min_up = analyze_profile(s, z)
        overall = min(min_down, min_up)
        status = "OK" if overall >= MIN_CLEARANCE else "SCRAPE"
        results.append(('quintic', arc, slope, min_down, min_up, overall >= MIN_CLEARANCE))
        if 20 <= slope <= 28:
            print(f"{arc:>8.1f} {slope:>8.1f}° {min_down*1000:>8.1f}mm {min_up*1000:>8.1f}mm {status:<10}")

    # Test linear with transitions
    print(f"\n--- LINEAR with PARABOLIC Transitions ---")
    print(f"{'Arc':>8} {'Trans':>6} {'Slope':>8} {'Down Cl':>10} {'Up Cl':>10} {'Status':<10}")
    for arc in np.arange(8.0, 18.0, 0.5):
        for trans in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            if trans * 2 < arc:  # Valid configuration
                s, z, slope = get_profile_linear_with_transitions(arc, trans)
                min_down, min_up = analyze_profile(s, z)
                overall = min(min_down, min_up)
                status = "OK" if overall >= MIN_CLEARANCE else "SCRAPE"
                results.append(('linear_trans', arc, slope, min_down, min_up, overall >= MIN_CLEARANCE, trans))
                if 22 <= slope <= 26 and overall >= MIN_CLEARANCE:
                    print(f"{arc:>8.1f} {trans:>6.1f} {slope:>8.1f}° {min_down*1000:>8.1f}mm {min_up*1000:>8.1f}mm {status:<10}")

    # Find best results for each profile type near target slope
    print("\n" + "=" * 90)
    print(f"BEST CONFIGURATIONS NEAR {TARGET_SLOPE}° SLOPE")
    print("=" * 90)

    best_results = []

    # Best cubic
    cubic_valid = [(r[1], r[2], r[3], r[4]) for r in results if r[0] == 'cubic' and r[5] and r[2] >= TARGET_SLOPE - 2]
    if cubic_valid:
        best_cubic = min(cubic_valid, key=lambda x: x[0])
        print(f"\nCUBIC: Arc={best_cubic[0]:.1f}m, Slope={best_cubic[1]:.1f}°, Down={best_cubic[2]*1000:.1f}mm, Up={best_cubic[3]*1000:.1f}mm")
        best_results.append(('cubic', best_cubic[0], best_cubic[1], min(best_cubic[2], best_cubic[3])))

    # Best quintic
    quintic_valid = [(r[1], r[2], r[3], r[4]) for r in results if r[0] == 'quintic' and r[5] and r[2] >= TARGET_SLOPE - 2]
    if quintic_valid:
        best_quintic = min(quintic_valid, key=lambda x: x[0])
        print(f"QUINTIC: Arc={best_quintic[0]:.1f}m, Slope={best_quintic[1]:.1f}°, Down={best_quintic[2]*1000:.1f}mm, Up={best_quintic[3]*1000:.1f}mm")
        best_results.append(('quintic', best_quintic[0], best_quintic[1], min(best_quintic[2], best_quintic[3])))

    # Best linear with transitions
    linear_valid = [(r[1], r[2], r[3], r[4], r[6]) for r in results if r[0] == 'linear_trans' and r[5] and r[2] >= TARGET_SLOPE - 2]
    if linear_valid:
        best_linear = min(linear_valid, key=lambda x: x[0])
        print(f"LINEAR+TRANS: Arc={best_linear[0]:.1f}m, Trans={best_linear[4]:.1f}m, Slope={best_linear[1]:.1f}°, Down={best_linear[2]*1000:.1f}mm, Up={best_linear[3]*1000:.1f}mm")
        best_results.append(('linear_trans', best_linear[0], best_linear[1], min(best_linear[2], best_linear[3]), best_linear[4]))

    return best_results


def create_comparison_blueprint(best_results):
    """Create blueprint comparing different profile types."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Find the best overall result
    if not best_results:
        print("No valid results found!")
        return

    # Use the shortest valid configuration
    best = min(best_results, key=lambda x: x[1])
    profile_type = best[0]
    arc_length = best[1]
    max_slope = best[2]

    if profile_type == 'cubic':
        s, z, _ = get_profile_cubic(arc_length)
    elif profile_type == 'quintic':
        s, z, _ = get_profile_quintic(arc_length)
    else:  # linear_trans
        trans = best[4]
        s, z, _ = get_profile_linear_with_transitions(arc_length, trans)

    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    # 1. Side Elevation
    ax1 = axes[0, 0]
    ax1.fill_between(s, z, z.min() - 0.5, color='#d4a574', alpha=0.7)
    ax1.plot(s, z, 'k-', linewidth=3, label=f'{profile_type.upper()} profile')
    ax1.axhline(y=0, color='green', linewidth=2, linestyle='--')
    ax1.axhline(y=-VERTICAL_DROP, color='blue', linewidth=2, linestyle='--')
    ax1.axvline(x=ENTRY_FLAT, color='red', linewidth=1, linestyle=':')
    ax1.axvline(x=ENTRY_FLAT + arc_length, color='red', linewidth=1, linestyle=':')

    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_title(f'SIDE ELEVATION\nArc={arc_length:.1f}m | Total={total_length:.1f}m | Slope={max_slope:.1f}°')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Profile comparison
    ax2 = axes[0, 1]

    # Plot all three profile types at same arc length for comparison
    s_cubic, z_cubic, slope_cubic = get_profile_cubic(arc_length)
    s_quintic, z_quintic, slope_quintic = get_profile_quintic(arc_length)

    ax2.plot(s_cubic, z_cubic, 'b-', linewidth=2, label=f'Cubic ({slope_cubic:.1f}°)')
    ax2.plot(s_quintic, z_quintic, 'r-', linewidth=2, label=f'Quintic ({slope_quintic:.1f}°)')
    if profile_type == 'linear_trans':
        ax2.plot(s, z, 'g-', linewidth=2, label=f'Linear+Trans ({max_slope:.1f}°)')

    ax2.axvline(x=ENTRY_FLAT, color='gray', linewidth=1, linestyle=':')
    ax2.axvline(x=ENTRY_FLAT + arc_length, color='gray', linewidth=1, linestyle=':')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Elevation (m)')
    ax2.set_title(f'Profile Comparison at Arc={arc_length:.1f}m')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Clearance analysis
    ax3 = axes[1, 0]

    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 300)

    down_cl = [check_car_clearance(p, s, z, 'downhill') * 1000 for p in positions]
    up_cl = [check_car_clearance(p, s, z, 'uphill') * 1000 for p in positions]

    ax3.plot(positions, down_cl, 'b-', linewidth=2, label='Downhill')
    ax3.plot(positions, up_cl, 'r-', linewidth=2, label='Uphill')
    ax3.axhline(y=MIN_CLEARANCE * 1000, color='orange', linewidth=2, linestyle='--', label='Min required')
    ax3.axhline(y=0, color='black', linewidth=1)

    ax3.set_xlabel('Car position (m)')
    ax3.set_ylabel('Clearance (mm)')
    ax3.set_title(f'Ground Clearance\nMin Down={min(down_cl):.1f}mm, Min Up={min(up_cl):.1f}mm')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-30, 120)

    # 4. Specifications
    ax4 = axes[1, 1]
    ax4.axis('off')

    R = arc_length * 2 / np.pi

    specs = f"""
    MAXIMUM SLOPE DESIGN - {max_slope:.1f}°
    ════════════════════════════════════════════════

    Profile type:       {profile_type.upper()}

    GEOMETRY:
      Horizontal radius:  {R:.2f}m
      Arc length:         {arc_length:.1f}m
      Entry flat:         {ENTRY_FLAT}m
      Exit flat:          {EXIT_FLAT}m
      TOTAL LENGTH:       {total_length:.1f}m

    SLOPES:
      Maximum slope:      {max_slope:.1f}°

    CLEARANCES:
      Downhill min:       {min(down_cl):.1f}mm
      Uphill min:         {min(up_cl):.1f}mm
      Status:             {'OK' if min(min(down_cl), min(up_cl)) >= MIN_CLEARANCE * 1000 else 'SCRAPE'}

    COMPARISON:
      Original (cubic):   11.78m arc, 24°, SCRAPES
      This design:        {arc_length:.1f}m arc, {max_slope:.1f}°, SAFE
    """

    ax4.text(0.05, 0.95, specs, transform=ax4.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    plt.suptitle(f'MAXIMUM SLOPE RAMP DESIGN\n'
                 f'Profile: {profile_type.upper()} | Arc: {arc_length:.1f}m | Slope: {max_slope:.1f}°',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspaces/RAMP/minimum_radial_3.5m_blueprint_max_slope.png', dpi=200, bbox_inches='tight')
    print(f"\nBlueprint saved: minimum_radial_3.5m_blueprint_max_slope.png")

    return best


def main():
    print("\n" + "=" * 90)
    print(f"MAXIMUM SLOPE RAMP DESIGN - TARGET {TARGET_SLOPE}°")
    print("Testing different profile types to minimize ramp length")
    print("=" * 90)

    best_results = find_minimum_for_target_slope()

    if best_results:
        best = create_comparison_blueprint(best_results)

        arc_length = best[1]
        max_slope = best[2]
        total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
        R = arc_length * 2 / np.pi

        print(f"\n{'=' * 90}")
        print("FINAL RESULT")
        print(f"{'=' * 90}")
        print(f"""
    Profile type:       {best[0].upper()}
    Horizontal radius:  {R:.2f}m
    Arc length:         {arc_length:.1f}m
    Total length:       {total_length:.1f}m
    Maximum slope:      {max_slope:.1f}°
    Minimum clearance:  {best[3]*1000:.1f}mm
        """)


if __name__ == '__main__':
    main()
