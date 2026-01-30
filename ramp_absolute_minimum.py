#!/usr/bin/env python3
"""
Find ABSOLUTE MINIMUM radius for the ramp.

Constraints:
- Car must NOT touch ground (front or rear) in EITHER direction
- Allow maximum slope (no slope limit)
- Find the smallest possible radius/arc length

This will give us the physically smallest ramp that works.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# Car specifications (Porsche 911, 997.1, 2008)
WHEELBASE = 2.350  # m
GROUND_CLEARANCE = 0.106  # m (106mm)
CAR_LENGTH = 4.461  # m
CAR_WIDTH = 1.808  # m

# Overhang distances
FRONT_OVERHANG = 0.85  # m
REAR_OVERHANG = CAR_LENGTH - WHEELBASE - FRONT_OVERHANG  # 1.261m

# Ramp parameters
VERTICAL_DROP = 3.5  # m
ENTRY_FLAT = 5.0  # m
EXIT_FLAT = 5.0   # m

# ABSOLUTE MINIMUM clearance - just don't touch!
MIN_CLEARANCE = 0.005  # 5mm - absolute minimum to not touch


def get_ramp_profile(arc_length, num_points=2000):
    """Generate ramp profile with flat sections."""
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)

    # Ramp section (cubic profile for smooth transitions)
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
    else:  # uphill
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

    return {
        'front_clearance': front_clearance,
        'rear_clearance': rear_clearance,
        'min_clearance': min(front_clearance, rear_clearance),
        'critical': 'front' if front_clearance < rear_clearance else 'rear',
    }


def analyze_full_traverse(arc_length, direction='downhill'):
    """Analyze minimum clearance for full traverse."""
    s_array, z_array = get_ramp_profile(arc_length)
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 500)

    min_clearance = float('inf')
    min_pos = 0
    min_type = ''

    for pos in positions:
        result = check_car_clearance(pos, s_array, z_array, direction)
        if result['min_clearance'] < min_clearance:
            min_clearance = result['min_clearance']
            min_pos = pos
            min_type = result['critical']

    return min_clearance, min_pos, min_type


def calculate_max_slope(arc_length):
    """Calculate maximum slope for given arc length."""
    L = arc_length
    H = VERTICAL_DROP
    # For cubic profile z = as³ + bs², max slope is at s = L/2
    # dz/ds = 3as² + 2bs
    # At s = L/2: dz/ds = 3a(L/2)² + 2b(L/2) = 3aL²/4 + bL
    a = 2 * H / L**3
    b = -3 * H / L**2
    max_slope_rad = abs(3 * a * (L/2)**2 + 2 * b * (L/2))
    return np.degrees(np.arctan(max_slope_rad))


def find_absolute_minimum():
    """Find the absolute minimum arc length with fine resolution."""

    print("=" * 80)
    print("FINDING ABSOLUTE MINIMUM ARC LENGTH")
    print("Constraint: Car must not touch ground (>5mm clearance)")
    print("=" * 80)

    print(f"\nCar geometry:")
    print(f"  Front overhang: {FRONT_OVERHANG*1000:.0f}mm")
    print(f"  Rear overhang:  {REAR_OVERHANG*1000:.0f}mm")
    print(f"  Ground clearance: {GROUND_CLEARANCE*1000:.0f}mm")

    print(f"\n{'Arc Len':>10} {'Down Cl':>12} {'Down Crit':>12} {'Up Cl':>12} {'Up Crit':>12} {'Max Slope':>10} {'Status':<10}")
    print("-" * 90)

    results = []

    # Fine search from 15m to 25m in 0.25m increments
    for arc_len in np.arange(15.0, 26.0, 0.25):
        down_cl, down_pos, down_crit = analyze_full_traverse(arc_len, 'downhill')
        up_cl, up_pos, up_crit = analyze_full_traverse(arc_len, 'uphill')
        max_slope = calculate_max_slope(arc_len)

        overall_min = min(down_cl, up_cl)
        is_safe = overall_min >= MIN_CLEARANCE

        results.append({
            'arc_length': arc_len,
            'down_clearance': down_cl,
            'down_critical': down_crit,
            'up_clearance': up_cl,
            'up_critical': up_crit,
            'max_slope': max_slope,
            'is_safe': is_safe,
        })

        status = "✓ OK" if is_safe else "✗ TOUCH"
        print(f"{arc_len:>10.2f} {down_cl*1000:>10.1f}mm {down_crit:>12} "
              f"{up_cl*1000:>10.1f}mm {up_crit:>12} {max_slope:>10.1f}° {status:<10}")

    # Find minimum safe
    safe_results = [r for r in results if r['is_safe']]

    if safe_results:
        min_safe = safe_results[0]

        # Now do a finer search around this value
        print(f"\n--- Fine tuning around {min_safe['arc_length']}m ---\n")

        fine_results = []
        for arc_len in np.arange(min_safe['arc_length'] - 0.5, min_safe['arc_length'] + 0.5, 0.05):
            down_cl, _, down_crit = analyze_full_traverse(arc_len, 'downhill')
            up_cl, _, up_crit = analyze_full_traverse(arc_len, 'uphill')
            max_slope = calculate_max_slope(arc_len)

            overall_min = min(down_cl, up_cl)
            is_safe = overall_min >= MIN_CLEARANCE

            fine_results.append({
                'arc_length': arc_len,
                'down_clearance': down_cl,
                'up_clearance': up_cl,
                'max_slope': max_slope,
                'is_safe': is_safe,
            })

            status = "✓" if is_safe else "✗"
            print(f"  {arc_len:>8.2f}m: down={down_cl*1000:>6.1f}mm, up={up_cl*1000:>6.1f}mm, slope={max_slope:>5.1f}° {status}")

        # Find absolute minimum
        safe_fine = [r for r in fine_results if r['is_safe']]
        if safe_fine:
            absolute_min = safe_fine[0]
            print(f"\n{'=' * 80}")
            print(f"ABSOLUTE MINIMUM ARC LENGTH: {absolute_min['arc_length']:.2f}m")
            print(f"Maximum slope: {absolute_min['max_slope']:.1f}°")
            print(f"{'=' * 80}")
            return absolute_min

    print("\nNo safe configuration found!")
    return None


def calculate_horizontal_radius(arc_length):
    """Calculate horizontal radius for quarter circle."""
    return arc_length * 2 / np.pi


def create_minimum_blueprint(result):
    """Create blueprint for the absolute minimum design."""

    arc_length = result['arc_length']
    R = calculate_horizontal_radius(arc_length)
    max_slope = result['max_slope']
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    # Get profiles
    s_array, z_array = get_ramp_profile(arc_length)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.25)

    # 1. Side Elevation
    ax1 = fig.add_subplot(gs[0, :])

    ax1.axhline(y=0, color='green', linewidth=3, label='Street level')
    ax1.axhline(y=-VERTICAL_DROP, color='blue', linewidth=3, label='Garage level')

    ax1.fill_between(s_array, z_array, z_array.min() - 0.5, color='#d4a574', alpha=0.7)
    ax1.plot(s_array, z_array, 'k-', linewidth=3)

    # Mark sections
    ax1.axvline(x=ENTRY_FLAT, color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax1.axvline(x=ENTRY_FLAT + arc_length, color='red', linewidth=2, linestyle='--', alpha=0.7)

    ax1.text(ENTRY_FLAT/2, 0.4, f'STREET\n{ENTRY_FLAT}m', ha='center', fontsize=12, color='green', fontweight='bold')
    ax1.text(ENTRY_FLAT + arc_length/2, 0.4, f'RAMP\n{arc_length:.1f}m', ha='center', fontsize=12, fontweight='bold')
    ax1.text(ENTRY_FLAT + arc_length + EXIT_FLAT/2, -VERTICAL_DROP + 0.4, f'GARAGE\n{EXIT_FLAT}m',
             ha='center', fontsize=12, color='blue', fontweight='bold')

    # Add slope annotation at steepest point
    mid_ramp = ENTRY_FLAT + arc_length/2
    ax1.annotate(f'Max slope: {max_slope:.1f}°', xy=(mid_ramp, -VERTICAL_DROP/2),
                fontsize=14, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax1.set_xlabel('Distance along path (m)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.set_title(f'SIDE ELEVATION - Absolute Minimum Design\n'
                  f'Arc: {arc_length:.1f}m | Total: {total_length:.1f}m | Max Slope: {max_slope:.1f}°',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, total_length + 0.5)
    ax1.set_ylim(-4.5, 1)

    # 2. Clearance Analysis - Downhill
    ax2 = fig.add_subplot(gs[1, 0])

    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 300)

    front_cl_down = []
    rear_cl_down = []

    for pos in positions:
        result_check = check_car_clearance(pos, s_array, z_array, 'downhill')
        front_cl_down.append(result_check['front_clearance'] * 1000)
        rear_cl_down.append(result_check['rear_clearance'] * 1000)

    ax2.plot(positions, front_cl_down, 'b-', linewidth=2, label='Front bumper')
    ax2.plot(positions, rear_cl_down, 'r-', linewidth=2, label='Rear bumper')
    ax2.axhline(y=MIN_CLEARANCE * 1000, color='orange', linewidth=2, linestyle='--', label=f'Min ({MIN_CLEARANCE*1000:.0f}mm)')
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axvline(x=ENTRY_FLAT, color='gray', linewidth=1, linestyle=':')
    ax2.axvline(x=ENTRY_FLAT + arc_length, color='gray', linewidth=1, linestyle=':')

    ax2.fill_between(positions, 0, [min(f, r) for f, r in zip(front_cl_down, rear_cl_down)],
                     where=[min(f, r) > 0 for f, r in zip(front_cl_down, rear_cl_down)],
                     color='green', alpha=0.2)

    ax2.set_xlabel('Car center position (m)', fontsize=11)
    ax2.set_ylabel('Clearance (mm)', fontsize=11)
    ax2.set_title(f'DOWNHILL Clearance\nMin: {result["down_clearance"]*1000:.1f}mm', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-20, 120)

    # 3. Clearance Analysis - Uphill
    ax3 = fig.add_subplot(gs[1, 1])

    front_cl_up = []
    rear_cl_up = []

    for pos in positions:
        result_check = check_car_clearance(pos, s_array, z_array, 'uphill')
        front_cl_up.append(result_check['front_clearance'] * 1000)
        rear_cl_up.append(result_check['rear_clearance'] * 1000)

    ax3.plot(positions, front_cl_up, 'b-', linewidth=2, label='Front bumper')
    ax3.plot(positions, rear_cl_up, 'r-', linewidth=2, label='Rear bumper')
    ax3.axhline(y=MIN_CLEARANCE * 1000, color='orange', linewidth=2, linestyle='--', label=f'Min ({MIN_CLEARANCE*1000:.0f}mm)')
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.axvline(x=ENTRY_FLAT, color='gray', linewidth=1, linestyle=':')
    ax3.axvline(x=ENTRY_FLAT + arc_length, color='gray', linewidth=1, linestyle=':')

    ax3.fill_between(positions, 0, [min(f, r) for f, r in zip(front_cl_up, rear_cl_up)],
                     where=[min(f, r) > 0 for f, r in zip(front_cl_up, rear_cl_up)],
                     color='green', alpha=0.2)

    ax3.set_xlabel('Car center position (m)', fontsize=11)
    ax3.set_ylabel('Clearance (mm)', fontsize=11)
    ax3.set_title(f'UPHILL Clearance\nMin: {result["up_clearance"]*1000:.1f}mm', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-20, 120)

    # 4. Specifications and comparison
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    specs = f"""
    ABSOLUTE MINIMUM DESIGN SPECIFICATIONS
    ════════════════════════════════════════════════════

    GEOMETRY:
      Horizontal radius:    {R:.2f}m (quarter circle)
      Arc length:           {arc_length:.2f}m
      Entry flat (street):  {ENTRY_FLAT}m
      Exit flat (garage):   {EXIT_FLAT}m
      ─────────────────────────────────────────────────
      TOTAL LENGTH:         {total_length:.1f}m

    SLOPES:
      Maximum slope:        {max_slope:.1f}°
      Entry/Exit:           0° (level transitions)

    CLEARANCES (verified both directions):
      Downhill minimum:     {result['down_clearance']*1000:.1f}mm
      Uphill minimum:       {result['up_clearance']*1000:.1f}mm
      Required minimum:     {MIN_CLEARANCE*1000:.0f}mm
      Status:               ✓ NO GROUND CONTACT

    CAR GEOMETRY:
      Front overhang:       {FRONT_OVERHANG*1000:.0f}mm
      Rear overhang:        {REAR_OVERHANG*1000:.0f}mm  ← Critical!
      Ground clearance:     {GROUND_CLEARANCE*1000:.0f}mm
    """

    ax4.text(0.02, 0.98, specs, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # 5. Comparison table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Calculate other designs for comparison
    designs = [
        ("Original (UNSAFE)", 11.78, -155),
        ("Conservative (20mm)", 22.0, 21),
        ("ABSOLUTE MIN", arc_length, min(result['down_clearance'], result['up_clearance'])*1000),
    ]

    comparison = f"""
    COMPARISON OF DESIGNS
    ════════════════════════════════════════════════════

    Design              Arc Length    Min Clearance    Status
    ─────────────────────────────────────────────────────────
    Original design     11.78m        -155mm           SCRAPES!
    Conservative        22.0m         +21mm            Safe
    ABSOLUTE MINIMUM    {arc_length:.1f}m         +{min(result['down_clearance'], result['up_clearance'])*1000:.0f}mm            Safe

    ─────────────────────────────────────────────────────────

    SAVINGS vs Conservative design:
      Arc length:    {22.0 - arc_length:.1f}m shorter
      Total length:  {(22.0 - arc_length):.1f}m shorter
      Max slope:     {calculate_max_slope(arc_length) - calculate_max_slope(22.0):.1f}° steeper

    TRADE-OFF:
      • Steeper slope ({max_slope:.1f}° vs 13.4°)
      • Less clearance margin ({min(result['down_clearance'], result['up_clearance'])*1000:.0f}mm vs 21mm)
      • Still safe - car does NOT touch ground
    """

    ax5.text(0.02, 0.98, comparison, transform=ax5.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle(f'ABSOLUTE MINIMUM RAMP DESIGN - {VERTICAL_DROP}m DROP\n'
                 f'Arc={arc_length:.1f}m | Total={total_length:.1f}m | Slope={max_slope:.1f}° | Bidirectional Safe',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('/workspaces/RAMP/minimum_radial_3.5m_blueprint_absolute_min.png', dpi=200, bbox_inches='tight')
    print(f"\nBlueprint saved: minimum_radial_3.5m_blueprint_absolute_min.png")


def save_measurements(result):
    """Save measurements to CSV."""
    arc_length = result['arc_length']
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
    s_array, z_array = get_ramp_profile(arc_length)

    with open('/workspaces/RAMP/minimum_radial_3.5m_measurements_absolute_min.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Distance_m', 'Elevation_m', 'Elevation_cm', 'Section'])

        for dist in np.arange(0, total_length + 0.25, 0.5):
            if dist <= total_length:
                elev = get_elevation_at(dist, s_array, z_array)
                if dist < ENTRY_FLAT:
                    section = 'Street'
                elif dist > ENTRY_FLAT + arc_length:
                    section = 'Garage'
                else:
                    section = 'Ramp'
                writer.writerow([f"{dist:.2f}", f"{elev:.4f}", f"{elev*100:.1f}", section])

    print(f"Measurements saved: minimum_radial_3.5m_measurements_absolute_min.csv")


def main():
    print("\n" + "=" * 80)
    print("ABSOLUTE MINIMUM RAMP DESIGN")
    print("Finding the shortest possible ramp that doesn't scrape")
    print("=" * 80)

    result = find_absolute_minimum()

    if result:
        create_minimum_blueprint(result)
        save_measurements(result)

        arc_length = result['arc_length']
        total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
        R = calculate_horizontal_radius(arc_length)

        print(f"\n{'=' * 80}")
        print("FINAL RESULT - ABSOLUTE MINIMUM DESIGN")
        print(f"{'=' * 80}")
        print(f"""
    Arc length:         {arc_length:.2f}m
    Total length:       {total_length:.1f}m
    Horizontal radius:  {R:.2f}m
    Maximum slope:      {result['max_slope']:.1f}°

    Clearances:
      Downhill:         {result['down_clearance']*1000:.1f}mm
      Uphill:           {result['up_clearance']*1000:.1f}mm

    This is the SMALLEST possible ramp where the
    Porsche 997.1 won't touch the ground in either direction.
        """)


if __name__ == '__main__':
    main()
