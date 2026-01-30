#!/usr/bin/env python3
"""
Ramp Ground Clearance Analysis - CORRECTED VERSION

This script properly analyzes ground clearance for BOTH directions:
- Downhill: Street → Garage (front of car leading)
- Uphill: Garage → Street (front of car leading, but climbing)

The CRITICAL issue with the previous design:
- It only checked clearance at the wheelbase center
- It did NOT check the front/rear overhangs which extend beyond the wheels
- On convex transitions (slope changes from steep to flat), the overhangs can scrape

Key insight: When going UPHILL, at the bottom transition (garage→ramp),
the FRONT overhang can hit the rising ramp surface.
"""

import numpy as np
import matplotlib.pyplot as plt

# Car specifications (Porsche 911, 997.1, 2008)
WHEELBASE = 2.350  # m - distance between front and rear axles
GROUND_CLEARANCE = 0.106  # m (106mm) - clearance at lowest point
CAR_LENGTH = 4.461  # m - total length
CAR_WIDTH = 1.808  # m

# Overhang distances (from axle to bumper)
FRONT_OVERHANG = 0.85  # m - front axle to front bumper
REAR_OVERHANG = CAR_LENGTH - WHEELBASE - FRONT_OVERHANG  # 1.261m

# Ramp parameters
VERTICAL_DROP = 3.5  # m
ENTRY_FLAT = 5.0  # m - flat section before ramp (street level)
EXIT_FLAT = 5.0   # m - flat section after ramp (garage level)

# Safety margin
MIN_CLEARANCE = 0.02  # m (2cm) - minimum acceptable clearance


def get_ramp_profile(arc_length, num_points=2000):
    """
    Generate the ramp profile using cubic polynomial.
    Returns arrays of position (s) and elevation (z).

    Profile: z = a*s³ + b*s² where:
    - a = 2H/L³
    - b = -3H/L²
    This gives zero slope at both ends and smooth transition.
    """
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)

    # Entry flat (street level = 0)
    # Already zeros

    # Ramp section
    ramp_mask = (s >= ENTRY_FLAT) & (s <= ENTRY_FLAT + arc_length)
    s_ramp = s[ramp_mask] - ENTRY_FLAT  # Local coordinate on ramp
    L = arc_length
    H = VERTICAL_DROP
    a = 2 * H / L**3
    b = -3 * H / L**2
    z[ramp_mask] = a * s_ramp**3 + b * s_ramp**2  # Negative (going down)

    # Exit flat (garage level)
    exit_mask = s > ENTRY_FLAT + arc_length
    z[exit_mask] = -VERTICAL_DROP

    return s, z


def get_elevation_at(s_query, s_array, z_array):
    """Interpolate elevation at a specific position."""
    return np.interp(s_query, s_array, z_array)


def get_slope_at(s_query, s_array, z_array):
    """Get slope (dz/ds) at a specific position."""
    # Use central difference
    ds = s_array[1] - s_array[0]
    dz_ds = np.gradient(z_array, ds)
    return np.interp(s_query, s_array, dz_ds)


def check_car_clearance(car_center_pos, s_array, z_array, direction='downhill'):
    """
    Check ground clearance for car at given position.

    Args:
        car_center_pos: Position of car center (midpoint between axles)
        s_array, z_array: Ramp profile arrays
        direction: 'downhill' (front leads going down) or 'uphill' (front leads going up)

    Returns:
        dict with clearance info for front bumper, center, and rear bumper
    """
    # Axle positions (car center is midpoint between axles)
    half_wheelbase = WHEELBASE / 2

    if direction == 'downhill':
        # Front axle is ahead (larger s), rear axle behind (smaller s)
        front_axle_pos = car_center_pos + half_wheelbase
        rear_axle_pos = car_center_pos - half_wheelbase
        front_bumper_pos = front_axle_pos + FRONT_OVERHANG
        rear_bumper_pos = rear_axle_pos - REAR_OVERHANG
    else:  # uphill - car is reversed relative to travel direction
        # Front axle is behind (smaller s), rear axle ahead (larger s)
        front_axle_pos = car_center_pos - half_wheelbase
        rear_axle_pos = car_center_pos + half_wheelbase
        front_bumper_pos = front_axle_pos - FRONT_OVERHANG
        rear_bumper_pos = rear_axle_pos + REAR_OVERHANG

    # Get ground elevations at axle positions
    front_axle_ground = get_elevation_at(front_axle_pos, s_array, z_array)
    rear_axle_ground = get_elevation_at(rear_axle_pos, s_array, z_array)

    # Car body follows straight line between axles (wheels on ground)
    # Wheel centers are at ground + some wheel radius, but for clearance
    # we care about the lowest point of the body

    # The car body line: from (front_axle_pos, front_axle_ground) to (rear_axle_pos, rear_axle_ground)
    # Car body height at any point x along this line:
    def car_body_height(x_pos):
        """Height of car body (bottom) at position x, assuming wheels on ground."""
        # Linear interpolation along car body line
        t = (x_pos - rear_axle_pos) / (front_axle_pos - rear_axle_pos)
        body_line_z = rear_axle_ground + t * (front_axle_ground - rear_axle_ground)
        # Add ground clearance (body is above wheel contact points)
        return body_line_z + GROUND_CLEARANCE

    # Check clearance at front bumper
    front_bumper_body_height = car_body_height(front_bumper_pos)
    front_bumper_ground = get_elevation_at(front_bumper_pos, s_array, z_array)
    front_bumper_clearance = front_bumper_body_height - front_bumper_ground

    # Check clearance at rear bumper
    rear_bumper_body_height = car_body_height(rear_bumper_pos)
    rear_bumper_ground = get_elevation_at(rear_bumper_pos, s_array, z_array)
    rear_bumper_clearance = rear_bumper_body_height - rear_bumper_ground

    # Check clearance at center (between axles)
    center_body_height = car_body_height(car_center_pos)
    center_ground = get_elevation_at(car_center_pos, s_array, z_array)
    center_clearance = center_body_height - center_ground

    return {
        'front_bumper_pos': front_bumper_pos,
        'front_bumper_clearance': front_bumper_clearance,
        'rear_bumper_pos': rear_bumper_pos,
        'rear_bumper_clearance': rear_bumper_clearance,
        'center_clearance': center_clearance,
        'min_clearance': min(front_bumper_clearance, rear_bumper_clearance, center_clearance),
        'front_axle_pos': front_axle_pos,
        'rear_axle_pos': rear_axle_pos,
    }


def analyze_full_traverse(arc_length, direction='downhill'):
    """
    Analyze clearance for car traversing entire ramp.

    Returns minimum clearance found and position where it occurs.
    """
    s_array, z_array = get_ramp_profile(arc_length)
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    # Car center positions to check
    # Need to ensure entire car is within bounds
    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 500)

    min_clearance = float('inf')
    min_clearance_pos = 0
    min_clearance_type = ''

    all_clearances = []

    for pos in positions:
        result = check_car_clearance(pos, s_array, z_array, direction)
        all_clearances.append({
            'pos': pos,
            **result
        })

        if result['front_bumper_clearance'] < min_clearance:
            min_clearance = result['front_bumper_clearance']
            min_clearance_pos = pos
            min_clearance_type = 'front_bumper'

        if result['rear_bumper_clearance'] < min_clearance:
            min_clearance = result['rear_bumper_clearance']
            min_clearance_pos = pos
            min_clearance_type = 'rear_bumper'

    return {
        'min_clearance': min_clearance,
        'min_clearance_pos': min_clearance_pos,
        'min_clearance_type': min_clearance_type,
        'all_clearances': all_clearances,
        's_array': s_array,
        'z_array': z_array,
    }


def find_minimum_safe_arc_length():
    """
    Find the minimum arc length that provides safe clearance in BOTH directions.
    """
    print("=" * 80)
    print("FINDING MINIMUM SAFE ARC LENGTH FOR BIDIRECTIONAL TRAVEL")
    print("=" * 80)
    print(f"\nCar specifications:")
    print(f"  Wheelbase:       {WHEELBASE:.3f}m")
    print(f"  Front overhang:  {FRONT_OVERHANG:.3f}m")
    print(f"  Rear overhang:   {REAR_OVERHANG:.3f}m")
    print(f"  Ground clearance: {GROUND_CLEARANCE*1000:.0f}mm")
    print(f"  Total length:    {CAR_LENGTH:.3f}m")
    print(f"\nRamp requirements:")
    print(f"  Vertical drop:   {VERTICAL_DROP}m")
    print(f"  Min clearance:   {MIN_CLEARANCE*1000:.0f}mm")

    print(f"\n{'Arc Len':>10} {'Down Min':>12} {'Down Type':>12} {'Up Min':>12} {'Up Type':>12} {'Status':<10}")
    print("-" * 80)

    results = []

    # Search arc lengths from 10m to 25m
    for arc_len in np.arange(10.0, 30.0, 0.5):
        down_result = analyze_full_traverse(arc_len, 'downhill')
        up_result = analyze_full_traverse(arc_len, 'uphill')

        down_min = down_result['min_clearance']
        up_min = up_result['min_clearance']
        overall_min = min(down_min, up_min)

        is_safe = overall_min >= MIN_CLEARANCE
        status = "✓ SAFE" if is_safe else "✗ FAIL"

        results.append({
            'arc_length': arc_len,
            'down_min': down_min,
            'down_type': down_result['min_clearance_type'],
            'up_min': up_min,
            'up_type': up_result['min_clearance_type'],
            'overall_min': overall_min,
            'is_safe': is_safe,
        })

        print(f"{arc_len:>10.1f} {down_min*1000:>10.1f}mm {down_result['min_clearance_type']:>12} "
              f"{up_min*1000:>10.1f}mm {up_result['min_clearance_type']:>12} {status:<10}")

    # Find minimum safe arc length
    safe_results = [r for r in results if r['is_safe']]
    if safe_results:
        min_safe = safe_results[0]
        print(f"\n{'=' * 80}")
        print(f"MINIMUM SAFE ARC LENGTH: {min_safe['arc_length']:.1f}m")
        print(f"{'=' * 80}")
        return min_safe
    else:
        print("\nNo safe configuration found in search range!")
        return None


def create_clearance_visualization(arc_length):
    """Create detailed visualization of clearance analysis."""

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))

    # Get profiles
    s_array, z_array = get_ramp_profile(arc_length)
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    down_result = analyze_full_traverse(arc_length, 'downhill')
    up_result = analyze_full_traverse(arc_length, 'uphill')

    # 1. Ramp profile with transitions marked
    ax1 = axes[0, 0]
    ax1.fill_between(s_array, z_array, z_array.min() - 0.5, color='#d4a574', alpha=0.7)
    ax1.plot(s_array, z_array, 'k-', linewidth=2, label='Ramp surface')
    ax1.axhline(y=0, color='green', linewidth=2, linestyle='--', label='Street level')
    ax1.axhline(y=-VERTICAL_DROP, color='blue', linewidth=2, linestyle='--', label='Garage level')
    ax1.axvline(x=ENTRY_FLAT, color='red', linewidth=1, linestyle=':', alpha=0.7)
    ax1.axvline(x=ENTRY_FLAT + arc_length, color='red', linewidth=1, linestyle=':', alpha=0.7)
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_title(f'Ramp Profile - Arc Length {arc_length:.1f}m')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, total_length + 0.5)

    # 2. Downhill clearance along path
    ax2 = axes[0, 1]
    positions = [c['pos'] for c in down_result['all_clearances']]
    front_cl = [c['front_bumper_clearance'] * 1000 for c in down_result['all_clearances']]
    rear_cl = [c['rear_bumper_clearance'] * 1000 for c in down_result['all_clearances']]

    ax2.plot(positions, front_cl, 'b-', linewidth=2, label='Front bumper')
    ax2.plot(positions, rear_cl, 'r-', linewidth=2, label='Rear bumper')
    ax2.axhline(y=MIN_CLEARANCE * 1000, color='orange', linewidth=2, linestyle='--', label=f'Min required ({MIN_CLEARANCE*1000:.0f}mm)')
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axvline(x=ENTRY_FLAT, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax2.axvline(x=ENTRY_FLAT + arc_length, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax2.set_xlabel('Car center position (m)')
    ax2.set_ylabel('Clearance (mm)')
    ax2.set_title(f'DOWNHILL Clearance - Min: {down_result["min_clearance"]*1000:.1f}mm at {down_result["min_clearance_type"]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-50, 150)

    # 3. Uphill clearance along path
    ax3 = axes[1, 0]
    positions = [c['pos'] for c in up_result['all_clearances']]
    front_cl = [c['front_bumper_clearance'] * 1000 for c in up_result['all_clearances']]
    rear_cl = [c['rear_bumper_clearance'] * 1000 for c in up_result['all_clearances']]

    ax3.plot(positions, front_cl, 'b-', linewidth=2, label='Front bumper')
    ax3.plot(positions, rear_cl, 'r-', linewidth=2, label='Rear bumper')
    ax3.axhline(y=MIN_CLEARANCE * 1000, color='orange', linewidth=2, linestyle='--', label=f'Min required ({MIN_CLEARANCE*1000:.0f}mm)')
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.axvline(x=ENTRY_FLAT, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax3.axvline(x=ENTRY_FLAT + arc_length, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax3.set_xlabel('Car center position (m)')
    ax3.set_ylabel('Clearance (mm)')
    ax3.set_title(f'UPHILL Clearance - Min: {up_result["min_clearance"]*1000:.1f}mm at {up_result["min_clearance_type"]}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-50, 150)

    # 4. Car at critical uphill position
    ax4 = axes[1, 1]

    # Find worst position for uphill
    worst_idx = np.argmin([c['front_bumper_clearance'] for c in up_result['all_clearances']])
    worst_pos = up_result['all_clearances'][worst_idx]['pos']

    # Draw ramp
    ax4.fill_between(s_array, z_array, z_array.min() - 0.3, color='#d4a574', alpha=0.7)
    ax4.plot(s_array, z_array, 'k-', linewidth=2)

    # Draw car at worst position (uphill orientation)
    half_wb = WHEELBASE / 2
    front_axle_pos = worst_pos - half_wb
    rear_axle_pos = worst_pos + half_wb
    front_bumper_pos = front_axle_pos - FRONT_OVERHANG
    rear_bumper_pos = rear_axle_pos + REAR_OVERHANG

    front_axle_z = get_elevation_at(front_axle_pos, s_array, z_array)
    rear_axle_z = get_elevation_at(rear_axle_pos, s_array, z_array)

    # Car body line (at ground clearance above axles)
    car_x = [front_bumper_pos, front_axle_pos, rear_axle_pos, rear_bumper_pos]

    # Interpolate body line
    def body_z(x):
        t = (x - rear_axle_pos) / (front_axle_pos - rear_axle_pos)
        return rear_axle_z + t * (front_axle_z - rear_axle_z) + GROUND_CLEARANCE

    car_z = [body_z(x) for x in car_x]

    ax4.plot(car_x, car_z, 'b-', linewidth=3, label='Car body (bottom)')
    ax4.plot([front_axle_pos, rear_axle_pos], [front_axle_z, rear_axle_z], 'ko', markersize=10, label='Wheels')
    ax4.plot(front_bumper_pos, body_z(front_bumper_pos), 'r^', markersize=12, label='Front bumper')
    ax4.plot(rear_bumper_pos, body_z(rear_bumper_pos), 'rs', markersize=10, label='Rear bumper')

    # Show clearance
    fb_ground = get_elevation_at(front_bumper_pos, s_array, z_array)
    ax4.plot([front_bumper_pos, front_bumper_pos], [fb_ground, body_z(front_bumper_pos)], 'r--', linewidth=2)
    clearance = (body_z(front_bumper_pos) - fb_ground) * 1000
    ax4.text(front_bumper_pos + 0.3, (fb_ground + body_z(front_bumper_pos))/2,
             f'{clearance:.0f}mm', fontsize=10, color='red', fontweight='bold')

    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Elevation (m)')
    ax4.set_title(f'CRITICAL POSITION - Uphill at s={worst_pos:.1f}m')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(worst_pos - 5, worst_pos + 5)
    ax4.set_ylim(z_array.min() - 0.3, 0.5)
    ax4.set_aspect('equal')

    # 5. Comparison: different arc lengths
    ax5 = axes[2, 0]
    arc_lengths = np.arange(10, 25, 1)
    down_mins = []
    up_mins = []

    for al in arc_lengths:
        dr = analyze_full_traverse(al, 'downhill')
        ur = analyze_full_traverse(al, 'uphill')
        down_mins.append(dr['min_clearance'] * 1000)
        up_mins.append(ur['min_clearance'] * 1000)

    ax5.plot(arc_lengths, down_mins, 'b-o', linewidth=2, label='Downhill min clearance')
    ax5.plot(arc_lengths, up_mins, 'r-s', linewidth=2, label='Uphill min clearance')
    ax5.axhline(y=MIN_CLEARANCE * 1000, color='orange', linewidth=2, linestyle='--', label=f'Required minimum')
    ax5.axhline(y=0, color='black', linewidth=1)
    ax5.axvline(x=arc_length, color='green', linewidth=2, linestyle=':', label=f'Selected ({arc_length}m)')
    ax5.set_xlabel('Arc Length (m)')
    ax5.set_ylabel('Minimum Clearance (mm)')
    ax5.set_title('Clearance vs Arc Length')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Summary
    ax6 = axes[2, 1]
    ax6.axis('off')

    # 7. Bird's Eye View (Top-down view of the track)
    ax7 = axes[3, 0]

    # Define ramp width (wider than car for safety margins)
    ramp_width = CAR_WIDTH + 0.4  # 0.2m margin on each side

    # Draw the track from above
    # Street section (green)
    street_rect = plt.Rectangle((0, -ramp_width/2), ENTRY_FLAT, ramp_width,
                                  facecolor='#90EE90', edgecolor='green', linewidth=2, alpha=0.7)
    ax7.add_patch(street_rect)
    ax7.text(ENTRY_FLAT/2, 0, 'STREET\n(Level 0m)', ha='center', va='center',
             fontsize=10, fontweight='bold', color='darkgreen')

    # Ramp section (gradient from green to blue)
    n_ramp_sections = 20
    for i in range(n_ramp_sections):
        x_start = ENTRY_FLAT + (arc_length / n_ramp_sections) * i
        section_width = arc_length / n_ramp_sections
        # Color gradient from green to blue
        ratio = i / n_ramp_sections
        color = (0.3 * (1-ratio), 0.6 * (1-ratio) + 0.4 * ratio, 0.3 * (1-ratio) + 0.8 * ratio)
        rect = plt.Rectangle((x_start, -ramp_width/2), section_width, ramp_width,
                              facecolor=color, edgecolor='none', alpha=0.7)
        ax7.add_patch(rect)

    # Ramp outline
    ramp_outline = plt.Rectangle((ENTRY_FLAT, -ramp_width/2), arc_length, ramp_width,
                                   facecolor='none', edgecolor='#444444', linewidth=2)
    ax7.add_patch(ramp_outline)
    ax7.text(ENTRY_FLAT + arc_length/2, 0, f'RAMP\n({arc_length:.1f}m)\n↓ {VERTICAL_DROP}m drop',
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Garage section (blue)
    garage_rect = plt.Rectangle((ENTRY_FLAT + arc_length, -ramp_width/2), EXIT_FLAT, ramp_width,
                                  facecolor='#87CEEB', edgecolor='blue', linewidth=2, alpha=0.7)
    ax7.add_patch(garage_rect)
    ax7.text(ENTRY_FLAT + arc_length + EXIT_FLAT/2, 0, f'GARAGE\n(Level -{VERTICAL_DROP}m)',
             ha='center', va='center', fontsize=10, fontweight='bold', color='darkblue')

    # Draw car positions along the path (showing trajectory)
    car_positions = [2.5, ENTRY_FLAT + arc_length * 0.25,
                     ENTRY_FLAT + arc_length * 0.5,
                     ENTRY_FLAT + arc_length * 0.75,
                     ENTRY_FLAT + arc_length + 2.5]

    for i, car_pos in enumerate(car_positions):
        alpha = 0.3 if i != 2 else 0.8  # Highlight middle position
        car_color = 'red' if i == 2 else '#FF6B6B'
        # Car rectangle (from above)
        car_rect = plt.Rectangle((car_pos - CAR_LENGTH/2, -CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH,
                                   facecolor=car_color, edgecolor='darkred', linewidth=1.5, alpha=alpha)
        ax7.add_patch(car_rect)
        # Front indicator (triangle)
        front_x = car_pos + CAR_LENGTH/2
        ax7.plot([front_x, front_x - 0.3, front_x - 0.3],
                 [0, 0.15, -0.15], color='darkred', linewidth=1, alpha=alpha)

    # Draw direction arrow
    ax7.annotate('', xy=(total_length - 1, ramp_width/2 + 0.3),
                 xytext=(1, ramp_width/2 + 0.3),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax7.text(total_length/2, ramp_width/2 + 0.5, 'DOWNHILL DIRECTION',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add dimension annotations
    ax7.annotate('', xy=(0, -ramp_width/2 - 0.4),
                 xytext=(ENTRY_FLAT, -ramp_width/2 - 0.4),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax7.text(ENTRY_FLAT/2, -ramp_width/2 - 0.6, f'{ENTRY_FLAT}m', ha='center', fontsize=8, color='green')

    ax7.annotate('', xy=(ENTRY_FLAT, -ramp_width/2 - 0.4),
                 xytext=(ENTRY_FLAT + arc_length, -ramp_width/2 - 0.4),
                 arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.5))
    ax7.text(ENTRY_FLAT + arc_length/2, -ramp_width/2 - 0.6, f'{arc_length:.1f}m', ha='center', fontsize=8)

    ax7.annotate('', xy=(ENTRY_FLAT + arc_length, -ramp_width/2 - 0.4),
                 xytext=(total_length, -ramp_width/2 - 0.4),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax7.text(ENTRY_FLAT + arc_length + EXIT_FLAT/2, -ramp_width/2 - 0.6, f'{EXIT_FLAT}m', ha='center', fontsize=8, color='blue')

    ax7.set_xlim(-1, total_length + 1)
    ax7.set_ylim(-ramp_width/2 - 1, ramp_width/2 + 1)
    ax7.set_xlabel('Distance (m)')
    ax7.set_ylabel('Width (m)')
    ax7.set_title("BIRD'S EYE VIEW - Track from Street to Garage", fontsize=11, fontweight='bold')
    ax7.set_aspect('equal')
    ax7.grid(True, alpha=0.3)

    # 8. Elevation profile along center line (side companion to bird's eye)
    ax8 = axes[3, 1]
    ax8.fill_between(s_array, z_array, -VERTICAL_DROP - 0.5, color='#d4a574', alpha=0.5)
    ax8.plot(s_array, z_array, 'k-', linewidth=2)

    # Mark the zones
    ax8.axvspan(0, ENTRY_FLAT, alpha=0.3, color='green', label='Street')
    ax8.axvspan(ENTRY_FLAT, ENTRY_FLAT + arc_length, alpha=0.2, color='gray', label='Ramp')
    ax8.axvspan(ENTRY_FLAT + arc_length, total_length, alpha=0.3, color='blue', label='Garage')

    # Add car silhouettes at same positions
    for i, car_pos in enumerate(car_positions):
        car_z = get_elevation_at(car_pos, s_array, z_array)
        alpha = 0.4 if i != 2 else 1.0
        ax8.plot([car_pos - CAR_LENGTH/2, car_pos + CAR_LENGTH/2],
                 [car_z + GROUND_CLEARANCE, car_z + GROUND_CLEARANCE],
                 'r-', linewidth=3, alpha=alpha)
        ax8.plot(car_pos, car_z, 'ko', markersize=4, alpha=alpha)

    ax8.set_xlabel('Distance (m)')
    ax8.set_ylabel('Elevation (m)')
    ax8.set_title('SIDE VIEW - Elevation Profile with Car Positions', fontsize=11, fontweight='bold')
    ax8.legend(loc='upper right', fontsize=8)
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(-1, total_length + 1)

    overall_safe = min(down_result['min_clearance'], up_result['min_clearance']) >= MIN_CLEARANCE
    summary = f"""CLEARANCE ANALYSIS SUMMARY
{'═'*40}
ARC LENGTH: {arc_length:.1f}m | TOTAL: {total_length:.1f}m

DOWNHILL: {down_result['min_clearance']*1000:.1f}mm {'✓' if down_result['min_clearance'] >= MIN_CLEARANCE else '✗'}
UPHILL:   {up_result['min_clearance']*1000:.1f}mm {'✓' if up_result['min_clearance'] >= MIN_CLEARANCE else '✗'}

CAR: WB={WHEELBASE*1000:.0f}mm GC={GROUND_CLEARANCE*1000:.0f}mm

{'✓ BIDIRECTIONAL SAFE' if overall_safe else '✗ NOT SAFE'}"""

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle(f'GROUND CLEARANCE ANALYSIS - Porsche 997.1\n'
                 f'Arc Length: {arc_length:.1f}m | Vertical Drop: {VERTICAL_DROP}m',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(h_pad=2.0)
    plt.savefig('/workspaces/RAMP/clearance_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nClearance analysis saved: clearance_analysis.png")

    return fig


def main():
    print("\n" + "=" * 80)
    print("RAMP CLEARANCE ANALYSIS - BIDIRECTIONAL TRAVEL")
    print("Checking both DOWNHILL and UPHILL clearance")
    print("=" * 80)

    # First, analyze the current design (arc length 11.78m)
    current_arc = 11.78
    print(f"\n--- Analyzing CURRENT design (arc={current_arc}m) ---")

    down = analyze_full_traverse(current_arc, 'downhill')
    up = analyze_full_traverse(current_arc, 'uphill')

    print(f"\nCurrent design ({current_arc}m arc):")
    print(f"  Downhill min clearance: {down['min_clearance']*1000:.1f}mm at {down['min_clearance_type']}")
    print(f"  Uphill min clearance:   {up['min_clearance']*1000:.1f}mm at {up['min_clearance_type']}")

    if up['min_clearance'] < MIN_CLEARANCE:
        print(f"\n  ⚠️  PROBLEM: Uphill clearance ({up['min_clearance']*1000:.1f}mm) is below minimum ({MIN_CLEARANCE*1000:.0f}mm)!")
        print(f"      The {up['min_clearance_type']} will scrape when going uphill.")

    # Find minimum safe arc length
    print("\n")
    safe_result = find_minimum_safe_arc_length()

    if safe_result:
        # Create visualization for the safe design
        create_clearance_visualization(safe_result['arc_length'])

        print(f"\n{'=' * 80}")
        print("RECOMMENDED NEW DESIGN")
        print(f"{'=' * 80}")
        print(f"""
    CURRENT DESIGN: Arc length = {current_arc}m
    - Downhill clearance: {down['min_clearance']*1000:.1f}mm ({'OK' if down['min_clearance'] >= MIN_CLEARANCE else 'FAIL'})
    - Uphill clearance:   {up['min_clearance']*1000:.1f}mm ({'OK' if up['min_clearance'] >= MIN_CLEARANCE else 'FAIL'})

    NEW SAFE DESIGN: Arc length = {safe_result['arc_length']:.1f}m
    - Downhill clearance: {safe_result['down_min']*1000:.1f}mm (OK)
    - Uphill clearance:   {safe_result['up_min']*1000:.1f}mm (OK)

    CHANGES NEEDED:
    - Arc length: {current_arc}m → {safe_result['arc_length']:.1f}m (+{safe_result['arc_length'] - current_arc:.1f}m)
    - Total ramp: {ENTRY_FLAT + current_arc + EXIT_FLAT:.1f}m → {ENTRY_FLAT + safe_result['arc_length'] + EXIT_FLAT:.1f}m
        """)


if __name__ == '__main__':
    main()
