#!/usr/bin/env python3
"""
24° Slope Ramp - Both Transitions Gentle

For BIDIRECTIONAL travel, both transitions need to handle overhangs:
- Downhill: enters one way, exits the other
- Uphill: reversed

Both circular arc transitions need to be gentle enough for the car's overhangs.
"""

import numpy as np
import matplotlib.pyplot as plt

# Car specifications
WHEELBASE = 2.350
GROUND_CLEARANCE = 0.106
CAR_LENGTH = 4.461
FRONT_OVERHANG = 0.85
REAR_OVERHANG = CAR_LENGTH - WHEELBASE - FRONT_OVERHANG

VERTICAL_DROP = 3.5
ENTRY_FLAT = 5.0
EXIT_FLAT = 5.0
TARGET_SLOPE = 24.0
MIN_CLEARANCE = 0.005


def create_profile_circular_transitions(R_trans, num_points=3000):
    """
    Create profile with circular arc transitions at both ends.
    Same radius for both transitions to ensure bidirectional safety.
    """
    slope_rad = np.radians(TARGET_SLOPE)
    slope_tan = np.tan(slope_rad)

    # Each circular arc transition
    trans_horiz = R_trans * np.sin(slope_rad)  # horizontal distance
    trans_drop = R_trans * (1 - np.cos(slope_rad))  # vertical drop

    # Check if transitions fit within the vertical drop
    if 2 * trans_drop >= VERTICAL_DROP:
        return None, None, None, None

    # Middle straight section
    middle_drop = VERTICAL_DROP - 2 * trans_drop
    middle_horiz = middle_drop / slope_tan

    # Total arc length
    arc_length = 2 * trans_horiz + middle_horiz
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    # Generate profile
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)

    ramp_start = ENTRY_FLAT
    entry_end = ramp_start + trans_horiz
    middle_end = entry_end + middle_horiz
    ramp_end = ramp_start + arc_length

    for i, pos in enumerate(s):
        if pos <= ramp_start:
            z[i] = 0
        elif pos <= entry_end:
            # Entry circular arc (slope increases from 0 to TARGET_SLOPE)
            local_x = pos - ramp_start
            theta = np.arcsin(min(local_x / R_trans, 1.0))
            z[i] = -R_trans * (1 - np.cos(theta))
        elif pos <= middle_end:
            # Straight middle at constant slope
            local_x = pos - entry_end
            z[i] = -trans_drop - local_x * slope_tan
        elif pos <= ramp_end:
            # Exit circular arc (slope decreases from TARGET_SLOPE to 0)
            local_x = pos - middle_end
            # Need to solve for position on arc where x starts from 0
            # At start: slope = TARGET_SLOPE, at end: slope = 0
            theta = np.arcsin(min(local_x / R_trans, 1.0))
            z[i] = -trans_drop - middle_drop - R_trans * np.sin(slope_rad - theta) * slope_tan + (R_trans * (np.cos(slope_rad - theta) - np.cos(slope_rad)))
            # Simpler: parameterize from end
            remaining = trans_horiz - local_x
            if remaining > 0:
                theta_from_end = np.arcsin(min(remaining / R_trans, 1.0))
                z[i] = -VERTICAL_DROP + R_trans * (1 - np.cos(theta_from_end))
            else:
                z[i] = -VERTICAL_DROP
        else:
            z[i] = -VERTICAL_DROP

    return s, z, arc_length, trans_horiz


def get_elevation_at(s_q, s_arr, z_arr):
    return np.interp(s_q, s_arr, z_arr)


def check_clearance(pos, s_arr, z_arr, direction='downhill'):
    half_wb = WHEELBASE / 2

    if direction == 'downhill':
        fa_pos = pos + half_wb
        ra_pos = pos - half_wb
        fb_pos = fa_pos + FRONT_OVERHANG
        rb_pos = ra_pos - REAR_OVERHANG
    else:
        fa_pos = pos - half_wb
        ra_pos = pos + half_wb
        fb_pos = fa_pos - FRONT_OVERHANG
        rb_pos = ra_pos + REAR_OVERHANG

    fa_z = get_elevation_at(fa_pos, s_arr, z_arr)
    ra_z = get_elevation_at(ra_pos, s_arr, z_arr)

    def body_z(x):
        t = (x - ra_pos) / (fa_pos - ra_pos)
        return ra_z + t * (fa_z - ra_z) + GROUND_CLEARANCE

    fb_cl = body_z(fb_pos) - get_elevation_at(fb_pos, s_arr, z_arr)
    rb_cl = body_z(rb_pos) - get_elevation_at(rb_pos, s_arr, z_arr)

    return min(fb_cl, rb_cl)


def analyze(s_arr, z_arr):
    total = s_arr[-1]
    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total - margin, 500)

    min_down = min(check_clearance(p, s_arr, z_arr, 'downhill') for p in positions)
    min_up = min(check_clearance(p, s_arr, z_arr, 'uphill') for p in positions)

    return min_down, min_up


def main():
    print("=" * 80)
    print("24° SLOPE WITH SYMMETRIC GENTLE TRANSITIONS")
    print("=" * 80)

    # Calculate minimum radius needed
    # For overhang d on curve radius R: clearance loss ≈ d²/(2R)
    # Need d²/(2R) < GROUND_CLEARANCE
    # R > d²/(2 * GROUND_CLEARANCE)
    R_min_rear = REAR_OVERHANG**2 / (2 * GROUND_CLEARANCE)
    R_min_front = FRONT_OVERHANG**2 / (2 * GROUND_CLEARANCE)

    print(f"\nTheoretical minimum R for rear overhang: {R_min_rear:.1f}m")
    print(f"Theoretical minimum R for front overhang: {R_min_front:.1f}m")

    print(f"\n{'R_trans':>10} {'Arc':>10} {'Total':>10} {'Down':>12} {'Up':>12} {'Status':<10}")
    print("-" * 70)

    best = None

    for R in np.arange(8.0, 40.0, 0.5):
        result = create_profile_circular_transitions(R)
        if result[0] is None:
            continue

        s, z, arc, trans_h = result
        total = ENTRY_FLAT + arc + EXIT_FLAT

        min_down, min_up = analyze(s, z)
        overall = min(min_down, min_up)
        is_valid = overall >= MIN_CLEARANCE

        status = "✓ OK" if is_valid else "✗"
        print(f"{R:>10.1f} {arc:>10.2f} {total:>10.1f} {min_down*1000:>10.1f}mm {min_up*1000:>10.1f}mm {status:<10}")

        if is_valid and best is None:
            best = {'R': R, 'arc': arc, 'total': total, 'down': min_down, 'up': min_up, 'trans_h': trans_h}

    if best:
        print(f"\n{'=' * 80}")
        print("MINIMUM VALID DESIGN FOUND")
        print(f"{'=' * 80}")
        print(f"Transition radius: {best['R']:.1f}m")
        print(f"Arc length: {best['arc']:.1f}m")
        print(f"Total length: {best['total']:.1f}m")
        print(f"Clearances: Down={best['down']*1000:.1f}mm, Up={best['up']*1000:.1f}mm")

        # Create blueprint
        s, z, arc, trans_h = create_profile_circular_transitions(best['R'])
        create_blueprint(s, z, best)

    return best


def create_blueprint(s, z, design):
    """Create the blueprint."""
    R = design['R']
    arc = design['arc']
    total = design['total']
    trans_h = design['trans_h']

    slope_rad = np.radians(TARGET_SLOPE)
    trans_drop = R * (1 - np.cos(slope_rad))
    middle_drop = VERTICAL_DROP - 2 * trans_drop
    middle_h = arc - 2 * trans_h

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Side Elevation
    ax1 = axes[0, 0]
    ax1.fill_between(s, z, z.min() - 0.5, color='#d4a574', alpha=0.7)
    ax1.plot(s, z, 'k-', linewidth=3)
    ax1.axhline(y=0, color='green', linewidth=3, label='Street')
    ax1.axhline(y=-VERTICAL_DROP, color='blue', linewidth=3, label='Garage')

    # Mark sections
    ax1.axvline(x=ENTRY_FLAT, color='red', linewidth=2, linestyle='--')
    ax1.axvline(x=ENTRY_FLAT + trans_h, color='orange', linewidth=1, linestyle=':')
    ax1.axvline(x=ENTRY_FLAT + trans_h + middle_h, color='orange', linewidth=1, linestyle=':')
    ax1.axvline(x=ENTRY_FLAT + arc, color='red', linewidth=2, linestyle='--')

    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_title(f'24° Slope Design - R={R:.1f}m Transitions\nArc: {arc:.1f}m | Total: {total:.1f}m')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Slope profile
    ax2 = axes[0, 1]
    ds = s[1] - s[0]
    dz_ds = np.gradient(z, ds)
    slope_deg = np.degrees(np.arctan(-dz_ds))
    ax2.plot(s, slope_deg, 'b-', linewidth=2)
    ax2.axhline(y=TARGET_SLOPE, color='red', linestyle='--', label=f'{TARGET_SLOPE}°')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Slope (°)')
    ax2.set_title('Slope Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Clearance
    ax3 = axes[1, 0]
    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total - margin, 300)
    down_cl = [check_clearance(p, s, z, 'downhill') * 1000 for p in positions]
    up_cl = [check_clearance(p, s, z, 'uphill') * 1000 for p in positions]

    ax3.plot(positions, down_cl, 'b-', linewidth=2, label='Downhill')
    ax3.plot(positions, up_cl, 'r-', linewidth=2, label='Uphill')
    ax3.axhline(y=MIN_CLEARANCE * 1000, color='orange', linestyle='--', label='Min')
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_xlabel('Car position (m)')
    ax3.set_ylabel('Clearance (mm)')
    ax3.set_title(f'Clearance: Down={min(down_cl):.1f}mm, Up={min(up_cl):.1f}mm')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-20, 120)

    # 4. Specs
    ax4 = axes[1, 1]
    ax4.axis('off')
    specs = f"""
    24° SLOPE RAMP DESIGN
    ════════════════════════════════════════

    GEOMETRY:
      Transition radius:  {R:.1f}m (both ends)
      Transition length:  {trans_h:.2f}m (each)
      Middle (24° const): {middle_h:.2f}m
      Arc length:         {arc:.1f}m
      Total length:       {total:.1f}m

    VERTICAL:
      Trans drop (each):  {trans_drop:.2f}m
      Middle drop:        {middle_drop:.2f}m
      Total drop:         {VERTICAL_DROP}m

    CLEARANCES:
      Downhill:           {design['down']*1000:.1f}mm
      Uphill:             {design['up']*1000:.1f}mm

    COMPARISON:
      Original (11.8m):   SCRAPES
      This ({arc:.1f}m):      SAFE at 24°!
    """
    ax4.text(0.05, 0.95, specs, transform=ax4.transAxes, fontsize=11,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    plt.tight_layout()
    plt.savefig('/workspaces/RAMP/minimum_radial_3.5m_blueprint_24deg.png', dpi=200, bbox_inches='tight')
    print(f"\nBlueprint saved: minimum_radial_3.5m_blueprint_24deg.png")


if __name__ == '__main__':
    main()
