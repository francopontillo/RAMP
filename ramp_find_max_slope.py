#!/usr/bin/env python3
"""
Find the MAXIMUM achievable slope with circular arc transitions.
Searches across all possible slopes to find the steepest that works.
"""

import numpy as np
import matplotlib.pyplot as plt

# Car specifications
WHEELBASE = 2.350
GROUND_CLEARANCE = 0.106
CAR_LENGTH = 4.461
FRONT_OVERHANG = 0.85
REAR_OVERHANG = CAR_LENGTH - WHEELBASE - FRONT_OVERHANG  # 1.261m

VERTICAL_DROP = 3.5
ENTRY_FLAT = 5.0
EXIT_FLAT = 5.0
MIN_CLEARANCE = 0.005  # 5mm minimum


def create_profile(R_trans, slope_deg, num_points=3000):
    """Create profile with circular arc transitions at both ends."""
    slope_rad = np.radians(slope_deg)
    slope_tan = np.tan(slope_rad)

    # Each circular arc transition
    trans_horiz = R_trans * np.sin(slope_rad)
    trans_drop = R_trans * (1 - np.cos(slope_rad))

    # Check if transitions fit
    if 2 * trans_drop >= VERTICAL_DROP:
        return None

    # Middle straight section
    middle_drop = VERTICAL_DROP - 2 * trans_drop
    middle_horiz = middle_drop / slope_tan

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
            local_x = pos - ramp_start
            theta = np.arcsin(min(local_x / R_trans, 1.0))
            z[i] = -R_trans * (1 - np.cos(theta))
        elif pos <= middle_end:
            local_x = pos - entry_end
            z[i] = -trans_drop - local_x * slope_tan
        elif pos <= ramp_end:
            local_x = pos - middle_end
            remaining = trans_horiz - local_x
            if remaining > 0:
                theta_from_end = np.arcsin(min(remaining / R_trans, 1.0))
                z[i] = -VERTICAL_DROP + R_trans * (1 - np.cos(theta_from_end))
            else:
                z[i] = -VERTICAL_DROP
        else:
            z[i] = -VERTICAL_DROP

    return {'s': s, 'z': z, 'arc': arc_length, 'trans_h': trans_horiz,
            'trans_drop': trans_drop, 'middle_h': middle_horiz, 'middle_drop': middle_drop}


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
    print("FINDING MAXIMUM ACHIEVABLE SLOPE")
    print("=" * 80)
    print(f"\nConstraints:")
    print(f"  Vertical drop: {VERTICAL_DROP}m")
    print(f"  Rear overhang: {REAR_OVERHANG:.3f}m (critical)")
    print(f"  Ground clearance: {GROUND_CLEARANCE*1000:.0f}mm")
    print(f"  Minimum safety margin: {MIN_CLEARANCE*1000:.0f}mm")

    best_designs = []

    # Search from high to low slopes
    for slope in np.arange(25.0, 10.0, -0.5):
        for R in np.arange(5.0, 50.0, 0.5):
            result = create_profile(R, slope)
            if result is None:
                continue

            min_down, min_up = analyze(result['s'], result['z'])
            overall = min(min_down, min_up)

            if overall >= MIN_CLEARANCE:
                best_designs.append({
                    'slope': slope,
                    'R': R,
                    'arc': result['arc'],
                    'total': ENTRY_FLAT + result['arc'] + EXIT_FLAT,
                    'down': min_down,
                    'up': min_up,
                    'result': result
                })
                break  # Found valid R for this slope, move to next slope

    if not best_designs:
        print("\nNo valid designs found!")
        return None

    # Sort by slope (highest first)
    best_designs.sort(key=lambda x: -x['slope'])

    print(f"\n{'Slope':>8} {'R':>8} {'Arc':>10} {'Total':>10} {'Down':>12} {'Up':>12}")
    print("-" * 70)

    for d in best_designs[:10]:
        print(f"{d['slope']:>7.1f}° {d['R']:>8.1f} {d['arc']:>10.2f} {d['total']:>10.1f} "
              f"{d['down']*1000:>10.1f}mm {d['up']*1000:>10.1f}mm")

    best = best_designs[0]

    print(f"\n{'=' * 80}")
    print("MAXIMUM ACHIEVABLE DESIGN")
    print(f"{'=' * 80}")
    print(f"  Maximum slope:       {best['slope']:.1f}°")
    print(f"  Transition radius:   {best['R']:.1f}m")
    print(f"  Arc length:          {best['arc']:.2f}m")
    print(f"  Total length:        {best['total']:.1f}m")
    print(f"  Clearance (down):    {best['down']*1000:.1f}mm")
    print(f"  Clearance (up):      {best['up']*1000:.1f}mm")

    # Create blueprint
    create_blueprint(best)

    return best


def create_blueprint(design):
    """Create detailed blueprint."""
    slope = design['slope']
    R = design['R']
    result = design['result']
    s, z = result['s'], result['z']
    arc = result['arc']
    total = design['total']
    trans_h = result['trans_h']
    trans_drop = result['trans_drop']
    middle_h = result['middle_h']
    middle_drop = result['middle_drop']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Side Elevation
    ax1 = axes[0, 0]
    ax1.fill_between(s, z, z.min() - 0.5, color='#d4a574', alpha=0.7)
    ax1.plot(s, z, 'k-', linewidth=3)
    ax1.axhline(y=0, color='green', linewidth=3, label='Street Level')
    ax1.axhline(y=-VERTICAL_DROP, color='blue', linewidth=3, label='Garage Level')

    ax1.axvline(x=ENTRY_FLAT, color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax1.axvline(x=ENTRY_FLAT + trans_h, color='orange', linewidth=1, linestyle=':')
    ax1.axvline(x=ENTRY_FLAT + trans_h + middle_h, color='orange', linewidth=1, linestyle=':')
    ax1.axvline(x=ENTRY_FLAT + arc, color='red', linewidth=2, linestyle='--', alpha=0.7)

    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.set_title(f'MAXIMUM SLOPE DESIGN: {slope:.1f}° with R={R:.1f}m Transitions', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. Slope profile
    ax2 = axes[0, 1]
    ds = s[1] - s[0]
    dz_ds = np.gradient(z, ds)
    slope_deg = np.degrees(np.arctan(-dz_ds))
    ax2.plot(s, slope_deg, 'b-', linewidth=2)
    ax2.axhline(y=slope, color='red', linestyle='--', label=f'Max: {slope}°')
    ax2.fill_between(s, 0, slope_deg, alpha=0.3)
    ax2.set_xlabel('Distance (m)', fontsize=12)
    ax2.set_ylabel('Slope (°)', fontsize=12)
    ax2.set_title('Slope Profile Along Ramp', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-2, slope + 5)

    # 3. Clearance
    ax3 = axes[1, 0]
    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total - margin, 300)
    down_cl = [check_clearance(p, s, z, 'downhill') * 1000 for p in positions]
    up_cl = [check_clearance(p, s, z, 'uphill') * 1000 for p in positions]

    ax3.plot(positions, down_cl, 'b-', linewidth=2, label='Downhill')
    ax3.plot(positions, up_cl, 'r-', linewidth=2, label='Uphill')
    ax3.axhline(y=MIN_CLEARANCE * 1000, color='orange', linestyle='--', linewidth=2, label=f'Min Safety ({MIN_CLEARANCE*1000:.0f}mm)')
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.fill_between(positions, 0, [min(d, u) for d, u in zip(down_cl, up_cl)],
                     where=[min(d, u) > 0 for d, u in zip(down_cl, up_cl)], alpha=0.3, color='green')
    ax3.set_xlabel('Car Center Position (m)', fontsize=12)
    ax3.set_ylabel('Ground Clearance (mm)', fontsize=12)
    ax3.set_title(f'Clearance Check: Down={min(down_cl):.1f}mm, Up={min(up_cl):.1f}mm', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-20, 120)

    # 4. Specifications
    ax4 = axes[1, 1]
    ax4.axis('off')

    specs = f"""
    ╔════════════════════════════════════════════════════════════╗
    ║         MAXIMUM SLOPE RAMP - FINAL DESIGN                  ║
    ╠════════════════════════════════════════════════════════════╣
    ║                                                            ║
    ║  MAXIMUM ACHIEVABLE SLOPE: {slope:.1f}°                          ║
    ║                                                            ║
    ║  GEOMETRY:                                                 ║
    ║    Entry flat (street):    {ENTRY_FLAT:.1f}m                          ║
    ║    Transition arc (entry): {trans_h:.2f}m  (R = {R:.1f}m)            ║
    ║    Middle straight:        {middle_h:.2f}m  (at {slope:.1f}° constant)   ║
    ║    Transition arc (exit):  {trans_h:.2f}m  (R = {R:.1f}m)            ║
    ║    Exit flat (garage):     {EXIT_FLAT:.1f}m                          ║
    ║    ─────────────────────────────────────                   ║
    ║    TOTAL LENGTH:           {total:.1f}m                         ║
    ║    ARC LENGTH:             {arc:.2f}m                        ║
    ║                                                            ║
    ║  VERTICAL:                                                 ║
    ║    Entry transition drop:  {trans_drop:.3f}m                        ║
    ║    Middle section drop:    {middle_drop:.3f}m                        ║
    ║    Exit transition drop:   {trans_drop:.3f}m                        ║
    ║    ─────────────────────────────────────                   ║
    ║    TOTAL DROP:             {VERTICAL_DROP:.1f}m                          ║
    ║                                                            ║
    ║  CLEARANCES:                                               ║
    ║    Downhill travel:        {design['down']*1000:.1f}mm                        ║
    ║    Uphill travel:          {design['up']*1000:.1f}mm                        ║
    ║    Minimum required:       {MIN_CLEARANCE*1000:.0f}mm                          ║
    ║                                                            ║
    ║  CAR: Porsche 997.1 Turbo (2008)                           ║
    ║    Ground clearance: {GROUND_CLEARANCE*1000:.0f}mm                             ║
    ║    Wheelbase: {WHEELBASE}m                                    ║
    ║    Rear overhang: {REAR_OVERHANG:.3f}m (critical constraint)        ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝

    NOTE: 24° is PHYSICALLY IMPOSSIBLE with these constraints.
    The maximum achievable slope is {slope:.1f}°.
    """

    ax4.text(0.02, 0.98, specs, transform=ax4.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.95, edgecolor='green'))

    plt.tight_layout()
    plt.savefig('/workspaces/RAMP/maximum_slope_final_design.png', dpi=200, bbox_inches='tight')
    print(f"\nBlueprint saved: maximum_slope_final_design.png")

    # Also save measurements
    save_measurements(s, z, design)


def save_measurements(s, z, design):
    """Save measurements to CSV."""
    with open('/workspaces/RAMP/maximum_slope_measurements.csv', 'w') as f:
        f.write("Distance_m,Elevation_m,Elevation_cm,Section\n")

        for i in range(0, len(s), len(s)//60):
            pos = s[i]
            elev = z[i]

            if pos < ENTRY_FLAT:
                section = "Street"
            elif pos > ENTRY_FLAT + design['arc']:
                section = "Garage"
            else:
                section = "Ramp"

            f.write(f"{pos:.2f},{elev:.4f},{elev*100:.1f},{section}\n")

    print(f"Measurements saved: maximum_slope_measurements.csv")


if __name__ == '__main__':
    main()
