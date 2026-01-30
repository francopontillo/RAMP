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

    # Car width for top view
    CAR_WIDTH = 1.808
    ramp_width = CAR_WIDTH + 0.5  # margin

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 0.8], hspace=0.3, wspace=0.25)

    # 1. Side Elevation
    ax1 = fig.add_subplot(gs[0, :])
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
    ax2 = fig.add_subplot(gs[1, 0])
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
    ax3 = fig.add_subplot(gs[1, 1])
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
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    specs = f"""
MAXIMUM SLOPE RAMP - {slope:.1f}°
════════════════════════════════════════

GEOMETRY:
  Transition radius:   {R:.1f}m
  Transition length:   {trans_h:.1f}m each
  Middle ({slope:.1f}° const): {middle_h:.1f}m
  Arc length:          {arc:.1f}m
  Total length:        {total:.1f}m

CLEARANCES:
  Downhill minimum:    {design['down']*1000:.1f}mm
  Uphill minimum:      {design['up']*1000:.1f}mm
  Required minimum:    {MIN_CLEARANCE*1000:.0f}mm

CONSTRAINTS:
  The {slope:.1f}° slope is the MAXIMUM achievable
  with this car geometry ({REAR_OVERHANG*1000:.0f}mm rear overhang)
  and {VERTICAL_DROP}m vertical drop.

  Higher slopes require gentler transitions,
  but gentler transitions need more vertical drop
  than available ({VERTICAL_DROP}m).
    """

    ax4.text(0.02, 0.98, specs, transform=ax4.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.95, edgecolor='green'))

    # 5. TOP VIEW (Plan)
    ax5 = fig.add_subplot(gs[2, 1])

    # Street section (gray - 5 meters)
    street_rect = plt.Rectangle((0, -ramp_width/2), ENTRY_FLAT, ramp_width,
                                  facecolor='#808080', edgecolor='#404040', linewidth=2, alpha=0.8)
    ax5.add_patch(street_rect)
    ax5.text(ENTRY_FLAT/2, 0, f'STREET\n{ENTRY_FLAT:.0f}m', ha='center', va='center',
             fontsize=9, fontweight='bold', color='white')

    # Ramp section (gradient from gray to blue)
    n_sections = 25
    for i in range(n_sections):
        x_start = ENTRY_FLAT + (arc / n_sections) * i
        section_width = arc / n_sections
        ratio = i / n_sections
        gray_val = 0.5 * (1 - ratio)
        blue_val = 0.5 + 0.5 * ratio
        color = (gray_val, gray_val, blue_val)
        rect = plt.Rectangle((x_start, -ramp_width/2), section_width, ramp_width,
                              facecolor=color, edgecolor='none', alpha=0.8)
        ax5.add_patch(rect)

    # Ramp outline
    ramp_outline = plt.Rectangle((ENTRY_FLAT, -ramp_width/2), arc, ramp_width,
                                   facecolor='none', edgecolor='#404040', linewidth=2)
    ax5.add_patch(ramp_outline)
    ax5.text(ENTRY_FLAT + arc/2, 0, f'RAMP\n{arc:.1f}m',
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Garage section (blue - 5 meters)
    garage_rect = plt.Rectangle((ENTRY_FLAT + arc, -ramp_width/2), EXIT_FLAT, ramp_width,
                                  facecolor='#4169E1', edgecolor='#1E3A8A', linewidth=2, alpha=0.8)
    ax5.add_patch(garage_rect)
    ax5.text(ENTRY_FLAT + arc + EXIT_FLAT/2, 0, f'GARAGE\n{EXIT_FLAT:.0f}m',
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Draw car silhouettes
    car_positions = [2.5, ENTRY_FLAT + arc * 0.33, ENTRY_FLAT + arc * 0.66, ENTRY_FLAT + arc + 2.5]
    for i, car_pos in enumerate(car_positions):
        alpha = 0.5
        car_color = '#FF6666'
        car_rect = plt.Rectangle((car_pos - CAR_LENGTH/2, -CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH,
                                   facecolor=car_color, edgecolor='darkred', linewidth=1, alpha=alpha)
        ax5.add_patch(car_rect)

    # Direction arrow
    ax5.annotate('', xy=(total - 0.5, ramp_width/2 + 0.3),
                 xytext=(0.5, ramp_width/2 + 0.3),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax5.text(total/2, ramp_width/2 + 0.5, 'DOWNHILL', ha='center', fontsize=8, fontweight='bold')

    # Dimension annotations
    ax5.annotate('', xy=(0, -ramp_width/2 - 0.4), xytext=(ENTRY_FLAT, -ramp_width/2 - 0.4),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax5.text(ENTRY_FLAT/2, -ramp_width/2 - 0.6, f'{ENTRY_FLAT:.0f}m', ha='center', fontsize=7, color='gray')

    ax5.annotate('', xy=(ENTRY_FLAT, -ramp_width/2 - 0.4), xytext=(ENTRY_FLAT + arc, -ramp_width/2 - 0.4),
                 arrowprops=dict(arrowstyle='<->', color='#404040', lw=1))
    ax5.text(ENTRY_FLAT + arc/2, -ramp_width/2 - 0.6, f'{arc:.1f}m', ha='center', fontsize=7)

    ax5.annotate('', xy=(ENTRY_FLAT + arc, -ramp_width/2 - 0.4), xytext=(total, -ramp_width/2 - 0.4),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=1))
    ax5.text(ENTRY_FLAT + arc + EXIT_FLAT/2, -ramp_width/2 - 0.6, f'{EXIT_FLAT:.0f}m', ha='center', fontsize=7, color='blue')

    ax5.set_xlim(-0.5, total + 0.5)
    ax5.set_ylim(-ramp_width/2 - 1, ramp_width/2 + 0.8)
    ax5.set_xlabel('Distance (m)', fontsize=9)
    ax5.set_ylabel('Width (m)', fontsize=9)
    ax5.set_title(f'TOP VIEW (Plan)', fontsize=10, fontweight='bold')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    plt.suptitle(f'MAXIMUM SLOPE RAMP DESIGN - {slope:.1f}°\n'
                 f'Arc: {arc:.1f}m | Total: {total:.1f}m | R: {R:.1f}m',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('/workspaces/RAMP/minimum_radial_3.5m_blueprint_24deg.png', dpi=200, bbox_inches='tight')
    print(f"\nBlueprint saved: minimum_radial_3.5m_blueprint_24deg.png")

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
