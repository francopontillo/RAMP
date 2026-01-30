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

    # Car width for top view
    CAR_WIDTH = 1.808
    ramp_width = CAR_WIDTH + 0.5  # Some margin

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.25)

    # 1. Side Elevation
    ax1 = fig.add_subplot(gs[0, 0])
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
    ax2 = fig.add_subplot(gs[0, 1])
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
    ax3 = fig.add_subplot(gs[1, 0])
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
    ax4 = fig.add_subplot(gs[1, 1])
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

    # 5. TOP VIEW (Plan) - spanning both columns
    ax5 = fig.add_subplot(gs[2, :])

    # Draw the track from above (street -> ramp -> garage)
    # Street section (gray)
    street_rect = plt.Rectangle((0, -ramp_width/2), ENTRY_FLAT, ramp_width,
                                  facecolor='#808080', edgecolor='#404040', linewidth=2, alpha=0.8)
    ax5.add_patch(street_rect)
    ax5.text(ENTRY_FLAT/2, 0, f'STREET\n{ENTRY_FLAT:.0f}m\n(Level 0m)', ha='center', va='center',
             fontsize=11, fontweight='bold', color='white')

    # Ramp section (gradient from gray to blue)
    n_sections = 30
    for i in range(n_sections):
        x_start = ENTRY_FLAT + (arc / n_sections) * i
        section_width = arc / n_sections
        # Color gradient from gray to light blue
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
    ax5.text(ENTRY_FLAT + arc/2, 0, f'RAMP\n{arc:.1f}m\n↓ {VERTICAL_DROP}m drop',
             ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Garage section (blue)
    garage_rect = plt.Rectangle((ENTRY_FLAT + arc, -ramp_width/2), EXIT_FLAT, ramp_width,
                                  facecolor='#4169E1', edgecolor='#1E3A8A', linewidth=2, alpha=0.8)
    ax5.add_patch(garage_rect)
    ax5.text(ENTRY_FLAT + arc + EXIT_FLAT/2, 0, f'GARAGE\n{EXIT_FLAT:.0f}m\n(Level -{VERTICAL_DROP}m)',
             ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Draw car silhouettes along the path
    car_positions = [2.5, ENTRY_FLAT + arc * 0.25, ENTRY_FLAT + arc * 0.5,
                     ENTRY_FLAT + arc * 0.75, ENTRY_FLAT + arc + 2.5]
    for i, car_pos in enumerate(car_positions):
        alpha = 0.4 if i != 2 else 0.9
        car_color = '#FF4444' if i == 2 else '#FF8888'
        car_rect = plt.Rectangle((car_pos - CAR_LENGTH/2, -CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH,
                                   facecolor=car_color, edgecolor='darkred', linewidth=1.5, alpha=alpha)
        ax5.add_patch(car_rect)
        # Front indicator
        front_x = car_pos + CAR_LENGTH/2
        ax5.fill([front_x, front_x - 0.3, front_x - 0.3], [0, 0.2, -0.2],
                 color='darkred', alpha=alpha)

    # Direction arrow
    ax5.annotate('', xy=(total - 1, ramp_width/2 + 0.4),
                 xytext=(1, ramp_width/2 + 0.4),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax5.text(total/2, ramp_width/2 + 0.6, 'DOWNHILL DIRECTION', ha='center', fontsize=10, fontweight='bold')

    # Dimension annotations
    ax5.annotate('', xy=(0, -ramp_width/2 - 0.5), xytext=(ENTRY_FLAT, -ramp_width/2 - 0.5),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax5.text(ENTRY_FLAT/2, -ramp_width/2 - 0.8, f'{ENTRY_FLAT:.0f}m', ha='center', fontsize=9, color='gray')

    ax5.annotate('', xy=(ENTRY_FLAT, -ramp_width/2 - 0.5), xytext=(ENTRY_FLAT + arc, -ramp_width/2 - 0.5),
                 arrowprops=dict(arrowstyle='<->', color='#404040', lw=1.5))
    ax5.text(ENTRY_FLAT + arc/2, -ramp_width/2 - 0.8, f'{arc:.1f}m', ha='center', fontsize=9)

    ax5.annotate('', xy=(ENTRY_FLAT + arc, -ramp_width/2 - 0.5), xytext=(total, -ramp_width/2 - 0.5),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax5.text(ENTRY_FLAT + arc + EXIT_FLAT/2, -ramp_width/2 - 0.8, f'{EXIT_FLAT:.0f}m', ha='center', fontsize=9, color='blue')

    ax5.set_xlim(-1, total + 1)
    ax5.set_ylim(-ramp_width/2 - 1.5, ramp_width/2 + 1.2)
    ax5.set_xlabel('Distance (m)', fontsize=11)
    ax5.set_ylabel('Width (m)', fontsize=11)
    ax5.set_title(f'TOP VIEW (Plan) - Total Length: {total:.1f}m', fontsize=12, fontweight='bold')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout(h_pad=2.0)
    plt.savefig('/workspaces/RAMP/minimum_radial_3.5m_blueprint_24deg.png', dpi=200, bbox_inches='tight')
    print(f"\nBlueprint saved: minimum_radial_3.5m_blueprint_24deg.png")


if __name__ == '__main__':
    main()
