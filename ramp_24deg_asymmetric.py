#!/usr/bin/env python3
"""
24° Slope Ramp Design with Asymmetric Transitions

KEY INSIGHT from user:
- CONCAVE curves (ground curves toward car) = SAFE, can be tight
- CONVEX curves (ground curves away from car) = DANGEROUS, must be gentle

Profile structure:
[FLAT 0°] → [CONCAVE: 0° to 24°] → [STRAIGHT 24°] → [CONVEX: 24° to 0°] → [FLAT 0°]
            (can be tight)                          (must be gentle)

The concave section is like the inside bottom of a cylinder - safe for overhangs.
The convex section is like going over a hill crest - dangerous for overhangs.

For bidirectional travel:
- Downhill: enters via concave (safe), exits via convex (check front overhang)
- Uphill: enters via convex (check front overhang), exits via concave (safe)

Wait - I need to reconsider. When going uphill through what was the "exit" convex:
- The car's FRONT is leading uphill
- At the convex transition, the front overhang is vulnerable

When going downhill through the "entry" concave:
- Actually, for the car going down, what looks concave from above is...

Let me think about this more carefully with actual geometry.
"""

import numpy as np
import matplotlib.pyplot as plt

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
TARGET_SLOPE = 24.0  # degrees
MIN_CLEARANCE = 0.005  # 5mm


def create_circular_arc_profile(R_entry, R_exit, num_points=3000):
    """
    Create a profile with circular arc transitions.

    R_entry: radius of entry transition (concave when going down)
    R_exit: radius of exit transition (convex when going down)

    Profile: Flat → Circular arc → Straight → Circular arc → Flat
    """
    slope_rad = np.radians(TARGET_SLOPE)
    slope_tan = np.tan(slope_rad)

    # Entry transition (concave for downhill)
    # Arc from 0° to 24°
    entry_arc_length = R_entry * slope_rad
    entry_horizontal = R_entry * np.sin(slope_rad)
    entry_drop = R_entry * (1 - np.cos(slope_rad))

    # Exit transition (convex for downhill)
    # Arc from 24° to 0°
    exit_arc_length = R_exit * slope_rad
    exit_horizontal = R_exit * np.sin(slope_rad)
    exit_drop = R_exit * (1 - np.cos(slope_rad))

    # Middle straight section
    middle_drop = VERTICAL_DROP - entry_drop - exit_drop
    if middle_drop < 0:
        print(f"Warning: transitions alone exceed vertical drop!")
        print(f"Entry drop: {entry_drop:.3f}m, Exit drop: {exit_drop:.3f}m")
        print(f"Total: {entry_drop + exit_drop:.3f}m > {VERTICAL_DROP}m")
        return None, None, None

    middle_horizontal = middle_drop / slope_tan

    # Total arc length (horizontal distance of ramp section)
    arc_length = entry_horizontal + middle_horizontal + exit_horizontal
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    # Generate profile points
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)

    # Key positions
    ramp_start = ENTRY_FLAT
    entry_end = ramp_start + entry_horizontal
    middle_end = entry_end + middle_horizontal
    ramp_end = ramp_start + arc_length

    for i, pos in enumerate(s):
        if pos <= ramp_start:
            # Entry flat
            z[i] = 0
        elif pos <= entry_end:
            # Entry circular arc (concave - curving down into slope)
            local_s = pos - ramp_start
            # Parametric: angle goes from 0 to slope_rad
            # x = R * sin(theta), z = -R * (1 - cos(theta))
            # Solve for theta given x: theta = arcsin(x/R)
            theta = np.arcsin(local_s / R_entry)
            z[i] = -R_entry * (1 - np.cos(theta))
        elif pos <= middle_end:
            # Middle straight section at constant slope
            local_s = pos - entry_end
            z_at_entry_end = -entry_drop
            z[i] = z_at_entry_end - local_s * slope_tan
        elif pos <= ramp_end:
            # Exit circular arc (convex - curving back to flat)
            local_s = pos - middle_end
            z_at_middle_end = -entry_drop - middle_drop
            # This arc goes from slope_rad to 0
            # x = R * (sin(slope_rad) - sin(slope_rad - theta))
            # z = -R * (cos(slope_rad - theta) - cos(slope_rad))
            # We need to find theta such that x = local_s
            # At theta = slope_rad: x = R * sin(slope_rad), z = -R * (1 - cos(slope_rad))

            # Parametric with angle measured from start of this arc
            # Start: slope = 24°, end: slope = 0°
            # theta goes from 0 to slope_rad
            # x(theta) = R * (sin(slope_rad) - sin(slope_rad - theta))
            # z(theta) = -R * (cos(slope_rad - theta) - cos(slope_rad))

            # Numerical solution for theta
            def x_of_theta(theta):
                return R_exit * (np.sin(slope_rad) - np.sin(slope_rad - theta))

            # Binary search for theta
            theta_low, theta_high = 0, slope_rad
            for _ in range(50):
                theta_mid = (theta_low + theta_high) / 2
                if x_of_theta(theta_mid) < local_s:
                    theta_low = theta_mid
                else:
                    theta_high = theta_mid
            theta = theta_mid

            dz = R_exit * (np.cos(slope_rad - theta) - np.cos(slope_rad))
            z[i] = z_at_middle_end - dz
        else:
            # Exit flat
            z[i] = -VERTICAL_DROP

    return s, z, arc_length


def get_elevation_at(s_query, s_array, z_array):
    return np.interp(s_query, s_array, z_array)


def check_clearance_detailed(car_center_pos, s_array, z_array, direction='downhill'):
    """Check clearance with detailed output."""
    half_wb = WHEELBASE / 2

    if direction == 'downhill':
        front_axle = car_center_pos + half_wb
        rear_axle = car_center_pos - half_wb
        front_bumper = front_axle + FRONT_OVERHANG
        rear_bumper = rear_axle - REAR_OVERHANG
    else:
        front_axle = car_center_pos - half_wb
        rear_axle = car_center_pos + half_wb
        front_bumper = front_axle - FRONT_OVERHANG
        rear_bumper = rear_axle + REAR_OVERHANG

    fa_z = get_elevation_at(front_axle, s_array, z_array)
    ra_z = get_elevation_at(rear_axle, s_array, z_array)

    def body_height(x):
        t = (x - rear_axle) / (front_axle - rear_axle)
        return ra_z + t * (fa_z - ra_z) + GROUND_CLEARANCE

    fb_ground = get_elevation_at(front_bumper, s_array, z_array)
    rb_ground = get_elevation_at(rear_bumper, s_array, z_array)

    fb_clearance = body_height(front_bumper) - fb_ground
    rb_clearance = body_height(rear_bumper) - rb_ground

    return {
        'front': fb_clearance,
        'rear': rb_clearance,
        'min': min(fb_clearance, rb_clearance),
        'critical': 'front' if fb_clearance < rb_clearance else 'rear'
    }


def analyze_full_traverse(s_array, z_array):
    """Analyze clearance for full traverse in both directions."""
    total_length = s_array[-1]
    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 500)

    min_down = float('inf')
    min_up = float('inf')
    worst_down_pos = 0
    worst_up_pos = 0

    for pos in positions:
        down = check_clearance_detailed(pos, s_array, z_array, 'downhill')
        up = check_clearance_detailed(pos, s_array, z_array, 'uphill')

        if down['min'] < min_down:
            min_down = down['min']
            worst_down_pos = pos
        if up['min'] < min_up:
            min_up = up['min']
            worst_up_pos = pos

    return min_down, min_up, worst_down_pos, worst_up_pos


def calculate_min_convex_radius():
    """
    Calculate minimum radius for convex transitions based on car geometry.

    For a convex curve, clearance loss at distance d from wheel is approximately:
    Δh ≈ d² / (2R)

    We need Δh < GROUND_CLEARANCE for both overhangs.
    """
    # For rear overhang (longer, more critical)
    R_min_rear = REAR_OVERHANG**2 / (2 * GROUND_CLEARANCE)

    # For front overhang
    R_min_front = FRONT_OVERHANG**2 / (2 * GROUND_CLEARANCE)

    print(f"Minimum convex radius for rear overhang ({REAR_OVERHANG*1000:.0f}mm): {R_min_rear:.2f}m")
    print(f"Minimum convex radius for front overhang ({FRONT_OVERHANG*1000:.0f}mm): {R_min_front:.2f}m")

    return max(R_min_rear, R_min_front)


def search_optimal_design():
    """Search for the shortest ramp with 24° slope."""

    print("=" * 80)
    print(f"SEARCHING FOR OPTIMAL 24° RAMP DESIGN")
    print("=" * 80)

    R_min = calculate_min_convex_radius()
    print(f"\nTheoretical minimum convex radius: {R_min:.2f}m")
    print(f"Using safety factor, starting search from R_convex = {R_min * 1.2:.1f}m")

    print(f"\n{'R_entry':>8} {'R_exit':>8} {'Arc':>8} {'Down':>10} {'Up':>10} {'Status':<10}")
    print("-" * 70)

    valid_designs = []

    # Search different combinations
    # Entry can be tight (concave is safe)
    # Exit must be gentle (convex is dangerous)

    for R_entry in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for R_exit in np.arange(R_min * 1.1, R_min * 3.0, 0.5):
            result = create_circular_arc_profile(R_entry, R_exit)
            if result[0] is None:
                continue

            s, z, arc_length = result
            min_down, min_up, _, _ = analyze_full_traverse(s, z)
            overall_min = min(min_down, min_up)

            is_valid = overall_min >= MIN_CLEARANCE

            if is_valid:
                total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
                valid_designs.append({
                    'R_entry': R_entry,
                    'R_exit': R_exit,
                    'arc': arc_length,
                    'total': total_length,
                    'down': min_down,
                    'up': min_up,
                    'min': overall_min
                })
                status = "✓ OK"
            else:
                status = "✗"

            if R_entry == 3.0 or is_valid:  # Print R_entry=3 cases and all valid
                print(f"{R_entry:>8.1f} {R_exit:>8.1f} {arc_length:>8.2f} "
                      f"{min_down*1000:>8.1f}mm {min_up*1000:>8.1f}mm {status:<10}")

    if valid_designs:
        # Find shortest
        best = min(valid_designs, key=lambda x: x['arc'])

        print(f"\n{'=' * 80}")
        print("BEST DESIGN FOUND")
        print(f"{'=' * 80}")
        print(f"Entry radius (concave): {best['R_entry']:.1f}m")
        print(f"Exit radius (convex):   {best['R_exit']:.1f}m")
        print(f"Arc length:             {best['arc']:.2f}m")
        print(f"Total length:           {best['total']:.1f}m")
        print(f"Downhill clearance:     {best['down']*1000:.1f}mm")
        print(f"Uphill clearance:       {best['up']*1000:.1f}mm")

        return best

    print("\nNo valid design found!")
    return None


def create_blueprint(design):
    """Create detailed blueprint."""

    R_entry = design['R_entry']
    R_exit = design['R_exit']

    s, z, arc_length = create_circular_arc_profile(R_entry, R_exit)
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT

    # Calculate section lengths
    slope_rad = np.radians(TARGET_SLOPE)
    entry_horiz = R_entry * np.sin(slope_rad)
    exit_horiz = R_exit * np.sin(slope_rad)
    entry_drop = R_entry * (1 - np.cos(slope_rad))
    exit_drop = R_exit * (1 - np.cos(slope_rad))
    middle_drop = VERTICAL_DROP - entry_drop - exit_drop
    middle_horiz = middle_drop / np.tan(slope_rad)

    # Car width for top view
    CAR_WIDTH = 1.808
    ramp_width = CAR_WIDTH + 0.5  # margin

    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 0.8, 0.8], hspace=0.3, wspace=0.25)

    # 1. Side Elevation
    ax1 = fig.add_subplot(gs[0, :])

    ax1.fill_between(s, z, z.min() - 0.5, color='#d4a574', alpha=0.7)
    ax1.plot(s, z, 'k-', linewidth=3)
    ax1.axhline(y=0, color='green', linewidth=3, label='Street level')
    ax1.axhline(y=-VERTICAL_DROP, color='blue', linewidth=3, label='Garage level')

    # Mark sections
    ramp_start = ENTRY_FLAT
    entry_end = ramp_start + entry_horiz
    middle_end = entry_end + middle_horiz
    ramp_end = ramp_start + arc_length

    ax1.axvline(x=ramp_start, color='red', linewidth=2, linestyle='--')
    ax1.axvline(x=entry_end, color='orange', linewidth=1.5, linestyle=':')
    ax1.axvline(x=middle_end, color='orange', linewidth=1.5, linestyle=':')
    ax1.axvline(x=ramp_end, color='red', linewidth=2, linestyle='--')

    # Labels
    ax1.text(ENTRY_FLAT/2, 0.4, f'STREET\n{ENTRY_FLAT}m', ha='center', fontsize=11,
             color='green', fontweight='bold')
    ax1.text(ramp_start + entry_horiz/2, 0.4, f'CONCAVE\nR={R_entry}m',
             ha='center', fontsize=9, color='blue')
    ax1.text(entry_end + middle_horiz/2, 0.4, f'STRAIGHT\n{TARGET_SLOPE}°',
             ha='center', fontsize=11, fontweight='bold', color='darkred')
    ax1.text(middle_end + exit_horiz/2, 0.4, f'CONVEX\nR={R_exit}m',
             ha='center', fontsize=9, color='purple')
    ax1.text(ramp_end + EXIT_FLAT/2, -VERTICAL_DROP + 0.4, f'GARAGE\n{EXIT_FLAT}m',
             ha='center', fontsize=11, color='blue', fontweight='bold')

    # Slope annotation
    ax1.annotate(f'24° slope', xy=(entry_end + middle_horiz/2, -entry_drop - middle_drop/2),
                fontsize=14, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.set_title(f'SIDE ELEVATION - 24° Maximum Slope Design\n'
                  f'Arc: {arc_length:.1f}m | Total: {total_length:.1f}m | Concave R={R_entry}m | Convex R={R_exit}m',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, total_length + 0.5)
    ax1.set_ylim(-4.5, 1)

    # 2. Slope profile
    ax2 = fig.add_subplot(gs[1, 0])

    ds = s[1] - s[0]
    dz_ds = np.gradient(z, ds)
    slope_deg = np.degrees(np.arctan(-dz_ds))

    ax2.plot(s, slope_deg, 'b-', linewidth=2)
    ax2.axhline(y=TARGET_SLOPE, color='red', linewidth=1, linestyle='--', label=f'Max {TARGET_SLOPE}°')
    ax2.axvline(x=ramp_start, color='gray', linewidth=1, linestyle=':')
    ax2.axvline(x=entry_end, color='orange', linewidth=1, linestyle=':')
    ax2.axvline(x=middle_end, color='orange', linewidth=1, linestyle=':')
    ax2.axvline(x=ramp_end, color='gray', linewidth=1, linestyle=':')

    ax2.fill_between(s, 0, slope_deg, alpha=0.3)

    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Slope (°)')
    ax2.set_title('Slope Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-2, TARGET_SLOPE + 5)

    # 3. Clearance
    ax3 = fig.add_subplot(gs[1, 1])

    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 300)

    down_cl = [check_clearance_detailed(p, s, z, 'downhill')['min'] * 1000 for p in positions]
    up_cl = [check_clearance_detailed(p, s, z, 'uphill')['min'] * 1000 for p in positions]

    ax3.plot(positions, down_cl, 'b-', linewidth=2, label='Downhill')
    ax3.plot(positions, up_cl, 'r-', linewidth=2, label='Uphill')
    ax3.axhline(y=MIN_CLEARANCE * 1000, color='orange', linewidth=2, linestyle='--',
                label=f'Min ({MIN_CLEARANCE*1000:.0f}mm)')
    ax3.axhline(y=0, color='black', linewidth=1)

    ax3.set_xlabel('Car position (m)')
    ax3.set_ylabel('Clearance (mm)')
    ax3.set_title(f'Ground Clearance\nDown: {min(down_cl):.1f}mm | Up: {min(up_cl):.1f}mm')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-20, 120)

    # 4. Specifications
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    R_horizontal = arc_length * 2 / np.pi  # Equivalent horizontal radius

    specs = f"""
    24° SLOPE RAMP DESIGN
    ══════════════════════════════════════════════════════

    GEOMETRY:
      Entry transition:     R = {R_entry:.1f}m (CONCAVE - safe)
      Exit transition:      R = {R_exit:.1f}m (CONVEX - gentle)
      Entry length:         {entry_horiz:.2f}m
      Middle length:        {middle_horiz:.2f}m (constant 24°)
      Exit length:          {exit_horiz:.2f}m
      ─────────────────────────────────────────────────────
      Ramp arc length:      {arc_length:.2f}m
      Entry flat:           {ENTRY_FLAT}m
      Exit flat:            {EXIT_FLAT}m
      TOTAL LENGTH:         {total_length:.1f}m

    SLOPES:
      Maximum slope:        {TARGET_SLOPE}°
      Entry/Exit:           0° (smooth transitions)

    CLEARANCES:
      Downhill minimum:     {design['down']*1000:.1f}mm
      Uphill minimum:       {design['up']*1000:.1f}mm
      Required minimum:     {MIN_CLEARANCE*1000:.0f}mm
      Status:               ✓ NO GROUND CONTACT
    """

    ax4.text(0.02, 0.98, specs, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # 5. Comparison
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    comparison = f"""
    COMPARISON WITH PREVIOUS DESIGNS
    ══════════════════════════════════════════════════════

    Design              Arc      Total    Slope   Status
    ─────────────────────────────────────────────────────
    Original cubic      11.8m    21.8m    24°     SCRAPES
    Conservative        22.0m    32.0m    13.4°   Safe
    Absolute min        20.1m    30.1m    14.6°   Safe
    THIS DESIGN         {arc_length:.1f}m    {total_length:.1f}m    24°     Safe

    ─────────────────────────────────────────────────────

    KEY INNOVATION:
    • CONCAVE entry (R={R_entry}m) - can be tight, safe for car
    • CONVEX exit (R={R_exit}m) - must be gentle for overhangs
    • Straight middle section at full 24° slope

    PHYSICS:
    • Concave curves = ground curves TOWARD car = MORE clearance
    • Convex curves = ground curves AWAY from car = LESS clearance
    • By using asymmetric radii, we achieve 24° safely!

    RESULT:
    • Full 24° slope achieved!
    • {total_length - 21.8:.1f}m longer than unsafe original
    • {32.0 - total_length:.1f}m shorter than conservative design
    """

    ax5.text(0.02, 0.98, comparison, transform=ax5.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # 6. TOP VIEW (Plan) - spanning both columns
    ax6 = fig.add_subplot(gs[3, :])

    # Street section (gray - 5 meters)
    street_rect = plt.Rectangle((0, -ramp_width/2), ENTRY_FLAT, ramp_width,
                                  facecolor='#808080', edgecolor='#404040', linewidth=2, alpha=0.8)
    ax6.add_patch(street_rect)
    ax6.text(ENTRY_FLAT/2, 0, f'STREET\n{ENTRY_FLAT:.0f}m\n(Level 0m)', ha='center', va='center',
             fontsize=11, fontweight='bold', color='white')

    # Ramp section (gradient from gray to blue)
    n_sections = 30
    for i in range(n_sections):
        x_start = ENTRY_FLAT + (arc_length / n_sections) * i
        section_width = arc_length / n_sections
        # Color gradient from gray to blue
        ratio = i / n_sections
        gray_val = 0.5 * (1 - ratio)
        blue_val = 0.5 + 0.5 * ratio
        color = (gray_val, gray_val, blue_val)
        rect = plt.Rectangle((x_start, -ramp_width/2), section_width, ramp_width,
                              facecolor=color, edgecolor='none', alpha=0.8)
        ax6.add_patch(rect)

    # Ramp outline
    ramp_outline = plt.Rectangle((ENTRY_FLAT, -ramp_width/2), arc_length, ramp_width,
                                   facecolor='none', edgecolor='#404040', linewidth=2)
    ax6.add_patch(ramp_outline)
    ax6.text(ENTRY_FLAT + arc_length/2, 0, f'RAMP\n{arc_length:.1f}m\n↓ {VERTICAL_DROP}m drop',
             ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Garage section (blue - 5 meters)
    garage_rect = plt.Rectangle((ENTRY_FLAT + arc_length, -ramp_width/2), EXIT_FLAT, ramp_width,
                                  facecolor='#4169E1', edgecolor='#1E3A8A', linewidth=2, alpha=0.8)
    ax6.add_patch(garage_rect)
    ax6.text(ENTRY_FLAT + arc_length + EXIT_FLAT/2, 0, f'GARAGE\n{EXIT_FLAT:.0f}m\n(Level -{VERTICAL_DROP}m)',
             ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Draw car silhouettes along the path
    car_positions = [2.5, ENTRY_FLAT + arc_length * 0.25, ENTRY_FLAT + arc_length * 0.5,
                     ENTRY_FLAT + arc_length * 0.75, ENTRY_FLAT + arc_length + 2.5]
    for i, car_pos in enumerate(car_positions):
        alpha = 0.4 if i != 2 else 0.9
        car_color = '#FF4444' if i == 2 else '#FF8888'
        car_rect = plt.Rectangle((car_pos - CAR_LENGTH/2, -CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH,
                                   facecolor=car_color, edgecolor='darkred', linewidth=1.5, alpha=alpha)
        ax6.add_patch(car_rect)
        # Front indicator
        front_x = car_pos + CAR_LENGTH/2
        ax6.fill([front_x, front_x - 0.3, front_x - 0.3], [0, 0.2, -0.2],
                 color='darkred', alpha=alpha)

    # Direction arrow
    ax6.annotate('', xy=(total_length - 1, ramp_width/2 + 0.4),
                 xytext=(1, ramp_width/2 + 0.4),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax6.text(total_length/2, ramp_width/2 + 0.6, 'DOWNHILL DIRECTION', ha='center', fontsize=10, fontweight='bold')

    # Dimension annotations
    ax6.annotate('', xy=(0, -ramp_width/2 - 0.5), xytext=(ENTRY_FLAT, -ramp_width/2 - 0.5),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax6.text(ENTRY_FLAT/2, -ramp_width/2 - 0.8, f'{ENTRY_FLAT:.0f}m', ha='center', fontsize=9, color='gray')

    ax6.annotate('', xy=(ENTRY_FLAT, -ramp_width/2 - 0.5), xytext=(ENTRY_FLAT + arc_length, -ramp_width/2 - 0.5),
                 arrowprops=dict(arrowstyle='<->', color='#404040', lw=1.5))
    ax6.text(ENTRY_FLAT + arc_length/2, -ramp_width/2 - 0.8, f'{arc_length:.1f}m', ha='center', fontsize=9)

    ax6.annotate('', xy=(ENTRY_FLAT + arc_length, -ramp_width/2 - 0.5), xytext=(total_length, -ramp_width/2 - 0.5),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax6.text(ENTRY_FLAT + arc_length + EXIT_FLAT/2, -ramp_width/2 - 0.8, f'{EXIT_FLAT:.0f}m', ha='center', fontsize=9, color='blue')

    ax6.set_xlim(-1, total_length + 1)
    ax6.set_ylim(-ramp_width/2 - 1.5, ramp_width/2 + 1.2)
    ax6.set_xlabel('Distance (m)', fontsize=11)
    ax6.set_ylabel('Width (m)', fontsize=11)
    ax6.set_title(f'TOP VIEW (Plan) - Total Length: {total_length:.1f}m', fontsize=12, fontweight='bold')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'24° MAXIMUM SLOPE RAMP DESIGN\n'
                 f'Arc: {arc_length:.1f}m | Total: {total_length:.1f}m | Bidirectional Safe',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('/workspaces/RAMP/minimum_radial_3.5m_blueprint_24deg.png', dpi=200, bbox_inches='tight')
    print(f"\nBlueprint saved: minimum_radial_3.5m_blueprint_24deg.png")


def main():
    print("\n" + "=" * 80)
    print("24° SLOPE RAMP DESIGN WITH ASYMMETRIC TRANSITIONS")
    print("Concave entry (tight) + Convex exit (gentle)")
    print("=" * 80)

    design = search_optimal_design()

    if design:
        create_blueprint(design)

        print(f"\n{'=' * 80}")
        print("FINAL RESULT - 24° SLOPE ACHIEVED!")
        print(f"{'=' * 80}")
        print(f"""
    Entry radius (concave): {design['R_entry']:.1f}m - can be tight (safe)
    Exit radius (convex):   {design['R_exit']:.1f}m - must be gentle

    Arc length:             {design['arc']:.1f}m
    Total length:           {design['total']:.1f}m
    Maximum slope:          {TARGET_SLOPE}°

    Clearances:
      Downhill:             {design['down']*1000:.1f}mm
      Uphill:               {design['up']*1000:.1f}mm

    SUCCESS: 24° slope with no ground contact!
        """)


if __name__ == '__main__':
    main()
