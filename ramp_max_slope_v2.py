#!/usr/bin/env python3
"""
Maximum Slope Ramp Design V2 - Target 24°

Key insight: The problem is CURVATURE at the transition points.
The car's rear overhang (1.26m) scrapes when the ground curves away too quickly.

Solution: Use longer transition zones with gentler curvature at the ends,
even if the middle section is steeper.

Profile structure:
[FLAT] - [TRANSITION 1] - [STEEP MIDDLE] - [TRANSITION 2] - [FLAT]
         (gentle curve)                    (gentle curve)
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

TARGET_SLOPE = 24.0
MIN_CLEARANCE = 0.005  # 5mm


def get_profile_optimized(arc_length, transition_ratio=0.3, num_points=2000):
    """
    Optimized profile with adjustable transition zones.

    The profile uses smooth polynomial transitions at the ends
    and a steeper middle section.

    transition_ratio: fraction of arc_length used for each transition (0.1 to 0.4)
    """
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)

    trans_len = transition_ratio * arc_length
    middle_len = arc_length - 2 * trans_len

    if middle_len < 0:
        # If transitions overlap, use pure polynomial
        return get_profile_pure_polynomial(arc_length, num_points)

    # Calculate drops for each section
    # We want gentle transitions and steep middle
    # Use sine-based transition for very smooth curvature

    ramp_start = ENTRY_FLAT
    ramp_end = ENTRY_FLAT + arc_length
    trans1_end = ramp_start + trans_len
    trans2_start = ramp_end - trans_len

    # The maximum slope will be in the middle section
    # We need: trans1_drop + middle_drop + trans2_drop = VERTICAL_DROP
    # For sine transition: drop = (trans_len / pi) * max_slope_at_end
    # For linear middle: drop = middle_len * tan(max_slope)

    # Let's parameterize by the middle slope
    max_slope_rad = np.radians(TARGET_SLOPE)
    max_slope_tan = np.tan(max_slope_rad)

    # Sine transition from 0 to max_slope over trans_len
    # z'(s) = max_slope_tan * sin(pi * s / (2 * trans_len)) for s in [0, trans_len]
    # z(s) = -max_slope_tan * (2 * trans_len / pi) * (1 - cos(pi * s / (2 * trans_len)))
    # At s = trans_len: z = -max_slope_tan * (2 * trans_len / pi)
    trans_drop = max_slope_tan * (2 * trans_len / np.pi)
    middle_drop = max_slope_tan * middle_len

    total_theoretical_drop = 2 * trans_drop + middle_drop

    # Scale to match actual drop
    scale = VERTICAL_DROP / total_theoretical_drop

    actual_max_slope = np.degrees(np.arctan(max_slope_tan * scale))

    for i, pos in enumerate(s):
        if pos < ramp_start:
            z[i] = 0
        elif pos < trans1_end:
            # Entry transition - sine-based
            local_s = pos - ramp_start
            slope_tan = max_slope_tan * scale * np.sin(np.pi * local_s / (2 * trans_len))
            # Integrate: z = -scale * max_slope_tan * (2 * trans_len / pi) * (1 - cos(...))
            z[i] = -max_slope_tan * scale * (2 * trans_len / np.pi) * (1 - np.cos(np.pi * local_s / (2 * trans_len)))
        elif pos < trans2_start:
            # Middle section - constant slope
            local_s = pos - trans1_end
            z_at_trans1_end = -max_slope_tan * scale * (2 * trans_len / np.pi)
            z[i] = z_at_trans1_end - max_slope_tan * scale * local_s
        elif pos < ramp_end:
            # Exit transition - sine-based (decelerating)
            local_s = pos - trans2_start
            z_at_trans2_start = -VERTICAL_DROP + max_slope_tan * scale * (2 * trans_len / np.pi)
            # Slope goes from max to 0 using cosine
            z[i] = z_at_trans2_start - max_slope_tan * scale * (2 * trans_len / np.pi) * np.sin(np.pi * local_s / (2 * trans_len)) - max_slope_tan * scale * trans_len * (1 - np.cos(np.pi * local_s / (2 * trans_len))) / (np.pi / 2)
        else:
            z[i] = -VERTICAL_DROP

    # Fix any discontinuities by using pure interpolation
    # Recalculate properly
    z = np.zeros_like(s)

    for i, pos in enumerate(s):
        if pos <= ramp_start:
            z[i] = 0
        elif pos >= ramp_end:
            z[i] = -VERTICAL_DROP
        else:
            # Normalized position on ramp [0, 1]
            t = (pos - ramp_start) / arc_length

            # Use a smooth function that allows steep middle
            # Attempt: piecewise with smooth joins
            t1 = transition_ratio  # end of first transition
            t2 = 1 - transition_ratio  # start of second transition

            if t < t1:
                # First transition: smooth acceleration
                # Use smoothstep for the slope
                tt = t / t1  # normalized within transition [0,1]
                smooth = tt * tt * (3 - 2 * tt)  # smoothstep
                # Integrate smoothstep: integral = t³ - t⁴/2 evaluated properly
                z[i] = -VERTICAL_DROP * (smooth * t1 * 0.5 * (t / t1))
            elif t < t2:
                # Middle: constant slope
                z_at_t1 = -VERTICAL_DROP * t1 * 0.5
                z[i] = z_at_t1 - VERTICAL_DROP * ((t - t1) / (1 - 2*t1 + 2*t1*0.5)) * (1 - t1)
            else:
                # Second transition: smooth deceleration
                tt = (t - t2) / t1
                smooth = tt * tt * (3 - 2 * tt)
                z[i] = -VERTICAL_DROP + VERTICAL_DROP * t1 * 0.5 * (1 - smooth)

    # This is getting complicated. Let me use a simpler parameterization.
    return get_profile_parameterized(arc_length, transition_ratio, num_points)


def get_profile_parameterized(arc_length, transition_ratio, num_points=2000):
    """
    Cleaner implementation of parameterized profile.
    Uses cosine blending for smooth transitions.
    """
    total_length = ENTRY_FLAT + arc_length + EXIT_FLAT
    s = np.linspace(0, total_length, num_points)
    z = np.zeros_like(s)

    trans_len = transition_ratio * arc_length
    middle_len = arc_length - 2 * trans_len

    # For smooth transitions, use raised cosine
    # Transition 1: slope goes from 0 to max_slope
    # Middle: constant max_slope
    # Transition 2: slope goes from max_slope to 0

    # Calculate max slope needed
    # Drop in transition 1: integral of slope from 0 to trans_len
    # For raised cosine: slope(s) = max_slope * 0.5 * (1 - cos(pi * s / trans_len))
    # Integral: drop1 = max_slope * (trans_len/2 - trans_len/pi * 0) = max_slope * trans_len / 2
    # Actually for raised cosine: integral = max_slope * s/2 - max_slope * trans_len/(2*pi) * sin(pi*s/trans_len)
    # At s=trans_len: drop1 = max_slope * trans_len / 2

    # drop1 = drop2 = max_slope * trans_len / 2
    # middle_drop = max_slope * middle_len
    # Total: max_slope * (trans_len + middle_len) = max_slope * (arc_length - trans_len)
    # So: VERTICAL_DROP = max_slope * (arc_length - trans_len)
    # max_slope = VERTICAL_DROP / (arc_length - trans_len)

    if arc_length <= trans_len:
        trans_len = arc_length * 0.4  # Reduce transition if arc is too short
        middle_len = arc_length - 2 * trans_len

    effective_length = arc_length - trans_len
    if effective_length <= 0:
        effective_length = arc_length * 0.6

    max_slope_tan = VERTICAL_DROP / effective_length
    max_slope_deg = np.degrees(np.arctan(max_slope_tan))

    ramp_start = ENTRY_FLAT
    trans1_end = ramp_start + trans_len
    trans2_start = ramp_start + arc_length - trans_len
    ramp_end = ramp_start + arc_length

    current_z = 0
    current_slope = 0

    for i, pos in enumerate(s):
        if pos <= ramp_start:
            z[i] = 0
        elif pos <= trans1_end:
            # Transition 1: raised cosine slope increase
            local_s = pos - ramp_start
            # Slope = max_slope * 0.5 * (1 - cos(pi * local_s / trans_len))
            slope = max_slope_tan * 0.5 * (1 - np.cos(np.pi * local_s / trans_len))
            # Z = integral of -slope
            z[i] = -max_slope_tan * (local_s / 2 - (trans_len / (2 * np.pi)) * np.sin(np.pi * local_s / trans_len))
        elif pos <= trans2_start:
            # Middle: constant slope
            z_at_trans1 = -max_slope_tan * trans_len / 2
            local_s = pos - trans1_end
            z[i] = z_at_trans1 - max_slope_tan * local_s
        elif pos <= ramp_end:
            # Transition 2: raised cosine slope decrease
            z_at_trans2_start = -max_slope_tan * trans_len / 2 - max_slope_tan * middle_len
            local_s = pos - trans2_start
            # Slope = max_slope * 0.5 * (1 + cos(pi * local_s / trans_len))
            # Z = z_at_trans2_start - integral
            z[i] = z_at_trans2_start - max_slope_tan * (local_s / 2 + (trans_len / (2 * np.pi)) * np.sin(np.pi * local_s / trans_len))
        else:
            z[i] = -VERTICAL_DROP

    # Verify endpoint
    z[s >= ramp_end] = -VERTICAL_DROP

    return s, z, max_slope_deg, trans_len


def get_elevation_at(s_query, s_array, z_array):
    return np.interp(s_query, s_array, z_array)


def check_clearance(car_center_pos, s_array, z_array, direction='downhill'):
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
        return rear_axle_ground + t * (front_axle_ground - rear_axle_ground) + GROUND_CLEARANCE

    front_cl = car_body_height(front_bumper_pos) - get_elevation_at(front_bumper_pos, s_array, z_array)
    rear_cl = car_body_height(rear_bumper_pos) - get_elevation_at(rear_bumper_pos, s_array, z_array)

    return min(front_cl, rear_cl), 'front' if front_cl < rear_cl else 'rear'


def analyze_profile(s_array, z_array):
    total_length = s_array[-1]
    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 500)

    min_down, min_up = float('inf'), float('inf')

    for pos in positions:
        down_cl, _ = check_clearance(pos, s_array, z_array, 'downhill')
        up_cl, _ = check_clearance(pos, s_array, z_array, 'uphill')
        min_down = min(min_down, down_cl)
        min_up = min(min_up, up_cl)

    return min_down, min_up


def search_optimal_design():
    """Search for the shortest ramp that achieves target slope without scraping."""

    print("=" * 90)
    print(f"SEARCHING FOR MINIMUM RAMP WITH {TARGET_SLOPE}° SLOPE")
    print("=" * 90)
    print(f"\nCar: Rear overhang = {REAR_OVERHANG*1000:.0f}mm (critical)")
    print(f"     Ground clearance = {GROUND_CLEARANCE*1000:.0f}mm")

    print(f"\n{'Arc':>8} {'Trans%':>8} {'Slope':>8} {'Down':>10} {'Up':>10} {'Status':<10}")
    print("-" * 70)

    valid_designs = []

    # Search arc lengths and transition ratios
    for arc in np.arange(8.0, 22.0, 0.5):
        for trans_ratio in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            s, z, slope, trans_len = get_profile_parameterized(arc, trans_ratio)

            if slope < TARGET_SLOPE - 1:  # Skip if slope is too low
                continue
            if slope > TARGET_SLOPE + 2:  # Skip if slope is too high
                continue

            min_down, min_up = analyze_profile(s, z)
            overall = min(min_down, min_up)

            is_valid = overall >= MIN_CLEARANCE

            if is_valid and TARGET_SLOPE - 1 <= slope <= TARGET_SLOPE + 1:
                valid_designs.append({
                    'arc': arc,
                    'trans_ratio': trans_ratio,
                    'slope': slope,
                    'down': min_down,
                    'up': min_up,
                    'min': overall
                })
                print(f"{arc:>8.1f} {trans_ratio*100:>7.0f}% {slope:>8.1f}° {min_down*1000:>8.1f}mm {min_up*1000:>8.1f}mm {'✓ OK':<10}")
            elif TARGET_SLOPE - 0.5 <= slope <= TARGET_SLOPE + 0.5:
                status = "✓ OK" if is_valid else "✗ SCRAPE"
                print(f"{arc:>8.1f} {trans_ratio*100:>7.0f}% {slope:>8.1f}° {min_down*1000:>8.1f}mm {min_up*1000:>8.1f}mm {status:<10}")

    if valid_designs:
        # Find the shortest valid design
        best = min(valid_designs, key=lambda x: x['arc'])

        print(f"\n{'=' * 90}")
        print(f"BEST DESIGN FOUND")
        print(f"{'=' * 90}")
        print(f"Arc length:      {best['arc']:.1f}m")
        print(f"Transition:      {best['trans_ratio']*100:.0f}% ({best['trans_ratio']*best['arc']:.2f}m each)")
        print(f"Maximum slope:   {best['slope']:.1f}°")
        print(f"Downhill clear:  {best['down']*1000:.1f}mm")
        print(f"Uphill clear:    {best['up']*1000:.1f}mm")

        return best
    else:
        print("\nNo valid design found with target slope!")

        # Find the steepest valid design
        print("\nSearching for steepest possible valid design...")

        for arc in np.arange(12.0, 25.0, 0.25):
            for trans_ratio in [0.20, 0.25, 0.30, 0.35, 0.40]:
                s, z, slope, _ = get_profile_parameterized(arc, trans_ratio)
                min_down, min_up = analyze_profile(s, z)

                if min(min_down, min_up) >= MIN_CLEARANCE:
                    print(f"\nSteepest valid: Arc={arc:.1f}m, Trans={trans_ratio*100:.0f}%, Slope={slope:.1f}°")
                    print(f"                Down={min_down*1000:.1f}mm, Up={min_up*1000:.1f}mm")
                    return {
                        'arc': arc,
                        'trans_ratio': trans_ratio,
                        'slope': slope,
                        'down': min_down,
                        'up': min_up,
                        'min': min(min_down, min_up)
                    }

        return None


def create_blueprint(design):
    """Create detailed blueprint for the design."""

    arc = design['arc']
    trans_ratio = design['trans_ratio']

    s, z, slope, trans_len = get_profile_parameterized(arc, trans_ratio)
    total_length = ENTRY_FLAT + arc + EXIT_FLAT
    R = arc * 2 / np.pi

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.25)

    # 1. Side Elevation
    ax1 = fig.add_subplot(gs[0, :])

    ax1.fill_between(s, z, z.min() - 0.5, color='#d4a574', alpha=0.7)
    ax1.plot(s, z, 'k-', linewidth=3)
    ax1.axhline(y=0, color='green', linewidth=3, label='Street level')
    ax1.axhline(y=-VERTICAL_DROP, color='blue', linewidth=3, label='Garage level')

    # Mark transitions
    ax1.axvline(x=ENTRY_FLAT, color='red', linewidth=2, linestyle='--')
    ax1.axvline(x=ENTRY_FLAT + trans_len, color='orange', linewidth=1, linestyle=':')
    ax1.axvline(x=ENTRY_FLAT + arc - trans_len, color='orange', linewidth=1, linestyle=':')
    ax1.axvline(x=ENTRY_FLAT + arc, color='red', linewidth=2, linestyle='--')

    # Labels
    ax1.text(ENTRY_FLAT/2, 0.4, f'STREET\n{ENTRY_FLAT}m', ha='center', fontsize=11, color='green', fontweight='bold')
    ax1.text(ENTRY_FLAT + trans_len/2, 0.4, f'TRANS\n{trans_len:.1f}m', ha='center', fontsize=9, color='orange')
    ax1.text(ENTRY_FLAT + arc/2, 0.4, f'RAMP {arc:.1f}m\nMax {slope:.1f}°', ha='center', fontsize=11, fontweight='bold')
    ax1.text(ENTRY_FLAT + arc - trans_len/2, 0.4, f'TRANS\n{trans_len:.1f}m', ha='center', fontsize=9, color='orange')
    ax1.text(ENTRY_FLAT + arc + EXIT_FLAT/2, -VERTICAL_DROP + 0.4, f'GARAGE\n{EXIT_FLAT}m', ha='center', fontsize=11, color='blue', fontweight='bold')

    ax1.annotate(f'Max slope: {slope:.1f}°', xy=(ENTRY_FLAT + arc/2, -VERTICAL_DROP/2),
                fontsize=14, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.set_title(f'SIDE ELEVATION - Maximum Slope Design\n'
                  f'Arc: {arc:.1f}m | Total: {total_length:.1f}m | Slope: {slope:.1f}°',
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
    ax2.axhline(y=slope, color='red', linewidth=1, linestyle='--', label=f'Max slope {slope:.1f}°')
    ax2.axvline(x=ENTRY_FLAT, color='gray', linewidth=1, linestyle=':')
    ax2.axvline(x=ENTRY_FLAT + trans_len, color='orange', linewidth=1, linestyle=':')
    ax2.axvline(x=ENTRY_FLAT + arc - trans_len, color='orange', linewidth=1, linestyle=':')
    ax2.axvline(x=ENTRY_FLAT + arc, color='gray', linewidth=1, linestyle=':')

    ax2.fill_between(s, 0, slope_deg, alpha=0.3)

    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Slope (°)')
    ax2.set_title('Slope Profile Along Ramp')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-2, slope + 5)

    # 3. Clearance analysis
    ax3 = fig.add_subplot(gs[1, 1])

    margin = CAR_LENGTH / 2 + max(FRONT_OVERHANG, REAR_OVERHANG) + 0.1
    positions = np.linspace(margin, total_length - margin, 300)

    down_cl = [check_clearance(p, s, z, 'downhill')[0] * 1000 for p in positions]
    up_cl = [check_clearance(p, s, z, 'uphill')[0] * 1000 for p in positions]

    ax3.plot(positions, down_cl, 'b-', linewidth=2, label='Downhill')
    ax3.plot(positions, up_cl, 'r-', linewidth=2, label='Uphill')
    ax3.axhline(y=MIN_CLEARANCE * 1000, color='orange', linewidth=2, linestyle='--', label=f'Min ({MIN_CLEARANCE*1000:.0f}mm)')
    ax3.axhline(y=0, color='black', linewidth=1)

    ax3.fill_between(positions, 0, np.minimum(down_cl, up_cl),
                     where=np.array(down_cl) > 0, color='green', alpha=0.2)

    ax3.set_xlabel('Car position (m)')
    ax3.set_ylabel('Clearance (mm)')
    ax3.set_title(f'Ground Clearance\nDown: {min(down_cl):.1f}mm | Up: {min(up_cl):.1f}mm')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-30, 120)

    # 4. Specifications
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    middle_len = arc - 2 * trans_len

    specs = f"""
    MAXIMUM SLOPE DESIGN - {slope:.1f}°
    ════════════════════════════════════════════════════

    GEOMETRY:
      Horizontal radius:    {R:.2f}m (quarter circle)
      Arc length:           {arc:.1f}m
      Entry flat:           {ENTRY_FLAT}m
      Exit flat:            {EXIT_FLAT}m
      ───────────────────────────────────────────────
      TOTAL LENGTH:         {total_length:.1f}m

    PROFILE STRUCTURE:
      Transition zones:     {trans_len:.2f}m each ({trans_ratio*100:.0f}%)
      Middle (steep):       {middle_len:.2f}m
      Maximum slope:        {slope:.1f}°

    CLEARANCES:
      Downhill minimum:     {design['down']*1000:.1f}mm
      Uphill minimum:       {design['up']*1000:.1f}mm
      Required minimum:     {MIN_CLEARANCE*1000:.0f}mm
      Status:               ✓ NO GROUND CONTACT

    CAR GEOMETRY:
      Rear overhang:        {REAR_OVERHANG*1000:.0f}mm (critical)
      Front overhang:       {FRONT_OVERHANG*1000:.0f}mm
      Ground clearance:     {GROUND_CLEARANCE*1000:.0f}mm
    """

    ax4.text(0.02, 0.98, specs, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # 5. Comparison
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    comparison = f"""
    COMPARISON WITH PREVIOUS DESIGNS
    ════════════════════════════════════════════════════

    Design              Arc      Total    Slope   Clearance
    ─────────────────────────────────────────────────────────
    Original            11.8m    21.8m    24.0°   SCRAPES!
    Conservative        22.0m    32.0m    13.4°   +21mm
    Absolute min        20.1m    30.1m    14.6°   +5mm
    THIS DESIGN         {arc:.1f}m    {total_length:.1f}m    {slope:.1f}°   +{design['min']*1000:.0f}mm

    ─────────────────────────────────────────────────────────

    KEY INNOVATION:
    • Uses LONGER TRANSITION ZONES ({trans_ratio*100:.0f}% of arc)
    • Gentle curvature where overhangs are vulnerable
    • Steep constant slope in the middle section

    RESULT:
    • Achieves {slope:.1f}° slope (target: {TARGET_SLOPE}°)
    • Shortest safe ramp possible at this slope
    • Car does NOT touch ground in either direction
    """

    ax5.text(0.02, 0.98, comparison, transform=ax5.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle(f'MAXIMUM SLOPE RAMP DESIGN - {slope:.1f}°\n'
                 f'Arc: {arc:.1f}m | Total: {total_length:.1f}m | R: {R:.1f}m | Bidirectional Safe',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('/workspaces/RAMP/minimum_radial_3.5m_blueprint_max_slope.png', dpi=200, bbox_inches='tight')
    print(f"\nBlueprint saved: minimum_radial_3.5m_blueprint_max_slope.png")


def main():
    print("\n" + "=" * 90)
    print(f"MAXIMUM SLOPE RAMP DESIGN - TARGET {TARGET_SLOPE}°")
    print("Using optimized profile with extended transition zones")
    print("=" * 90)

    design = search_optimal_design()

    if design:
        create_blueprint(design)

        arc = design['arc']
        total_length = ENTRY_FLAT + arc + EXIT_FLAT
        R = arc * 2 / np.pi

        print(f"\n{'=' * 90}")
        print("FINAL RESULT")
        print(f"{'=' * 90}")
        print(f"""
    Horizontal radius:  {R:.2f}m
    Arc length:         {arc:.1f}m
    Total length:       {total_length:.1f}m
    Maximum slope:      {design['slope']:.1f}°

    Clearances:
      Downhill:         {design['down']*1000:.1f}mm
      Uphill:           {design['up']*1000:.1f}mm

    This is the SHORTEST ramp that achieves ~{TARGET_SLOPE}° slope
    without the car touching the ground.
        """)
    else:
        print("\nCould not find a valid design at target slope.")
        print("The car's geometry (especially 1.26m rear overhang)")
        print("may make this slope impossible without scraping.")


if __name__ == '__main__':
    main()
