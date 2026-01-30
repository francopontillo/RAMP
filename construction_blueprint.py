#!/usr/bin/env python3
"""
Construction Blueprint Generator for 12m Garage Ramp
Creates detailed construction drawings and specifications
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Ramp specifications
RAMP_LENGTH = 12.0  # meters
VERTICAL_DROP = 2.8  # meters
RAMP_WIDTH = 3.0  # meters (typical single car width)

# Car specifications (for reference)
WHEELBASE = 2.350  # m
GROUND_CLEARANCE = 0.106  # m


def cubic_spline(x, L, H):
    """Cubic spline: y = ax³ + bx²"""
    a = 2 * H / L**3
    b = -3 * H / L**2
    return a * x**3 + b * x**2


def generate_construction_points(num_points=25):
    """Generate detailed construction points."""
    x = np.linspace(0, RAMP_LENGTH, num_points)
    y = cubic_spline(x, RAMP_LENGTH, VERTICAL_DROP)

    # Calculate slope at each point
    dy_dx = np.gradient(y, x)
    angles = np.degrees(np.arctan(-dy_dx))

    return x, y, angles


def create_blueprint():
    """Create comprehensive construction blueprint."""

    # Generate profile
    x_smooth = np.linspace(0, RAMP_LENGTH, 1000)
    y_smooth = cubic_spline(x_smooth, RAMP_LENGTH, VERTICAL_DROP)

    # Construction points (every 0.5m)
    x_points, y_points, angles = generate_construction_points(25)

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('white')

    # Title block
    fig.text(0.5, 0.97, 'GARAGE RAMP CONSTRUCTION BLUEPRINT',
             ha='center', fontsize=20, fontweight='bold', family='monospace')
    fig.text(0.5, 0.95, 'PORSCHE 911 (997.1) - 12m CUBIC SPLINE RAMP',
             ha='center', fontsize=14, fontweight='bold', family='monospace')

    # Project info
    fig.text(0.05, 0.93, 'Project: Residential Garage Access Ramp', fontsize=10, family='monospace')
    fig.text(0.05, 0.915, f'Ramp Length: {RAMP_LENGTH}m | Vertical Drop: {VERTICAL_DROP}m | Width: {RAMP_WIDTH}m',
             fontsize=10, family='monospace')
    fig.text(0.05, 0.900, 'Design: Cubic Spline Profile with Zero Entry/Exit Slopes',
             fontsize=10, family='monospace')

    # Grid layout - give much more space to top row for side elevation graph
    # Large hspace for lots of breathing room between graph and table
    gs = fig.add_gridspec(4, 2, left=0.08, right=0.95, top=0.88, bottom=0.05,
                          hspace=1.4, wspace=0.3, height_ratios=[1.6, 1, 1, 0.8])

    # ============================================================
    # MAIN PROFILE VIEW (Side Elevation)
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Draw ground/ramp profile
    ax1.plot(x_smooth, y_smooth, 'b-', linewidth=3, label='Ramp profile')
    ax1.plot([0, 0], [0, 0.5], 'k-', linewidth=2)  # Street level marker
    ax1.plot([RAMP_LENGTH, RAMP_LENGTH], [-VERTICAL_DROP, -VERTICAL_DROP+0.5],
             'k-', linewidth=2)  # Garage level marker

    # Draw measurement points (every 0.5m)
    for i, (x_p, y_p) in enumerate(zip(x_points, y_points)):
        ax1.plot([x_p, x_p], [y_p, 0], 'r--', linewidth=0.8, alpha=0.4)
        ax1.plot(x_p, y_p, 'ro', markersize=7, markeredgecolor='darkred', markeredgewidth=1.2)

        # Distance labels at top (alternating height to avoid overlap)
        if i % 2 == 0:
            ax1.text(x_p, 0.2, f'{x_p:.1f}m', ha='center', fontsize=8.5, fontweight='bold')
        else:
            ax1.text(x_p, 0.4, f'{x_p:.1f}m', ha='center', fontsize=8.5, fontweight='bold', style='italic')

        # Depth labels - all at same distance below their red dot
        # Every label is horizontally centered on its red dot at consistent vertical offset
        label_offset = 0.15  # consistent distance below each red dot
        label_y = y_p - label_offset

        ax1.text(x_p, label_y, f'{abs(y_p):.2f}m', ha='center', va='top', fontsize=6.5,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='yellow', alpha=0.85, edgecolor='orange', linewidth=0.5))

    # Draw construction grid lines
    ax1.plot([0, RAMP_LENGTH], [0, 0], 'k-', linewidth=2, label='Street level (0.00m)')
    ax1.plot([0, RAMP_LENGTH], [-VERTICAL_DROP, -VERTICAL_DROP], 'k-',
             linewidth=2, label=f'Garage level (-{VERTICAL_DROP}m)')

    # Dimension arrows
    ax1.annotate('', xy=(RAMP_LENGTH, -VERTICAL_DROP-0.5), xytext=(0, -VERTICAL_DROP-0.5),
                arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    ax1.text(RAMP_LENGTH/2, -VERTICAL_DROP-0.7, f'{RAMP_LENGTH}m HORIZONTAL',
            ha='center', fontsize=12, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))

    ax1.annotate('', xy=(RAMP_LENGTH+0.5, -VERTICAL_DROP), xytext=(RAMP_LENGTH+0.5, 0),
                arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    ax1.text(RAMP_LENGTH+1.0, -VERTICAL_DROP/2, f'{VERTICAL_DROP}m\nVERTICAL',
            ha='left', va='center', fontsize=12, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))

    ax1.set_xlabel('HORIZONTAL DISTANCE (meters)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('VERTICAL POSITION (meters)', fontsize=13, fontweight='bold')
    ax1.set_title('SIDE ELEVATION VIEW - RAMP PROFILE (All measurements at 0.5m intervals)',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    # Don't use equal aspect - let horizontal stretch for better label spacing
    ax1.set_aspect('auto')
    ax1.legend(loc='lower left', fontsize=10)
    ax1.set_xlim(-0.5, RAMP_LENGTH+0.5)
    ax1.set_ylim(-VERTICAL_DROP-0.6, 0.7)

    # ============================================================
    # TOP VIEW (Plan View)
    # ============================================================
    ax2 = fig.add_subplot(gs[1, 0])

    # Draw ramp outline
    ramp_rect = Rectangle((0, -RAMP_WIDTH/2), RAMP_LENGTH, RAMP_WIDTH,
                          linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.3)
    ax2.add_patch(ramp_rect)

    # Draw centerline
    ax2.plot([0, RAMP_LENGTH], [0, 0], 'r--', linewidth=2, label='Centerline')

    # Draw measurement points (every 0.5m)
    for i, x_p in enumerate(x_points):
        ax2.plot([x_p, x_p], [-RAMP_WIDTH/2, RAMP_WIDTH/2], 'k-', linewidth=0.8, alpha=0.4)
        # Alternate labels to avoid clutter
        if i % 2 == 0:
            ax2.text(x_p, RAMP_WIDTH/2 + 0.2, f'{x_p:.1f}', ha='center', fontsize=7)
        else:
            ax2.text(x_p, RAMP_WIDTH/2 + 0.35, f'{x_p:.1f}', ha='center', fontsize=7, style='italic')

    # Draw car outline for scale
    car_length = 4.461
    car_width = 1.852
    car_rect = Rectangle((RAMP_LENGTH/2 - car_length/2, -car_width/2),
                         car_length, car_width,
                         linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax2.add_patch(car_rect)
    ax2.text(RAMP_LENGTH/2, 0, 'PORSCHE 911\n(for scale)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='red')

    # Dimensions
    ax2.annotate('', xy=(RAMP_LENGTH, -RAMP_WIDTH/2-0.3), xytext=(0, -RAMP_WIDTH/2-0.3),
                arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
    ax2.text(RAMP_LENGTH/2, -RAMP_WIDTH/2-0.5, f'{RAMP_LENGTH}m',
            ha='center', fontsize=11, fontweight='bold', color='blue')

    ax2.annotate('', xy=(RAMP_LENGTH+0.5, RAMP_WIDTH/2), xytext=(RAMP_LENGTH+0.5, -RAMP_WIDTH/2),
                arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
    ax2.text(RAMP_LENGTH+0.8, 0, f'{RAMP_WIDTH}m',
            ha='left', va='center', fontsize=11, fontweight='bold', color='blue', rotation=90)

    ax2.set_xlabel('LENGTH (m)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('WIDTH (m)', fontsize=11, fontweight='bold')
    ax2.set_title('TOP VIEW - PLAN', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-1, RAMP_LENGTH+1.5)
    ax2.set_ylim(-RAMP_WIDTH/2-1, RAMP_WIDTH/2+1)

    # ============================================================
    # SLOPE ANGLE DIAGRAM
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 1])

    ax3.plot(x_smooth, np.degrees(np.arctan(-np.gradient(y_smooth, x_smooth))),
            'g-', linewidth=3)
    ax3.fill_between(x_smooth, 0, np.degrees(np.arctan(-np.gradient(y_smooth, x_smooth))),
                     alpha=0.3, color='green')

    max_angle = np.max(np.degrees(np.arctan(-np.gradient(y_smooth, x_smooth))))
    max_angle_x = x_smooth[np.argmax(np.degrees(np.arctan(-np.gradient(y_smooth, x_smooth))))]

    ax3.axhline(y=max_angle, color='red', linestyle='--', linewidth=2,
               label=f'Maximum: {max_angle:.1f}°')
    ax3.plot(max_angle_x, max_angle, 'ro', markersize=12, markeredgecolor='darkred',
            markeredgewidth=2)

    ax3.set_xlabel('DISTANCE (m)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('SLOPE ANGLE (degrees)', fontsize=11, fontweight='bold')
    ax3.set_title('SLOPE ANGLE PROFILE', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlim(0, RAMP_LENGTH)
    ax3.set_ylim(0, 25)

    # ============================================================
    # CONSTRUCTION POINTS TABLE
    # ============================================================
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Table data
    table_data = []
    table_data.append(['Distance\n(m)', 'Depth from\nStreet (m)', 'Depth from\nStreet (cm)',
                      'Slope\nAngle (°)', 'Formwork\nHeight (cm)', 'Notes'])

    for i, (x_p, y_p, angle) in enumerate(zip(x_points, y_points, angles)):
        formwork_height = abs(y_p) * 100
        note = ''
        if x_p == 0:
            note = 'START - Street level'
        elif x_p == RAMP_LENGTH:
            note = 'END - Garage level'
        elif abs(x_p - RAMP_LENGTH/2) < 0.1:
            note = 'MIDPOINT - Max slope'
        elif i % 4 == 0:
            note = 'Reference stake'

        table_data.append([
            f'{x_p:.1f}',
            f'{y_p:.3f}',
            f'{abs(y_p)*100:.1f}',
            f'{angle:.1f}',
            f'{formwork_height:.1f}',
            note
        ])

    # Create table
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.15, 0.15, 0.12, 0.15, 0.31])

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)

    # Style header row
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=9)

    # Style data rows (alternate colors)
    for i in range(1, len(table_data)):
        for j in range(6):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')

            # Highlight key points
            if 'START' in table_data[i][5] or 'END' in table_data[i][5]:
                cell.set_facecolor('#FFD966')
                cell.set_text_props(weight='bold')
            elif 'MIDPOINT' in table_data[i][5]:
                cell.set_facecolor('#F4B183')
                cell.set_text_props(weight='bold')

    ax4.set_title('CONSTRUCTION MEASUREMENT TABLE', fontsize=13, fontweight='bold',
                 pad=20, loc='left')

    # ============================================================
    # SPECIFICATIONS AND NOTES
    # ============================================================
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.axis('off')

    specs_text = """
TECHNICAL SPECIFICATIONS

Curve Equation: y = ax³ + bx²
  where: a = 0.003240741
         b = -0.058333333
         x = horizontal distance (m)
         y = vertical drop (m)

Maximum Slope: 19.3° (at 6m mark)
Minimum Radius of Curvature: 8.6m
Surface Grade: Smooth finish required

MATERIALS REQUIRED
• Concrete: ~42 m³ (estimate)
• Rebar: 12mm spacing grid
• Formwork: Flexible or stepped
• Surface finish: Broom finish for traction
• Drainage: 100mm channel each side

VEHICLE CLEARANCES
✓ Ground clearance: 106mm + margin
✓ Approach angle: Safe
✓ Departure angle: Safe
✓ Breakover angle: Safe
    """

    ax5.text(0.05, 0.95, specs_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue',
                     edgecolor='blue', linewidth=2))

    # ============================================================
    # CONSTRUCTION INSTRUCTIONS
    # ============================================================
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')

    instructions_text = """
CONSTRUCTION SEQUENCE

1. SITE PREPARATION
   □ Mark centerline with string/laser
   □ Set reference stakes every 0.5m
   □ Excavate to depth + 150mm for base

2. BASE & DRAINAGE
   □ Install drainage channels both sides
   □ Compact 100mm gravel base
   □ Check levels against table

3. FORMWORK
   □ Build side forms at 3m width
   □ Use flexible material for curve OR
   □ Use stepped formwork every 0.5m
   □ Set top edge to profile elevations
   □ Support forms every 1m

4. REINFORCEMENT
   □ 12mm rebar grid: 300mm spacing
   □ 50mm cover to surface
   □ Secure against movement

5. CONCRETE POUR
   □ Pour in sections if needed
   □ Vibrate thoroughly
   □ Screed to match profile
   □ Smooth but not slippery finish

6. FINISHING
   □ Broom finish for traction
   □ Cure minimum 7 days
   □ Remove forms carefully
   □ Seal edges if required

⚠ CRITICAL: Entry/exit transitions
   must be smooth with no bumps!
    """

    ax6.text(0.05, 0.95, instructions_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                     edgecolor='orange', linewidth=2))

    # Add warning box
    warning_text = "⚠ WARNING: This is a complex curved structure. Consider professional consultation for construction."
    fig.text(0.5, 0.02, warning_text, ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', edgecolor='red', linewidth=2))

    # Save
    plt.savefig('/workspaces/read-it-refactor/other-projects/franco-garage-slope/construction_blueprint.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    print("✓ Construction blueprint saved: construction_blueprint.png")

    plt.show()


def export_csv_measurements():
    """Export detailed measurements to CSV for builders."""

    # Generate high-resolution points
    x_detailed = np.linspace(0, RAMP_LENGTH, 121)  # Every 10cm
    y_detailed = cubic_spline(x_detailed, RAMP_LENGTH, VERTICAL_DROP)

    dy_dx = np.gradient(y_detailed, x_detailed)
    angles = np.degrees(np.arctan(-dy_dx))

    # Write to CSV
    filename = '/workspaces/read-it-refactor/other-projects/franco-garage-slope/measurements.csv'
    with open(filename, 'w') as f:
        f.write('# GARAGE RAMP CONSTRUCTION MEASUREMENTS\n')
        f.write(f'# Ramp Length: {RAMP_LENGTH}m, Vertical Drop: {VERTICAL_DROP}m\n')
        f.write('# Cubic Spline Profile: y = ax³ + bx² where a=0.003240741, b=-0.058333333\n')
        f.write('#\n')
        f.write('Distance_m,Distance_cm,Depth_m,Depth_cm,Depth_mm,Slope_degrees,Formwork_Height_cm\n')

        for x_val, y_val, angle in zip(x_detailed, y_detailed, angles):
            f.write(f'{x_val:.3f},{x_val*100:.1f},{y_val:.4f},{abs(y_val)*100:.2f},'
                   f'{abs(y_val)*1000:.1f},{angle:.2f},{abs(y_val)*100:.2f}\n')

    print(f"✓ Detailed measurements exported: measurements.csv")


def export_construction_checklist():
    """Export construction checklist."""

    filename = '/workspaces/read-it-refactor/other-projects/franco-garage-slope/construction_checklist.txt'
    with open(filename, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write('GARAGE RAMP CONSTRUCTION CHECKLIST\n')
        f.write('=' * 70 + '\n\n')

        f.write('PROJECT SPECIFICATIONS:\n')
        f.write(f'  Ramp Length: {RAMP_LENGTH}m horizontal\n')
        f.write(f'  Vertical Drop: {VERTICAL_DROP}m\n')
        f.write(f'  Ramp Width: {RAMP_WIDTH}m\n')
        f.write(f'  Profile: Cubic spline (smooth curve)\n')
        f.write(f'  Maximum Slope: 19.3° (at midpoint)\n\n')

        f.write('PRE-CONSTRUCTION:\n')
        f.write('  [ ] Obtain necessary permits\n')
        f.write('  [ ] Mark all utility lines (call before you dig!)\n')
        f.write('  [ ] Verify measurements on site\n')
        f.write('  [ ] Order materials (see specifications)\n')
        f.write('  [ ] Arrange concrete delivery timing\n')
        f.write('  [ ] Set up site safety barriers\n\n')

        f.write('SITE PREPARATION:\n')
        f.write('  [ ] Clear area of debris and vegetation\n')
        f.write('  [ ] Establish reference point at street level\n')
        f.write('  [ ] Mark centerline with string line or laser\n')
        f.write('  [ ] Set measurement stakes every 0.5m (25 stakes total)\n')
        f.write('  [ ] Mark depth at each stake per measurement table\n')
        f.write('  [ ] Excavate to required depth + 150mm for base\n\n')

        f.write('BASE PREPARATION:\n')
        f.write('  [ ] Install drainage channels on both sides\n')
        f.write('  [ ] Place and compact 100mm gravel base\n')
        f.write('  [ ] Verify base levels match profile\n')
        f.write('  [ ] Ensure proper slope for water drainage\n\n')

        f.write('FORMWORK:\n')
        f.write('  [ ] Build side forms to 3m width\n')
        f.write('  [ ] Set form top edges to match profile elevations\n')
        f.write('  [ ] Option A: Use flexible material for smooth curve\n')
        f.write('  [ ] Option B: Use stepped formwork every 0.5m\n')
        f.write('  [ ] Brace formwork securely every 1m\n')
        f.write('  [ ] Double-check all elevations\n')
        f.write('  [ ] Ensure smooth transitions (no bumps!)\n\n')

        f.write('REINFORCEMENT:\n')
        f.write('  [ ] Cut and bend rebar as required\n')
        f.write('  [ ] Install 12mm rebar grid at 300mm spacing\n')
        f.write('  [ ] Maintain 50mm concrete cover\n')
        f.write('  [ ] Tie all intersections securely\n')
        f.write('  [ ] Use rebar chairs to maintain height\n')
        f.write('  [ ] Final inspection before pour\n\n')

        f.write('CONCRETE POUR:\n')
        f.write('  [ ] Verify concrete mix design (min 32 MPa)\n')
        f.write('  [ ] Have sufficient crew for continuous pour\n')
        f.write('  [ ] Pour concrete in systematic sections\n')
        f.write('  [ ] Vibrate thoroughly to eliminate voids\n')
        f.write('  [ ] Screed surface to match profile curve\n')
        f.write('  [ ] Float surface smooth\n')
        f.write('  [ ] Apply broom finish for traction\n')
        f.write('  [ ] Check edges and transitions\n\n')

        f.write('CURING & FINISHING:\n')
        f.write('  [ ] Cover with plastic sheeting\n')
        f.write('  [ ] Keep moist for minimum 7 days\n')
        f.write('  [ ] Remove formwork carefully (after 2-3 days)\n')
        f.write('  [ ] Fill any small voids or gaps\n')
        f.write('  [ ] Seal edges if required\n')
        f.write('  [ ] Allow full cure (28 days) before heavy use\n\n')

        f.write('FINAL CHECKS:\n')
        f.write('  [ ] Verify profile matches design (use template)\n')
        f.write('  [ ] Check drainage function\n')
        f.write('  [ ] Test drive with actual vehicle (SLOW!)\n')
        f.write('  [ ] Check for any scraping points\n')
        f.write('  [ ] Verify entry and exit transitions are smooth\n')
        f.write('  [ ] Clean up site\n')
        f.write('  [ ] Document completion with photos\n\n')

        f.write('SAFETY NOTES:\n')
        f.write('  ⚠ Maximum slope of 19.3° - take care during construction\n')
        f.write('  ⚠ Ensure adequate drainage to prevent ice formation\n')
        f.write('  ⚠ Smooth curve is critical - no bumps or dips\n')
        f.write('  ⚠ Consider professional consultation for this complex curve\n\n')

        f.write('=' * 70 + '\n')

    print(f"✓ Construction checklist exported: construction_checklist.txt")


if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING CONSTRUCTION BLUEPRINT")
    print("=" * 70)
    print()

    create_blueprint()
    export_csv_measurements()
    export_construction_checklist()

    print()
    print("=" * 70)
    print("CONSTRUCTION DOCUMENTS GENERATED")
    print("=" * 70)
    print()
    print("Files created:")
    print("  1. construction_blueprint.png - Visual blueprint with all views")
    print("  2. measurements.csv - Detailed measurements every 10cm")
    print("  3. construction_checklist.txt - Complete construction checklist")
    print()
    print("Ready for construction!")
