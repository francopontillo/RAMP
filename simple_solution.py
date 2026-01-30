#!/usr/bin/env python3
"""
Simplified and Correct Garage Slope Analysis

The key physics:
- A car with wheelbase L and ground clearance h can traverse a curve
- The limiting factor is the minimum radius of curvature R_min
- R_min ≈ L^2 / (8*h) for small angles

For sharp transitions (like entering/exiting the ramp), the breakover angle matters.
For smooth curves, the radius of curvature is the key constraint.
"""

import numpy as np
import matplotlib.pyplot as plt

# Car specifications
WHEELBASE = 2.350  # m
GROUND_CLEARANCE = 0.106  # m
CAR_LENGTH = 4.461  # m

# Slope requirements
VERTICAL_DROP = 2.8  # m
DESIRED_LENGTH = 12.0  # m


def calculate_minimum_radius():
    """
    Calculate the minimum radius of curvature the car can handle.

    For a car with wheelbase L and ground clearance h traversing a concave
    curve (curving downward), the minimum radius is approximately:
    R_min = L^2 / (8 * h)

    This comes from the geometry of a chord (the car) inscribed in a circle.
    """
    R_min = WHEELBASE**2 / (8 * GROUND_CLEARANCE)
    return R_min


def cubic_spline(x, L, H):
    """Cubic spline with zero slope at ends: y = ax^3 + bx^2"""
    a = 2 * H / L**3
    b = -3 * H / L**2
    return a * x**3 + b * x**2


def calculate_curvature(x, y):
    """Calculate the curvature κ = |y''| / (1 + y'^2)^(3/2)"""
    dx = x[1] - x[0]
    dy_dx = np.gradient(y, dx)
    d2y_dx2 = np.gradient(dy_dx, dx)

    curvature = np.abs(d2y_dx2) / (1 + dy_dx**2)**1.5
    return curvature


def analyze_ramp(length, drop=VERTICAL_DROP):
    """Analyze a ramp design."""
    # Generate profile
    x = np.linspace(0, length, 2000)
    y = cubic_spline(x, length, drop)

    # Calculate curvature
    curvature = calculate_curvature(x, y)
    max_curvature = np.max(curvature)
    min_radius = 1 / max_curvature if max_curvature > 0 else float('inf')

    # Calculate slope
    dy_dx = np.gradient(y, x)
    angles = np.degrees(np.arctan(-dy_dx))
    max_angle = np.max(angles)

    # Check if safe
    R_min_required = calculate_minimum_radius()
    is_safe = min_radius >= R_min_required

    return {
        'length': length,
        'x': x,
        'y': y,
        'curvature': curvature,
        'max_curvature': max_curvature,
        'min_radius': min_radius,
        'angles': angles,
        'max_angle': max_angle,
        'is_safe': is_safe,
        'R_min_required': R_min_required
    }


def find_minimum_length():
    """Find minimum length where radius of curvature is acceptable."""
    R_min_required = calculate_minimum_radius()

    print("=" * 70)
    print("FINDING MINIMUM SAFE RAMP LENGTH")
    print("=" * 70)
    print(f"\nCar requires minimum radius of curvature: {R_min_required:.2f}m")
    print(f"(Based on wheelbase {WHEELBASE}m and ground clearance {GROUND_CLEARANCE*1000}mm)\n")

    # Search for minimum length
    for length in np.arange(10, 25, 0.5):
        result = analyze_ramp(length)
        status = "✓ SAFE" if result['is_safe'] else "✗ UNSAFE"
        print(f"{length:4.1f}m: Min radius = {result['min_radius']:6.2f}m  {status}")

        if result['is_safe']:
            return length

    print("\n⚠ No safe solution found in reasonable range!")
    return None


def create_visualization():
    """Create comprehensive visualization."""

    print("\n" + "=" * 70)
    print("PORSCHE 911 GARAGE SLOPE DESIGN")
    print("=" * 70)

    print(f"\nVehicle: Porsche 911 (997.1, 2008)")
    print(f"  Wheelbase: {WHEELBASE}m")
    print(f"  Ground Clearance: {GROUND_CLEARANCE*1000}mm")
    print(f"  Length: {CAR_LENGTH}m")

    print(f"\nRequirements:")
    print(f"  Vertical drop: {VERTICAL_DROP}m")
    print(f"  Desired max length: {DESIRED_LENGTH}m")

    R_min = calculate_minimum_radius()
    print(f"\nMinimum radius of curvature required: {R_min:.2f}m")

    # Find minimum safe length
    min_safe_length = find_minimum_length()

    if not min_safe_length:
        print("\nCannot find safe solution!")
        return

    # Analyze various lengths
    lengths = [10, 12, 14, min_safe_length, 18, 20]
    if min_safe_length not in lengths:
        lengths.append(min_safe_length)
    lengths = sorted(set([l for l in lengths if l <= 25]))

    results = {l: analyze_ramp(l) for l in lengths}

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"{'Length':<10} {'Min Radius':<12} {'Max Angle':<12} {'Status':<10}")
    print(f"{'(m)':<10} {'(m)':<12} {'(°)':<12} {'':<10}")
    print("-" * 70)

    for length in lengths:
        r = results[length]
        status = "✓ SAFE" if r['is_safe'] else "✗ UNSAFE"
        print(f"{r['length']:<10.1f} {r['min_radius']:<12.2f} {r['max_angle']:<12.2f} {status:<10}")

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Profile comparison
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(lengths)))

    for length, color in zip(lengths, colors):
        r = results[length]
        style = '-' if r['is_safe'] else '--'
        width = 3 if length == min_safe_length else 2
        label = f"{length:.0f}m (R={r['min_radius']:.1f}m)"
        if length == min_safe_length:
            label += " ★ MIN SAFE"
        label += " ✓" if r['is_safe'] else " ✗"

        ax1.plot(r['x'], r['y'], color=color, linestyle=style,
                linewidth=width, label=label, alpha=0.8)

    ax1.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
    ax1.axhline(y=-VERTICAL_DROP, color='k', linestyle='-', linewidth=1, alpha=0.3)
    ax1.set_xlabel('Horizontal Distance (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Vertical Position (m)', fontsize=11, fontweight='bold')
    ax1.set_title('Ramp Profile Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Recommended solution
    rec = results[min_safe_length]

    # Plot 2: Recommended profile
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rec['x'], rec['y'], 'b-', linewidth=3)

    # Draw car at steepest point
    steepest_idx = np.argmax(np.abs(rec['angles']))
    car_center_x = rec['x'][steepest_idx]

    rear_x = car_center_x - WHEELBASE/2
    front_x = car_center_x + WHEELBASE/2

    rear_idx = np.argmin(np.abs(rec['x'] - rear_x))
    front_idx = np.argmin(np.abs(rec['x'] - front_x))

    rear_y = rec['y'][rear_idx]
    front_y = rec['y'][front_idx]

    ax2.plot([rear_x, front_x], [rear_y, front_y],
            'r-', linewidth=5, alpha=0.7, label=f'Car at steepest point')
    ax2.plot([rear_x, front_x], [rear_y, front_y], 'ro', markersize=10)

    ax2.set_xlabel('Distance (m)', fontsize=10)
    ax2.set_ylabel('Height (m)', fontsize=10)
    ax2.set_title(f'Recommended Solution: {min_safe_length:.1f}m Ramp',
                 fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Plot 3: Slope angles
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(rec['x'], rec['angles'], 'g-', linewidth=2.5)
    ax3.fill_between(rec['x'], 0, rec['angles'], alpha=0.3, color='green')
    ax3.axhline(y=rec['max_angle'], color='orange', linestyle='--',
               linewidth=2, label=f'Max: {rec["max_angle"]:.1f}°')
    ax3.set_xlabel('Distance (m)', fontsize=10)
    ax3.set_ylabel('Slope Angle (°)', fontsize=10)
    ax3.set_title(f'Slope Angle Profile', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Radius of curvature
    ax4 = fig.add_subplot(gs[1, 2])
    radii = 1 / (rec['curvature'] + 1e-10)  # Avoid division by zero
    radii = np.minimum(radii, 100)  # Cap for visualization

    ax4.plot(rec['x'], radii, 'purple', linewidth=2.5)
    ax4.axhline(y=R_min, color='r', linestyle='--', linewidth=2,
               label=f'Min required: {R_min:.1f}m')
    ax4.fill_between(rec['x'], R_min, radii, where=(radii >= R_min),
                    alpha=0.3, color='green', label='Safe zone')
    ax4.set_xlabel('Distance (m)', fontsize=10)
    ax4.set_ylabel('Radius of Curvature (m)', fontsize=10)
    ax4.set_title('Radius of Curvature Along Ramp', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)

    # Plot 5: Min radius vs length (KEY PLOT)
    ax5 = fig.add_subplot(gs[2, :])

    test_lengths = np.linspace(8, 25, 100)
    test_radii = []

    for length in test_lengths:
        r = analyze_ramp(length)
        test_radii.append(r['min_radius'])

    ax5.plot(test_lengths, test_radii, 'b-', linewidth=3, label='Minimum radius of curvature')
    ax5.axhline(y=R_min, color='r', linestyle='--', linewidth=2.5,
               label=f'Required minimum: {R_min:.1f}m')
    ax5.axvline(x=min_safe_length, color='orange', linestyle='--', linewidth=2.5,
               label=f'Min safe length: {min_safe_length:.1f}m')
    ax5.axvline(x=DESIRED_LENGTH, color='gray', linestyle=':', linewidth=2,
               alpha=0.7, label=f'Desired length: {DESIRED_LENGTH:.0f}m')

    # Shade safe region
    safe_mask = np.array(test_radii) >= R_min
    ax5.fill_between(test_lengths, 0, 50,
                     where=safe_mask, alpha=0.2, color='green', label='Safe region')
    ax5.fill_between(test_lengths, 0, 50,
                     where=~safe_mask, alpha=0.2, color='red', label='Unsafe region')

    ax5.set_xlabel('Ramp Length (m)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Minimum Radius of Curvature (m)', fontsize=12, fontweight='bold')
    ax5.set_title('KEY FINDING: Radius of Curvature vs Ramp Length', fontsize=13, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 50)

    plt.suptitle(f'Garage Slope Design: {min_safe_length:.1f}m Cubic Spline Required',
                 fontsize=14, fontweight='bold')

    plt.savefig('/workspaces/read-it-refactor/other-projects/franco-garage-slope/final_solution.png',
                dpi=150, bbox_inches='tight')
    print("\nVisualization saved: final_solution.png")

    # Print construction guide
    print(f"\n{'=' * 70}")
    print("CONSTRUCTION REFERENCE")
    print(f"{'=' * 70}")
    print(f"\nFor {min_safe_length:.1f}m cubic spline ramp:")
    print(f"\n{'Distance (m)':<15} {'Depth (m)':<15} {'Depth (cm)':<15} {'Slope (°)':<15}")
    print("-" * 60)

    for x_val in np.arange(0, min_safe_length + 0.5, 0.5):
        if x_val <= min_safe_length:
            idx = np.argmin(np.abs(rec['x'] - x_val))
            y_val = rec['y'][idx]
            angle_val = rec['angles'][idx]
            print(f"{x_val:<15.1f} {y_val:<15.3f} {y_val*100:<15.1f} {angle_val:<15.2f}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"""
MINIMUM SAFE RAMP LENGTH: {min_safe_length:.1f}m

Your desired 12m length is {'SUFFICIENT ✓' if DESIRED_LENGTH >= min_safe_length else 'INSUFFICIENT ✗'}

{'Additional length needed: {:.1f}m'.format(min_safe_length - DESIRED_LENGTH) if min_safe_length > DESIRED_LENGTH else 'You have {:.1f}m extra margin'.format(DESIRED_LENGTH - min_safe_length)}

SPECIFICATIONS FOR {min_safe_length:.1f}m RAMP:
• Curve type: Cubic spline (smooth entry and exit)
• Maximum slope angle: {rec['max_angle']:.1f}°
• Minimum radius of curvature: {rec['min_radius']:.1f}m
• Required minimum radius: {R_min:.1f}m
• Safety factor: {rec['min_radius']/R_min:.2f}x

CONSTRUCTION NOTES:
1. Use the measurement table above
2. Mark points every 0.5m horizontally
3. Measure depth at each point from street level
4. Connect points with smooth curve (no kinks!)
5. Entry and exit should be nearly flat (0° slope)
6. Maximum slope occurs at the midpoint

The cubic spline equation is: y = ax³ + bx²
where:
  a = {2 * VERTICAL_DROP / min_safe_length**3:.8f}
  b = {-3 * VERTICAL_DROP / min_safe_length**2:.8f}

This design ensures your Porsche 911 can safely navigate the slope
without scraping its undercarriage.
    """)

    plt.show()


if __name__ == '__main__':
    create_visualization()
