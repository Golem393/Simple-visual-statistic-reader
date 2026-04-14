import cv2
import numpy as np
import os
import csv
from collections import Counter
import matplotlib.pyplot as plt # Make sure you have matplotlib installed (pip install matplotlib)

def extract_with_lasers_v6(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    debug_dir = "debug_output"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Saving debug images to ./{debug_dir}/")

    # Resize
    target_width = 800
    target_height = int(target_width * img.shape[0] / img.shape[1])
    img = cv2.resize(img, (target_width, target_height))

    # --- 1. AUTOMATIC SCREEN DETECTION (Color/Brightness Based) ---
    # Convert to HSV to target the bright bluish-white backlight
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for the light blue/white LCD backlight
    # Hue: ~90-130 (Blueish), Sat: 0-100 (Pale/Whiteish), Val: 150-255 (Bright)
    lower_blue_light = np.array([90, 0, 150])
    upper_blue_light = np.array([130, 100, 255])
    
    mask = cv2.inRange(hsv, lower_blue_light, upper_blue_light)
    cv2.imwrite(f"{debug_dir}/0a_HSV_Mask.png", mask) # Debug the color mask
    
    # Clean up the mask (remove small noise, fill holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find the largest contour in the illuminated area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    screen_contour = None
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Verify it's actually a large screen area (e.g., > 5% of total image)
        if cv2.contourArea(contours[0]) > (img.shape[0] * img.shape[1] * 0.05):
            screen_contour = contours[0]

    if screen_contour is not None:
        x, y, w, h = cv2.boundingRect(screen_contour)
        pad = 8 # Slightly larger inset to avoid the dark inner bezel shadows
        roi_y1, roi_y2 = max(0, y + pad), min(img.shape[0], y + h - pad)
        roi_x1, roi_x2 = max(0, x + pad), min(img.shape[1], x + w - pad)
    else:
        print("WARNING: Screen color detection failed. Using hardcoded fallback.")
        roi_y1, roi_y2 = 100, 325
        roi_x1, roi_x2 = 225, 575

    roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
    print(f"Detected Screen ROI: Y({roi_y1}:{roi_y2}), X({roi_x1}:{roi_x2})")
    
    cv2.imwrite(f"{debug_dir}/0b_Detected_ROI.png", roi)

    # --- 1. ROI & THRESHOLDING ---
    #roi_y1, roi_y2 = 100, 325
    # roi_x1, roi_x2 = 225, 575
    #roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 2)
    cv2.imwrite(f"{debug_dir}/0_Threshold.png", thresh)

    # --- 2. Y-AXIS DETECTION (STRUCTURAL MORPHOLOGY) ---
    vert_axis_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    vert_axis_only = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_axis_kernel)
    cv2.imwrite(f"{debug_dir}/1A_Isolated_Vertical_Axis.png", vert_axis_only)

    col_sums_axis = np.sum(vert_axis_only / 255, axis=0)
    height = thresh.shape[0]
    width = thresh.shape[1]
    
    min_axis_coverage = height * 0.30
    max_axis_coverage = height * 0.70
    axis_x_candidates = np.where((col_sums_axis >= min_axis_coverage) & (col_sums_axis <= max_axis_coverage))[0]

    lines_x = []
    if len(axis_x_candidates) > 0:
        current_group = [axis_x_candidates[0]]
        for x in axis_x_candidates[1:]:
            if x - current_group[-1] <= 3:
                current_group.append(x)
            else:
                lines_x.append(int(np.mean(current_group)))
                current_group = [x]
        lines_x.append(int(np.mean(current_group)))

    lines_x.sort()

    if not lines_x:
        print("ERROR: Failed to find any main vertical structural lines matching the 30%-70% span criteria.")
        return

    # --- 3. HORIZONTAL LINES & AXIS PAIR VERIFICATION ---
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    horiz_lines_only = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
    cv2.imwrite(f"{debug_dir}/2_Isolated_Horizontal_Lines.png", horiz_lines_only)

    row_sums = np.sum(horiz_lines_only / 255, axis=1)
    
    min_line_coverage = width * 0.35 
    line_y_candidates = np.where(row_sums >= min_line_coverage)[0]

    lines_y = []
    if len(line_y_candidates) > 0:
        current_group = [line_y_candidates[0]]
        for y in line_y_candidates[1:]:
            if y - current_group[-1] <= 3: 
                current_group.append(y)
            else:
                lines_y.append(int(np.mean(current_group)))
                current_group = [y]
        lines_y.append(int(np.mean(current_group)))

    lines_y.sort(reverse=True)
    
    valid_pairs = []
    for x_cand in lines_x:
        for i, y_cand in enumerate(lines_y):
            has_close_line = False
            for y_above in lines_y[i+1:]:
                dist = y_cand - y_above
                if 10 < dist < 50: 
                    has_close_line = True
                    break
            
            if not has_close_line:
                continue

            window_vert = vert_axis_only[max(0, y_cand-5):min(height, y_cand+6), max(0, x_cand-5):min(width, x_cand+6)]
            window_horiz = horiz_lines_only[max(0, y_cand-5):min(height, y_cand+6), max(0, x_cand-5):min(width, x_cand+6)]
            
            intersects = (np.sum(window_vert) > 0) and (np.sum(window_horiz) > 0)
            
            if intersects:
                valid_pairs.append((x_cand, y_cand))

    if not valid_pairs:
        print("ERROR: Failed to find an intersecting X and Y axis pair meeting all constraints.")
        return

    y_axis_x, baseline_y = valid_pairs[0]
    
    axes_debug = roi.copy()
    cv2.line(axes_debug, (y_axis_x, 0), (y_axis_x, height), (0, 0, 255), 1)
    cv2.line(axes_debug, (0, baseline_y), (width, baseline_y), (0, 255, 0), 1)
    cv2.circle(axes_debug, (y_axis_x, baseline_y), 5, (255, 0, 0), -1)
    cv2.imwrite(f"{debug_dir}/3_Origin_and_Baseline.png", axes_debug)

    # --- 4. MEASURE AXES AND DIVIDE (NEW LOGIC) ---
    # 1. Measure the length of the y-axis
    y_pixels = np.where(vert_axis_only[:, y_axis_x] > 0)[0]
    y_pixels_above = y_pixels[y_pixels <= baseline_y]
    top_y = np.min(y_pixels_above) if len(y_pixels_above) > 0 else 0
    y_length = baseline_y - top_y

    # 2. & 3. Find horizontal lines in this region, measure, and take the longest
    horiz_lengths = []
    for y_cand in lines_y:
        if top_y - 10 <= y_cand <= baseline_y + 10:
            # Look at pixels to the right of the y-axis
            row_pixels = np.where(horiz_lines_only[y_cand, y_axis_x:] > 0)[0]
            if len(row_pixels) > 0:
                length = np.max(row_pixels)
                horiz_lengths.append(length)

    if not horiz_lengths:
        print("ERROR: No horizontal lines found to measure x-axis length.")
        return

    horiz_lengths.sort()
    x_length = horiz_lengths[-1]
    
    # Remove outliers: if the longest line is artificially huge (>15% longer than the second longest), drop it.
    if len(horiz_lengths) > 1 and horiz_lengths[-1] > horiz_lengths[-2] * 1.15:
        x_length = horiz_lengths[-2]

    # 4. We now have x_length and y_length
    print(f"Calculated Lengths -> X length: {x_length}, Y length: {y_length}")

    # 5. Draw a picture of y-axis, x-axis, and sections
    section_img = roi.copy()
    
    # Calculate perfect blind divisions (13 x-lines for 12 sections, 6 y-lines for 5 sections)
    x_grid_lines = [y_axis_x + int(i * (x_length / 12)) for i in range(13)]
    y_grid_lines = [baseline_y - int(i * (y_length / 5)) for i in range(6)]

   # --- 5. CALCULATE & FIRE LASERS (MULTI-RAY SCATTER WITH SANITY CHECK) ---
    if len(y_grid_lines) < 2:
        print("ERROR: Y-axis is too short to calculate the 'second x axis from below'.")
        return

    start_y = y_grid_lines[1] 
    
    ignore_y_coords = set()
    for y in lines_y:
        for dy in range(-2, 3): 
            ignore_y_coords.add(y + dy)

    laser_img = roi.copy()
    cv2.line(laser_img, (0, start_y), (width, start_y), (255, 255, 0), 1)

    section_width = x_length / 12
    num_rays = 40 

    csv_data = []
    plot_labels = []
    plot_today = []
    plot_yesterday = []

    for i in range(12):
        section_start_x = y_axis_x + (i * section_width)
        
        left_xs = np.linspace(section_start_x + (0.10 * section_width), 
                              section_start_x + (0.50 * section_width), num_rays, dtype=int)
        
        right_xs = np.linspace(section_start_x + (0.45 * section_width), 
                               section_start_x + (0.75 * section_width), num_rays, dtype=int)

        def get_best_collision(hits):
            valid_hits = [h for h in hits if top_y - 10 <= h <= start_y - 2]
            if not valid_hits:
                return start_y 
            counts = Counter(valid_hits)
            return counts.most_common(1)[0][0]

        # ---------------------------------------------------------
        # 1. FIRE LEFT RAYS ("Today" - Gap Tolerant + Sanity Check)
        # ---------------------------------------------------------
        left_hits = []
        for x in left_xs:
            if 0 <= x < width:
                col_y = start_y
                gap_count = 0
                max_gap = 15 
                for y in range(start_y - 2, top_y - 10, -1):
                    if y in ignore_y_coords: continue
                    if thresh[y, x] == 255: 
                        col_y = y 
                        gap_count = 0      
                    else:
                        gap_count += 1
                        if gap_count > max_gap: break 
                
                # SANITY CHECK: Look at a 10x5 patch inside the bar
                patch_y1 = min(col_y + 2, start_y - 1)
                patch_y2 = min(col_y + 12, start_y)
                patch_x1 = max(0, x - 2)
                patch_x2 = min(width, x + 3)
                patch = thresh[patch_y1:patch_y2, patch_x1:patch_x2]
                
                if patch.size > 0:
                    white_ratio = np.sum(patch == 255) / patch.size
                    # Left bar should be "filled", expecting > 5% white pixels/noise
                    if white_ratio >= 0.05: 
                        left_hits.append(col_y)
                        cv2.line(laser_img, (x, start_y), (x, col_y), (0, 165, 255), 1) # Orange = Valid
                    else:
                        cv2.line(laser_img, (x, start_y), (x, col_y), (100, 100, 100), 1) # Gray = Rejected Crossover
                else:
                    left_hits.append(col_y)
        print(f"Section {i} (Left Ray at X={x}): Stopped at Y={col_y}. Gap count reached: {gap_count}. White ratio in patch: {white_ratio:.2f}")
        best_left_y = get_best_collision(left_hits)
        mid_left_x = int(np.mean(left_xs))
        cv2.circle(laser_img, (mid_left_x, best_left_y), 5, (0, 0, 255), -1)

        # ---------------------------------------------------------
        # 2. FIRE RIGHT RAYS ("Yesterday" - First Hit + Sanity Check)
        # ---------------------------------------------------------
        right_hits = []
        for x in right_xs:
            if 0 <= x < width:
                col_y = start_y
                # Start slightly higher (-5) to bypass immediate baseline sludge
                for y in range(start_y - 5, top_y - 10, -1):
                    if y in ignore_y_coords: continue
                    
                    if thresh[y, x] == 255: 
                        # Run sanity check BEFORE breaking
                        patch_y1 = min(y + 4, start_y - 1)  # TUNED: Moved down from y+2 to avoid thick top borders
                        patch_y2 = min(y + 12, start_y)
                        patch_x1 = max(0, x - 1)            # TUNED: Narrowed from x-2 to avoid side walls
                        patch_x2 = min(width, x + 2)        # TUNED: Narrowed from x+3
                        patch = thresh[patch_y1:patch_y2, patch_x1:patch_x2]
                        
                        if patch.size > 0:
                            white_ratio = np.sum(patch == 255) / patch.size
                            # Right bar should be "empty", expecting low white pixels
                            if white_ratio < 0.40:          # TUNED: Relaxed from 0.20 to 0.40
                                col_y = y 
                                right_hits.append(col_y)
                                cv2.line(laser_img, (x, start_y), (x, col_y), (0, 255, 0), 1)
                                break
                        
                # If we get here and col_y is still start_y, it failed to find a valid top
                if col_y == start_y:
                    cv2.line(laser_img, (x, start_y), (x, start_y - 10), (100, 100, 100), 1) # Gray = Rejected
                    
        print(f"Section {i} (Right Ray): Best Hit Y={col_y}.")
        best_right_y = get_best_collision(right_hits)
        mid_right_x = int(np.mean(right_xs))
        cv2.circle(laser_img, (mid_right_x, best_right_y), 5, (0, 0, 255), -1)

        # ---------------------------------------------------------
        # 3. RECORD DATA
        # ---------------------------------------------------------
        section_name = i * 2
        today_val = start_y - best_left_y
        yesterday_val = start_y - best_right_y
        
        csv_data.append({
            "Section": section_name,
            "Today": today_val,
            "Yesterday": yesterday_val
        })
        
        plot_labels.append(str(section_name))
        plot_today.append(today_val)
        plot_yesterday.append(yesterday_val)

    cv2.imwrite(f"{debug_dir}/5_Final_Lasers_MultiRay.png", laser_img)
    
    # --- 6. EXPORT STATS & CSV ---
    # Save CSV
    csv_path = f"{debug_dir}/extracted_data.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Section", "Today", "Yesterday"])
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"Data saved to {csv_path}")

    # Generate Statistic Plot
    x_pos = np.arange(len(plot_labels))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x_pos - bar_width/2, plot_today, bar_width, label='Today (Left - Mode)', color='orange')
    plt.bar(x_pos + bar_width/2, plot_yesterday, bar_width, label='Yesterday (Right - Mode)', color='green')

    plt.xlabel('Sections (0, 2, 4 ... 22)')
    plt.ylabel('Extracted Pixel Height (Start Y - Best Hit Y)')
    plt.title('Extracted Bar Heights using Multi-Ray Mode Aggregation')
    plt.xticks(x_pos, plot_labels)
    plt.legend()
    
    stats_img_path = f"{debug_dir}/6_Extraction_Statistics.png"
    plt.savefig(stats_img_path)
    plt.close()
    
    print(f"Statistics visualized and saved to {stats_img_path}")
    print("Extraction complete. Check the debug_output folder.")
# Run it
extract_with_lasers_v6('graph.png')