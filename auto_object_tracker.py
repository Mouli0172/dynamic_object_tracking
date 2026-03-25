import cv2
import numpy as np
import math

# Video tracking script for rolling tire using OpenCV CSRT
# Setting up initial parameters

# Configuration
VIDEO_PATH       = "videoplayback_264.mp4"   
OUTPUT_PATH      = "tracked_output.mp4"      
FRAME_WIDTH      = 1280      # Processing resolution
FRAME_HEIGHT     = 720
MARGIN           = 40        # Buffer zone
SEARCH_RADIUS    = 400       # fallback search radius
HORIZON_RATIO    = 0.35      # Top 35% of the frame is the "No-Search Zone"
THRESH_VALUE     = 70        # Binary threshold for dark-object isolation
AREA_MIN_FACTOR  = 0.15      # Contour area must be >= 15% of last known area
AREA_MAX_FACTOR  = 4.0       # Contour area must be <= 400% of last known area
ASPECT_MIN       = 0.15      # Minimum aspect ratio 
ASPECT_MAX       = 6.0       # Maximum aspect ratio
ERODE_KERNEL_SZ  = (3, 3)    # kernel size
ERODE_ITERS      = 1         # Lighter erosion to preserve small tire contours
DILATE_ITERS     = 1         # Match erosion to restore size
MIN_CONTOUR_AREA = 80        # Absolute minimum area 
CSRT_EDGE_MARGIN = 10        
REINIT_PADDING   = 15        # CSRT re-init

# Video Capture
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video file:", VIDEO_PATH)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # fallback
FRAME_DELAY = max(1, int(1000 / fps))

# Horizon Y-coordinate (top 35% = off-limits)
HORIZON_Y = int(FRAME_HEIGHT * HORIZON_RATIO)

# Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))
if not out_writer.isOpened():
    print("[WARN] Could not open video writer. Output will not be saved.")

# Play video, pause, draw bounding box
while True:
    success, frame = cap.read()
    if not success:
        print("Error: Reached end of video before selection. Exiting.")
        exit()

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    cv2.imshow("Tracker", frame)

    key = cv2.waitKey(FRAME_DELAY) & 0xFF
    if key == ord('p'):
        print("[PAUSE] Video paused. Draw a bounding box around the tire.")
        print("[PAUSE] Press ENTER or SPACE to confirm. Press 'C' to cancel.")
        break
    elif key == ord('q'):
        print("[QUIT] User exited during playback.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# draw a bounding box
bbox = cv2.selectROI("Tracker", frame, fromCenter=False, showCrosshair=True)

if bbox[2] == 0 or bbox[3] == 0:
    print("[ABORT] No region selected. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

print(f"Setup Selected ROI: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")

# Create and initialize the CSRT tracker
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)
print("Setup CSRT tracker initialized successfully.")

last_known_good_box = bbox
initial_area = bbox[2] * bbox[3]
print(f"Setup Initial tire area: {initial_area}px² (anchor for area filter)")

# prev_center: center of the tire in the previous frame
prev_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

lost_frame_count = 0
frame_number = 0

print("[TRACK] Tracking started! Green = CSRT locked, Yellow = RECOVERED")
print("=" * 65)

def is_box_sane(bx, by, bw, bh):
    """
    Returns False if the bounding box is clearly garbage:
    - Near the frame edges (CSRT ghost tracking returns ~0,0)
    - Zero or negative dimensions
    NOTE: We do NOT check the horizon here! CSRT can legitimately
    track in the upper screen when the drone is looking down at
    the dune. The horizon rule only applies to the fallback scanner.
    """
    # Zero/negative dimensions
    if bw <= 0 or bh <= 0:
        return False
    # Too close to any frame edge
    if bx < CSRT_EDGE_MARGIN or by < CSRT_EDGE_MARGIN:
        return False
    if (bx + bw) > (FRAME_WIDTH - CSRT_EDGE_MARGIN):
        return False
    if (by + bh) > (FRAME_HEIGHT - CSRT_EDGE_MARGIN):
        return False
    return True

def refine_with_contours(frame, rx, ry, rw, rh, margin):
    """
    Given a rough bounding box, expand it by 'margin' pixels, threshold
    the ROI, apply shadow-severing morphology, and find the tightest
    contour bounding box for the dark object.

    Returns (success, x, y, w, h) in full-frame coordinates.
    """
    # Expand the ROI with the margin, clamp to frame bounds
    buf_x1 = max(0, rx - margin)
    buf_y1 = max(0, ry - margin)
    buf_x2 = min(FRAME_WIDTH,  rx + rw + margin)
    buf_y2 = min(FRAME_HEIGHT, ry + rh + margin)

    roi = frame[buf_y1:buf_y2, buf_x1:buf_x2]
    if roi.size == 0:
        return False, rx, ry, rw, rh

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Threshold: dark pixels → white on black
    _, mask = cv2.threshold(blurred, THRESH_VALUE, 255, cv2.THRESH_BINARY_INV)

    # Shadow Severing: erode to pinch the shadow neck, dilate to restore tire
    kernel = np.ones(ERODE_KERNEL_SZ, np.uint8)
    mask = cv2.erode(mask, kernel, iterations=ERODE_ITERS)
    mask = cv2.dilate(mask, kernel, iterations=DILATE_ITERS)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, rx, ry, rw, rh

    # Pick the largest contour that meets minimum area
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not valid:
        # If no contour meets min area, take the largest anyway
        valid = contours

    largest = max(valid, key=cv2.contourArea)
    cx, cy, cw, ch = cv2.boundingRect(largest)

    # Convert from ROI-local to full-frame coordinates
    real_x = buf_x1 + cx
    real_y = buf_y1 + cy

    return True, real_x, real_y, cw, ch

tracker_alive = True  # CSRT is live

# Main Block
while True:
    success, frame = cap.read()
    if not success:
        print("Video finished.")
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_number += 1

    # Draw the Horizon Line
    cv2.line(frame, (0, HORIZON_Y), (FRAME_WIDTH, HORIZON_Y), (0, 0, 255), 1)
    cv2.putText(frame, "HORIZON LIMIT",
                (10, HORIZON_Y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 255), 1, cv2.LINE_AA)
    found_this_frame = False

    if tracker_alive:
        is_tracking, csrt_bbox = tracker.update(frame)

        if is_tracking:
            cx, cy, cw, ch = [int(v) for v in csrt_bbox]

            if is_box_sane(cx, cy, cw, ch):
                # CSRT box is plausible — refine it with contour snapping
                refined, rx, ry, rw, rh = refine_with_contours(
                    frame, cx, cy, cw, ch, MARGIN)

                if refined and rw > 0 and rh > 0:
                    # Drawing the refined bounding box 
                    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, "TRACKING",
                                (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    # Show tire dimensions for debugging auto-adjust
                    cv2.putText(frame, f"{rw}x{rh}",
                                (rx, ry + rh + 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 255, 0), 1, cv2.LINE_AA)

                    # Update dynamic memory
                    last_known_good_box = (rx, ry, rw, rh)
                    prev_center = (rx + rw // 2, ry + rh // 2)
                    found_this_frame = True
                else:
                    # Contour refinement failed — use raw CSRT box
                    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch),
                                  (0, 200, 0), 2)
                    cv2.putText(frame, "TRACKING (raw)",
                                (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 200, 0), 2, cv2.LINE_AA)
                    last_known_good_box = (cx, cy, cw, ch)
                    prev_center = (cx + cw // 2, cy + ch // 2)
                    found_this_frame = True
            else:
                # Ghost tracking — CSRT at edge/origin. Kill the tracker.
                if lost_frame_count == 0:
                    print(f"[CSRT GHOST] Frame {frame_number}: Box at ({cx},{cy}) "
                          f"is at frame edge — killing CSRT, switching to fallback")
                tracker_alive = False

        else:
            # CSRT explicitly returned False
            if lost_frame_count == 0:
                print(f"[CSRT LOST] Frame {frame_number}: Tracker returned False")
            tracker_alive = False

    # Reset or increment the lost counter
    if found_this_frame:
        lost_frame_count = 0
    else:
        lost_frame_count += 1

    if not found_this_frame:
        if lost_frame_count == 1:
            print(f"Scanning: Frame {frame_number}: Triggering WIDE RADIUS search...")
            lx, ly, lw, lh = last_known_good_box
            print(f"Scanning: Search center: ({lx + lw//2}, {ly + lh//2}) | "
                  f"Radius: {SEARCH_RADIUS}px | Horizon: Y > {HORIZON_Y}")
        elif lost_frame_count % 30 == 0:
            print(f"Scanning: Still searching... ({lost_frame_count} frames lost)")

        cv2.putText(frame, f"SCANNING... (lost {lost_frame_count}f)",
                    (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 165, 255), 2, cv2.LINE_AA)

        # Unpack last known good position
        last_x, last_y, last_w, last_h = last_known_good_box
        last_area = max(last_w * last_h, MIN_CONTOUR_AREA)
        search_cx = last_x + last_w // 2
        search_cy = last_y + last_h // 2

        area_ref_min = min(initial_area, last_area)
        area_ref_max = max(initial_area, last_area)

        max_acceptable_area = initial_area * 8.0

        # Expand search radius when lost
        effective_radius = min(SEARCH_RADIUS + (lost_frame_count * 3), 600)

        # Draw the search circle
        cv2.circle(frame, (search_cx, search_cy), effective_radius,
                   (255, 100, 0), 2)

        # Full-frame threshold
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_full = cv2.GaussianBlur(gray_full, (7, 7), 0)
        _, mask_full = cv2.threshold(blurred_full, THRESH_VALUE, 255,
                                     cv2.THRESH_BINARY_INV)

        kernel = np.ones(ERODE_KERNEL_SZ, np.uint8)
        mask_full = cv2.erode(mask_full, kernel, iterations=ERODE_ITERS)
        mask_full = cv2.dilate(mask_full, kernel, iterations=DILATE_ITERS)
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best_candidate = None
        best_score     = float('inf')

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            b_area = bw * bh
            b_cx = bx + bw // 2
            b_cy = by + bh // 2
            if b_cy < HORIZON_Y:
                continue
            if b_area < MIN_CONTOUR_AREA:
                continue
            if b_area > max_acceptable_area:
                continue
            distance = math.hypot(b_cx - search_cx, b_cy - search_cy)
            if distance > effective_radius:
                continue
            matches_initial = (initial_area * AREA_MIN_FACTOR < b_area < initial_area * AREA_MAX_FACTOR)
            matches_last    = (last_area * AREA_MIN_FACTOR < b_area < last_area * AREA_MAX_FACTOR)
            if not (matches_initial or matches_last):
                continue
            aspect = float(bw) / bh if bh > 0 else 0
            if not (ASPECT_MIN < aspect < ASPECT_MAX):
                continue

            area_diff = abs(b_area - initial_area) / max(initial_area, 1)
            score = distance + (area_diff * 30)

            if score < best_score:
                best_score     = score
                best_candidate = (bx, by, bw, bh)

        # Re-initialize CSRT
        if best_candidate is not None:
            bx, by, bw, bh = best_candidate

            # Pad the recovered box
            pad = REINIT_PADDING
            init_x = max(0, bx - pad)
            init_y = max(0, by - pad)
            init_w = min(FRAME_WIDTH - init_x,  bw + 2 * pad)
            init_h = min(FRAME_HEIGHT - init_y, bh + 2 * pad)

            # Draw the recovery box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                          (0, 255, 255), 3)
            cv2.putText(frame, "RECOVERED!",
                        (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Update memory
            last_known_good_box = (bx, by, bw, bh)
            prev_center = (bx + bw // 2, by + bh // 2)

            # Re-create a FRESH CSRT tracker
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (init_x, init_y, init_w, init_h))
            tracker_alive = True  # CSRT is back in the game

            print(f"[RECOVERED] Frame {frame_number}: Re-acquired at "
                  f"({bx},{by}) size {bw}x{bh} after {lost_frame_count} lost frames")
            lost_frame_count = 0
    out_writer.write(frame)

    # Display
    cv2.imshow("Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[QUIT] User exited.")
        break
    
out_writer.release()
cap.release()
cv2.destroyAllWindows()
print(f"[SAVED] Output video saved to: {OUTPUT_PATH}")
print("[DONE] Tracker shut down cleanly.")