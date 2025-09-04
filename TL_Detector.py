"""
Color Detection System using HSV Color Space
Author: [Your Name]
Date: 2024

This script detects red, yellow, and green colored objects in real-time
using webcam feed or video files. It uses HSV color space for better
color segmentation and includes real-time tuning capabilities.
"""

import cv2
import numpy as np
import argparse
import time

def parse_arguments():
    """Parse command line arguments for video source selection"""
    parser = argparse.ArgumentParser(description='Real-time Color Detection using HSV')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (uses webcam if not provided)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0 for primary camera)')
    parser.add_argument('--record', action='store_true',
                       help='Record output to video file')
    return parser.parse_args()

def initialize_video_writer(frame_width, frame_height, fps=20):
    """Initialize video writer for recording output"""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"color_detection_output_{timestamp}.avi"
    return cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

def create_tuning_window():
    """Create tuning window with trackbars for real-time HSV adjustment"""
    cv2.namedWindow('HSV Tuning Panel', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('HSV Tuning Panel', 400, 400)
    
    # Default HSV values for red (will be used for tuning)
    cv2.createTrackbar('Hue Min', 'HSV Tuning Panel', 0, 179, lambda x: None)
    cv2.createTrackbar('Hue Max', 'HSV Tuning Panel', 10, 179, lambda x: None)
    cv2.createTrackbar('Sat Min', 'HSV Tuning Panel', 100, 255, lambda x: None)
    cv2.createTrackbar('Sat Max', 'HSV Tuning Panel', 255, 255, lambda x: None)
    cv2.createTrackbar('Val Min', 'HSV Tuning Panel', 100, 255, lambda x: None)
    cv2.createTrackbar('Val Max', 'HSV Tuning Panel', 255, 255, lambda x: None)

def get_tuning_values():
    """Get current values from HSV tuning trackbars"""
    return {
        'h_min': cv2.getTrackbarPos('Hue Min', 'HSV Tuning Panel'),
        'h_max': cv2.getTrackbarPos('Hue Max', 'HSV Tuning Panel'),
        's_min': cv2.getTrackbarPos('Sat Min', 'HSV Tuning Panel'),
        's_max': cv2.getTrackbarPos('Sat Max', 'HSV Tuning Panel'),
        'v_min': cv2.getTrackbarPos('Val Min', 'HSV Tuning Panel'),
        'v_max': cv2.getTrackbarPos('Val Max', 'HSV Tuning Panel')
    }

def apply_morphological_operations(mask):
    """Apply morphological operations to clean up the mask"""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Fill holes
    return mask

def calculate_circularity(contour):
    """Calculate circularity of a contour (1.0 = perfect circle)"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        return (4 * np.pi * area) / (perimeter * perimeter)
    return 0

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize video capture
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = f"Video: {args.video}"
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Webcam {args.camera}"
    
    # Verify video source
    if not cap.isOpened():
        print(f"Error: Could not open {source_name}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if args.video else 30
    
    # Initialize video writer if recording is enabled
    video_writer = None
    if args.record:
        video_writer = initialize_video_writer(frame_width, frame_height, fps)
        print("Recording enabled - output will be saved to file")
    
    # Create tuning window
    create_tuning_window()
    
    # Detection parameters (tune these based on your environment)
    MIN_AREA = 500           # Minimum contour area in pixels
    MIN_CIRCULARITY = 0.6    # Minimum circularity for circle detection
    KERNEL_SIZE = 5          # Size of morphological operation kernel
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    detection_stats = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
    
    print("=" * 50)
    print("COLOR DETECTION SYSTEM")
    print("=" * 50)
    print(f"Source: {source_name}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print("\nControls:")
    print("  'q' - Quit application")
    print("  'r' - Reset detection counters")
    print("  's' - Save screenshot")
    print("  'p' - Pause/Resume processing")
    print("  Use trackbars for real-time HSV tuning")
    print("=" * 50)
    
    paused = False
    
    while True:
        if not paused:
            # Read frame from video source
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            frame_count += 1
            
            # Flip frame horizontally for webcam (mirror effect)
            if not args.video:
                frame = cv2.flip(frame, 1)
            
            # Convert to HSV color space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Get current tuning values
            tune_vals = get_tuning_values()
            
            # Define HSV ranges for each color
            # Red (dual range due to HSV wrap-around)
            lower_red1 = np.array([tune_vals['h_min'], tune_vals['s_min'], tune_vals['v_min']])
            upper_red1 = np.array([tune_vals['h_max'], tune_vals['s_max'], tune_vals['v_max']])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            # Yellow
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            # Green
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            
            # Create color masks
            red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
            green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
            
            # Clean up masks
            red_mask = apply_morphological_operations(red_mask)
            yellow_mask = apply_morphological_operations(yellow_mask)
            green_mask = apply_morphological_operations(green_mask)
            
            # Color processing configuration
            colors_config = {
                'RED': (red_mask, (0, 0, 255), lower_red1, upper_red1),
                'YELLOW': (yellow_mask, (0, 255, 255), lower_yellow, upper_yellow),
                'GREEN': (green_mask, (0, 255, 0), lower_green, upper_green)
            }
            
            # Reset frame detection count
            frame_detections = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
            
            # Process each color
            for color_name, (mask, bgr_color, lower, upper) in colors_config.items():
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < MIN_AREA:
                        continue
                    
                    # Calculate circularity and aspect ratio
                    circularity = calculate_circularity(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h) if h > 0 else 0
                    
                    # Filter based on shape properties
                    if (circularity >= MIN_CIRCULARITY and 
                        0.5 <= aspect_ratio <= 2.0):  # Reasonable aspect ratio range
                        
                        # Draw bounding box and information
                        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)
                        cv2.putText(frame, color_name, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
                        
                        # Update detection counters
                        frame_detections[color_name] += 1
                        detection_stats[color_name] += 1
            
            # Calculate and display performance metrics
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Display information overlay
            info_text = [
                f"FPS: {current_fps:.1f}",
                f"RED: {frame_detections['RED']}",
                f"YELLOW: {frame_detections['YELLOW']}", 
                f"GREEN: {frame_detections['GREEN']}",
                f"Frame: {frame_count}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display main output
            cv2.imshow('Color Detection System', frame)
            
            # Display masks for debugging
            cv2.imshow('Red Mask', red_mask)
            cv2.imshow('Yellow Mask', yellow_mask) 
            cv2.imshow('Green Mask', green_mask)
            
            # Write frame to output video if recording
            if video_writer is not None:
                video_writer.write(frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detection_stats = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
            print("Detection counters reset")
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    total_detections = sum(detection_stats.values())
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("FINAL DETECTION STATISTICS")
    print("=" * 50)
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {frame_count/total_time:.2f}")
    print(f"Total detections: {total_detections}")
    print(f"  RED: {detection_stats['RED']}")
    print(f"  YELLOW: {detection_stats['YELLOW']}")
    print(f"  GREEN: {detection_stats['GREEN']}")
    print("=" * 50)

if __name__ == "__main__":
    main()