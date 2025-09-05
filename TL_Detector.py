import cv2
import numpy as np
import argparse
from collections import deque
import time
from datetime import datetime

class TrafficLightDetector:
    def __init__(self):
        
        self.color_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),   
                (np.array([170, 120, 70]), np.array([180, 255, 255]))   
            ],
            'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
            'green': [(np.array([40, 50, 50]), np.array([90, 255, 255]))]
        }
        
        self.min_contour_area = 200
        
        self.min_aspect_ratio = 0.3
        self.max_aspect_ratio = 1.2
        
        self.state_buffer = deque(maxlen=15)
        self.state_history = []   
        self.current_state = 'unknown'
        self.last_state_change_time = time.time()
        
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
    
    def preprocess_frame(self, frame):
     
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        return blurred
    
    def detect_color(self, frame_hsv, color):
        """Detect specific color regions in the frame"""
        masks = []
        for lower, upper in self.color_ranges[color]:
            mask = cv2.inRange(frame_hsv, lower, upper)
            masks.append(mask)
        
        combined_mask = masks[0]
        for i in range(1, len(masks)):
            combined_mask = cv2.bitwise_or(combined_mask, masks[i])
        
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def validate_contour(self, contour):
        """Validate if contour resembles a traffic light"""
        area = cv2.contourArea(contour)
        if area < self.min_contour_area:
            return False
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False
      
        return True
    
    def detect_traffic_lights(self, frame):
       
        processed_frame = self.preprocess_frame(frame)
        
        hsv_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        
        detected_lights = []
        
        for color in self.color_ranges.keys():
            mask = self.detect_color(hsv_frame, color)
            
             
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if self.validate_contour(contour):
                    x, y, w, h = cv2.boundingRect(contour)
                    
                     
                    roi = frame[y:y+h, x:x+w]
                    if np.mean(roi) < 50:   
                        continue
                    
                    detected_lights.append({
                        'bbox': (x, y, w, h),
                        'color': color,
                        'confidence': min(cv2.contourArea(contour) / 1000, 1.0)   
                    })
        
        return detected_lights
    
    def classify_state(self, detected_lights):
         
        if not detected_lights:
            new_state = 'unknown'
        else:
             
            best_detection = max(detected_lights, key=lambda x: x['confidence'])
            new_state = best_detection['color']
         
        self.state_buffer.append(new_state)
         
        stable_state = max(set(self.state_buffer), key=self.state_buffer.count)
        
        if stable_state != self.current_state:
             
            timestamp = time.time() - self.start_time
            human_time = datetime.now().strftime("%H:%M:%S")
            self.state_history.append({
                'timestamp': timestamp,
                'human_time': human_time,
                'state': stable_state,
                'frame': self.frame_count
            })
            self.current_state = stable_state
            self.last_state_change_time = time.time()
        
        return stable_state
    
    def draw_annotations(self, frame, detected_lights, overall_state):
        """Draw bounding boxes and labels on the frame"""
        annotated_frame = frame.copy()
        
        for light in detected_lights:
            x, y, w, h = light['bbox']
            color = light['color']
            confidence = light['confidence']
             
            if color == 'red':
                box_color = (0, 0, 255)
            elif color == 'yellow':
                box_color = (0, 255, 255)
            else:   
                box_color = (0, 255, 0)
             
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
             
            label = f"{color} ({confidence:.2f})"
            cv2.putText(annotated_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
         
        state_duration = time.time() - self.last_state_change_time
        cv2.putText(annotated_frame, f"State: {overall_state} ({state_duration:.1f}s)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
         
        current_time = time.time() - self.start_time
        cv2.putText(annotated_frame, f"Time: {current_time:.1f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
         
        fps = self.frame_count / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0
        accuracy = self.detection_count / self.frame_count if self.frame_count > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Detection Rate: {accuracy:.2%}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(self.state_history) > 0:
            history_text = "Recent states:"
            start_idx = max(0, len(self.state_history) - 3)
            for i in range(start_idx, len(self.state_history)):
                state_info = self.state_history[i]
                history_text += f" {state_info['state']}@{state_info['timestamp']:.1f}s"
            
            cv2.putText(annotated_frame, history_text, (10, annotated_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def process_video(self, video_source=0):
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Press 'q' to quit, 's' to save screenshot, 'r' to show state report")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            self.frame_count += 1
            
            detected_lights = self.detect_traffic_lights(frame)
            
            if detected_lights:
                self.detection_count += 1
             
            overall_state = self.classify_state(detected_lights)
             
            annotated_frame = self.draw_annotations(frame, detected_lights, overall_state)
             
            cv2.imshow('Traffic Light Detection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"traffic_light_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Screenshot saved as {filename}")
            elif key == ord('r'):
                 
                self.print_state_report()
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.print_state_report()
        self.print_detection_report()
    
    def print_state_report(self):
        """Print state change report with timestamps"""
        if not self.state_history:
            print("\nNo state changes detected during the session.")
            return
            
        print("\n" + "="*70)
        print("TRAFFIC LIGHT STATE CHANGE REPORT")
        print("="*70)
        print(f"{'Time (s)':<10} {'Clock Time':<10} {'Frame':<8} {'State':<8}")
        print("-" * 70)
        
        for state_info in self.state_history:
            print(f"{state_info['timestamp']:<10.2f} {state_info['human_time']:<10} "
                  f"{state_info['frame']:<8} {state_info['state']:<8}")
        
        state_durations = {}
        for i in range(len(self.state_history)):
            state = self.state_history[i]['state']
            start_time = self.state_history[i]['timestamp']
            end_time = self.state_history[i+1]['timestamp'] if i < len(self.state_history)-1 else time.time() - self.start_time
            duration = end_time - start_time
            
            if state not in state_durations:
                state_durations[state] = 0
            state_durations[state] += duration
        
        print("-" * 70)
        print("TIME SPENT IN EACH STATE:")
        for state, duration in state_durations.items():
            print(f"{state}: {duration:.2f}s ({duration/(time.time()-self.start_time):.1%})")
        
        print("="*70)
    
    def print_detection_report(self):
        """Print detection accuracy report"""
        total_time = time.time() - self.start_time
        fps = self.frame_count / total_time if total_time > 0 else 0
        accuracy = self.detection_count / self.frame_count if self.frame_count > 0 else 0
        
        print("\n" + "="*50)
        print("DETECTION PERFORMANCE REPORT")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Frames with detection: {self.detection_count}")
        print(f"Detection accuracy: {accuracy:.2%}")
        print(f"Average FPS: {fps:.2f}")
        print(f"Processing time: {total_time:.2f} seconds")
        print(f"State changes detected: {len(self.state_history)}")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Traffic Light Detection System')
    parser.add_argument('--input', type=str, default='0',
                       help='Video file path or camera device ID (default: 0)')
    args = parser.parse_args()
   
    detector = TrafficLightDetector()
   
    try:
        video_source = int(args.input)
    except ValueError:
        video_source = args.input
    
    detector.process_video(video_source)

if __name__ == "__main__":
    main()