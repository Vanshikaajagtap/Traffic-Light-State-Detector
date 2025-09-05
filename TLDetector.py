import cv2
import numpy as np
import argparse
from collections import deque
import time

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
        
         
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
    
    def preprocess_frame(self, frame):
         
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        return blurred
    
    def detect_color(self, frame_hsv, color):
         
        masks = []
        for lower, upper in self.color_ranges[color]:
            mask = cv2.inRange(frame_hsv, lower, upper)
            masks.append(mask)
        
        
        combined_mask = masks[0]
        for i in range(1, len(masks)):
            combined_mask = cv2.bitwise_or(combined_mask, masks[i])
        
        return combined_mask
    
    def validate_contour(self, contour):
         
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
        """Classify the overall traffic light state"""
        if not detected_lights:
            self.state_buffer.append('unknown')
        else:
             
            best_detection = max(detected_lights, key=lambda x: x['confidence'])
            self.state_buffer.append(best_detection['color'])
        
         
        if len(self.state_buffer) > 0:
            return max(set(self.state_buffer), key=self.state_buffer.count)
        return 'unknown'
    
    def draw_annotations(self, frame, detected_lights, overall_state):
         
        annotated_frame = frame.copy()
        
        for light in detected_lights:
            x, y, w, h = light['bbox']
            color = light['color']
            confidence = light['confidence']
            
             
            if color == 'red':
                box_color = (0, 0, 255)
            elif color == 'yellow':
                box_color = (0, 255, 255)
            else:  # green
                box_color = (0, 255, 0)
            
             
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
            
             
            label = f"{color} ({confidence:.2f})"
            cv2.putText(annotated_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
         
        cv2.putText(annotated_frame, f"State: {overall_state}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
         
        fps = self.frame_count / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0
        accuracy = self.detection_count / self.frame_count if self.frame_count > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Detection Rate: {accuracy:.2%}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video(self, video_source=0):
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Press 'q' to quit, 's' to save screenshot")
        
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
        
         
        cap.release()
        cv2.destroyAllWindows()
        
         
        self.print_report()
    
    def print_report(self):
         
        total_time = time.time() - self.start_time
        fps = self.frame_count / total_time if total_time > 0 else 0
        accuracy = self.detection_count / self.frame_count if self.frame_count > 0 else 0
        
        print("\n" + "="*50)
        print("TRAFFIC LIGHT DETECTION REPORT")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Frames with detection: {self.detection_count}")
        print(f"Detection accuracy: {accuracy:.2%}")
        print(f"Average FPS: {fps:.2f}")
        print(f"Processing time: {total_time:.2f} seconds")
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