import cv2
import numpy as np
import pytesseract
import os

# If needed, uncomment and set your tesseract path:
# pytesseract.pytesseract.tesseract_cmd = r""

cascade_path = r"/home/Vados/Documents/AdoVs/VPSc/haarcascade_russian_plate_number.xml"
if not os.path.exists(cascade_path):
    print(f"Error: Haar cascade file not found at {cascade_path}")
    exit(1)

cascade = cv2.CascadeClassifier(cascade_path)

states = {
    "AN": "Andaman and Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "DN": "Dadra and Nagar Haveli",
    "DD": "Daman and Diu", "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat",
    "HR": "Haryana", "HP": "Himachal Pradesh", "JK": "Jammu and Kashmir",
    "KA": "Karnataka", "KL": "Kerala", "LD": "Lakshadweep", "MP": "Madhya Pradesh",
    "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya", "MZ": "Mizoram",
    "NL": "Nagaland", "OD": "Odisha", "PY": "Pondicherry", "PN": "Punjab",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura",
    "UP": "Uttar Pradesh", "WB": "West Bengal", "CG": "Chhattisgarh",
    "TS": "Telangana", "JH": "Jharkhand", "UK": "Uttarakhand"
}

locked_state = None
reset_flag = False

capture_dir = r"/home/Vados/Documents/AdoVs/VPSc/car"
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir, exist_ok=True)
    print(f"Created directory: {capture_dir}")

def preprocess_plate(plate):
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    plate_blur = cv2.bilateralFilter(plate_gray, 9, 75, 75)
    plate_thresh = cv2.adaptiveThreshold(plate_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    plate_clean = cv2.morphologyEx(plate_thresh, cv2.MORPH_CLOSE, kernel)
    return plate_clean

def extract_text_from_plate(plate):
    config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(plate, config=config)
    text = ''.join(e for e in text if e.isalnum())
    return text

def detect_and_recognize_plate(frame):
    global locked_state, reset_flag

    if locked_state and not reset_flag:
        cv2.putText(frame, f"State Locked: {locked_state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = cascade.detectMultiScale(gray, 1.1, 10)
    print(f"Detected {len(plates)} plate(s)")

    for (x, y, w, h) in plates:
        plate = frame[y:y + h, x:x + w]
        cv2.imshow("Cropped Plate", plate)

        plate_preprocessed = preprocess_plate(plate)
        text = extract_text_from_plate(plate_preprocessed)
        print(f"[INFO] Extracted Text: {text}")

        if len(text) >= 2:
            state_code = text[:2].upper()
            if state_code in states:
                locked_state = states[state_code]
                print(f"[INFO] Detected State: {locked_state}")

                plate_filename = os.path.join(capture_dir, f"{text}.png")
                success = cv2.imwrite(plate_filename, plate)
                if success:
                    print(f"[INFO] License plate saved as: {plate_filename}")
                else:
                    print("[ERROR] Failed to save the plate image.")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Plate: {text}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"State: {locked_state}", (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                with open("plate_log.txt", "a") as f:
                    f.write(f"{text}, {locked_state}, {plate_filename}\n")
                
                break  
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Scanning...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open the camera.")
        return
    
    print("🔁 Press 'q' to quit, 'r' to reset state.")
    global reset_flag, locked_state

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture an image.")
            break

        if reset_flag:
            locked_state = None
            reset_flag = False

        processed_frame = detect_and_recognize_plate(frame)
        cv2.imshow("License Plate Detection", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_flag = True
            print("[INFO] Resetting state...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
