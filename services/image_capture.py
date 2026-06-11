import os
import sys
import uuid
import json
import base64
import shutil
import subprocess
import cv2
import numpy as np
from scipy import stats
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Enable CORS for frontend/Ionic frameworks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory Setup
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
ASSESSED_DIR = "assessed_cracks"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ASSESSED_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")
app.mount("/assessed", StaticFiles(directory=ASSESSED_DIR), name="assessed")

PIXEL_TO_MM = 0.1
GAP_THRESHOLD_PIXELS = 5

# =========================================================================
# --- GEOMETRIC MODELING HELPERS (UNION-FIND & DISTANCE) ---
# =========================================================================

class CrackUnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def calculate_min_edge_distance(contour_a, contour_b):
    pts_a = contour_a.reshape(-1, 2)
    pts_b = contour_b.reshape(-1, 2)
    dist_matrix = np.linalg.norm(pts_a[:, np.newaxis] - pts_b, axis=2)
    return np.min(dist_matrix)

def convert_to_base64(image):
    success, buffer = cv2.imencode('.jpg', image)
    if success:
        img_base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64_str}"
    return None

# =========================================================================
# --- CORE STRUCTURAL ANALYSIS ENGINE ---
# =========================================================================

def preprocess_and_save_format(image_path):
    base_path, _ = os.path.splitext(image_path)
    txt_path = f"{base_path}.txt"
    
    # 1. Read and standardize
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image_path) 
    
    if img_gray is None:
        return np.zeros((416, 416), dtype=np.uint8), np.zeros((416, 416, 3), dtype=np.uint8), [], [], ""
        
    # Resize BOTH
    resized_gray = cv2.resize(img_gray, (416, 416), interpolation=cv2.INTER_AREA)
    resized_color = cv2.resize(img_color, (416, 416), interpolation=cv2.INTER_AREA)
    
    # Convert to Base64 only when needed (e.g., for JSON response), not for image processing
    resized_img_base64 = convert_to_base64(resized_color)

    if "," in resized_img_base64:
        base64_clean = resized_img_base64.split(",")[1]
    else:
        base64_clean = resized_img_base64

    img_bytes = base64.b64decode(base64_clean)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    debug_img_matrix = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if debug_img_matrix is not None:
        debug_dir = "debug_crops"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "resized_base64_debug.jpg"), debug_img_matrix)
        print("DEBUG: Successfully saved base64 image to debug_outputs/resized_base64_debug.jpg", flush=True)

    # 2. Advanced Preprocessing (Use the resized_gray array, NOT the base64 string)
    smoothed_small = cv2.bilateralFilter(resized_gray, d=7, sigmaColor=50, sigmaSpace=50)
    whitehot_crack = cv2.adaptiveThreshold(
        smoothed_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 14
    )

    # 3. Morphological Cleaning
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    cleaned_mask = cv2.morphologyEx(whitehot_crack, cv2.MORPH_OPEN, kernel_small)

    # 4. Connected Components & Contour Extraction
    num_labels, labels, stats_map, centroids = cv2.connectedComponentsWithStats(
        cleaned_mask, connectivity=8, ltype=cv2.CV_32S
    )
    
    final_cleaned_mask = np.zeros_like(cleaned_mask)
    for i in range(1, num_labels):
        if stats_map[i, cv2.CC_STAT_AREA] >= 20:  
            final_cleaned_mask[labels == i] = 255

    contours, _ = cv2.findContours(final_cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    crack_records = []

    # 5. Extract Individual Component Geometric Morphologies
    for count in contours:
        if cv2.contourArea(count) < 5:
            continue
    
        x, y, w, h = cv2.boundingRect(count)
        region_of_interest = np.zeros((h + 10, w + 10), dtype=np.uint8)
        shifted_count = count - [x - 5, y - 5]
        cv2.drawContours(region_of_interest, [shifted_count], -1, 255, -1)
        
        dist_map = cv2.distanceTransform(region_of_interest, cv2.DIST_L2, 5)
        skeleton = cv2.ximgproc.thinning(region_of_interest, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        
        pixel_length = np.sum(skeleton == 255)
        length_mm = pixel_length * PIXEL_TO_MM

        if pixel_length == 0:
            continue

        widths_mm = dist_map[skeleton == 255] * 2.0 * PIXEL_TO_MM
        max_w = np.max(widths_mm)
        mean_w = np.mean(widths_mm)
        
        # --- SCIPY FIX ---
        mode_result = stats.mode(np.round(widths_mm, 2), keepdims=True)
        # Handle different scipy versions for mode extraction
        mode_w = float(mode_result.mode.item(0) if hasattr(mode_result.mode, 'item') else mode_result.mode)
        
        rotated_box = cv2.minAreaRect(count)
        
        # Access the angle at index 2
        orientation_angle = float(rotated_box[2])

        crack_records.append({
            "contour": count,
            "length_mm": length_mm,
            "max_width_mm": max_w,
            "mean_width_mm": mean_w,
            "mode_width_mm": mode_w,
            "orientation_deg": orientation_angle,
            "widths_raw": widths_mm
        })

    num_fragments = len(crack_records)
    
    # If no cracks, return early
    if num_fragments == 0:
        return final_cleaned_mask, resized_color, [], [], resized_img_base64

    # 6. Execute Union-Find
    uf = CrackUnionFind(num_fragments)
    for i in range(num_fragments):
        for j in range(i + 1, num_fragments):
            gap = calculate_min_edge_distance(crack_records[i]["contour"], crack_records[j]["contour"])
            if gap <= GAP_THRESHOLD_PIXELS:
                uf.union(i, j)

    clusters = {}
    for i in range(num_fragments):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # 7. Build Metrics and Draw Overlays
    bounding_boxes = []
    cropped_roi_objectStore = []
    output_img = resized_color.copy() # Use the color version for drawing
    
    target_w, target_h = 416, 416

    with open(txt_path, "w") as txt_file:
        for chain_id, indices in enumerate(clusters.values()):
            sub_cracks = [crack_records[idx] for idx in indices]
            total_length = sum(c["length_mm"] for c in sub_cracks)
            max_width = max(c["max_width_mm"] for c in sub_cracks)
            mean_width = np.mean(np.concatenate([c["widths_raw"] for c in sub_cracks]))
            
            # Weighted average angle calculations
            angles = np.array([c["orientation_deg"] for c in sub_cracks])
            lengths = np.array([c["length_mm"] for c in sub_cracks])
            if total_length > 0:
                percentage_weights = lengths / total_length
                weighted_average_angle = np.sum(angles * percentage_weights)
                within_tolerance = np.abs(angles - weighted_average_angle) <= 10.0
                final_orientation = f"{weighted_average_angle:.1f}°" if np.sum(within_tolerance) > (len(sub_cracks) / 2.0) else "Curve"
            else:
                final_orientation = "Unknown"

            # Combine contour points to calculate absolute global parameters
            all_contour_points = np.concatenate([c["contour"] for c in sub_cracks], axis=0)
            bx, by, bw, bh = cv2.boundingRect(all_contour_points)
            
            # --- SAFETY CHECK & ROI PROCESSING ---
            if bw > 0 and bh > 0:
                # Calculate Relative Percentages
                pct_x = float((bx / target_w) * 100.0)
                pct_y = float((by / target_h) * 100.0)
                pct_w = float((bw / target_w) * 100.0)
                pct_h = float((bh / target_h) * 100.0)

                # Write Absolute data to metrics log text file
                txt_file.write(f"{bw} {bh} {bx} {by}\n")

                # Draw Absolute boxes and tracking points to the validation frame
                cv2.rectangle(output_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cx, cy = bx + (bw / 2), by + (bh / 2)
                cv2.circle(output_img, (int(cx), int(cy)), 4, (0, 0, 255), -1)

                # Log validation statements to the environment console
                print(f"\n[Structural Unit {chain_id + 1}]", flush=True)
                print(f" -> Absolute Pixels: Box(x={bx}, y={by}, w={bw}, h={bh})", flush=True)
                print(f" -> Relative Proportions: Box(x={pct_x:.1f}%, y={pct_y:.1f}%, w={pct_w:.1f}%, h={pct_h:.1f}%)", flush=True)
                print(f" -> Geometric Profile: Length={total_length:.2f}mm, AvgWidth={mean_width:.2f}mm, Orient={final_orientation}", flush=True)
                print(f"DEBUG: Creating Box for ID {int(chain_id + 1)} -> x:{bx}, y:{by}, w:{bw}, h:{bh}", flush=True)

                # 8. Region Of Interest (ROI) Absolute Cropping & Matrix Extraction
                cropped_roi = resized_color[by:by+bh, bx:bx+bw]

                # cv2.imshow(f"Debug Crop {chain_id + 1}", cropped_roi)
                # cv2.waitKey(0) # Waits indefinitely until you press a key
                # cv2.destroyAllWindows() # Closes the window
                # ------------------------------

                success, buffer = cv2.imencode('.jpg', cropped_roi)
                
                # --- A. Generate Base64 for the API response ---
                success, buffer = cv2.imencode('.jpg', cropped_roi)
                image_data_uri = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}" if success else None
                
                # --- B. Save locally for debug_crops ---
                box_id = int(chain_id + 1)
                debug_dir = "debug_crops"
                os.makedirs(debug_dir, exist_ok=True) # Ensure folder exists
                save_path = os.path.join(debug_dir, f"box_{box_id}.jpg")
                
                if success:
                    cv2.imwrite(save_path, cropped_roi)
                    print(f"DEBUG: Successfully saved cropped image {box_id} to {save_path}", flush=True)
                
                # --- C. Store in the object store for the API return ---
                cropped_roi_objectStore.append({
                    "box_id": box_id,
                    "image_data": image_data_uri
                })


                # 9. Pixel Density Vector Computations inside Cropped Matrix
                # 9. Pixel Density Vector Computations inside Cropped Matrix
                # Convert the color crop to grayscale first
                gray_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)

                # Apply threshold to the 1-channel grayscale image
                _, bitmap_cropped = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
                
                total_pixels = bitmap_cropped.size
                white_pixels = cv2.countNonZero(bitmap_cropped) if total_pixels > 0 else 0
                black_pixels = total_pixels - white_pixels
                white_percentage = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                black_percentage = (black_pixels / total_pixels) * 100 if total_pixels > 0 else 0

                # Store payload containing both coordinate types and structural analytics (Explicitly cast for JSON safety)
                # bounding_boxes.append({
                #     "id": int(chain_id + 1),
                #     "absolute": {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)},
                #     "relative": {
                #         "pct_x": float(round(pct_x, 2)), 
                #         "pct_y": float(round(pct_y, 2)), 
                #         "pct_width": float(round(pct_w, 2)), 
                #         "pct_height": float(round(pct_h, 2))
                #     },
                #     "metrics": {
                #         "avgWidth_mm": float(round(mean_width, 2)),
                #         "maxWidth_mm": float(round(max_width, 2)),
                #         "crackLength_mm": float(round(total_length, 2)),
                #         "orientation": str(final_orientation)
                #     },
                #     "pixel_analysis": {
                #         "total_pixels": int(total_pixels), 
                #         "white_percentage": float(round(white_percentage, 2)), 
                #         "black_percentage": float(round(black_percentage, 2))
                #     }
                # })
                bounding_boxes.append(
                    {"x": int(bx),
                     "y": int(by),
                     "w": int(bw),
                     "h": int(bh)}
                )
            else:
                print(f"DEBUG: Skipping invalid crop for Box {chain_id + 1} (Zero width/height)", flush=True)

    # Sort items based on absolute bounding spatial size surface areas
    bounding_boxes.sort(key=lambda b: b["w"] * b["h"], reverse=True)

    # ADDED: Print the bounding boxes after sorting to the terminal
    print("\n--- OPENCV CHECK --- Sorted bounding_boxes array:", flush=True)
    print(json.dumps(bounding_boxes, indent=2), flush=True)

    

    return final_cleaned_mask, output_img, bounding_boxes, cropped_roi_objectStore, resized_img_base64

# =========================================================================
# --- API ROUTING LAYER ---
# =========================================================================

@app.get("/api/hello")
async def get_message():
    return {"message": "Hello from your advanced FastAPI analysis engine! jUNE 9 🚀"}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    raw_file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(raw_file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # Execute structural pipeline parsing
    # final_cleaned_mask, visual_img, bounding_boxes, cropped_roi_objectStore, resized_img_base64 = preprocess_and_save_format(raw_file_path)
    final_cleaned_mask, output_img, bounding_boxes, cropped_roi_objectStore, resized_img_base64 = preprocess_and_save_format(raw_file_path)

    # Save validation image representation to system directories
    processed_filename = f"processed_{file.filename}"
    processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)
    cv2.imwrite(processed_file_path, output_img)
    

    
    # # SYSTEM VISUAL VALIDATION: Open output image automatically using OS native graphics module
    # try:
    #     if sys.platform == "win32":
    #         os.startfile(processed_file_path)
    #     elif sys.platform == "darwin":
    #         subprocess.Popen(["open", processed_file_path])
    #     else:
    #         subprocess.Popen(["xdg-open", processed_file_path])
        
    #     # ADDED flush=True here as well
    #     print(f"Validation Display Active: Opened '{processed_filename}' in OS system window.", flush=True)
    # except Exception as e:
    #     print(f"Log: Could not spawn automated native window workspace validation. Error: {e}", flush=True)
        
    return {
        "message": "Image analyzed successfully using Union-Find geometric modeling!",
        "rawImagePath": f"/uploads/{file.filename}",
        "processedImagePath": f"/processed/{processed_filename}",
        "bounding_boxes": bounding_boxes, 
        "cropped_roi_objectStore": cropped_roi_objectStore, 
        "resizedImagePath": resized_img_base64 
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)