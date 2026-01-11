An interactive Streamlit-based computer vision application for detecting and analyzing basic geometric shapes using contour-based feature extraction.

The Shape & Contour Analyzer allows users to upload an image and automatically:
1. Detect basic geometric shapes
2. Count the number of objects
3. Compute area and perimeter of each shape
4. Visualize detected contours and labels, 
5. Export analysis results as a CSV file
   
The application is designed as an educational tool to demonstrate concepts such as edge detection, contour extraction, and shape feature analysis.
The system identifies the following shapes: Triangle, Rectangle, Circle, Polygon (all other shapes)

Technologies Used: 
Python
Streamlit – interactive web dashboard
OpenCV – image processing & contour detection
NumPy – numerical operations
Pandas – tabular data handling
Plotly – data visualization
Pillow (PIL) – image loading

Methodology: 
Image Upload: User uploads an image via the Streamlit interface.
Preprocessing
1. Convert image to grayscale
2. Apply Gaussian blur to reduce noise
Edge Detection- Use Canny edge detector to highlight object boundaries.
Contour Detection- Extract external contours using OpenCV.
Shape Classification
1. Approximate contours to determine number of vertices
2. Use geometric properties (vertices & circularity) to classify shapes.
Feature Extraction: Area, Perimeter, Centroid
Visualization & Analytics
1. Display original image, edge map, and detected shapes
2. Show summary metrics and charts
3. Allow CSV download of results

How to Run Locally?
Install dependencies:
pip install -r requirements.txt
Run the application:
streamlit run app.py

NOTE: PERFORMANCE MAY VARY WITH NOISY OR LOW-CONTRAST IMAGES 
