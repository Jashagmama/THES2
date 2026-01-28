"""
Image processing utilities for worksheet scanning
"""
from PIL import Image, ImageFilter, ImageOps
import numpy as np

class BoundaryDetector:
    """Detect document boundaries in images"""
    
    @staticmethod
    def detect_document_bounds(image_path):
        """
        Detect document boundaries using edge detection
        Returns: dict with x1, y1, x2, y2 coordinates
        """
        try:
            # Open image
            img = Image.open(image_path)
            
            # Convert to grayscale
            gray = img.convert('L')
            
            # Apply edge detection
            edges = gray.filter(ImageFilter.FIND_EDGES)
            
            # Convert to numpy array for processing
            img_array = np.array(edges)
            
            # Find non-zero pixels (edges)
            rows = np.any(img_array > 30, axis=1)
            cols = np.any(img_array > 30, axis=0)
            
            # Find boundaries
            if rows.any() and cols.any():
                row_indices = np.where(rows)[0]
                col_indices = np.where(cols)[0]
                
                y1 = int(row_indices[0])
                y2 = int(row_indices[-1])
                x1 = int(col_indices[0])
                x2 = int(col_indices[-1])
                
                # Add padding (5% margin)
                width = img.width
                height = img.height
                padding_x = int(width * 0.05)
                padding_y = int(height * 0.05)
                
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(width, x2 + padding_x)
                y2 = min(height, y2 + padding_y)
                
                return {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'detected': True
                }
            else:
                # Fallback to default margins
                return BoundaryDetector.get_default_bounds(img)
                
        except Exception as e:
            print(f"Boundary detection error: {e}")
            # Return default bounds on error
            return BoundaryDetector.get_default_bounds(img)
    
    @staticmethod
    def get_default_bounds(img):
        """Get default bounds with 5% margin"""
        width = img.width
        height = img.height
        
        return {
            'x1': int(width * 0.05),
            'y1': int(height * 0.05),
            'x2': int(width * 0.95),
            'y2': int(height * 0.95),
            'detected': False
        }
    
    @staticmethod
    def detect_using_contours(image_path):
        """
        Advanced detection using OpenCV contours
        Requires: pip install opencv-python
        """
        try:
            import cv2
            
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                # Fallback to PIL
                pil_img = Image.open(image_path)
                return BoundaryDetector.get_default_bounds(pil_img)
            
            height, width = img.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Check if contour is significant (> 10% of image area)
                contour_area = w * h
                image_area = width * height
                
                if contour_area > image_area * 0.1:
                    return {
                        'x1': int(x),
                        'y1': int(y),
                        'x2': int(x + w),
                        'y2': int(y + h),
                        'detected': True
                    }
            
            # Fallback to default
            pil_img = Image.open(image_path)
            return BoundaryDetector.get_default_bounds(pil_img)
            
        except ImportError:
            # OpenCV not installed, use PIL method
            return BoundaryDetector.detect_document_bounds(image_path)
        except Exception as e:
            print(f"Contour detection error: {e}")
            pil_img = Image.open(image_path)
            return BoundaryDetector.get_default_bounds(pil_img)


class WorksheetProcessor:
    """Additional image processing utilities"""
    
    @staticmethod
    def enhance_for_grading(image_file):
        """Optimize image for automated grading"""
        from io import BytesIO
        from django.core.files.uploadedfile import InMemoryUploadedFile
        import sys
        
        img = Image.open(image_file)
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Auto-rotate based on EXIF
        img = ImageOps.exif_transpose(img)
        
        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.8)
        
        # Save
        output = BytesIO()
        img.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return InMemoryUploadedFile(
            output, 'ImageField',
            f"processed_{image_file.name}",
            'image/jpeg',
            sys.getsizeof(output), None
        )
