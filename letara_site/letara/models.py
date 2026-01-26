from django.db import models
from django.utils import timezone
from PIL import Image, ImageDraw
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
import sys
import json

class ScannedDocument(models.Model):
    title = models.CharField(max_length=255, blank=True)
    original_image = models.ImageField(upload_to='original_documents/%Y/%m/%d/')
    cropped_image = models.ImageField(upload_to='cropped_documents/%Y/%m/%d/', null=True, blank=True)
    upload_method = models.CharField(
        max_length=10,
        choices=[('camera', 'Camera Capture'), ('upload', 'File Upload')],
        default='upload'
    )
    uploaded_at = models.DateTimeField(default=timezone.now)
    file_size = models.IntegerField(null=True, blank=True)
    
    # Store crop coordinates as JSON
    crop_coordinates = models.JSONField(null=True, blank=True)
    is_cropped = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return self.title or f"Worksheet {self.id}"
    
    def apply_crop(self, x1, y1, x2, y2):
        """Apply crop based on user-defined boundaries"""
        if not self.original_image:
            return False
        
        try:
            # Open original image
            img = Image.open(self.original_image)
            
            # Ensure coordinates are in correct order
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            
            # Crop the image
            cropped = img.crop((left, top, right, bottom))
            
            # Save cropped image
            output = BytesIO()
            cropped.save(output, format='JPEG', quality=95)
            output.seek(0)
            
            # Store crop coordinates
            self.crop_coordinates = {
                'x1': left, 'y1': top,
                'x2': right, 'y2': bottom
            }
            
            # Create file
            self.cropped_image = InMemoryUploadedFile(
                output, 'ImageField',
                f"cropped_{self.original_image.name.split('/')[-1]}",
                'image/jpeg',
                sys.getsizeof(output), None
            )
            
            self.is_cropped = True
            self.save()
            return True
            
        except Exception as e:
            print(f"Crop error: {e}")
            return False
    
    def get_display_image(self):
        """Return the best image to display"""
        return self.cropped_image if self.is_cropped and self.cropped_image else self.original_image
    
    def save(self, *args, **kwargs):
        if self.original_image and not self.file_size:
            self.file_size = self.original_image.size
        super().save(*args, **kwargs)
