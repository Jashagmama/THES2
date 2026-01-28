from django.core import validators
from django.db import models
from django.forms import FloatField
from django.utils import timezone
from django.core.validators import MaxValueValidator, MinValueValidator
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
    is_graded = models.BooleanField(default=False)
    
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


class HandwritingGrade(models.Model):
    document = models.OneToOneField(
        ScannedDocument,
        on_delete=models.CASCADE,
        related_name='hw_grade',
    )

    letter_form = models.FloatField(
        validators=[validators.MinValueValidator(0), validators.MaxValueValidator(100)],
        help_text="Quality of letter formation"
    )

    size = models.FloatField(
        validators=[validators.MinValueValidator(0), validators.MaxValueValidator(100)],
        help_text="Letter size in relation to template character size"
    )

    line_align = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Alignment with baseline and proper spacing"
    )
    orientation = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Proper slant and angle of letters"
    )

    # Overall grade (auto-calculated or manual)
    overall_score = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Overall handwriting score"
    )

    graded_at = models.DateTimeField(auto_now_add=True)

    comments = models.TextField(blank=True, help_text="Detailed feedback for the student")
    strengths = models.TextField(blank=True, help_text="What the student does well")
    areas_for_improvement = models.TextField(blank=True, help_text="Areas that need work")

    def __str__(self):
        return f"Grade for {self.document.title or f'Worksheet {self.document.id}'}"

    def get_overall_score(self):
        """Calculate overall score from individual criteria"""
        if self.overall_score is not None:
            return self.overall_score

        # Calculate weighted average
        criteria = [self.letter_form, self.size, self.line_align, self.orientation]
        weights = [0.3, 0.25, 0.25, 0.2]  # Adjust weights as needed

        # Normalize weights to sum to 1
        total_weight = sum(weights[:len(criteria)])
        normalized_weights = [w / total_weight for w in weights[:len(criteria)]]

        # Calculate weighted score
        score = sum(c * w for c, w in zip(criteria, normalized_weights))
        return round(score, 2)

