from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
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
    
    # Grading status
    is_graded = models.BooleanField(default=False)
    graded_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return self.title or f"Worksheet {self.id}"
    
    def apply_crop(self, x1, y1, x2, y2):
        """Apply crop based on user-defined boundaries"""
        if not self.original_image:
            return False
        
        try:
            img = Image.open(self.original_image)
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            cropped = img.crop((left, top, right, bottom))
            output = BytesIO()
            cropped.save(output, format='JPEG', quality=95)
            output.seek(0)
            self.crop_coordinates = {'x1': left, 'y1': top, 'x2': right, 'y2': bottom}
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
    
    def get_overall_grade(self):
        """Calculate overall grade from letter averages"""
        letter_summaries = self.letter_summaries.all()
        if letter_summaries.exists():
            avg_letter_form = sum(l.avg_letter_form for l in letter_summaries) / len(letter_summaries)
            avg_size = sum(l.avg_size for l in letter_summaries) / len(letter_summaries)
            avg_line_align = sum(l.avg_line_align for l in letter_summaries) / len(letter_summaries)
            avg_orientation = sum(l.avg_orientation for l in letter_summaries) / len(letter_summaries)
            overall = (avg_letter_form + avg_size + avg_line_align + avg_orientation) / 4
            return round(overall, 2)
        return None
    
    def get_letter_count(self):
        """Get number of unique letters"""
        return self.letter_summaries.count()
    
    def get_total_repetitions(self):
        """Get total number of letter repetitions graded"""
        return self.letter_grades.count()
    
    def save(self, *args, **kwargs):
        if self.original_image and not self.file_size:
            self.file_size = self.original_image.size
        super().save(*args, **kwargs)


class LetterGrade(models.Model):
    """Store grades for individual letter instances (e.g., 5 repetitions of 'A')"""
    
    document = models.ForeignKey(
        ScannedDocument, 
        on_delete=models.CASCADE,
        related_name='letter_grades'
    )
    
    # Letter information
    letter = models.CharField(max_length=1, help_text="The letter (A-Z)")
    repetition_number = models.IntegerField(help_text="Which repetition (1-5)")
    # position_in_worksheet = models.IntegerField(help_text="Overall position in worksheet")
    
    # Grading criteria for this specific instance
    letter_form = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Quality of letter formation"
    )
    size = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Appropriateness of size"
    )
    line_align = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Baseline alignment"
    )
    orientation = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Letter slant/angle"
    )
    
    # Bounding box for this letter instance
    bbox_x = models.IntegerField(null=True, blank=True)
    bbox_y = models.IntegerField(null=True, blank=True)
    bbox_width = models.IntegerField(null=True, blank=True)
    bbox_height = models.IntegerField(null=True, blank=True)
    
    # Score for this instance
    instance_score = models.FloatField(null=True, blank=True)
    
    # Comments for this specific instance
    comments = models.TextField(blank=True)
    
    graded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        # ordering = ['position_in_worksheet']
        unique_together = ['document', 'letter', 'repetition_number']
    
    def __str__(self):
        return f"{self.letter} (rep {self.repetition_number}) - {self.document.title or f'Doc {self.document.id}'}"
    
    def get_instance_score(self):
        """Calculate score for this instance"""
        if self.instance_score is not None:
            return self.instance_score
        score = (self.letter_form + self.size + self.line_align) / 3
        return round(score, 2)
    
    def save(self, *args, **kwargs):
        if not self.document.is_graded:
            self.document.is_graded = True
            self.document.graded_at = timezone.now()
            self.document.save()
        super().save(*args, **kwargs)


class LetterSummary(models.Model):
    """Aggregated scores for each letter across all repetitions"""
    
    document = models.ForeignKey(
        ScannedDocument,
        on_delete=models.CASCADE,
        related_name='letter_summaries'
    )
    
    letter = models.CharField(max_length=1, help_text="The letter (A-Z)")
    
    # Average scores across all repetitions
    avg_letter_form = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    avg_size = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    avg_line_align = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    avg_orientation = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    
    # Overall average for this letter
    letter_average = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    
    # Number of repetitions graded
    repetition_count = models.IntegerField(default=5)
    
    # Best and worst scores
    best_score = models.FloatField(null=True, blank=True)
    worst_score = models.FloatField(null=True, blank=True)
    
    # Feedback for this letter
    comments = models.TextField(blank=True)
    
    class Meta:
        ordering = ['letter']
        unique_together = ['document', 'letter']
    
    def __str__(self):
        return f"Summary for '{self.letter}' - {self.document.title or f'Doc {self.document.id}'}"
    
    def get_letter_grade(self):
        """Convert to letter grade"""
        score = self.letter_average
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'


class WorksheetSummary(models.Model):
    """Overall summary for the entire worksheet"""
    
    document = models.OneToOneField(
        ScannedDocument,
        on_delete=models.CASCADE,
        related_name='worksheet_summary'
    )
    
    # Overall scores (aggregated from letter averages)
    overall_letter_form = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    overall_size = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    overall_line_align = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    overall_orientation = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    
    overall_score = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    
    # Statistics
    total_letters = models.IntegerField(help_text="Number of unique letters")
    total_repetitions = models.IntegerField(help_text="Total instances graded")
    
    # Grading metadata
    grading_method = models.CharField(
        max_length=20,
        choices=[
            ('manual', 'Manual Grading'),
            ('automatic', 'Automatic AI Grading'),
            ('hybrid', 'Hybrid (AI + Manual Review)')
        ],
        default='automatic'
    )
    graded_by = models.CharField(max_length=255, blank=True)
    
    # Overall feedback
    comments = models.TextField(blank=True)
    strengths = models.TextField(blank=True)
    areas_for_improvement = models.TextField(blank=True)
    
    graded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Summary for {self.document.title or f'Doc {self.document.id}'}"
    
    def get_letter_grade(self):
        """Convert to letter grade"""
        score = self.overall_score
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
