from django.db import models
from django.utils import timezone

# Create your models here.

class ScannedDocument(models.Model):
    title = models.CharField(max_length=255, blank=True)
    image = models.ImageField(upload_to='scanned_documents%Y%m%d')
    upload_method = models.CharField(
        max_length=10,
        choices=[('camera', 'Camera Capture'), ('upload', 'File Upload')],
        default='upload'
    )
    uploaded_at = models.DateTimeField(default=timezone.now)
    file_size = models.IntegerField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return self.title or f"Document {self.id}"
    
    def save(self, *args, **kwargs):
        if self.image and not self.file_size:
            self.file_size = self.image.size
        super().save(*args, **kwargs)
