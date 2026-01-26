from django.contrib import admin
from .models import ScannedDocument

@admin.register(ScannedDocument)
class ScannedDocumentAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'upload_method', 'is_cropped', 'uploaded_at', 'file_size']
    list_filter = ['upload_method', 'is_cropped', 'uploaded_at']
    search_fields = ['title']
    readonly_fields = ['uploaded_at', 'file_size', 'crop_coordinates']
    
    fieldsets = (
        ('Document Information', {
            'fields': ('title', 'original_image', 'cropped_image', 'upload_method')
        }),
        ('Crop Information', {
            'fields': ('is_cropped', 'crop_coordinates'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('uploaded_at', 'file_size'),
            'classes': ('collapse',)
        }),
    )
