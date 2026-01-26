from django import forms
from .models import ScannedDocument

class DocumentUploadForm(forms.ModelForm):
    class Meta:
        model = ScannedDocument
        fields = ['title', 'original_image', 'upload_method']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Worksheet title (optional)'
            }),
            'original_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'upload_method': forms.HiddenInput()
        }
        labels = {
            'original_image': 'Worksheet Image'
        }

class CameraCaptureForm(forms.ModelForm):
    captured_image = forms.CharField(widget=forms.HiddenInput(), required=False)
    
    class Meta:
        model = ScannedDocument
        fields = ['title', 'upload_method']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Worksheet title (optional)'
            }),
            'upload_method': forms.HiddenInput()
        }

class CropForm(forms.Form):
    """Form for handling crop coordinates"""
    x1 = forms.FloatField()
    y1 = forms.FloatField()
    x2 = forms.FloatField()
    y2 = forms.FloatField()
