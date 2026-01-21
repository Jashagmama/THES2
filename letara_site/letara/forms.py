from django import forms
from .models import ScannedDocument

class DocumentUploadForm(forms.ModelForm):
    class Meta:
        model = ScannedDocument
        fields = ['title', 'image', 'upload_method']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Document title (optional)'
            }),
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'upload_method': forms.HiddenInput()
        }

class CameraCaptureForm(forms.ModelForm):
    captured_image = forms.CharField(widget=forms.HiddenInput(), required=False)
    
    class Meta:
        model = ScannedDocument
        fields = ['title', 'upload_method']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Document title (optional)'
            }),
            'upload_method': forms.HiddenInput()
        }
