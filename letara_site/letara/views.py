from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.files.base import ContentFile
from django.views.generic import ListView, DetailView
import base64
from .models import ScannedDocument
from .forms import DocumentUploadForm, CameraCaptureForm

# Create your views here.

def document_scanner(request):
    """Main view for document scanning with camera or upload options"""
    upload_form = DocumentUploadForm()
    camera_form = CameraCaptureForm()
    
    if request.method == 'POST':
        upload_method = request.POST.get('upload_method')
        
        if upload_method == 'camera':
            return handle_camera_capture(request)
        elif upload_method == 'upload':
            return handle_file_upload(request)
    
    context = {
        'upload_form': upload_form,
        'camera_form': camera_form,
    }
    return render(request, 'letara/document_scanner.html', context)

def handle_camera_capture(request):
    """Handle camera capture submission"""
    form = CameraCaptureForm(request.POST)
    
    if form.is_valid():
        captured_image_data = request.POST.get('captured_image')
        
        if captured_image_data:
            # Remove the data URL prefix
            format, imgstr = captured_image_data.split(';base64,')
            ext = format.split('/')[-1]
            
            # Decode base64 image
            image_data = ContentFile(base64.b64decode(imgstr), name=f'camera_capture.{ext}')
            
            # Save the document
            document = form.save(commit=False)
            document.image = image_data
            document.upload_method = 'camera'
            document.save()
            
            messages.success(request, 'Document captured successfully!')
            return redirect('document_detail', pk=document.pk)
        else:
            messages.error(request, 'No image was captured.')
    else:
        messages.error(request, 'Please fill in the required fields.')
    
    return redirect('document_scanner')

def handle_file_upload(request):
    """Handle file upload submission"""
    form = DocumentUploadForm(request.POST, request.FILES)
    
    if form.is_valid():
        document = form.save(commit=False)
        document.upload_method = 'upload'
        document.save()
        
        messages.success(request, 'Document uploaded successfully!')
        return redirect('document_detail', pk=document.pk)
    else:
        messages.error(request, 'Please select a valid image file.')
    
    return redirect('document_scanner')

class DocumentListView(ListView):
    """View to list all scanned documents"""
    model = ScannedDocument
    template_name = 'letara/document_list.html'
    context_object_name = 'documents'
    paginate_by = 12

class DocumentDetailView(DetailView):
    """View to display a single scanned document"""
    model = ScannedDocument
    template_name = 'letara/document_detail.html'
    context_object_name = 'document'
