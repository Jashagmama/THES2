from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.files.base import ContentFile
from django.views.generic import ListView, DetailView
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_http_methods
import base64
import json
from .models import ScannedDocument
from .forms import DocumentUploadForm, CameraCaptureForm, CropForm

from . import fullPipe

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
            document.original_image = image_data
            document.upload_method = 'camera'
            document.save()
            
            messages.success(request, 'Document captured! Now adjust the boundaries.')
            return redirect('adjust_boundaries', pk=document.pk)
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
        
        messages.success(request, 'Document uploaded! Now adjust the boundaries.')
        return redirect('adjust_boundaries', pk=document.pk)
    else:
        messages.error(request, 'Please select a valid image file.')
    
    return redirect('document_scanner')

def adjust_boundaries(request, pk):
    """View for adjusting document boundaries"""
    document = get_object_or_404(ScannedDocument, pk=pk)
    
    context = {
        'document': document,
    }
    return render(request, 'letara/adjust_boundaries.html', context)

@require_POST
def apply_crop(request, pk):
    """Apply crop to document based on user-defined boundaries"""
    document = get_object_or_404(ScannedDocument, pk=pk)
    
    try:
        data = json.loads(request.body)
        x1 = float(data.get('x1'))
        y1 = float(data.get('y1'))
        x2 = float(data.get('x2'))
        y2 = float(data.get('y2'))
        
        success = document.apply_crop(x1, y1, x2, y2)
        
        if success:
            return JsonResponse({
                'success': True,
                'message': 'Boundaries applied successfully!',
                'redirect_url': f'/document/{document.pk}/'
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'Failed to apply crop.'
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        })

@require_POST
def skip_crop(request, pk):
    """Skip cropping and use original image"""
    document = get_object_or_404(ScannedDocument, pk=pk)
    document.is_cropped = False
    document.save()
    
    return JsonResponse({
        'success': True,
        'message': 'Using original image',
        'redirect_url': f'/document/{document.pk}/'
    })

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

@require_http_methods(["POST", "DELETE"])
def delete_document(request, pk):
    """Delete a document"""
    document = get_object_or_404(ScannedDocument, pk=pk)
    
    try:
        # Delete the image files
        if document.original_image:
            document.original_image.delete()
        if document.cropped_image:
            document.cropped_image.delete()
        
        # Delete the document record
        document.delete()
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'message': 'Document deleted successfully!'
            })
        else:
            messages.success(request, 'Document deleted successfully!')
            return redirect('document_list')
            
    except Exception as e:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': False,
                'message': f'Error deleting document: {str(e)}'
            }, status=500)
        else:
            messages.error(request, f'Error deleting document: {str(e)}')
            return redirect('document_detail', pk=pk)
