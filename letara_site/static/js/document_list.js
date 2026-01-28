const uploadSection = document.getElementById('uploadSection');
const showUploadBtn = document.getElementById('showUploadBtn');
const showUploadBtnEmpty = document.getElementById('showUploadBtnEmpty');
const closeUploadBtn = document.getElementById('closeUploadBtn');
const cancelUploadBtn = document.getElementById('cancelUploadBtn');
const uploadPreview = document.getElementById('uploadPreview');
const fileInput = document.getElementById('id_image');
// const selectImgs = document.getElementById('btn-check-2-outlined');

// Show upload section
function showUploadSection(e) {
    e.preventDefault();
    uploadSection.style.display = 'block';
    uploadSection.scrollIntoView({ behavior: 'smooth' });
}

// Hide upload section
function hideUploadSection() {
    uploadSection.style.display = 'none';
    fileInput.value = '';
    uploadPreview.style.display = 'none';
}

// Event listeners
if (showUploadBtn) {
    showUploadBtn.addEventListener('click', showUploadSection);
}

if (showUploadBtnEmpty) {
    showUploadBtnEmpty.addEventListener('click', showUploadSection);
}

closeUploadBtn.addEventListener('click', hideUploadSection);
cancelUploadBtn.addEventListener('click', hideUploadSection);

// File preview
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreview.src = e.target.result;
            uploadPreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

document.addEventListener("DOMContentLoaded", () => {
    const checkbox = document.getElementById("btn-check-2-outlined");

    checkbox.addEventListener("change", () => {
        console.log("test");
        console.log("checked:", checkbox.checked);
    });
});
