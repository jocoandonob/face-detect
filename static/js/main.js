document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('imageInput');
    const uploadButton = document.getElementById('uploadButton');
    const resultArea = document.getElementById('resultArea');
    const resultImage = document.getElementById('resultImage');
    const detectionDetails = document.getElementById('detectionDetails');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');
    
    // Preview the selected image
    fileInput.addEventListener('change', function() {
        const file = fileInput.files[0];
        
        if (file) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                resultImage.src = e.target.result;
                resultImage.style.display = 'block';
                resultArea.style.display = 'block';
                detectionDetails.innerHTML = '';
                errorMessage.style.display = 'none';
            };
            
            reader.readAsDataURL(file);
        }
    });
    
    // Handle image upload and face detection
    uploadButton.addEventListener('click', async function() {
        if (!fileInput.files.length) {
            errorMessage.textContent = 'Please select an image first.';
            errorMessage.style.display = 'block';
            return;
        }
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading spinner
        loadingSpinner.style.display = 'block';
        errorMessage.style.display = 'none';
        
        try {
            // Process the image to get detections
            const response = await fetch('/api/process_image', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Display the processed image
            if (data.processed_image) {
                resultImage.src = `data:image/jpeg;base64,${data.processed_image}`;
                resultImage.style.display = 'block';
            }
            
            // Get detailed detections
            const detailResponse = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });
            
            if (!detailResponse.ok) {
                throw new Error(`Server responded with ${detailResponse.status}: ${detailResponse.statusText}`);
            }
            
            const detailData = await detailResponse.json();
            displayDetectionSummary(detailData);
            
        } catch (error) {
            console.error('Error:', error);
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.style.display = 'block';
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });
    
    // Display detection results summary
    function displayDetectionSummary(data) {
        const faceCount = data.faces ? data.faces.length : 0;
        const eyeCount = data.eyes ? data.eyes.length : 0;
        const lipCount = data.lips ? data.lips.length : 0;
        
        let summaryHTML = `
            <div class="alert alert-info mt-3">
                <h5>Detection Summary:</h5>
                <ul>
                    <li><strong>Faces detected:</strong> ${faceCount}</li>
                    <li><strong>Eyes detected:</strong> ${eyeCount}</li>
                    <li><strong>Lips detected:</strong> ${lipCount}</li>
                </ul>
            </div>
        `;
        
        if (faceCount > 0) {
            summaryHTML += `<div class="accordion mt-3" id="detectionAccordion">`;
            
            // Face details
            summaryHTML += createAccordionItem('faceDetails', 'Face Detections', 
                createDetailsList(data.faces, face => {
                    return `Face at position (${face.box[0]}, ${face.box[1]}) with width ${face.box[2]} and height ${face.box[3]}. 
                           Confidence: ${(face.confidence * 100).toFixed(2)}%`;
                })
            );
            
            // Eye details
            if (eyeCount > 0) {
                summaryHTML += createAccordionItem('eyeDetails', 'Eye Detections', 
                    createDetailsList(data.eyes, eye => {
                        const side = eye.side ? `${eye.side} eye` : 'Eye';
                        const confidence = eye.confidence ? `Confidence: ${(eye.confidence * 100).toFixed(2)}%` : '';
                        const estimated = eye.estimated ? ' (estimated from facial landmarks)' : '';
                        return `${side} at position (${eye.box[0]}, ${eye.box[1]}) with width ${eye.box[2]} and height ${eye.box[3]}. ${confidence}${estimated}`;
                    })
                );
            }
            
            // Lip details
            if (lipCount > 0) {
                summaryHTML += createAccordionItem('lipDetails', 'Lip Detections', 
                    createDetailsList(data.lips, lip => {
                        return `Lips at position (${lip.box[0]}, ${lip.box[1]}) with width ${lip.box[2]} and height ${lip.box[3]}`;
                    })
                );
            }
            
            summaryHTML += `</div>`;
        }
        
        detectionDetails.innerHTML = summaryHTML;
    }
    
    // Create accordion item for detection details
    function createAccordionItem(id, title, content) {
        return `
            <div class="accordion-item">
                <h2 class="accordion-header" id="heading${id}">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                            data-bs-target="#collapse${id}" aria-expanded="false" aria-controls="collapse${id}">
                        ${title}
                    </button>
                </h2>
                <div id="collapse${id}" class="accordion-collapse collapse" 
                     aria-labelledby="heading${id}" data-bs-parent="#detectionAccordion">
                    <div class="accordion-body">
                        ${content}
                    </div>
                </div>
            </div>
        `;
    }
    
    // Create details list for detections
    function createDetailsList(items, formatter) {
        if (!items || items.length === 0) {
            return '<p>No detections found.</p>';
        }
        
        let listHTML = '<ul class="list-group">';
        items.forEach((item, index) => {
            listHTML += `<li class="list-group-item">${formatter(item)}</li>`;
        });
        listHTML += '</ul>';
        
        return listHTML;
    }
});
