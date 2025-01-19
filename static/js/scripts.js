$(document).ready(function () {
    $('#uploadForm').on('submit', function (e) {
        e.preventDefault();
        let formData = new FormData();
        let fileInput = $('#imageInput')[0];

        // Check if a file is selected
        if (fileInput.files.length === 0) {
            alert("Please select an image to upload.");
            return;
        }

        formData.append('image', fileInput.files[0]);

        // Show loading indicator
        $('#result').hide();
        $('#loading').show();

        // Send the image to the Flask API
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                // Hide loading indicator
                $('#loading').hide();

                // Display the prediction result
                if (response.prediction && response.confidence) {
                    $('#predictionResult').text(response.prediction);
                    $('#confidenceResult').text((response.confidence * 100).toFixed(2) + '%');
                    $('#result').show();
                } else {
                    alert("Invalid response from the server.");
                }
            },
            error: function (xhr, status, error) {
                // Hide loading indicator
                $('#loading').hide();

                // Display the error message
                alert("Error: " + xhr.responseText);
                console.error(xhr);
            }
        });
    });
});