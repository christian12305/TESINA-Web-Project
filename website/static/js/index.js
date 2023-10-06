function updateResult(predResult ,id) {

    const result = document.getElementById('result');
    // Clear the existing value
    result.innerHTML = '';
    result.innerHTML = `<strong>Cardiac Disease:</strong> ${predResult}`;

    // Get a reference to the anchor tag by its id
    var predictiveLink = document.getElementById("predictiveLink");

    //`{{url_for('views.predictive_analysis', visitId=${id})}}`
    predictiveLink.href = `/predictive_analysis?visitId=${id}`

    // Get a reference to the button element by its id
    var predictiveAnalysisButton = document.getElementById("predictiveButton");

    // Enable the button
    predictiveAnalysisButton.disabled = false;

    
}