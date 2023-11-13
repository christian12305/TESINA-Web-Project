//This function updates the patient_record page when 
//  the user clicks a visit.
function updateResult(predResult, id, fecha) {

    const result = document.getElementById('result');
    const date = document.getElementById('date');

    //Change to readable value
    if(predResult == 1){
        predResult = "Positive"
    }else{
        predResult = "Negative"
    }

    // Clear the existing value and replace
    result.innerHTML = '';
    result.innerHTML = `<strong>Cardiac Disease:</strong> ${predResult}`;

    date.innerHTML = '';
    date.innerHTML = `<strong>${fecha}</strong>`

    // Get a reference to the anchor tag by its id
    var predictiveLink = document.getElementById("predictiveLink");

    //`{{url_for('views.predictive_analysis', visitId=${id})}}`
    //predictiveLink.href = `/predictive_analysis?visitId=${id}`
    predictiveLink.href = `/predictive_analysis`


    // Get a reference to the button element by its id
    var predictiveAnalysisButton = document.getElementById("predictiveButton");

    // Enable the button
    predictiveAnalysisButton.disabled = false;
    
}