function selectPatient(patientId){
  //window.location.href = "{{url_for('views.patient_record', patientId)}}";


  fetch("{{url_for('views.patient_record')}}", {
    method: "POST",
    body: JSON.stringify({patientId: patientId}),
  }).then((_res) => {
    //window.location.href = "{{url_for('views.patient_record', patientId)}}";
  })
}

  fetch('/patient_record' + new URLSearchParams({
    patientId: patientId,
  }))