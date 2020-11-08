
function change_label() {
    $.ajax({
        url: 'change_label',
        type: 'POST',
        success: function (response) {
            // Perform operation on the return value
            $('#acc').html(response[0]);
            $("#change_btn").removeClass("btn-success btn-danger btn-warning").addClass("btn-" + response[1]);

            // console.log(response);
            // console.log(response[1]);
        }
    });
}

$(document).ready(function () {
    setInterval(change_label, 1000);
});