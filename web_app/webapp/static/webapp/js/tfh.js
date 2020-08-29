$(document).ready(function() {
              $('.navbar').removeClass('fixed-top');
              $("#month_picker").MonthPicker({
                ShowIcon: false,
                StartYear: 2020
              });
              $(".forecast-btn").click(function(){
                    var tfh_month_select = $("#tfh_month_select").children("option:selected").val();
                    data = {'tfh_month_select' : tfh_month_select };
                    $.ajax.get({
                         url: '/tourist_hotspot_forecast/',
                         method : 'GET',
                         data: data,
                         beforeSend: function() {
                         },
                         success: function(){
                         }
                    });
              });
              $(".historical-btn").click(function(){
                    var month_picker = $("#month_picker").MonthPicker('GetSelectedMonthYear');
                    data = {'month_picker' : month_picker };
                    $.ajax.get({
                         url: '/tourist_hotspot_forecast/',
                         method : 'GET',
                         data: data,
                         beforeSend: function() {
                         },
                         success: function(){
                         }
                    });
              });
});



