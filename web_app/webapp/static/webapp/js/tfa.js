$(document).ready(function() {
              $('.navbar').removeClass('fixed-top');
              $(".refresh-btn").click(function(){
                    var tfa_season_select = $("#tfa_season_select").children("option:selected").val();
                    var tfa_place_select = $("#tfa_place_select").children("option:selected").val();
                    data = {'tfa_season_select' : tfa_season_select, 'tfa_place_select' : tfa_place_select};
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



