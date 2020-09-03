$(document).ready(function() {
              $('.navbar').removeClass('fixed-top');
              var trs_country_select = $("#trs_country_select").children("option:selected").val();

              if(trs_country_select == "Germany") {
                        $('.div-cities').removeClass('display-od');
                    }
              else {
                        $('.div-cities').addClass('display-od');
                    }
              $("#trs_country_select").on('change', function(){
                    var trs_country_select = $("#trs_country_select").children("option:selected").val();
                    if(trs_country_select == "Germany") {
                        $('.div-cities').removeClass('display-od');
                    }
                    else {
                        $('.div-cities').addClass('display-od');
                    }
              })
});