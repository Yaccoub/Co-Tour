$(document).ready(function() {
        // Transition effect for navbar 
        $(window).scroll(function() {
          // checks if window is scrolled more than 250px, adds/removes solid class
          if($(this).scrollTop() > 250) { 
              $('.navbar').addClass('active');
          } else {
              $('.navbar').removeClass('active');
          }
        });
});
