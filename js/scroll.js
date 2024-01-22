function sidebarHandler() {
  var sidebar_lis_html = document.getElementById("sidebar_tracker").getElementsByTagName("a");
  var sidebar_lis      = Array.from(sidebar_lis_html);
  sidebar_lis[0].classList.add("sidebar_active");

  var heads = document.getElementsByTagName("H3");
  for (var i = heads.length - 1; i >= 0; i--) {
    var box = heads[i].getBoundingClientRect();
    if (box.top < 10 && box.top > -20) {
      var id  = heads[i].innerHTML;
      var li  = "";
      for (var j = sidebar_lis.length - 1; j >= 0; j--) {
        var list_value = sidebar_lis[j].innerHTML;
        sidebar_lis[j].classList.remove("sidebar_active");
        if (list_value == id) {
          li = sidebar_lis[j]
        }
      }
      if (li != "") {
        li.classList.add("sidebar_active");
      }
    }
  }
}