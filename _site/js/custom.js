var items = document.querySelectorAll(".my-timeline li");
 
function isElementInViewport(el) {
  var rect = el.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}
 
function callbackFunc() {
  for (var i = 0; i < items.length; i++) {
    if (isElementInViewport(items[i])) {
      items[i].classList.add("in-view");
    }
  }
}
 
window.addEventListener("load", callbackFunc);
window.addEventListener("scroll", callbackFunc);

var toggle = 0;
function showComments() {
  var div_master = document.getElementById("comment_count");
  var div_disqus = document.getElementById('disqus_thread');

  if (toggle == 0) {
    div_disqus.style.display = "block";
    toggle = 1;
  } else {
    temp = comment_count;
    div_disqus.style.display = "none";
    toggle = 0;
  }
}

nav_on = 0;
function top_navigation() {
    var x = document.getElementById("top_navigator");
    if (nav_on == 0) {
        //x.className += " responsive";
        nav_on = 1;
    } else {
        x.className = "topnav";
        nav_on = 0;
    }
}

nav_on_splash = 0;
function top_splash_navigation() {
    var x = document.getElementById("top_navigator");
    if (nav_on_splash == 0) {
        //x.className += " responsive";
        nav_on_splash = 1;
    } else {
        x.className = "splashnav";
        nav_on_splash = 0;
    }
}

function windowScrollHandler() {
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        document.getElementById("btnScrollTop").style.display = "block";
    } else {
        document.getElementById("btnScrollTop").style.display = "none";
    }
}

function handleSideBarLinks(id) {
  var elements = document.getElementsByClassName("sidebar_links");
  for (var i = 0; i < elements.length; i++) {
    elements[i].classList.remove("sidebar_active");
  }
  document.getElementById(id).classList.add("sidebar_active");
}

function topScroller() {
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
}

function boxHandler(btnId) {
  var boxId  = btnId.replace("btn", "box");
  var status = document.getElementById(boxId).style.display;
  if (status == "block"){
    document.getElementById(boxId).style.display = "none";
  } else {
    document.getElementById(boxId).style.display = "block";
  }
}

function showShareBox(id) {
  var box = document.getElementById(id.replace("fab", "box"))
  if (box.style.display == "block") {
    box.style.display = "none";
    document.getElementById(id).style.backgroundColor = "#bc3b2f";
  } else {
    box.style.display = "block";
    document.getElementById(id).style.backgroundColor = "#fc0";
  }
}

function closeSidebar(id) {
  document.getElementById(id).style.display = "none";
}

function showSidebar(id) {
  if(document.getElementById(id).style.display == "block") {
    document.getElementById(id).style.display = "none";
  } else {
    document.getElementById(id).style.display = "block";
  }
}

function showTabBox(id) {
  var tab = document.getElementById(id);
  var box = document.getElementById(id.replace("tab", "box"));

  var boxes = document.getElementsByClassName("blog-category-box");
  var tabs = document.getElementById("category-tab").getElementsByTagName("li");

  for (var i = 0; i < boxes.length; i++) {
    boxes[i].style.display = "none";
  }

  for (var i = 0; i < tabs.length; i++) {
    tabs[i].style.backgroundColor = "white";
    tabs[i].style.fontWeight = "100";
  }

  box.style.display = "block";
  tab.style.backgroundColor = "#ffea82";
  tab.style.fontWeight = "bold";
}

var modalBool = 0;
function showHideModal(imgId) {
  // Get the modal
  var modal = document.getElementById('creative_modal');

  // Get the image and insert it inside the modal - use its "alt" text as a caption
  var img         = document.getElementById(imgId);
  var modalImg    = document.getElementById("modal_image");

  if (modalBool == 0) {
      modal.style.display = "block";
      modalImg.src = img.src;
      modalBool = 1;
  } else {
    modal.style.display = "none";
    modalBool = 0;
  }

  // Get the <span> element that closes the modal
  var span = document.getElementsByClassName("close")[0];

  // When the user clicks on <span> (x), close the modal
  span.onclick = function() { 
      modal.style.display = "none";
      modalBool = 0;
  }
}

function downloadImage() {
  var modalImg = document.getElementById("modal_image");
  var link = document.createElement('a');
  link.href = modalImg.src;
  link.download = modalImg.alt;
  document.body.appendChild(link);
  link.click();
  link.style.display="none";
}

function closeSideNav() {
  document.getElementById("awesomeSideNav").style.width = "0";
  document.getElementById("awesomeSideNav").style.padding = "0";
}

function openSideNav() {
  document.getElementById("awesomeSideNav").style.width = "250px";
  document.getElementById("awesomeSideNav").style.padding = "15px";
}

var slideIndex = 1;
showDivs(slideIndex);

function plusDivs(n) {
  showDivs(slideIndex += n);
}

function showDivs(n) {
  var i;
  var x = document.getElementsByClassName("moocs_slides");
  if (n > x.length) {slideIndex = 1} 
  if (n < 1) {slideIndex = x.length} ;
  for (i = 0; i < x.length; i++) {
    x[i].style.display = "none"; 
  }
  x[slideIndex-1].style.display = "block"; 
}