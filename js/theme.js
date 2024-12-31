//---------------------------
// Light or Dark Theme
//---------------------------
var theme = localStorage.getItem('theme');
var body  = document.body;
if (theme) {
  body.classList.add(theme);
} else {
  body.classList.add("light");
  localStorage.setItem('theme', 'light');
  theme = localStorage.getItem('theme');
  document.getElementById("nav_theme").style.backgroundImage = "url('/images/icons/light-theme.png')";
}

function switchTheme() {
  var sheet     = document.getElementById("main-style-sheet");
  var btnTheme  = document.getElementById("nav_theme");
  var metaTheme = document.querySelector('meta[name="theme-color"]');

  if(theme == "light") {
    localStorage.setItem('theme', 'dark');
    theme = localStorage.getItem('theme');
    body.classList.remove("light");
    body.classList.add("dark");
    btnTheme.style.backgroundImage = "url('/images/icons/dark-theme.png')";
    metaTheme.setAttribute("content", "#000000");
  } else {
    localStorage.setItem('theme', 'light');
    theme = localStorage.getItem('theme');
    body.classList.remove("dark");
    body.classList.add("light");
    btnTheme.style.backgroundImage = "url('/images/icons/light-theme.png')";
    metaTheme.setAttribute("content", "#000000");
  }
}