var codes = [

"code-tcl-std-io", 
"code-tcl-math", 
"code-tcl-loops",
"code-tcl-conditions",
"code-tcl-logical",
"code-tcl-switch",
"code-tcl-lists",
"code-tcl-dicts",
"code-tcl-strings",
"code-tcl-regex",
"code-tcl-procs",
"code-tcl-files",
"code-tcl-utils",
"code-tcl-sources"

];

for (i=0; i<codes.length; i++) {
	document.getElementById(codes[i]).style.display = "none";
}

function toggle_div (div) { 
	var block_div = div.id;
	var code_div  = block_div.replace("block", "code");
	
	var block_element = document.getElementById(block_div);
	var code_element = document.getElementById(code_div);
	var status = code_element.style.display;

	if (status=="none") {
		block_element.style.backgroundImage = "url('/images/up.png')";
		code_element.style.display = "block";
	} else {
		block_element.style.backgroundImage = "url('/images/down.png')";
		code_element.style.display = "none";
	}
}

function toggle_code(toggleBtn) {
	if (toggleBtn.checked) {
		for (i=0; i<codes.length; i++) {
			document.getElementById(codes[i]).style.display = "block";
		}
	} else {
		for (i=0; i<codes.length; i++) {
			document.getElementById(codes[i]).style.display = "none";
		}
	}
}