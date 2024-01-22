function selectBoxHandler(type, id, base_path, img_id) {
	var select_box = document.getElementById(id);
	var selected   = select_box.options[select_box.selectedIndex].id;
	var img_box    = document.getElementById(img_id);

	if (type == "target_vs_") {
		var img_path = base_path + type + selected + ".png";
	} else {
		var img_path = base_path + type + "_" + selected + ".png";
	}
	img_box.src = img_path;
}