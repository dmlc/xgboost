function auto_index(module) {
  $(document).ready(function () {
    // find all classes or functions
    var div_query = "div[class='section'][id='module-" + module + "']";
    var class_query = div_query + " dl[class='class'] > dt";
    var func_query = div_query + " dl[class='function'] > dt";
    var targets = $(class_query + ',' + func_query);

    var li_node = $("li a[href='#module-" + module + "']").parent();
    var html = "<ul>";

    for (var i = 0; i < targets.length; ++i) {
      var id = $(targets[i]).attr('id');
      // remove 'mxnet.' prefix to make menus shorter
      var id_simple = id.replace(/^mxnet\./, '');
      html += "<li><a class='reference internal' href='#";
      html += id;
      html += "'>" + id_simple + "</a></li>";
    }

    html += "</ul>";
    li_node.append(html);
  });
}

