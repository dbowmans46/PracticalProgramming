& "C:\Software\Graphviz\bin\dot.exe" -Tpng .\decision_tree.dot -o decision_tree.png

& "C:\Software\Graphviz\bin\dot.exe" -Tsvg .\decision_tree.dot -o decision_tree.svg

# SVG can be read in HTML, so can also store as an HTML file
& "C:\Software\Graphviz\bin\dot.exe" -Tsvg .\decision_tree.dot -o decision_tree.html

& "C:\Software\Graphviz\bin\dot.exe" -Tpdf .\decision_tree.dot -o decision_tree.pdf

& "C:\Software\Graphviz\bin\dot.exe" -Tps .\decision_tree.dot -o decision_tree.ps