"""
Workaround for castalign graph visualization issues with GraphViz.

The built-in g.visualise() uses PDF format which has DLL dependency issues
on Windows. This provides an alternative using PNG format.

Usage in Jupyter:
    from graph_viz_fix import visualise_graph_png
    visualise_graph_png(g)  # Opens PNG in browser/viewer
    visualise_graph_png(g, filename='my_graph.png')  # Saves to file
"""

import sys
import tempfile
import os
from pathlib import Path

# Add project root to path for local_config import
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from local_config import CASTALIGN_ROOT
except ImportError:
    raise ImportError(
        "local_config.py not found.\n"
        "Copy local_config.example.py to local_config.py and fill in your paths:\n"
        "    cp local_config.example.py local_config.py"
    )

# Add castalign to path
if CASTALIGN_ROOT not in sys.path:
    sys.path.insert(0, CASTALIGN_ROOT)


def visualise_graph_png(graph, filename=None, nearby=None, format='svg', view=True):
    """
    Visualize a LineStuffUp graph using SVG format instead of PDF.

    This is a drop-in replacement for g.visualise() that works around
    GraphViz DLL dependency issues on Windows. SVG is vector format (like PDF)
    so it preserves text layout and quality, but doesn't need Pango plugin.

    Parameters:
    -----------
    graph : castalign.Graph
        The graph to visualize
    filename : str, optional
        Output filename (without extension). If None, uses temporary file.
    nearby : str, optional
        Only show edges connected to this node
    format : str, default='svg'
        Output format. Recommended: 'svg' (vector, clean text).
        Alternative: 'png' (raster, may have text overflow).
        Available: svg, png, jpg, webp, etc.
    view : bool, default=True
        Whether to open the file after rendering

    Returns:
    --------
    str : Path to the rendered file
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError("Please install graphviz package to visualise")

    # Create temporary or specified filename
    fn = filename
    cleanup_temp = False
    if fn is None:
        fn = tempfile.mktemp(suffix=f'.{format}')
        cleanup_temp = True

    # Remove extension if provided
    fn = str(Path(fn).with_suffix(''))

    # Create Digraph with specified format
    g = graphviz.Digraph(graph.name, filename=fn, format=format)

    # Find all nodes that have an Identity edge and choose one as the 'base' node
    ur_node = {}
    ur_node_names = {}
    for e1 in graph.edges.keys():
        found = False
        ident_edges = [e2 for e2 in graph.edges[e1] if graph.edges[e1][e2].__class__.__name__ == "Identity"]
        for e2 in ident_edges:
            if e2 in ur_node.keys() and ur_node[e2] == e2:
                ur_node[e1] = e2
                ur_node_names[e2] += "\n" + e1
                found = True
                break
        if not found:
            ur_node[e1] = e1
            ur_node_names[e1] = e1

    # Add edges
    ur_nodes_used = set()
    for e1 in graph.edges.keys():
        for e2 in graph.edges[e1].keys():
            if nearby is not None and e1 != nearby and e2 != nearby:
                continue
            if e1 in graph.edges[e2].keys() and graph.edges[e1][e2].__class__.__name__ == graph.edges[e2][e1].__class__.__name__:
                if e1 > e2 and graph.edges[e1][e2].__class__.__name__ != "Identity":
                    g.edge(ur_node[e1], ur_node[e2], label=graph.edges[e1][e2].__class__.__name__, dir="both")
                    ur_nodes_used.add(ur_node[e1])
                    ur_nodes_used.add(ur_node[e2])
            else:
                g.edge(ur_node[e1], ur_node[e2], label=graph.edges[e1][e2].__class__.__name__)
                ur_nodes_used.add(ur_node[e1])
                ur_nodes_used.add(ur_node[e2])

    # Add nodes
    for n in sorted(ur_nodes_used):
        g.node(n, label=ur_node_names[n])

    # Render and optionally view
    if view:
        output_path = g.render(cleanup=cleanup_temp, view=True)
    else:
        output_path = g.render(cleanup=False, view=False)

    print(f"Graph visualization saved to: {output_path}")
    return output_path


def visualise_graph_inline(graph, nearby=None, format='svg', dpi=150, save_path=None):
    """
    Visualize a LineStuffUp graph inline in Jupyter notebook.

    Parameters:
    -----------
    graph : castalign.Graph
        The graph to visualize
    nearby : str, optional
        Only show edges connected to this node
    format : str, default='svg'
        Output format ('svg' or 'png'). SVG is vector and scales better.
    dpi : int, default=150
        DPI for PNG format (ignored for SVG)
    save_path : str, optional
        Directory to save the file. If None, saves to graph's directory.

    Returns:
    --------
    IPython.display object
    """
    try:
        import graphviz
        from IPython.display import display, SVG, Image
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")

    # Create Digraph
    g = graphviz.Digraph(graph.name, format=format)

    if format == 'png':
        g.attr(dpi=str(dpi))

    # Same logic as above for building the graph
    ur_node = {}
    ur_node_names = {}
    for e1 in graph.edges.keys():
        found = False
        ident_edges = [e2 for e2 in graph.edges[e1] if graph.edges[e1][e2].__class__.__name__ == "Identity"]
        for e2 in ident_edges:
            if e2 in ur_node.keys() and ur_node[e2] == e2:
                ur_node[e1] = e2
                ur_node_names[e2] += "\n" + e1
                found = True
                break
        if not found:
            ur_node[e1] = e1
            ur_node_names[e1] = e1

    ur_nodes_used = set()
    for e1 in graph.edges.keys():
        for e2 in graph.edges[e1].keys():
            if nearby is not None and e1 != nearby and e2 != nearby:
                continue
            if e1 in graph.edges[e2].keys() and graph.edges[e1][e2].__class__.__name__ == graph.edges[e2][e1].__class__.__name__:
                if e1 > e2 and graph.edges[e1][e2].__class__.__name__ != "Identity":
                    g.edge(ur_node[e1], ur_node[e2], label=graph.edges[e1][e2].__class__.__name__, dir="both")
                    ur_nodes_used.add(ur_node[e1])
                    ur_nodes_used.add(ur_node[e2])
            else:
                g.edge(ur_node[e1], ur_node[e2], label=graph.edges[e1][e2].__class__.__name__)
                ur_nodes_used.add(ur_node[e1])
                ur_nodes_used.add(ur_node[e2])

    for n in sorted(ur_nodes_used):
        g.node(n, label=ur_node_names[n])

    # Determine save location
    if save_path is None:
        if hasattr(graph, 'filename') and graph.filename:
            from pathlib import Path
            save_path = str(Path(graph.filename).parent / f"{graph.name}_structure")
        else:
            save_path = f"{graph.name}_structure"

    # Save to file
    output_file = g.render(filename=save_path, cleanup=True)
    print(f"Graph saved to: {output_file}")

    # Render to bytes and display inline
    if format == 'svg':
        svg_data = g.pipe(format='svg')
        return SVG(svg_data)
    else:  # png
        png_data = g.pipe(format='png')
        return Image(png_data)


# Monkey-patch the Graph class (optional - use at your own risk)
def patch_castalign_visualise():
    """
    Monkey-patch castalign's Graph.visualise() to use SVG instead of PDF.

    SVG is vector format (like PDF) so text and layout stay clean, but it
    doesn't require the broken Pango plugin.

    Call this once at the start of your notebook:
        from graph_viz_fix import patch_castalign_visualise
        patch_castalign_visualise()

    Then g.visualise() will work without errors.
    """
    import castalign

    # Save original method
    if not hasattr(castalign.Graph, '_original_visualise'):
        castalign.Graph._original_visualise = castalign.Graph.visualise

    # Replace with inline SVG version
    def visualise_svg(self, filename=None, nearby=None):
        return visualise_graph_inline(self, nearby=nearby, format='svg')

    castalign.Graph.visualise = visualise_svg
    print("✓ Patched Graph.visualise() to use SVG format (vector, preserves text layout)")
