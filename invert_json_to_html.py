import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from pyvis.network import Network
    _PYVIS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    Network = None  # type: ignore[assignment]
    _PYVIS_IMPORT_ERROR = exc

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_EXTRACT_DIR = PROJECT_ROOT / "output_extract"
DEFAULT_HTML_NAME = "knowledge_graph.html"

COLOR_MAP = {
    "chemical": "#4f81bd",
    "chemical_entity": "#4f81bd",
    "sample": "#4f81bd",
    "testing": "#f79646",
    "test": "#f79646",
    "experimental_condition": "#ffc000",
    "synthesis": "#c0504d",
    "process": "#c0504d",
    "characterization": "#9bbb59",
    "characterization_data": "#d8e4bc",
    "figure": "#8064a2",
    "image": "#8064a2",
}
DEFAULT_COLOR = "#a6a6a6"


def _format_value(value: Any) -> str:
    if isinstance(value, dict):
        combined = f"{value.get('value', '')} {value.get('unit', '')}".strip()
        return combined if combined else str(value)
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value)


def generate_info_html(node: Dict[str, Any], color: str) -> str:
    title = node.get("name", node["id"])
    html = (
        f"<h3 style='margin-top:0; color:#333; border-bottom: 2px solid {color};"
        f" padding-bottom: 10px;'>{title}</h3>"
    )
    html += "<table style='width:100%; border-collapse: collapse; font-size: 13px;'>"
    html += (
        "<tr><td style='padding:4px; font-weight:bold; color:#666;'>Type</td>"
        f"<td style='padding:4px;'>{node.get('type', 'N/A')}</td></tr>"
    )
    html += (
        "<tr><td style='padding:4px; font-weight:bold; color:#666;'>ID</td>"
        f"<td style='padding:4px; color:#999;'>{node['id']}</td></tr>"
    )

    exclude_keys = {
        "id",
        "name",
        "type",
        "label",
        "title",
        "color",
        "size",
        "x",
        "y",
        "shape",
        "group",
    }
    for key, value in node.items():
        if key in exclude_keys:
            continue
        display_key = key.replace("_", " ").title()
        display_value = _format_value(value)
        html += (
            "<tr style='border-top: 1px solid #eee;'>"
            "<td style='padding:4px; font-weight:bold; color:#666; width: 40%; vertical-align: top;'>"
            f"{display_key}</td><td style='padding:4px;'>{display_value}</td></tr>"
        )

    html += "</table>"
    return html


def build_network(data: Dict[str, Any]) -> Network:
    if Network is None:
        raise RuntimeError(
            "pyvis is required to build graph HTML. Install it with `pip install pyvis`."
        ) from _PYVIS_IMPORT_ERROR

    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#f4f4f4",
        font_color="#333",
        select_menu=True,
        cdn_resources="remote",
    )
    net.barnes_hut(gravity=-3000, spring_length=150, spring_strength=0.04, damping=0.09)

    nodes_registry: Dict[str, bool] = {}
    for _, content in data.items():
        if not isinstance(content, dict):
            continue

        nodes = content.get("nodes", [])
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict) or "id" not in node:
                    continue
                node_id = str(node["id"])
                if node_id in nodes_registry:
                    continue

                node_type = str(node.get("type", "unknown"))
                color = COLOR_MAP.get(node_type, DEFAULT_COLOR)
                label = str(node.get("name", node_id))
                if len(label) > 15:
                    label = f"{label[:15]}..."

                info_html = generate_info_html(node, color=color)
                net.add_node(
                    node_id,
                    label=label,
                    title="Click to view details",
                    color=color,
                    size=20,
                    group=node_type,
                    info_html=info_html,
                )
                nodes_registry[node_id] = True

        edges = content.get("edges", [])
        if isinstance(edges, list):
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                src = edge.get("source_id")
                dst = edge.get("target_id")
                if src is None or dst is None:
                    continue
                src, dst = str(src), str(dst)

                for endpoint in (src, dst):
                    if endpoint not in nodes_registry:
                        net.add_node(
                            endpoint,
                            label=endpoint,
                            color="#ccc",
                            size=10,
                            info_html="<p>Data not available</p>",
                        )
                        nodes_registry[endpoint] = True

                net.add_edge(src, dst, title=str(edge.get("type", "")), arrows="to", color="#ccc")

    return net


def inject_sidebar(html_content: str) -> str:
    custom_css = """
    <style>
        #sidebar {
            position: fixed;
            top: 10px;
            left: 10px;
            width: 300px;
            max-height: 90vh;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            z-index: 999;
            overflow-y: auto;
            display: none;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        #sidebar .close-btn {
            position: absolute;
            top: 5px;
            right: 10px;
            cursor: pointer;
            font-size: 20px;
            color: #999;
        }
        #sidebar .close-btn:hover { color: #333; }
        div.vis-network { outline: none; }
    </style>
    """

    sidebar_div = """
    <div id="sidebar">
        <div class="close-btn" onclick="document.getElementById('sidebar').style.display='none'">&times;</div>
        <div id="node-details">
            <p style="color:#666; text-align:center;">Click any node to inspect details.</p>
        </div>
    </div>
    """

    custom_js = """
    <script type="text/javascript">
        network.on("click", function (params) {
            var sidebar = document.getElementById('sidebar');
            var contentDiv = document.getElementById('node-details');

            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                var nodeData = nodes.get(nodeId);

                if (nodeData && nodeData.info_html) {
                    contentDiv.innerHTML = nodeData.info_html;
                    sidebar.style.display = 'block';
                }
            } else {
                sidebar.style.display = 'none';
            }
        });
    </script>
    """

    injected = f"{custom_css}\n{sidebar_div}\n{custom_js}\n"
    if "</body>" in html_content:
        return html_content.replace("</body>", f"{injected}</body>")
    return f"{html_content}\n{injected}"


def create_graph_html(input_file: Path, output_file: Path) -> Path:
    if not input_file.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    net = build_network(data)
    net.save_graph(str(output_file))

    html_content = output_file.read_text(encoding="utf-8")
    output_file.write_text(inject_sidebar(html_content), encoding="utf-8")
    return output_file


def _output_sort_key(path: Path) -> Tuple[int, str]:
    output_dir_name = path.parent.parent.name
    match = re.fullmatch(r"output_?(\d+)", output_dir_name)
    if match:
        return (int(match.group(1)), output_dir_name)
    return (10**9, output_dir_name)


def discover_input_files(output_extract_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    patterns = (
        "output_*/graph/full_output.json",
        "output*/graph/full_output.json",
    )
    for pattern in patterns:
        candidates.extend(output_extract_dir.glob(pattern))

    unique_files = sorted({path.resolve() for path in candidates if path.is_file()}, key=_output_sort_key)
    return unique_files


def _find_input_in_graph_dir(graph_dir: Path) -> Optional[Path]:
    preferred_names = ("full_output.json", "output_output.json")
    for file_name in preferred_names:
        candidate = graph_dir / file_name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate interactive knowledge-graph HTML from MAGE graph JSON. "
            "Without arguments, scans output_extract/output_*/graph/full_output.json."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input-file", help="Path to one graph JSON file.")
    group.add_argument("--graph-dir", help="Path to one graph directory (contains full_output.json).")
    parser.add_argument("--output-file", help="Output HTML path (default: <graph_dir>/knowledge_graph.html).")
    parser.add_argument(
        "--output-extract-dir",
        default=str(DEFAULT_OUTPUT_EXTRACT_DIR),
        help=f"Directory used for batch scan (default: {DEFAULT_OUTPUT_EXTRACT_DIR}).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing HTML files.")
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    jobs: List[Tuple[Path, Path]] = []

    if args.input_file:
        input_file = Path(args.input_file).expanduser().resolve()
        output_file = Path(args.output_file).expanduser().resolve() if args.output_file else input_file.parent / DEFAULT_HTML_NAME
        jobs.append((input_file, output_file))
        return jobs

    if args.graph_dir:
        graph_dir = Path(args.graph_dir).expanduser().resolve()
        input_file = _find_input_in_graph_dir(graph_dir)
        if input_file is None:
            raise FileNotFoundError(f"No full_output.json or output_output.json in {graph_dir}")
        output_file = Path(args.output_file).expanduser().resolve() if args.output_file else graph_dir / DEFAULT_HTML_NAME
        jobs.append((input_file, output_file))
        return jobs

    output_extract_dir = Path(args.output_extract_dir).expanduser().resolve()
    discovered = discover_input_files(output_extract_dir)
    for input_file in discovered:
        jobs.append((input_file, input_file.parent / DEFAULT_HTML_NAME))
    return jobs


def main() -> int:
    args = parse_args()
    try:
        jobs = build_jobs(args)
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    if not jobs:
        print(f"[error] No graph JSON files found under: {Path(args.output_extract_dir).expanduser().resolve()}")
        return 1

    generated = 0
    skipped = 0
    failed = 0

    for input_file, output_file in jobs:
        try:
            if output_file.exists() and not args.force:
                skipped += 1
                print(f"[skip] {output_file}")
                continue

            create_graph_html(input_file, output_file)
            generated += 1
            print(f"[ok] {input_file} -> {output_file}")
        except Exception as exc:
            failed += 1
            print(f"[fail] {input_file}: {exc}")

    print(f"[summary] generated={generated}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
