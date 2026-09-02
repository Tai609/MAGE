import json
import unittest
from pathlib import Path

from tools.cat_graph.image_utils import (
    image_media_type,
    namespace_generated_node_ids,
    parse_graph_response,
    response_content_to_text,
)


class ImageResponseTests(unittest.TestCase):
    def test_direct_nested_json_is_parsed(self):
        payload = {
            "nodes": [{"id": "node_1", "properties": {"size": {"value": 12}}}],
            "edges": [{"source_id": "node_1", "target_id": "node_2"}],
        }

        self.assertEqual(parse_graph_response(json.dumps(payload)), payload)

    def test_fenced_or_prose_wrapped_json_is_parsed(self):
        payload = {"nodes": [{"id": "img_chem_A"}], "edges": []}
        encoded = json.dumps(payload)

        self.assertEqual(parse_graph_response(f"```json\n{encoded}\n```"), payload)
        self.assertEqual(parse_graph_response(f"Result follows: {encoded}\nDone."), payload)

    def test_response_content_blocks_are_joined(self):
        content = [{"type": "text", "text": "{\"nodes\": []"}, {"type": "text", "text": ", \"edges\": []}"}]

        self.assertEqual(response_content_to_text(content), '{"nodes": [], "edges": []}')

    def test_namespacing_updates_nodes_and_edges(self):
        graph = {
            "nodes": [{"id": "node_1"}, {"id": 2}, {"id": "img_chem_stable"}],
            "edges": [
                {"source_id": "node_1", "target_id": 2},
                {"source": "img_chem_stable", "target": "node_1"},
            ],
        }

        namespace_generated_node_ids(graph, "figure_1")

        self.assertEqual([node["id"] for node in graph["nodes"]], ["figure_1_node_1", "figure_1_2", "img_chem_stable"])
        self.assertEqual(graph["edges"][0]["source_id"], "figure_1_node_1")
        self.assertEqual(graph["edges"][0]["target_id"], "figure_1_2")
        self.assertEqual(graph["edges"][1]["target"], "figure_1_node_1")

    def test_media_type_matches_extension(self):
        self.assertEqual(image_media_type(Path("figure.png")), "image/png")
        self.assertEqual(image_media_type(Path("figure.JPG")), "image/jpeg")
        self.assertEqual(image_media_type(Path("figure.webp")), "image/webp")


if __name__ == "__main__":
    unittest.main()
