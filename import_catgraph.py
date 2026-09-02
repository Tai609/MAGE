import json
import argparse
import logging
import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_nodes(session, nodes, paper_name):
    """Create nodes in Neo4j database"""
    logger.info(f"Creating {len(nodes)} nodes...")
    
    for node in nodes:
        # Prepare node properties
        props = {k: v for k, v in node.items() if k not in ['id', 'type']}
        props['original_id'] = node['id']
        props['paper_name'] = paper_name
        
        # Add type as label
        label = node['type'].capitalize()  # chemical -> Chemical, synthesis -> Synthesis, etc.
        
        # Create Cypher query
        query = f"""
        MERGE (n:{label} {{original_id: $original_id, paper_name: $paper_name}})
        SET n += $props
        """
        
        try:
            session.run(query, original_id=f"{paper_name}_{node['id']}", paper_name=paper_name, props=props)
        except Exception as e:
            logger.error(f"Failed to create node {node['id']}: {e}")
    
    logger.info("Node creation completed.")

def create_edges(session, edges, paper_name):
    """Create relationships in Neo4j database"""
    logger.info(f"Creating {len(edges)} relationships...")
    
    for edge in edges:
        # Prepare relationship properties
        props = {k: v for k, v in edge.items() if k not in ['id', 'type', 'source_id', 'target_id']}
        props['original_id'] = edge['id']
        props['original_type'] = edge['type']
        props['paper_name'] = paper_name
        
        # Map edge type to relationship type
        edge_type_map = {
            'synthesis_input': 'SYNTHESIS_INPUT',
            'synthesis_output': 'SYNTHESIS_OUTPUT',
            'tested_in': 'TESTED_IN',
            'characterized_in': 'CHARACTERIZED_IN'
        }
        
        rel_type = edge_type_map.get(edge['type'], 'RELATED_TO')
        
        # Create Cypher query
        query = f"""
        MATCH (source {{original_id: $source_id, paper_name: $paper_name}})
        MATCH (target {{original_id: $target_id, paper_name: $paper_name}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $props
        """
        
        try:
            session.run(query, 
                       source_id=f"{paper_name}_{edge['source_id']}", 
                       target_id=f"{paper_name}_{edge['target_id']}", 
                       paper_name=paper_name, 
                       props=props)
        except Exception as e:
            logger.error(f"Failed to create relationship {edge['id']}: {e}")
    
    logger.info("Relationship creation completed.")

def import_json_to_neo4j(uri, username, password, json_file_path, paper_name=None):
    """Main function to import JSON data to Neo4j"""
    # Load JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine paper name
    if not paper_name:
        paper_name = Path(json_file_path).stem
    
    logger.info(f"Importing data from {json_file_path} with paper name: {paper_name}")
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        # Verify connectivity
        driver.verify_connectivity()
        logger.info("Connected to Neo4j database successfully.")
        
        with driver.session() as session:
            # Process synthesis data if present
            if 'synthesis' in data:
                synthesis_data = data['synthesis']
                if 'nodes' in synthesis_data:
                    create_nodes(session, synthesis_data['nodes'], paper_name)
                if 'edges' in synthesis_data:
                    create_edges(session, synthesis_data['edges'], paper_name)
            
            # Process testing data if present
            if 'testing' in data:
                testing_data = data['testing']
                if 'nodes' in testing_data:
                    create_nodes(session, testing_data['nodes'], paper_name)
                if 'edges' in testing_data:
                    create_edges(session, testing_data['edges'], paper_name)
            
            # Process characterization data if present
            if 'characterization' in data:
                characterization_data = data['characterization']
                if 'nodes' in characterization_data:
                    create_nodes(session, characterization_data['nodes'], paper_name)
                if 'edges' in characterization_data:
                    create_edges(session, characterization_data['edges'], paper_name)
            
            logger.info("Data import completed successfully!")
            
    except Exception as e:
        logger.error(f"Error during import: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import CatGraph JSON data into Neo4j")
    parser.add_argument("json_file", help="Path to the JSON file to import")
    parser.add_argument(
        "--uri", "--neo4j_uri", dest="uri",
        default=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        help="Neo4j URI (default: NEO4J_URI or neo4j://localhost:7687)",
    )
    parser.add_argument(
        "--username", "--neo4j_user", dest="username",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: NEO4J_USER or neo4j)",
    )
    parser.add_argument(
        "--password", "--neo4j_password", dest="password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j password (prefer NEO4J_PASSWORD in the private environment)",
    )
    parser.add_argument("--paper_name", help="Paper name for the graph")
    
    args = parser.parse_args()
    
    if not args.password:
        parser.error("Neo4j password is required; set NEO4J_PASSWORD or pass --password")

    import_json_to_neo4j(
        uri=args.uri,
        username=args.username,
        password=args.password,
        json_file_path=args.json_file,
        paper_name=args.paper_name
    )
