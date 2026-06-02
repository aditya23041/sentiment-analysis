import logging
import networkx as nx
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationalMemoryGraph:
    """
    A NetworkX directed graph to store conversation history and emotional state drift.
    Used to track sarcasm context over time (e.g., if a user was previously angry, 
    a subsequent "positive" statement might be sarcastic).
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self._counter = 0

    def add_utterance(self, session_id: str, text: str, emotion_scores: Dict[str, float] = None) -> int:
        """
        Adds a new user utterance to their session history.
        Links it chronologically to their previous utterance.
        """
        node_id = self._counter
        self._counter += 1
        
        timestamp = datetime.now().isoformat()
        
        # Add node
        self.graph.add_node(
            node_id, 
            session_id=session_id, 
            text=text, 
            timestamp=timestamp,
            emotions=emotion_scores or {}
        )
        
        # Find the most recent node for this session to create a temporal edge
        # Optimize: In a production DB, we wouldn't iterate all nodes.
        last_node = None
        last_time = ""
        for n, data in self.graph.nodes(data=True):
            if data.get("session_id") == session_id and n != node_id:
                if data.get("timestamp", "") > last_time:
                    last_time = data["timestamp"]
                    last_node = n
                    
        if last_node is not None:
            # Create a temporal link representing state drift
            self.graph.add_edge(last_node, node_id, type="temporal_progression")
            
        logger.debug(f"Added utterance {node_id} to session {session_id}")
        return node_id

    def get_session_history(self, session_id: str, limit: int = 5) -> List[Dict]:
        """
        Retrieves the most recent utterances for a given session.
        Returns them in chronological order.
        """
        session_nodes = [
            (n, data) for n, data in self.graph.nodes(data=True) 
            if data.get("session_id") == session_id
        ]
        
        # Sort by timestamp ascending
        session_nodes.sort(key=lambda x: x[1].get("timestamp", ""))
        
        # Return the last 'limit' nodes
        recent_nodes = session_nodes[-limit:]
        return [data for n, data in recent_nodes]

    def clear_session(self, session_id: str):
        """Removes a session from memory."""
        nodes_to_remove = [n for n, data in self.graph.nodes(data=True) if data.get("session_id") == session_id]
        self.graph.remove_nodes_from(nodes_to_remove)
        logger.info(f"Cleared {len(nodes_to_remove)} nodes for session {session_id}")

# Singleton instance for the running app
memory_graph = ConversationalMemoryGraph()
