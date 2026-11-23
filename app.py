import os
import sys
import json
import pickle
import logging
import numpy as np
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Flask imports
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ML/NLP imports
import torch
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# --- Configuration Section ---
INDEX_DIR = 'table_index'
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
# --- End Configuration Section ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedRecordSearchEngine:
    """
    Manages the search indices and performs hybrid search operations.
    """
    def __init__(self):
        """Initializes the search engine by loading pre-trained models and indices."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device} for sentence embeddings.")
        
        self.sentence_model = None
        self.faiss_index = None
        self.bm25 = None
        self.documents = []
        self.metadata = []
        self.lemmatizer = None
        self.stop_words = set()
        
        self.load_indices()
    
    def _setup_nltk_data(self):
        """Helper method to download required NLTK data bundles."""
        required_data = ['punkt', 'stopwords', 'wordnet']
        for data_id in required_data:
            try:
                if data_id == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif data_id == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif data_id == 'wordnet':
                    nltk.data.find('corpora/wordnet')
                logger.info(f"NLTK data '{data_id}' already available.")
            except LookupError:
                logger.info(f"Downloading NLTK data: {data_id}...")
                try:
                    nltk.download(data_id, quiet=True)
                    logger.info(f"Successfully downloaded NLTK data: {data_id}")
                except Exception as e:
                    logger.warning(f"Failed to download NLTK data '{data_id}': {e}")

    def load_indices(self):
        """Loads all saved indices and data from disk."""
        try:
            logger.info("Loading search indices...")
            
            # Setup NLTK
            self._setup_nltk_data()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Load sentence transformer
            self.sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME, device=self.device)
            
            # Load FAISS index
            faiss_path = os.path.join(INDEX_DIR, 'faiss.index')
            if not os.path.exists(faiss_path):
                raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
            self.faiss_index = faiss.read_index(faiss_path)
            
            # Load BM25 index
            bm25_path = os.path.join(INDEX_DIR, 'bm25.pkl')
            if not os.path.exists(bm25_path):
                raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
            with open(bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            
            # Load processed documents
            docs_path = os.path.join(INDEX_DIR, 'processed_records.json')
            if not os.path.exists(docs_path):
                raise FileNotFoundError(f"Documents file not found: {docs_path}")
            with open(docs_path, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

            self.documents = [doc['text'] for doc in processed_data]
            self.metadata = [doc['metadata'] for doc in processed_data]
            
            logger.info(f"Successfully loaded {len(self.documents)} documents and all indices.")
            
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            raise
            
    def semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Performs semantic search using FAISS."""
        if not self.faiss_index or not self.sentence_model:
            return []
        
        try:
            query_embedding = self.sentence_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))
            
            valid_results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx < len(self.documents):
                    valid_results.append((int(idx), float(score)))
            
            return valid_results
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extracts keywords for BM25 search."""
        if not text:
            return []
        
        try:
            text = text.lower()
            tokens = word_tokenize(text)
            
            keywords = []
            for token in tokens:
                if len(token) > 2 and token.isalnum() and token not in self.stop_words:
                    keywords.append(self.lemmatizer.lemmatize(token))
            return keywords
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Performs keyword search using BM25."""
        if not self.bm25:
            return []
        
        try:
            tokenized_query = self._extract_keywords(query)
            if not tokenized_query:
                return []
                
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0 and len(results) < top_k and idx < len(self.documents):
                    results.append((int(idx), float(scores[idx])))
                elif len(results) >= top_k:
                    break
            return results
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []

    def parse_filters(self, query: str) -> Tuple[str, Dict]:
        """Extract filters from query and return clean query."""
        filters = {}
        clean_query = query
        
        # Extract price filters
        price_match = re.search(r'under\s*\$?(\d+)', query.lower())
        if price_match:
            filters['max_price'] = float(price_match.group(1))
            clean_query = re.sub(r'under\s*\$?\d+', '', clean_query, flags=re.IGNORECASE)
        
        # Extract year filters
        year_match = re.search(r'after\s*(\d{4})', query.lower())
        if year_match:
            filters['min_year'] = int(year_match.group(1))
            clean_query = re.sub(r'after\s*\d{4}', '', clean_query, flags=re.IGNORECASE)
        
        # Subject/category detection
        if any(word in query.lower() for word in ['children', 'kids', 'juvenile']):
            filters['subject'] = 'children'
        elif any(word in query.lower() for word in ['fiction', 'novel', 'story']):
            filters['subject'] = 'fiction'
        elif any(word in query.lower() for word in ['history', 'historical']):
            filters['subject'] = 'history'
        
        return clean_query.strip(), filters

    def hybrid_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Performs hybrid search with automatic filter extraction."""
        try:
            # Parse query for filters
            clean_query, filters = self.parse_filters(query)
            
            # Get results from both search methods
            semantic_results = self.semantic_search(clean_query, top_k * 3)
            keyword_results = self.keyword_search(clean_query, top_k * 3)
            
            # Combine scores
            combined_scores = defaultdict(float)
            
            # Add semantic scores (weighted 0.6)
            if semantic_results:
                max_semantic = max(score for _, score in semantic_results) if semantic_results else 1.0
                if max_semantic > 0:
                    for idx, score in semantic_results:
                        combined_scores[idx] += 0.6 * (score / max_semantic)
            
            # Add keyword scores (weighted 0.4)
            if keyword_results:
                max_keyword = max(score for _, score in keyword_results) if keyword_results else 1.0
                if max_keyword > 0:
                    for idx, score in keyword_results:
                        combined_scores[idx] += 0.4 * (score / max_keyword)

            # Sort and filter results
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            final_results = []
            for idx, score in sorted_results:
                if len(final_results) >= top_k:
                    break
                
                if idx < len(self.metadata):
                    metadata = self.metadata[idx]
                    
                    if self._apply_filters(metadata, filters):
                        result = metadata.copy()
                        result['relevance_score'] = round(float(score), 3)
                        final_results.append(result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []
        
    def _apply_filters(self, result: Dict, filters: Dict) -> bool:
        """Apply filters to search results."""
        if not filters:
            return True
        
        try:
            # Price filter
            if 'max_price' in filters:
                try:
                    price = float(result.get('price', 0))
                    if price > filters['max_price']: 
                        return False
                except (ValueError, TypeError):
                    pass
                
            # Year filter
            if 'min_year' in filters:
                try:
                    year = int(result.get('year', 0))
                    if year < filters['min_year']: 
                        return False
                except (ValueError, TypeError):
                    pass
            
            # Subject filter
            if 'subject' in filters:
                filter_subject = filters['subject'].lower()
                categories = result.get('categories', {})
                subjects = categories.get('subjects', [])
                genres = categories.get('genres', [])
                
                if not (filter_subject in result.get('subject', '').lower() or
                        any(filter_subject in s.lower() for s in subjects) or
                        any(filter_subject in g.lower() for g in genres)):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Filter application error: {e}")
            return True

class EnhancedResponseGenerator:
    """
    Generates well-formatted, professional responses for record search results.
    """
    
    def generate_response(self, query: str, results: List[Dict]) -> str:
        """Generate a comprehensive, well-formatted response."""
        if not results:
            return self._generate_no_results_response(query)
        
        response_parts = []
        
        # Header
        response_parts.append(f"Found {len(results)} record(s) for '{query}':")
        response_parts.append("")  # Empty line for spacing
        
        # Format each result
        for i, record in enumerate(results, 1):
            record_section = self._format_record_result(i, record)
            response_parts.append(record_section)
            response_parts.append("")  # Empty line between results
        
        # Add summary if many results
        if len(results) > 10:
            response_parts.append("📚 Tip: Use more specific keywords to narrow down your search.")
        
        return "\n".join(response_parts)
    
    def _format_record_result(self, index: int, record: Dict) -> str:
        """Format a single record result professionally."""
        title = record.get('title', 'Unknown Title')
        author = record.get('author', '')
        publisher = record.get('publisher', '')
        year = record.get('year', '')
        price = record.get('price', '0')
        identifier = record.get('identifier', '')
        format_type = record.get('format', '')
        subject = record.get('subject', '')
        
        # Build the formatted result
        result_lines = [f"**{index}. {title}**"]
        
        # Author line
        if author and author.strip():
            result_lines.append(f"   Author: {author}")
        
        # Publication details
        pub_details = []
        if publisher and publisher.strip():
            pub_details.append(f"Publisher: {publisher}")
        if year and year != '0' and year.strip():
            pub_details.append(f"Year: {year}")
        if format_type and format_type.strip():
            pub_details.append(f"Format: {format_type}")
        
        if pub_details:
            result_lines.append(f"   {' | '.join(pub_details)}")
        
        # Subject/Category
        if subject and subject.strip():
            result_lines.append(f"   Subject: {subject}")
        
        # Price and identifier
        additional_info = []
        if price and price != '0':
            try:
                price_float = float(price)
                if price_float > 0:
                    additional_info.append(f"Price: ₹{price_float:,.2f}")
            except (ValueError, TypeError):
                pass
        
        if identifier and identifier.strip():
            additional_info.append(f"identifier: {identifier}")
        
        if additional_info:
            result_lines.append(f"   {' | '.join(additional_info)}")
        
        return "\n".join(result_lines)
    
    def _generate_no_results_response(self, query: str) -> str:
        """Generate a helpful response when no results are found."""
        suggestions = [
            "• Try using different keywords or synonyms",
            "• Check spelling and try simpler terms", 
            "• Search by author name or record title",
            "• Try broader category terms like 'fiction', 'science', or 'history'"
        ]
        
        return f"""No records found for '{query}'.

**Search Tips:**
{chr(10).join(suggestions)}

**Example searches:**
• "records by Stephen King"
• "children's records"
• "science fiction novels"
• "history textrecords"
"""

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'record-search-engine-2024'
CORS(app)

# Initialize search components
search_engine = None
response_generator = None

def initialize_search_system():
    """Initialize the search system with proper error handling."""
    global search_engine, response_generator
    
    try:
        # Check if index files exist
        required_files = ['faiss.index', 'bm25.pkl', 'processed_records.json']
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(os.path.join(INDEX_DIR, file)):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing index files: {missing_files}")
            logger.error("Please run train.py first to create the search indices.")
            return False
        
        search_engine = AdvancedRecordSearchEngine()
        response_generator = EnhancedResponseGenerator()
        logger.info("Search system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize search system: {e}")
        return False

@app.route('/')
def index():
    """Serves the main HTML interface."""
    try:
        # Try to serve external HTML file first
        if os.path.exists('index.html'):
            with open('index.html', 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Fallback message if no external HTML
            return """
            <h1>RecordBot Search Engine</h1>
            <p>Please create an index.html file or use the API endpoint at /api/search</p>
            <p>API Usage: POST to /api/search with JSON: {"query": "your search query"}</p>
            """
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        return "Error loading the search interface", 500

@app.route('/api/search', methods=['POST'])
def search_records():
    """Main search API endpoint."""
    if not search_engine or not response_generator:
        return jsonify({
            'success': False, 
            'error': 'Search system not initialized. Please run train.py first.'
        }), 500
    
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request format'}), 400
            
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'success': False, 'error': 'Search query cannot be empty'}), 400
        
        # Get search parameters
        max_results = min(int(data.get('max_results', 20)), 50)  # Limit to 50 max
        
        # Perform search
        start_time = time.time()
        results = search_engine.hybrid_search(query, top_k=max_results)
        search_time = round(time.time() - start_time, 3)
        
        # Generate formatted response
        response_text = response_generator.generate_response(query, results)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'response': response_text,
            'total_results': len(results),
            'search_time_seconds': search_time,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Search API error: {e}", exc_info=True)
        return jsonify({
            'success': False, 
            'error': 'Search failed. Please try again with a different query.'
        }), 500

@app.route('/api/health')
def health_check():
    """System health check endpoint."""
    if not search_engine:
        return jsonify({'status': 'unhealthy', 'error': 'Search system not initialized'}), 500
    
    try:
        return jsonify({
            'status': 'healthy',
            'total_records_indexed': len(search_engine.documents),
            'device': search_engine.device,
            'index_directory': INDEX_DIR,
            'model': SENTENCE_MODEL_NAME,
            'features': ['hybrid_search', 'semantic_search', 'keyword_search', 'filtering']
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get search system statistics."""
    if not search_engine:
        return jsonify({'error': 'Search system not initialized'}), 500
    
    try:
        # Calculate basic statistics
        total_records = len(search_engine.metadata)
        
        # Count by categories
        category_counts = defaultdict(int)
        language_counts = defaultdict(int)
        
        for metadata in search_engine.metadata:
            # Count subjects
            subjects = metadata.get('categories', {}).get('subjects', [])
            for subject in subjects:
                category_counts[subject] += 1
            
            # Count languages
            lang = metadata.get('language', 'Unknown')
            language_counts[lang] += 1
        
        return jsonify({
            'total_records': total_records,
            'top_categories': dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'languages': dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': 'Failed to generate statistics'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to start the Flask application."""
    print("Initializing RecordBot Search Engine...")
    
    # Initialize search system
    if not initialize_search_system():
        print("❌ Failed to initialize search system!")
        print("📝 Please run train.py first to create the search indices.")
        sys.exit(1)
    
    print(f"✅ Search system ready with {len(search_engine.documents)} records indexed")
    print(f"🖥️  Using device: {search_engine.device}")
    print("🌐 Starting web server...")
    print("📖 RecordBot available at: http://localhost:5000")
    print("🔍 API endpoint: http://localhost:5000/api/search")
    print("📊 Health check: http://localhost:5000/api/health")
    
    # Start Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 RecordBot stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print("❌ Failed to start the web server")

if __name__ == '__main__':
    main()