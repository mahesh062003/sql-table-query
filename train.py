import os
import sys
import json
import pickle
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')
import re
from collections import defaultdict

# Database imports
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ML/NLP imports
import torch
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Section ---
INDEX_DIR = 'table_index'
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': os.environ.get('DB_NAME', 'sample'),
    'charset': 'utf8mb4'
}

# Table and column names the indexer reads. The defaults below describe an
# illustrative catalogue schema; point them at your own tables.
SCHEMA = {
    'items_table': os.environ.get('ITEMS_TABLE', 'items'),
    'catalogs_table': os.environ.get('CATALOGS_TABLE', 'catalogs'),
    'key_column': os.environ.get('KEY_COLUMN', 'record_id'),
}
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
# --- End Configuration Section ---

class TableIndexProcessor:
    """
    Enhanced processor that indexes ALL available records from the database
    with comprehensive error tracking and data validation.
    """

    def __init__(self):
        """Initialize all components with comprehensive setup."""
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device} for sentence embeddings.")
        
        # Load sentence transformer model
        self.sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME, device=self.device)
        self.embedding_dimension = self.sentence_model.get_sentence_embedding_dimension()
        
        # Ensure index directory exists
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        # Setup NLTK
        self._setup_nltk_data()
        
        # Initialize text processing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load categorization keywords
        self.subject_keywords = self._load_subject_keywords()
        self.age_keywords = self._load_age_keywords()
        self.genre_keywords = self._load_genre_keywords()
        
        # Processing statistics
        self.stats = {
            'total_fetched': 0,
            'successfully_processed': 0,
            'skipped_invalid': 0,
            'skipped_errors': 0,
            'duplicate_record_ids': 0
        }

    def _setup_nltk_data(self):
        """Download required NLTK data."""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for data_id in required_data:
            try:
                if data_id == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif data_id == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif data_id == 'wordnet':
                    nltk.data.find('corpora/wordnet')
                elif data_id == 'averaged_perceptron_tagger':
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                logger.info(f"NLTK data '{data_id}' already available.")
            except LookupError:
                logger.info(f"Downloading NLTK data: {data_id}...")
                try:
                    nltk.download(data_id, quiet=True)
                    logger.info(f"Successfully downloaded NLTK data: {data_id}")
                except Exception as e:
                    logger.warning(f"Failed to download NLTK data '{data_id}': {e}")

    def _load_subject_keywords(self) -> Dict[str, List[str]]:
        """Enhanced subject keywords for better categorization."""
        return {
            'children': ['children', 'kids', 'juvenile', 'young readers', 'picture record', 'bedtime', 'fairy tale', 'nursery', 'toddler', 'preschool'],
            'history': ['history', 'historical', 'ancient', 'medieval', 'war', 'biography', 'civilization', 'chronicle', 'past', 'empire', 'revolution'],
            'science': ['science', 'physics', 'chemistry', 'biology', 'mathematics', 'astronomy', 'research', 'experiment', 'laboratory', 'theory'],
            'fiction': ['novel', 'story', 'tales', 'adventure', 'mystery', 'romance', 'thriller', 'fantasy', 'sci-fi', 'fiction', 'narrative'],
            'education': ['textrecord', 'curriculum', 'academic', 'study', 'learning', 'educational', 'course', 'school', 'university', 'college'],
            'religion': ['religious', 'spiritual', 'faith', 'prayer', 'theology', 'bible', 'quran', 'hinduism', 'buddhism', 'christianity'],
            'business': ['business', 'management', 'economics', 'finance', 'marketing', 'entrepreneurship', 'leadership', 'corporate', 'strategy'],
            'technology': ['computer', 'programming', 'software', 'digital', 'internet', 'AI', 'machine learning', 'data', 'technology', 'coding'],
            'health': ['health', 'medical', 'medicine', 'wellness', 'fitness', 'nutrition', 'psychology', 'therapy', 'healthcare', 'disease'],
            'arts': ['art', 'music', 'painting', 'sculpture', 'photography', 'design', 'creative', 'aesthetic', 'culture', 'literature'],
            'cooking': ['cooking', 'recipe', 'food', 'cuisine', 'culinary', 'kitchen', 'chef', 'baking', 'nutrition', 'diet'],
            'travel': ['travel', 'tourism', 'guide', 'geography', 'culture', 'destination', 'adventure', 'exploration', 'world'],
            'sports': ['sports', 'fitness', 'exercise', 'training', 'athletic', 'game', 'competition', 'physical', 'outdoor']
        }
    
    def _load_age_keywords(self) -> Dict[str, List[str]]:
        """Age group classification keywords."""
        return {
            'children': ['children', 'kids', 'juvenile', 'young readers', 'ages 3-8', 'ages 5-12', 'elementary', 'primary'],
            'young_adult': ['young adult', 'teen', 'teenager', 'ya', 'ages 12-18', 'high school', 'adolescent'],
            'adult': ['adult', 'mature', 'grown-up', 'professional', 'university', 'college', 'advanced']
        }
    
    def _load_genre_keywords(self) -> Dict[str, List[str]]:
        """Genre classification keywords."""
        return {
            'mystery': ['mystery', 'detective', 'crime', 'investigation', 'suspense', 'whodunit', 'thriller'],
            'romance': ['romance', 'love story', 'romantic', 'relationship', 'passion', 'love'],
            'fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'mythical', 'supernatural', 'fairy'],
            'sci_fi': ['science fiction', 'sci-fi', 'futuristic', 'space', 'alien', 'robot', 'dystopian', 'cyberpunk'],
            'horror': ['horror', 'scary', 'ghost', 'vampire', 'zombie', 'supernatural', 'dark'],
            'biography': ['biography', 'autobiography', 'memoir', 'life story', 'personal account'],
            'self_help': ['self-help', 'personal development', 'motivation', 'success', 'improvement', 'guide'],
            'technical': ['manual', 'handrecord', 'reference', 'guide', 'technical', 'documentation'],
            'poetry': ['poetry', 'poems', 'verse', 'rhyme', 'sonnet', 'ballad']
        }

    def _clean_text(self, text: Any) -> str:
        """Safely clean and normalize text fields."""
        if pd.isna(text) or text is None or text == '':
            return ""
        
        text = str(text).strip()
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause encoding issues
        text = re.sub(r'[^\w\s\-.,;:()[\]{}/"\'&@#%]', ' ', text)
        return text.strip()

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract and normalize keywords with better processing."""
        if not text or text.strip() == "":
            return []
        
        try:
            text = text.lower()
            tokens = word_tokenize(text)
            
            keywords = []
            for token in tokens:
                # More permissive token filtering
                if len(token) > 1 and token not in self.stop_words:
                    # Add original token
                    if token.isalnum():
                        keywords.append(token)
                    # Add lemmatized version
                    lemmatized = self.lemmatizer.lemmatize(token)
                    if lemmatized != token and lemmatized.isalnum():
                        keywords.append(lemmatized)
            
            return list(set(keywords))  # Remove duplicates
        except Exception as e:
            logger.warning(f"Keyword extraction failed for text: {str(text)[:50]}... Error: {e}")
            return []

    def _categorize_record(self, row: Dict) -> Dict[str, Any]:
        """Enhanced record categorization with better text handling."""
        # Combine all text fields for analysis
        text_fields = [
            row.get('Title', ''),
            row.get('subtitle', ''),
            row.get('Subject', ''),
            row.get('details', ''),
            row.get('contents', ''),
            row.get('author', ''),
            row.get('publisher', ''),
            row.get('item_type', '')
        ]
        
        full_text = ' '.join(self._clean_text(field) for field in text_fields).lower()
        
        categories = {
            'subjects': [],
            'age_groups': [],
            'genres': [],
            'keywords': self._extract_keywords(full_text)
        }
        
        # Categorize using keyword matching
        for cat_type, keyword_dict in [
            ('subjects', self.subject_keywords), 
            ('age_groups', self.age_keywords), 
            ('genres', self.genre_keywords)
        ]:
            for category, keywords in keyword_dict.items():
                if any(keyword.lower() in full_text for keyword in keywords):
                    categories[cat_type].append(category)
        
        return categories

    def connect_database(self):
        """Establish database connection with comprehensive error handling."""
        try:
            # Create connection string
            connection_string = (
                f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
                f"{DB_CONFIG['host']}/{DB_CONFIG['database']}?charset={DB_CONFIG['charset']}"
            )
            
            engine = create_engine(connection_string, pool_pre_ping=True, pool_recycle=300)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Database connection successful.")
            
            return engine
            
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {e}")
            logger.error("Please check your database configuration in DB_CONFIG")
            raise
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
            raise

    def get_database_stats(self, engine):
        """Get comprehensive database statistics."""
        try:
            with engine.connect() as conn:
                # Total records
                total_result = conn.execute(text("SELECT COUNT(*) as total FROM items"))
                total_count = total_result.fetchone()[0]
                
                # Active records
                active_result = conn.execute(text("""
                    SELECT COUNT(*) as active 
                    FROM items 
                    WHERE status = 'Active' OR status IS NULL
                """))
                active_count = active_result.fetchone()[0]
                
                # Status breakdown
                status_result = conn.execute(text("""
                    SELECT status, COUNT(*) as count 
                    FROM items 
                    GROUP BY status
                """))
                status_breakdown = dict(status_result.fetchall())
                
                logger.info(f"Database Statistics:")
                logger.info(f"  Total records: {total_count}")
                logger.info(f"  Active/NULL status: {active_count}")
                logger.info(f"  Status breakdown: {status_breakdown}")
                
                return {
                    'total': total_count,
                    'active': active_count,
                    'status_breakdown': status_breakdown
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return None

    def fetch_records_data(self) -> pd.DataFrame:
        """Fetch ALL available record data with comprehensive logging."""
        logger.info("Connecting to database...")
        engine = self.connect_database()
        
        # Get database statistics first
        db_stats = self.get_database_stats(engine)
        
        # Modified query to get ALL records (remove restrictive WHERE clause)
        query = """
        SELECT 
            p.*,
            c.Title as CatalogTitle,
            c.Type as CatalogType
        FROM items p
        LEFT JOIN catalogs c ON FIND_IN_SET(p.record_id, REPLACE(c.record_id, ' ', '')) > 0
        """
        
        try:
            logger.info("Executing database query...")
            df = pd.read_sql(query, engine)
            self.stats['total_fetched'] = len(df)
            logger.info(f"Fetched {len(df)} total records from database")
            
            # Check for duplicates
            initial_count = len(df)
            df = df.drop_duplicates(subset=['record_id'], keep='first')
            duplicates_removed = initial_count - len(df)
            
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate records")
                self.stats['duplicate_record_ids'] = duplicates_removed
            
            logger.info(f"Final dataset: {len(df)} unique records")
            
            # Show sample of data for verification
            logger.info("Sample record fields:")
            for col in df.columns:
                sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
                logger.info(f"  {col}: {str(sample_value)[:50]}...")
            
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data fetch: {e}")
            raise
        finally:
            engine.dispose()

    def validate_record_record(self, row_data: Dict) -> tuple[bool, str]:
        """Validate if a record record has minimum required data."""
        record_id = row_data.get('record_id')
        title = self._clean_text(row_data.get('Title', ''))
        
        # Must have record_id and some form of title
        if not record_id:
            return False, "Missing record_id"
        
        if not title and not self._clean_text(row_data.get('CatalogTitle', '')):
            return False, "Missing Title"
        
        return True, "Valid"

    def process_single_record(self, row_data: Dict) -> Optional[Dict]:
        """Process a single record record with comprehensive error handling."""
        try:
            # Validate record
            is_valid, reason = self.validate_record_record(row_data)
            if not is_valid:
                self.stats['skipped_invalid'] += 1
                logger.debug(f"Skipping record_id {row_data.get('record_id', 'Unknown')}: {reason}")
                return None
            
            # Clean and prepare all fields
            row = {}
            for key, value in row_data.items():
                row[key] = self._clean_text(value)
            
            # Use title priority: Title > CatalogTitle
            final_title = row.get('Title') or row.get('CatalogTitle', '')
            
            # Categorize the record
            categories = self._categorize_record(row)
            
            # Create comprehensive searchable document
            document_text = f"""
            Title: {final_title}
            Subtitle: {row.get('subtitle', '')}
            Author: {row.get('author', '')}
            Publisher: {row.get('publisher', '')}
            Subject: {row.get('Subject', '')}
            Language: {row.get('Language', '')}
            Publication Year: {row.get('year', '')}
            Country: {row.get('country', '')}
            identifier: {row.get('identifier', '')}
            Record Details: {row.get('details', '')}
            Content Details: {row.get('contents', '')}
            Product Type: {row.get('item_type', '')}
            Format: {row.get('format', '')}
            Price: {row.get('Price', '')}
            Stock: {row.get('Stock', '')}
            Categories: {', '.join(categories['subjects'])} {', '.join(categories['age_groups'])} {', '.join(categories['genres'])}
            Keywords: {', '.join(categories['keywords'])}
            """.strip()
            
            # Create metadata object
            metadata = {
                'record_id': str(row.get('record_id', '')),
                'title': final_title,
                'subtitle': row.get('subtitle', ''),
                'author': row.get('author', ''),
                'publisher': row.get('publisher', ''),
                'subject': row.get('Subject', ''),
                'language': row.get('Language', ''),
                'year': row.get('year', ''),
                'country': row.get('country', ''),
                'identifier': row.get('identifier', ''),
                'record_details': row.get('details', ''),
                'content_details': row.get('contents', ''),
                'item_type': row.get('item_type', ''),
                'format': row.get('format', ''),
                'price': row.get('Price', '0'),
                'stock': row.get('Stock', '0'),
                'categories': categories
            }
            
            return {
                'id': str(row.get('record_id')),
                'text': document_text,
                'metadata': metadata
            }
            
        except Exception as e:
            self.stats['skipped_errors'] += 1
            logger.warning(f"Error processing record_id {row_data.get('record_id', 'Unknown')}: {e}")
            return None

    def process_and_index_data(self):
        """Main processing pipeline with comprehensive tracking."""
        logger.info("Starting comprehensive record indexing pipeline...")
        
        try:
            # Fetch data
            df = self.fetch_records_data()
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return False
        
        if df.empty:
            logger.error("No data retrieved from database.")
            return False

        # Process all records
        logger.info("Processing all record records...")
        processed_documents = []
        tokenized_docs = []
        seen_record_ids = set()
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processing record {idx + 1}/{len(df)}...")
            
            row_dict = row.to_dict()
            record_id = row_dict.get('record_id')
            
            # Skip duplicates
            if record_id in seen_record_ids:
                self.stats['duplicate_record_ids'] += 1
                continue
            seen_record_ids.add(record_id)
            
            # Process the record
            processed_record = self.process_single_record(row_dict)
            
            if processed_record:
                processed_documents.append(processed_record)
                
                # Prepare BM25 tokenized version
                bm25_text = f"{processed_record['text']} {row_dict.get('Title', '')} {row_dict.get('author', '')}"
                tokenized_docs.append(self._extract_keywords(bm25_text))
                
                self.stats['successfully_processed'] += 1
        
        # Log processing statistics
        self._log_processing_stats()
        
        if not processed_documents:
            logger.error("No valid documents were processed.")
            return False

        logger.info(f"Successfully processed {len(processed_documents)} records for indexing.")
        
        # Build indices
        return self._build_and_save_indices(processed_documents, tokenized_docs)

    def _log_processing_stats(self):
        """Log comprehensive processing statistics."""
        logger.info("=== Processing Statistics ===")
        logger.info(f"Total records fetched: {self.stats['total_fetched']}")
        logger.info(f"Successfully processed: {self.stats['successfully_processed']}")
        logger.info(f"Skipped (invalid data): {self.stats['skipped_invalid']}")
        logger.info(f"Skipped (processing errors): {self.stats['skipped_errors']}")
        logger.info(f"Duplicate record ids: {self.stats['duplicate_record_ids']}")
        
        if self.stats['total_fetched'] > 0:
            success_rate = (self.stats['successfully_processed'] / self.stats['total_fetched']) * 100
            logger.info(f"Processing success rate: {success_rate:.2f}%")

    def _build_and_save_indices(self, processed_documents: List[Dict], tokenized_docs: List[List[str]]) -> bool:
        """Build FAISS and BM25 indices and save everything."""
        
        # 1. Build FAISS semantic index
        logger.info("Building FAISS semantic search index...")
        try:
            all_texts = [doc['text'] for doc in processed_documents]
            
            # Process embeddings in batches to handle memory better
            batch_size = 100
            embeddings = []
            
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                logger.info(f"Processing embeddings batch {i//batch_size + 1}/{(len(all_texts)//batch_size) + 1}")
                
                batch_embeddings = self.sentence_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                embeddings.append(batch_embeddings)
            
            # Combine all embeddings
            all_embeddings = np.vstack(embeddings)
            
            # Create FAISS index
            faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
            faiss_index.add(all_embeddings.astype('float32'))
            
            logger.info(f"FAISS index created with {faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False

        # 2. Build BM25 keyword index
        logger.info("Building BM25 keyword search index...")
        try:
            # Filter empty documents
            valid_tokenized = [doc for doc in tokenized_docs if doc]
            
            if not valid_tokenized:
                logger.error("No valid tokenized documents for BM25")
                return False
            
            bm25_index = BM25Okapi(valid_tokenized)
            logger.info("BM25 index created successfully")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            return False

        # 3. Save all components
        return self._save_all_indices(faiss_index, bm25_index, processed_documents)

    def _save_all_indices(self, faiss_index, bm25_index, processed_documents: List[Dict]) -> bool:
        """Save all indices and data with verification."""
        logger.info("Saving all search components...")
        
        try:
            # Save FAISS index
            faiss_path = os.path.join(INDEX_DIR, 'faiss.index')
            faiss.write_index(faiss_index, faiss_path)
            logger.info(f"FAISS index saved: {faiss_path}")
            
            # Save BM25 index
            bm25_path = os.path.join(INDEX_DIR, 'bm25.pkl')
            with open(bm25_path, 'wb') as f:
                pickle.dump(bm25_index, f)
            logger.info(f"BM25 index saved: {bm25_path}")
            
            # Save processed documents
            docs_path = os.path.join(INDEX_DIR, 'processed_records.json')
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(processed_documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Processed documents saved: {docs_path}")
            
            # Save processing statistics
            stats_path = os.path.join(INDEX_DIR, 'processing_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2)
            logger.info(f"Processing statistics saved: {stats_path}")
            
            # Verify file sizes
            self._verify_saved_files()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save indices: {e}")
            return False

    def _verify_saved_files(self):
        """Verify that all saved files are valid."""
        files_to_check = [
            ('faiss.index', 'FAISS index'),
            ('bm25.pkl', 'BM25 index'),
            ('processed_records.json', 'Processed records'),
            ('processing_stats.json', 'Processing statistics')
        ]
        
        logger.info("Verifying saved files...")
        for filename, description in files_to_check:
            filepath = os.path.join(INDEX_DIR, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"  {description}: {size_mb:.2f} MB")
            else:
                logger.error(f"  {description}: FILE NOT FOUND!")

def main():
    """Main execution function."""
    print("=" * 60)
    print("📚 COMPREHENSIVE RECORD DATABASE INDEXING SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize processor
        processor = TableIndexProcessor()
        
        # Run the complete pipeline
        success = processor.process_and_index_data()
        
        if success:
            print("\n" + "=" * 60)
            print("✅ INDEXING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📁 All files saved in: '{INDEX_DIR}/'")
            print(f"📊 Records successfully indexed: {processor.stats['successfully_processed']}")
            print(f"⚡ Processing success rate: {(processor.stats['successfully_processed'] / processor.stats['total_fetched'] * 100):.1f}%")
            print("\n🚀 You can now run app.py to start the search application!")
            print("🌐 Access your RecordBot at: http://localhost:5000")
            
        else:
            print("\n" + "=" * 60)
            print("❌ INDEXING FAILED!")
            print("=" * 60)
            print("Please check the error messages above and:")
            print("1. Verify your database connection settings")
            print("2. Ensure your database contains valid record records")
            print("3. Check that you have sufficient disk space")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Indexing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
        print(f"\n❌ Critical error: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()