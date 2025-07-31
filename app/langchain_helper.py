import time
from typing import List, Dict, Any, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging
import os
import pickle
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

logger = logging.getLogger(__name__)

class UniversityRecommendationService:
    """Super optimized University Recommendation Service with proper caching"""
    
    def __init__(self, gemini_api_key: str, cache_dir: str = "vector_store_cache"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        self.vector_store = None
        self.university_data = None
        self.cache_path = cache_dir
        self.metadata_cache = {}
        self.query_cache = {}
        self.cache_lock = threading.Lock()
        self._cache_loaded = False  # Track if cache was successfully loaded
        
        # Try to load existing vector store
        self._cache_loaded = self._try_load_cached_vector_store()
    
    def _try_load_cached_vector_store(self):
        """Try to load existing vector store from cache"""
        try:
            if os.path.exists(self.cache_path) and os.path.exists(f"{self.cache_path}/index.faiss"):
                logger.info("Loading cached vector store...")
                start_time = time.time()
                
                self.vector_store = FAISS.load_local(
                    self.cache_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                load_time = time.time() - start_time
                logger.info(f"Successfully loaded cached vector store in {load_time:.2f}s")
                return True
        except Exception as e:
            logger.warning(f"Could not load cached vector store: {e}")
        return False
    
    def _get_data_hash(self, documents: List[Dict[str, Any]]) -> str:
        """Generate hash of document data to check if reprocessing is needed"""
        sample_data = {
            'count': len(documents),
            'sample': str(sorted([doc.get('_id', '') for doc in documents[:100]]))
        }
        return hashlib.md5(json.dumps(sample_data, sort_keys=True).encode()).hexdigest()
    
    def _is_cache_valid(self, mongodb_documents: List[Dict[str, Any]]) -> bool:
        """Check if current cache is valid for the given documents"""
        if not self._cache_loaded or not self.vector_store:
            return False
            
        hash_file = f"{self.cache_path}/data_hash.txt"
        if not os.path.exists(hash_file):
            return False
            
        try:
            current_hash = self._get_data_hash(mongodb_documents)
            with open(hash_file, 'r') as f:
                cached_hash = f.read().strip()
            return cached_hash == current_hash
        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False
    
    def _convert_to_enhanced_categories(self, doc: Dict[str, Any]) -> Dict[str, str]:
        """Optimized category conversion with caching"""
        doc_id = doc.get('_id', '')
        
        # Check cache first
        with self.cache_lock:
            if doc_id in self.metadata_cache:
                return self.metadata_cache[doc_id]
        
        enhanced = {}
        
        # Optimized ranking categories
        rank = doc.get('university_global_rank', 'unknown')
        if rank != 'unknown' and rank != '' and str(rank).isdigit():
            rank_val = int(rank)
            enhanced['rank_category'] = (
                "top-50-globally" if rank_val <= 50 else
                "top-100-globally" if rank_val <= 100 else
                "top-200-globally" if rank_val <= 200 else
                "top-500-globally" if rank_val <= 500 else
                "ranked-university"
            )
        else:
            enhanced['rank_category'] = "unranked-or-unknown"
        
        # Optimized tuition categories
        tuition = doc.get('university_course_tuition_usd', 'unknown')
        if tuition != 'unknown' and tuition != '' and str(tuition).isdigit():
            tuition_val = int(tuition)
            enhanced['tuition_category'] = (
                "very-affordable" if tuition_val <= 10000 else
                "affordable" if tuition_val <= 25000 else
                "moderate-cost" if tuition_val <= 50000 else
                "expensive"
            )
        else:
            enhanced['tuition_category'] = "unknown-cost"
        
        # Quick level mapping
        level_map = {
            'foundation': 'Foundation',
            'certificate': 'Certificate',
            'diploma': 'Diploma',
            'bachelors': 'Bachelors',
            'masters': 'Masters',
            'doctorate': 'Doctorate',
            'phd': 'Doctorate',
            
            # Common variations
            'undergraduate': 'Bachelors',
            'graduate': 'Masters',
            'postgraduate': 'Masters',
            'doctoral': 'Doctorate'
        }
       
        enhanced['program_level_category'] = level_map.get(str(doc.get('program_level', 'unknown')), "unknown-level")
        
        # Quick scholarship categories
        scholarships = doc.get('scholarship_count', 'unknown')
        if scholarships != 'unknown' and scholarships != '' and str(scholarships).isdigit():
            scholarship_val = int(scholarships)
            enhanced['scholarship_category'] = (
                "many-scholarships-available" if scholarship_val >= 10 else
                "several-scholarships-available" if scholarship_val >= 5 else
                "few-scholarships-available" if scholarship_val > 0 else
                "no-scholarships"
            )
        else:
            enhanced['scholarship_category'] = "unknown-scholarships"
        
        # Quick GRE requirement
        gre = str(doc.get('is_gre_required', 'unknown')).lower()
        enhanced['gre_category'] = (
            "gre-required" if gre in {'yes', 'true', '1'} else
            "gre-not-required" if gre in {'no', 'false', '0'} else
            "gre-unknown"
        )
        
        # Cache the result
        with self.cache_lock:
            self.metadata_cache[doc_id] = enhanced
        
        return enhanced
    
    def _create_optimized_document_text(self, doc: Dict[str, Any], enhanced: Dict[str, str]) -> str:
        """Create super optimized text representation"""
        text_parts = []
        
        # Core academic information (most important for matching)
        if doc.get('parent_course_name', 'unknown') != 'unknown':
            text_parts.append(f"Field: {doc['parent_course_name']}")
        
        if doc.get('university_course_name', 'unknown') != 'unknown':
            text_parts.append(f"Course: {doc['university_course_name']}")
        
        if doc.get('university_name', 'unknown') != 'unknown':
            text_parts.append(f"University: {doc['university_name']}")
        
        # Location (important for preferences)
        location_parts = []
        if doc.get('country_name', 'unknown') != 'unknown':
            location_parts.append(doc['country_name'])
        if doc.get('location_name', 'unknown') != 'unknown':
            location_parts.append(doc['location_name'])
        if location_parts:
            text_parts.append(f"Location: {', '.join(location_parts)}")
        
        # Enhanced categories (for semantic matching)
        if enhanced['program_level_category'] != "unknown-level":
            text_parts.append(f"Level: {enhanced['program_level_category'].replace('-', ' ')}")
        
        if enhanced['tuition_category'] != "unknown-cost":
            text_parts.append(f"Cost: {enhanced['tuition_category'].replace('-', ' ')}")
        
        if enhanced['rank_category'] != "unranked-or-unknown":
            text_parts.append(f"Ranking: {enhanced['rank_category'].replace('-', ' ')}")
        
        return " | ".join(text_parts) if text_parts else "University Program"
    
    def load_data_from_mongodb(self, mongodb_documents: List[Dict[str, Any]], force_reload: bool = False):
        """FIXED: Proper caching that avoids re-embedding unchanged data"""
        start_time = time.time()
        doc_count = len(mongodb_documents)
        
        # Store the university data regardless of cache status
        self.university_data = mongodb_documents
        
        # Check if we can use existing cache
        if not force_reload and self._is_cache_valid(mongodb_documents):
            logger.info(f"✅ Cache is valid! Using existing vector store for {doc_count:,} documents")
            logger.info(f"Skipped embedding creation - saved significant time!")
            return
        
        logger.info(f"🔄 Cache invalid or force reload requested")
        logger.info(f"Processing {doc_count:,} documents for vector store creation...")
        
        try:
            # STEP 1: Parallel document preprocessing (FAST)
            logger.info("Step 1/3: Preprocessing documents...")
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                def process_single_document(doc):
                    enhanced = self._convert_to_enhanced_categories(doc)
                    text = self._create_optimized_document_text(doc, enhanced)
                    metadata = {**doc, 'enhanced_features': enhanced}
                    return Document(page_content=text, metadata=metadata)
                
                # Process all documents in parallel with progress bar
                documents = list(tqdm(
                    executor.map(process_single_document, mongodb_documents),
                    total=doc_count,
                    desc="Processing docs",
                    unit="docs"
                ))
            
            step1_time = time.time() - start_time
            logger.info(f"Step 1 completed in {step1_time:.2f}s")
            
            # STEP 2: Create embeddings and vector store (THE SLOW PART)
            logger.info("Step 2/3: Creating vector store with embeddings...")
            logger.info("⏰ This is the slow step that creates embeddings via Gemini API...")
            
            embedding_start = time.time()
            
            # Use LangChain's built-in batch processing - much simpler!
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            embedding_time = time.time() - embedding_start
            logger.info(f"Step 2 completed in {embedding_time:.2f}s ({doc_count/embedding_time:.1f} docs/sec)")
            
            # STEP 3: Save cache (FAST)
            logger.info("Step 3/3: Saving cache...")
            cache_start = time.time()
            
            try:
                os.makedirs(self.cache_path, exist_ok=True)
                self.vector_store.save_local(self.cache_path)
                
                # Save data hash for future cache validation
                data_hash = self._get_data_hash(mongodb_documents)
                hash_file = f"{self.cache_path}/data_hash.txt"
                with open(hash_file, 'w') as f:
                    f.write(data_hash)
                
                # Update cache status
                self._cache_loaded = True
                
                cache_time = time.time() - cache_start
                logger.info(f"Step 3 completed in {cache_time:.2f}s")
                logger.info("✅ Vector store cache saved successfully")
                
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
            
            total_time = time.time() - start_time
            logger.info(f"🎉 COMPLETE! Vector store created in {total_time:.2f}s")
            logger.info(f"📊 Breakdown: Preprocessing: {step1_time:.1f}s, Embeddings: {embedding_time:.1f}s, Cache: {cache_time:.1f}s")
            logger.info(f"⚡ Average speed: {doc_count/total_time:.1f} documents/second")
            
        except Exception as e:
            logger.error(f"❌ Error loading data into vector store: {e}")
            raise
    
    def get_recommendations(self, user_preferences: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Optimized recommendation with query caching"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Create cache key for query
            query_key = hashlib.md5(
                json.dumps(user_preferences, sort_keys=True).encode()
            ).hexdigest()
            
            # Check query cache
            with self.cache_lock:
                if query_key in self.query_cache:
                    logger.info("Using cached query results")
                    return self.query_cache[query_key][:top_k]
            
            # Create optimized query
            query = self._create_optimized_query(user_preferences)
            logger.info(f"Searching with query: {query}")
            
            # Optimized search
            search_k = min(top_k * 3, 100)
            similar_docs = self.vector_store.similarity_search_with_score(query, k=search_k)
            
            # Fast processing
            recommendations = []
            seen_combinations = set()
            
            for doc, score in similar_docs:
                # Quick duplicate check
                unique_id = f"{doc.metadata.get('university_name', '')}-{doc.metadata.get('university_course_name', '')}"
                if unique_id in seen_combinations:
                    continue
                seen_combinations.add(unique_id)
                
                # Fast match calculation
                match_percentage = self._fast_calculate_match(doc.metadata, user_preferences)
                
                recommendation = {
                    'university_name': doc.metadata.get('university_name', 'Unknown'),
                    'course_name': doc.metadata.get('university_course_name', 'Unknown'),
                    'program_label': doc.metadata.get('course_program_label', 'Unknown'),
                    'parent_course': doc.metadata.get('parent_course_name', 'Unknown'),
                    'location': doc.metadata.get('location_name', 'Unknown'),
                    'country': doc.metadata.get('country_name', 'Unknown'),
                    'global_rank': doc.metadata.get('university_global_rank', 'Unknown'),
                    'tuition_usd': doc.metadata.get('university_course_tuition_usd', 'Unknown'),
                    'university_type': doc.metadata.get('university_type', 'Unknown'),
                    'scholarship_count': doc.metadata.get('scholarship_count', 'Unknown'),
                    'is_gre_required': doc.metadata.get('is_gre_required', 'Unknown'),
                    'similarity_score': float(1 - score),
                    'match_percentage': match_percentage,
                    'reasoning': self._generate_fast_reasoning(doc.metadata, user_preferences, match_percentage),
                    'relevance_score': (match_percentage / 100 + (1 - float(score))) / 2
                }
                recommendations.append(recommendation)
                
                if len(recommendations) >= top_k * 2:
                    break
            
            # Sort and limit
            recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
            final_results = recommendations[:top_k]
            
            # Cache results
            with self.cache_lock:
                if len(self.query_cache) > 100:
                    # Remove oldest entries
                    oldest_keys = list(self.query_cache.keys())[:20]
                    for key in oldest_keys:
                        del self.query_cache[key]
                
                self.query_cache[query_key] = recommendations[:50]
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def _create_optimized_query(self, preferences: Dict[str, Any]) -> str:
        """Create optimized search query"""
        query_parts = []
        
        if preferences.get('desired_program'):
            program = preferences['desired_program']
            query_parts.append(f"Field: {program} Course: {program}")
        
        if preferences.get('preferred_countries'):
            countries = preferences['preferred_countries'][:3]
            query_parts.append(f"Location: {' '.join(countries)}")
        
        if preferences.get('program_level'):
            query_parts.append(f"Level: {preferences['program_level']}")
        
        if preferences.get('max_tuition_usd'):
            budget = preferences['max_tuition_usd']
            if budget <= 25000:
                query_parts.append("Cost: affordable")
            elif budget <= 50000:
                query_parts.append("Cost: moderate")
        
        return " | ".join(query_parts) if query_parts else "university programs"
    
    def _fast_calculate_match(self, metadata: Dict[str, Any], preferences: Dict[str, Any]) -> float:
        """Calculate final match percentage using enhanced scoring"""
        # Weights for different components
        weights = {
            'program_relevance': 0.50,
            'location_fit': 0.20,
            'budget_match': 0.15,
            'ranking_score': 0.10,
            'additional_fit': 0.05
        }
        
        # Get detailed scores
        scores = self._calculate_enhanced_match(metadata, preferences)
        
        # Calculate weighted sum
        final_score = sum(
            score * weights[component]
            for component, score in scores.items()
        ) * 100
        
        return round(final_score, 1)
    
    def _generate_fast_reasoning(self, metadata: Dict[str, Any], preferences: Dict[str, Any], match_percentage: float) -> str:
        """Fast reasoning generation"""
        reasons = []
        
        if preferences.get('desired_program'):
            program = preferences['desired_program'].lower()
            if program in str(metadata.get('parent_course_name', '')).lower():
                reasons.append(f"Program: {preferences['desired_program']}")
        
        if (preferences.get('preferred_countries') and 
            metadata.get('country_name') in preferences.get('preferred_countries', [])):
            reasons.append(f"Location: {metadata['country_name']}")
        
        enhanced = metadata.get('enhanced_features', {})
        if enhanced.get('tuition_category') in ['very-affordable', 'affordable']:
            reasons.append("Affordable")
        
        if enhanced.get('rank_category', '').startswith('top-'):
            reasons.append("Highly ranked")
        
        reason_text = f"Match: {match_percentage:.0f}%"
        if reasons:
            reason_text += f" ({', '.join(reasons[:2])})"
        
        return reason_text
    
    def clear_cache(self):
        """Clear all caches to force rebuild"""
        try:
            import shutil
            if os.path.exists(self.cache_path):
                shutil.rmtree(self.cache_path)
            self.vector_store = None
            self._cache_loaded = False
            self.metadata_cache.clear()
            self.query_cache.clear()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def _calculate_enhanced_match(self, metadata: Dict[str, Any], preferences: Dict[str, Any]) -> Dict[str, float]:
        """Enhanced matching algorithm with detailed scoring components"""
        scores = {
            'program_relevance': 0.0,  # 50%
            'location_fit': 0.0,       # 20%
            'budget_match': 0.0,       # 15%
            'ranking_score': 0.0,      # 10%
            'additional_fit': 0.0      # 5%
        }
        
        # 1. Program Relevance (40%)
        if preferences.get('desired_program'):
            from difflib import SequenceMatcher
            program = preferences['desired_program'].lower()
            parent_course = str(metadata.get('parent_course_name', '')).lower()
            course_name = str(metadata.get('university_course_name', '')).lower()
            
            # Check exact matches first
            if program in parent_course or program in course_name:
                scores['program_relevance'] = 1.0
            else:
                # Use fuzzy matching for program names
                parent_ratio = SequenceMatcher(None, program, parent_course).ratio()
                course_ratio = SequenceMatcher(None, program, course_name).ratio()
                
                # Keywords matching
                keywords = set(program.split())
                parent_keywords = set(parent_course.split())
                course_keywords = set(course_name.split())
                keyword_match = len(keywords & (parent_keywords | course_keywords)) / len(keywords)
                
                # Combine fuzzy and keyword matching
                scores['program_relevance'] = max(
                    max(parent_ratio, course_ratio) * 0.7,  # Fuzzy match weight
                    keyword_match * 0.8  # Keyword match weight
                )
        
        # 2. Location Fit (20%)
        if preferences.get('preferred_countries'):
            country = metadata.get('country_name')
            if country in preferences['preferred_countries']:
                scores['location_fit'] = 1.0
            else:
                # Regional matching
                regions = {
                    'North America': ['USA', 'Canada'],
                    'Western Europe': ['UK', 'Ireland', 'Germany', 'France', 'Netherlands'],
                    'Asia Pacific': ['Australia', 'New Zealand', 'Singapore', 'Japan'],
                    'Nordic': ['Sweden', 'Norway', 'Denmark', 'Finland']
                }
                
                for region_countries in regions.values():
                    if (country in region_countries and 
                        any(c in preferences['preferred_countries'] for c in region_countries)):
                        scores['location_fit'] = 0.4
                        break
        
        # 3. Budget Match (15%)
        if preferences.get('max_tuition_usd'):
            tuition = metadata.get('university_course_tuition_usd')
            if tuition and str(tuition).isdigit():
                tuition_val = int(tuition)
                max_budget = int(preferences['max_tuition_usd'])
                
                if tuition_val <= max_budget:
                    # Better score for programs well within budget
                    ratio = tuition_val / max_budget
                    if ratio <= 0.7:
                        scores['budget_match'] = 1.0
                    else:
                        scores['budget_match'] = 0.9 - (ratio - 0.7)
                elif tuition_val <= max_budget * 1.2:
                    # Slight budget overflow
                    scores['budget_match'] = 0.3
        
        # 4. Ranking Score (15%)
        rank = metadata.get('university_global_rank')
        if rank and str(rank).isdigit():
            rank_val = int(rank)
            scores['ranking_score'] = max(0, min(1.0, (1000 - rank_val) / 1000))
        
        # 5. Additional Fit (10%)
        additional_score = 0.0
        
        # GRE requirement match
        if preferences.get('gre_preferred') is not None:
            gre_required = metadata.get('is_gre_required')
            if gre_required is not None and preferences['gre_preferred'] == gre_required:
                additional_score += 0.3
        
        # Scholarship availability
        if preferences.get('needs_scholarship'):
            scholarships = metadata.get('scholarship_count')
            if scholarships and int(scholarships) > 0:
                additional_score += 0.4
        
        # University type match
        if preferences.get('university_types'):
            if metadata.get('university_type') in preferences['university_types']:
                additional_score += 0.3
        
        scores['additional_fit'] = min(1.0, additional_score)
        
        return scores
    
    def _generate_enhanced_reasoning(self, metadata: Dict[str, Any], preferences: Dict[str, Any], match_percentage: float) -> str:
        """Generate detailed reasoning for the match"""
        reasons = []
        
        # Get detailed scores
        scores = self._calculate_enhanced_match(metadata, preferences)
        
        # Program match reasoning
        if scores['program_relevance'] > 0.8:
            reasons.append("Strong program match")
        elif scores['program_relevance'] > 0.5:
            reasons.append("Related program")
        
        # Location reasoning
        if scores['location_fit'] > 0.8:
            reasons.append(f"Preferred location: {metadata.get('country_name')}")
        elif scores['location_fit'] > 0.3:
            reasons.append("Similar region")
        
        # Budget reasoning
        if scores['budget_match'] > 0.8:
            reasons.append("Within budget")
        elif scores['budget_match'] > 0.3:
            reasons.append("Near budget")
        
        # Ranking reasoning
        rank = metadata.get('university_global_rank')
        if rank and str(rank).isdigit():
            rank_val = int(rank)
            if rank_val <= 100:
                reasons.append("Top 100 globally")
            elif rank_val <= 200:
                reasons.append("Top 200 globally")
        
        # Additional features
        if preferences.get('needs_scholarship') and metadata.get('scholarship_count', 0) > 0:
            reasons.append("Scholarships available")
        
        reason_text = f"Match: {match_percentage:.1f}%"
        if reasons:
            reason_text += f" ({', '.join(reasons[:3])})"
        
        return reason_text