import numpy as np
import pandas as pd
from bson import ObjectId
from typing import List, Dict, Any, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

# Only fetch the fields we actually need for recommendations
REQUIRED_FIELDS = {
    "_id": 1,
    "university_name": 1,
    "university_global_rank": 1,
    "location_name": 1,
    "program_level": 1,
    "university_type": 1,
    "country_name": 1,
    "scholarship_count": 1,
    "university_course_tuition_usd": 1,
    "university_course_name": 1,
    "course_program_label": 1,
    "parent_course_name": 1,
    "is_gre_required": 1,
    "program_type": 1,
    "university_courses_credential": 1,
    "country_currency": 1
}

def convert_to_string_safe(value: Any, field_name: str = "") -> str:
    """Optimized conversion with early returns"""
    # Fast path for already-string values
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in {'nan', 'null', 'none', '', 'unknown'}:
            return cleaned
        return "unknown"
    
    # Fast path for None/empty
    if value is None or value == "":
        return "unknown"
    
    # Handle numeric types
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            if pd.isna(value) or np.isnan(value):
                return "unknown"
            
            # Optimized numeric field handling
            if field_name in {'university_global_rank', 'program_level', 'scholarship_count'}:
                return str(int(value))
            elif field_name == 'university_course_tuition_usd':
                return str(int(value)) if value > 0 else "unknown"
            else:
                return str(value)
        except (ValueError, OverflowError):
            return "unknown"
    
    # Handle boolean
    if isinstance(value, bool):
        return "yes" if value else "no"
    
    # Handle ObjectId
    if isinstance(value, ObjectId):
        return str(value)
    
    # Default conversion
    try:
        result = str(value) if value else "unknown"
        return result if result != 'nan' else "unknown"
    except:
        return "unknown"

def document_helper(document) -> dict:
    """Optimized document conversion with batch processing"""
    # Pre-compute all conversions at once
    result = {}
    for field_name in REQUIRED_FIELDS.keys():
        if field_name in document:
            result[field_name] = convert_to_string_safe(document[field_name], field_name)
        else:
            result[field_name] = "unknown"
    
    return result

async def retrieve_documents(collection, batch_size: int = 1000) -> List[Dict[str, Any]]:
    """Optimized retrieval with projection and batching"""
    documents = []
    try:
        logger.info("Fetching documents from MongoDB with projection...")
        
        # Use projection to only fetch required fields
        cursor = collection.find({}, REQUIRED_FIELDS)
        
        # Process in batches to reduce memory usage
        batch = []
        async for doc in cursor:
            batch.append(document_helper(doc))
            
            if len(batch) >= batch_size:
                documents.extend(batch)
                batch = []
                # Optional: Add a small delay to prevent overwhelming the system
                if len(documents) % 5000 == 0:
                    logger.info(f"Processed {len(documents)} documents...")
                    await asyncio.sleep(0.01)  # 10ms delay every 5000 docs
        
        # Add remaining documents
        if batch:
            documents.extend(batch)
        
        logger.info(f"Retrieved {len(documents)} documents from MongoDB")
        return documents
        
    except Exception as e:
        logger.error(f"Error retrieving documents from MongoDB: {e}")
        return []

# Optional: Add caching for frequent queries
class DocumentCache:
    def __init__(self, ttl_seconds: int = 3600):  # 1 hour cache
        self._cache = {}
        self._timestamps = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        import time
        if key in self._cache:
            if time.time() - self._timestamps[key] < self.ttl:
                return self._cache[key]
            else:
                # Expired
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: List[Dict[str, Any]]):
        import time
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def clear(self):
        self._cache.clear()
        self._timestamps.clear()

# Global cache instance
document_cache = DocumentCache()

async def retrieve_documents_cached(collection, batch_size: int = 1000, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Cached version of document retrieval"""
    cache_key = "all_documents"
    
    if use_cache:
        cached_result = document_cache.get(cache_key)
        if cached_result:
            logger.info(f"Retrieved {len(cached_result)} documents from cache")
            return cached_result
    
    # Fetch from database
    documents = await retrieve_documents(collection, batch_size)
    
    if use_cache and documents:
        document_cache.set(cache_key, documents)
    
    return documents