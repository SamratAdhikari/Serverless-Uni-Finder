import hashlib
import json
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class HashUtils:
    """Utility class for hash-based data comparison"""
    
    @staticmethod
    def generate_data_hash(data: List[Dict]) -> str:
        """
        Generate a hash for a list of data objects.
        Excludes timestamp fields to focus on actual data changes.
        """
        if not data:
            return hashlib.md5("empty".encode()).hexdigest()
        
        # Create a normalized version of the data (excluding timestamps)
        normalized_data = []
        for item in data:
            normalized_item = {}
            for key, value in item.items():
                # Skip timestamp fields and internal MongoDB fields
                if key not in ['created_at', 'updated_at', '_id', 'source']:
                    normalized_item[key] = value
            normalized_data.append(normalized_item)
        
        # Sort by a consistent key to ensure hash stability
        if normalized_data and 'course_name' in normalized_data[0]:
            normalized_data.sort(key=lambda x: x.get('course_name', ''))
        elif normalized_data and 'university_name' in normalized_data[0]:
            normalized_data.sort(key=lambda x: x.get('university_name', ''))
        
        # Convert to JSON string and hash
        json_str = json.dumps(normalized_data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    @staticmethod
    def generate_collection_hash(collection_data: List[Dict], collection_type: str) -> str:
        """
        Generate a hash for a specific collection type.
        """
        return HashUtils.generate_data_hash(collection_data)
    
    @staticmethod
    async def get_stored_hash(db, collection_name: str, hash_key: str) -> Optional[str]:
        """
        Get the stored hash for a collection from the database.
        """
        try:
            hash_doc = await db["data_hashes"].find_one({"collection": collection_name, "hash_key": hash_key})
            return hash_doc.get("hash") if hash_doc else None
        except Exception as e:
            logger.error(f"Error getting stored hash for {collection_name}: {e}")
            return None
    
    @staticmethod
    async def store_hash(db, collection_name: str, hash_key: str, hash_value: str, data_count: int):
        """
        Store the hash for a collection in the database.
        """
        try:
            timestamp = datetime.now(timezone.utc)
            hash_doc = {
                "collection": collection_name,
                "hash_key": hash_key,
                "hash": hash_value,
                "data_count": data_count,
                "updated_at": timestamp
            }
            
            # Upsert the hash document
            await db["data_hashes"].update_one(
                {"collection": collection_name, "hash_key": hash_key},
                {"$set": hash_doc},
                upsert=True
            )
            
            logger.info(f"Stored hash for {collection_name} ({hash_key}): {hash_value[:8]}... ({data_count} items)")
            
        except Exception as e:
            logger.error(f"Error storing hash for {collection_name}: {e}")
    
    @staticmethod
    async def has_data_changed(db, collection_name: str, hash_key: str, new_data: List[Dict]) -> bool:
        """
        Check if data has changed by comparing hashes.
        Returns True if data has changed, False if it's the same.
        """
        try:
            # Generate hash for new data
            new_hash = HashUtils.generate_collection_hash(new_data, collection_name)
            
            # Get stored hash
            stored_hash = await HashUtils.get_stored_hash(db, collection_name, hash_key)
            
            if stored_hash is None:
                logger.info(f"No stored hash found for {collection_name} ({hash_key}), treating as changed")
                return True
            
            # Compare hashes
            if new_hash == stored_hash:
                logger.info(f"Data unchanged for {collection_name} ({hash_key})")
                return False
            else:
                logger.info(f"Data changed for {collection_name} ({hash_key})")
                logger.info(f"  Old hash: {stored_hash[:8]}...")
                logger.info(f"  New hash: {new_hash[:8]}...")
                return True
                
        except Exception as e:
            logger.error(f"Error checking data changes for {collection_name}: {e}")
            return True  # Treat as changed on error to be safe 