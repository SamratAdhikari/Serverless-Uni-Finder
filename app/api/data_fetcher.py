import aiohttp
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).parent.parent))
from db.mongo import db
from hash_utils import HashUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YocketDataFetcher:
    def __init__(self):
        self.base_url = "https://api.yocket.com"
        self.headers = {
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhbGdvcml0aG0iOiJFUzI1NiIsImlkIjoiN2QwZjAxYjEtYmIzNC00ODU2LTlmNTMtZmFhMGJlNzQzYjJmIiwiaWF0IjoxNzUzNTk0OTU3LCJleHAiOjE3NTYyMjQ3MDN9.rgq_zeju3SEkTpJjEWDCY45QJsLIjKXuDyyQKjGkXlM",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://yocket.com",
            "Referer": "https://yocket.com/",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        }
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.countries_collection = db["countries"]
        self.courses_collection = db["courses"]
        self.universities_collection = db["universities"]

    async def retry_request(self, session: aiohttp.ClientSession, url: str, payload: Dict = None, params: Dict = None, method: str = "GET", retries: int = 3) -> Optional[Dict]:

        for attempt in range(1, retries + 1):
            try:
                if method == "GET":
                    async with session.get(url, headers=self.headers, params=params) as response:
                        return await response.json()
                elif method == "POST":
                    async with session.post(url, headers=self.headers, json=payload) as response:
                        return await response.json()
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise e
                await asyncio.sleep(2 ** attempt)  
        return None

    async def save_courses_to_db(self, courses_data: List[Dict]):
        """Save courses to database with hash comparison"""
        timestamp = datetime.now(timezone.utc)
        
        # Check if data has changed using hash comparison
        has_changed = await HashUtils.has_data_changed(
            db, "courses", "all_courses", courses_data
        )
        
        if not has_changed:
            logger.info("[SKIP] Courses data unchanged - skipping database update")
            return False
        
        logger.info("[UPDATE] Courses data changed - updating database")
        
        # Clear existing data
        await self.courses_collection.delete_many({})
        
        # Prepare documents for bulk insert
        documents = []
        for course in courses_data:
            document = {
                "course_name": course["name"],
                "course_id": course["id"],
                "slug": course["slug"],
                "app_count": course.get("app_count", 0),
                "created_at": timestamp,
                "updated_at": timestamp,
                "source": "yocket_api"
            }
            documents.append(document)
        
        # Bulk insert all documents at once
        if documents:
            await self.courses_collection.insert_many(documents)
        
        # Store the new hash
        await HashUtils.store_hash(db, "courses", "all_courses", 
                                 HashUtils.generate_data_hash(courses_data), len(courses_data))
        
        logger.info(f"[SUCCESS] Saved {len(courses_data)} courses to database")
        return True

    async def save_universities_to_db(self, universities_data: List[Dict], level: int):
        """Save universities to database with hash comparison"""
        timestamp = datetime.now(timezone.utc)
        
        # Check if data has changed using hash comparison
        has_changed = await HashUtils.has_data_changed(
            db, "universities", f"level_{level}", universities_data
        )
        
        if not has_changed:
            logger.info(f"[SKIP] Universities data for level {level} unchanged - skipping database update")
            return False
        
        logger.info(f"[UPDATE] Universities data for level {level} changed - updating database")
        
        # Clear existing data for this level
        await self.universities_collection.delete_many({"level": level})
        
        # Prepare documents for bulk insert
        documents = []
        for university in universities_data:
            document = {
                **university,  
                "level": level,
                "created_at": timestamp,
                "updated_at": timestamp,
                "source": "yocket_api"
            }
            documents.append(document)
        
        # Bulk insert all documents at once
        if documents:
            await self.universities_collection.insert_many(documents)
        
        # Store the new hash
        await HashUtils.store_hash(db, "universities", f"level_{level}", 
                                 HashUtils.generate_data_hash(universities_data), len(universities_data))
        
        logger.info(f"[SUCCESS] Saved {len(universities_data)} universities for level {level} to database")
        return True

    async def fetch_all_courses(self) -> Dict[str, Any]:
        """Fetch all courses from Yocket API"""
        logger.info("Fetching all courses from Yocket API...")
        
        all_courses = []
        page = 1
        items_per_page = 50
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    logger.info(f"Fetching courses page {page}...")
                    
                    response = await self.retry_request(
                        session,
                        f"{self.base_url}/explore/parent-courses",
                        params={"page": page, "items": items_per_page}
                    )
                    
                    if response and response.get("data") and response["data"].get("results"):
                        courses = response["data"]["results"]
                        
                        if not courses:
                            logger.info("No more courses found, stopping pagination")
                            break
                        
                        all_courses.extend(courses)
                        logger.info(f"Fetched {len(courses)} courses from page {page}")
                        
                        if len(courses) < items_per_page:
                            logger.info("Reached last page, stopping pagination")
                            break
                        
                        page += 1
                    else:
                        logger.info("No data in response, stopping pagination")
                        break
                        
                except Exception as e:
                    logger.error(f"Error fetching page {page}: {e}")
                    break
        
        logger.info(f"Total courses fetched: {len(all_courses)}")
        
        # Save courses to courseList.json
        course_list_data = {
            "data": [
                {
                    "course_name": course["name"],
                    "course_id": course["id"],
                    "slug": course["slug"],
                    "app_count": course.get("app_count", 0)
                }
                for course in all_courses
            ]
        }
        
        course_list_path = self.data_dir / "courseList.json"
        with open(course_list_path, "w") as f:
            json.dump(course_list_data, f, indent=2)
        
        # Save courses to database with hash comparison
        db_updated = await self.save_courses_to_db(all_courses)
        
        logger.info(f"Saved {len(all_courses)} courses to courseList.json")
        
        return {
            "message": "Course list updated successfully",
            "courses_count": len(all_courses),
            "saved_to": "data/courseList.json",
            "database_updated": db_updated
        }

    async def fetch_single_level_data(self, level: int, course_names_array: List[str], country_names: str) -> Dict[str, Any]:
        """Fetch detailed data for a single level"""
        logger.info(f"=== Starting parallel fetch for Level {level} ===")
        all_results = []
        courses_per_request = 3
        
        async with aiohttp.ClientSession() as session:
            for start_index in range(0, len(course_names_array), courses_per_request):
                course_batch = ",".join(
                    course_names_array[start_index:start_index + courses_per_request]
                )
                
                current_page = 1
                logger.info(f"[Level {level}] Fetching data for courses: {course_batch} from page {current_page}")
                
                while True:
                    payload = {
                        "c_name": country_names,
                        "items": 21,
                        "level": level,
                        "page": current_page,
                        "pc_name": course_batch,
                        "query_source": "w",
                    }
                    
                    try:
                        response = await self.retry_request(
                            session,
                            f"{self.base_url}/explore/filter/7d0f01b1-bb34-4856-9f53-faa0be743b2f",
                            payload,
                            method="POST"
                        )
                        
                        logger.info(f"[Level {level}] Data fetched for page {current_page} of courses [{course_batch}]")
                        
                        if (response and response.get("data") and 
                            response["data"].get("result")):
                            
                            all_results.extend(response["data"]["result"])
                            
                            # Check if we've reached the last page
                            if current_page >= response["data"]["metadata"]["total_pages"]:
                                logger.info(f"[Level {level}] All pages fetched for courses [{course_batch}]. Moving to next batch.")
                                break
                            
                            logger.info(f"[Level {level}] All data of page {current_page} has been fetched, now fetching page {current_page + 1}")
                            current_page += 1
                        else:
                            logger.info(f"[Level {level}] No result found on page {current_page} for courses [{course_batch}]. Stopping.")
                            break
                            
                    except Exception as e:
                        logger.error(f"[Level {level}] Failed request payload: {payload}")
                        logger.error(f"[Level {level}] Error: {e}")
                        break
        
        # Save results for this level to JSON
        output_filename = f"Level{level}.json"
        output_path = self.data_dir / output_filename
        
        with open(output_path, "w") as f:
            json.dump({"result": all_results}, f, indent=2)
        
        # Save universities to database with hash comparison
        db_updated = await self.save_universities_to_db(all_results, level)
        
        logger.info(f"[Level {level}] Data successfully written to {output_filename} ({len(all_results)} results)")
        
        return {
            "level": level,
            "count": len(all_results),
            "saved_to": f"data/{output_filename}",
            "database_updated": db_updated
        }

    async def fetch_detailed_data(self, levels: List[int] = [1, 2, 3, 4, 5]) -> Dict[str, Any]:
        """Fetch detailed data for courses across multiple levels in parallel"""
        logger.info(f"Fetching detailed data for levels: {levels} in parallel...")
        
        # Read course list
        course_list_path = self.data_dir / "courseList.json"
        if not course_list_path.exists():
            raise Exception("courseList.json not found. Please run fetch_all_courses first.")
        
        with open(course_list_path, "r") as f:
            course_list = json.load(f)
        
        course_names_array = [course["course_name"] for course in course_list["data"]]
        
        # Read countries
        countries_path = self.data_dir / "countries.json"
        if not countries_path.exists():
            raise Exception("countries.json not found. Please run fetch_countries first.")
        
        with open(countries_path, "r") as f:
            countries_list = json.load(f)
        
        # Extract country names
        country_names_array = countries_list["data"].get("results", [])
        logger.info(f"Countries data structure: {type(country_names_array)}")
        logger.info(f"First few countries: {country_names_array[:3]}")
        
        # Handle different possible structures
        if isinstance(country_names_array, list):
            if country_names_array and isinstance(country_names_array[0], dict):
                country_names = ",".join([country["name"] for country in country_names_array])
            elif country_names_array and isinstance(country_names_array[0], str):
                country_names = ",".join(country_names_array)
            else:
                country_names = ""
        else:
            country_names = ""
        
        # Create tasks for parallel execution
        tasks = []
        for level in levels:
            task = self.fetch_single_level_data(level, course_names_array, country_names)
            tasks.append(task)
        
        # Execute all levels in parallel
        logger.info(f"Starting parallel execution for {len(levels)} levels...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        total_results = {}
        successful_levels = []
        failed_levels = []
        
        for i, result in enumerate(results):
            level = levels[i]
            if isinstance(result, Exception):
                logger.error(f"Level {level} failed: {result}")
                failed_levels.append(level)
            else:
                total_results[f"level_{level}"] = result
                successful_levels.append(level)
                logger.info(f"Level {level} completed successfully with {result['count']} results")
        
        logger.info(f"Parallel execution completed!")
        logger.info(f"Successful levels: {successful_levels}")
        if failed_levels:
            logger.warning(f"Failed levels: {failed_levels}")
        
        return {
            "message": "Detailed data successfully written for all levels",
            "levels_processed": successful_levels,
            "failed_levels": failed_levels,
            "results_summary": total_results
        }

    async def fetch_all_courses_and_data(self) -> Dict[str, Any]:
        """Fetch all courses and then detailed data for all levels"""
        logger.info("Starting complete data fetch process...")
        
        # Step 1: Fetch all courses
        courses_result = await self.fetch_all_courses()
        
        # Step 2: Fetch detailed data for all levels (1-5)
        detailed_result = await self.fetch_detailed_data(levels=[1, 2, 3, 4, 5])
        
        # Check if any database updates occurred
        any_db_updates = courses_result.get('database_updated', False)
        for level_result in detailed_result['results_summary'].values():
            if level_result.get('database_updated', False):
                any_db_updates = True
                break
        
        return {
            "message": "All courses fetched and detailed data saved for all levels",
            "courses_count": courses_result["courses_count"],
            "levels_processed": detailed_result["levels_processed"],
            "results_summary": detailed_result["results_summary"],
            "courses_saved_to": courses_result["saved_to"],
            "database_updated": any_db_updates
        }

# Global instance
data_fetcher = YocketDataFetcher() 