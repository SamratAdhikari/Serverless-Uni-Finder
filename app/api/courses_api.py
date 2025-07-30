from typing import List, Optional, Dict, Any
from db.mongo import db
import json
from pathlib import Path


class CoursesAPI:
    """Business logic for course-related API operations"""
    
    @staticmethod
    async def get_course_list(
        limit: Optional[int] = None,
        skip: int = 0,
        course_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get course list data from courseList.json file.
        
        Returns courses with the following fields:
        - course_name
        - course_id
        - slug
        - app_count
        """
        
        # Read course list from JSON file
        course_list_path = Path(__file__).parent / "data" / "courseList.json"
        
        if not course_list_path.exists():
            return {
                "courses": [],
                "total_returned": 0,
                "error": "courseList.json not found"
            }
        
        with open(course_list_path, "r") as f:
            course_list = json.load(f)
        
        courses = course_list["data"]
        
        # Apply course name filter if provided
        if course_name:
            courses = [
                course for course in courses 
                if course_name.lower() in course["course_name"].lower()
            ]
        
        # Apply pagination
        total_courses = len(courses)
        if skip > 0:
            courses = courses[skip:]
        
        if limit:
            courses = courses[:limit]
        
        return {
            "courses": courses,
            "total_returned": len(courses),
            "total_available": total_courses,
            "filters_applied": {
                "course_name": course_name
            }
        }
    
    @staticmethod
    async def get_courses(
        limit: int = 100,
        skip: int = 0,
        level: Optional[int] = None,
        country: Optional[str] = None,
        course_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get course data with specific fields from the universities collection.
        
        Returns courses with the following fields:
        - university_name
        - university_global_rank
        - university_tuition_rating
        - location_name
        - program_level
        - university_type
        - country_name
        - scholarship_count
        - university_course_tuition_usd
        - university_course_name
        - course_program_label
        - parent_course_name
        - is_gre_required
        """
        
        # Build filter query
        filter_query = {}
        
        if level is not None:
            filter_query["level"] = level
        
        if country is not None:
            filter_query["country_name"] = {"$regex": country, "$options": "i"}
        
        if course_name is not None:
            filter_query["$or"] = [
                {"university_course_name": {"$regex": course_name, "$options": "i"}},
                {"parent_course_name": {"$regex": course_name, "$options": "i"}}
            ]
        
        # Define the fields to return
        projection = {
            "university_name": 1,
            "university_global_rank": 1,
            "university_tuition_rating": 1,
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
            "_id": 0  # Exclude MongoDB _id field
        }
        
        # Execute query
        cursor = db["universities"].find(filter_query, projection).skip(skip).limit(limit)
        courses = await cursor.to_list(length=limit)
        
        return {
            "courses": courses,
            "total_returned": len(courses),
            "filters_applied": {
                "level": level,
                "country": country,
                "course_name": course_name
            }
        }

    @staticmethod
    async def get_course_stats() -> Dict[str, Any]:
        """
        Get statistics about the courses data including:
        - Total courses by level
        - Countries distribution
        - University types distribution
        """
        
        # Get total courses by level
        pipeline_level = [
            {"$group": {"_id": "$level", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        level_stats = await db["universities"].aggregate(pipeline_level).to_list(None)
        
        # Get countries distribution
        pipeline_countries = [
            {"$group": {"_id": "$country_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        country_stats = await db["universities"].aggregate(pipeline_countries).to_list(None)
        
        # Get university types distribution
        pipeline_types = [
            {"$group": {"_id": "$university_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        type_stats = await db["universities"].aggregate(pipeline_types).to_list(None)
        
        # Get total count
        total_courses = await db["universities"].count_documents({})
        
        return {
            "total_courses": total_courses,
            "by_level": level_stats,
            "top_countries": country_stats,
            "by_university_type": type_stats
        }

    @staticmethod
    async def get_available_countries() -> Dict[str, List[str]]:
        """Get list of all available countries"""
        pipeline = [
            {"$group": {"_id": "$country_name"}},
            {"$sort": {"_id": 1}}
        ]
        countries = await db["universities"].aggregate(pipeline).to_list(None)
        return {"countries": [country["_id"] for country in countries if country["_id"]]}

    @staticmethod
    async def get_available_levels() -> Dict[str, List[int]]:
        """Get list of all available program levels"""
        pipeline = [
            {"$group": {"_id": "$level"}},
            {"$sort": {"_id": 1}}
        ]
        levels = await db["universities"].aggregate(pipeline).to_list(None)
        return {"levels": [level["_id"] for level in levels if level["_id"] is not None]} 