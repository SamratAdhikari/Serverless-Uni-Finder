from fastapi import APIRouter, Query
from typing import Optional
from db.mongo import db
from .courses_api import CoursesAPI

router = APIRouter()

@router.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@router.get("/users")
async def get_users():
    users = await db["users"].find().to_list(100)
    return users

@router.get("/course-list")
async def get_course_list(
    limit: Optional[int] = Query(default=None, description="Number of courses to return (default: all)"),
    skip: Optional[int] = Query(default=0, description="Number of courses to skip"),
    course_name: Optional[str] = Query(default=None, description="Filter by course name")
):
    """
    Get course list data from courseList.json file.
    
    Returns courses with the following fields:
    - course_name
    - course_id
    - slug
    - app_count
    """
    return await CoursesAPI.get_course_list(
        limit=limit,
        skip=skip,
        course_name=course_name
    )

@router.get("/courses")
async def get_courses(
    limit: Optional[int] = Query(default=100, description="Number of courses to return"),
    skip: Optional[int] = Query(default=0, description="Number of courses to skip"),
    level: Optional[int] = Query(default=None, description="Filter by program level (1-5)"),
    country: Optional[str] = Query(default=None, description="Filter by country name"),
    course_name: Optional[str] = Query(default=None, description="Filter by course name")
):
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
    return await CoursesAPI.get_courses(
        limit=limit,
        skip=skip,
        level=level,
        country=country,
        course_name=course_name
    )

@router.get("/courses/stats")
async def get_course_stats():
    """
    Get statistics about the courses data including:
    - Total courses by level
    - Countries distribution
    - University types distribution
    """
    return await CoursesAPI.get_course_stats()

@router.get("/courses/countries")
async def get_available_countries():
    """Get list of all available countries"""
    return await CoursesAPI.get_available_countries()

@router.get("/courses/levels")
async def get_available_levels():
    """Get list of all available program levels"""
    return await CoursesAPI.get_available_levels()