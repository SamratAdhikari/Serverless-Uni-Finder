import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

from db.mongo import db


async def test_courses_collection():
    """Test what's in the courses collection"""
    print("🧪 Testing courses collection...")
    
    try:
        # Check if courses collection exists and has data
        courses_count = await db["courses"].count_documents({})
        print(f"📊 Total courses in collection: {courses_count}")
        
        if courses_count > 0:
            # Get a sample course
            sample_course = await db["courses"].find_one()
            print(f"📝 Sample course structure:")
            print(f"   Keys: {list(sample_course.keys())}")
            print(f"   Sample: {sample_course}")
        else:
            print("❌ No courses found in collection")
            
        # Check if we have the course list data
        print(f"\n📋 Checking courseList.json structure...")
        import json
        course_list_path = Path(__file__).parent / "data" / "courseList.json"
        if course_list_path.exists():
            with open(course_list_path, "r") as f:
                course_list = json.load(f)
            print(f"✅ courseList.json found with {len(course_list['data'])} courses")
            print(f"📝 Sample course from JSON: {course_list['data'][0]}")
        else:
            print("❌ courseList.json not found")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_courses_collection()) 