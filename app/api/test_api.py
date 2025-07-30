import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

from courses_api import CoursesAPI


async def test_courses_api():
    """Test the CoursesAPI methods"""
    print("🧪 Testing CoursesAPI methods...")
    
    try:
        # Test 1: Get course stats
        print("\n📊 Testing get_course_stats()...")
        stats = await CoursesAPI.get_course_stats()
        print(f"✅ Stats retrieved successfully!")
        print(f"   Total courses: {stats['total_courses']}")
        print(f"   Levels available: {[level['_id'] for level in stats['by_level']]}")
        
        # Test 2: Get available countries
        print("\n🌍 Testing get_available_countries()...")
        countries = await CoursesAPI.get_available_countries()
        print(f"✅ Countries retrieved successfully!")
        print(f"   Total countries: {len(countries['countries'])}")
        print(f"   First 5 countries: {countries['countries'][:5]}")
        
        # Test 3: Get available levels
        print("\n📚 Testing get_available_levels()...")
        levels = await CoursesAPI.get_available_levels()
        print(f"✅ Levels retrieved successfully!")
        print(f"   Available levels: {levels['levels']}")
        
        # Test 4: Get courses with filters
        print("\n🎓 Testing get_courses()...")
        courses = await CoursesAPI.get_courses(limit=5)
        print(f"✅ Courses retrieved successfully!")
        print(f"   Total returned: {courses['total_returned']}")
        if courses['courses']:
            print(f"   Sample course: {courses['courses'][0].get('university_name', 'N/A')}")
        
        # Test 5: Get courses with level filter
        print("\n🎓 Testing get_courses() with level filter...")
        level_courses = await CoursesAPI.get_courses(limit=3, level=1)
        print(f"Level 1 courses retrieved successfully!")
        print(f"Total returned: {level_courses['total_returned']}")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_courses_api()) 