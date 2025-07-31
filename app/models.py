from beanie import Document
from typing import Optional

class University(Document):
    university_name: Optional[str] = None
    university_global_rank: Optional[str] = None
    university_tuition_rating: Optional[str] = None
    location_name: Optional[str] = None
    program_level: Optional[str] = None
    university_type: Optional[str] = None
    country_name: Optional[str] = None
    scholarship_count: Optional[str] = None
    university_course_tuition_usd: Optional[str] = None
    university_course_name: Optional[str] = None
    course_program_label: Optional[str] = None
    parent_course_name: Optional[str] = None
    is_gre_required: Optional[str] = None
    program_type: Optional[str] = None
    university_courses_credential: Optional[str] = None
    country_currency: Optional[str] = None

    class Settings:
        name = "universities"