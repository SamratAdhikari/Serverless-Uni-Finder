from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import uvicorn
import logging
import os
import asyncio
from contextlib import asynccontextmanager

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from models import University
from langchain_helper import UniversityRecommendationService
from mongodb_helper import retrieve_documents_cached

# Load env vars
load_dotenv()

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
recommendation_service = None
mongo_client = None
app_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized lifespan management"""
    global recommendation_service, mongo_client, app_ready
    
    # Startup
    try:
        MONGO_URL = os.getenv("MONGO_URL")
        DATABASE_NAME = os.getenv("DATABASE_NAME")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        logger.info("Starting optimized startup...")
        
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        mongo_client = AsyncIOMotorClient(MONGO_URL)
        db = mongo_client[DATABASE_NAME]

        # Initialize Beanie
        logger.info("Initializing Beanie...")
        await init_beanie(database=db, document_models=[University])

        # Initialize Recommendation Service (this will try to load cached vector store)
        logger.info("Initializing Recommendation Service...")
        recommendation_service = UniversityRecommendationService(
            gemini_api_key=GEMINI_API_KEY,
            cache_dir="vector_store_cache"
        )

        # Load data in background if needed
        collection = db["universities"]
        
        # Start background task to load/update data if needed
        asyncio.create_task(load_data_background(collection))
        
        app_ready = True
        logger.info("Application startup completed!")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown
        if mongo_client:
            mongo_client.close()
            logger.info("MongoDB connection closed")

async def load_data_background(collection):
    """Background task to load data without blocking startup"""
    global recommendation_service
    try:
        logger.info("Loading data in background...")
        
        # Use cached retrieval
        documents = await retrieve_documents_cached(collection, batch_size=2000, use_cache=True)
        
        if documents:
            logger.info(f"Processing {len(documents)} documents...")
            recommendation_service.load_data_from_mongodb(documents)
            logger.info("Background data loading completed")
        else:
            logger.warning("No documents found in database")
            
    except Exception as e:
        logger.error(f"Background data loading failed: {e}")

# FastAPI app with optimized lifespan
app = FastAPI(
    title="University Recommendation API",
    description="AI-powered university course recommendations",
    version="2.0.0",
    lifespan=lifespan
)

# Optimized CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow needed methods
    allow_headers=["*"],
)

# Optimized Request Models
class RecommendationRequest(BaseModel):
    desired_program: Optional[str] = None
    preferred_countries: Optional[List[str]] = None
    program_level: Optional[str] = None
    max_tuition_usd: Optional[int] = None
    top_k: Optional[int] = 10
    
    class Config:
        # Optimize validation
        validate_assignment = True
        use_enum_values = True

class RecommendationResponse(BaseModel):
    university_name: str
    course_name: str
    program_label: str
    parent_course: str
    location: str
    country: str
    global_rank: str
    tuition_usd: str
    university_type: str
    scholarship_count: str
    is_gre_required: str
    similarity_score: float
    match_percentage: float
    reasoning: str
    relevance_score: float
    
    class Config:
        # Optimize serialization
        json_encoders = {
            float: lambda v: round(v, 3)  # Limit float precision
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "University Recommendation API v2.0",
        "status": "ready" if app_ready else "initializing",
        "service_ready": recommendation_service is not None and recommendation_service.vector_store is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "app_ready": app_ready,
        "recommendation_service": recommendation_service is not None,
        "vector_store_ready": recommendation_service.vector_store is not None if recommendation_service else False,
        "mongodb_connected": mongo_client is not None
    }

@app.post("/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(request: RecommendationRequest):
    """Optimized recommendations endpoint"""
    if not app_ready:
        raise HTTPException(status_code=503, detail="Service is still initializing, please try again in a moment")
    
    if not recommendation_service:
        raise HTTPException(status_code=503, detail="Recommendation service not available")
    
    if not recommendation_service.vector_store:
        raise HTTPException(status_code=503, detail="Vector store not ready, please try again in a moment")
    
    try:
        # Validate and limit top_k
        top_k = min(request.top_k or 10, 50)  # Limit to prevent excessive processing
        
        # Convert request to preferences dict
        preferences = request.dict(exclude_none=True, exclude={'top_k'})
        
        # Get recommendations
        recommendations = recommendation_service.get_recommendations(preferences, top_k=top_k)
        
        # Convert to response format efficiently
        response_data = []
        for rec in recommendations:
            response_data.append(RecommendationResponse(
                university_name=rec.get('university_name', 'Unknown'),
                course_name=rec.get('course_name', 'Unknown'),
                program_label=rec.get('program_label', 'Unknown'),
                parent_course=rec.get('parent_course', 'Unknown'),
                location=rec.get('location', 'Unknown'),
                country=rec.get('country', 'Unknown'),
                global_rank=rec.get('global_rank', 'Unknown'),
                tuition_usd=rec.get('tuition_usd', 'Unknown'),
                university_type=rec.get('university_type', 'Unknown'),
                scholarship_count=rec.get('scholarship_count', 'Unknown'),
                is_gre_required=rec.get('is_gre_required', 'Unknown'),
                similarity_score=rec.get('similarity_score', 0.0),
                match_percentage=rec.get('match_percentage', 0.0),
                reasoning=rec.get('reasoning', ''),
                relevance_score=rec.get('relevance_score', 0.0)
            ))
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        doc_count = await University.count() if app_ready else 0
        
        cache_stats = {}
        if recommendation_service:
            cache_stats = {
                "metadata_cache_size": len(recommendation_service.metadata_cache),
                "query_cache_size": len(recommendation_service.query_cache)
            }
        
        return {
            "app_ready": app_ready,
            "mongodb_documents": doc_count,
            "recommendation_service_ready": recommendation_service is not None,
            "vector_store_ready": recommendation_service.vector_store is not None if recommendation_service else False,
            **cache_stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh-data")
async def refresh_data(background_tasks: BackgroundTasks):
    """Manually refresh data from MongoDB"""
    if not app_ready or not mongo_client:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Clear caches
        if recommendation_service:
            recommendation_service.metadata_cache.clear()
            recommendation_service.query_cache.clear()
        
        # Refresh in background
        collection = mongo_client[os.getenv("DATABASE_NAME")]["universities"]
        background_tasks.add_task(load_data_background, collection)
        
        return {"message": "Data refresh initiated in background"}
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add endpoint to clear caches
@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches"""
    if recommendation_service:
        recommendation_service.metadata_cache.clear()
        recommendation_service.query_cache.clear()
        
        # Clear document cache
        from mongodb_helper import document_cache
        document_cache.clear()
        
        return {"message": "All caches cleared successfully"}
    else:
        raise HTTPException(status_code=503, detail="Service not available")

if __name__ == "__main__":
    SERVER_URL = os.getenv("SERVER_URL", "127.0.0.1")
    PORT = int(os.getenv("PORT", 8000))
    ENV = os.getenv("ENV", "dev")
    
    # Optimized uvicorn configuration
    config = {
        "host": SERVER_URL,
        "port": PORT,
        "reload": ENV == "dev",
        "workers": 1,  # Single worker for development
        "loop": "uvloop" if ENV == "prod" else "asyncio",  # Use uvloop in production
        "log_level": "info" if ENV == "prod" else "debug",
        "access_log": ENV != "prod",  # Disable access logs in production for performance
    }
    uvicorn.run("main:app", **config)
    