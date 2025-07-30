import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Create logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging for sequential fetcher
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'sequential_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from data_fetcher import data_fetcher
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

async def run_data_update():
    """Run the complete data update process"""
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting data fetch at {start_time}")
    
    try:
        # Run the complete data fetch process
        result = await data_fetcher.fetch_all_courses_and_data()
        
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        
        logger.info(f"Data fetch completed successfully!")
        logger.info(f"Summary:")
        logger.info(f"   - Courses fetched: {result['courses_count']}")
        logger.info(f"   - Levels processed: {result['levels_processed']}")
        logger.info(f"   - Database updates: {'[YES]' if result.get('database_updated', False) else '[NO] (data unchanged)'}")
        logger.info(f"   - Duration: {duration}")
        logger.info(f"   - Completed at: {end_time}")
        
        # Log hash comparison results
        if 'results_summary' in result:
            logger.info("Hash comparison results:")
            total_updates = 0
            total_skipped = 0
            for level_key, level_result in result['results_summary'].items():
                db_updated = level_result.get('database_updated', False)
                if db_updated:
                    status = "[UPDATED]"
                    total_updates += 1
                else:
                    status = "[SKIPPED] (no changes)"
                    total_skipped += 1
                logger.info(f"   - {level_key}: {status}")
            
            logger.info(f"[SUMMARY] {total_updates} levels updated, {total_skipped} levels skipped")
        
        return True
        
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        logger.error(f"Data fetch failed after {duration}: {e}")
        return False

async def run_sequential_fetcher():
    """Run data fetcher sequentially with 1-minute intervals"""
    logger.info("Starting sequential data fetcher...")
    logger.info("Will run every 1 minute after each complete fetch")
    logger.info("Press Ctrl+C to stop gracefully")
    
    fetch_count = 0
    
    try:
        while True:
            fetch_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"[FETCH] Fetch #{fetch_count}")
            logger.info(f"{'='*60}")
            
            # Run the data fetch
            success = await run_data_update()
            
            if success:
                logger.info("Waiting 1 minute before next fetch...")
                # Wait 1 minute (60 seconds) before next fetch - can be interrupted
                try:
                    await asyncio.wait_for(asyncio.sleep(60), timeout=60)
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue to next fetch
            else:
                logger.warning("Fetch failed. Waiting 2 minutes before retry...")
                # Wait 2 minutes on failure before retry - can be interrupted
                try:
                    await asyncio.wait_for(asyncio.sleep(120), timeout=120)
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue to next fetch
                    
    except asyncio.CancelledError:
        logger.info("Cancelled by user (Ctrl+C)")
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    
    logger.info("Sequential fetcher stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(run_sequential_fetcher())
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
