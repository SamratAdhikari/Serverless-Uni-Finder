# MongoDB Setup for Uni Finder Backend

## 1. Install MongoDB

### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install mongodb
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

### macOS:
```bash
brew install mongodb-community
brew services start mongodb-community
```

### Windows:
Download and install from [MongoDB Download Center](https://www.mongodb.com/try/download/community)

## 2. Environment Configuration

Create a `.env` file in the root directory:

```env
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=uni_finder_db

# Optional: If using MongoDB Atlas or remote MongoDB
# MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net
# MONGO_DB_NAME=uni_finder_db
```

## 3. Database Structure

The application will create a collection called `university_data` with the following document types:

### Document Types:
1. **countries** - Country data from Yocket API
2. **courses** - Course list data
3. **level_1_data** - Detailed data for level 1
4. **level_2_data** - Detailed data for level 2
5. **level_3_data** - Detailed data for level 3
6. **level_4_data** - Detailed data for level 4
7. **level_5_data** - Detailed data for level 5

### Document Structure:
```json
{
  "_id": "ObjectId",
  "data_type": "string",
  "data": "object",
  "metadata": {
    "count": "number",
    "file_path": "string",
    "level": "number (for level data)"
  },
  "created_at": "datetime",
  "updated_at": "datetime",
  "source": "yocket_api"
}
```

## 4. Testing the Setup

Run the test script to verify MongoDB connection:

```bash
cd app/api
python3 test_fetch.py
```

## 5. Data Tracking

The system automatically tracks:
- When data was fetched
- When data was last updated
- Source of the data (Yocket API)
- File paths where JSON data is saved
- Count of records for each data type

## 6. Data Updates

The system uses upsert operations, so:
- New data will be inserted
- Existing data will be updated with new timestamps
- No duplicate documents will be created 