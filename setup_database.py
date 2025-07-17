#!/usr/bin/env python3
"""
Database setup script for Emotion AI application
Creates the necessary tables and handles database initialization
"""

import os
import sys
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
SUPABASE_URL = "https://thvugbyogpuvcljlcplf.supabase.co"
# Anon key from Supabase dashboard
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRodnVnYnlvZ3B1dmNsamxjcGxmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk1MjQxODgsImV4cCI6MjA2NTEwMDE4OH0.zM9jr8RXEqTDSjFcFhOVBzpi-iSg8-BkxvdULwqAxww"

def create_mood_history_table(supabase: Client):
    """Create the mood_history table with all necessary columns"""
    
    # SQL to create the table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS mood_history (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        emotion TEXT NOT NULL,
        score FLOAT DEFAULT 0.0,
        confidence FLOAT DEFAULT 0.0,
        all_emotions JSONB,
        analysis_type TEXT DEFAULT 'text',
        session_id TEXT,
        user_input_length INTEGER DEFAULT 0,
        processing_time FLOAT DEFAULT 0.0,
        weather_condition TEXT,
        time_of_day INTEGER,
        day_of_week INTEGER,
        user_location TEXT,
        user_id TEXT
    );
    """
    
    # Create indexes for better performance
    create_indexes_sql = [
        "CREATE INDEX IF NOT EXISTS idx_mood_history_created_at ON mood_history(created_at);",
        "CREATE INDEX IF NOT EXISTS idx_mood_history_emotion ON mood_history(emotion);",
        "CREATE INDEX IF NOT EXISTS idx_mood_history_user_id ON mood_history(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_mood_history_session_id ON mood_history(session_id);"
    ]
    
    try:
        # Execute table creation
        logger.info("Creating mood_history table...")
        result = supabase.rpc('exec_sql', {'sql': create_table_sql}).execute()
        logger.info("✅ Table created successfully")
        
        # Create indexes
        for index_sql in create_indexes_sql:
            logger.info(f"Creating index...")
            supabase.rpc('exec_sql', {'sql': index_sql}).execute()
        
        logger.info("✅ All indexes created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create table: {e}")
        return False

def test_connection(supabase: Client):
    """Test the database connection"""
    try:
        # Try to select from the table
        result = supabase.table("mood_history").select("*").limit(1).execute()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def insert_sample_data(supabase: Client):
    """Insert some sample data for testing"""
    sample_data = [
        {
            "emotion": "joy",
            "score": 0.85,
            "confidence": 0.85,
            "analysis_type": "text",
            "session_id": "sample_session_1"
        },
        {
            "emotion": "neutral",
            "score": 0.60,
            "confidence": 0.60,
            "analysis_type": "text",
            "session_id": "sample_session_1"
        },
        {
            "emotion": "excitement",
            "score": 0.78,
            "confidence": 0.78,
            "analysis_type": "audio",
            "session_id": "sample_session_2"
        }
    ]
    
    try:
        logger.info("Inserting sample data...")
        result = supabase.table("mood_history").insert(sample_data).execute()
        logger.info("✅ Sample data inserted successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to insert sample data: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Emotion AI Database...")
    
    # Check if we have the anon key
    if SUPABASE_KEY == "YOUR_ANON_KEY_HERE":
        print("❌ Please update the SUPABASE_KEY in this script with your actual anon key")
        print("You can find it in your Supabase dashboard under Settings > API")
        return False
    
    try:
        # Create Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Test connection
        if not test_connection(supabase):
            return False
        
        # Create table
        if not create_mood_history_table(supabase):
            return False
        
        # Insert sample data
        if not insert_sample_data(supabase):
            return False
        
        print("✅ Database setup completed successfully!")
        print("Your Emotion AI app is now ready to use with the new database.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)