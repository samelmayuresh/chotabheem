#!/usr/bin/env python3
"""
Database fix script - works with existing permissions
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
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRodnVnYnlvZ3B1dmNsamxjcGxmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk1MjQxODgsImV4cCI6MjA2NTEwMDE4OH0.zM9jr8RXEqTDSjFcFhOVBzpi-iSg8-BkxvdULwqAxww"

def test_and_fix_database():
    """Test database and insert sample data if needed"""
    try:
        # Create Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Try to read existing data
        logger.info("Testing database connection...")
        result = supabase.table("mood_history").select("*").limit(5).execute()
        
        if result.data:
            logger.info(f"✅ Found {len(result.data)} existing records")
            for record in result.data:
                logger.info(f"Record: {record}")
        else:
            logger.info("No existing records found")
        
        # Try to insert a test record
        logger.info("Inserting test record...")
        test_record = {
            "emotion": "joy",
            "score": 0.85,
            "created_at": "2025-01-17T10:00:00Z"
        }
        
        insert_result = supabase.table("mood_history").insert(test_record).execute()
        logger.info("✅ Test record inserted successfully")
        
        # Verify the insert worked
        verify_result = supabase.table("mood_history").select("*").order("created_at", desc=True).limit(1).execute()
        if verify_result.data:
            logger.info(f"✅ Verified: Latest record = {verify_result.data[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database operation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_and_fix_database()
    if success:
        print("✅ Database is working correctly!")
    else:
        print("❌ Database issues detected")
    sys.exit(0 if success else 1)