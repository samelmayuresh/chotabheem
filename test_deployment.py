#!/usr/bin/env python3
"""
Test script to verify deployment configuration
"""

import streamlit as st
from config import api_config
from utils.hybrid_database import HybridEmotionDatabase

def main():
    st.title("🔧 Deployment Configuration Test")
    
    # Test configuration loading
    st.subheader("Configuration Status")
    
    config_status = {
        "Supabase URL": api_config.supabase_url,
        "Supabase Key": "✅ Configured" if api_config.supabase_key else "❌ Missing",
        "OpenRouter Key": "✅ Configured" if api_config.openrouter_key else "❌ Missing",
        "Tenor API Key": "✅ Configured" if api_config.tenor_key else "❌ Missing",
        "Weather API Key": "✅ Configured" if api_config.weather_key else "❌ Missing"
    }
    
    for key, value in config_status.items():
        if key == "Supabase URL":
            st.write(f"**{key}**: {value}")
        else:
            st.write(f"**{key}**: {value}")
    
    # Test database connection
    st.subheader("Database Connection Test")
    
    try:
        db = HybridEmotionDatabase(
            supabase_url=api_config.supabase_url,
            supabase_key=api_config.supabase_key
        )
        
        if db.supabase_available:
            st.success("✅ Supabase connection successful!")
            
            # Test data retrieval
            df = db.get_mood_history(days=7)
            st.write(f"📊 Found {len(df)} mood records in the last 7 days")
            
            if not df.empty:
                st.write("Recent records:")
                st.dataframe(df.head())
        else:
            st.warning("⚠️ Using local storage fallback")
            
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
    
    # Show environment info
    st.subheader("Environment Information")
    import os
    st.write(f"**Python Path**: {os.getcwd()}")
    st.write(f"**Streamlit Version**: {st.__version__}")

if __name__ == "__main__":
    main()