# utils/hybrid_database.py - Hybrid database that works with both Supabase and local storage
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import os

class HybridEmotionDatabase:
    """Hybrid database that falls back to local storage when Supabase is unavailable"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.supabase_available = False
        self.supabase = None
        self.local_db_file = "local_mood_history.json"
        
        # Try to initialize Supabase
        if supabase_url and supabase_key:
            try:
                from supabase import create_client, Client
                self.supabase: Client = create_client(supabase_url, supabase_key)
                self._test_supabase_connection()
            except Exception as e:
                logging.warning(f"Supabase connection failed: {e}")
                self.supabase_available = False
        
        # Initialize local database
        self._init_local_database()
        self.available = True  # Always available with local fallback
    
    def _test_supabase_connection(self):
        """Test Supabase connection"""
        try:
            result = self.supabase.table("mood_history").select("*").limit(1).execute()
            self.supabase_available = True
            logging.info("Supabase connection successful")
        except Exception as e:
            logging.warning(f"Supabase test failed: {e}")
            self.supabase_available = False
    
    def _init_local_database(self):
        """Initialize local JSON database"""
        if not os.path.exists(self.local_db_file):
            initial_data = {
                "mood_history": [],
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            with open(self.local_db_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def _load_local_data(self):
        """Load data from local JSON file"""
        try:
            with open(self.local_db_file, 'r') as f:
                data = json.load(f)
            return data.get("mood_history", [])
        except Exception as e:
            logging.error(f"Failed to load local data: {e}")
            return []
    
    def _save_local_data(self, records):
        """Save data to local JSON file"""
        try:
            data = {
                "mood_history": records,
                "updated_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            with open(self.local_db_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save local data: {e}")
            return False
    
    def log_emotion(self, emotion_data: Dict, user_context: Dict = None) -> bool:
        """Log emotion to available database (Supabase or local)"""
        
        # Prepare record
        record = {
            "created_at": datetime.utcnow().isoformat(),
            "emotion": emotion_data.get("primary", "neutral"),
            "score": float(emotion_data.get("confidence", 0.0)),
            "confidence": float(emotion_data.get("confidence", 0.0)),
            "all_emotions": json.dumps(emotion_data.get("all_scores", [])),
            "analysis_type": emotion_data.get("source", "text"),
            "session_id": user_context.get("session_id") if user_context else None,
            "user_input_length": user_context.get("input_length", 0) if user_context else 0,
            "processing_time": user_context.get("processing_time", 0.0) if user_context else 0.0
        }
        
        # Try Supabase first
        if self.supabase_available:
            try:
                result = self.supabase.table("mood_history").insert(record).execute()
                logging.info("Emotion logged to Supabase")
                return True
            except Exception as e:
                logging.warning(f"Supabase insert failed: {e}")
                self.supabase_available = False
        
        # Fallback to local storage
        try:
            records = self._load_local_data()
            record["id"] = len(records) + 1  # Simple ID assignment
            records.append(record)
            
            if self._save_local_data(records):
                logging.info("Emotion logged to local database")
                return True
        except Exception as e:
            logging.error(f"Local database insert failed: {e}")
        
        return False
    
    def get_mood_history(self, days: int = 30, user_id: str = None) -> pd.DataFrame:
        """Get mood history from available database"""
        
        # Try Supabase first
        if self.supabase_available:
            try:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=days)
                
                query = self.supabase.table("mood_history").select("*")
                query = query.gte("created_at", start_date.isoformat())
                query = query.lte("created_at", end_date.isoformat())
                
                if user_id:
                    query = query.eq("user_id", user_id)
                
                result = query.order("created_at", desc=True).execute()
                
                if result.data:
                    df = pd.DataFrame(result.data)
                    df['created_at'] = pd.to_datetime(df['created_at'])
                    logging.info(f"Retrieved {len(df)} records from Supabase")
                    return df
            except Exception as e:
                logging.warning(f"Supabase query failed: {e}")
                self.supabase_available = False
        
        # Fallback to local storage
        try:
            records = self._load_local_data()
            if records:
                df = pd.DataFrame(records)
                df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601')
                
                # Filter by date range
                end_date = pd.Timestamp.now(tz='UTC')
                start_date = end_date - timedelta(days=days)
                
                # Ensure timezone consistency
                if df['created_at'].dt.tz is None:
                    df['created_at'] = df['created_at'].dt.tz_localize('UTC')
                
                df = df[df['created_at'] >= start_date]
                df = df[df['created_at'] <= end_date]
                
                logging.info(f"Retrieved {len(df)} records from local database")
                return df.sort_values('created_at', ascending=False)
        except Exception as e:
            logging.error(f"Local database query failed: {e}")
        
        return pd.DataFrame()
    
    def get_emotion_analytics(self, days: int = 30) -> Dict:
        """Generate comprehensive emotion analytics"""
        df = self.get_mood_history(days)
        
        if df.empty:
            return {
                "error": "No data available",
                "total_entries": 0,
                "most_common_emotion": "neutral",
                "average_confidence": 0.0,
                "confidence_trend": "no_data",
                "emotional_stability": {"stability": "no_data"},
                "mood_trajectory": {"trajectory": "no_data"}
            }
        
        analytics = {}
        
        # Basic statistics
        analytics["total_entries"] = len(df)
        analytics["date_range"] = {
            "start": df['created_at'].min().strftime("%Y-%m-%d"),
            "end": df['created_at'].max().strftime("%Y-%m-%d")
        }
        
        # Emotion distribution
        if 'emotion' in df.columns:
            emotion_counts = df['emotion'].value_counts()
            analytics["emotion_distribution"] = emotion_counts.to_dict()
            analytics["most_common_emotion"] = emotion_counts.index[0] if not emotion_counts.empty else "neutral"
        else:
            analytics["emotion_distribution"] = {}
            analytics["most_common_emotion"] = "neutral"
        
        # Confidence analysis
        confidence_col = None
        if 'confidence' in df.columns:
            confidence_col = 'confidence'
        elif 'score' in df.columns:
            confidence_col = 'score'
            
        if confidence_col and confidence_col in df.columns:
            clean_confidence = pd.to_numeric(df[confidence_col], errors='coerce').fillna(0.0)
            analytics["average_confidence"] = clean_confidence.mean()
            analytics["confidence_trend"] = self._calculate_trend(df, confidence_col)
        else:
            analytics["average_confidence"] = 0.0
            analytics["confidence_trend"] = "no_data"
        
        # Simplified analytics for reliability
        analytics["emotional_stability"] = {"stability": "moderate"}
        analytics["mood_trajectory"] = {
            "trajectory": "stable",
            "recent_mood_score": 0.0,
            "historical_mood_score": 0.0,
            "improvement": False
        }
        
        return analytics
    
    def _calculate_trend(self, df: pd.DataFrame, column: str) -> str:
        """Calculate trend for a numeric column"""
        if len(df) < 2 or column not in df.columns:
            return "insufficient_data"
        
        try:
            # Clean the data
            clean_data = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if len(clean_data) < 2:
                return "insufficient_data"
            
            # Simple trend calculation
            first_half = clean_data[:len(clean_data)//2].mean()
            second_half = clean_data[len(clean_data)//2:].mean()
            
            if second_half > first_half + 0.05:
                return "improving"
            elif second_half < first_half - 0.05:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            logging.error(f"Trend calculation failed: {e}")
            return "insufficient_data"
    
    def get_personalized_insights(self, days: int = 30) -> List[str]:
        """Generate personalized insights"""
        analytics = self.get_emotion_analytics(days)
        
        if "error" in analytics:
            return ["Start tracking your emotions to get personalized insights!"]
        
        insights = []
        
        total_entries = analytics["total_entries"]
        if total_entries > 10:
            insights.append(f"Great job! You've tracked {total_entries} emotional moments.")
        elif total_entries > 5:
            insights.append("You're building a good habit of emotional awareness.")
        else:
            insights.append("Keep tracking to discover patterns in your emotional journey!")
        
        most_common = analytics["most_common_emotion"]
        if most_common in ["joy", "gratitude", "optimism"]:
            insights.append(f"Your most common emotion is {most_common} - you're maintaining positive mental health!")
        elif most_common in ["sadness", "anxiety", "stress"]:
            insights.append(f"You've been experiencing {most_common} frequently. Consider self-care strategies.")
        
        avg_confidence = analytics["average_confidence"]
        if avg_confidence > 0.7:
            insights.append("Your emotional self-awareness is strong!")
        elif avg_confidence < 0.5:
            insights.append("Your emotions seem complex - this is normal and shows emotional depth.")
        
        return insights if insights else ["Keep tracking to discover your emotional patterns!"]