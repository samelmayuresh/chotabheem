# utils/database.py - Enhanced database operations with analytics
import streamlit as st
from supabase import create_client, Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json

class EmotionDatabase:
    """Enhanced database operations for emotion tracking and analytics"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        try:
            self.supabase: Client = create_client(supabase_url, supabase_key)
            self.available = True
            self._test_connection()
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            self.available = False
    
    def _test_connection(self):
        """Test database connection"""
        try:
            # Try to fetch one record to test connection
            self.supabase.table("mood_history").select("*").limit(1).execute()
        except Exception as e:
            logging.warning(f"Database test failed: {e}")
            self.available = False
    
    def log_emotion(self, emotion_data: Dict, user_context: Dict = None) -> bool:
        """Enhanced emotion logging with context"""
        if not self.available:
            return False
        
        try:
            # Prepare comprehensive emotion record
            record = {
                "created_at": datetime.utcnow().isoformat(),
                "emotion": emotion_data.get("primary", "neutral"),
                "confidence": float(emotion_data.get("confidence", 0.0)),
                "all_emotions": json.dumps(emotion_data.get("all_scores", [])),
                "analysis_type": emotion_data.get("source", "text"),  # text, audio, combined
                "session_id": user_context.get("session_id") if user_context else None,
                "user_input_length": user_context.get("input_length", 0) if user_context else 0,
                "processing_time": user_context.get("processing_time", 0.0) if user_context else 0.0
            }
            
            # Add optional context fields
            if user_context:
                record.update({
                    "weather_condition": user_context.get("weather"),
                    "time_of_day": datetime.now().hour,
                    "day_of_week": datetime.now().weekday(),
                    "user_location": user_context.get("location")
                })
            
            result = self.supabase.table("mood_history").insert(record).execute()
            return True
            
        except Exception as e:
            logging.error(f"Failed to log emotion: {e}")
            return False
    
    def get_mood_history(self, days: int = 30, user_id: str = None) -> pd.DataFrame:
        """Get mood history with enhanced filtering"""
        if not self.available:
            return pd.DataFrame()
        
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Build query
            query = self.supabase.table("mood_history").select("*")
            query = query.gte("created_at", start_date.isoformat())
            query = query.lte("created_at", end_date.isoformat())
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.order("created_at", desc=True).execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                df['created_at'] = pd.to_datetime(df['created_at'])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Failed to fetch mood history: {e}")
            return pd.DataFrame()
    
    def get_emotion_analytics(self, days: int = 30) -> Dict:
        """Generate comprehensive emotion analytics"""
        df = self.get_mood_history(days)
        
        if df.empty:
            return {"error": "No data available"}
        
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
        
        # Confidence analysis (using 'score' column if 'confidence' doesn't exist)
        if 'confidence' in df.columns:
            confidence_col = 'confidence'
        elif 'score' in df.columns:
            confidence_col = 'score'
        else:
            confidence_col = None
            
        if confidence_col:
            analytics["average_confidence"] = df[confidence_col].mean()
            analytics["confidence_trend"] = self._calculate_trend(df, confidence_col)
        else:
            analytics["average_confidence"] = 0.0
            analytics["confidence_trend"] = "no_data"
        
        # Temporal patterns
        analytics["temporal_patterns"] = self._analyze_temporal_patterns(df)
        
        # Emotional stability
        analytics["emotional_stability"] = self._calculate_emotional_stability(df)
        
        # Weekly patterns
        analytics["weekly_patterns"] = self._analyze_weekly_patterns(df)
        
        # Mood trajectory
        analytics["mood_trajectory"] = self._calculate_mood_trajectory(df)
        
        return analytics
    
    def _calculate_trend(self, df: pd.DataFrame, column: str) -> str:
        """Calculate trend for a numeric column"""
        if len(df) < 2 or column not in df.columns:
            return "insufficient_data"
        
        # Clean the data - remove None values and convert to numeric
        clean_data = df[column].dropna()
        clean_data = pd.to_numeric(clean_data, errors='coerce').dropna()
        
        if len(clean_data) < 2:
            return "insufficient_data"
        
        # Use linear regression to determine trend
        x = np.arange(len(clean_data))
        y = clean_data.values
        
        try:
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "declining"
            else:
                return "stable"
        except (np.linalg.LinAlgError, TypeError, ValueError):
            return "insufficient_data"
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns by time of day and day of week"""
        patterns = {}
        
        # Hour of day analysis
        if 'time_of_day' in df.columns and 'emotion' in df.columns:
            hourly_emotions = df.groupby('time_of_day')['emotion'].apply(list).to_dict()
            patterns["hourly_patterns"] = {
                hour: self._get_dominant_emotion(emotions) 
                for hour, emotions in hourly_emotions.items()
            }
        
        # Day of week analysis
        if 'day_of_week' in df.columns and 'emotion' in df.columns:
            daily_emotions = df.groupby('day_of_week')['emotion'].apply(list).to_dict()
            patterns["daily_patterns"] = {
                day: self._get_dominant_emotion(emotions) 
                for day, emotions in daily_emotions.items()
            }
        
        return patterns
    
    def _calculate_emotional_stability(self, df: pd.DataFrame) -> Dict:
        """Calculate emotional stability metrics"""
        if len(df) < 5:
            return {"stability": "insufficient_data"}
        
        # Calculate emotion diversity (Shannon entropy)
        if 'emotion' in df.columns:
            emotion_probs = df['emotion'].value_counts(normalize=True)
            entropy = -sum(p * np.log2(p) for p in emotion_probs if p > 0)
        else:
            entropy = 0.0
        
        # Calculate confidence stability
        confidence_col = 'confidence' if 'confidence' in df.columns else 'score'
        if confidence_col in df.columns:
            confidence_data = pd.to_numeric(df[confidence_col], errors='coerce').dropna()
            confidence_std = confidence_data.std() if len(confidence_data) > 0 else 0.0
        else:
            confidence_std = 0.0
        
        # Determine stability level
        if entropy < 1.5 and confidence_std < 0.2:
            stability = "high"
        elif entropy < 2.5 and confidence_std < 0.3:
            stability = "moderate"
        else:
            stability = "low"
        
        return {
            "stability": stability,
            "emotion_diversity": entropy,
            "confidence_variability": confidence_std
        }
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze weekly emotional patterns"""
        if len(df) < 7:
            return {"pattern": "insufficient_data"}
        
        # Group by week and analyze
        df['week'] = df['created_at'].dt.isocalendar().week
        confidence_col = 'confidence' if 'confidence' in df.columns else 'score'
        
        agg_dict = {
            'emotion': lambda x: x.mode().iloc[0] if not x.mode().empty else 'neutral'
        }
        if confidence_col in df.columns:
            agg_dict[confidence_col] = 'mean'
        
        weekly_data = df.groupby('week').agg(agg_dict).to_dict()
        
        return {
            "weekly_dominant_emotions": weekly_data.get('emotion', {}),
            "weekly_confidence": weekly_data.get(confidence_col, {})
        }
    
    def _calculate_mood_trajectory(self, df: pd.DataFrame) -> Dict:
        """Calculate overall mood trajectory"""
        if len(df) < 3 or 'emotion' not in df.columns:
            return {"trajectory": "insufficient_data"}
        
        # Define emotion valence scores
        valence_scores = {
            "joy": 0.8, "gratitude": 0.7, "optimism": 0.6, "excitement": 0.7,
            "love": 0.8, "pride": 0.5, "relief": 0.4, "approval": 0.3,
            "neutral": 0.0, "curiosity": 0.1, "surprise": 0.0,
            "sadness": -0.6, "anger": -0.5, "fear": -0.7, "anxiety": -0.6,
            "stress": -0.4, "disappointment": -0.5, "grief": -0.8,
            "nervousness": -0.3, "annoyance": -0.3, "confusion": -0.2
        }
        
        # Calculate valence scores for each entry
        df_copy = df.copy()
        df_copy['valence'] = df_copy['emotion'].map(valence_scores).fillna(0.0)
        
        # Clean valence data
        valence_data = pd.to_numeric(df_copy['valence'], errors='coerce').fillna(0.0)
        df_copy['valence'] = valence_data
        
        # Calculate trend
        trend = self._calculate_trend(df_copy, 'valence')
        
        # Calculate recent vs historical average
        try:
            recent_valence = df_copy.head(7)['valence'].mean()  # Last 7 entries
            historical_valence = df_copy.tail(len(df_copy) - 7)['valence'].mean() if len(df_copy) > 7 else 0.0
            
            # Handle NaN values
            recent_valence = recent_valence if not pd.isna(recent_valence) else 0.0
            historical_valence = historical_valence if not pd.isna(historical_valence) else 0.0
            
        except (TypeError, ValueError):
            recent_valence = 0.0
            historical_valence = 0.0
        
        return {
            "trajectory": trend,
            "recent_mood_score": recent_valence,
            "historical_mood_score": historical_valence,
            "improvement": recent_valence > historical_valence
        }
    
    def _get_dominant_emotion(self, emotions: List[str]) -> str:
        """Get the most common emotion from a list"""
        if not emotions:
            return "neutral"
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return max(emotion_counts, key=emotion_counts.get)
    
    def get_personalized_insights(self, days: int = 30) -> List[str]:
        """Generate personalized insights based on user's data"""
        analytics = self.get_emotion_analytics(days)
        
        if "error" in analytics:
            return ["Start tracking your emotions to get personalized insights!"]
        
        insights = []
        
        # Frequency insights
        total_entries = analytics["total_entries"]
        if total_entries > 20:
            insights.append(f"Great job! You've tracked {total_entries} emotional moments this month.")
        elif total_entries > 10:
            insights.append("You're building a good habit of emotional awareness.")
        
        # Dominant emotion insights
        most_common = analytics["most_common_emotion"]
        if most_common in ["joy", "gratitude", "optimism"]:
            insights.append(f"Your most common emotion is {most_common} - you're maintaining positive mental health!")
        elif most_common in ["sadness", "anxiety", "stress"]:
            insights.append(f"You've been experiencing {most_common} frequently. Consider self-care strategies.")
        
        # Confidence insights
        avg_confidence = analytics["average_confidence"]
        if avg_confidence > 0.7:
            insights.append("Your emotional self-awareness is strong - you clearly recognize your feelings.")
        elif avg_confidence < 0.5:
            insights.append("Your emotions seem complex. This is normal - emotions can be nuanced.")
        
        # Stability insights
        stability = analytics.get("emotional_stability", {}).get("stability", "unknown")
        if stability == "high":
            insights.append("Your emotional patterns are stable and consistent.")
        elif stability == "low":
            insights.append("You're experiencing varied emotions - this shows emotional richness.")
        
        # Trajectory insights
        trajectory = analytics.get("mood_trajectory", {})
        if trajectory.get("improvement", False):
            insights.append("Your mood has been improving recently - keep up the positive momentum!")
        elif trajectory.get("trajectory") == "declining":
            insights.append("Consider reaching out for support or trying new coping strategies.")
        
        return insights if insights else ["Keep tracking to discover patterns in your emotional journey!"]