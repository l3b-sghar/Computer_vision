"""
Analytics package for emotion and body-language analysis.

This package contains modules for analyzing detected features,
computing satisfaction scores, tracking interactions, and
generating analytics outputs.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AnalyticsResult:
    """
    Container for analytics results.
    
    Attributes:
        customer_satisfaction_score: Overall satisfaction score (0-1)
        processing_time_seconds: Total interaction time in seconds
        attention_score: Optional attention score (0-1)
        stress_score: Optional stress level score (0-1)
        hesitancy_score: Optional hesitancy score (0-1)
        body_language_events: Optional list of body language events
        timestamp: Timestamp of the analysis
        metadata: Additional metadata
    """
    customer_satisfaction_score: float
    processing_time_seconds: float
    attention_score: Optional[float] = None
    stress_score: Optional[float] = None
    hesitancy_score: Optional[float] = None
    body_language_events: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert analytics result to dictionary.
        
        Returns:
            Dictionary representation of the analytics result
        """
        result = {
            'customer_satisfaction_score': self.customer_satisfaction_score,
            'processing_time_seconds': self.processing_time_seconds,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
        
        if self.attention_score is not None:
            result['attention_score'] = self.attention_score
        
        if self.stress_score is not None:
            result['stress_score'] = self.stress_score
        
        if self.hesitancy_score is not None:
            result['hesitancy_score'] = self.hesitancy_score
        
        if self.body_language_events is not None:
            result['body_language_events'] = self.body_language_events
        
        if self.metadata is not None:
            result['metadata'] = self.metadata
        
        return result


__all__ = ['AnalyticsResult']
