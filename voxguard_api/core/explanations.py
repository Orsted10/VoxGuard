"""
VoxGuard Explanation Generation Module
Generates human-readable explanations for deepfake detection results
"""

from typing import Dict, Any, List
import numpy as np


def build_explanation(
    features: Dict[str, Any], 
    probability: float, 
    language: str,
    classification: str
) -> str:
    """
    Generate a human-readable explanation for the classification.
    
    Args:
        features: Extracted audio features
        probability: Probability of AI_GENERATED classification
        language: The detected/specified language
        classification: The classification result
        
    Returns:
        Human-readable explanation string
    """
    reasons = analyze_features(features, probability)
    
    if classification == "AI_GENERATED":
        explanation = build_ai_explanation(reasons, language, probability)
    else:
        explanation = build_human_explanation(reasons, language, probability)
    
    return explanation


def analyze_features(features: Dict[str, Any], probability: float) -> Dict[str, Any]:
    """
    Analyze features to identify suspicious patterns.
    
    Args:
        features: Extracted audio features
        probability: Classification probability
        
    Returns:
        Dictionary of analysis results
    """
    reasons = {
        "suspicious": [],
        "natural": [],
        "details": {}
    }
    
    # 1. Pitch Analysis
    pitch_std = features.get("pitch_std", 0)
    pitch_range = features.get("pitch_range", 0)
    
    if pitch_std < 15:  # Very low pitch variation
        reasons["suspicious"].append("unnaturally consistent pitch")
        reasons["details"]["pitch"] = "low_variation"
    elif pitch_std > 80:  # Very high, natural variation
        reasons["natural"].append("natural pitch variation")
        reasons["details"]["pitch"] = "natural"
    else:
        reasons["details"]["pitch"] = "moderate"
    
    # 2. Energy/Volume Analysis
    energy_std = features.get("energy_std", 0)
    dynamic_range = features.get("dynamic_range", 0)
    
    if dynamic_range < 10:  # Very flat audio
        reasons["suspicious"].append("unnaturally uniform volume levels")
        reasons["details"]["energy"] = "too_uniform"
    elif dynamic_range > 40:  # Good dynamics
        reasons["natural"].append("natural volume dynamics")
        reasons["details"]["energy"] = "natural"
    else:
        reasons["details"]["energy"] = "moderate"
    
    # 3. Spectral Flatness Analysis
    spectral_flatness = features.get("spectral_flatness_mean", 0)
    
    if spectral_flatness < 0.01:  # Very tonal
        reasons["suspicious"].append("overly clean spectral characteristics")
        reasons["details"]["spectral"] = "too_clean"
    elif spectral_flatness > 0.1:  # More noise-like, natural
        reasons["natural"].append("natural spectral characteristics")
        reasons["details"]["spectral"] = "natural"
    else:
        reasons["details"]["spectral"] = "moderate"
    
    # 4. Harmonic Analysis
    harmonic_ratio = features.get("harmonic_ratio", 0)
    
    if harmonic_ratio > 0.95:  # Almost pure harmonic
        reasons["suspicious"].append("synthetic harmonic structure")
        reasons["details"]["harmonics"] = "too_pure"
    elif harmonic_ratio < 0.7:  # Natural mix
        reasons["natural"].append("natural harmonic balance")
        reasons["details"]["harmonics"] = "natural"
    else:
        reasons["details"]["harmonics"] = "moderate"
    
    # 5. Voiced Ratio Analysis
    voiced_ratio = features.get("voiced_ratio", 0)
    
    if voiced_ratio > 0.95:  # Almost no unvoiced segments
        reasons["suspicious"].append("absence of natural breath sounds")
        reasons["details"]["voiced"] = "no_breaths"
    elif voiced_ratio < 0.8:  # Natural pauses
        reasons["natural"].append("natural speech pauses")
        reasons["details"]["voiced"] = "natural"
    else:
        reasons["details"]["voiced"] = "moderate"
    
    # 6. MFCC Smoothness Analysis
    mfcc_delta = features.get("mfcc_delta", np.zeros(1))
    mfcc_smoothness = np.mean(mfcc_delta) if isinstance(mfcc_delta, np.ndarray) else mfcc_delta
    
    if mfcc_smoothness < 0.5:  # Very smooth transitions
        reasons["suspicious"].append("robotic formant transitions")
        reasons["details"]["mfcc"] = "too_smooth"
    elif mfcc_smoothness > 2.0:  # Natural variability
        reasons["natural"].append("natural formant patterns")
        reasons["details"]["mfcc"] = "natural"
    else:
        reasons["details"]["mfcc"] = "moderate"
    
    # 7. Zero Crossing Rate Variability
    zcr_std = features.get("zcr_std", 0)
    
    if zcr_std < 0.02:
        reasons["suspicious"].append("mechanical speech rhythm")
        reasons["details"]["zcr"] = "too_uniform"
    elif zcr_std > 0.1:
        reasons["natural"].append("expressive speech patterns")
        reasons["details"]["zcr"] = "natural"
    else:
        reasons["details"]["zcr"] = "moderate"
    
    return reasons


def build_ai_explanation(
    reasons: Dict[str, Any], 
    language: str, 
    probability: float
) -> str:
    """Build explanation for AI_GENERATED classification."""
    
    confidence_descriptor = get_confidence_descriptor(probability)
    
    if reasons["suspicious"]:
        # Use top 2-3 suspicious patterns
        top_reasons = reasons["suspicious"][:3]
        reason_text = ", ".join(top_reasons[:-1])
        if len(top_reasons) > 1:
            reason_text += f" and {top_reasons[-1]}"
        else:
            reason_text = top_reasons[0]
        
        explanation = f"Analysis {confidence_descriptor} indicates AI-generated audio. Detected: {reason_text}."
    else:
        # Generic AI explanation based on probability
        explanation = f"Analysis {confidence_descriptor} indicates this audio has characteristics typical of AI-generated speech synthesis."
    
    # Add language context
    explanation += f" Voice patterns analyzed for {language} language characteristics."
    
    return explanation


def build_human_explanation(
    reasons: Dict[str, Any], 
    language: str, 
    probability: float
) -> str:
    """Build explanation for HUMAN classification."""
    
    confidence_descriptor = get_confidence_descriptor(1 - probability)
    
    if reasons["natural"]:
        # Use natural indicators
        top_reasons = reasons["natural"][:3]
        reason_text = ", ".join(top_reasons[:-1])
        if len(top_reasons) > 1:
            reason_text += f" and {top_reasons[-1]}"
        else:
            reason_text = top_reasons[0]
        
        explanation = f"Analysis {confidence_descriptor} indicates authentic human speech. Detected: {reason_text}."
    else:
        # Generic human explanation
        explanation = f"Analysis {confidence_descriptor} indicates this audio exhibits natural human speech characteristics."
    
    # Add language context
    explanation += f" Voice authenticity verified for {language} speech."
    
    return explanation


def get_confidence_descriptor(probability: float) -> str:
    """Get a descriptor based on confidence level."""
    if probability >= 0.9:
        return "strongly"
    elif probability >= 0.75:
        return "confidently"
    elif probability >= 0.6:
        return "moderately"
    else:
        return "marginally"


def get_suspicion_reasons(features: Dict[str, Any]) -> List[str]:
    """
    Get a list of suspicion reasons for showcase/debugging.
    
    Returns a detailed list of what makes the audio suspicious.
    """
    reasons = analyze_features(features, 0.5)
    return reasons["suspicious"]


def get_naturalness_indicators(features: Dict[str, Any]) -> List[str]:
    """
    Get a list of naturalness indicators for showcase/debugging.
    
    Returns a detailed list of what makes the audio natural.
    """
    reasons = analyze_features(features, 0.5)
    return reasons["natural"]
