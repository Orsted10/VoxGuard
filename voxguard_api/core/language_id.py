"""
VoxGuard Language Validation Module
Handles language validation and consistency checks
"""

from voxguard_api.core.config import SUPPORTED_LANGUAGES


def validate_language(lang: str) -> str:
    """
    Validate that the language is one of the supported languages.
    
    Args:
        lang: The language string to validate
        
    Returns:
        The validated language string
        
    Raises:
        ValueError: If the language is not supported
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{lang}'. "
            f"Supported languages are: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    return lang


def get_language_code(lang: str) -> str:
    """
    Get ISO language code for a supported language.
    
    Args:
        lang: The language name
        
    Returns:
        ISO 639-1 language code
    """
    language_codes = {
        "Tamil": "ta",
        "English": "en",
        "Hindi": "hi",
        "Malayalam": "ml",
        "Telugu": "te"
    }
    return language_codes.get(lang, "en")


def get_language_display_name(lang: str) -> str:
    """
    Get a display-friendly language name with native script.
    
    Args:
        lang: The language name
        
    Returns:
        Display name with native script
    """
    display_names = {
        "Tamil": "Tamil (தமிழ்)",
        "English": "English",
        "Hindi": "Hindi (हिंदी)",
        "Malayalam": "Malayalam (മലയാളം)",
        "Telugu": "Telugu (తెలుగు)"
    }
    return display_names.get(lang, lang)
