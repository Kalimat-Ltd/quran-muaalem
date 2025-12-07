"""
Tests for the /reference endpoint.

The /reference endpoint retrieves phonetizer output for a specific range of words
from a surah/ayah, with support for multi-ayah word ranges.
"""

import pytest
import requests


BASE_URL = "http://localhost:8000"
REFERENCE_ENDPOINT = f"{BASE_URL}/reference"


class TestReferenceBasic:
    """Basic tests for the /reference endpoint."""

    def test_reference_single_word(self):
        """Test retrieving a single word from a surah/ayah."""
        payload = {
            "surah": 76,
            "ayah": 15,
            "start_word": 1,
            "num_words": 40,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "surah" in data
        assert "ayah" in data
        assert "start_word" in data
        assert "num_words_retrieved" in data
        assert "num_words_requested" in data
        assert "uthmani_text" in data
        assert "phonetizer_out" in data
        assert "waqf_phonemes" in data
        assert "wasl_waqf_phonemes" in data
        assert "waqf_wasl_phonemes" in data
        assert "spaced_phonemes" in data
        assert "char_map" in data
        assert "spaced_char_map" in data
        assert "offsets" in data
        
        assert data["surah"] == 1
        assert data["ayah"] == 1
        assert data["start_word"] == 1
        assert data["num_words_retrieved"] == 1

    def test_reference_multiple_words(self):
        """Test retrieving multiple words from a surah/ayah."""
        payload = {
            "surah": 1,
            "ayah": 1,
            "start_word": 1,
            "num_words": 3,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["surah"] == 1
        assert data["ayah"] == 1
        assert data["num_words_retrieved"] == 3
        assert data["spans_multiple_ayahs"] is False

    def test_reference_with_start_word(self):
        """Test retrieving words starting from a middle word."""
        payload = {
            "surah": 2,
            "ayah": 1,
            "start_word": 2,
            "num_words": 2,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["start_word"] == 2
        assert data["num_words_retrieved"] == 2

    def test_reference_without_num_words(self):
        """Test retrieving all words without specifying num_words."""
        payload = {
            "surah": 1,
            "ayah": 1,
            "start_word": 1,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Should retrieve all words from start_word to end of ayah
        assert data["num_words_retrieved"] > 0

    def test_reference_span_multiple_ayahs(self):
        """Test retrieving words that span across multiple ayahs."""
        # Start near the end of an ayah with enough num_words to span to next ayah
        payload = {
            "surah": 2,
            "ayah": 1,
            "start_word": 1,
            "num_words": 50,  # Likely to span multiple ayahs
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Check if it spans multiple ayahs
        assert "spans_multiple_ayahs" in data
        assert "ayah_segments" in data
        assert isinstance(data["ayah_segments"], list)

    def test_reference_phonemes_structure(self):
        """Test that phonemes and char_map are properly structured."""
        payload = {
            "surah": 1,
            "ayah": 1,
            "start_word": 1,
            "num_words": 2,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Verify phonemes are strings
        assert isinstance(data["phonetizer_out"]["phonemes"], str)
        assert isinstance(data["spaced_phonemes"], str)
        assert isinstance(data["waqf_phonemes"], str)
        assert isinstance(data["wasl_waqf_phonemes"], str)
        assert isinstance(data["waqf_wasl_phonemes"], str)
        
        # Verify char_map is a list
        assert isinstance(data["phonetizer_out"]["char_map"], list)
        assert isinstance(data["spaced_char_map"], list)

    def test_reference_moshaf_attributes(self):
        """Test specifying different moshaf attributes."""
        payload = {
            "surah": 1,
            "ayah": 1,
            "start_word": 1,
            "num_words": 1,
            "rewaya": "hafs",
            "madd_monfasel_len": 4,
            "madd_mottasel_len": 4,
            "madd_mottasel_waqf": 4,
            "madd_aared_len": 4,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["num_words_retrieved"] == 1


class TestReferenceErrors:
    """Test error handling in the /reference endpoint."""

    def test_reference_missing_surah(self):
        """Test error when surah is missing."""
        payload = {
            "ayah": 1,
            "start_word": 1,
            "num_words": 1,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_reference_missing_ayah(self):
        """Test error when ayah is missing."""
        payload = {
            "surah": 1,
            "start_word": 1,
            "num_words": 1,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_reference_invalid_surah_low(self):
        """Test error when surah is too low."""
        payload = {
            "surah": 0,
            "ayah": 1,
            "start_word": 1,
            "num_words": 1,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_reference_invalid_surah_high(self):
        """Test error when surah is too high."""
        payload = {
            "surah": 115,
            "ayah": 1,
            "start_word": 1,
            "num_words": 1,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_reference_invalid_start_word(self):
        """Test error when start_word is invalid."""
        payload = {
            "surah": 1,
            "ayah": 1,
            "start_word": 0,
            "num_words": 1,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_reference_invalid_num_words(self):
        """Test error when num_words is invalid."""
        payload = {
            "surah": 1,
            "ayah": 1,
            "start_word": 1,
            "num_words": 0,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_reference_invalid_surah_type(self):
        """Test error when surah is not an integer."""
        payload = {
            "surah": "abc",
            "ayah": 1,
            "start_word": 1,
            "num_words": 1,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_reference_invalid_ayah_type(self):
        """Test error when ayah is not an integer."""
        payload = {
            "surah": 1,
            "ayah": "xyz",
            "start_word": 1,
            "num_words": 1,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data


class TestReferenceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_reference_first_surah_first_ayah(self):
        """Test retrieving from the very first surah and ayah."""
        payload = {
            "surah": 1,
            "ayah": 1,
            "start_word": 1,
            "num_words": 1,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["surah"] == 1
        assert data["ayah"] == 1

    def test_reference_last_surah(self):
        """Test retrieving from the last surah."""
        payload = {
            "surah": 114,
            "ayah": 1,
            "start_word": 1,
            "num_words": 1,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["surah"] == 114

    def test_reference_large_num_words(self):
        """Test retrieving a large number of words."""
        payload = {
            "surah": 1,
            "ayah": 1,
            "start_word": 1,
            "num_words": 100,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Should return whatever is available
        assert data["num_words_retrieved"] > 0


class TestReferenceOffsets:
    """Test offset calculations in responses."""

    def test_reference_offsets_structure(self):
        """Test that offsets are properly structured."""
        payload = {
            "surah": 76,
            "ayah": 15,
            "start_word": 1,
            "num_words": 40,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "offsets" in data
        assert "uthmani_word_offset" in data["offsets"]
        assert "uthmani_char_offset" in data["offsets"]
        assert isinstance(data["offsets"]["uthmani_word_offset"], int)
        assert isinstance(data["offsets"]["uthmani_char_offset"], int)

    def test_reference_offsets_values(self):
        """Test that offset values are correct."""
        payload = {
            "surah": 76,
            "ayah": 15,
            "start_word": 1,
            "num_words": 40,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # start_word is 1-based, so offset should be start_word - 1
        assert data["offsets"]["uthmani_word_offset"] == 1


class TestReferenceUthmaniText:
    """Test the uthmani_text in responses."""

    def test_reference_uthmani_text_not_empty(self):
        """Test that uthmani_text is not empty."""
        payload = {
            "surah": 76,
            "ayah": 15,
            "start_word": 1,
            "num_words": 40,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["uthmani_text"]) > 0

    def test_reference_uthmani_text_is_string(self):
        """Test that uthmani_text is a string."""
        payload = {
            "surah": 76,
            "ayah": 15,
            "start_word": 1,
            "num_words": 40,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data["uthmani_text"], str)


class TestReferenceAyahSegments:
    """Test ayah_segments in responses."""

    def test_reference_ayah_segments_single_ayah(self):
        """Test ayah_segments for single ayah retrieval."""
        payload = {
            "surah": 76,
            "ayah": 15,
            "start_word": 1,
            "num_words": 40,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["ayah_segments"]) >= 1
        segment = data["ayah_segments"][0]
        assert "surah" in segment
        assert "ayah" in segment
        assert "start_word" in segment
        assert "end_word" in segment
        assert "word_count" in segment

    def test_reference_ayah_segments_values(self):
        """Test that ayah_segments have correct values."""
        payload = {
            "surah": 76,
            "ayah": 15,
            "start_word": 1,
            "num_words": 40,
        }
        response = requests.post(REFERENCE_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        
        segment = data["ayah_segments"][0]
        assert segment["surah"] == 1
        assert segment["ayah"] == 1
        assert segment["start_word"] >= 1
        assert segment["end_word"] >= segment["start_word"]
        assert segment["word_count"] == segment["end_word"] - segment["start_word"] + 1


def test_reference_basic_functionality():
    """Simple integration test for basic /reference functionality."""
    payload = {
        "surah": 78,
        "ayah": 16,
        "start_word": 2,
        "num_words": 1,
    }
    response = requests.post(REFERENCE_ENDPOINT, json=payload)
    
    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 200
    data = response.json()
    assert "uthmani_text" in data
    assert "phonetizer_out" in data
    assert len(data["uthmani_text"]) > 0


if __name__ == "__main__":
    # Run basic test if executed directly
    test_reference_basic_functionality()
