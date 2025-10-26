import requests
import pytest


API_BASE = "http://localhost:8000"


def test_uthmani_basic():
    """Test uthmani endpoint with valid phonemes input."""
    url = f"{API_BASE}/uthmani"
    payload = {
        "phonemes": "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن"
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 200
    data = response.json()
    assert "input_phonemes" in data
    assert "uthmani_text" in data
    assert data["input_phonemes"] == "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن"  
    assert isinstance(data["uthmani_text"], str)
    assert len(data["uthmani_text"]) > 0
    print(f"Input: {data['input_phonemes']}")
    print(f"Output: {data['uthmani_text']}")


def test_uthmani_with_spaces():
    """Test uthmani endpoint removes spaces from phonemes."""
    url = f"{API_BASE}/uthmani"
    payload = {
        "phonemes": "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن"
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 200
    data = response.json()
    assert "input_phonemes" in data
    assert "uthmani_text" in data
    assert data["input_phonemes"] == "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن"
    assert isinstance(data["uthmani_text"], str)
    print(f"Input: {data['input_phonemes']}")
    print(f"Output: {data['uthmani_text']}")


def test_uthmani_missing_phonemes():
    """Test uthmani endpoint with missing phonemes field."""
    url = f"{API_BASE}/uthmani"
    payload = {}
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "phonemes" in data["error"].lower()


def test_uthmani_empty_phonemes():
    """Test uthmani endpoint with empty phonemes string."""
    url = f"{API_BASE}/uthmani"
    payload = {
        "phonemes": ""
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_uthmani_invalid_type():
    """Test uthmani endpoint with invalid phonemes type (not string)."""
    url = f"{API_BASE}/uthmani"
    payload = {
        "phonemes": 12345
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_uthmani_long_phonemes():
    """Test uthmani endpoint with longer phoneme sequence."""
    url = f"{API_BASE}/uthmani"
    payload = {
        "phonemes": "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن"
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 200
    data = response.json()
    assert "input_phonemes" in data
    assert "uthmani_text" in data
    assert isinstance(data["uthmani_text"], str)
    print(f"Input: {data['input_phonemes']}")
    print(f"Output: {data['uthmani_text']}")


def test_uthmani_batch_basic():
    """Test uthmani endpoint with batch phonemes input."""
    url = f"{API_BASE}/uthmani"
    payload = {
        "phonemes_list": [
            "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن",
            "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن",
            "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن"
        ],
        "batch_size": 8
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "batch"
    assert data["count"] == 3
    assert "results" in data
    assert len(data["results"]) == 3
    
    # Check each result structure
    for i, result in enumerate(data["results"]):
        assert "input_phonemes" in result
        assert "phonemes_no_spaces" in result
        assert "uthmani_text" in result
        assert isinstance(result["uthmani_text"], str)
        print(f"\nResult {i+1}:")
        print(f"  Input: {result['input_phonemes']}")
        print(f"  Output: {result['uthmani_text']}")


def test_uthmani_batch_empty_list():
    """Test uthmani endpoint with empty batch list."""
    url = f"{API_BASE}/uthmani"
    payload = {
        "phonemes_list": []
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_uthmani_batch_invalid_type():
    """Test uthmani endpoint with invalid item in batch list."""
    url = f"{API_BASE}/uthmani"
    payload = {
        "phonemes_list": [
            "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن",
            12345,  # Invalid: not a string
            "a l h a m d u l i l l a h"
        ]
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "index 1" in data["error"]


def test_uthmani_batch_not_list():
    """Test uthmani endpoint with phonemes_list that is not a list."""
    url = f"{API_BASE}/uthmani"
    payload = {
        "phonemes_list": "not a list"
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_uthmani_batch_custom_batch_size():
    """Test uthmani endpoint with custom batch_size."""
    url = f"{API_BASE}/uthmani"
    payload = {
        "phonemes_list": [
            "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن",
            "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن",
            "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن",
            "ءِننننَاااا ءَںںںزَلنَااهُ قُرءَاانَن عَرَبِييَ للَعَللَكُم تَعقِلُۥۥنَ نَحنُ نَقُصصُ عَلَيكَ ءَحسَنَ لقَصَصِ بِمَاااا ءَوحَينَاااا ءِلَيكَ هَااذَ لقُرءَاانَ وَءِںںں كُںںںتَ مِںںں قَبڇلِهِۦۦ لَمِنَ لغَاافِلِۦۦۦۦۦن"
        ],
        "batch_size": 2  # Smaller batch size
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "batch"
    assert data["count"] == 4
    assert len(data["results"]) == 4


if __name__ == "__main__":
    # Run individual tests
    print("=" * 60)
    print("Test 1: Basic phonemes input")
    print("=" * 60)
    test_uthmani_basic()
    
    print("\n" + "=" * 60)
    print("Test 2: Phonemes with spaces")
    print("=" * 60)
    test_uthmani_with_spaces()
    
    print("\n" + "=" * 60)
    print("Test 3: Missing phonemes field")
    print("=" * 60)
    test_uthmani_missing_phonemes()
    
    print("\n" + "=" * 60)
    print("Test 4: Empty phonemes")
    print("=" * 60)
    test_uthmani_empty_phonemes()
    
    print("\n" + "=" * 60)
    print("Test 5: Invalid phonemes type")
    print("=" * 60)
    test_uthmani_invalid_type()
    
    print("\n" + "=" * 60)
    print("Test 6: Long phoneme sequence")
    print("=" * 60)
    test_uthmani_long_phonemes()
    
    print("\n" + "=" * 60)
    print("Test 7: Batch conversion - basic")
    print("=" * 60)
    test_uthmani_batch_basic()
    
    print("\n" + "=" * 60)
    print("Test 8: Batch conversion - empty list")
    print("=" * 60)
    test_uthmani_batch_empty_list()
    
    print("\n" + "=" * 60)
    print("Test 9: Batch conversion - invalid type in list")
    print("=" * 60)
    test_uthmani_batch_invalid_type()
    
    print("\n" + "=" * 60)
    print("Test 10: Batch conversion - phonemes_list not a list")
    print("=" * 60)
    test_uthmani_batch_not_list()
    
    print("\n" + "=" * 60)
    print("Test 11: Batch conversion - custom batch size")
    print("=" * 60)
    test_uthmani_batch_custom_batch_size()
