import requests

def test_phonetize():
    url = "http://localhost:8000/phonetize"
    payload = {
        "text": "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    if response.status_code == 200:
        data = response.json()
        assert "phonemes" in data
        assert "waqf_phonemes" in data
        print("Phonemes:", repr(data["phonemes"]))
        print("Waqf Phonemes:", repr(data["waqf_phonemes"]))
        print("Waqf WSL Phonemes:", repr(data["waqf_wsl_phonemes"]))

    else:
        print("Failed with status", response.status_code)

if __name__ == "__main__":
    test_phonetize()
