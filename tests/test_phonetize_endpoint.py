import requests

def test_phonetize():
    url = "http://localhost:8000/phonetize"
    payload = {
        "text": "إِنَّآ أَنزَلْنَـٰهُ قُرْءَٰنًا عَرَبِيًّۭا لَّعَلَّكُمْ تَعْقِلُونَ نَحْنُ نَقُصُّ عَلَيْكَ أَحْسَنَ ٱلْقَصَصِ بِمَآ أَوْحَيْنَآ إِلَيْكَ هَـٰذَا ٱلْقُرْءَانَ وَإِن كُنتَ مِن قَبْلِهِۦ لَمِنَ ٱلْغَـٰفِلِينَ "
    }
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    if response.status_code == 200:
        data = response.json()
        assert "phonemes" in data
        assert "waqf_phonemes" in data
        print("Phonemes:", repr(data["phonemes"]))
        print("Segmented Phonemes:", repr(data["segmented_phonemes"]))
        print("Waqf Phonemes:", repr(data["waqf_phonemes"]))
        print("Waqf Wasl Phonemes:", repr(data["waqf_wasl_phonemes"]))
        print("Wasl Waqf Phonemes:", repr(data["wasl_waqf_phonemes"]))

    else:
        print("Failed with status", response.status_code)

if __name__ == "__main__":
    test_phonetize()
