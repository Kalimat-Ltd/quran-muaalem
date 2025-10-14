#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Quranic Word Waqf (Stopping) Rules Implementation
========================================================

This script implements the 10 main Tajweed Waqf rules for Arabic Quranic words.
When given an Arabic word with diacritics, it returns both the original pronunciation
and the Waqf (stopping) pronunciation according to Tajweed rules.

Author: GitHub Copilot Assistant
Date: October 2025
"""

import re
import json
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from urllib import request as urllib_request
from urllib import error as urllib_error


DEFAULT_PHONEME_SERVICE_URL = "https://joey-wondrous-wasp.ngrok-free.app/phonetize"


@dataclass
class WaqfResult:
    """Result of applying Waqf rules to a word"""
    original: str
    waqf: str
    rule_applied: str
    description: str


class ArabicTextProcessor:
    """Utility class for processing Arabic text and diacritics"""
    
    # Arabic diacritics and symbols
    DIACRITICS = {
        'fatha': '\u064E',      # َ
        'damma': '\u064F',      # ُ
        'kasra': '\u0650',      # ِ
        'sukoon': '\u0652',     # ْ
        'shadda': '\u0651',     # ّ
        'tanween_fath': '\u064B',   # ً
        'tanween_damm': '\u064C',   # ٌ
        'tanween_kasr': '\u064D',   # ٍ
        'alif_khanjariyya': '\u0670', # ٰ
        'maddah': '\u0653',     # ٓ
    }
    
    # Arabic letters
    TAA_MARBOOTA = '\u0629'     # ة
    HAA = '\u0647'              # ه
    ALIF = '\u0627'             # ا
    WAW = '\u0648'              # و
    YAA = '\u064A'              # ي
    ALIF_MAKSURA = '\u0649'     # ى
    HAMZA = '\u0621'            # ء
    HAMZA_ON_ALIF = '\u0623'    # أ
    HAMZA_UNDER_ALIF = '\u0625' # إ
    HAMZA_ON_WAW = '\u0624'     # ؤ
    HAMZA_ON_YAA = '\u0626'     # ئ
    
    # Hamzat Wasl
    HAMZAT_WASL = '\u0671'      # ٱ

    # Orthographic marks that may appear at word endings but are not pronounced at Waqf
    ORTHOGRAPHIC_ENDING_SIGNS: Tuple[str, ...] = ('\u06E2', '\u06E5')
    
    @classmethod
    def remove_diacritics(cls, text: str) -> str:
        """Remove all diacritics from Arabic text"""
        diacritics_pattern = '[' + ''.join(cls.DIACRITICS.values()) + ']'
        return re.sub(diacritics_pattern, '', text)
    
    @classmethod
    def get_last_character_info(cls, word: str) -> Tuple[str, str, List[str]]:
        """
        Get information about the last character in the word
        Returns: (base_char, last_char_with_diacritics, diacritics_list)
        """
        if not word:
            return '', '', []
        
        # Find diacritics on the last character
        diacritics = []
        base_char = ''
        
        # Skip orthographic ending signs like small meem (ۢ)
        index = len(word) - 1
        while index >= 0 and word[index] in cls.ORTHOGRAPHIC_ENDING_SIGNS:
            index -= 1

        if index < 0:
            return '', '', []

        # Work backwards from the first non-orthographic mark to find the base character
        for i in range(index, -1, -1):
            char = word[i]
            if char in cls.DIACRITICS.values():
                diacritics.insert(0, char)
            else:
                base_char = char
                break
        
        last_char_with_diacritics = base_char + ''.join(diacritics)
        return base_char, last_char_with_diacritics, diacritics

    @classmethod
    def strip_orthographic_end_marks(cls, text: str) -> str:
        """Remove trailing orthographic marks that should vanish at Waqf."""
        while text and text[-1] in cls.ORTHOGRAPHIC_ENDING_SIGNS:
            text = text[:-1]
        return text
    
    @classmethod
    def has_tanween_fath_ending(cls, word: str) -> bool:
        """Check if word ends with tanween fath (including ًا pattern)"""
        if not word:
            return False
        
        # Check for tanween fath directly
        if word.endswith(cls.DIACRITICS['tanween_fath']):
            return True
        
        # Check for ًا pattern (tanween fath + alif)
        tanween_alif_pattern = cls.DIACRITICS['tanween_fath'] + cls.ALIF
        if word.endswith(tanween_alif_pattern):
            return True
        
        return False
    
    @classmethod
    def is_consonant(cls, char: str) -> bool:
        """Check if character is a consonant (not a long vowel)"""
        long_vowels = {cls.ALIF, cls.WAW, cls.YAA, cls.ALIF_MAKSURA}
        return char not in long_vowels and not char in cls.DIACRITICS.values()
    
    @classmethod
    def has_tanween(cls, diacritics: List[str]) -> Optional[str]:
        """Check if diacritics contain tanween and return its type"""
        for diacritic in diacritics:
            if diacritic == cls.DIACRITICS['tanween_fath']:
                return 'fath'
            elif diacritic == cls.DIACRITICS['tanween_damm']:
                return 'damm'
            elif diacritic == cls.DIACRITICS['tanween_kasr']:
                return 'kasr'
        return None
    
    @classmethod
    def has_short_vowel(cls, diacritics: List[str]) -> Optional[str]:
        """Check if diacritics contain short vowels"""
        for diacritic in diacritics:
            if diacritic == cls.DIACRITICS['fatha']:
                return 'fatha'
            elif diacritic == cls.DIACRITICS['damma']:
                return 'damma'
            elif diacritic == cls.DIACRITICS['kasra']:
                return 'kasra'
        return None


class WaqfProcessor:
    """Main class for applying Waqf rules to Arabic words"""
    
    def __init__(self):
        self.processor = ArabicTextProcessor()
    
    def apply_waqf_rules(self, word: str) -> WaqfResult:
        """
        Apply appropriate Waqf rules to an Arabic word
        Returns both original and Waqf pronunciations
        """
        if not word.strip():
            return WaqfResult(word, word, "No rule", "Empty word")
        
        word = word.strip()
        original_word = word
        word = self.processor.strip_orthographic_end_marks(word)
        
        # Get character info first
        base_char, _, diacritics = self.processor.get_last_character_info(word)
        
        # Priority 1: Taa Marboota (ة) - highest priority, even with tanween fath
        if base_char == self.processor.TAA_MARBOOTA:
            result = self._apply_taa_marboota_rule(word)
            result.original = original_word
            return result
        
        # Priority 2: Check for tanween fath patterns (including ًا)
        if self.processor.has_tanween_fath_ending(word):
            result = self._apply_tanween_fath_rule(word)
            result.original = original_word
            return result
        
        # Priority 3: Other tanween types
        tanween_type = self.processor.has_tanween(diacritics)
        if tanween_type in ['damm', 'kasr']:
            result = self._apply_tanween_damm_kasr_rule(word, tanween_type)
            result.original = original_word
            return result
        
        # Priority 4: Hamza endings (before checking for alif)
        if self._is_hamza_ending(base_char):
            result = self._apply_hamza_rule(word)
            result.original = original_word
            return result
        
        # Priority 5: Final Alif (ا)
        if base_char == self.processor.ALIF:
            result = self._apply_alif_rule(word)
            result.original = original_word
            return result
        
        # Priority 6: Long vowels (و ي as vowels)
        if self._is_long_vowel_ending(word, base_char):
            result = self._apply_long_vowel_rule(word)
            result.original = original_word
            return result
        
        # Priority 7: Soft letters (حرف لين)
        if self._is_soft_letter_ending(word):
            result = self._apply_soft_letter_rule(word)
            result.original = original_word
            return result
        
        # Priority 8: Regular consonants with vowels
        if self.processor.is_consonant(base_char) and self.processor.has_short_vowel(diacritics):
            result = self._apply_consonant_rule(word)
            result.original = original_word
            return result
        
        # Default: no change needed
        return WaqfResult(
            original=original_word,
            waqf=word,
            rule_applied="No change needed",
            description="Word already ends appropriately for Waqf"
        )
    
    def _apply_taa_marboota_rule(self, word: str) -> WaqfResult:
        """Rule 2: Taa Marboota becomes silent Haa"""
        # Find the taa marboota and remove everything after the base word
        taa_pos = word.rfind(self.processor.TAA_MARBOOTA)
        if taa_pos >= 0:
            word_base = word[:taa_pos]
            waqf_word = word_base + self.processor.HAA + self.processor.DIACRITICS['sukoon']
        else:
            waqf_word = word  # Fallback
        
        return WaqfResult(
            original=word,
            waqf=waqf_word,
            rule_applied="Taa Marboota Rule",
            description="تاء مربوطة تُبدل هاء ساكنة"
        )
    
    def _apply_tanween_fath_rule(self, word: str) -> WaqfResult:
        """Rule 5: Tanween Fath becomes Alif or gets removed for Hamza"""
        # Handle ًا pattern (tanween fath + alif)
        tanween_alif_pattern = self.processor.DIACRITICS['tanween_fath'] + self.processor.ALIF
        
        # Special case: Check for hamza before the tanween pattern
        if word.endswith(tanween_alif_pattern):
            # Remove the ًا pattern first
            base_word = word[:-len(tanween_alif_pattern)]
            
            # Check what character comes before the tanween
            if base_word and self._is_hamza_ending(base_word[-1]):
                # For hamza + tanween fath: hamza gets sukoon
                waqf_word = base_word + self.processor.DIACRITICS['sukoon']
                description = "همزة مع تنوين الفتح تُسكن"
            else:
                # Regular case: becomes alif
                waqf_word = base_word + self.processor.ALIF
                description = "تنوين الفتح يُبدل ألفًا"
        
        # Check for tanween fath directly (not followed by alif)
        elif word.endswith(self.processor.DIACRITICS['tanween_fath']):
            # Check if the character before tanween is hamza
            base_word = word[:-len(self.processor.DIACRITICS['tanween_fath'])]
            
            if base_word and self._is_hamza_ending(base_word[-1]):
                # For hamza + tanween fath: hamza gets sukoon
                waqf_word = base_word + self.processor.DIACRITICS['sukoon']
                description = "همزة مع تنوين الفتح تُسكن"
            else:
                # Regular tanween fath - replace with alif
                waqf_word = word.replace(self.processor.DIACRITICS['tanween_fath'], self.processor.ALIF)
                description = "تنوين الفتح يُبدل ألفًا"
        
        else:
            # Should not reach here based on calling logic, but fallback
            waqf_word = word
            description = "تنوين الفتح - حالة استثنائية"
        
        return WaqfResult(
            original=word,
            waqf=waqf_word,
            rule_applied="Tanween Fath Rule",
            description=description
        )
    
    def _apply_tanween_damm_kasr_rule(self, word: str, tanween_type: str) -> WaqfResult:
        """Rule 7: Tanween Damm/Kasr is removed and letter is made silent"""
        # Remove tanween, add sukoon
        if tanween_type == 'damm':
            word_without_tanween = word.replace(self.processor.DIACRITICS['tanween_damm'], '')
        else:
            word_without_tanween = word.replace(self.processor.DIACRITICS['tanween_kasr'], '')
        
        word_without_tanween = self.processor.strip_orthographic_end_marks(word_without_tanween)

        vowel_diacritics = [
            self.processor.DIACRITICS['fatha'],
            self.processor.DIACRITICS['damma'],
            self.processor.DIACRITICS['kasra'],
        ]

        for diacritic in vowel_diacritics:
            if word_without_tanween.endswith(diacritic):
                word_without_tanween = word_without_tanween[:-len(diacritic)]
                break

        if not word_without_tanween.endswith(self.processor.DIACRITICS['sukoon']):
            waqf_word = word_without_tanween + self.processor.DIACRITICS['sukoon']
        else:
            waqf_word = word_without_tanween
        
        return WaqfResult(
            original=word,
            waqf=waqf_word,
            rule_applied="Tanween Damm/Kasr Rule",
            description="تنوين الضم/الكسر يُحذف ويُسكن الحرف"
        )
    
    def _apply_alif_rule(self, word: str) -> WaqfResult:
        """Rule 3: Final Alif remains with natural lengthening"""
        return WaqfResult(
            original=word,
            waqf=word,  # No change
            rule_applied="Final Alif Rule",
            description="الألف المتطرفة تبقى كما هي بالمد الطبيعي"
        )
    
    def _apply_long_vowel_rule(self, word: str) -> WaqfResult:
        """Rule 4: Long vowels (و ي) remain with natural lengthening"""
        return WaqfResult(
            original=word,
            waqf=word,  # No change
            rule_applied="Long Vowel Rule",
            description="حرف المد يبقى كما هو بالمد الطبيعي"
        )
    
    def _apply_hamza_rule(self, word: str) -> WaqfResult:
        """Rule 6: Final Hamza is made silent"""
        # Remove vowel diacritics from hamza, add sukoon
        word_clean = word
        vowel_diacritics = [
            self.processor.DIACRITICS['fatha'],
            self.processor.DIACRITICS['damma'], 
            self.processor.DIACRITICS['kasra']
        ]
        
        # Remove only the last occurrence of vowel diacritics
        for diacritic in vowel_diacritics:
            if word_clean.endswith(diacritic):
                word_clean = word_clean[:-len(diacritic)]
                break
        
        if not word_clean.endswith(self.processor.DIACRITICS['sukoon']):
            waqf_word = word_clean + self.processor.DIACRITICS['sukoon']
        else:
            waqf_word = word_clean
        
        return WaqfResult(
            original=word,
            waqf=waqf_word,
            rule_applied="Final Hamza Rule",
            description="الهمزة المتطرفة تُسكن"
        )
    
    def _apply_soft_letter_rule(self, word: str) -> WaqfResult:
        """Rule 10: Soft letters (حرف لين) remain as is"""
        return WaqfResult(
            original=word,
            waqf=word,
            rule_applied="Soft Letter Rule",
            description="حرف اللين يبقى كما هو"
        )
    
    def _apply_consonant_rule(self, word: str) -> WaqfResult:
        """Rule 1: Regular consonants with vowels become silent"""
        # Remove only the final vowel diacritics, preserve internal ones
        word_result = word
        vowel_diacritics = [
            self.processor.DIACRITICS['fatha'],
            self.processor.DIACRITICS['damma'], 
            self.processor.DIACRITICS['kasra']
        ]
        
        # Remove the last vowel diacritic
        for diacritic in vowel_diacritics:
            if word_result.endswith(diacritic):
                word_result = word_result[:-len(diacritic)]
                break
        
        # Add sukoon if not already present
        if not word_result.endswith(self.processor.DIACRITICS['sukoon']):
            waqf_word = word_result + self.processor.DIACRITICS['sukoon']
        else:
            waqf_word = word_result
        
        return WaqfResult(
            original=word,
            waqf=waqf_word,
            rule_applied="Consonant Sukoon Rule", 
            description="الحرف المتحرك يُسكن عند الوقف"
        )
    
    def _is_long_vowel_ending(self, word: str, base_char: str) -> bool:
        """Check if word ends with a long vowel (و or ي as vowels)"""
        if base_char not in [self.processor.WAW, self.processor.YAA]:
            return False
        
        # Check if it's preceded by appropriate vowel
        if len(word) < 2:
            return False
        
        # Get the character before the last one
        prev_char_info = self.processor.get_last_character_info(word[:-1])
        prev_diacritics = prev_char_info[2]
        
        # For WAW: should be preceded by damma
        if base_char == self.processor.WAW:
            return self.processor.DIACRITICS['damma'] in prev_diacritics
        
        # For YAA: should be preceded by kasra
        if base_char == self.processor.YAA:
            return self.processor.DIACRITICS['kasra'] in prev_diacritics
        
        return False
    
    def _is_hamza_ending(self, base_char: str) -> bool:
        """Check if character is any form of hamza"""
        hamza_forms = [
            self.processor.HAMZA,
            self.processor.HAMZA_ON_ALIF,
            self.processor.HAMZA_UNDER_ALIF,
            self.processor.HAMZA_ON_WAW,
            self.processor.HAMZA_ON_YAA
        ]
        return base_char in hamza_forms
    
    def _is_soft_letter_ending(self, word: str) -> bool:
        """Check if word ends with soft letter (حرف لين): و or ي preceded by fatha"""
        if len(word) < 2:
            return False
        
        base_char, _, _ = self.processor.get_last_character_info(word)
        
        if base_char not in [self.processor.WAW, self.processor.YAA]:
            return False
        
        # Check if preceded by fatha (sukoon is implicit after fatha for soft letters)
        # This is for patterns like خَوْف، بَيْت
        prev_chars = word[:-1]
        return self.processor.DIACRITICS['fatha'] in prev_chars


@dataclass
class MaddInfo:
    """Information about a detected madd segment"""
    madd_type: str
    letter: str
    phoneme_symbols: List[str]
    has_maddah_mark: bool = False


@dataclass
class PhonemeConfig:
    """Configuration for phoneme processing"""
    madd_lin_duration: int = 3
    madd_aaridh_duration: int = 4
    natural_madd_duration: int = 2
    qalqala_letters: Tuple[str, ...] = ('ق', 'ط', 'ب', 'ج', 'د')
    qalqala_symbol: str = 'ڇ'
    short_vowel_symbols: Tuple[str, ...] = ('َ', 'ُ', 'ِ', 'ً', 'ٌ', 'ٍ')
    madd_symbol_map: Dict[str, Dict[str, List[str]]] = field(
        default_factory=lambda: {
            ArabicTextProcessor.YAA: {
                'lin': [ArabicTextProcessor.YAA],
                'aaridh': ['ۦ', ArabicTextProcessor.YAA],
            },
            ArabicTextProcessor.WAW: {
                'lin': [ArabicTextProcessor.WAW],
                'aaridh': ['ۥ', ArabicTextProcessor.WAW],
            },
            ArabicTextProcessor.ALIF: {
                'lin': [ArabicTextProcessor.ALIF],
                'aaridh': [ArabicTextProcessor.ALIF],
            },
            ArabicTextProcessor.ALIF_MAKSURA: {
                'lin': [ArabicTextProcessor.YAA],
                'aaridh': ['ۦ', ArabicTextProcessor.YAA],
            },
        }
    )


class PhonemeProcessor:
    """Processor for adjusting phoneme strings according to Waqf rules"""

    def __init__(self, config: Optional[PhonemeConfig] = None):
        self.config = config or PhonemeConfig()
        self.processor = ArabicTextProcessor()
        self._diacritics_set = set(self.processor.DIACRITICS.values())
        self._short_vowel_set = set(self.config.short_vowel_symbols)

    def process(self, phonemes: str, waqf_word: str) -> str:
        phoneme_chars = list(phonemes.strip())

        if not phoneme_chars:
            return phonemes

        waqf_tokens = self._parse_word(waqf_word)
        waqf_final_char = self._get_last_base_char(waqf_tokens)

        self._strip_tanween_suffix(phoneme_chars, waqf_final_char)
        self._remove_trailing_short_vowels(phoneme_chars)
        final_consonant_idx = self._find_final_consonant_index(
            phoneme_chars,
            target_char=waqf_final_char,
        )

        if (
            final_consonant_idx is not None
            and final_consonant_idx + 1 < len(phoneme_chars)
        ):
            phoneme_chars = phoneme_chars[: final_consonant_idx + 1]
            final_consonant_idx = len(phoneme_chars) - 1

        madd_info = self._detect_madd(waqf_tokens)

        if madd_info and final_consonant_idx is not None:
            if (
                madd_info.has_maddah_mark
                and final_consonant_idx == len(phoneme_chars) - 1
            ):
                final_consonant_idx = len(phoneme_chars)
            target_length = self._determine_madd_target_length(madd_info)
            phoneme_chars = self._adjust_madd_sequence(
                phoneme_chars,
                final_consonant_idx,
                madd_info.phoneme_symbols,
                target_length,
            )
            final_consonant_idx = self._find_final_consonant_index(
                phoneme_chars,
                target_char=waqf_final_char,
            )

        final_consonant_char = None
        if final_consonant_idx is not None and 0 <= final_consonant_idx < len(phoneme_chars):
            final_consonant_char = phoneme_chars[final_consonant_idx]

        if (
            waqf_final_char in self.config.qalqala_letters
            and final_consonant_char is not None
            and phoneme_chars[-1] != self.config.qalqala_symbol
        ):
            phoneme_chars.append(self.config.qalqala_symbol)

        return ''.join(phoneme_chars)

    def _remove_trailing_short_vowels(self, chars: List[str]) -> None:
        """Pop short vowel markers from the tail of a phoneme list in-place."""
        while chars and chars[-1] in self._short_vowel_set:
            chars.pop()

    def _strip_tanween_suffix(self, chars: List[str], waqf_final_char: Optional[str]) -> None:
        """Remove a trailing tanween-derived noon if it vanishes in the Waqf form."""
        if not chars:
            return

        if chars[-1] == 'ن' and waqf_final_char != 'ن':
            chars.pop()
            while chars and chars[-1] in self._short_vowel_set:
                chars.pop()

    def _determine_madd_target_length(self, madd_info: MaddInfo) -> int:
        """Compute the desired length (in counts) for the detected Madd segment."""
        if madd_info.has_maddah_mark:
            return self.config.natural_madd_duration
        if madd_info.madd_type == 'lin':
            return self.config.madd_lin_duration
        return self.config.madd_aaridh_duration

    def _parse_word(self, word: str) -> List[Dict[str, List[str]]]:
        """Split a vocalised word into tokens of base characters and diacritics."""
        tokens: List[Dict[str, List[str]]] = []
        current: Optional[Dict[str, List[str]]] = None

        for char in word:
            if char in self._diacritics_set:
                if current is not None:
                    current['diacritics'].append(char)
            else:
                current = {'char': char, 'diacritics': []}
                tokens.append(current)

        return tokens

    def _detect_madd(self, tokens: List[Dict[str, List[str]]]) -> Optional[MaddInfo]:
        """Analyse the final tokens to determine whether a Madd is present."""
        if not tokens:
            return None

        last_token = tokens[-1]
        has_maddah = self.processor.DIACRITICS['maddah'] in last_token['diacritics']

        if has_maddah and last_token['char'] in (
            self.processor.ALIF,
            self.processor.WAW,
            self.processor.YAA,
            self.processor.ALIF_MAKSURA,
        ):
            phoneme_symbols = self.config.madd_symbol_map.get(
                last_token['char'],
                {},
            ).get('aaridh', [last_token['char']])
            return MaddInfo('aaridh', last_token['char'], phoneme_symbols, has_maddah_mark=True)

        if len(tokens) < 2:
            return None

        if self.processor.DIACRITICS['sukoon'] not in last_token['diacritics']:
            return None

        prev_token = tokens[-2]
        prev_char = prev_token['char']

        # Madd lin: و/ي ساكن مسبوق بفتح، مع الحرف الأخير ساكن للوقف
        if (
            prev_char in (self.processor.WAW, self.processor.YAA)
            and self.processor.DIACRITICS['sukoon'] in prev_token['diacritics']
            and len(tokens) >= 3
            and self.processor.DIACRITICS['fatha'] in tokens[-3]['diacritics']
        ):
            phoneme_symbols = self.config.madd_symbol_map.get(prev_char, {}).get('lin', [prev_char])
            return MaddInfo('lin', prev_char, phoneme_symbols, has_maddah_mark=False)

        # Madd aaridh: الحرف السابق حرف مد (ا، و، ي)
        if prev_char in (
            self.processor.ALIF,
            self.processor.WAW,
            self.processor.YAA,
            self.processor.ALIF_MAKSURA,
        ):
            phoneme_symbols = self.config.madd_symbol_map.get(prev_char, {}).get('aaridh', [prev_char])
            return MaddInfo('aaridh', prev_char, phoneme_symbols, has_maddah_mark=False)

        return None

    def _find_final_consonant_index(
        self,
        chars: List[str],
        target_char: Optional[str] = None,
    ) -> Optional[int]:
        """Locate the index of the final pronounced consonant within phoneme chars."""
        if not chars:
            return None

        if target_char is not None:
            for idx in range(len(chars) - 1, -1, -1):
                if chars[idx] == target_char:
                    return idx

        idx = len(chars) - 1
        while idx >= 0 and chars[idx] == self.config.qalqala_symbol:
            idx -= 1

        return idx if idx >= 0 else None

    def _adjust_madd_sequence(
        self,
        chars: List[str],
        final_idx: int,
        candidates: List[str],
        target_length: int,
    ) -> List[str]:
        """Resize the Madd symbol run so it matches the requested target length."""
        if target_length <= 0:
            return chars

        for symbol in candidates:
            block = self._locate_madd_block(chars, final_idx, symbol)
            if block is not None:
                start, trailing_diacritics = block
                replacement = [symbol] * target_length
                final_segment = chars[final_idx:]
                return (
                    chars[:start]
                    + replacement
                    + final_segment
                    + trailing_diacritics
                )

        return chars

    def _locate_madd_block(
        self, chars: List[str], final_idx: int, symbol: str
    ) -> Optional[Tuple[int, List[str]]]:
        """Identify the contiguous Madd block before the final consonant."""
        if final_idx <= 0:
            return None

        idx = final_idx - 1
        trailing_diacritics: List[str] = []
        while idx >= 0 and chars[idx] in self._diacritics_set:
            trailing_diacritics.append(chars[idx])
            idx -= 1

        trailing_diacritics.reverse()

        if idx < 0 or chars[idx] != symbol:
            return None

        start = idx
        while start - 1 >= 0 and chars[start - 1] == symbol:
            start -= 1

        return start, trailing_diacritics

    def _get_last_base_char(self, tokens: List[Dict[str, List[str]]]) -> Optional[str]:
        """Return the final base character from a parsed token list, if any."""
        if not tokens:
            return None
        return tokens[-1]['char']


def fetch_phonemes_from_service(
    waqf_text: str,
    service_url: str,
    timeout: float = 10.0,
) -> str:
    """Fetch phoneme string for the given Waqf text from a remote service."""
    if not service_url:
        raise RuntimeError("Phoneme service URL is not configured")

    waqf_text = waqf_text.strip()
    if not waqf_text:
        raise RuntimeError("Cannot fetch phonemes for an empty Waqf word")

    payload = json.dumps({"text": waqf_text}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def _perform_request(url: str) -> str:
        request = urllib_request.Request(url, data=payload, headers=headers, method="POST")
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
            charset = "utf-8"
            if hasattr(response, "headers") and hasattr(response.headers, "get_content_charset"):
                charset = response.headers.get_content_charset() or charset
            return raw_body.decode(charset, errors="replace").strip()

    def _raise_http_error(exc: urllib_error.HTTPError, url: str) -> RuntimeError:
        snippet: str = ""
        try:
            body = exc.read()
            if body:
                decoded = body.decode("utf-8", errors="replace").strip()
                if decoded:
                    if len(decoded) > 200:
                        decoded = decoded[:200] + "…"
                    snippet = f" - Response: {decoded}"
        except Exception:
            snippet = ""
        reason = exc.reason if hasattr(exc, "reason") else "Unknown error"
        return RuntimeError(
            f"Phoneme service returned HTTP {exc.code} at {url} ({reason}){snippet}"
        )

    try:
        body_text = _perform_request(service_url)
    except urllib_error.HTTPError as exc:
        if exc.code == 404:
            alternate_url = service_url.rstrip('/') + '/'
            if alternate_url != service_url:
                try:
                    body_text = _perform_request(alternate_url)
                    service_url = alternate_url
                except urllib_error.HTTPError as exc_alt:
                    raise _raise_http_error(exc_alt, alternate_url) from exc_alt
                except urllib_error.URLError as exc_alt:
                    raise RuntimeError(f"Unable to reach phoneme service ({exc_alt})") from exc_alt
            else:
                raise _raise_http_error(exc, service_url) from exc
        else:
            raise _raise_http_error(exc, service_url) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Unable to reach phoneme service ({exc})") from exc

    if not body_text:
        raise RuntimeError("Phoneme service returned an empty response")

    try:
        parsed = json.loads(body_text)
    except json.JSONDecodeError:
        return body_text

    if isinstance(parsed, dict):
        for key in ("phonemes", "phoneme", "text", "result"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise RuntimeError("Phoneme service response did not contain a phoneme string")

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str) and item.strip():
                return item.strip()
        raise RuntimeError("Phoneme service response list did not contain a phoneme string")

    if isinstance(parsed, str):
        return parsed.strip()

    raise RuntimeError("Unexpected phoneme service response format")

def apply_waqf_to_word(
    word: str,
    phonemes: Optional[str] = None,
    phoneme_config: Optional[PhonemeConfig] = None,
) -> Tuple[WaqfResult, Optional[str]]:
    """Apply Waqf rules to a word (and optionally its phoneme string)."""
    processor = WaqfProcessor()
    result = processor.apply_waqf_rules(word)

    updated_phonemes: Optional[str] = None
    if phonemes is not None:
        phoneme_processor = PhonemeProcessor(phoneme_config)
        updated_phonemes = phoneme_processor.process(phonemes, result.waqf)

    return result, updated_phonemes


def process_single_word(
    word: str,
    phoneme_config: Optional[PhonemeConfig] = None,
    phoneme_service_url: Optional[str] = DEFAULT_PHONEME_SERVICE_URL,
    phoneme_timeout: float = 10.0,
) -> None:
    """Process a single Arabic word and synchronise its phoneme string via the remote service."""
    processor = WaqfProcessor()
    result = processor.apply_waqf_rules(word)

    if not phoneme_service_url:
        raise RuntimeError("Phoneme service URL is required for processing")

    raw_phonemes = fetch_phonemes_from_service(
        result.waqf,
        phoneme_service_url,
        timeout=phoneme_timeout,
    )

    phoneme_processor = PhonemeProcessor(phoneme_config)
    updated_phonemes = phoneme_processor.process(raw_phonemes, result.waqf)

    print(f"{result.waqf} {updated_phonemes}")


def main(
    argv: Optional[List[str]] = None,
    *,
    word: Optional[str] = None,
    phoneme_config: Optional[PhonemeConfig] = None,
    phoneme_timeout: float = 10.0,
):
    """Entry point for both CLI usage and direct invocation.

    When *word* is provided, argument parsing is skipped and the
    function processes the request directly. Otherwise, it falls back to the
    traditional CLI behaviour using ``argv`` (defaulting to ``sys.argv[1:]``).
    """

    if word is not None:
        phoneme_config = phoneme_config or PhonemeConfig()

        process_single_word(
            word,
            phoneme_config=phoneme_config,
            phoneme_timeout=phoneme_timeout,
        )

        return

    parser = argparse.ArgumentParser(
        description="🕌 Arabic Quranic Word Waqf (Stopping) Rules Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fix_word_endings.py نَعْبُدُ               # Process single word
    python fix_word_endings.py رَيْبَ --phonemes رَيْبَ  # Update phoneme string
    python fix_word_endings.py --help                 # Show this help

Supported Waqf Rules:
  • Regular consonants with vowels → sukoon
  • Taa Marboota (ة) → silent Haa (ه)  
  • Tanween Fath (ً) → Alif or sukoon for Hamza
  • Tanween Damm/Kasr (ٌ/ٍ) → remove and add sukoon
  • Final Alif, Long vowels, Soft letters → no change
  • Final Hamza → sukoon
        """
    )
    
    parser.add_argument(
        'word', 
        help='Arabic word to process (with diacritics)'
    )
    parser.add_argument(
        '--madd-lin-length',
        type=int,
        default=4,
        help='Duration (in counts) for Madd Lin at Waqf (default: 4)',
    )
    parser.add_argument(
        '--madd-aaridh-length',
        type=int,
        default=4,
        help='Duration (in counts) for Madd Aaridh lil-sukoon (default: 4)',
    )
    parser.add_argument(
        '--madd-natural-length',
        type=int,
        default=2,
        help='Duration (in counts) for natural Madd (default: 2)',
    )
    parser.add_argument(
        '--phoneme-timeout',
        type=float,
        default=10.0,
        help='Timeout in seconds for contacting the default phoneme service (default: 10)',
    )
    parser.add_argument(
        '--version', 
        action='version', 
        version='Arabic Waqf Processor v1.0'
    )
    
    args = parser.parse_args(argv)

    try:
        phoneme_config = PhonemeConfig(
            madd_lin_duration=args.madd_lin_length,
            madd_aaridh_duration=args.madd_aaridh_length,
            natural_madd_duration=args.madd_natural_length,
        )

        process_single_word(
            args.word,
            phoneme_config=phoneme_config,
            phoneme_timeout=args.phoneme_timeout,
        )

    except RuntimeError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main(word='نَعْبُدُ')