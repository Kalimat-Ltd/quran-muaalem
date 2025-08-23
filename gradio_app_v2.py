import gradio as gr
import logging
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import torch

from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
from quran_muaalem.inference import Muaalem
from quran_muaalem.muaalem_typing import MuaalemOutput
from quran_muaalem.explain import explain_for_terminal
from torchcodec.decoders import AudioDecoder
from quran_transcript.utils import PartOfUthmaniWord

# Initialize components
logging.basicConfig(level=logging.INFO)
device = "cpu"
muaalem = Muaalem(device=device)
sampling_rate = 16000

# Load Sura information
sura_idx_to_name = {}
sura_to_aya_count = {}
start_aya = Aya()
for sura_idx in range(1, 115):
    start_aya.set(sura_idx, 1)
    sura_idx_to_name[sura_idx] = start_aya.get().sura_name
    sura_to_aya_count[sura_idx] = start_aya.get().num_ayat_in_sura

moshaf = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=2,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=2,
    sakt_man_raq="idraj",
)


def update_aya_dropdown(sura_idx):
    if not sura_idx:
        sura_idx = 1
    return gr.update(
        choices=list(range(1, sura_to_aya_count[int(sura_idx)] + 1)), value=1
    )


def update_uthmani_ref(sura_idx, aya_idx, start_idx, num_words):
    if not all([sura_idx, aya_idx, start_idx is not None, num_words is not None]):
        return ""
    try:
        uthmani_ref = (
            Aya(int(sura_idx), int(aya_idx))
            .get_by_imlaey_words(int(start_idx), int(num_words))
            .uthmani
        )
        return uthmani_ref
    except PartOfUthmaniWord as e:
        return f"⚠️ Warning: You've selected part of a Uthmani word. Please adjust the number of words to include complete words only.\n\nError details: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def process_audio(audio, sura_idx, aya_idx, start_idx, num_words):
    if audio is None:
        return "Please upload an audio file first"

    try:
        # Get Uthmani reference text
        uthmani_ref = (
            Aya(int(sura_idx), int(aya_idx))
            .get_by_imlaey_words(int(start_idx), int(num_words))
            .uthmani
        )
        phonetizer_out = quran_phonetizer(uthmani_ref, moshaf, remove_spaces=True)

        # Process audio
        decoder = AudioDecoder(audio, sample_rate=sampling_rate, num_channels=1)
        outs = muaalem(
            [decoder.get_all_samples().data[0]],
            [phonetizer_out],
            sampling_rate=sampling_rate,
        )

        # Prepare output
        output_text = f"Phonemes: {outs[0].phonemes}\n\n"
        for sifa in outs[0].sifat:
            output_text += json.dumps(asdict(sifa), indent=2, ensure_ascii=False) + "\n"
            output_text += "*" * 30 + "\n"
        output_text += "-" * 40 + "\n\n"

        # Add explanation
        explanation = explain_for_terminal(
            outs[0].phonemes.text,
            phonetizer_out.phonemes,
            outs[0].sifat,
            phonetizer_out.sifat,
        )
        output_text += f"Explanation:\n{explanation}"

        return output_text
    except PartOfUthmaniWord as e:
        return f"⚠️ Error: The selected word range includes partial Uthmani words. Please adjust the number of words to include complete words only.\n\nError details: {str(e)}"
    # except Exception as e:
    #     return f"Error processing audio: {str(e)}"


with gr.Blocks(title="Quran Recitation Analysis") as app:
    gr.Markdown("# Quran Recitation Analysis Tool")
    gr.Markdown(
        "Select the Sura, Aya, and word range, then upload an audio recording of the recitation for analysis."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Quran Reference")

            # Create sura dropdown with both index and name
            sura_choices = [
                (f"{idx} - {sura_idx_to_name[idx]}", idx) for idx in range(1, 115)
            ]
            sura_dropdown = gr.Dropdown(
                choices=sura_choices, label="Sura", value=1, elem_id="sura_dropdown"
            )

            aya_dropdown = gr.Dropdown(
                choices=list(range(1, sura_to_aya_count[1] + 1)),
                label="Aya Number",
                value=1,
                elem_id="aya_dropdown",
            )
            start_idx = gr.Number(
                value=0,
                label="Start Word Index",
                minimum=0,
                step=1,
                elem_id="start_idx",
            )
            num_words = gr.Number(
                value=5,
                label="Number of Words",
                minimum=1,
                step=1,
                elem_id="num_words",
            )
            uthmani_text = gr.Textbox(
                label="Uthmani Reference Text",
                interactive=False,
                elem_id="uthmani_text",
            )

        with gr.Column(scale=2):
            gr.Markdown("### Audio Input & Analysis")
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                label="Upload or Record Audio",
                type="filepath",
                elem_id="audio_input",
            )
            analyze_btn = gr.Button(
                "Analyze Recitation", variant="primary", elem_id="analyze_btn"
            )
            output_text = gr.Textbox(
                label="Analysis Results", lines=20, max_lines=50, elem_id="output_text"
            )

    # Initial update of uthmani text
    app.load(
        update_uthmani_ref,
        inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
        outputs=uthmani_text,
    )

    # Update aya dropdown when sura changes and reset aya_idx to 1
    sura_dropdown.change(
        update_aya_dropdown, inputs=sura_dropdown, outputs=aya_dropdown
    ).then(
        update_uthmani_ref,
        inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
        outputs=uthmani_text,
    )

    # Update uthmani text when any parameter changes
    for component in [aya_dropdown, start_idx, num_words]:
        component.change(
            update_uthmani_ref,
            inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
            outputs=uthmani_text,
        )

    # Process audio when button is clicked
    analyze_btn.click(
        process_audio,
        inputs=[audio_input, sura_dropdown, aya_dropdown, start_idx, num_words],
        outputs=output_text,
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")

