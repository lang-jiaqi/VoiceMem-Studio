"""Provider-neutral attribution contract and small acoustic implementation."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from .types import TurnAttributionLLMResult, TurnEmotionRecord

@runtime_checkable
class TurnAttributor(Protocol):
    def analyze_turn_with_audio(
        self,
        *,
        audio_path: str,
        asr_text: str | None,
        left_memory_block: str,
        emotion_graph_context: str | None,
        turn: TurnEmotionRecord,
    ) -> TurnAttributionLLMResult:
        ...


@dataclass
class FixtureTurnAttributor:
    """Fixed attribution fixture without model inference."""

    result: TurnAttributionLLMResult | None = None

    def analyze_turn_with_audio(
        self,
        *,
        audio_path: str,
        asr_text: str | None,
        left_memory_block: str,
        emotion_graph_context: str | None,
        turn: TurnEmotionRecord,
    ) -> TurnAttributionLLMResult:
        _ = (audio_path, asr_text, left_memory_block, emotion_graph_context, turn)
        if self.result is not None:
            return self.result
        from voicemem.utils.audio.emotion.types import EmotionGraphDelta, EmotionGraphNodeInput, EmotionSignal

        return TurnAttributionLLMResult(
            analysis_text="Fixture attribution: subdued tone; likely frustration about a blocked experiment.",
            emotion=EmotionSignal(
                label="frustration",
                valence=turn.vad.valence,
                arousal=turn.vad.arousal,
                intensity="high",
                confidence=0.75,
            ),
            acoustic_evidence=["lower vocal energy", "noticeable pauses"],
            semantic_evidence=["mentions experiment failure"],
            related_nodes=[
                EmotionGraphNodeInput(local_id="topic_1", name="main experiment", node_type="Topic")
            ],
            graph_delta=EmotionGraphDelta(
                nodes=[
                    EmotionGraphNodeInput(local_id="topic_1", name="main experiment", node_type="Topic")
                ],
                edges=[],
            ),
            retrieval_snippet=["main experiment", "frustration"],
        )


class SmallEmotionAttributor:
    """Adapt acoustic classification to the existing emotion-memory result schema."""
    def __init__(self, detector=None):
        from .detector import SmallEmotionDetector
        self.detector = detector or SmallEmotionDetector()

    def analyze_turn_with_audio(self, *, audio_path, asr_text, left_memory_block,
                               emotion_graph_context, turn):
        """Return observed affect; no inferred causes or graph relations are invented."""
        from .types import EmotionSignal
        label = self.detector.detect(audio_path)
        return TurnAttributionLLMResult(
            analysis_text=f'Acoustic emotion: {label}.',
            emotion=EmotionSignal(label=label, valence=turn.vad.valence,
                                  arousal=turn.vad.arousal),
            acoustic_evidence=[label],
            retrieval_snippet=[asr_text] if asr_text else [],
        )
