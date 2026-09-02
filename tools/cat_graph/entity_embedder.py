import logging
import os
from typing import Any, Optional, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class ImageTextEmbedder:
    """
    Joint text/image embedder for structure-guided alignment.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.processor_is_fast = False
        self.cache_dir = (os.getenv("CATGRAPH_CLIP_CACHE_DIR", "") or "").strip() or None
        self.source_mode = self._resolve_source_mode()
        self.local_files_only = self.source_mode == "local"

        logger.info(
            "Loading CLIP model: %s on %s (mode=%s, local_files_only=%s, cache_dir=%s)",
            model_name,
            self.device,
            self.source_mode,
            self.local_files_only,
            self.cache_dir or "<default>",
        )
        try:
            model_load_kwargs = self._build_pretrained_kwargs()
            self.model = CLIPModel.from_pretrained(
                model_name,
                **model_load_kwargs,
            ).to(self.device)
            self.processor = self._load_processor(prefer_fast=True)
            self.model.eval()
        except Exception as e:
            logger.error("Failed to load CLIP model: %s", e, exc_info=True)
            self.model = None
            self.processor = None
            self.processor_is_fast = False

    @staticmethod
    def _resolve_source_mode() -> str:
        mode = (os.getenv("CATGRAPH_CLIP_SOURCE_MODE", "online") or "").strip().lower()
        if mode in {"local", "online"}:
            return mode
        logger.warning(
            "Invalid CATGRAPH_CLIP_SOURCE_MODE='%s'. Falling back to 'online'.",
            mode,
        )
        return "online"

    def _build_pretrained_kwargs(self) -> dict:
        # Some transformers/huggingface-hub combinations are sensitive to cache_dir=None.
        kwargs = {"local_files_only": self.local_files_only}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        return kwargs

    def _load_processor(self, prefer_fast: bool) -> CLIPProcessor:
        processor_kwargs = self._build_pretrained_kwargs()
        if prefer_fast:
            try:
                proc = CLIPProcessor.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    **processor_kwargs,
                )
                self.processor_is_fast = True

                image_proc = getattr(proc, "image_processor", None)
                if image_proc is not None and not hasattr(image_proc, "_valid_processor_keys"):
                    logger.warning(
                        "Fast CLIP image processor is incompatible (missing _valid_processor_keys). "
                        "Falling back to slow processor."
                    )
                    raise RuntimeError("Fast CLIP processor compatibility issue")
                return proc
            except Exception as fast_err:
                logger.warning(
                    "Fast CLIP processor unavailable (%s). Falling back to slow processor.",
                    fast_err,
                )

        proc = CLIPProcessor.from_pretrained(
            self.model_name,
            use_fast=False,
            **processor_kwargs,
        )
        self.processor_is_fast = False
        return proc

    def _fallback_to_slow_processor(self) -> bool:
        if not self.processor_is_fast:
            return self.processor is not None
        try:
            self.processor = self._load_processor(prefer_fast=False)
            logger.warning("Switched CLIP processor to slow mode (use_fast=False).")
            return True
        except Exception as e:
            logger.error(f"Failed to switch CLIP processor to slow mode: {e}")
            return False

    def _processor_call_with_fallback(self, **kwargs: Any) -> Any:
        if self.processor is None:
            raise RuntimeError("CLIP processor is not initialized.")
        try:
            return self.processor(**kwargs)
        except AttributeError as e:
            # Known compatibility issue with some transformers builds.
            if "_valid_processor_keys" in str(e):
                logger.warning(
                    "CLIP fast processor call failed (%s). Retrying with slow processor.",
                    e,
                )
                if self._fallback_to_slow_processor():
                    return self.processor(**kwargs)
            raise

    def get_embeddings(
        self,
        texts: Optional[list] = None,
        image_paths: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.model is None or self.processor is None:
            return None, None

        text_embeds = None
        image_embeds = None

        with torch.no_grad():
            if texts:
                inputs_text = self._processor_call_with_fallback(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                text_embeds = self.model.get_text_features(**inputs_text).cpu()

            if image_paths:
                images = []
                for path in image_paths:
                    try:
                        images.append(Image.open(path).convert("RGB"))
                    except Exception as e:
                        logger.warning(f"Cannot read image for embedding: {path} - {e}")

                if images:
                    inputs_img = self._processor_call_with_fallback(
                        images=images,
                        return_tensors="pt",
                    ).to(self.device)
                    image_embeds = self.model.get_image_features(**inputs_img).cpu()

        return text_embeds, image_embeds
