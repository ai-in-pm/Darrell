"""
Florence-2-large Model Integration for Darrell Agent
Advanced computer vision for UI detection and meeting analysis
"""

import asyncio
import time
import io
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import base64

try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: Transformers package not available. Install with: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not available. Install with: pip install opencv-python")
    OPENCV_AVAILABLE = False

from ..utils.logger import DarrellLogger, LogCategory


@dataclass
class VisionResult:
    """Result from Florence-2 vision processing"""
    task: str
    result: Dict[str, Any]
    confidence: float
    processing_time: float
    image_size: Tuple[int, int]
    metadata: Dict[str, Any] = None


@dataclass
class UIElement:
    """Detected UI element"""
    element_type: str
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    clickable: bool = False
    attributes: Dict[str, Any] = None


class Florence2Integration:
    """
    Florence-2-large model integration for advanced computer vision
    Handles UI detection, OCR, object detection, and visual understanding
    """
    
    def __init__(self, model_path: str = "microsoft/Florence-2-large", 
                 device: str = "auto", precision: str = "fp16"):
        """
        Initialize Florence-2 integration
        
        Args:
            model_path: HuggingFace model path
            device: Device to run on (auto, cpu, cuda)
            precision: Model precision (fp16, fp32)
        """
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.precision = precision
        self.logger = DarrellLogger("Florence2Vision")
        
        # Model components
        self.model = None
        self.processor = None
        
        # Performance tracking
        self.processing_times = []
        self.total_images_processed = 0
        
        # Task configurations
        self.task_configs = {
            'OCR': '<OCR>',
            'OCR_WITH_REGION': '<OCR_WITH_REGION>',
            'CAPTION': '<CAPTION>',
            'DETAILED_CAPTION': '<DETAILED_CAPTION>',
            'MORE_DETAILED_CAPTION': '<MORE_DETAILED_CAPTION>',
            'OD': '<OD>',  # Object Detection
            'DENSE_REGION_CAPTION': '<DENSE_REGION_CAPTION>',
            'REGION_PROPOSAL': '<REGION_PROPOSAL>',
            'REFERRING_EXPRESSION_SEGMENTATION': '<REFERRING_EXPRESSION_SEGMENTATION>',
            'REGION_TO_SEGMENTATION': '<REGION_TO_SEGMENTATION>',
            'OPEN_VOCABULARY_DETECTION': '<OPEN_VOCABULARY_DETECTION>',
            'REGION_TO_CATEGORY': '<REGION_TO_CATEGORY>',
            'REGION_TO_DESCRIPTION': '<REGION_TO_DESCRIPTION>'
        }
        
        # State management
        self.is_initialized = False
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers package not available", LogCategory.AI_MODEL)
            raise ImportError("Transformers package required for Florence-2")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    async def initialize(self) -> bool:
        """Initialize Florence-2 model"""
        try:
            self.logger.info(f"Initializing Florence-2 model: {self.model_path}", LogCategory.AI_MODEL)
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Load model
            if self.precision == "fp16" and self.device != "cpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                ).to(self.device)
            
            # Test model with a simple task
            test_success = await self._test_model()
            
            if test_success:
                self.is_initialized = True
                self.logger.info("Florence-2 model initialized successfully", LogCategory.AI_MODEL,
                               performance_data={
                                   "device": self.device,
                                   "precision": self.precision,
                                   "model_path": self.model_path
                               })
                return True
            else:
                self.logger.error("Florence-2 model test failed", LogCategory.AI_MODEL)
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Florence-2: {e}", LogCategory.AI_MODEL, error=e)
            return False
    
    async def _test_model(self) -> bool:
        """Test model with a simple task"""
        try:
            # Create a simple test image
            test_image = Image.new('RGB', (100, 100), color='white')
            
            # Test caption generation
            result = await self.process_image(test_image, 'CAPTION')
            
            return result is not None and result.result is not None
            
        except Exception as e:
            self.logger.error(f"Model test failed: {e}", LogCategory.AI_MODEL, error=e)
            return False
    
    async def process_image(self, image: Union[Image.Image, np.ndarray, str], 
                          task: str, text_input: str = None) -> Optional[VisionResult]:
        """
        Process image with Florence-2 model
        
        Args:
            image: PIL Image, numpy array, or file path
            task: Task to perform (OCR, CAPTION, OD, etc.)
            text_input: Additional text input for some tasks
        
        Returns:
            VisionResult or None if failed
        """
        try:
            start_time = time.time()
            
            # Convert image to PIL if needed
            pil_image = self._convert_to_pil(image)
            if pil_image is None:
                return None
            
            # Get task prompt
            task_prompt = self.task_configs.get(task.upper())
            if not task_prompt:
                self.logger.error(f"Unknown task: {task}", LogCategory.AI_MODEL)
                return None
            
            # Prepare prompt
            if text_input:
                prompt = f"{task_prompt} {text_input}"
            else:
                prompt = task_prompt
            
            # Process with model
            inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )
            
            # Decode result
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(pil_image.width, pil_image.height)
            )
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self.processing_times.append(processing_time)
            self.total_images_processed += 1
            
            # Create result
            result = VisionResult(
                task=task,
                result=parsed_answer,
                confidence=self._estimate_confidence(parsed_answer, task),
                processing_time=processing_time,
                image_size=(pil_image.width, pil_image.height),
                metadata={
                    "prompt": prompt,
                    "device": self.device,
                    "model": self.model_path
                }
            )
            
            self.logger.log_ai_model_usage("Florence-2", task, processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}", LogCategory.AI_MODEL, error=e)
            return None
    
    def _convert_to_pil(self, image: Union[Image.Image, np.ndarray, str]) -> Optional[Image.Image]:
        """Convert various image formats to PIL Image"""
        try:
            if isinstance(image, Image.Image):
                return image
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    # Convert BGR to RGB if needed
                    if image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(image)
            elif isinstance(image, str):
                # File path or base64
                if image.startswith('data:image'):
                    # Base64 encoded image
                    header, data = image.split(',', 1)
                    image_data = base64.b64decode(data)
                    return Image.open(io.BytesIO(image_data))
                else:
                    # File path
                    return Image.open(image)
            else:
                self.logger.error(f"Unsupported image type: {type(image)}", LogCategory.AI_MODEL)
                return None
                
        except Exception as e:
            self.logger.error(f"Image conversion failed: {e}", LogCategory.AI_MODEL, error=e)
            return None
    
    def _estimate_confidence(self, result: Dict[str, Any], task: str) -> float:
        """Estimate confidence score for result"""
        # This is a simplified confidence estimation
        # In practice, you might use model logits or other metrics
        
        if not result:
            return 0.0
        
        if task.upper() == 'OCR':
            # For OCR, confidence based on text length and structure
            text = result.get('text', '')
            if len(text) > 0:
                return min(0.9, 0.5 + len(text) * 0.01)
            return 0.1
        
        elif task.upper() in ['CAPTION', 'DETAILED_CAPTION', 'MORE_DETAILED_CAPTION']:
            # For captions, confidence based on description length
            caption = result.get('caption', result.get('text', ''))
            if len(caption) > 10:
                return 0.8
            elif len(caption) > 0:
                return 0.6
            return 0.2
        
        elif task.upper() == 'OD':
            # For object detection, average confidence of detections
            bboxes = result.get('bboxes', [])
            labels = result.get('labels', [])
            if bboxes and labels:
                return 0.7  # Default confidence for detections
            return 0.1
        
        # Default confidence
        return 0.5
    
    async def extract_text(self, image: Union[Image.Image, np.ndarray, str], 
                          with_regions: bool = False) -> Optional[Dict[str, Any]]:
        """
        Extract text from image using OCR
        
        Args:
            image: Image to process
            with_regions: Whether to include text regions
        
        Returns:
            OCR result with text and optionally regions
        """
        task = 'OCR_WITH_REGION' if with_regions else 'OCR'
        result = await self.process_image(image, task)
        
        if result and result.result:
            return {
                'text': result.result.get('text', ''),
                'regions': result.result.get('quad_boxes', []) if with_regions else [],
                'confidence': result.confidence,
                'processing_time': result.processing_time
            }
        
        return None
    
    async def detect_objects(self, image: Union[Image.Image, np.ndarray, str]) -> Optional[List[Dict[str, Any]]]:
        """
        Detect objects in image
        
        Args:
            image: Image to process
        
        Returns:
            List of detected objects with bboxes and labels
        """
        result = await self.process_image(image, 'OD')
        
        if result and result.result:
            bboxes = result.result.get('bboxes', [])
            labels = result.result.get('labels', [])
            
            objects = []
            for bbox, label in zip(bboxes, labels):
                objects.append({
                    'label': label,
                    'bbox': bbox,
                    'confidence': result.confidence
                })
            
            return objects
        
        return []
    
    async def generate_caption(self, image: Union[Image.Image, np.ndarray, str], 
                             detail_level: str = "normal") -> Optional[str]:
        """
        Generate caption for image
        
        Args:
            image: Image to process
            detail_level: Level of detail (normal, detailed, more_detailed)
        
        Returns:
            Generated caption
        """
        task_map = {
            "normal": "CAPTION",
            "detailed": "DETAILED_CAPTION", 
            "more_detailed": "MORE_DETAILED_CAPTION"
        }
        
        task = task_map.get(detail_level, "CAPTION")
        result = await self.process_image(image, task)
        
        if result and result.result:
            return result.result.get('caption', result.result.get('text', ''))
        
        return None
    
    async def find_ui_elements(self, image: Union[Image.Image, np.ndarray, str]) -> List[UIElement]:
        """
        Find UI elements in image using multiple Florence-2 tasks
        
        Args:
            image: Screenshot or image to analyze
        
        Returns:
            List of detected UI elements
        """
        ui_elements = []
        
        try:
            # Extract text with regions for text-based UI elements
            ocr_result = await self.extract_text(image, with_regions=True)
            if ocr_result and ocr_result['regions']:
                for i, region in enumerate(ocr_result['regions']):
                    if len(region) >= 4:  # Ensure we have bbox coordinates
                        # Convert quad to bbox (simplified)
                        x_coords = [region[j] for j in range(0, len(region), 2)]
                        y_coords = [region[j] for j in range(1, len(region), 2)]
                        
                        bbox = (
                            min(x_coords), min(y_coords),
                            max(x_coords), max(y_coords)
                        )
                        
                        # Extract text for this region (simplified)
                        text = ocr_result['text'] if i == 0 else ""
                        
                        ui_elements.append(UIElement(
                            element_type="text",
                            text=text,
                            bbox=bbox,
                            confidence=ocr_result['confidence'],
                            clickable=self._is_likely_clickable(text),
                            attributes={"ocr_region": i}
                        ))
            
            # Detect objects that might be UI elements
            objects = await self.detect_objects(image)
            for obj in objects:
                if self._is_ui_object(obj['label']):
                    ui_elements.append(UIElement(
                        element_type="object",
                        text=obj['label'],
                        bbox=tuple(obj['bbox']),
                        confidence=obj['confidence'],
                        clickable=True,
                        attributes={"object_type": obj['label']}
                    ))
            
            self.logger.info(f"Found {len(ui_elements)} UI elements", LogCategory.AI_MODEL)
            return ui_elements
            
        except Exception as e:
            self.logger.error(f"UI element detection failed: {e}", LogCategory.AI_MODEL, error=e)
            return []
    
    def _is_likely_clickable(self, text: str) -> bool:
        """Determine if text is likely clickable"""
        clickable_keywords = [
            'button', 'click', 'join', 'start', 'stop', 'mute', 'unmute',
            'video', 'audio', 'share', 'chat', 'participants', 'settings',
            'leave', 'end', 'record', 'more', 'options'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in clickable_keywords)
    
    def _is_ui_object(self, label: str) -> bool:
        """Determine if detected object is a UI element"""
        ui_objects = [
            'button', 'icon', 'menu', 'window', 'dialog', 'panel',
            'toolbar', 'tab', 'checkbox', 'radio', 'slider'
        ]
        
        label_lower = label.lower()
        return any(ui_obj in label_lower for ui_obj in ui_objects)
    
    async def analyze_meeting_screen(self, image: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        """
        Analyze meeting screen for participants, UI state, etc.
        
        Args:
            image: Meeting screenshot
        
        Returns:
            Analysis results
        """
        try:
            # Generate detailed caption
            caption = await self.generate_caption(image, "detailed")
            
            # Extract text for UI state
            ocr_result = await self.extract_text(image)
            
            # Detect objects
            objects = await self.detect_objects(image)
            
            # Find UI elements
            ui_elements = await self.find_ui_elements(image)
            
            analysis = {
                'caption': caption,
                'extracted_text': ocr_result['text'] if ocr_result else '',
                'detected_objects': objects,
                'ui_elements': [
                    {
                        'type': elem.element_type,
                        'text': elem.text,
                        'bbox': elem.bbox,
                        'clickable': elem.clickable
                    } for elem in ui_elements
                ],
                'meeting_state': self._analyze_meeting_state(caption, ocr_result, objects),
                'participant_count': self._estimate_participant_count(caption, objects)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Meeting screen analysis failed: {e}", LogCategory.AI_MODEL, error=e)
            return {}
    
    def _analyze_meeting_state(self, caption: str, ocr_result: Dict, objects: List) -> Dict[str, Any]:
        """Analyze current meeting state from visual information"""
        state = {
            'in_meeting': False,
            'muted': False,
            'video_on': False,
            'screen_sharing': False,
            'chat_open': False,
            'participants_visible': False
        }
        
        if caption:
            caption_lower = caption.lower()
            state['in_meeting'] = any(word in caption_lower for word in ['meeting', 'zoom', 'participants', 'video call'])
            state['screen_sharing'] = 'sharing' in caption_lower or 'screen' in caption_lower
        
        if ocr_result and ocr_result.get('text'):
            text_lower = ocr_result['text'].lower()
            state['muted'] = 'mute' in text_lower or 'unmute' in text_lower
            state['video_on'] = 'start video' not in text_lower and 'stop video' in text_lower
            state['chat_open'] = 'chat' in text_lower
            state['participants_visible'] = 'participants' in text_lower
        
        return state
    
    def _estimate_participant_count(self, caption: str, objects: List) -> int:
        """Estimate number of participants from visual analysis"""
        # This is a simplified estimation
        # In practice, you might look for face detection or video tiles
        
        count = 0
        
        if caption:
            # Look for mentions of people or participants
            import re
            numbers = re.findall(r'\d+', caption)
            for num in numbers:
                if 1 <= int(num) <= 100:  # Reasonable participant count
                    count = max(count, int(num))
        
        # Count face-like objects
        face_objects = [obj for obj in objects if 'person' in obj.get('label', '').lower() or 'face' in obj.get('label', '').lower()]
        count = max(count, len(face_objects))
        
        return count
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get Florence-2 performance metrics"""
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            "total_images_processed": self.total_images_processed,
            "average_processing_time": avg_processing_time,
            "device": self.device,
            "precision": self.precision,
            "model_path": self.model_path,
            "is_initialized": self.is_initialized,
            "available_tasks": list(self.task_configs.keys())
        }
    
    async def cleanup(self):
        """Cleanup Florence-2 resources"""
        self.logger.info("Cleaning up Florence-2 integration...", LogCategory.AI_MODEL)
        
        if self.model:
            del self.model
            self.model = None
        
        if self.processor:
            del self.processor
            self.processor = None
        
        # Clear CUDA cache if using GPU
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        self.logger.info("Florence-2 cleanup completed", LogCategory.AI_MODEL)
