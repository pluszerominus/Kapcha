from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Модель на основе ViT
def TrOCR():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    return model, processor