import logging
import io
import os

import dtlpy as dl

logger = logging.getLogger('video-utils.prompt-to-text')

DEFAULT_OUTPUT_DIR = '/text_responses_dir'


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.output_dir = DEFAULT_OUTPUT_DIR

    def run(self, item: dl.Item, context: dl.Context) -> dl.Item:
        """
        Receives a prompt item (after VLM predict), extracts the model response,
        creates a text item from it, and uploads it to the same dataset.

        Args:
            item (dl.Item): A prompt item that has been through VLM prediction.
            context (dl.Context): Pipeline context containing node configuration

        Returns:
            dl.Item: The newly uploaded text item containing the model response.
        """
        logger.info(f"Processing prompt item: {item.id} ({item.name})")

        node_config = context.node.metadata.get('customNodeConfig', {})
        self.output_dir = node_config.get('output_dir', DEFAULT_OUTPUT_DIR)
                
        # Load the item as a PromptItem to access assistant responses
        prompt_item = dl.PromptItem.from_item(item=item)

        # Extract the model response text from assistant prompts
        response_texts = []
        for assistant_prompt in prompt_item.assistant_prompts:
            for element in assistant_prompt.elements:
                if element.get('mimetype') == dl.PromptType.TEXT:
                    response_texts.append(element['value'])

        if not response_texts:
            raise ValueError(f"No assistant response found in prompt item {item.id}")

        # Combine all response texts (typically there's one per prompt key)
        full_response = "\n\n".join(response_texts)
        logger.info(f"Extracted response ({len(full_response)} chars) from prompt item {item.id}")

        # Build the text item name from the prompt item name
        base_name = os.path.splitext(item.name)[0]
        text_item_name = f"{base_name}-response.txt"

        # Create a BytesIO buffer for the text content
        buffer = io.BytesIO()
        buffer.name = text_item_name
        buffer.write(full_response.encode('utf-8'))
        buffer.seek(0)

        remote_path = self.output_dir
        dataset = item.dataset
        uploaded_item = dataset.items.upload(
            local_path=buffer,
            remote_path=remote_path,
        )
        logger.info(f"Uploaded text item: {uploaded_item.id} ({uploaded_item.name})")

        # Copy lineage metadata from the prompt item
        uploaded_item.metadata['user'] = uploaded_item.metadata.get('user', {})

        frame_indices = item.metadata.get('user', {}).get('frame_indices', None)
        if frame_indices is not None:
            uploaded_item.metadata['user']['frame_indices'] = frame_indices
            logger.info(f"Copied frame_indices {frame_indices} to text item metadata")
        else:
            logger.warning(f"No frame_indices found in prompt item {item.id} user metadata")

        origin_video_name = item.metadata.get('origin_video_name', None)
        if origin_video_name is not None:
            uploaded_item.metadata['origin_video_name'] = origin_video_name
        created_time = item.metadata.get('created_time', None)
        if created_time is not None:
            uploaded_item.metadata['created_time'] = created_time

        uploaded_item = uploaded_item.update()

        logger.info(f"returning uploaded item {uploaded_item.id} ({uploaded_item.name})")
        return uploaded_item
