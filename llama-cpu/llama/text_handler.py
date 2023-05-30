# A torchserve text handler to serve llama models

from typing import Tuple
from abc import ABC
import json
import logging
import os
import ast
import time
import torch

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("gloo")
    initialize_model_parallel(world_size)
    print('Setup parallel complete!')
    # torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_store")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        world_size, local_rank = setup_model_parallel()
        # Read model serialize/pt file
        start_time = time.time()
        checkpoints = sorted(Path(model_dir).glob("*.pth"))
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]

        # Load model
        logger.debug("Loading model")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(model_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=512, max_batch_size=32, **params
        )
        tokenizer = Tokenizer(model_path=model_dir)
        model_args.vocab_size = tokenizer.n_words
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)
        torch.set_default_tensor_type(torch.BFloat16Tensor)
        model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

        self.model = LLaMA(model, tokenizer)

        self.initialized = True

        logger.debug(f"Loaded llama model in {time.time() - start_time:.2f} seconds")

    def preprocess(self, data):
        """ Very basic preprocessing code - only extract prompts. 
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        text = text.decode('utf-8')
        input_text = ast.literal_eval(text)
        prompts = input_text['contents']

        return prompts

    def inference(self, prompts):
        """
        Predict the class of a text using a trained transformer model.
        """

        if self.mapping:
            prediction = self.mapping[str(prediction)]
        results = self.model.generate(
            prompts, max_gen_len=512, temperature=0.8, top_p=0.95
        )

        return prediction

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e