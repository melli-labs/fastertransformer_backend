import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
from torch.nn import Module
from transformers import OPTConfig
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation_logits_process import LogitsProcessorList
from transformers.utils import ModelOutput
from transformers.generation_stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import torch.distributed as dist
from torch import nn
from dataclasses import dataclass
import warnings


try:
    # noinspection PyUnresolvedReferences
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BatchEncoding,
    PretrainedConfig,
    PreTrainedTokenizer,
    TensorType,
)


@dataclass
class SampleDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


SampleOutput = SampleDecoderOnlyOutput


# WARNING: some things are hardcoded here like seq length, tokenizer name, etc.!
class OPTModelInference(Module, GenerationMixin):
    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
        inference: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.config: PretrainedConfig = config
        self.device: torch.device = device
        self.inference: Callable[[torch.Tensor], torch.Tensor] = inference
        self.main_input_name = (
            "input_ids"  # https://github.com/huggingface/transformers/pull/14803
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        print("inside prepare inputs")
        return {
            self.main_input_name: input_ids.type(dtype=torch.int32),
        }

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        print("inside prepare model innputs")
        # signature copied from https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/generation_utils.py#L914
        input_name = self.main_input_name
        return inputs, input_name, model_kwargs

    def forward(self, input_ids, **_):
        logits = self.inference(input_ids)
        output = CausalLMOutputWithPast(logits=logits)
        return output

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        print("huhu")
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )

        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        logits_warper = (
            logits_warper if logits_warper is not None else LogitsProcessorList()
        )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        output_scores = (
            output_scores if output_scores is not None else self.config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if self.config.is_encoder_decoder:
            raise NotImplementedError("Only decoder only models are supported")

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            print(model_inputs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=None,
                output_hidden_states=None,
            )
            print(outputs.logits.shape)
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]  # TODO fix me
            print("1")
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            print("2")
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
            print("3")
            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            print("sample works")
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            print("cat works")
            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long()
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids


class TritonPythonModel:
    tokenizer: PreTrainedTokenizer
    device: str

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        self.device = (
            torch.device("cpu")
            if args["model_instance_kind"] == "CPU"
            else torch.device("cuda")
        )
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        # model_config = OPTConfig.from_pretrained("facebook/opt-350m")
        model_config = OPTConfig()
        target_model = args["model_name"].replace("-generate", "-model")

        def inference_triton(input_ids: torch.Tensor) -> torch.Tensor:
            input_ids = input_ids.type(dtype=torch.int32)

            inputs = [pb_utils.Tensor.from_dlpack("input_ids", to_dlpack(input_ids))]
            inference_request = pb_utils.InferenceRequest(
                model_name=target_model,
                requested_output_names=["output"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()

            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, "output"
                )
                tensor: torch.Tensor = torch.from_dlpack(output.to_dlpack())
                if self.device == "cuda":
                    tensor = tensor.cuda()
                return tensor

        self.model = OPTModelInference(
            config=model_config, device=self.device, inference=inference_triton
        )
        if self.device == "cuda":
            self.model = self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        # to silent a warning during seq generation
        # self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
                .as_numpy()
                .tolist()
            ]
            tokens: BatchEncoding = self.tokenizer(
                text=query[0],
                return_tensors=TensorType.PYTORCH,
                return_attention_mask=False,
            )
            # tensorrt uses int32 as input type, ort also because we force the format
            input_ids = tokens.input_ids.type(dtype=torch.int32)
            if self.device == "cuda":
                input_ids = input_ids.to("cuda")
            output_seq: torch.Tensor = self.model.generate(
                input_ids, max_length=64, do_sample=True, top_k=20
            )
            # output_seq: torch.Tensor = self.model.forward(input_ids, max_length=64)
            print("are we done generating?")
            decoded_texts: List[str] = [
                self.tokenizer.decode(seq, skip_special_tokens=True)
                for seq in output_seq
            ]
            tensor_output = [
                pb_utils.Tensor("output", np.array(t, dtype=object))
                for t in decoded_texts
            ]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses
