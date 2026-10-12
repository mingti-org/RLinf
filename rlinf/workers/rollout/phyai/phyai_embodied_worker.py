# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embodied phyai rollout worker: drive a registered phyai action converter over
channels against a driver-launched ``phyai serve`` (no worker-owned HTTP
server, no in-worker subprocess).

Used by embodied phyai-convert-action models (e.g. DreamZero). The eval driver
launches the ``phyai serve`` server group via
:func:`launch_phyai_router_and_server` and pushes the server URLs to each
rollout worker via :meth:`set_phyai_server_urls`; the worker picks the URL
at its own rank (for N-server parallel throughput), loads the phyai action converter
registered for ``rollout.model.model_type``, and is driven by
``EmbodiedEvalRunner`` over channels (``recv_from``/``send_to``). It does NOT
host its own HTTP server (the agent path uses
:class:`phyaiAgentWorkerWithHTTPServer`).
"""

from typing import Any, Literal, Optional

import torch
from omegaconf import DictConfig

from rlinf.scheduler import Worker
from rlinf.utils.placement import HybridComponentPlacement
import numpy as np


class PhyaiEmbodiedWorker(Worker):
    """Use a driver-launched ``phyai serve`` + phyai action converter + channel eval."""

    def __init__(
        self,
        config: DictConfig,
        placement: HybridComponentPlacement,
        config_rollout: Optional[DictConfig] = None,
    ):
        Worker.__init__(self)
        self.cfg = config
        self.cfg_rollout = (
            config_rollout if config_rollout is not None else config.rollout
        )
        self.model_type = str(
            getattr(getattr(self.cfg_rollout, "model", None), "model_type", "")
        ).lower()
        self.model_cfg = self.cfg_rollout.model
        self.http_client = None
        self.phyai_server_url = None
        self._phyai_server_urls = None
        # This worker is eval-only (drives a serve + channel eval; no training).
        assert config.runner.get("only_eval", True), (
            "PhyaiEmbodiedWorker is eval-only; set runner.only_eval: true"
        )
        # Decoupled env/rollout is not implemented on the phyai embodied path.
        assert not config.runner.get("enable_decoupled_mode", False), (
            "PhyaiEmbodiedWorker does not support runner.enable_decoupled_mode"
        )
        eval_env_cfg = config.env.get("eval", None)
        self.num_pipeline_stages = int(config.rollout.pipeline_stage_num)
        total_eval = int(eval_env_cfg.total_num_envs) if eval_env_cfg else 0
        self.eval_batch_size = (
            total_eval // self.num_pipeline_stages
            if self.num_pipeline_stages
            else total_eval
        )
        self.eval_rollout_epoch = int(eval_env_cfg.rollout_epoch) if eval_env_cfg else 1
        if eval_env_cfg is not None:
            self.n_eval_chunk_steps = int(
                eval_env_cfg.max_steps_per_rollout_epoch
            ) // int(self.model_cfg.num_action_chunks)
        else:
            self.n_eval_chunk_steps = 0

    async def init_worker(self):
        self._init_phyai_server()
        from rlinf.utils.http_client import InferenceHTTPClient

        self.http_client = InferenceHTTPClient(self.phyai_server_url)
        phyai_cfg = self.cfg.rollout.get("phyai", {})
        self._http_timeout_s = float(
            phyai_cfg.get("http_timeout_s", phyai_cfg.get("timeout_s", 120.0))
        )
        self._http_max_retries = int(phyai_cfg.get("http_max_retries", 5))
        self._http_retry_backoff_s = float(phyai_cfg.get("http_retry_backoff_s", 1.0))

    def set_phyai_server_urls(self, urls) -> None:
        """Receive the phyai server URLs the driver launched."""
        self._phyai_server_urls = list(urls)

    def _init_phyai_server(self) -> None:
        """Pick the pre-launched phyai server URL assigned to this rank."""
        urls = self._phyai_server_urls
        if not urls:
            raise RuntimeError(
                "phyai server URLs not set; the eval driver must call "
                "rollout_group.set_phyai_server_urls(urls) (after "
                "launch_phyai_router_and_server) before init_workers()."
            )
        self.phyai_server_url = urls[int(self._rank) % len(urls)]
        self.log_info(
            f"phyai server assigned: rank={self._rank} -> "
            f"{self.phyai_server_url} ({len(urls)} server(s))"
        )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [b["obs"] if "obs" in b else b for b in obs_batches]
        merged: dict[str, Any] = {}
        for key in obs_dicts[0].keys():
            values = [d[key] for d in obs_dicts]
            first = next((v for v in values if v is not None), None)
            if first is None:
                merged[key] = None
            elif isinstance(first, torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(first, list):
                merged[key] = [item for sub in values for item in sub]
            else:
                merged[key] = values
        reset = any(b.get("final_obs") is not None for b in obs_batches)
        return {"obs": merged, "reset": reset}

    def build_phyai_request(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if mode != "eval":
            raise NotImplementedError("PhyAI embodied worker supports eval only.")
        routing_keys = {"_rlinf_stage_id", "_rlinf_reset"}
        # Preserve the raw observations from the RLinf EnvWorker without applying DreamZero preprocessing.
        observation = {
            key: value
            for key, value in env_obs.items()
            if key not in routing_keys
        }

        states=observation["states"]
        if torch.is_tensor(states):
            observation["states"]=states.to(dtype=torch.float32)
        else:
            import numpy as np
            observation["states"]=np.asarray(states,dtype=np.float32)

        batch_size = self._infer_env_batch_size(observation)
        stage_id = env_obs.get("_rlinf_stage_id", 0)
        if torch.is_tensor(stage_id):
            stage_id = stage_id.item()
        reset = env_obs.get("_rlinf_reset", False)
        if torch.is_tensor(reset):
            reset = bool(reset.item())
        else:
            reset = bool(reset)
        action_horizon = self.model_cfg.get("action_horizon", None)
        if action_horizon is None:
            action_horizon = self.model_cfg.get("num_action_chunks", 1)
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError(
                f"requested action horizon must be positive, got {action_horizon}"
            )
        metadata = {
            "mode": mode,
            "batch_size": batch_size,
            "stage_id": int(stage_id),
            "reset": reset,
        }
        payload = {
            "observation": observation,
            "model_name":"pi05",
            "model":"pi05",
            "metadata": metadata,
            "requested_action_horizon": action_horizon,
        }
        state = {
            "batch_size": batch_size,
            "requested_action_horizon": action_horizon,
            "stage_id": int(stage_id),
        }
        return payload, state

    def parse_phyai_response(
      self,
      resp: dict[str, Any],
      state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        try:
            actions = torch.as_tensor(resp["actions"], dtype=torch.float32)
        except KeyError as exc:
            raise RuntimeError(
                f"PhyAI response missing actions: {resp}"
            ) from exc  
        expected_shape = (
            state["batch_size"],
            state["requested_action_horizon"],
        )
        if actions.ndim != 3 or tuple(actions.shape[:2]) != expected_shape:
            raise RuntimeError(
                f"PhyAI actions must have shape "
                f"[{expected_shape[0]},{expected_shape[1]},D], "
                f"got {tuple(actions.shape)}"
            )   
        flat = actions.reshape(actions.shape[0], -1)
        info = {
            "prev_logprobs": torch.zeros_like(flat),
            "prev_values": torch.zeros(
                (actions.shape[0], 1),
                dtype=torch.float32,
            ),
            "forward_inputs": {"action": flat.cpu()},
        }
        return actions, info

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """env_obs -> action chunks [N, num_action_chunks, action_dim].

        Owns the phyai HTTP round-trip: the adapter builds the request
        payload and parses the response; this worker performs the msgpack POST.
        """
        payload, state = self.build_phyai_request(env_obs, mode=mode)
        resp = self.http_client.post(
            "/v1/actions/generations",
            payload,
            msgpack=True,
            timeout_s=self._http_timeout_s,
            max_retries=self._http_max_retries,
            retry_backoff_s=self._http_retry_backoff_s,
        )
        return self.parse_phyai_response(resp, state)

    async def evaluate(self, input_channel, output_channel):
        """Channel-based embodied eval loop, driven by EmbodiedEvalRunner."""
        from tqdm import tqdm

        for _ in tqdm(
            range(self.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.n_eval_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_from(
                        group_name=self.cfg.env.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                        merge_fn=self._merge_obs_batches,
                        infer_batch_size_fn=self._infer_env_batch_size,
                    ).async_wait()
                    obs = {
                        **env_output["obs"],
                        "_rlinf_stage_id": stage_id,
                        "_rlinf_reset": env_output.get("reset", False),
                    }
                    actions, _ = self.predict(obs, mode="eval")
                    if isinstance(actions, torch.Tensor):
                        actions = actions.detach().cpu().contiguous()
                    self.send_to(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=actions,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                    )

