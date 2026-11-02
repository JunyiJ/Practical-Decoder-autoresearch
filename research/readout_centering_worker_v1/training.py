"""Narrow instrumented trainer; frozen incumbent helpers and model math are reused."""
import json, logging, math, time
from pathlib import Path
import torch
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from research.readout_centering_model_v1.runtime.models.gpt import GPT
from research.readout_centering_model_v1.runtime.train.train import (
    _resolve_device, _make_loader, _training_contract as inherited_contract,
    _contracts_match, _evaluate, _get_rss_mb, _get_accel_mem_mb,
    _get_cuda_usage, _summarize_moe_usage, _append_metric,
)
from research.readout_centering_model_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.readout_centering_model_v1.runtime.train.weight_averaging import TailWeightAverages
from research.readout_centering_model_v1.runtime.utils.device import synchronize_device
from research.readout_centering_model_v1.runtime.utils.checkpoint import (
    save_checkpoint, read_checkpoint, capture_training_state, restore_training_state,
)
from research.readout_centering_model_v1.integration import identity, build_optimizer, load_model_weights
from research.gradient_clip_model_v1.clipping import clip_gradients
log = logging.getLogger(__name__)


from research.readout_centering_model_v1.training_state import _training_contract, validate_history


def run_training(cfg: DictConfig, *, output_sink=None) -> dict:
    identity(cfg)
    if output_sink is not None:
        output_sink.validate_config(cfg)
    eval_every = int(getattr(cfg.training, "eval_every", 400))
    eval_iters = int(getattr(cfg.training, "eval_iters", 100))
    if eval_every < 1:
        raise ValueError("cfg.training.eval_every must be >= 1")
    if eval_iters < 1:
        raise ValueError("cfg.training.eval_iters must be >= 1")
    metrics_file = getattr(cfg.training, "metrics_file", None)
    summary_file = getattr(cfg.training, "summary_file", None)
    metrics_path = Path(to_absolute_path(metrics_file)) if metrics_file else None
    summary_path = Path(to_absolute_path(summary_file)) if summary_file else None

    seed = int(getattr(cfg.training, "seed", 0))
    torch.manual_seed(seed)
    if cfg.training.max_iters < 1:
        raise ValueError("training.max_iters must be >= 1")
    device = _resolve_device(cfg)
    loader = _make_loader(cfg, device)
    cfg.model.vocab_size = loader.vocab_size
    from research.readout_centering_model_v1.runtime.models.output_bias import validate_output_bias_training_data
    validate_output_bias_training_data(cfg.model, loader, to_absolute_path(cfg.data.path))
    model = GPT(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg)
    # Capture configured base rates before loading any scheduled checkpoint.
    lr_schedule = LearningRateSchedule(optimizer, cfg.training)
    weight_averages = TailWeightAverages(model, cfg.training)
    checkpoint_dir = to_absolute_path(cfg.training.checkpoint_dir) if cfg.training.checkpoint_dir else None
    checkpoint_every = cfg.training.checkpoint_every
    if checkpoint_dir and output_sink is None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log.info("Starting training: %s + %s on %s", cfg.model.attn_type, cfg.model.mlp_type, device)

    budget = getattr(cfg.training, "max_train_seconds", None)
    if budget is not None and (not math.isfinite(float(budget)) or float(budget) <= 0):
        raise ValueError("max_train_seconds must be positive or null")
    contract = _training_contract(cfg, device, loader)
    metrics = []
    clipping_observations = []
    completed_steps = 0
    training_seconds = 0.0
    best_metric = None
    best_checkpoint_path = None
    resume = getattr(cfg.training, "resume_from", None)
    if resume:
        ckpt = read_checkpoint(to_absolute_path(resume), map_location='cpu')
        state = ckpt.get('training_state')
        if state is None or not _contracts_match(state.get('contract'), contract):
            raise ValueError("Resume requires matching model, data, optimizer, device, PyTorch and evaluation settings")
        clipping_observations = validate_history(state.get('clipping_observations'), int(ckpt['step']), cfg)
        load_model_weights(model, ckpt, cfg, for_resume=True)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        completed_steps = int(ckpt['step'])
        training_seconds = state['training_seconds']
        metrics = state['metrics']
        best_metric = state['best_metric']
        best_checkpoint_path = state['best_checkpoint']
        restore_training_state(state, loader, device)
        weight_averages.load_state_dict(state.get('weight_averaging_state'), completed_steps)

    def save_run(path):
        state = capture_training_state(
            loader, device, contract=contract, training_seconds=training_seconds,
            metrics=metrics, best_metric=best_metric, best_checkpoint=best_checkpoint_path,
        )
        state['clipping_observations'] = validate_history(clipping_observations, completed_steps, cfg)
        if weight_averages.spec is not None:
            state['weight_averaging_state'] = weight_averages.state_dict()
        save_checkpoint(path, model, optimizer=optimizer, step=completed_steps,
                        cfg=cfg, training_state=state,
                        **({"payload_writer": output_sink.checkpoint} if output_sink is not None else {}))

    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    final_checkpoint_path = None
    stop_reason = "max_iters"
    for iter in range(completed_steps, cfg.training.max_iters):
        if budget is not None and training_seconds >= budget:
            stop_reason = "time_budget"
            break
        model.train()
        synchronize_device(device)
        iter_start = time.perf_counter()
        xb, yb = loader.get_batch('train')
        objective_kwargs = ({"include_training_objective": True}
                            if getattr(model, "logit_z_signature", None) is not None else {})
        logits, loss, train_loss_breakdown = model(xb, yb, return_loss_breakdown=True, **objective_kwargs)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        clipping = dict(step=iter+1, **clip_gradients(model.parameters(), cfg.training.grad_clip))
        lr_multiplier = lr_schedule.apply(iter)
        optimizer.step()
        weight_averages.update(iter + 1)
        synchronize_device(device)
        iter_time = time.perf_counter() - iter_start
        completed_steps = iter + 1
        clipping_observations.append(clipping)
        training_seconds += iter_time
        budget_reached = budget is not None and training_seconds >= budget

        if iter % eval_every == 0 or iter == cfg.training.max_iters - 1 or budget_reached:
            elapsed = time.perf_counter() - start_time
            rss_mb = _get_rss_mb()
            accel_mb = _get_accel_mem_mb(device)
            cuda_usage = _get_cuda_usage(device)
            # Snapshot training routing before evaluation overwrites the histograms.
            moe_usage = _summarize_moe_usage(model)
            avg_eval_loss, avg_eval_aux_loss = _evaluate(model, loader, 'val', eval_iters)
            ppl = math.exp(avg_eval_loss)
            train_ce_loss = train_loss_breakdown["ce_loss"]
            train_aux_loss = train_loss_breakdown["aux_loss"]

            if train_aux_loss is None and accel_mb is None:
                log.info(
                    "Step %d: Loss %.4f | val %.4f | ppl %.2f | iter %.3fs | elapsed %.1fs | rss %.1f MB",
                    iter,
                    loss.item(),
                    avg_eval_loss,
                    ppl,
                    iter_time,
                    elapsed,
                    rss_mb,
                )
            elif train_aux_loss is None:
                log.info(
                    "Step %d: Loss %.4f | val %.4f | ppl %.2f | iter %.3fs | elapsed %.1fs | rss %.1f MB | accel %.1f MB",
                    iter,
                    loss.item(),
                    avg_eval_loss,
                    ppl,
                    iter_time,
                    elapsed,
                    rss_mb,
                    accel_mb,
                )
            elif accel_mb is None:
                log.info(
                    "Step %d: Loss %.4f | ce %.4f | aux %.4f | val_ce %.4f | val_aux %.4f | ppl %.2f | iter %.3fs | elapsed %.1fs | rss %.1f MB",
                    iter,
                    loss.item(),
                    train_ce_loss.item(),
                    train_aux_loss.item(),
                    avg_eval_loss,
                    0.0 if avg_eval_aux_loss is None else avg_eval_aux_loss,
                    ppl,
                    iter_time,
                    elapsed,
                    rss_mb,
                )
            else:
                log.info(
                    "Step %d: Loss %.4f | ce %.4f | aux %.4f | val_ce %.4f | val_aux %.4f | ppl %.2f | iter %.3fs | elapsed %.1fs | rss %.1f MB | accel %.1f MB",
                    iter,
                    loss.item(),
                    train_ce_loss.item(),
                    train_aux_loss.item(),
                    avg_eval_loss,
                    0.0 if avg_eval_aux_loss is None else avg_eval_aux_loss,
                    ppl,
                    iter_time,
                    elapsed,
                    rss_mb,
                    accel_mb,
                )
            if "logit_z_coefficient" in train_loss_breakdown:
                log.info("Training objective: ce %.6f | logit_z_coefficient %.6g | logit_z_squared_mean %s | logit_z_penalty %s | logit_z_mean %s | total %.6f",
                         train_ce_loss.item(), train_loss_breakdown["logit_z_coefficient"],
                         *[None if train_loss_breakdown[k] is None else train_loss_breakdown[k].item()
                           for k in ("logit_z_squared_mean", "logit_z_penalty", "logit_z_mean")], loss.item())
            if cuda_usage is not None:
                log.info(
                    "CUDA usage (gpu %d): util %.1f%% | used %.1f / %.1f MB",
                    cuda_usage["gpu_index"],
                    cuda_usage["utilization_pct"],
                    cuda_usage["memory_used_mb"],
                    cuda_usage["memory_total_mb"],
                )
            if moe_usage is not None:
                usage_str = ", ".join(f"{u:.3f}" for u in moe_usage["aggregate"]) or "n/a"
                log.info("MoE expert usage (train batch, top-k assignments): %s", usage_str)
                for layer_usage in moe_usage["per_layer"]:
                    layer_usage_str = ", ".join(f"{u:.3f}" for u in layer_usage["usage"]) or "n/a"
                    log.info("MoE expert usage (layer %d): %s", layer_usage["layer_idx"], layer_usage_str)

            metric = {
                "clipping": clipping,
                **({'averaging_snapshots': weight_averages.snapshots}
                   if weight_averages.spec is not None else {}),
                **({'lr_multiplier': lr_multiplier,
                    'base_learning_rate': float(cfg.training.learning_rate),
                    'scheduled_learning_rate': float(cfg.training.learning_rate) * lr_multiplier}
                   if lr_multiplier is not None else {}),
                "step": iter,
                "tokens_processed": (iter + 1) * cfg.training.batch_size * cfg.model.block_size,
                "train_loss": float(loss.item()),
                "train_total_loss": float(loss.item()),
                "train_ce_loss": float(train_ce_loss.item()),
                "train_aux_loss": None if train_aux_loss is None else float(train_aux_loss.item()),
                "val_loss": float(avg_eval_loss),
                "val_ce_loss": float(avg_eval_loss),
                "val_aux_loss": None if avg_eval_aux_loss is None else float(avg_eval_aux_loss),
                "ppl": float(ppl),
                "iter_time_sec": float(iter_time),
                "elapsed_time_sec": float(elapsed),
                "training_time_sec": training_seconds,
                "accel_mem_kind": "cuda_peak_allocated" if str(device).startswith('cuda') else "mps_current_allocated" if device == 'mps' else None,
                "rss_kind": "process_lifetime_peak",
                "rss_mb": float(rss_mb),
                "accel_mem_mb": None if accel_mb is None else float(accel_mb),
                "device": device,
                "cuda": cuda_usage,
                "moe_usage": moe_usage,
            }
            if "logit_z_coefficient" in train_loss_breakdown:
                metric["train_logit_z_coefficient"] = train_loss_breakdown["logit_z_coefficient"]
                for key in ("logit_z_squared_mean", "logit_z_penalty", "logit_z_mean"):
                    value = train_loss_breakdown[key]
                    metric["train_" + key] = None if value is None else float(value.item())
            metrics.append(metric)
            if best_metric is None or metric['val_loss'] < best_metric['val_loss']:
                best_metric = metric
                if checkpoint_dir and getattr(cfg.training, 'keep_best_checkpoint', True):
                    best_checkpoint_path = str(Path(checkpoint_dir) / f"best_{completed_steps:06d}_{time.time_ns()}.pt")
                    save_run(best_checkpoint_path)
            if metrics_path is not None:
                if output_sink is None:
                    _append_metric(metrics_path, metric)
                else:
                    output_sink.metric(metrics_path, metric)
        if checkpoint_dir and checkpoint_every > 0 and completed_steps % checkpoint_every == 0:
            ckpt_path = Path(checkpoint_dir) / f"step_{completed_steps:06d}_{time.time_ns()}.pt"
            save_run(ckpt_path)
            final_checkpoint_path = str(ckpt_path)
        if budget_reached:
            stop_reason = "time_budget"
            break
    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir) / f"final_{completed_steps:06d}_{time.time_ns()}.pt"
        save_run(ckpt_path)
        final_checkpoint_path = str(ckpt_path)

    result = {
        "clipping_observations": validate_history(clipping_observations, completed_steps, cfg),
        "device": device,
        "metrics": metrics,
        "best_metric": best_metric,
        "final_metric": metrics[-1] if metrics else None,
        "final_checkpoint": final_checkpoint_path,
        "max_iters": int(cfg.training.max_iters),
        "completed_steps": completed_steps,
        "training_time_sec": training_seconds,
        "stop_reason": stop_reason,
        "best_checkpoint": best_checkpoint_path,
        "eval_every": eval_every,
        "eval_iters": eval_iters,
    }
    if weight_averages.spec is not None:
        result['weight_averaging'] = dict(spec=weight_averages.spec,
            completed_steps=weight_averages.completed_steps, snapshots=weight_averages.snapshots)
    if summary_path is not None:
        if output_sink is None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, sort_keys=True)
        else:
            output_sink.summary(summary_path, result)
    return result

