p = "/srv/workspace/flychess/src/nnue-pytorch/model/nnue.py"
t = open(p).read()
old = '''    def val_step(self, batch, current_epoch, global_step):
        _ = current_epoch
        loss = self.compute_loss(batch, global_step)
        self.loss_metrics["val_loss_epoch"].update(loss)
        return {"val_loss": loss}'''
new = '''    def val_step(self, batch, current_epoch, global_step):
        _ = current_epoch
        loss = self.compute_loss(batch, global_step)
        self.loss_metrics["val_loss_epoch"].update(loss)
        out = {"val_loss": loss}
        # per-epoch d12-space MAE/corr (operator metric, 2026-10-04)
        try:
            (
                us, them, white_indices, black_indices,
                _outcome, score, piece_count,
            ) = batch
            scorenet = self.model(
                us, them, white_indices, black_indices, piece_count,
                self.config.use_fake_act_quantization,
                self.config.use_fake_weight_quantization,
            )
            pred = scorenet * self.model.quantization.nnue2score
            if not hasattr(self, "_mae_accum"):
                self._mae_accum = {"n": 0, "ae": 0.0,
                                   "sx": 0.0, "sy": 0.0,
                                   "sxx": 0.0, "syy": 0.0, "sxy": 0.0}
            d = self.config.loss_params.in_scaling
            o = self.config.loss_params.in_offset
            # batch score is scaled: cp = (s*scaling)+offset
            pred_cp = pred * d + o
            true_cp = score * d + o
            diff = (pred_cp - true_cp).detach()
            n = diff.numel()
            self._mae_accum["n"] += n
            self._mae_accum["ae"] += float(diff.abs().sum())
            self._mae_accum["sx"] += float(true_cp.sum())
            self._mae_accum["sy"] += float(pred_cp.sum())
            self._mae_accum["sxx"] += float((true_cp * true_cp).sum())
            self._mae_accum["syy"] += float((pred_cp * pred_cp).sum())
            self._mae_accum["sxy"] += float((true_cp * pred_cp).sum())
        except Exception:
            pass
        return out

    def val_metrics_reset(self):
        self._mae_accum = {"n": 0, "ae": 0.0,
                           "sx": 0.0, "sy": 0.0,
                           "sxx": 0.0, "syy": 0.0, "sxy": 0.0}

    def val_metrics_report(self):
        a = getattr(self, "_mae_accum", None)
        if not a or a["n"] == 0:
            return None
        n = a["n"]
        mae = a["ae"] / n
        mx, my = a["sx"] / n, a["sy"] / n
        cov = a["sxy"] / n - mx * my
        vx = a["sxx"] / n - mx * mx
        vy = a["syy"] / n - my * my
        import math as _m
        corr = cov / _m.sqrt(max(vx * vy, 1e-12))
        return {"val_mae": mae, "val_corr": corr}'''
assert old in t
t = t.replace(old, new)
open(p, "w").write(t)
print("val_step patched")
