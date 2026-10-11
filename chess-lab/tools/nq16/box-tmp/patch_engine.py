p = "/srv/workspace/flychess/src/nnue-pytorch/trainer/engine.py"
t = open(p).read()
old = '''        self.callback_metrics["val_loss_epoch"] = mean_val_loss
        self._log_metrics(
            {"val_loss_epoch": mean_val_loss, "epoch": self.current_epoch},
            step=self.global_step,
        )'''
new = '''        self.callback_metrics["val_loss_epoch"] = mean_val_loss
        epoch_metrics = {"val_loss_epoch": mean_val_loss, "epoch": self.current_epoch}
        # d12-space MAE/corr per epoch (operator metric)
        try:
            if hasattr(self.model, "val_metrics_report"):
                rep = self.model.val_metrics_report()
                if rep:
                    epoch_metrics.update(rep)
                    self.callback_metrics.update(rep)
        except Exception:
            pass
        self._log_metrics(epoch_metrics, step=self.global_step)'''
assert old in t
t = t.replace(old, new)
old2 = '''        if hasattr(self.model, "on_validation_epoch_start"):
            self.model.on_validation_epoch_start()'''
new2 = '''        if hasattr(self.model, "val_metrics_reset"):
            self.model.val_metrics_reset()
        if hasattr(self.model, "on_validation_epoch_start"):
            self.model.on_validation_epoch_start()'''
assert old2 in t
t = t.replace(old2, new2)
open(p, "w").write(t)
print("engine patched")
